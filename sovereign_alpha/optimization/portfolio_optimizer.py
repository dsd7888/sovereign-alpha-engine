"""
L4 — Portfolio Optimizer.

Ledoit-Wolf shrinkage covariance + a fully-vectorised, Cholesky-correlated
Monte Carlo simulation over the CVaR/Sharpe efficient frontier.

Note on the Monte Carlo engine: a naive implementation simulates one
portfolio at a time in a Python ``for`` loop (5,000 iterations, each doing an
(n_assets, 252) matrix multiply). That is roughly two orders of magnitude
slower than necessary — every simulation draws from the same ``mu``/``cov``,
so the whole batch can be generated as one tensor and reduced with
``einsum``. This implementation still processes simulations in bounded-size
batches (not all 5,000 at once) purely to cap peak memory, not because the
math requires it.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — required for unattended/cron runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import settings
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.optimization.transaction_costs import GrowwCostModel
from sovereign_alpha.utils.helpers import today_str
from sovereign_alpha.utils.logger import logger
from sovereign_alpha.utils.validators import is_valid_price, normalise_weights

# Cap the in-memory footprint of a single Monte Carlo batch to roughly this
# many float64 elements (n_batch * n_assets * TRADING_DAYS) ~= 200 MB.
_MAX_BATCH_ELEMENTS = 25_000_000


class PortfolioOptimizerError(RuntimeError):
    pass


def _shrink_mu(raw_mu: np.ndarray, sigma_sq_diag: np.ndarray, n_obs: int) -> np.ndarray:
    """
    James-Stein shrinkage: pull individual asset return estimates
    toward the grand mean. Reduces overfitting in short sample windows.

    Formula:
      grand_mean = mean(raw_mu)
      shrinkage  = 1 - ((n_assets - 2) / n_obs) * TRADING_DAYS *
                       sum(sigma_sq_diag) / sum((raw_mu - grand_mean)^2)
      mu_shrunk  = grand_mean + shrinkage * (raw_mu - grand_mean)

    ``sigma_sq_diag`` is annualised variance and ``raw_mu``/``ss_dev`` are in
    annualised-return units, but ``n_obs`` counts *daily* observations — the
    James-Stein estimation-noise term Var(mu_hat) therefore needs an extra
    factor of ``TRADING_DAYS`` to convert the per-day estimator variance into
    annualised terms (Var(252 * daily_mean_hat) = 252 * annual_var / n_obs).
    Without it, sigma_sq_diag/n_obs understates estimation noise by ~252x and
    the shrinkage is nearly inert even on a 3-year window.

    Clip shrinkage to [0, 1] — never extrapolate, never invert.
    """
    n_assets = len(raw_mu)
    grand_mean = float(np.mean(raw_mu))
    deviations = raw_mu - grand_mean
    ss_dev = float(np.sum(deviations ** 2))

    if ss_dev < 1e-10 or n_assets <= 2:
        logger.info("[OPT] James-Stein shrinkage skipped (degenerate mu spread), mu unchanged")
        return raw_mu.copy()

    shrinkage_factor = 1.0 - (
        ((n_assets - 2) * settings.TRADING_DAYS * np.sum(sigma_sq_diag)) /
        (n_obs * ss_dev)
    )
    shrinkage_factor = float(np.clip(shrinkage_factor, 0.0, 1.0))

    mu_shrunk = grand_mean + shrinkage_factor * deviations
    logger.info(
        f"[OPT] James-Stein mu shrinkage: raw range [{raw_mu.min():.1%}, {raw_mu.max():.1%}] -> "
        f"shrunk range [{mu_shrunk.min():.1%}, {mu_shrunk.max():.1%}], "
        f"shrinkage_factor={shrinkage_factor:.4f}"
    )
    return mu_shrunk


class PortfolioOptimizer:
    def __init__(self) -> None:
        self.ingestor = DataIngestor()
        self.cost_model = GrowwCostModel()
        self._rng = np.random.default_rng(settings.MC_RANDOM_SEED)

    # ── Returns & covariance ────────────────────────────────────────────────

    def load_returns(self, tickers: list[str]) -> pd.DataFrame:
        """
        Daily log-returns matrix (trading_days x n_tickers). Drops tickers
        with fewer than ``settings.MIN_OHLCV_HISTORY_DAYS`` observations or
        more than 10% missing data — a thinly-traded/newly-listed stock
        would otherwise inject a mostly-fabricated (forward-filled) return
        series into the covariance estimate.
        """
        ohlcv = self.ingestor.fetch_ohlcv(tickers)
        series_by_ticker: dict[str, pd.Series] = {}

        for ticker in tickers:
            subset = ohlcv.loc[ohlcv["ticker"] == ticker, "Adj_Close"].dropna()
            if len(subset) < settings.MIN_OHLCV_HISTORY_DAYS:
                logger.warning(f"[OPT] {ticker}: only {len(subset)} price points, skipping")
                continue
            log_ret = np.log(subset / subset.shift(1)).dropna()
            series_by_ticker[ticker] = log_ret

        if not series_by_ticker:
            raise PortfolioOptimizerError("No ticker had sufficient return history to optimise.")

        returns_df = pd.DataFrame(series_by_ticker)
        missing_frac = returns_df.isna().mean()
        keep = missing_frac[missing_frac < 0.10].index.tolist()
        if not keep:
            raise PortfolioOptimizerError("Every ticker exceeded the 10% missing-data threshold.")

        returns_df = returns_df[keep].dropna(how="all").fillna(0.0)
        logger.info(f"[OPT] Returns matrix: {returns_df.shape} — {len(keep)} tickers x {len(returns_df)} days")
        return returns_df

    def compute_covariance_matrix(self, returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Ledoit-Wolf shrinkage covariance, annualised. Returns (annual_cov, annual_mean)."""
        if returns_df.shape[1] < 1:
            raise PortfolioOptimizerError("compute_covariance_matrix: no assets in returns_df")

        lw = LedoitWolf()
        lw.fit(returns_df.values)
        annual_cov = lw.covariance_ * settings.TRADING_DAYS
        annual_cov += np.eye(annual_cov.shape[0]) * 1e-8  # guarantee PD for Cholesky

        annual_mean = returns_df.mean().values * settings.TRADING_DAYS
        logger.info(f"[OPT] Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}")

        if settings.MU_SHRINKAGE:
            sigma_sq_diag = np.diag(lw.covariance_) * settings.TRADING_DAYS
            annual_mean = _shrink_mu(annual_mean, sigma_sq_diag, n_obs=len(returns_df))

        return annual_cov, annual_mean

    # ── Vectorised Monte Carlo ──────────────────────────────────────────────

    def cholesky_monte_carlo(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        tickers: list[str],
        n_sim: int | None = None,
    ) -> pd.DataFrame:
        """
        Cholesky-correlated Monte Carlo over random simplex weight draws.
        ``Sigma = L @ L.T`` transforms independent daily shocks into
        correlated ones, so the simulation reflects reality (e.g. IT stocks
        selling off together) rather than treating assets as independent.
        """
        n_sim = n_sim or settings.N_SIMULATIONS
        n_assets = len(tickers)
        if mu.shape[0] != n_assets or cov.shape != (n_assets, n_assets):
            raise PortfolioOptimizerError(
                f"cholesky_monte_carlo: shape mismatch — mu={mu.shape}, cov={cov.shape}, "
                f"n_tickers={n_assets}"
            )

        T = settings.TRADING_DAYS
        dt = 1.0 / T
        rf = settings.RISK_FREE_RATE

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            logger.warning("[OPT] Covariance not positive-definite, regularising for Cholesky")
            L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-6)

        batch_size = max(1, min(n_sim, _MAX_BATCH_ELEMENTS // max(n_assets * T, 1)))
        logger.info(f"[OPT] Running {n_sim:,} Monte Carlo simulations in batches of {batch_size}...")

        var_idx = max(math.floor((1 - settings.CVAR_CONFIDENCE) * T), 1)
        weights_all, exp_return_all, port_std_all, cvar_all, sharpe_all = [], [], [], [], []

        remaining = n_sim
        while remaining > 0:
            b = min(batch_size, remaining)
            remaining -= b

            w = self._rng.dirichlet(np.ones(n_assets), size=b)  # (b, n_assets)
            z = self._rng.standard_normal((b, n_assets, T))  # (b, n_assets, T)

            # correlated[b, i, t] = mu[i]*dt + sqrt(dt) * sum_j L[i,j] * z[b,j,t]
            correlated = np.einsum("ij,bjt->bit", L, z) * math.sqrt(dt) + (mu * dt)[None, :, None]
            port_r = np.einsum("bi,bit->bt", w, correlated)  # (b, T)

            # exp_return/port_std come from the closed-form portfolio moments
            # (w.mu, w'Sigma w), not the single simulated path — the best of
            # 5,000 noisy realised-return draws is a large positive outlier
            # by construction, regardless of the true mu. CVaR has no
            # closed form under correlated multi-day compounding, so it still
            # needs the simulated path.
            exp_return = w @ mu
            port_std = np.sqrt(np.einsum("bi,ij,bj->b", w, cov, w))

            sorted_r = np.sort(port_r, axis=1)
            cvar_95 = -sorted_r[:, :var_idx].mean(axis=1) * math.sqrt(T)

            sharpe = (exp_return - rf) / np.clip(port_std, 1e-9, None)

            weights_all.extend(dict(zip(tickers, row)) for row in w)
            exp_return_all.append(exp_return)
            port_std_all.append(port_std)
            cvar_all.append(cvar_95)
            sharpe_all.append(sharpe)

        sim_df = pd.DataFrame({
            "weights": weights_all,
            "exp_return": np.concatenate(exp_return_all),
            "port_std": np.concatenate(port_std_all),
            "cvar_95": np.concatenate(cvar_all),
            "sharpe": np.concatenate(sharpe_all),
        })
        logger.info(
            f"[OPT] Simulation complete. Max Sharpe: {sim_df['sharpe'].max():.2f}, "
            f"Max Return: {sim_df['exp_return'].max():.1%}"
        )
        return sim_df

    def find_optimal_portfolios(self, sim_df: pd.DataFrame, annual_turnover: float = 1.2) -> dict:
        """Applies the Groww cost penalty and picks the max-Sharpe and min-CVaR portfolios."""
        rt_cost = self.cost_model.round_trip_cost_fraction(
            trade_value=settings.INITIAL_CAPITAL_INR / settings.MAX_STOCKS_IN_PORTFOLIO
        )
        sim_df = sim_df.copy()
        sim_df["exp_return_net"] = sim_df["exp_return"] - (rt_cost * annual_turnover)
        sim_df["sharpe_net"] = (
            (sim_df["exp_return_net"] - settings.RISK_FREE_RATE) / sim_df["port_std"].clip(lower=1e-9)
        )

        max_sharpe_row = sim_df.loc[sim_df["sharpe_net"].idxmax()]
        min_cvar_row = sim_df.loc[sim_df["cvar_95"].idxmin()]

        cols = ["weights", "exp_return", "exp_return_net", "port_std", "cvar_95", "sharpe", "sharpe_net"]
        return {
            "max_sharpe": max_sharpe_row[cols].to_dict(),
            "min_cvar": min_cvar_row[cols].to_dict(),
            "all_sims": sim_df,
        }

    def plot_frontier(self, sim_df: pd.DataFrame, max_sharpe: dict, min_cvar: dict) -> Path:
        """Saves the efficient-frontier scatter (CVaR vs net return, coloured by net Sharpe) as PNG."""
        fig, ax = plt.subplots(figsize=(12, 7), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        scatter = ax.scatter(
            sim_df["cvar_95"] * 100, sim_df["exp_return_net"] * 100,
            c=sim_df["sharpe_net"], cmap="RdYlGn", alpha=0.5, s=8, linewidths=0,
        )
        plt.colorbar(scatter, ax=ax, label="Sharpe Ratio (net of costs)")

        ax.scatter(max_sharpe["cvar_95"] * 100, max_sharpe["exp_return_net"] * 100,
                   marker="*", s=500, color="#FFD700", zorder=10, label="Max Sharpe (net)")
        ax.scatter(min_cvar["cvar_95"] * 100, min_cvar["exp_return_net"] * 100,
                   marker="D", s=200, color="#00CED1", zorder=10, label="Min CVaR")

        ax.set_xlabel("CVaR 95% (Annual %) — lower is safer", color="white", fontsize=12)
        ax.set_ylabel("Expected Net Return (Annual %)", color="white", fontsize=12)
        ax.set_title(f"Sovereign Alpha Engine — Efficient Frontier\n"
                     f"(Net of Groww Delivery Costs | {today_str()})", color="white", fontsize=14)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#2a2a3e", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#555")

        out_path = settings.OUTPUT_DIR / f"efficient_frontier_{today_str()}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[FRONTIER] Chart saved: {out_path}")
        return out_path

    def allocate_capital(
        self, weights: dict[str, float], capital_inr: float, prices: dict[str, float]
    ) -> dict[str, dict]:
        """
        Converts portfolio weights into whole-share quantities (fractional
        shares don't exist on NSE), largest positions first, respecting
        ``settings.MIN_POSITION_VALUE_INR``.
        """
        weights = normalise_weights(weights)
        allocation: dict[str, dict] = {}
        remaining_capital = capital_inr

        for ticker, weight in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            price = prices.get(ticker)
            if not is_valid_price(price):
                continue

            target_value = weight * capital_inr
            if target_value < settings.MIN_POSITION_VALUE_INR:
                continue

            qty = int(target_value // price)
            if qty < 1:
                continue

            actual_value = qty * price
            if actual_value > remaining_capital:
                qty = int(remaining_capital // price)
                if qty < 1:
                    continue
                actual_value = qty * price

            remaining_capital -= actual_value
            allocation[ticker] = {
                "qty": qty, "price": price, "value": actual_value,
                "weight_actual": actual_value / capital_inr,
            }

        return allocation
