"""
Day-by-day backtest engine.

Deliberately thin: every trading decision routes through the exact
production modules (``FundamentalScorer``, ``PortfolioOptimizer``,
``PaperBroker``, ``RiskGuardian``, ``GrowwCostModel``) — this file only
supplies point-in-time data and drives the daily loop. A backtest and a
live run can never silently diverge on what "the strategy" does, because
there is only one implementation of it.

Two modes (see ``sovereign_alpha.backtesting.point_in_time`` and
``sovereign_alpha.ingestion.technical_features`` module docstrings for the
full data-availability reasoning):

``"technical"`` — stock selection driven by momentum + low-vol only
(point-in-time-safe over the full available price history, ~10yr). Every
fundamentals factor is absent from the per-day scoring input, which
``FundamentalScorer`` already treats as neutral (50) for missing data — no
special-casing needed. Because 8 of 11 USHS weights are then pinned at
exactly 50, the composite score can mathematically never reach
``MIN_USHS_THRESHOLD`` (0.82*50 + 0.18*100 = 59 < 60), so eligibility
always falls through to top-K-by-USHS ranking — the same fallback
``run_engine.py`` already uses when too few tickers clear the threshold,
engaged deliberately here rather than as an edge case.

``"full"`` — replays the complete USHS including fundamentals,
reconstructed at annual resolution from real statements (~5yr for most
NSE names). Promoter pledge and sentiment are neutral throughout — no
historical source exists for either.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from config import settings
from config.universe import sector_of
from sovereign_alpha.backtesting.point_in_time import (
    as_of_fundamentals,
    build_universe_pit_store,
    fetch_current_shares_outstanding,
)
from sovereign_alpha.execution.paper_broker import PaperBroker
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.ingestion.technical_features import compute_technical_features
from sovereign_alpha.optimization.portfolio_optimizer import PortfolioOptimizer, PortfolioOptimizerError
from sovereign_alpha.risk.risk_guardian import Action, RiskGuardian
from sovereign_alpha.scoring.fundamental_scorer import FundamentalScorer
from sovereign_alpha.utils.logger import logger
from sovereign_alpha.utils.validators import is_valid_price


@dataclass
class BacktestResult:
    nav: pd.Series
    trade_log: list[dict]
    initial_capital: float
    mode: str
    tickers: list[str]
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)


class BacktestEngine:
    def __init__(
        self,
        mode: str = "technical",
        initial_capital: float | None = None,
        rebalance_frequency_days: int | None = None,
        n_simulations: int | None = None,
    ) -> None:
        if mode not in ("technical", "full"):
            raise ValueError(f"mode must be 'technical' or 'full', got {mode!r}")
        self.mode = mode
        self.initial_capital = initial_capital or settings.INITIAL_CAPITAL_INR
        self.rebalance_frequency_days = rebalance_frequency_days or settings.BACKTEST_REBALANCE_FREQUENCY_DAYS
        self.n_simulations = n_simulations
        self.optimizer = PortfolioOptimizer()
        self.risk_guard = RiskGuardian()
        self.scorer = FundamentalScorer()

    def run(self, tickers: list[str], start: str, end: str) -> BacktestResult:
        tickers = list(dict.fromkeys(tickers))
        logger.info(f"[BACKTEST:{self.mode}] {len(tickers)} tickers, {start} -> {end}")

        warmup_start = (pd.Timestamp(start) - pd.DateOffset(years=settings.LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        # use_cache=False: DataIngestor's on-disk cache isn't range-aware (see
        # its docstring) — a backtest legitimately requests different date
        # ranges across runs/CLI invocations, which a cache hit would betray.
        ohlcv = DataIngestor().fetch_ohlcv(tickers, start=warmup_start, end=end, use_cache=False)
        vix_series = self._fetch_vix_series(warmup_start, end)

        wide_prices = (
            ohlcv.pivot_table(index=ohlcv.index, columns="ticker", values="Adj_Close")
            .sort_index()
            .ffill()
        )

        pit_store: dict[str, pd.DataFrame] = {}
        shares_outstanding: dict[str, float] = {}
        if self.mode == "full":
            pit_store = build_universe_pit_store(tickers, {t: sector_of(t) for t in tickers})
            shares_outstanding = fetch_current_shares_outstanding(tickers)

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        sim_dates = sorted(d for d in wide_prices.index if start_ts <= d <= end_ts)
        if not sim_dates:
            raise ValueError(f"No trading days with data between {start} and {end}")

        current_date = {"str": None}
        broker = PaperBroker(
            persist=False, initial_capital=self.initial_capital, clock=lambda: current_date["str"]
        )
        target_weights: dict[str, float] = {}
        rebalance_dates: list[pd.Timestamp] = []

        for day_idx, d in enumerate(sim_dates):
            current_date["str"] = d.strftime("%Y%m%d")
            current_prices = wide_prices.loc[d].dropna().to_dict()

            portfolio_value = broker.update_prices(current_prices)
            peak_value = broker.state["peak_value"]
            vix_today = self._vix_on(vix_series, d)
            circuit_action = self.risk_guard.check_circuit_breaker(portfolio_value, peak_value, vix_today)

            self._apply_stop_losses(broker, current_prices)

            is_scheduled = day_idx % self.rebalance_frequency_days == 0
            if is_scheduled:
                try:
                    target_weights = self._compute_target_weights(
                        ohlcv, tickers, d, current_prices, shares_outstanding, pit_store
                    )
                    rebalance_dates.append(d)
                except PortfolioOptimizerError as e:
                    logger.warning(f"[BACKTEST] {d.date()}: optimisation failed ({e}), holding prior targets")

            if is_scheduled or circuit_action != Action.HOLD:
                broker.rebalance(target_weights, current_prices, circuit_action)

            broker.update_prices(current_prices)

            if day_idx % 100 == 0 or day_idx == len(sim_dates) - 1:
                logger.info(
                    f"[BACKTEST:{self.mode}] {d.date()} ({day_idx + 1}/{len(sim_dates)}) "
                    f"NAV=Rs.{portfolio_value:,.0f}"
                )

        nav = pd.Series(broker.state["daily_nav"], dtype=float)
        nav.index = pd.to_datetime(nav.index, format="%Y%m%d")
        return BacktestResult(
            nav=nav.sort_index(), trade_log=broker.state["trade_log"], initial_capital=self.initial_capital,
            mode=self.mode, tickers=tickers, rebalance_dates=rebalance_dates,
        )

    # ── Per-day helpers ──────────────────────────────────────────────────

    def _compute_target_weights(
        self,
        ohlcv: pd.DataFrame,
        tickers: list[str],
        as_of: pd.Timestamp,
        current_prices: dict[str, float],
        shares_outstanding: dict[str, float],
        pit_store: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        trailing = ohlcv[
            (ohlcv.index <= as_of) & (ohlcv.index > as_of - pd.DateOffset(years=settings.LOOKBACK_YEARS))
        ]

        df = compute_technical_features(trailing, tickers)

        if self.mode == "full":
            rows = {
                t: as_of_fundamentals(
                    pit_store.get(t, pd.DataFrame()), as_of,
                    current_prices.get(t, np.nan), shares_outstanding.get(t),
                )
                for t in df.index
            }
            df = df.join(pd.DataFrame.from_dict(rows, orient="index"))

        scored_df = self.scorer.compute_ushs(df, save_output=False)

        eligible = self.scorer.get_eligible_tickers(scored_df)
        if len(eligible) < 3:
            eligible = scored_df.sort_values("ushs", ascending=False).head(10).index.tolist()
        eligible = eligible[: settings.MAX_STOCKS_IN_PORTFOLIO]

        returns_df = self.optimizer.load_returns(eligible, ohlcv=trailing)
        cov, mu = self.optimizer.compute_covariance_matrix(returns_df)
        sim_df = self.optimizer.cholesky_monte_carlo(mu, cov, list(returns_df.columns), n_sim=self.n_simulations)
        optimal = self.optimizer.find_optimal_portfolios(sim_df)
        return optimal["max_sharpe"]["weights"]

    def _apply_stop_losses(self, broker: PaperBroker, current_prices: dict[str, float]) -> None:
        # No promoter-pledge check in backtest — pledge has no historical
        # source at all (see point_in_time.py), unlike production which at
        # least has today's real Screener reading.
        for ticker, holding in list(broker.state["holdings"].items()):
            if holding.get("qty", 0) <= 0:
                continue
            price = current_prices.get(ticker)
            if not is_valid_price(price):
                continue
            action = self.risk_guard.check_single_position(ticker, holding.get("avg_cost", 0.0), price)
            if action == Action.FULL_EXIT:
                broker.sell_all(ticker, price)

    @staticmethod
    def _fetch_vix_series(start: str, end: str) -> pd.Series:
        try:
            raw = yf.download("^INDIAVIX", start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if raw is None or raw.empty:
                return pd.Series(dtype=float)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            return raw["Close"].dropna().sort_index()
        except Exception as e:
            logger.warning(f"[BACKTEST] India VIX history unavailable, using flat fallback: {e}")
            return pd.Series(dtype=float)

    @staticmethod
    def _vix_on(vix_series: pd.Series, d: pd.Timestamp) -> float:
        if vix_series.empty:
            return settings.VIX_DEFAULT_FALLBACK
        past = vix_series[vix_series.index <= d]
        return float(past.iloc[-1]) if len(past) else settings.VIX_DEFAULT_FALLBACK
