"""
Performance metrics for a NAV time series and trade log.

Pure functions, no I/O — used by both the single-run backtest engine and
the walk-forward aggregator, and by tests, without any dependency on how
the NAV series was produced (backtest, paper-trading history, whatever).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings


def to_return_series(nav: pd.Series) -> pd.Series:
    """Simple daily returns from a date-sorted NAV series."""
    nav = nav.sort_index()
    return nav.pct_change().dropna()


def cagr(nav: pd.Series) -> float:
    """
    Compound annual growth rate from the first to the last NAV point,
    annualised by the actual elapsed calendar time (not trading-day count)
    — a short backtest window annualised off ``TRADING_DAYS`` would wildly
    overstate a lucky short run's implied yearly return.
    """
    nav = nav.sort_index()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return float("nan")
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def annualised_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(settings.TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf: float = settings.RISK_FREE_RATE) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - (rf / settings.TRADING_DAYS)
    std = excess.std(ddof=1)
    if std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(settings.TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = settings.RISK_FREE_RATE) -> float:
    """Like Sharpe, but penalises only downside deviation — a strategy with
    large, rare upside jumps and small, frequent downside noise scores
    better here than under Sharpe, which penalises both symmetrically."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - (rf / settings.TRADING_DAYS)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std < 1e-12:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(settings.TRADING_DAYS))


def max_drawdown(nav: pd.Series) -> dict:
    """Deepest peak-to-trough decline, plus when it happened."""
    nav = nav.sort_index()
    if nav.empty:
        return {"max_drawdown_pct": float("nan"), "peak_date": None, "trough_date": None}
    running_peak = nav.cummax()
    drawdown = (nav - running_peak) / running_peak
    trough_idx = drawdown.idxmin()
    peak_idx = nav.loc[:trough_idx].idxmax()
    return {
        "max_drawdown_pct": float(-drawdown.loc[trough_idx] * 100),
        "peak_date": peak_idx,
        "trough_date": trough_idx,
    }


def calmar_ratio(cagr_value: float, max_dd_pct: float) -> float:
    """CAGR / max drawdown — return earned per unit of worst pain endured."""
    if max_dd_pct is None or np.isnan(max_dd_pct) or max_dd_pct < 1e-9:
        return float("nan")
    return float(cagr_value / (max_dd_pct / 100.0))


def annualised_turnover(trade_log: list[dict], avg_nav: float, years: float) -> float:
    """
    Total traded value / average NAV / years — "how many times over did the
    book get replaced per year". High turnover matters because it's the
    channel through which the (already-modeled) Groww cost drag compounds;
    this metric makes that drag visible independent of net P&L.
    """
    if avg_nav <= 0 or years <= 0:
        return float("nan")
    total_traded = sum(t.get("value", 0.0) for t in trade_log)
    return float(total_traded / avg_nav / years)


def win_rate(trade_log: list[dict]) -> float:
    """Fraction of SELL trades (realised round-trips) that were profitable."""
    sells = [t for t in trade_log if t.get("action") == "SELL" and "realised_pnl" in t]
    if not sells:
        return float("nan")
    wins = sum(1 for t in sells if t["realised_pnl"] > 0)
    return float(wins / len(sells))


def benchmark_comparison(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    """
    Beta, annualised alpha, tracking error, information ratio against a
    benchmark return series, aligned on overlapping dates only.
    """
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1, sort=True
    ).dropna()
    if len(aligned) < 10:
        return {"beta": float("nan"), "alpha_annualised": float("nan"),
                "tracking_error_annualised": float("nan"), "information_ratio": float("nan")}

    cov = np.cov(aligned["strategy"], aligned["benchmark"])
    bench_var = cov[1, 1]
    beta = float(cov[0, 1] / bench_var) if bench_var > 1e-12 else float("nan")

    strat_ann = float(aligned["strategy"].mean() * settings.TRADING_DAYS)
    bench_ann = float(aligned["benchmark"].mean() * settings.TRADING_DAYS)
    alpha = strat_ann - beta * bench_ann if not np.isnan(beta) else float("nan")

    active = aligned["strategy"] - aligned["benchmark"]
    tracking_error = float(active.std(ddof=1) * np.sqrt(settings.TRADING_DAYS))
    info_ratio = float(active.mean() / active.std(ddof=1) * np.sqrt(settings.TRADING_DAYS)) if active.std(ddof=1) > 1e-12 else float("nan")

    return {
        "beta": beta,
        "alpha_annualised": alpha,
        "tracking_error_annualised": tracking_error,
        "information_ratio": info_ratio,
    }


def summarize(
    nav: pd.Series,
    trade_log: list[dict],
    initial_capital: float,
    benchmark_returns: pd.Series | None = None,
) -> dict:
    """Full metrics bundle for one NAV series + trade log."""
    nav = nav.sort_index()
    returns = to_return_series(nav)
    dd = max_drawdown(nav)
    years = (nav.index[-1] - nav.index[0]).days / 365.25 if len(nav) >= 2 else float("nan")

    result = {
        "start_date": nav.index[0].strftime("%Y-%m-%d") if len(nav) else None,
        "end_date": nav.index[-1].strftime("%Y-%m-%d") if len(nav) else None,
        "n_days": len(nav),
        "initial_capital": initial_capital,
        "final_nav": float(nav.iloc[-1]) if len(nav) else float("nan"),
        "total_return_pct": float((nav.iloc[-1] / initial_capital - 1.0) * 100) if len(nav) else float("nan"),
        "cagr_pct": cagr(nav) * 100 if len(nav) >= 2 else float("nan"),
        "annualised_volatility_pct": annualised_volatility(returns) * 100,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "max_drawdown_peak_date": dd["peak_date"].strftime("%Y-%m-%d") if dd["peak_date"] is not None else None,
        "max_drawdown_trough_date": dd["trough_date"].strftime("%Y-%m-%d") if dd["trough_date"] is not None else None,
        "calmar_ratio": calmar_ratio(cagr(nav) if len(nav) >= 2 else float("nan"), dd["max_drawdown_pct"]),
        "annualised_turnover": annualised_turnover(trade_log, nav.mean() if len(nav) else 0.0, years),
        "win_rate_pct": win_rate(trade_log) * 100 if trade_log else float("nan"),
        "n_trades": len(trade_log),
        "total_costs_paid": float(sum(t.get("total_cost", 0.0) for t in trade_log)),
    }

    if benchmark_returns is not None and not benchmark_returns.empty:
        result.update(benchmark_comparison(returns, benchmark_returns))

    return result
