"""
Walk-forward validation.

USHS's factor weights are fixed configuration, not fitted parameters —
there is no "training phase" to speak of, so walk-forward here does not
mean the classic train-on-fold-N/test-on-fold-N+1 hyperparameter search.
What it means, and what actually matters for this strategy: the entire
``BacktestEngine`` run is already point-in-time (every day only sees data
knowable by that day), so it is a single continuous out-of-sample
evaluation by construction. Re-running it separately per calendar-year
fold and resetting capital to the starting amount each time would be
worse, not more rigorous — it would hide compounding effects and make
folds artificially independent when the portfolio genuinely carries
positions across fold boundaries in reality.

So this module runs the engine ONCE over the full window, then slices the
resulting NAV/trade log into folds purely for *reporting* — showing
whether performance holds up across distinct calendar periods/regimes
(a bull run, a correction, a crash) rather than being an artifact of one
favourable window. A strategy that only looks good in a single fold and
falls apart in others has not been validated, no matter how good the
full-period number looks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import settings
from sovereign_alpha.backtesting import metrics
from sovereign_alpha.backtesting.engine import BacktestEngine, BacktestResult
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.utils.logger import logger


@dataclass
class WalkForwardResult:
    overall: dict
    folds: list[dict] = field(default_factory=list)
    backtest: BacktestResult | None = None


class WalkForwardValidator:
    def __init__(self, engine: BacktestEngine | None = None, fold_months: int | None = None) -> None:
        self.engine = engine or BacktestEngine()
        self.fold_months = fold_months or settings.BACKTEST_WALK_FORWARD_FOLD_MONTHS

    def run(self, tickers: list[str], start: str, end: str) -> WalkForwardResult:
        result = self.engine.run(tickers, start, end)
        if result.nav.empty:
            raise ValueError("Backtest produced an empty NAV series — nothing to validate.")

        benchmark_returns = self._fetch_benchmark_returns(start, end)
        overall = metrics.summarize(result.nav, result.trade_log, result.initial_capital, benchmark_returns)
        overall["mode"] = result.mode
        overall["n_rebalances"] = len(result.rebalance_dates)

        folds = []
        for fold_start, fold_end in self._fold_boundaries(result.nav.index.min(), result.nav.index.max()):
            fold_nav = result.nav[(result.nav.index >= fold_start) & (result.nav.index <= fold_end)]
            if len(fold_nav) < 5:
                logger.info(f"[WALK-FORWARD] Skipping fold {fold_start.date()}-{fold_end.date()}: too few data points")
                continue

            fold_trades = [
                t for t in result.trade_log
                if fold_start <= pd.to_datetime(t["date"], format="%Y%m%d") <= fold_end
            ]
            fold_bench = None
            if benchmark_returns is not None and not benchmark_returns.empty:
                fold_bench = benchmark_returns[
                    (benchmark_returns.index >= fold_start) & (benchmark_returns.index <= fold_end)
                ]

            fold_metrics = metrics.summarize(fold_nav, fold_trades, float(fold_nav.iloc[0]), fold_bench)
            folds.append({"fold_start": fold_start.strftime("%Y-%m-%d"), "fold_end": fold_end.strftime("%Y-%m-%d"), **fold_metrics})

        logger.info(
            f"[WALK-FORWARD] {len(folds)} folds | Overall CAGR: {overall.get('cagr_pct', float('nan')):.1f}% | "
            f"Overall Sharpe: {overall.get('sharpe_ratio', float('nan')):.2f}"
        )
        return WalkForwardResult(overall=overall, folds=folds, backtest=result)

    def _fold_boundaries(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        boundaries = []
        cursor = start
        while cursor < end:
            fold_end = min(cursor + pd.DateOffset(months=self.fold_months) - pd.Timedelta(days=1), end)
            boundaries.append((cursor, fold_end))
            cursor = fold_end + pd.Timedelta(days=1)
        return boundaries

    @staticmethod
    def _fetch_benchmark_returns(start: str, end: str) -> pd.Series | None:
        try:
            returns = DataIngestor().fetch_benchmark_returns(start=start)
            return returns[returns.index <= pd.Timestamp(end)]
        except Exception as e:
            logger.warning(f"[WALK-FORWARD] Benchmark fetch failed, skipping alpha/beta: {e}")
            return None
