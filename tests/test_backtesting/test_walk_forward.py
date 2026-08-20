import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.backtesting.engine import BacktestEngine
from sovereign_alpha.backtesting.walk_forward import WalkForwardValidator
from sovereign_alpha.ingestion.data_ingestor import DataIngestor

_TICKERS = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]


def _synthetic_ohlcv(tickers: list[str], n_days: int = 1100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    frames = []
    for t in tickers:
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
        frames.append(pd.DataFrame(
            {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
             "Close": prices, "Volume": 1_000_000.0, "Adj_Close": prices, "ticker": t},
            index=dates,
        ))
    return pd.concat(frames).sort_index()


@pytest.fixture(autouse=True)
def _mock_market_data(monkeypatch):
    def fake_fetch_ohlcv(self, tickers, start=None, end=None, interval="1d", use_cache=True):
        full = _synthetic_ohlcv(tickers)
        if start is not None:
            full = full[full.index >= pd.Timestamp(start)]
        if end is not None:
            full = full[full.index <= pd.Timestamp(end)]
        return full

    def fake_benchmark_returns(self, start=None):
        dates = pd.bdate_range(start=start or "2023-01-01", end="2024-12-31")
        rng = np.random.default_rng(1)
        return pd.Series(rng.normal(0.0003, 0.012, len(dates)), index=dates)

    monkeypatch.setattr(DataIngestor, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(DataIngestor, "fetch_benchmark_returns", fake_benchmark_returns)
    monkeypatch.setattr("sovereign_alpha.backtesting.engine.yf.download", lambda *a, **k: pd.DataFrame())


class TestWalkForwardValidator:
    def test_produces_overall_and_fold_metrics(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=15, n_simulations=200)
        validator = WalkForwardValidator(engine=engine, fold_months=6)
        result = validator.run(_TICKERS, start="2024-01-01", end="2024-12-01")

        assert "sharpe_ratio" in result.overall
        assert "cagr_pct" in result.overall
        assert len(result.folds) >= 1
        for fold in result.folds:
            assert "fold_start" in fold and "fold_end" in fold
            assert "sharpe_ratio" in fold

    def test_folds_are_contiguous_and_ordered(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=15, n_simulations=200)
        validator = WalkForwardValidator(engine=engine, fold_months=6)
        result = validator.run(_TICKERS, start="2024-01-01", end="2024-12-01")

        ends = [pd.Timestamp(f["fold_end"]) for f in result.folds]
        starts = [pd.Timestamp(f["fold_start"]) for f in result.folds]
        assert starts == sorted(starts)
        for i in range(len(result.folds) - 1):
            assert starts[i + 1] > ends[i]

    def test_overall_includes_benchmark_comparison_when_available(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=15, n_simulations=200)
        validator = WalkForwardValidator(engine=engine, fold_months=6)
        result = validator.run(_TICKERS, start="2024-01-01", end="2024-12-01")
        assert "beta" in result.overall
        assert "alpha_annualised" in result.overall

    def test_empty_nav_raises_rather_than_silently_reporting_nothing(self, monkeypatch):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=15, n_simulations=200)

        def empty_run(self, tickers, start, end):
            from sovereign_alpha.backtesting.engine import BacktestResult
            return BacktestResult(nav=pd.Series(dtype=float), trade_log=[], initial_capital=100_000.0,
                                   mode="technical", tickers=tickers)

        monkeypatch.setattr(BacktestEngine, "run", empty_run)
        validator = WalkForwardValidator(engine=engine)
        with pytest.raises(ValueError):
            validator.run(_TICKERS, start="2024-01-01", end="2024-12-01")
