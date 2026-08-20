import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.backtesting.engine import BacktestEngine, BacktestResult
from sovereign_alpha.ingestion.data_ingestor import DataIngestor

_TICKERS = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]


def _synthetic_ohlcv(tickers: list[str], n_days: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2024-06-30", periods=n_days)
    frames = []
    for i, t in enumerate(tickers):
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n_days)))
        frames.append(pd.DataFrame(
            {"Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
             "Close": prices, "Volume": 1_000_000.0, "Adj_Close": prices, "ticker": t},
            index=dates,
        ))
    return pd.concat(frames).sort_index()


@pytest.fixture(autouse=True)
def _mock_market_data(monkeypatch):
    """
    The engine's `run()` hits two network boundaries — DataIngestor.fetch_ohlcv
    and a raw yf.download for India VIX. Replacing both with deterministic
    synthetic data keeps this suite fast and offline, matching how the rest
    of the project tests pure computation rather than live network calls.
    """
    def fake_fetch_ohlcv(self, tickers, start=None, end=None, interval="1d", use_cache=True):
        full = _synthetic_ohlcv(tickers)
        if start is not None:
            full = full[full.index >= pd.Timestamp(start)]
        if end is not None:
            full = full[full.index <= pd.Timestamp(end)]
        return full

    monkeypatch.setattr(DataIngestor, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr("sovereign_alpha.backtesting.engine.yf.download", lambda *a, **k: pd.DataFrame())


class TestBacktestEngineRun:
    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            BacktestEngine(mode="not_a_real_mode")

    def test_produces_backtest_result(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS, start="2024-03-01", end="2024-06-01")
        assert isinstance(result, BacktestResult)
        assert result.mode == "technical"
        assert len(result.nav) > 0
        assert result.initial_capital == pytest.approx(100_000.0)

    def test_nav_series_is_sorted_datetime_indexed(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS, start="2024-03-01", end="2024-06-01")
        assert isinstance(result.nav.index, pd.DatetimeIndex)
        assert result.nav.index.is_monotonic_increasing

    def test_nav_starts_near_initial_capital(self):
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS, start="2024-03-01", end="2024-06-01")
        # First day may already have one rebalance's worth of cost drag, but
        # should be nowhere near a different order of magnitude.
        assert result.nav.iloc[0] == pytest.approx(100_000.0, rel=0.05)

    def test_custom_initial_capital_respected(self):
        engine = BacktestEngine(mode="technical", initial_capital=50_000.0, rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS, start="2024-03-01", end="2024-06-01")
        assert result.initial_capital == pytest.approx(50_000.0)
        assert result.nav.iloc[0] == pytest.approx(50_000.0, rel=0.05)

    def test_trade_dates_are_simulated_dates_not_real_today(self):
        """
        The exact bug the injectable clock (PaperBroker) exists to prevent:
        every trade in the log must be stamped with a date that falls
        inside the simulated window, never with the real wall-clock date
        the test happens to run on.
        """
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS, start="2024-03-01", end="2024-06-01")
        assert len(result.trade_log) > 0
        for trade in result.trade_log:
            trade_date = pd.Timestamp(trade["date"])
            assert pd.Timestamp("2024-01-01") <= trade_date <= pd.Timestamp("2024-07-01")

    def test_rebalance_frequency_controls_optimisation_cadence(self):
        engine_daily = BacktestEngine(mode="technical", rebalance_frequency_days=1, n_simulations=200)
        result_daily = engine_daily.run(_TICKERS, start="2024-05-01", end="2024-06-01")

        engine_sparse = BacktestEngine(mode="technical", rebalance_frequency_days=20, n_simulations=200)
        result_sparse = engine_sparse.run(_TICKERS, start="2024-05-01", end="2024-06-01")

        assert len(result_daily.rebalance_dates) > len(result_sparse.rebalance_dates)

    def test_empty_ticker_list_after_dedup_still_runs(self):
        # Duplicate tickers in the input must not double-count anything.
        engine = BacktestEngine(mode="technical", rebalance_frequency_days=10, n_simulations=200)
        result = engine.run(_TICKERS + _TICKERS, start="2024-03-01", end="2024-06-01")
        assert result.tickers == _TICKERS
