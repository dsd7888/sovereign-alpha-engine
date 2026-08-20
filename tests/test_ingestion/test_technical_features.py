import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.ingestion.technical_features import compute_technical_features


def _synthetic_ohlcv(ticker: str, n_days: int, daily_drift: float, daily_vol: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-08-20", periods=n_days)
    log_rets = rng.normal(daily_drift, daily_vol, n_days)
    prices = 100.0 * np.exp(np.cumsum(log_rets))
    return pd.DataFrame({"Adj_Close": prices, "ticker": ticker}, index=dates)


class TestComputeTechnicalFeatures:
    def test_insufficient_history_returns_nan(self):
        short = _synthetic_ohlcv("SHORT.NS", n_days=50, daily_drift=0.001, daily_vol=0.01, seed=1)
        out = compute_technical_features(short, ["SHORT.NS"])
        assert np.isnan(out.loc["SHORT.NS", "momentum_12_1"])
        assert np.isnan(out.loc["SHORT.NS", "low_vol_6m"])

    def test_strong_uptrend_has_positive_momentum(self):
        up = _synthetic_ohlcv("UP.NS", n_days=400, daily_drift=0.003, daily_vol=0.01, seed=2)
        out = compute_technical_features(up, ["UP.NS"])
        assert out.loc["UP.NS", "momentum_12_1"] > 0

    def test_higher_daily_vol_gives_higher_low_vol_reading(self):
        calm = _synthetic_ohlcv("CALM.NS", n_days=400, daily_drift=0.0, daily_vol=0.005, seed=3)
        wild = _synthetic_ohlcv("WILD.NS", n_days=400, daily_drift=0.0, daily_vol=0.04, seed=4)
        combined = pd.concat([calm, wild])
        out = compute_technical_features(combined, ["CALM.NS", "WILD.NS"])
        assert out.loc["WILD.NS", "low_vol_6m"] > out.loc["CALM.NS", "low_vol_6m"]

    def test_as_of_excludes_future_prices(self):
        """
        The core lookahead-safety guarantee this module exists to provide:
        restricting to an as_of date must change the result relative to
        using the full series, proving future rows were actually excluded
        rather than silently ignored.
        """
        prices = _synthetic_ohlcv("T.NS", n_days=400, daily_drift=0.002, daily_vol=0.01, seed=5)
        cutoff = prices.index[300]

        full = compute_technical_features(prices, ["T.NS"])
        as_of = compute_technical_features(prices, ["T.NS"], as_of=cutoff)

        assert full.loc["T.NS", "momentum_12_1"] != pytest.approx(as_of.loc["T.NS", "momentum_12_1"])

    def test_as_of_never_sees_rows_after_cutoff(self):
        prices = _synthetic_ohlcv("T.NS", n_days=400, daily_drift=0.001, daily_vol=0.01, seed=6)
        cutoff = prices.index[350]  # leaves 351 days, comfortably above the 273-day minimum
        # A hand-truncated series must give the identical answer to as_of=cutoff —
        # if it doesn't, as_of is leaking rows past the cutoff into the calc.
        truncated = prices[prices.index <= cutoff]

        as_of_result = compute_technical_features(prices, ["T.NS"], as_of=cutoff)
        truncated_result = compute_technical_features(truncated, ["T.NS"])

        assert as_of_result.loc["T.NS", "momentum_12_1"] == pytest.approx(
            truncated_result.loc["T.NS", "momentum_12_1"]
        )
        assert as_of_result.loc["T.NS", "low_vol_6m"] == pytest.approx(
            truncated_result.loc["T.NS", "low_vol_6m"]
        )

    def test_missing_ticker_in_ohlcv_returns_nan_row(self):
        prices = _synthetic_ohlcv("A.NS", n_days=400, daily_drift=0.001, daily_vol=0.01, seed=7)
        out = compute_technical_features(prices, ["A.NS", "MISSING.NS"])
        assert "MISSING.NS" in out.index
        assert np.isnan(out.loc["MISSING.NS", "momentum_12_1"])
