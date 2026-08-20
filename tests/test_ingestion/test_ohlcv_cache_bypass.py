from unittest.mock import patch

import pandas as pd

from sovereign_alpha.ingestion.data_ingestor import DataIngestor


def _fake_ohlcv(price: float, n: int = 5) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "Open": price, "High": price, "Low": price, "Close": price,
        "Volume": 1000.0, "Adj_Close": price,
    }, index=dates)


class TestUseCacheFalse:
    def test_use_cache_false_ignores_a_fresh_but_wrong_range_cache(self):
        """
        The bug this flag exists to prevent: a ticker cached moments ago
        for one date range must not be silently served to a caller asking
        about a different range, just because the cache file is still
        within OHLCV_CACHE_HOURS.
        """
        ingestor = DataIngestor()
        with patch.object(ingestor, "_fetch_single_ohlcv", return_value=_fake_ohlcv(100.0)) as mock_fetch:
            first = ingestor.fetch_ohlcv(["A.NS"], start="2024-01-01", end="2024-01-10")
        assert (first["Close"] == 100.0).all()
        assert mock_fetch.call_count == 1

        # Second call, different range, cache file from the first call is
        # still fresh (just written) — with use_cache=False it must hit the
        # network again rather than replay the first call's cached data.
        with patch.object(ingestor, "_fetch_single_ohlcv", return_value=_fake_ohlcv(200.0)) as mock_fetch:
            second = ingestor.fetch_ohlcv(["A.NS"], start="2024-02-01", end="2024-02-10", use_cache=False)
        assert (second["Close"] == 200.0).all()
        assert mock_fetch.call_count == 1  # actually called, not skipped via cache hit

    def test_use_cache_true_default_does_reuse_a_fresh_cache(self):
        """Confirms the default (production) behaviour is unchanged by the new flag."""
        ingestor = DataIngestor()
        with patch.object(ingestor, "_fetch_single_ohlcv", return_value=_fake_ohlcv(100.0)) as mock_fetch:
            ingestor.fetch_ohlcv(["A.NS"], start="2024-01-01", end="2024-01-10")

        with patch.object(ingestor, "_fetch_single_ohlcv", return_value=_fake_ohlcv(999.0)) as mock_fetch:
            second = ingestor.fetch_ohlcv(["A.NS"], start="2024-01-01", end="2024-01-10")
        assert mock_fetch.call_count == 0  # cache hit — network not called at all
        assert (second["Close"] == 100.0).all()  # served the cached value, not 999
