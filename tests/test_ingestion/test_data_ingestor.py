import json
import os
import time
from unittest.mock import patch

import pandas as pd
import pytest

from config import settings
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.utils.helpers import is_cache_valid


@pytest.fixture
def ingestor():
    return DataIngestor()


class TestFetchIndiaVix:
    def test_valid_yfinance_data_returns_float(self, ingestor):
        fake_close = pd.DataFrame({"Close": [15.5, 16.0, 16.25]})
        with patch("sovereign_alpha.ingestion.data_ingestor.yf.download", return_value=fake_close) as mock_dl:
            vix = ingestor.fetch_india_vix()
        assert mock_dl.called
        assert isinstance(vix, float)
        assert vix == pytest.approx(16.25)

    def test_yfinance_exception_returns_fallback(self, ingestor):
        with patch("sovereign_alpha.ingestion.data_ingestor.yf.download", side_effect=Exception("network down")):
            vix = ingestor.fetch_india_vix()
        assert vix == settings.VIX_DEFAULT_FALLBACK
        assert vix == 18.0

    def test_fresh_cache_skips_yfinance_call(self, ingestor):
        cache_path = settings.RAW_VIX_DIR / "vix_latest.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"vix": 14.2, "fetched_at": "2026-08-07T00:00:00"}), encoding="utf-8")

        with patch("sovereign_alpha.ingestion.data_ingestor.yf.download") as mock_dl:
            vix = ingestor.fetch_india_vix()

        mock_dl.assert_not_called()
        assert vix == 14.2


class TestFetchSingleFundamentals:
    def test_roe_none_in_info_is_preserved_as_none_key(self, ingestor):
        info = {
            "symbol": "TCS", "trailingPE": 25.0, "priceToBook": 10.0,
            "returnOnEquity": None, "freeCashflow": 5e9,
            "debtToEquity": 20.0, "marketCap": 1e12,
        }
        with patch("sovereign_alpha.ingestion.data_ingestor.yf.Ticker") as mock_ticker_cls, \
             patch.object(
                 DataIngestor, "_derive_fundamentals_from_statements",
                 return_value={"roe_derived": None, "fcf_derived": None},
             ):
            mock_ticker_cls.return_value.get_info.return_value = info
            result = ingestor._fetch_single_fundamentals("TCS.NS")

        assert result is not None
        assert "roe" in result  # key present, not dropped
        assert result["roe"] is None
        assert result["roe_source"] == "none"

    def test_info_exception_returns_none(self, ingestor):
        with patch("sovereign_alpha.ingestion.data_ingestor.yf.Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.get_info.side_effect = Exception("rate limited")
            result = ingestor._fetch_single_fundamentals("RELIANCE.NS")
        assert result is None

    def test_fetch_fundamentals_continues_past_a_failed_ticker(self, ingestor):
        # Patch _fetch_single_fundamentals directly to isolate the
        # looping/continuation behaviour of fetch_fundamentals from the
        # yfinance mocking details already covered above.
        with patch.object(
            DataIngestor, "_fetch_single_fundamentals",
            side_effect=[None, {"ticker": "GOOD.NS", "roe": 0.25, "roe_source": "info"}],
        ):
            df = ingestor.fetch_fundamentals(["BAD.NS", "GOOD.NS"])

        assert list(df.index) == ["BAD.NS", "GOOD.NS"]
        assert df.loc["GOOD.NS", "roe"] == 0.25


class TestIsCacheValid:
    def test_missing_path_is_invalid(self, tmp_path):
        assert is_cache_valid(tmp_path / "does_not_exist.parquet", max_age_hours=4) is False

    def test_one_hour_old_file_is_valid_for_four_hour_max(self, tmp_path):
        path = tmp_path / "fresh.parquet"
        path.write_text("data", encoding="utf-8")
        one_hour_ago = time.time() - 3600
        os.utime(path, (one_hour_ago, one_hour_ago))
        assert is_cache_valid(path, max_age_hours=4) is True

    def test_five_hour_old_file_is_invalid_for_four_hour_max(self, tmp_path):
        path = tmp_path / "stale.parquet"
        path.write_text("data", encoding="utf-8")
        five_hours_ago = time.time() - 5 * 3600
        os.utime(path, (five_hours_ago, five_hours_ago))
        assert is_cache_valid(path, max_age_hours=4) is False
