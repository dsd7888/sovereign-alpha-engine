from unittest.mock import patch

import pandas as pd
import pytest

from sovereign_alpha.ingestion.universe_builder import (
    INDEX_SOURCES,
    UniverseBuilderError,
    build_universe,
    fetch_index_constituents,
)


def _fake_response(status_code: int, text: str = "") -> object:
    class _R:
        pass
    r = _R()
    r.status_code = status_code
    r.text = text
    return r


def _fake_csv(symbols: list[str], industry: str = "Capital Goods") -> str:
    header = "Company Name,Industry,Symbol,Series,ISIN Code\n"
    rows = "".join(f"{s} Ltd.,{industry},{s},EQ,INE000000000\n" for s in symbols)
    return header + rows


class TestFetchIndexConstituents:
    def test_http_error_returns_none(self):
        with patch("requests.get", return_value=_fake_response(404)):
            assert fetch_index_constituents("bad_slug.csv") is None

    def test_network_exception_returns_none(self):
        with patch("requests.get", side_effect=ConnectionError("boom")):
            assert fetch_index_constituents("slug.csv") is None

    def test_valid_csv_parsed(self):
        csv_text = _fake_csv(["AAA", "BBB"])
        with patch("requests.get", return_value=_fake_response(200, csv_text)):
            df = fetch_index_constituents("slug.csv")
        assert list(df["Symbol"]) == ["AAA", "BBB"]

    def test_missing_symbol_column_returns_none(self):
        with patch("requests.get", return_value=_fake_response(200, "A,B\n1,2\n")):
            assert fetch_index_constituents("slug.csv") is None


def _mock_all_sources(mapping: dict[str, pd.DataFrame | None]):
    def fake_fetch(slug: str):
        return mapping.get(slug)
    return patch("sovereign_alpha.ingestion.universe_builder.fetch_index_constituents", side_effect=fake_fetch)


class TestBuildUniverse:
    def test_every_source_failing_raises(self):
        with _mock_all_sources({slug: None for slug, _, _ in INDEX_SOURCES}), pytest.raises(UniverseBuilderError):
            build_universe()

    def test_partial_source_failure_still_produces_a_universe(self):
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_nifty50list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(["AAA", "BBB"], "IT")))
        with _mock_all_sources(mapping):
            result = build_universe()
        assert result["tickers"] == ["AAA.NS", "BBB.NS"]
        assert result["sector_map"]["AAA.NS"] == "IT"

    def test_deduplicates_a_ticker_appearing_in_two_sources(self):
        # AAA appears in both Nifty50 (higher priority) and Next50 — must
        # only be counted/added once, keeping the first (large) tier's data.
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_nifty50list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(["AAA"], "IT")))
        mapping["ind_niftynext50list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(["AAA", "BBB"], "Auto")))
        with _mock_all_sources(mapping):
            result = build_universe()
        assert result["tickers"].count("AAA.NS") == 1
        assert result["sector_map"]["AAA.NS"] == "IT"  # from the higher-priority source
        assert "AAA.NS" in result["large_cap_tickers"]

    def test_overall_cap_respected(self):
        big_list = [f"T{i}" for i in range(500)]
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_niftysmallcap250list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(big_list)))
        with _mock_all_sources(mapping):
            result = build_universe(max_size=50)
        assert len(result["tickers"]) == 50

    def test_per_source_cap_respected_even_under_overall_cap(self):
        big_list = [f"T{i}" for i in range(500)]
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_niftysmallcap250list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(big_list)))
        with _mock_all_sources(mapping):
            result = build_universe(max_size=100_000)  # overall cap not binding
        small_cap_source = next(cap for slug, tier, cap in INDEX_SOURCES if tier == "small")
        assert len(result["non_large_cap_tickers"]) == small_cap_source

    def test_large_cap_and_non_large_cap_partition_all_tickers(self):
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_nifty50list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(["AAA"])))
        mapping["ind_niftymidcap150list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(["BBB"])))
        with _mock_all_sources(mapping):
            result = build_universe()
        assert set(result["large_cap_tickers"]) | set(result["non_large_cap_tickers"]) == set(result["tickers"])
        assert set(result["large_cap_tickers"]) & set(result["non_large_cap_tickers"]) == set()

    def test_sample_selection_is_reproducible_across_calls(self):
        big_list = [f"T{i}" for i in range(500)]
        mapping = {slug: None for slug, _, _ in INDEX_SOURCES}
        mapping["ind_niftysmallcap250list.csv"] = pd.read_csv(pd.io.common.StringIO(_fake_csv(big_list)))
        with _mock_all_sources(mapping):
            r1 = build_universe()
        with _mock_all_sources(mapping):
            r2 = build_universe()
        assert r1["tickers"] == r2["tickers"]
