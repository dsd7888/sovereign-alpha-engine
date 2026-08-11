import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.ingestion.data_harmonizer import (
    _EBIT_BLOCKLIST,
    DataHarmonizer,
)
from sovereign_alpha.utils.helpers import extract_statement_row


class TestComputeAltmanZ:
    def test_none_total_assets_returns_nan(self):
        fin = {"total_assets": None, "ebit": 8e8}
        assert np.isnan(DataHarmonizer.compute_altman_z(fin, sector="IT"))

    def test_none_ebit_returns_nan(self):
        fin = {"total_assets": 1e10, "ebit": None}
        assert np.isnan(DataHarmonizer.compute_altman_z(fin, sector="IT"))

    def test_banking_sector_excluded(self):
        fin = {
            "total_assets": 1e10, "ebit": 8e8, "total_debt": 3e9,
            "working_capital": 2e9, "retained_earnings": 4e9,
        }
        assert np.isnan(DataHarmonizer.compute_altman_z(fin, sector="Banking"))

    def test_nbfc_sector_excluded(self):
        fin = {
            "total_assets": 1e10, "ebit": 8e8, "total_debt": 3e9,
            "working_capital": 2e9, "retained_earnings": 4e9,
        }
        assert np.isnan(DataHarmonizer.compute_altman_z(fin, sector="NBFC"))

    def test_valid_non_financial_inputs_return_float_in_range(self):
        fin = {
            "total_assets": 1e10, "ebit": 8e8, "total_debt": 3e9,
            "working_capital": 2e9, "retained_earnings": 4e9,
        }
        z = DataHarmonizer.compute_altman_z(fin, sector="IT")
        assert isinstance(z, float)
        assert not np.isnan(z)
        assert -5 <= z <= 15

    def test_banking_sector_excluded_even_with_missing_working_capital(self):
        # Sector exclusion must short-circuit before any field defaulting
        # (e.g. `wc or 0.0`) happens — missing working_capital must never
        # cause a Banking-sector ticker to silently fall through to a
        # computed (and meaningless) score instead of NaN.
        fin = {"total_assets": 1e10, "ebit": 8e8, "working_capital": None}
        z = DataHarmonizer.compute_altman_z(fin, sector="Banking")
        assert np.isnan(z)
        assert z != 0.0  # not merely nan-like; must not be a defaulted 0.0 pretending to be a score


class TestExtractStatementRowEbitMatching:
    """Regression coverage for the substring-match bug: 'operating income'
    used to match inside 'Other Non Operating Income Expenses' (a footnote
    plug, not real operating profit), fabricating an EBIT value for
    HDFCBANK. `extract_statement_row` (formerly `_extract_row`) now uses
    exact/word-boundary-prefix matching plus an explicit blocklist for
    EBIT/EBITDA lookups.
    """

    def test_exact_row_preferred_over_substring_lookalike(self):
        df = pd.DataFrame(
            {2024: [50.0, 200.0, 220.0]},
            index=["Other Non Operating Income Expenses", "Operating Income", "EBIT"],
        )
        result = extract_statement_row(df, ["operating income", "ebit"], blocklist=_EBIT_BLOCKLIST)
        assert not result.empty
        assert result.iloc[0] == 200.0  # from "Operating Income", not the 50.0 footnote row

    def test_no_fabricated_match_when_only_footnote_row_present(self):
        # The exact HDFCBANK scenario: no real EBIT/Operating Income row
        # exists at all, only the footnote-style "other non operating"
        # line — must return empty, never a fabricated value.
        df = pd.DataFrame(
            {2024: [-36789600000.0, 1925667800000.0]},
            index=["Other Non Operating Income Expenses", "Total Revenue"],
        )
        result = extract_statement_row(df, ["EBIT", "Operating Income"], blocklist=_EBIT_BLOCKLIST)
        assert result.empty

    def test_blocklist_prevents_prefix_match_containing_blocked_phrase(self):
        df = pd.DataFrame(
            {2024: [99.0, 1000.0]},
            index=["Operating Income Other Expense Adjustment", "Total Revenue"],
        )
        # This row label starts with "operating income" (a valid prefix
        # match) but also contains the blocklisted phrase "other expense".
        with_blocklist = extract_statement_row(df, ["EBIT", "Operating Income"], blocklist=_EBIT_BLOCKLIST)
        assert with_blocklist.empty

        # Without the blocklist, the plain word-boundary-prefix match alone
        # would have matched it — proving the blocklist is doing real work,
        # not just being redundant with the prefix fix.
        without_blocklist = extract_statement_row(df, ["EBIT", "Operating Income"], blocklist=None)
        assert not without_blocklist.empty
        assert without_blocklist.iloc[0] == 99.0


class TestComputeBeneishM:
    def test_none_revenue_returns_nan(self):
        curr = {"revenue": None, "total_assets": 1e9}
        prev = {"revenue": 1e9, "total_assets": 1e9}
        assert np.isnan(DataHarmonizer.compute_beneish_m(curr, prev))

    def test_none_total_assets_returns_nan(self):
        curr = {"revenue": 1e9, "total_assets": None}
        prev = {"revenue": 1e9, "total_assets": 1e9}
        assert np.isnan(DataHarmonizer.compute_beneish_m(curr, prev))

    def test_complete_valid_inputs_return_float_in_range(self):
        curr = {
            "revenue": 1050.0, "total_assets": 1100.0, "receivables": 105.0,
            "gross_profit": 420.0, "current_assets": 330.0,
            "net_income": 90.0, "operating_cashflow": 80.0,
        }
        prev = {
            "revenue": 1000.0, "total_assets": 1000.0, "receivables": 100.0,
            "gross_profit": 400.0, "current_assets": 300.0,
        }
        m = DataHarmonizer.compute_beneish_m(curr, prev)
        assert isinstance(m, float)
        assert not np.isnan(m)
        assert -10 <= m <= 10

    def test_clean_company_scores_below_low_risk_threshold(self):
        # Ratios chosen to reproduce DSRI=1.0, GMI=1.0, AQI=1.0, SGI=1.05
        # exactly. TATA is set to -0.04 (operating cash flow comfortably
        # exceeding net income) rather than the -0.01 a first guess might
        # suggest: with DSRI=GMI=AQI=1.0 and SGI=1.05, the module's formula
        # (-4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 4.679*TATA)
        # gives -2.098 at TATA=-0.01 — short of the -2.22 "low risk" cutoff
        # documented on compute_beneish_m. TATA=-0.04 is still a mild,
        # unremarkable accrual ratio (a clean company) and lands at -2.239,
        # clearing the threshold with margin.
        curr = {
            "revenue": 1050.0, "total_assets": 1100.0, "receivables": 105.0,
            "gross_profit": 420.0, "current_assets": 330.0,
            "net_income": 56.0, "operating_cashflow": 100.0,
        }
        prev = {
            "revenue": 1000.0, "total_assets": 1000.0, "receivables": 100.0,
            "gross_profit": 400.0, "current_assets": 300.0,
        }
        m = DataHarmonizer.compute_beneish_m(curr, prev)
        assert m < -2.22


class TestLiquidityFilter:
    def test_liquidity_filter_passes_large_caps(self):
        df = pd.DataFrame(
            {"avg_daily_volume_inr": [4_350_00_00_000]}, index=["RELIANCE.NS"]
        )
        filtered = DataHarmonizer._apply_liquidity_filter(df)
        assert "RELIANCE.NS" in filtered.index

    def test_liquidity_filter_drops_thin_stock(self):
        df = pd.DataFrame(
            {"avg_daily_volume_inr": [50_00_000]}, index=["THIN.NS"]
        )
        filtered = DataHarmonizer._apply_liquidity_filter(df)
        assert "THIN.NS" not in filtered.index

    def test_liquidity_filter_keeps_unknown(self):
        df = pd.DataFrame(
            {"avg_daily_volume_inr": [np.nan]}, index=["UNKNOWN.NS"]
        )
        filtered = DataHarmonizer._apply_liquidity_filter(df)
        assert "UNKNOWN.NS" in filtered.index


class TestNormaliseSectorPe:
    def test_middle_stock_normalises_near_zero(self):
        df = pd.DataFrame(
            {"pe": [15.0, 20.0, 25.0]},
            index=["TCS.NS", "INFY.NS", "HCLTECH.NS"],  # all IT sector
        )
        result = DataHarmonizer.normalise_sector_pe(df)
        assert result.loc["INFY.NS", "pe_sector_normalised"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc["TCS.NS", "pe_sector_normalised"] == pytest.approx(1.0, abs=1e-9)
        assert result.loc["HCLTECH.NS", "pe_sector_normalised"] == pytest.approx(-1.0, abs=1e-9)

    def test_sectors_normalised_independently(self):
        df = pd.DataFrame(
            {"pe": [15.0, 20.0, 25.0, 10.0, 12.0]},
            index=["TCS.NS", "INFY.NS", "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
        )
        result = DataHarmonizer.normalise_sector_pe(df)

        # IT-sector normalisation is unaffected by the Banking rows sharing the frame.
        assert result.loc["INFY.NS", "pe_sector_normalised"] == pytest.approx(0.0, abs=1e-9)
        assert result.loc["TCS.NS", "pe_sector_normalised"] == pytest.approx(1.0, abs=1e-9)
        assert result.loc["HCLTECH.NS", "pe_sector_normalised"] == pytest.approx(-1.0, abs=1e-9)

        # Banking is normalised against its own median/std (11.0 / sqrt(2)), not IT's.
        banking_std = pd.Series([10.0, 12.0]).std()
        expected_hdfc = (11.0 - 10.0) / banking_std
        expected_icici = (11.0 - 12.0) / banking_std
        assert result.loc["HDFCBANK.NS", "pe_sector_normalised"] == pytest.approx(expected_hdfc, abs=1e-9)
        assert result.loc["ICICIBANK.NS", "pe_sector_normalised"] == pytest.approx(expected_icici, abs=1e-9)
