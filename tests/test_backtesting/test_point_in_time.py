import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.backtesting import point_in_time as pit


class _FakeTicker:
    def __init__(self, balance_sheet, financials, cashflow):
        self.balance_sheet = balance_sheet
        self.financials = financials
        self.cashflow = cashflow


def _fake_statements(n_periods: int = 4, growth: float = 1.08):
    """
    Synthetic annual statements shaped like yfinance's real output: columns
    are period-end timestamps, newest first; rows are line-item labels this
    module's candidate lists search for. Each period grows off the last by
    ``growth`` so Beneish M / Altman Z have real period-over-period movement
    to compute, not degenerate identical numbers.
    """
    period_ends = pd.DatetimeIndex(
        [pd.Timestamp(f"{2026 - i}-03-31") for i in range(n_periods)]
    )  # newest first, matching real yfinance ordering

    def series(base: float, g: float = growth) -> list[float]:
        return [base * (g ** -i) for i in range(n_periods)]  # newest (i=0) is largest

    # Different growth rates per line item (not just a uniform scale-up) so
    # ratio-based scores like Altman Z actually move period to period —
    # scaling every field by the same factor would leave every ratio
    # identical across periods, which is a fixture flaw, not a real
    # property of the code being tested.
    balance_sheet = pd.DataFrame({
        period_ends[i]: {
            "Total Assets": series(1_000_000.0, 1.08)[i],
            "Current Assets": series(400_000.0, 1.05)[i],
            "Current Liabilities": series(200_000.0, 1.10)[i],
            "Retained Earnings": series(300_000.0, 1.15)[i],
            "Stockholders Equity": series(500_000.0, 1.06)[i],
            "Total Debt": series(150_000.0, 1.20)[i],
            "Receivables": series(80_000.0, 1.03)[i],
        }
        for i in range(n_periods)
    })

    financials = pd.DataFrame({
        period_ends[i]: {
            "Total Revenue": series(900_000.0, 1.09)[i],
            "Gross Profit": series(300_000.0, 1.07)[i],
            "EBIT": series(120_000.0, 1.25)[i],
            "Net Income": series(80_000.0, 1.30)[i],
        }
        for i in range(n_periods)
    })

    cashflow = pd.DataFrame({
        period_ends[i]: {
            "Operating Cash Flow": series(100_000.0)[i],
            "Capital Expenditure": series(-40_000.0)[i],
        }
        for i in range(n_periods)
    })

    return balance_sheet, financials, cashflow


@pytest.fixture(autouse=True)
def _mock_yfinance_ticker(monkeypatch):
    bs, fin, cf = _fake_statements()

    def fake_ticker(symbol):
        return _FakeTicker(bs, fin, cf)

    monkeypatch.setattr(pit.yf, "Ticker", fake_ticker)


class TestFetchPointInTimeFundamentals:
    def test_returns_one_row_per_period_pair(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        # 4 periods -> 3 curr/prev pairs (needs a prior year for Beneish M)
        assert len(df) == 3

    def test_known_date_is_period_end_plus_reporting_lag(self):
        from config import settings
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        lag = pd.Timedelta(days=settings.ANNUAL_RESULTS_REPORTING_LAG_DAYS)
        for _, row in df.iterrows():
            assert row["known_date"] == row["period_end"] + lag

    def test_sorted_ascending_by_known_date(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        assert df["known_date"].is_monotonic_increasing

    def test_altman_z_matches_data_harmonizer_directly(self):
        """
        Single source of truth: this module must reuse
        DataHarmonizer.compute_altman_z, not reimplement the formula. If
        someone changes the formula in one place and not the other, this
        test catches the drift.
        """
        from sovereign_alpha.ingestion.data_harmonizer import DataHarmonizer

        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        row = df.iloc[-1]  # most recent pair
        fin = {
            "total_assets": 1_000_000.0, "ebit": 120_000.0, "working_capital": 200_000.0,
            "retained_earnings": 300_000.0, "book_equity": 500_000.0, "total_debt": 150_000.0,
        }
        expected = DataHarmonizer.compute_altman_z(fin, sector="IT")
        assert row["altman_z"] == pytest.approx(expected)

    def test_banking_sector_excluded_from_altman_z(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="Banking")
        assert df["altman_z"].isna().all()

    def test_ticker_with_no_statements_returns_empty_df(self, monkeypatch):
        monkeypatch.setattr(pit.yf, "Ticker", lambda s: _FakeTicker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
        df = pit.fetch_point_in_time_fundamentals("EMPTY.NS", sector="IT")
        assert df.empty


class TestAsOfFundamentals:
    def test_before_first_known_date_returns_nan(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        way_before = df["known_date"].min() - pd.Timedelta(days=3650)
        result = pit.as_of_fundamentals(df, way_before, current_price=100.0, shares_outstanding=1000.0)
        assert np.isnan(result["altman_z"])

    def test_uses_most_recent_known_period_not_a_future_one(self):
        """
        The core lookahead-safety property: as_of pinned exactly between two
        known_dates must return the OLDER period's figures, never the newer
        one that hadn't been reported yet as of that date.
        """
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        assert len(df) >= 2
        older, newer = df.iloc[0], df.iloc[1]
        midpoint = older["known_date"] + (newer["known_date"] - older["known_date"]) / 2

        result = pit.as_of_fundamentals(df, midpoint, current_price=100.0, shares_outstanding=1000.0)
        assert result["altman_z"] == pytest.approx(older["altman_z"])
        assert result["altman_z"] != pytest.approx(newer["altman_z"])

    def test_empty_pit_df_returns_all_nan(self):
        result = pit.as_of_fundamentals(pd.DataFrame(), pd.Timestamp("2024-01-01"), 100.0, 1000.0)
        assert all(np.isnan(v) for v in result.values())

    def test_pe_pb_fcf_yield_computed_from_price_and_shares(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        as_of = df["known_date"].max()
        result = pit.as_of_fundamentals(df, as_of, current_price=50.0, shares_outstanding=10_000.0)
        assert not np.isnan(result["pe"])
        assert not np.isnan(result["pb"])
        assert not np.isnan(result["fcf_yield"])

    def test_missing_shares_outstanding_gives_nan_ratios_not_a_crash(self):
        df = pit.fetch_point_in_time_fundamentals("FAKE.NS", sector="IT")
        as_of = df["known_date"].max()
        result = pit.as_of_fundamentals(df, as_of, current_price=50.0, shares_outstanding=None)
        assert np.isnan(result["pe"])
        assert np.isnan(result["pb"])
        assert np.isnan(result["fcf_yield"])
        assert not np.isnan(result["altman_z"])  # unaffected — doesn't need price/shares
