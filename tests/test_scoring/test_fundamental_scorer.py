import numpy as np
import pandas as pd
import pytest

from config import settings
from sovereign_alpha.scoring.fundamental_scorer import FundamentalScorer


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "pe": [10.0, 20.0, np.nan, 15.0],
        "pe_sector_normalised": [1.0, -1.0, 0.0, np.nan],
        "pb": [1.0, 30.0, 2.0, np.nan],
        "roe": [0.20, -0.60, 0.10, np.nan],
        "fcf_yield": [0.05, np.nan, 0.02, 0.08],
        "debt_equity": [50.0, 600.0, -150.0, np.nan],
        "promoter_pledge_pct": [0.0, 60.0, np.nan, 25.0],
        "altman_z": [3.0, 1.0, np.nan, 2.0],
        "beneish_m": [-3.0, -1.5, np.nan, -2.0],
        "momentum_12_1": [0.30, -0.10, np.nan, 0.05],
        "low_vol_6m": [0.15, 0.45, np.nan, 0.25],
        "sector": ["IT", "Banking", "IT", "Auto"],
    }, index=["A.NS", "B.NS", "C.NS", "D.NS"])


class TestWeights:
    def test_ushs_weights_sum_to_one(self):
        assert abs(sum(settings.USHS_WEIGHTS.values()) - 1.0) < 1e-9


class TestPercentileScore:
    def test_all_nan_series_returns_neutral(self):
        scorer = FundamentalScorer()
        s = pd.Series([np.nan, np.nan, np.nan], index=["a", "b", "c"])
        result = scorer._percentile_score(s)
        assert (result == 50.0).all()

    def test_higher_is_better_ranks_ascending(self):
        scorer = FundamentalScorer()
        s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        result = scorer._percentile_score(s, higher_is_better=True)
        assert result["c"] > result["b"] > result["a"]

    def test_lower_is_better_inverts_ranking(self):
        scorer = FundamentalScorer()
        s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        result = scorer._percentile_score(s, higher_is_better=False)
        assert result["a"] > result["b"] > result["c"]


class TestFactorScorers:
    def setup_method(self):
        self.df = _sample_df()
        self.scorer = FundamentalScorer()

    def test_debt_equity_extreme_leverage_forced_to_zero(self):
        scored = self.scorer.score_debt_equity(self.df)
        assert scored["B.NS"] == 0.0  # debt_equity=600 > 500

    def test_debt_equity_negative_equity_forced_to_zero(self):
        scored = self.scorer.score_debt_equity(self.df)
        assert scored["C.NS"] == 0.0  # debt_equity=-150 < -100

    def test_promoter_pledge_zero_is_max_score(self):
        scored = self.scorer.score_promoter_pledge(self.df)
        assert scored["A.NS"] == 100.0

    def test_promoter_pledge_missing_is_neutral(self):
        scored = self.scorer.score_promoter_pledge(self.df)
        assert scored["C.NS"] == 50.0

    def test_altman_z_safe_zone_is_100(self):
        scored = self.scorer.score_altman_z(self.df)
        assert scored["A.NS"] == 100.0

    def test_altman_z_distress_zone_is_zero(self):
        scored = self.scorer.score_altman_z(self.df)
        assert scored["B.NS"] == 0.0

    def test_altman_z_missing_is_slightly_below_neutral(self):
        scored = self.scorer.score_altman_z(self.df)
        assert scored["C.NS"] == 40.0

    def test_beneish_m_safe_is_100(self):
        scored = self.scorer.score_beneish_m(self.df)
        assert scored["A.NS"] == 100.0

    def test_beneish_m_flagged_is_zero(self):
        scored = self.scorer.score_beneish_m(self.df)
        assert scored["B.NS"] == 0.0

    def test_beneish_m_missing_is_neutral(self):
        scored = self.scorer.score_beneish_m(self.df)
        assert scored["C.NS"] == 50.0

    def test_momentum_higher_return_scores_higher(self):
        scored = self.scorer.score_momentum(self.df)
        assert scored["A.NS"] > scored["D.NS"] > scored["B.NS"]

    def test_momentum_missing_is_neutral(self):
        scored = self.scorer.score_momentum(self.df)
        assert scored["C.NS"] == 50.0

    def test_low_vol_lower_volatility_scores_higher(self):
        scored = self.scorer.score_low_vol(self.df)
        assert scored["A.NS"] > scored["D.NS"] > scored["B.NS"]

    def test_low_vol_missing_is_neutral(self):
        scored = self.scorer.score_low_vol(self.df)
        assert scored["C.NS"] == 50.0


class TestMissingColumn:
    def test_absent_column_does_not_misalign_index(self):
        # Regression test: df.get(col, pd.Series(dtype=float)) would return an
        # EMPTY series with no index when a column is entirely absent, silently
        # misaligning every downstream vectorised op. `_col` must not do that.
        df = pd.DataFrame({"pe": [10.0, 20.0]}, index=["A.NS", "B.NS"])
        scorer = FundamentalScorer()
        scored = scorer.score_roe(df)  # "roe" column doesn't exist at all
        assert list(scored.index) == ["A.NS", "B.NS"]
        assert (scored == 50.0).all()


class TestComputeUSHS:
    def test_empty_dataframe_raises(self):
        scorer = FundamentalScorer()
        with pytest.raises(ValueError):
            scorer.compute_ushs(pd.DataFrame())

    def test_ushs_within_0_100_bounds(self):
        df = _sample_df()
        scored = FundamentalScorer().compute_ushs(df)
        assert scored["ushs"].between(0, 100).all()

    def test_ushs_rank_is_dense_and_starts_at_one(self):
        df = _sample_df()
        scored = FundamentalScorer().compute_ushs(df)
        assert scored["ushs_rank"].min() == 1


class TestEligibleTickers:
    def test_unknown_altman_and_beneish_do_not_exclude(self):
        # C.NS has NaN altman_z/beneish_m but otherwise good fundamentals —
        # "unknown" must not be treated as "disqualified".
        df = _sample_df()
        df.loc["C.NS", ["pe", "pb", "roe", "fcf_yield", "debt_equity", "promoter_pledge_pct"]] = [
            8.0, 1.5, 0.25, 0.10, 40.0, 0.0,
        ]
        scored = FundamentalScorer().compute_ushs(df)
        scored.loc["C.NS", "ushs"] = 90.0  # force it above threshold deterministically
        eligible = FundamentalScorer().get_eligible_tickers(scored)
        assert "C.NS" in eligible

    def test_confirmed_distress_excludes(self):
        df = _sample_df()
        scored = FundamentalScorer().compute_ushs(df)
        scored.loc["B.NS", "ushs"] = 99.0  # even with a top USHS score...
        eligible = FundamentalScorer().get_eligible_tickers(scored)
        assert "B.NS" not in eligible  # ...confirmed Altman distress (z=1.0) must still exclude it

    def test_illiquid_names_are_excluded_when_volume_column_present(self):
        df = _sample_df()
        scored = FundamentalScorer().compute_ushs(df)
        scored.loc["A.NS", "ushs"] = 95.0
        scored["avg_daily_volume_inr"] = [
            settings.MIN_AVG_DAILY_VOLUME_INR * 2,  # A liquid
            settings.MIN_AVG_DAILY_VOLUME_INR * 2,  # B distress anyway
            settings.MIN_AVG_DAILY_VOLUME_INR * 2,  # C
            1_00_000,  # D thin book
        ]
        scored.loc["D.NS", "ushs"] = 95.0
        eligible = FundamentalScorer().get_eligible_tickers(scored)
        assert "A.NS" in eligible
        assert "D.NS" not in eligible

    def test_missing_volume_column_does_not_block_eligibility(self):
        # Older harmonized frames without ADV must still score/filter on USHS alone.
        df = _sample_df()
        scored = FundamentalScorer().compute_ushs(df)
        scored.loc["A.NS", "ushs"] = 95.0
        assert "avg_daily_volume_inr" not in scored.columns
        eligible = FundamentalScorer().get_eligible_tickers(scored)
        assert "A.NS" in eligible
