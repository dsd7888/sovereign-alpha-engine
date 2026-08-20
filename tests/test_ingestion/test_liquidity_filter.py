import pandas as pd

from config import settings
from config.universe import LARGE_CAP_UNIVERSE, NON_LARGECAP_UNIVERSE, PILOT_UNIVERSE
from sovereign_alpha.ingestion.data_harmonizer import DataHarmonizer


class TestUniverseExpansion:
    def test_non_largecap_list_nonempty_and_ns_suffixed(self):
        assert len(NON_LARGECAP_UNIVERSE) >= 50
        assert all(t.endswith(".NS") for t in NON_LARGECAP_UNIVERSE)

    def test_pilot_includes_large_and_non_largecap(self):
        assert set(LARGE_CAP_UNIVERSE).issubset(PILOT_UNIVERSE)
        assert set(NON_LARGECAP_UNIVERSE).issubset(PILOT_UNIVERSE)
        assert len(PILOT_UNIVERSE) == len(set(PILOT_UNIVERSE))


class TestLiquidityFilter:
    def test_drops_below_adv_floor(self):
        df = pd.DataFrame({
            "avg_daily_volume_inr": [
                settings.MIN_AVG_DAILY_VOLUME_INR * 2,
                settings.MIN_AVG_DAILY_VOLUME_INR / 2,
                None,
            ],
        }, index=["LIQUID.NS", "THIN.NS", "UNKNOWN.NS"])
        filtered = DataHarmonizer._apply_liquidity_filter(df)
        # UNKNOWN.NS has no known ADV — unknown liquidity is neutral, not
        # zero, so it must survive the filter alongside LIQUID.NS.
        assert list(filtered.index) == ["LIQUID.NS", "UNKNOWN.NS"]

    def test_passes_through_when_column_absent(self):
        df = pd.DataFrame({"pe": [10.0, 12.0]}, index=["A.NS", "B.NS"])
        filtered = DataHarmonizer._apply_liquidity_filter(df)
        assert list(filtered.index) == ["A.NS", "B.NS"]
