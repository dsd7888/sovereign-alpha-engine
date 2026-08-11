from sovereign_alpha.optimization.transaction_costs import GrowwCostModel


class TestBuyCost:
    def test_zero_or_negative_trade_value_is_free(self):
        assert GrowwCostModel.compute_buy_cost(0).total == 0
        assert GrowwCostModel.compute_buy_cost(-100).total == 0

    def test_brokerage_floor_applies_on_tiny_trade(self):
        # 0.1% of 1000 = Rs.1, below the Rs.5 floor.
        costs = GrowwCostModel.compute_buy_cost(1_000)
        assert costs.brokerage == 5.00

    def test_brokerage_cap_applies_on_large_trade(self):
        # 0.1% of 50,000 = Rs.50, above the Rs.20 cap.
        costs = GrowwCostModel.compute_buy_cost(50_000)
        assert costs.brokerage == 20.00

    def test_stamp_duty_charged_on_buy_only(self):
        costs = GrowwCostModel.compute_buy_cost(100_000)
        assert abs(costs.stamp_duty - 0.00015 * 100_000) < 1e-9
        assert costs.dp_charge == 0.0

    def test_gst_excludes_stt_and_stamp_duty(self):
        tv = 100_000
        costs = GrowwCostModel.compute_buy_cost(tv)
        expected_gst = 0.18 * (costs.brokerage + costs.exchange_tc + costs.sebi_charge)
        assert abs(costs.gst - expected_gst) < 1e-9

    def test_total_pct_matches_total_over_value(self):
        costs = GrowwCostModel.compute_buy_cost(100_000)
        assert abs(costs.total_pct - costs.total / 100_000) < 1e-12


class TestSellCost:
    def test_zero_or_negative_trade_value_is_free(self):
        assert GrowwCostModel.compute_sell_cost(0).total == 0
        assert GrowwCostModel.compute_sell_cost(-1).total == 0

    def test_dp_charge_flat_20_per_scrip(self):
        costs = GrowwCostModel.compute_sell_cost(100_000)
        assert costs.dp_charge == 20.00

    def test_no_stamp_duty_on_sell(self):
        costs = GrowwCostModel.compute_sell_cost(100_000)
        assert costs.stamp_duty == 0.0

    def test_gst_includes_dp_charge(self):
        tv = 100_000
        costs = GrowwCostModel.compute_sell_cost(tv)
        expected_gst = 0.18 * (costs.brokerage + costs.exchange_tc + costs.sebi_charge + costs.dp_charge)
        assert abs(costs.gst - expected_gst) < 1e-9


class TestRoundTrip:
    def test_round_trip_on_1_lakh_is_roughly_0_3_pct(self):
        # The spec's own worked example: ~0.295% round trip on a Rs.1,00,000 trade,
        # before impact cost. Assert it lands in a sane neighbourhood, not an exact
        # figure, since impact cost is volume/volatility-dependent.
        rt = GrowwCostModel.round_trip_cost_fraction(100_000, avg_daily_volume_inr=1e10, daily_volatility=0.0)
        assert 0.002 < rt < 0.004

    def test_impact_cost_zero_order_value(self):
        assert GrowwCostModel.compute_impact_cost(0, 1e8, 0.02) == 0.0

    def test_impact_cost_no_volume_data_uses_conservative_default(self):
        assert GrowwCostModel.compute_impact_cost(50_000, 0, 0.02) == 0.005

    def test_impact_cost_capped(self):
        # A huge order relative to a tiny daily volume should hit the cap, not blow up.
        ic = GrowwCostModel.compute_impact_cost(order_value=10_000_000, avg_daily_volume_inr=1_000, daily_volatility=0.05)
        assert ic <= 0.02 + 1e-12

    def test_penalise_return_reduces_gross_return(self):
        net = GrowwCostModel.penalise_return(expected_annual_return=0.15, annual_turnover=1.0, avg_trade_value=50_000)
        assert net < 0.15
