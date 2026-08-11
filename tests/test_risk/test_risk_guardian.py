from sovereign_alpha.risk.risk_guardian import Action, RiskGuardian


class TestCircuitBreaker:
    def setup_method(self):
        self.rg = RiskGuardian()

    def test_normal_market_holds(self):
        assert self.rg.check_circuit_breaker(99_000, 100_000, vix=17.0) == Action.HOLD

    def test_vix_crisis_overrides_everything_to_halt(self):
        # Even with zero drawdown, a VIX crisis reading must HALT.
        assert self.rg.check_circuit_breaker(100_000, 100_000, vix=36.0) == Action.HALT

    def test_vix_high_alert_below_crisis_hedges(self):
        assert self.rg.check_circuit_breaker(97_000, 100_000, vix=27.0) == Action.HEDGE_PUT

    def test_full_exit_drawdown_without_elevated_vix(self):
        assert self.rg.check_circuit_breaker(84_000, 100_000, vix=15.0) == Action.FULL_EXIT

    def test_reduce_50_on_moderate_drawdown(self):
        assert self.rg.check_circuit_breaker(90_000, 100_000, vix=15.0) == Action.REDUCE_50

    def test_reduce_50_on_elevated_vix_alone(self):
        assert self.rg.check_circuit_breaker(99_500, 100_000, vix=22.0) == Action.REDUCE_50

    def test_vix_priority_over_drawdown(self):
        # 20% drawdown (would be FULL_EXIT) but VIX crisis must win.
        assert self.rg.check_circuit_breaker(80_000, 100_000, vix=40.0) == Action.HALT

    def test_zero_peak_value_does_not_divide_by_zero(self):
        # Degenerate state (fresh portfolio, peak not yet set) must not raise.
        action = self.rg.check_circuit_breaker(0, 0, vix=15.0)
        assert action in Action


class TestSinglePosition:
    def setup_method(self):
        self.rg = RiskGuardian()

    def test_stop_loss_triggers_at_threshold(self):
        # 12% loss threshold exactly for large caps.
        assert self.rg.check_single_position("RELIANCE.NS", entry_price=100.0, current_price=88.0) == Action.FULL_EXIT

    def test_holds_below_threshold(self):
        assert self.rg.check_single_position("RELIANCE.NS", entry_price=100.0, current_price=90.0) == Action.HOLD

    def test_zero_entry_price_holds_not_crashes(self):
        assert self.rg.check_single_position("RELIANCE.NS", entry_price=0.0, current_price=50.0) == Action.HOLD

    def test_gain_holds(self):
        assert self.rg.check_single_position("RELIANCE.NS", entry_price=100.0, current_price=120.0) == Action.HOLD

    def test_midcap_allows_deeper_drawdown_before_stop(self):
        # 13% loss: large-cap stop (12%) would exit; midcap stop (15%) holds.
        assert self.rg.check_single_position("PERSISTENT.NS", entry_price=100.0, current_price=87.0) == Action.HOLD

    def test_midcap_stop_triggers_at_15_pct(self):
        assert self.rg.check_single_position("PERSISTENT.NS", entry_price=100.0, current_price=85.0) == Action.FULL_EXIT


class TestPromoterPledge:
    def setup_method(self):
        self.rg = RiskGuardian()

    def test_surge_past_50_triggers_exit(self):
        assert self.rg.check_promoter_pledge("X", new_pct=55, prev_pct=40) == Action.FULL_EXIT

    def test_high_but_stable_pledge_holds(self):
        # Above 50% but not a recent surge — spec's rule requires both conditions.
        assert self.rg.check_promoter_pledge("X", new_pct=55, prev_pct=53) == Action.HOLD

    def test_below_50_holds_even_with_surge(self):
        assert self.rg.check_promoter_pledge("X", new_pct=45, prev_pct=20) == Action.HOLD


class TestKellyPositionSizing:
    def setup_method(self):
        self.rg = RiskGuardian()

    def test_position_fraction_capped_at_25_pct(self):
        # Absurdly high mu/low sigma would blow past 25% uncapped.
        result = self.rg.compute_position_size("X", total_capital=100_000, mu=0.9, sigma=0.05)
        assert result["position_fraction"] <= 0.25 + 1e-9

    def test_negative_edge_gives_zero_position(self):
        result = self.rg.compute_position_size("X", total_capital=100_000, mu=0.0, sigma=0.2)
        assert result["position_fraction"] == 0.0
        assert result["position_inr"] == 0.0

    def test_high_impact_cost_shrinks_position(self):
        # Thin liquidity relative to position size should trigger the impact-cost trim.
        result = self.rg.compute_position_size(
            "X", total_capital=10_000_000, mu=0.5, sigma=0.1, avg_daily_volume_inr=10_000
        )
        assert result["impact_cost_pct"] <= 0.5 + 1e-6 or result["position_fraction"] < 0.25
