import pytest

from sovereign_alpha.execution.paper_broker import PaperBroker
from sovereign_alpha.risk.risk_guardian import Action


@pytest.fixture
def broker(tmp_path):
    return PaperBroker(state_path=tmp_path / "portfolio_state.json")


class TestInitialisation:
    def test_starts_with_configured_capital(self, broker):
        from config import settings
        assert broker.state["cash"] == settings.INITIAL_CAPITAL_INR
        assert broker.state["holdings"] == {}

    def test_state_persists_across_instances(self, tmp_path):
        path = tmp_path / "state.json"
        b1 = PaperBroker(state_path=path)
        b1.buy("A.NS", qty=1, price=100.0)
        b2 = PaperBroker(state_path=path)
        assert "A.NS" in b2.state["holdings"]


class TestBacktestMode:
    def test_persist_false_never_touches_disk(self, tmp_path):
        path = tmp_path / "should_not_be_created.json"
        b = PaperBroker(state_path=path, persist=False)
        b.buy("A.NS", qty=1, price=100.0)
        assert not path.exists()

    def test_persist_false_starts_fresh_regardless_of_existing_file(self, tmp_path):
        path = tmp_path / "state.json"
        seeded = PaperBroker(state_path=path)
        seeded.buy("A.NS", qty=1, price=100.0)  # writes real state to disk

        b = PaperBroker(state_path=path, persist=False)
        assert b.state["holdings"] == {}  # ignores what's on disk

    def test_custom_initial_capital(self):
        b = PaperBroker(persist=False, initial_capital=50_000.0)
        assert b.state["cash"] == 50_000.0
        assert b.state["initial_capital"] == 50_000.0

    def test_clock_controls_trade_and_nav_dates(self):
        # clock is read multiple times per call (trade record + _save_state),
        # so it must be a stable "what day is it" read, not a one-shot value —
        # a mutable holder the test advances between logical days, exactly
        # like a backtest loop would.
        day = {"value": "20200101"}
        b = PaperBroker(persist=False, clock=lambda: day["value"])

        trade = b.buy("A.NS", qty=1, price=100.0)
        assert trade["date"] == "20200101"

        day["value"] = "20200102"
        b.update_prices({"A.NS": 105.0})
        assert "20200102" in b.state["daily_nav"]

        day["value"] = "20200103"
        trade2 = b.sell("A.NS", qty=1, price=105.0)
        assert trade2["date"] == "20200103"

    def test_moving_clock_across_days_keeps_every_nav_point(self):
        """
        The bug this clock exists to prevent: with the real wall clock,
        every simulated day's NAV would be stamped with today's actual
        date and overwrite the same daily_nav key, collapsing a multi-day
        backtest into a single point.
        """
        day = {"value": "20200101"}
        b = PaperBroker(persist=False, clock=lambda: day["value"])
        for d, price in [("20200101", 100.0), ("20200102", 101.0), ("20200103", 99.0)]:
            day["value"] = d
            b.update_prices({})
        assert set(b.state["daily_nav"].keys()) == {"20200101", "20200102", "20200103"}


class TestBuy:
    def test_buy_deducts_cash_including_costs(self, broker):
        cash_before = broker.state["cash"]
        trade = broker.buy("A.NS", qty=10, price=100.0)
        assert trade is not None
        assert broker.state["cash"] < cash_before - 1_000  # trade value + costs both deducted

    def test_buy_rejects_insufficient_cash(self, broker):
        trade = broker.buy("A.NS", qty=1_000_000, price=100.0)
        assert trade is None
        assert "A.NS" not in broker.state["holdings"]

    def test_buy_rejects_non_positive_qty_or_price(self, broker):
        assert broker.buy("A.NS", qty=0, price=100.0) is None
        assert broker.buy("A.NS", qty=10, price=0.0) is None
        assert broker.buy("A.NS", qty=-5, price=100.0) is None

    def test_repeated_buy_computes_weighted_average_cost(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        broker.buy("A.NS", qty=10, price=200.0)
        holding = broker.state["holdings"]["A.NS"]
        assert holding["qty"] == 20
        assert abs(holding["avg_cost"] - 150.0) < 1e-6


class TestSell:
    def test_sell_more_than_held_rejected(self, broker):
        broker.buy("A.NS", qty=5, price=100.0)
        assert broker.sell("A.NS", qty=10, price=110.0) is None

    def test_sell_all_liquidates_position(self, broker):
        broker.buy("A.NS", qty=5, price=100.0)
        broker.sell_all("A.NS", price=110.0)
        assert broker.state["holdings"]["A.NS"]["qty"] == 0

    def test_sell_nonexistent_position_returns_none(self, broker):
        assert broker.sell("Z.NS", qty=1, price=100.0) is None


class TestUpdatePrices:
    def test_missing_fresh_price_falls_back_to_last_known(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        v1 = broker.update_prices({"A.NS": 120.0})
        # No fresh price this time — must NOT drop the position's value to zero.
        v2 = broker.update_prices({})
        assert v2 >= v1 - 1.0  # cash unchanged, holdings value should carry over via stale price

    def test_peak_value_only_increases(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        broker.update_prices({"A.NS": 200.0})
        peak_after_gain = broker.state["peak_value"]
        broker.update_prices({"A.NS": 50.0})
        assert broker.state["peak_value"] == peak_after_gain


class TestRebalance:
    def test_halt_executes_no_trades(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        trades = broker.rebalance({"A.NS": 1.0}, {"A.NS": 100.0}, circuit_action=Action.HALT)
        assert trades == []

    def test_full_exit_liquidates_everything(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        broker.buy("B.NS", qty=5, price=200.0)
        broker.rebalance({}, {"A.NS": 100.0, "B.NS": 200.0}, circuit_action=Action.FULL_EXIT)
        assert broker.state["holdings"]["A.NS"]["qty"] == 0
        assert broker.state["holdings"]["B.NS"]["qty"] == 0

    def test_reduce_50_halves_positions(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        broker.rebalance({"A.NS": 1.0}, {"A.NS": 100.0}, circuit_action=Action.REDUCE_50)
        assert broker.state["holdings"]["A.NS"]["qty"] == 5

    def test_normal_rebalance_buys_toward_target(self, broker):
        broker.rebalance({"A.NS": 0.5, "B.NS": 0.5}, {"A.NS": 100.0, "B.NS": 100.0})
        assert broker.state["holdings"].get("A.NS", {}).get("qty", 0) > 0
        assert broker.state["holdings"].get("B.NS", {}).get("qty", 0) > 0


class TestSummary:
    def test_summary_on_fresh_portfolio(self, broker):
        summary = broker.get_summary()
        assert summary["total_pnl"] == 0.0
        assert summary["total_trades"] == 0

    def test_summary_reflects_costs_paid(self, broker):
        broker.buy("A.NS", qty=10, price=100.0)
        broker.update_prices({"A.NS": 100.0})
        summary = broker.get_summary()
        assert summary["total_costs_paid"] > 0
        assert summary["total_trades"] == 1
