import numpy as np
import pandas as pd
import pytest

from sovereign_alpha.backtesting import metrics


def _nav(values: list[float], start="2023-01-01", freq="D") -> pd.Series:
    dates = pd.date_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=dates)


class TestCagr:
    def test_doubling_over_one_year(self):
        nav = _nav([100.0, 200.0], start="2023-01-01", freq="365D")
        assert metrics.cagr(nav) == pytest.approx(1.0, abs=0.01)

    def test_flat_nav_is_zero(self):
        nav = _nav([100.0] * 10)
        assert metrics.cagr(nav) == pytest.approx(0.0, abs=1e-6)

    def test_single_point_is_nan(self):
        nav = _nav([100.0])
        assert np.isnan(metrics.cagr(nav))


class TestMaxDrawdown:
    def test_identifies_peak_and_trough(self):
        # 100 -> 150 (peak) -> 90 (trough, -40% from peak) -> 120
        nav = _nav([100.0, 150.0, 120.0, 90.0, 120.0])
        result = metrics.max_drawdown(nav)
        assert result["max_drawdown_pct"] == pytest.approx(40.0, abs=0.01)

    def test_monotonic_increase_has_zero_drawdown(self):
        nav = _nav([100.0, 110.0, 120.0, 130.0])
        result = metrics.max_drawdown(nav)
        assert result["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_series_returns_nan_not_crash(self):
        nav = pd.Series(dtype=float)
        result = metrics.max_drawdown(nav)
        assert np.isnan(result["max_drawdown_pct"])


class TestSharpeSortino:
    def test_zero_volatility_returns_nan(self):
        returns = pd.Series([0.001] * 20)  # constant — std is 0
        assert np.isnan(metrics.sharpe_ratio(returns, rf=0.0))

    def test_positive_drift_gives_positive_sharpe(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, 500))
        assert metrics.sharpe_ratio(returns, rf=0.0) > 0

    def test_sortino_penalises_downside_jumps_more_than_upside_jumps(self):
        # Same symmetric base noise (so both series have genuine, non-empty
        # downside for the ratio to be defined on), then periodic big jumps
        # added to one side only. The downside-jump series must score
        # noticeably worse — Sortino should barely react to the upside jumps.
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 0.01, 300)
        upside_jumps = base.copy()
        upside_jumps[::10] += 0.05
        downside_jumps = base.copy()
        downside_jumps[::10] -= 0.05

        up_sortino = metrics.sortino_ratio(pd.Series(upside_jumps), rf=0.0)
        down_sortino = metrics.sortino_ratio(pd.Series(downside_jumps), rf=0.0)
        assert up_sortino > down_sortino


class TestCalmarRatio:
    def test_basic_ratio(self):
        assert metrics.calmar_ratio(0.20, 10.0) == pytest.approx(2.0)

    def test_zero_drawdown_returns_nan(self):
        assert np.isnan(metrics.calmar_ratio(0.20, 0.0))


class TestTurnoverAndWinRate:
    def test_annualised_turnover(self):
        trade_log = [{"value": 50_000.0}, {"value": 50_000.0}]
        # 100,000 traded / 100,000 avg NAV / 1 year = 1.0x turnover
        assert metrics.annualised_turnover(trade_log, avg_nav=100_000.0, years=1.0) == pytest.approx(1.0)

    def test_win_rate_counts_only_sells_with_realised_pnl(self):
        trade_log = [
            {"action": "BUY", "value": 100.0},
            {"action": "SELL", "realised_pnl": 10.0},
            {"action": "SELL", "realised_pnl": -5.0},
            {"action": "SELL", "realised_pnl": 3.0},
        ]
        assert metrics.win_rate(trade_log) == pytest.approx(2 / 3)

    def test_win_rate_no_sells_is_nan(self):
        assert np.isnan(metrics.win_rate([{"action": "BUY", "value": 100.0}]))


class TestBenchmarkComparison:
    def test_beta_one_for_identical_series(self):
        rng = np.random.default_rng(1)
        bench = pd.Series(rng.normal(0.0005, 0.01, 300))
        result = metrics.benchmark_comparison(bench, bench)
        assert result["beta"] == pytest.approx(1.0, abs=1e-6)
        assert result["alpha_annualised"] == pytest.approx(0.0, abs=1e-6)

    def test_too_few_overlapping_points_returns_nan(self):
        s = pd.Series([0.01, 0.02, 0.01])
        result = metrics.benchmark_comparison(s, s)
        assert np.isnan(result["beta"])


class TestSummarize:
    def test_produces_all_expected_keys(self):
        nav = _nav([100_000.0, 101_000.0, 99_000.0, 103_000.0])
        trade_log = [{"action": "BUY", "value": 5000.0, "total_cost": 10.0}]
        result = metrics.summarize(nav, trade_log, initial_capital=100_000.0)
        for key in ("cagr_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio", "win_rate_pct"):
            assert key in result

    def test_total_return_matches_first_last_nav(self):
        nav = _nav([100_000.0, 150_000.0])
        result = metrics.summarize(nav, [], initial_capital=100_000.0)
        assert result["total_return_pct"] == pytest.approx(50.0)
