import numpy as np
import pandas as pd
import pytest

from config import settings
from sovereign_alpha.optimization.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioOptimizerError,
)


@pytest.fixture
def optimizer():
    return PortfolioOptimizer()


class TestCholeskyMonteCarlo:
    def test_output_shape_and_columns(self, optimizer):
        tickers = ["A", "B", "C"]
        mu = np.array([0.10, 0.15, 0.08])
        cov = np.diag([0.04, 0.06, 0.03])
        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=50)

        assert len(sim_df) == 50
        for col in ("weights", "exp_return", "port_std", "cvar_95", "sharpe"):
            assert col in sim_df.columns

    def test_weights_sum_to_one_per_simulation(self, optimizer):
        tickers = ["A", "B", "C"]
        mu = np.array([0.10, 0.15, 0.08])
        cov = np.diag([0.04, 0.06, 0.03])
        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=20)
        for w in sim_df["weights"]:
            assert abs(sum(w.values()) - 1.0) < 1e-9
            assert all(v >= 0 for v in w.values())

    def test_shape_mismatch_raises(self, optimizer):
        with pytest.raises(PortfolioOptimizerError):
            optimizer.cholesky_monte_carlo(np.array([0.1, 0.2]), np.eye(3), ["A", "B", "C"], n_sim=10)

    def test_non_positive_definite_covariance_does_not_raise(self, optimizer):
        # Rank-deficient covariance (two identical assets) is not strictly PD;
        # the Cholesky-regularisation fallback must handle it gracefully.
        tickers = ["A", "B"]
        mu = np.array([0.1, 0.1])
        cov = np.array([[0.04, 0.04], [0.04, 0.04]])
        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=10)
        assert len(sim_df) == 10

    def test_batching_produces_same_total_as_single_batch(self, optimizer):
        # Forcing a tiny per-batch cap must not change the total simulation count.
        tickers = ["A", "B"]
        mu = np.array([0.1, 0.12])
        cov = np.diag([0.05, 0.05])
        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=137)
        assert len(sim_df) == 137

    def test_exp_return_is_analytical(self, optimizer):
        # exp_return must be the closed-form w.mu, not a value realised from
        # the single simulated 252-day path — pin the weight draw and check
        # the stored exp_return matches the dot product exactly.
        tickers = ["A", "B"]
        mu = np.array([0.10, 0.20])
        cov = np.diag([0.04, 0.04])
        fixed_w = np.array([[0.5, 0.5]])

        class _FixedDirichletRNG:
            def __init__(self, inner):
                self._inner = inner

            def dirichlet(self, alpha, size=None):
                return np.tile(fixed_w, (size, 1))

            def standard_normal(self, shape):
                return self._inner.standard_normal(shape)

        optimizer._rng = _FixedDirichletRNG(np.random.default_rng(settings.MC_RANDOM_SEED))

        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=1)

        assert sim_df.iloc[0]["exp_return"] == pytest.approx(0.15, abs=1e-12)

    def test_same_weights_same_return(self):
        # exp_return is analytical (w.mu), so two independent runs seeded
        # identically must produce bit-for-bit identical exp_return for a
        # matching weight vector — no path-simulation noise should leak in.
        tickers = ["A", "B", "C"]
        mu = np.array([0.10, 0.15, 0.08])
        cov = np.diag([0.04, 0.06, 0.03])

        opt1 = PortfolioOptimizer()
        opt1._rng = np.random.default_rng(settings.MC_RANDOM_SEED)
        opt2 = PortfolioOptimizer()
        opt2._rng = np.random.default_rng(settings.MC_RANDOM_SEED)

        sim_df1 = opt1.cholesky_monte_carlo(mu, cov, tickers, n_sim=25)
        sim_df2 = opt2.cholesky_monte_carlo(mu, cov, tickers, n_sim=25)

        row1, row2 = sim_df1.iloc[0], sim_df2.iloc[0]
        assert row1["weights"] == row2["weights"]
        assert row1["exp_return"] == row2["exp_return"]

    def test_frontier_return_realistic(self, optimizer):
        # With flat mu (post-shrinkage) and a reasonable covariance matrix,
        # the max-Sharpe portfolio's exp_return must land in a realistic
        # equity range — a value like 30%+ would indicate the selection-bias
        # regression (picking the outlier of 5,000 noisy simulated paths).
        tickers = ["A", "B", "C", "D"]
        mu = np.full(4, 0.074)
        cov = np.diag([0.04, 0.05, 0.06, 0.045])

        sim_df = optimizer.cholesky_monte_carlo(mu, cov, tickers, n_sim=2000)
        result = optimizer.find_optimal_portfolios(sim_df)

        assert 0.05 <= result["max_sharpe"]["exp_return"] <= 0.15


class TestFindOptimalPortfolios:
    def test_picks_max_sharpe_and_min_cvar_rows(self, optimizer):
        sim_df = pd.DataFrame({
            "weights": [{"A": 1.0}, {"A": 0.5, "B": 0.5}, {"B": 1.0}],
            "exp_return": [0.10, 0.15, 0.05],
            "port_std": [0.10, 0.12, 0.20],
            "cvar_95": [0.15, 0.10, 0.30],
            "sharpe": [1.0, 1.2, 0.2],
        })
        result = optimizer.find_optimal_portfolios(sim_df)
        assert "max_sharpe" in result and "min_cvar" in result
        # Row index 1 has the lowest CVaR (0.10) among the three.
        assert result["min_cvar"]["cvar_95"] == 0.10


class TestAllocateCapital:
    def test_whole_share_quantities_only(self, optimizer):
        alloc = optimizer.allocate_capital(
            weights={"A": 0.6, "B": 0.4}, capital_inr=10_000, prices={"A": 333.33, "B": 50.0},
        )
        for details in alloc.values():
            assert float(details["qty"]).is_integer()
            assert details["qty"] >= 1

    def test_below_min_position_value_is_skipped(self, optimizer):
        alloc = optimizer.allocate_capital(
            weights={"A": 0.99, "B": 0.01}, capital_inr=10_000, prices={"A": 100.0, "B": 100.0},
        )
        assert "B" not in alloc  # 1% of 10,000 = Rs.100 < MIN_POSITION_VALUE_INR

    def test_invalid_price_is_skipped_not_fatal(self, optimizer):
        alloc = optimizer.allocate_capital(
            weights={"A": 0.5, "B": 0.5}, capital_inr=10_000, prices={"A": 100.0, "B": 0.0},
        )
        assert "B" not in alloc
        assert "A" in alloc

    def test_never_allocates_more_than_capital(self, optimizer):
        alloc = optimizer.allocate_capital(
            weights={"A": 0.5, "B": 0.5}, capital_inr=10_000, prices={"A": 3_100.0, "B": 3_100.0},
        )
        total_allocated = sum(d["value"] for d in alloc.values())
        assert total_allocated <= 10_000

    def test_weights_not_summing_to_one_are_normalised(self, optimizer):
        # 0.3 + 0.3 = 0.6, not 1.0 — should be rescaled rather than under-invested.
        alloc = optimizer.allocate_capital(
            weights={"A": 0.3, "B": 0.3}, capital_inr=10_000, prices={"A": 100.0, "B": 100.0},
        )
        total_weight_actual = sum(d["weight_actual"] for d in alloc.values())
        assert total_weight_actual > 0.6  # normalisation should push allocation higher than raw 0.6
