"""Deterministic invariants for the DCC-GARCH estimator and experiments.

These check the mathematical properties the paper relies on (correlation-matrix
validity, causal timing, (a,b) recovery, tracking dominance over static,
crisis risk misstatement, parameter-count scaling) rather than pinning
machine-specific magnitudes. Run: python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import dcc          # noqa: E402
import run_all      # noqa: E402


# --------------------------------------------------------------------------- #
# estimator internals
# --------------------------------------------------------------------------- #
def test_corr_target_is_valid_correlation():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 3))
    Q = dcc.corr_target(Z)
    assert np.allclose(np.diag(Q), 1.0)
    assert np.allclose(Q, Q.T)
    assert np.min(np.linalg.eigvalsh(Q)) > 0.0


def test_dcc_filter_produces_valid_correlations():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((400, 3))
    Rp = dcc.dcc_filter(Z, 0.05, 0.90)
    for t in (0, 50, 399):
        R = Rp[t]
        assert np.allclose(np.diag(R), 1.0)
        assert np.allclose(R, R.T)
        assert np.min(np.linalg.eigvalsh(R)) > -1e-10
        assert np.all(np.abs(R[np.triu_indices(3, 1)]) <= 1.0 + 1e-12)


def test_filter_is_causal_first_step_is_target():
    """R_path[0] must equal the (normalized) target Qbar -- it cannot depend on
    z_0 (that would be look-ahead)."""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((300, 2))
    Qbar = dcc.corr_target(Z)
    Rp = dcc.dcc_filter(Z, 0.05, 0.90, Qbar)
    assert np.allclose(Rp[0], Qbar)


def test_negloglik_penalizes_infeasible():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((200, 2))
    Qbar = dcc.corr_target(Z)
    assert dcc.dcc_negloglik([0.6, 0.6], Z, Qbar) >= 1e11   # a+b>=1
    assert dcc.dcc_negloglik([-0.1, 0.9], Z, Qbar) >= 1e11  # a<=0
    assert dcc.dcc_negloglik([0.05, 0.9], Z, Qbar) < 1e11   # feasible


# --------------------------------------------------------------------------- #
# the headline invariant: recovery of a known (a,b)
# --------------------------------------------------------------------------- #
def test_dcc_recovers_known_ab():
    rng = np.random.default_rng(11)
    Qbar = np.array([[1.0, 0.5], [0.5, 1.0]])
    Z = dcc.simulate_dcc_z(4000, Qbar, 0.05, 0.90, rng)
    fit = dcc.fit_dcc(Z)
    assert abs(fit["a"] - 0.05) < 0.03
    assert abs(fit["b"] - 0.90) < 0.06
    assert 0.0 < fit["a"] and 0.0 < fit["b"] and fit["sum"] < 1.0


def test_dcc_recovers_known_ab_three_asset():
    rng = np.random.default_rng(12)
    Qbar = np.array([[1, 0.4, 0.3], [0.4, 1, 0.5], [0.3, 0.5, 1]], float)
    Z = dcc.simulate_dcc_z(4000, Qbar, 0.04, 0.92, rng)
    fit = dcc.fit_dcc(Z)
    assert abs(fit["a"] - 0.04) < 0.03
    assert abs(fit["b"] - 0.92) < 0.06


# --------------------------------------------------------------------------- #
# baselines and metrics
# --------------------------------------------------------------------------- #
def test_rolling_corr_is_causal():
    """rolling_corr_path[t] must use only rows < t (no look-ahead)."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 2))
    Rp = dcc.rolling_corr_path(X, 30)
    # planting a huge outlier at time t must not change the matrix AT t
    Y = X.copy()
    Y[100] *= 100.0
    Rp2 = dcc.rolling_corr_path(Y, 30)
    assert np.allclose(Rp[100], Rp2[100])
    assert not np.allclose(Rp[101], Rp2[101])   # but it affects t+1


def test_mae_rmse_ordering():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.1, 0.2, 0.3])
    assert dcc.rmse(a, b) >= dcc.mae(a, b) > 0.0


def test_var_breaches_monotone_in_variance():
    """A smaller covariance forecast (tighter VaR) can only produce >= breaches."""
    rng = np.random.default_rng(5)
    R = rng.standard_normal((500, 2)) * 0.02
    w = np.array([0.5, 0.5])
    big = np.tile(np.eye(2) * 1e-2, (500, 1, 1))
    small = np.tile(np.eye(2) * 1e-4, (500, 1, 1))
    b_big, _, _ = dcc.portfolio_var_breaches(R, big, w, 1.645)
    b_small, _, _ = dcc.portfolio_var_breaches(R, small, w, 1.645)
    assert b_small.sum() >= b_big.sum()


# --------------------------------------------------------------------------- #
# experiment-level invariants (small, fast configs)
# --------------------------------------------------------------------------- #
def test_tracking_dcc_beats_static_in_crisis():
    res = run_all.corr_tracking_case(T=800, d=2, seed=101)
    assert res["dcc"]["mae_crisis"] < res["static"]["mae_crisis"]
    # and DCC beats the best rolling window in the crisis window
    assert res["dcc"]["mae_crisis"] <= res["best_rolling_crisis_mae"] + 1e-9
    # a valid, mean-reverting fit
    assert 0.0 < res["dcc"]["a"] and 0.0 < res["dcc"]["b"]
    assert res["dcc"]["sum"] < 1.0 + 1e-6


def test_static_understates_crisis_risk():
    res = run_all.risk_misstatement_case(T=800, d=2, seed=101, roll_window=60)
    nominal = res["nominal"]
    # static breaches far above nominal in the crisis...
    assert res["static"]["breach_crisis"] > 1.8 * nominal
    # ...while DCC stays much closer to nominal than static does
    assert (abs(res["dcc"]["breach_crisis"] - nominal)
            < abs(res["static"]["breach_crisis"] - nominal))


def test_hedge_reduces_spread_variance():
    res = run_all.hedge_ratio_case(T=800, seed=101)
    assert res["var_reduction_crisis"] > 0.0
    assert res["beta_track_mae_dcc_crisis"] < res["beta_track_mae_static_crisis"]


def test_dimensionality_scaling():
    # VECH grows faster than BEKK grows faster than DCC
    assert run_all.dcc_params(10) < run_all.bekk_params(10) < run_all.vech_params(10)
    # exact small-d formulas quoted in the paper
    assert run_all.vech_params(2) == 21
    assert run_all.bekk_params(2) == 11
    assert run_all.dcc_params(2) == 8
    assert run_all.dcc_params(50) == 152 and run_all.bekk_params(50) == 6275
    # DCC correlation layer is O(1): the growth is entirely the univariate step
    assert run_all.dcc_params(20) - run_all.dcc_params(10) == 3 * (20 - 10)


def test_determinism():
    a = run_all.corr_tracking_case(T=400, d=2, seed=7)
    b = run_all.corr_tracking_case(T=400, d=2, seed=7)
    assert a["dcc"]["a"] == b["dcc"]["a"]
    assert a["dcc"]["mae_crisis"] == b["dcc"]["mae_crisis"]
