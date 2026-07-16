"""Self-contained DCC-GARCH estimator, baselines, and risk metrics.

The `arch` library (Kevin Sheppard) fits *univariate* GARCH only; the Dynamic
Conditional Correlation (DCC) layer is implemented here from scratch. This is
the two-step estimator of Engle (2002):

  Step 1 (per asset, delegated to `arch`): fit a univariate GARCH(1,1),
          extract the conditional volatility sigma_{i,t} and the standardized
          residual z_{i,t} = eps_{i,t}/sigma_{i,t}.
  Step 2 (here): with Qbar the sample correlation of the standardized
          residuals (correlation targeting), run the Q-recursion
              Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},
          normalize Q_t -> R_t (unit diagonal), and maximize the Gaussian
          DCC quasi-log-likelihood
              L(a,b) = -1/2 sum_t ( log|R_t| + z_t' R_t^{-1} z_t )
          over the two scalars (a,b) with a,b>0, a+b<1.

Timing is strictly causal: R_t is scored against z_t and *then* Q is advanced
with z_t z_t' for step t+1, so R_t uses information through t-1 only.

Everything is deterministic given a seed. Estimators live here; the synthetic
data-generating processes live in run_all.py.
"""
import warnings

import numpy as np
from scipy.optimize import minimize


# --------------------------------------------------------------------------- #
# univariate step (arch wrapper)
# --------------------------------------------------------------------------- #
def garch_std_resid(returns, scale=1.0):
    """Fit a univariate GARCH(1,1) with constant mean via `arch`.

    Returns (z, sigma) where z is the standardized residual and sigma the
    conditional volatility, both in the *original* return units (the internal
    x100 scaling `arch` prefers is undone before returning). Deterministic:
    `arch`'s analytic MLE has no random component.
    """
    from arch import arch_model
    r = np.asarray(returns, dtype=float) * scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1,
                         dist="normal").fit(disp="off")
    z = np.asarray(res.std_resid, dtype=float)
    sigma = np.asarray(res.conditional_volatility, dtype=float) / scale
    return z, sigma


def garch_panel(returns, scale=100.0):
    """Fit univariate GARCH to every column of a (T,d) return panel.

    Returns (Z, Sigma): (T,d) standardized residuals and (T,d) conditional
    volatilities (in original units).
    """
    R = np.asarray(returns, dtype=float)
    T, d = R.shape
    Z = np.empty((T, d))
    Sigma = np.empty((T, d))
    for i in range(d):
        z, s = garch_std_resid(R[:, i], scale=scale)
        Z[:, i] = z
        Sigma[:, i] = s
    return Z, Sigma


# --------------------------------------------------------------------------- #
# correlation targeting + the Q-recursion
# --------------------------------------------------------------------------- #
def corr_target(Z):
    """Unconditional correlation matrix Qbar of the standardized residuals."""
    Z = np.asarray(Z, dtype=float)
    C = np.cov(Z, rowvar=False, bias=True)
    dinv = np.diag(1.0 / np.sqrt(np.diag(C)))
    return dinv @ C @ dinv


def _q_to_r(Q):
    q = np.sqrt(np.diag(Q))
    return Q / np.outer(q, q)


def dcc_filter(Z, a, b, Qbar=None):
    """Run the DCC recursion with fixed (a,b); return the R_t path.

    Causal timing: R_path[t] depends on z_0..z_{t-1} only. Q is initialized at
    Qbar (so R_path[0] == Qbar as a correlation matrix).
    Returns R_path with shape (T, d, d).
    """
    Z = np.asarray(Z, dtype=float)
    T, d = Z.shape
    if Qbar is None:
        Qbar = corr_target(Z)
    Q = Qbar.copy()
    R_path = np.empty((T, d, d))
    for t in range(T):
        R_path[t] = _q_to_r(Q)
        z = Z[t]
        Q = (1.0 - a - b) * Qbar + a * np.outer(z, z) + b * Q
    return R_path


def dcc_negloglik(params, Z, Qbar):
    """Negative Gaussian DCC quasi-log-likelihood in (a,b). Infeasible points
    (a,b<=0 or a+b>=1) return a large finite penalty so derivative-free
    optimizers stay in the stationarity region."""
    a, b = float(params[0]), float(params[1])
    if a <= 0.0 or b <= 0.0 or a + b >= 1.0:
        return 1e12
    Z = np.asarray(Z, dtype=float)
    T = Z.shape[0]
    Q = Qbar.copy()
    ll = 0.0
    for t in range(T):
        R = _q_to_r(Q)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0.0:
            return 1e12
        z = Z[t]
        ll += -0.5 * (logdet + z @ np.linalg.solve(R, z))
        Q = (1.0 - a - b) * Qbar + a * np.outer(z, z) + b * Q
    return -ll


def fit_dcc(Z, starts=((0.03, 0.94), (0.05, 0.90), (0.02, 0.97), (0.10, 0.80))):
    """Estimate (a,b) by maximizing the DCC quasi-log-likelihood.

    Multi-start Nelder-Mead (derivative-free; robust to the a+b<1 feasibility
    cliff). Deterministic. Returns dict(a, b, sum=a+b, loglik, success).
    """
    Z = np.asarray(Z, dtype=float)
    Qbar = corr_target(Z)
    best = None
    for x0 in starts:
        res = minimize(dcc_negloglik, np.array(x0, dtype=float),
                       args=(Z, Qbar), method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    a, b = float(best.x[0]), float(best.x[1])
    return dict(a=a, b=b, sum=a + b, loglik=float(-best.fun),
                success=bool(best.success))


# --------------------------------------------------------------------------- #
# simulate standardized residuals from a TRUE DCC recursion (for (a,b) recovery)
# --------------------------------------------------------------------------- #
def simulate_dcc_z(T, Qbar, a, b, rng, burn=500):
    """Draw z_t ~ N(0, R_t) where R_t follows the DCC recursion with the given
    true (a,b) and target Qbar. Returns Z of shape (T, d)."""
    Qbar = np.asarray(Qbar, dtype=float)
    d = Qbar.shape[0]
    Q = Qbar.copy()
    z = rng.standard_normal(d)
    out = np.empty((T, d))
    for t in range(-burn, T):
        Q = (1.0 - a - b) * Qbar + a * np.outer(z, z) + b * Q
        R = _q_to_r(Q)
        L = np.linalg.cholesky(R)
        z = L @ rng.standard_normal(d)
        if t >= 0:
            out[t] = z
    return out


# --------------------------------------------------------------------------- #
# baseline correlation / covariance estimators
# --------------------------------------------------------------------------- #
def rolling_corr_path(returns, window):
    """Causal rolling Pearson correlation matrix path (T,d,d). For t<window the
    matrix is the identity-diagonal correlation of the data seen so far
    (min 2 points); entry t uses returns[max(0,t-window):t] (info < t)."""
    R = np.asarray(returns, dtype=float)
    T, d = R.shape
    out = np.empty((T, d, d))
    for t in range(T):
        lo = max(0, t - window)
        win = R[lo:t]
        if win.shape[0] < 2:
            out[t] = np.eye(d)
            continue
        C = np.corrcoef(win, rowvar=False)
        out[t] = np.nan_to_num(C, nan=0.0) + np.eye(d) * 0.0
        np.fill_diagonal(out[t], 1.0)
    return out


def rolling_cov_path(returns, window):
    """Causal rolling covariance matrix path (T,d,d); entry t uses
    returns[max(0,t-window):t]. For t<2 falls back to a diagonal of ones."""
    R = np.asarray(returns, dtype=float)
    T, d = R.shape
    out = np.empty((T, d, d))
    for t in range(T):
        lo = max(0, t - window)
        win = R[lo:t]
        if win.shape[0] < 2:
            out[t] = np.eye(d)
            continue
        out[t] = np.cov(win, rowvar=False)
    return out


def static_corr(returns):
    """Full-sample Pearson correlation matrix (d,d)."""
    return np.corrcoef(np.asarray(returns, dtype=float), rowvar=False)


def static_cov(returns):
    """Full-sample covariance matrix (d,d)."""
    return np.cov(np.asarray(returns, dtype=float), rowvar=False)


# --------------------------------------------------------------------------- #
# summaries / metrics
# --------------------------------------------------------------------------- #
def avg_pairwise(R_path):
    """Average off-diagonal (pairwise) correlation at each t: shape (T,)."""
    R_path = np.asarray(R_path, dtype=float)
    T, d, _ = R_path.shape
    iu = np.triu_indices(d, k=1)
    return R_path[:, iu[0], iu[1]].mean(axis=1)


def pair_corr(R_path, i, j):
    """The (i,j) correlation entry at each t: shape (T,)."""
    return np.asarray(R_path, dtype=float)[:, i, j]


def mae(est, true, mask=None):
    est = np.asarray(est, float); true = np.asarray(true, float)
    if mask is not None:
        est, true = est[mask], true[mask]
    return float(np.mean(np.abs(est - true)))


def rmse(est, true, mask=None):
    est = np.asarray(est, float); true = np.asarray(true, float)
    if mask is not None:
        est, true = est[mask], true[mask]
    return float(np.sqrt(np.mean((est - true) ** 2)))


# --------------------------------------------------------------------------- #
# portfolio VaR backtest
# --------------------------------------------------------------------------- #
def portfolio_var_breaches(returns, cov_path, weights, z_quantile):
    """One-step Gaussian VaR backtest for an equal/other-weight portfolio.

    cov_path[t] is the covariance forecast for period t built from information
    through t-1. Portfolio VaR at t is VaR_t = z_quantile * sqrt(w' Sigma_t w)
    (zero-mean Gaussian). A breach is realized r_p,t < -VaR_t. Returns
    (breaches boolean array (T,), var_series (T,), pnl (T,)).
    """
    R = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    T = R.shape[0]
    pnl = R @ w
    var = np.empty(T)
    for t in range(T):
        v = float(w @ cov_path[t] @ w)
        var[t] = z_quantile * np.sqrt(max(v, 0.0))
    breaches = pnl < -var
    return breaches, var, pnl


def breach_rate(breaches, mask=None):
    b = np.asarray(breaches, dtype=bool)
    if mask is not None:
        b = b[mask]
    return float(np.mean(b))
