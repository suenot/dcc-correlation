"""DCC-GARCH controlled-study harness -> results/results.json.

Four seeded experiments with KNOWN ground truth (synthetic DGPs; no market
data):

  1. corr_tracking          : a 2- and 3-asset DCC-GARCH DGP with a KNOWN
     time-varying correlation path (calm rho=0.3 -> crisis ramp to rho=0.9 ->
     partial recovery to 0.5). Fit the self-contained DCC estimator and compare
     its tracking of the TRUE rho path against rolling-window Pearson (several
     windows) and static full-sample correlation, overall and inside the crisis
     window. Also recover the TRUE DCC (a,b) from a stationary DCC simulation.
  2. static_risk_misstatement : equal-weight portfolio one-step Gaussian VaR
     under static / rolling / DCC / TRUE covariance; backtest breach coverage
     overall and in the crisis window. Static and rolling UNDER-state risk in
     the correlation+volatility spike; DCC stays near nominal.
  3. dynamic_hedge_ratio    : on a simulated pair, time-varying hedge ratio
     beta_t = rho_t sigma2_t/sigma1_t from DCC vs a static OLS hedge; measure
     hedged-spread variance reduction and tracking error to the TRUE beta_t.
  4. dimensionality         : analytical free-parameter counts (VECH O(d^4) vs
     BEKK O(d^2) vs DCC O(d)) as d grows, plus empirical DCC (a,b) recovery
     stability across d.

Everything is seeded and deterministic. Run: python scripts/run_all.py [--quick]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

VAR_CONF = 0.95                          # one-sided VaR confidence
Z_VAR = float(norm.ppf(VAR_CONF))        # ~1.645
NOMINAL = round(1.0 - VAR_CONF, 4)       # nominal breach rate 0.05

# regime correlation levels (the KNOWN ground truth of the DGP)
RHO_CALM, RHO_CRISIS, RHO_RECOVER = 0.30, 0.90, 0.50
# per-asset GARCH(1,1) coefficients (shared across assets in the DGP)
G_ALPHA, G_BETA = 0.10, 0.85
OMEGA_CALM, OMEGA_CRISIS = 0.05, 0.20    # variance intercept: calm vs crisis
ROLL_WINDOWS = [30, 60, 120, 250]


# --------------------------------------------------------------------------- #
# the regime DGP: known time-varying correlation AND volatility
# --------------------------------------------------------------------------- #
def _equicorr(rho, d):
    R = np.full((d, d), rho)
    np.fill_diagonal(R, 1.0)
    return R


def _schedule(T):
    """Piecewise regime path over T steps. Returns (rho_t, omega_t, crisis_mask)
    with a calm -> ramp-up -> crisis plateau -> recovery -> calm2 structure."""
    t1, t2, t3, t4 = int(0.30 * T), int(0.40 * T), int(0.65 * T), int(0.75 * T)
    rho = np.empty(T)
    omega = np.empty(T)
    for t in range(T):
        if t < t1:                                   # calm
            f = 0.0; reg = "calm"
        elif t < t2:                                 # ramp up
            f = (t - t1) / (t2 - t1); reg = "ramp"
        elif t < t3:                                 # crisis plateau
            f = 1.0; reg = "crisis"
        elif t < t4:                                 # recovery
            g = (t - t3) / (t4 - t3)
            rho[t] = RHO_CRISIS + g * (RHO_RECOVER - RHO_CRISIS)
            omega[t] = OMEGA_CRISIS + g * (OMEGA_CALM * 1.6 - OMEGA_CRISIS)
            continue
        else:                                        # calm2 (partial recovery)
            rho[t] = RHO_RECOVER
            omega[t] = OMEGA_CALM * 1.6
            continue
        rho[t] = RHO_CALM + f * (RHO_CRISIS - RHO_CALM)
        omega[t] = OMEGA_CALM + f * (OMEGA_CRISIS - OMEGA_CALM)
    crisis_mask = np.zeros(T, dtype=bool)
    crisis_mask[t1:t3] = True                        # ramp + plateau = stress
    return rho, omega, crisis_mask, (t1, t2, t3, t4)


def simulate_regime_dgp(T, d, seed):
    """Simulate d correlated GARCH(1,1) series with a KNOWN time-varying
    equicorrelation path. Returns dict with returns, true rho path, true sigma
    path, true covariance path, and the crisis mask."""
    rng = np.random.default_rng(seed)
    rho_t, omega_t, crisis_mask, cuts = _schedule(T)
    uncond = OMEGA_CALM / (1.0 - G_ALPHA - G_BETA)   # calm unconditional var
    sigma2 = np.full(d, uncond)
    eps_prev = np.zeros(d)
    returns = np.empty((T, d))
    sigma_path = np.empty((T, d))
    R_true = np.empty((T, d, d))
    H_true = np.empty((T, d, d))
    for t in range(T):
        sigma2 = omega_t[t] + G_ALPHA * eps_prev ** 2 + G_BETA * sigma2
        sigma = np.sqrt(sigma2)
        R = _equicorr(rho_t[t], d)
        L = np.linalg.cholesky(R)
        u = L @ rng.standard_normal(d)
        eps = sigma * u
        returns[t] = eps
        sigma_path[t] = sigma
        R_true[t] = R
        H_true[t] = np.diag(sigma) @ R @ np.diag(sigma)
        eps_prev = eps
    return dict(returns=returns, rho_t=rho_t, sigma_path=sigma_path,
                R_true=R_true, H_true=H_true, crisis_mask=crisis_mask,
                cuts=cuts)


# --------------------------------------------------------------------------- #
# experiment 1: correlation tracking + (a,b) recovery
# --------------------------------------------------------------------------- #
def track_metrics(est_path_scalar, true_scalar, mask):
    return dict(
        mae_overall=dcc.mae(est_path_scalar, true_scalar),
        rmse_overall=dcc.rmse(est_path_scalar, true_scalar),
        mae_crisis=dcc.mae(est_path_scalar, true_scalar, mask),
        rmse_crisis=dcc.rmse(est_path_scalar, true_scalar, mask),
    )


def corr_tracking_case(T, d, seed):
    sim = simulate_regime_dgp(T, d, seed)
    ret = sim["returns"]
    true_rho = sim["rho_t"]                          # equicorr scalar path
    mask = sim["crisis_mask"]

    # DCC: univariate GARCH per asset (arch) -> standardized residuals -> fit
    Z, Sigma = dcc.garch_panel(ret, scale=100.0)
    fit = dcc.fit_dcc(Z)
    R_hat = dcc.dcc_filter(Z, fit["a"], fit["b"])
    dcc_scalar = dcc.avg_pairwise(R_hat)
    dcc_m = track_metrics(dcc_scalar, true_rho, mask)
    dcc_m.update(a=fit["a"], b=fit["b"], sum=fit["sum"])

    # rolling-window Pearson baselines
    rolling = {}
    for w in ROLL_WINDOWS:
        Rr = dcc.rolling_corr_path(ret, w)
        rolling[str(w)] = track_metrics(dcc.avg_pairwise(Rr), true_rho, mask)

    # static full-sample correlation baseline
    sc = dcc.static_corr(ret)
    iu = np.triu_indices(d, k=1)
    static_scalar = np.full(T, float(sc[iu].mean()))
    static_m = track_metrics(static_scalar, true_rho, mask)
    static_m["value"] = float(sc[iu].mean())

    best_roll_crisis = min(rolling, key=lambda w: rolling[w]["mae_crisis"])
    best_roll_overall = min(rolling, key=lambda w: rolling[w]["mae_overall"])
    return dict(
        d=d, T=T, crisis_window=[int(sim["cuts"][0]), int(sim["cuts"][2])],
        rho_calm=RHO_CALM, rho_crisis=RHO_CRISIS, rho_recover=RHO_RECOVER,
        dcc=dcc_m, rolling=rolling, static=static_m,
        best_rolling_crisis=best_roll_crisis,
        best_rolling_overall=best_roll_overall,
        best_rolling_crisis_mae=rolling[best_roll_crisis]["mae_crisis"],
        best_rolling_overall_mae=rolling[best_roll_overall]["mae_overall"],
    )


def dcc_recovery_case(T, d, seed, a_true, b_true, off):
    """Recover known (a,b) from a stationary DCC simulation (fixed Qbar)."""
    rng = np.random.default_rng(seed)
    if d == 2:
        Qbar = _equicorr(off, 2)
    else:
        Qbar = _equicorr(off, d)
    Z = dcc.simulate_dcc_z(T, Qbar, a_true, b_true, rng)
    fit = dcc.fit_dcc(Z)
    return dict(d=d, T=T, a_true=a_true, b_true=b_true, qbar_off=off,
                a_hat=fit["a"], b_hat=fit["b"], sum_hat=fit["sum"],
                a_abs_err=abs(fit["a"] - a_true),
                b_abs_err=abs(fit["b"] - b_true),
                sum_abs_err=abs(fit["sum"] - (a_true + b_true)))


# --------------------------------------------------------------------------- #
# experiment 2: static/rolling covariance under-states risk in the spike
# --------------------------------------------------------------------------- #
def risk_misstatement_case(T, d, seed, roll_window):
    sim = simulate_regime_dgp(T, d, seed)
    ret = sim["returns"]
    mask = sim["crisis_mask"]
    w = np.full(d, 1.0 / d)                           # equal weight

    Z, Sigma = dcc.garch_panel(ret, scale=100.0)
    fit = dcc.fit_dcc(Z)
    R_hat = dcc.dcc_filter(Z, fit["a"], fit["b"])
    # DCC one-step covariance path H_t = D_t R_t D_t (causal: R_t uses info<t,
    # Sigma_t is the arch conditional vol for t)
    H_dcc = np.empty((T, d, d))
    for t in range(T):
        D = np.diag(Sigma[t])
        H_dcc[t] = D @ R_hat[t] @ D

    cov_static = dcc.static_cov(ret)
    static_path = np.broadcast_to(cov_static, (T, d, d))
    roll_path = dcc.rolling_cov_path(ret, roll_window)
    true_path = sim["H_true"]

    out = dict(d=d, T=T, weights="equal", var_conf=VAR_CONF, nominal=NOMINAL,
               roll_window=roll_window,
               crisis_window=[int(sim["cuts"][0]), int(sim["cuts"][2])])
    for name, path in (("static", static_path), ("rolling", roll_path),
                       ("dcc", H_dcc), ("true", true_path)):
        br, var, pnl = dcc.portfolio_var_breaches(ret, path, w, Z_VAR)
        out[name] = dict(breach_overall=dcc.breach_rate(br),
                         breach_crisis=dcc.breach_rate(br, mask),
                         breach_calm=dcc.breach_rate(br, ~mask))
    out["dcc_a"], out["dcc_b"] = fit["a"], fit["b"]
    # how badly each mis-states the crisis relative to nominal (ratio)
    out["static_crisis_ratio"] = out["static"]["breach_crisis"] / NOMINAL
    out["rolling_crisis_ratio"] = out["rolling"]["breach_crisis"] / NOMINAL
    out["dcc_crisis_ratio"] = out["dcc"]["breach_crisis"] / NOMINAL
    return out


# --------------------------------------------------------------------------- #
# experiment 3: dynamic hedge ratio
# --------------------------------------------------------------------------- #
def hedge_ratio_case(T, seed):
    d = 2
    sim = simulate_regime_dgp(T, d, seed)
    ret = sim["returns"]
    mask = sim["crisis_mask"]
    sig = sim["sigma_path"]
    rho = sim["rho_t"]
    # hedge asset 0 exposure with asset 1: beta_t = rho_t sigma0_t / sigma1_t
    beta_true = rho * sig[:, 0] / sig[:, 1]

    Z, Sigma = dcc.garch_panel(ret, scale=100.0)
    fit = dcc.fit_dcc(Z)
    R_hat = dcc.dcc_filter(Z, fit["a"], fit["b"])
    rho_hat = dcc.pair_corr(R_hat, 0, 1)
    beta_dcc = rho_hat * Sigma[:, 0] / Sigma[:, 1]

    cov = dcc.static_cov(ret)
    beta_static = float(cov[0, 1] / cov[1, 1])        # OLS hedge ratio

    # hedged spread with LAGGED hedge (use info through t-1)
    def spread_var(beta_series, m=None):
        b = np.asarray(beta_series)
        sp = ret[1:, 0] - b[:-1] * ret[1:, 1]
        if m is not None:
            sp = sp[m[1:]]
        return float(np.var(sp))

    v_dcc = spread_var(beta_dcc)
    v_static = spread_var(np.full(T, beta_static))
    v_unhedged = float(np.var(ret[1:, 0]))
    v_dcc_cr = spread_var(beta_dcc, mask)
    v_static_cr = spread_var(np.full(T, beta_static), mask)

    return dict(
        T=T, d=d, crisis_window=[int(sim["cuts"][0]), int(sim["cuts"][2])],
        beta_static=beta_static,
        beta_true_mean=float(beta_true.mean()),
        beta_dcc_mean=float(beta_dcc.mean()),
        var_reduction_overall=float(1.0 - v_dcc / v_static),
        var_reduction_crisis=float(1.0 - v_dcc_cr / v_static_cr),
        var_dcc=v_dcc, var_static=v_static, var_unhedged=v_unhedged,
        beta_track_mae_dcc=dcc.mae(beta_dcc, beta_true),
        beta_track_mae_static=dcc.mae(np.full(T, beta_static), beta_true),
        beta_track_mae_dcc_crisis=dcc.mae(beta_dcc, beta_true, mask),
        beta_track_mae_static_crisis=dcc.mae(np.full(T, beta_static),
                                             beta_true, mask),
        dcc_a=fit["a"], dcc_b=fit["b"],
    )


# --------------------------------------------------------------------------- #
# experiment 4: dimensionality
# --------------------------------------------------------------------------- #
def vech_params(d):
    m = d * (d + 1) // 2
    return int(2 * m * m + m)          # intercept (m) + A (m^2) + B (m^2)


def bekk_params(d):
    return int(d * (d + 1) // 2 + 2 * d * d)   # C'C + A (d^2) + B (d^2)


def dcc_params(d):
    return int(3 * d + 2)              # 3 per univariate GARCH(1,1) + (a,b)


def dimensionality_block(dims_table, stability_cfg, seed0):
    table = []
    for d in dims_table:
        table.append(dict(d=d, vech=vech_params(d), bekk=bekk_params(d),
                          dcc_total=dcc_params(d), dcc_corr=2))
    stability = []
    for k, d in enumerate(stability_cfg["dims"]):
        rec = dcc_recovery_case(stability_cfg["T"], d, seed0 + 17 * (k + 1),
                                stability_cfg["a_true"], stability_cfg["b_true"],
                                stability_cfg["qbar_off"])
        stability.append(rec)
    return dict(table=table, stability=stability,
                a_true=stability_cfg["a_true"], b_true=stability_cfg["b_true"])


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        cfg = dict(T=800, T_rec=1200, roll_window=60,
                   dims_table=[2, 3, 5, 10],
                   stab=dict(dims=[2, 3, 5], T=1000,
                             a_true=0.05, b_true=0.90, qbar_off=0.4))
    else:
        cfg = dict(T=2000, T_rec=4000, roll_window=60,
                   dims_table=[2, 3, 5, 10, 20, 50],
                   stab=dict(dims=[2, 3, 5, 10], T=4000,
                             a_true=0.05, b_true=0.90, qbar_off=0.4))

    print("[1/4] correlation tracking (2- and 3-asset) ...", flush=True)
    track2 = corr_tracking_case(cfg["T"], 2, seed=101)
    track3 = corr_tracking_case(cfg["T"], 3, seed=202)
    print("[1b] DCC (a,b) recovery ...", flush=True)
    rec2 = dcc_recovery_case(cfg["T_rec"], 2, 303, 0.05, 0.90, 0.4)
    rec3 = dcc_recovery_case(cfg["T_rec"], 3, 404, 0.04, 0.92, 0.4)

    print("[2/4] static-risk misstatement (VaR backtest) ...", flush=True)
    risk2 = risk_misstatement_case(cfg["T"], 2, seed=101,
                                   roll_window=cfg["roll_window"])
    risk3 = risk_misstatement_case(cfg["T"], 3, seed=202,
                                   roll_window=cfg["roll_window"])

    print("[3/4] dynamic hedge ratio ...", flush=True)
    hedge = hedge_ratio_case(cfg["T"], seed=101)

    print("[4/4] dimensionality (param counts + stability) ...", flush=True)
    dim = dimensionality_block(cfg["dims_table"], cfg["stab"], seed0=900)

    results = dict(
        meta=dict(
            quick=bool(args.quick),
            python=sys.version.split()[0], numpy=np.__version__,
            var_conf=VAR_CONF, z_var=Z_VAR, nominal=NOMINAL,
            rho_calm=RHO_CALM, rho_crisis=RHO_CRISIS, rho_recover=RHO_RECOVER,
            garch_alpha=G_ALPHA, garch_beta=G_BETA,
            omega_calm=OMEGA_CALM, omega_crisis=OMEGA_CRISIS,
            roll_windows=ROLL_WINDOWS, roll_window_risk=cfg["roll_window"],
            config=cfg,
        ),
        corr_tracking=dict(d2=track2, d3=track3,
                           recovery=dict(d2=rec2, d3=rec3)),
        static_risk_misstatement=dict(d2=risk2, d3=risk3),
        dynamic_hedge_ratio=hedge,
        dimensionality=dim,
    )

    out = os.path.join(ROOT, "results",
                       "results_quick.json" if args.quick else "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)

    # headline
    print("\n=== headline ===")
    t = track2
    print(f"[track d=2] DCC a={t['dcc']['a']:.3f} b={t['dcc']['b']:.3f} "
          f"| crisis MAE: DCC {t['dcc']['mae_crisis']:.3f}, "
          f"best rolling({t['best_rolling_crisis']}) "
          f"{t['best_rolling_crisis_mae']:.3f}, "
          f"static {t['static']['mae_crisis']:.3f}")
    print(f"[recover d=2] true(0.05,0.90) -> ({rec2['a_hat']:.3f},"
          f"{rec2['b_hat']:.3f})  d=3 true(0.04,0.92) -> "
          f"({rec3['a_hat']:.3f},{rec3['b_hat']:.3f})")
    r = risk2
    print(f"[risk d=2] crisis breach (nominal {NOMINAL}): "
          f"static {r['static']['breach_crisis']:.3f}, "
          f"rolling {r['rolling']['breach_crisis']:.3f}, "
          f"dcc {r['dcc']['breach_crisis']:.3f}, "
          f"true {r['true']['breach_crisis']:.3f}")
    print(f"[hedge] var reduction overall {hedge['var_reduction_overall']:.3f}, "
          f"crisis {hedge['var_reduction_crisis']:.3f}")
    print("[dims] " + ", ".join(
        f"d={r['d']}:VECH{r['vech']}/BEKK{r['bekk']}/DCC{r['dcc_total']}"
        for r in dim["table"]))


if __name__ == "__main__":
    main()
