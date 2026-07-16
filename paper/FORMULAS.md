# Formula spec: DCC-GARCH, baselines, and the risk/hedge metrics

Notation: `r_t` is the `d`-vector of returns at time `t`; `H_t` the conditional
covariance; `D_t = diag(sigma_{1,t}, ..., sigma_{d,t})` the diagonal of
conditional standard deviations; `R_t` the conditional correlation matrix;
`z_t` the standardized-residual vector; `Qbar` the unconditional (targeted)
correlation of the standardized residuals. All quantities are computed in the
returns' native frequency; nothing is annualized in this paper.

---

## 1. Factorization (CCC / DCC)

**Source:** Bollerslev (1990) [`bollerslev1990ccc`] for the constant-correlation
factorization; Engle (2002) [`engle2002dcc`] for the dynamic version.

```
H_t = D_t R_t D_t,     h_{ij,t} = rho_{ij,t} * sigma_{i,t} * sigma_{j,t}
```

Bollerslev's CCC holds `R_t = R` constant; Engle's DCC lets it move. The
factorization is what makes the two-step estimator possible: `D_t` comes from
`d` independent univariate GARCH fits, `R_t` from the correlation recursion
below.

---

## 2. Step 1 — univariate GARCH(1,1) margins (via `arch`)

**Source:** Engle (1982) [`engle1982arch`], Bollerslev (1986)
[`bollerslev1986garch`]; implemented with the `arch` package
[`sheppard_arch`].

For each asset `i`, a constant-mean GARCH(1,1):

```
eps_{i,t} = r_{i,t} - mu_i
sigma_{i,t}^2 = omega_i + alpha_i eps_{i,t-1}^2 + beta_i sigma_{i,t-1}^2
z_{i,t}       = eps_{i,t} / sigma_{i,t}          (standardized residual)
```

`arch` returns `sigma_{i,t}` as `.conditional_volatility` and `z_{i,t}` as
`.std_resid`. We fit on returns scaled by 100 for the optimizer's numerical
health and undo the scaling on the way out (`scripts/dcc.py:garch_std_resid`).

---

## 3. Step 2 — the DCC correlation recursion

**Source:** Engle (2002) [`engle2002dcc`], the scalar (mean-reverting) DCC with
correlation targeting; theoretical/empirical properties in Engle & Sheppard
(2001) [`englesheppard2001dcc`].

```
Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}                    (Q-recursion)
R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}     rho_{ij,t} = q_{ij,t} / sqrt(q_{ii,t} q_{jj,t})
```

- `Qbar` — the unconditional correlation matrix of the standardized residuals,
  **plugged in** (correlation targeting), not optimized. This is what confines
  the entire correlation layer to the two scalars `(a, b)`.
- `a > 0` — the shock loading (how strongly `z_{t-1}z_{t-1}'` pulls `R_t`).
- `b > 0` — the persistence (how much of `Q_{t-1}` carries forward).
- `a + b < 1` — mean reversion, exactly analogous to `alpha + beta < 1` in
  univariate GARCH.

`Q_t` is a convex combination of positive-semidefinite matrices (`Qbar`, the
rank-1 outer product, and `Q_{t-1}`), so it stays positive definite as long as
`Qbar` is; normalizing gives a valid unit-diagonal correlation matrix at every
step. Because `a` and `b` are scalars, every pair of assets shares the same
adjustment speed and persistence — the scalar restriction, the price of
tractability.

**Causal timing (implementation-critical).** In the filter loop `R_t` is
computed from `Q_t` and scored *before* `Q` is advanced with `z_t z_t'`, so
`R_t` depends on `z_0 ... z_{t-1}` only. Reversing the order lets `R_t` see the
contemporaneous shock — the off-by-one that silently inflates in-sample fit.
The first output `R_0` equals the targeted `Qbar` (tested).

---

## 4. Step 2 — the DCC quasi-log-likelihood

**Source:** Engle (2002) [`engle2002dcc`]. Under
`eps_t | F_{t-1} ~ N(0, H_t)` the Gaussian log-likelihood separates into a
volatility part (a function of `D_t` only) and a correlation part (a function
of `R_t`). Maximizing the volatility part is exactly the `d` univariate fits of
Step 1; the correlation step maximizes only

```
L^C(a, b) = -1/2 * sum_t ( log|R_t| + z_t' R_t^{-1} z_t )
```

over `(a, b)`. (The `z_t' z_t` term of the full separation does not depend on
`a, b` and is dropped.) It is a *quasi*-likelihood: the two-step estimator is
consistent but not fully efficient, and correct standard errors need the
Engle & Sheppard (2001) [`englesheppard2001dcc`] correction. We optimize with a
derivative-free multi-start Nelder-Mead search over the feasible region
`a, b > 0, a + b < 1` (`scripts/dcc.py:fit_dcc`); infeasible points return a
large finite penalty.

---

## 5. Baselines

- **Static (full-sample):** the Pearson correlation / covariance of the whole
  return sample, one matrix held fixed for all `t`.
- **Rolling window `w`:** the Pearson correlation / covariance of the last `w`
  observations, `returns[t-w : t]` — strictly causal (uses info `< t`).

Both are what a practitioner reaches for before any GARCH machinery, and are
the estimators DCC must beat to justify itself.

---

## 6. Downstream quantities

**Dynamic hedge ratio** (minimum-variance hedge of asset 1 with asset 2):

```
beta_t = Cov_t(r1, r2) / Var_t(r2) = h_{12,t} / sigma_{2,t}^2
       = rho_{12,t} * sigma_{1,t} / sigma_{2,t}
```

Traded with the lagged `beta_{t-1}` (no look-ahead). The static comparison is
the full-sample OLS hedge ratio `Cov(r1, r2) / Var(r2)`.

**Portfolio Value-at-Risk** (one-step Gaussian, zero-mean, weight vector `w`):

```
sigma_{p,t}^2 = w' Sigma_t w,      VaR_t = z_q * sigma_{p,t},   z_q = Phi^{-1}(conf)
```

A breach is a realized loss `w' r_t < -VaR_t`. `Sigma_t` is in turn the static,
rolling, DCC (`D_t R_t D_t`), or true covariance, each built from information
through `t-1`. The breach rate is compared to the nominal `1 - conf`.

**Asymmetric DCC (aDCC)** — mentioned, not implemented. Cappiello, Engle &
Sheppard (2006) [`cappiello2006asymmetric`] add a term in the negative-part
standardized residuals `z_t^- = min(z_t, 0)` to capture correlations rising more
after joint downside shocks.

---

## 7. Multivariate GARCH parameter counts (dimensionality)

For `d` assets, with `m = d(d+1)/2` the length of the half-vectorization:

- **VECH** [`bollerslev1988vech`]: intercept `vech(H)` (length `m`) plus two
  `m x m` coefficient matrices A, B ⇒ `2 m^2 + m` free parameters, `O(d^4)`.
- **BEKK** [`englekroner1995bekk`]: triangular intercept `C` (`m` params) plus
  two `d x d` matrices A, B ⇒ `m + 2 d^2` free parameters, `O(d^2)`.
- **DCC** [`engle2002dcc`]: 3 per univariate GARCH(1,1) (`omega, alpha, beta`)
  plus the two correlation scalars ⇒ `3d + 2`, `O(d)` — with the correlation
  layer fixed at 2 regardless of `d`.

These are the counts in `scripts/run_all.py` (`vech_params`, `bekk_params`,
`dcc_params`) and Table 3 of the paper.

---

## Verification notes

- The DCC recursion, targeting, quasi-likelihood, and factorization are the
  standard scalar-DCC construction of Engle (2002) and match the reference
  characterization in the accompanying blog post. Equation forms were taken
  from the standard presentation; the primary-source page/equation numbers were
  not re-extracted from the paywalled originals in this pass.
- All bibliographic metadata in `refs.bib` reuses the DOIs/identifiers already
  collected in the accompanying blog draft's References section.
- The synthetic DGPs (Section 2 of the paper) are our own construction for
  controlled ground truth; they are not from any source and make no claim about
  real markets.
