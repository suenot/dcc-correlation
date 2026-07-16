#!/usr/bin/env python3
"""Verify every numeric claim in paper/main.tex against results/results.json.

Two-way check (matching the flagship arxiv_paper_dsr harness):
  1. Forward: every claim in CLAIMS must appear in the manuscript body as the
     exact literal (word-boundary matched, at least `min_count` times) AND agree
     with the value computed from results.json within rounding tolerance (half a
     unit in the last quoted decimal).
  2. Reverse: after removing all claim literals and a small structural
     allowlist, the body must contain NO remaining multi-digit or decimal
     numeric literals (single digits are exempt: math notation like GARCH(1,1),
     d=2, 1/2).

Plus internal-consistency checks on results.json itself and a check that the
DGP constants quoted in the paper are literally present in the source.

Exit code 0 iff everything passes. Run: python3 scripts/check_paper_numbers.py
"""
import json
import math
import os
import re
import sys

from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "paper", "main.tex")
RESULTS = os.path.join(ROOT, "results", "results.json")

with open(RESULTS) as f:
    R = json.load(f)

M = R["meta"]
T2 = R["corr_tracking"]["d2"]
T3 = R["corr_tracking"]["d3"]
REC = R["corr_tracking"]["recovery"]
RK2 = R["static_risk_misstatement"]["d2"]
RK3 = R["static_risk_misstatement"]["d3"]
HG = R["dynamic_hedge_ratio"]
DIM = R["dimensionality"]
DTAB = {row["d"]: row for row in DIM["table"]}
STAB = DIM["stability"]


# --------------------------------------------------------------------------- #
# internal consistency of results.json
# --------------------------------------------------------------------------- #
def _consistency():
    e = []
    if abs(M["nominal"] - (1.0 - M["var_conf"])) > 1e-12:
        e.append("nominal != 1 - var_conf")
    if abs(M["z_var"] - norm.ppf(M["var_conf"])) > 1e-9:
        e.append("z_var != Phi^-1(var_conf)")
    for tag, RK in (("d2", RK2), ("d3", RK3)):
        if abs(RK["static_crisis_ratio"]
               - RK["static"]["breach_crisis"] / M["nominal"]) > 1e-9:
            e.append(f"{tag}: static_crisis_ratio mismatch")
    for tag, d in (("d2", REC["d2"]), ("d3", REC["d3"])):
        if abs(d["sum_hat"] - (d["a_hat"] + d["b_hat"])) > 1e-9:
            e.append(f"rec {tag}: sum_hat != a_hat+b_hat")
        if abs(d["a_abs_err"] - abs(d["a_hat"] - d["a_true"])) > 1e-9:
            e.append(f"rec {tag}: a_abs_err mismatch")
        if abs(d["b_abs_err"] - abs(d["b_hat"] - d["b_true"])) > 1e-9:
            e.append(f"rec {tag}: b_abs_err mismatch")
    for tag, T in (("d2", T2), ("d3", T3)):
        if abs(T["dcc"]["sum"] - (T["dcc"]["a"] + T["dcc"]["b"])) > 1e-9:
            e.append(f"track {tag}: dcc.sum != a+b")
    for d, row in DTAB.items():
        m = d * (d + 1) // 2
        if row["vech"] != 2 * m * m + m:
            e.append(f"dim d={d}: vech formula")
        if row["bekk"] != m + 2 * d * d:
            e.append(f"dim d={d}: bekk formula")
        if row["dcc_total"] != 3 * d + 2:
            e.append(f"dim d={d}: dcc_total formula")
        if row["dcc_corr"] != 2:
            e.append(f"dim d={d}: dcc_corr != 2")
    return e


# --------------------------------------------------------------------------- #
# DGP constants quoted in the paper, verified present in the source
# --------------------------------------------------------------------------- #
CODE_CONSTANTS = [
    ("G_ALPHA, G_BETA = 0.10, 0.85", "scripts/run_all.py", "GARCH alpha,beta"),
    ("OMEGA_CALM, OMEGA_CRISIS = 0.05, 0.20", "scripts/run_all.py", "omegas"),
    ("RHO_CALM, RHO_CRISIS, RHO_RECOVER = 0.30, 0.90, 0.50",
     "scripts/run_all.py", "regime rho levels"),
    ("VAR_CONF = 0.95", "scripts/run_all.py", "VaR confidence"),
]


def _code_constants():
    e = []
    for needle, rel, what in CODE_CONSTANTS:
        with open(os.path.join(ROOT, rel)) as f:
            if needle not in f.read():
                e.append(f"code constant '{needle}' ({what}) not in {rel}")
    return e


# --------------------------------------------------------------------------- #
# the claims: (label, tex literal, value, min_count)
# --------------------------------------------------------------------------- #
CLAIMS = [
    # --- meta / DGP ---------------------------------------------------------
    ("obs T", "2000", M["config"]["T"], 1),
    ("recovery T", "4000", M["config"]["T_rec"], 1),
    ("crisis start", "600", T2["crisis_window"][0], 1),
    ("crisis end", "1300", T2["crisis_window"][1], 1),
    ("rho calm", "0.3", M["rho_calm"], 1),
    ("rho crisis", "0.9", M["rho_crisis"], 1),
    ("rho recover", "0.5", M["rho_recover"], 1),
    ("garch alpha", "0.10", M["garch_alpha"], 1),
    ("garch beta", "0.85", M["garch_beta"], 1),
    ("omega calm", "0.05", M["omega_calm"], 1),
    ("omega crisis", "0.20", M["omega_crisis"], 1),
    ("var conf", "0.95", M["var_conf"], 1),
    ("var conf pct", "95", M["var_conf"] * 100, 1),
    ("z var", "1.645", M["z_var"], 1),
    ("nominal", "0.05", M["nominal"], 1),
    ("python", "3.14.6", M["python"], 1),
    ("numpy", "2.5.1", M["numpy"], 1),
    # --- experiment 1: tracking (d2 table) ----------------------------------
    ("static value d2", "0.673", T2["static"]["value"], 1),
    ("static mae_o d2", "0.238", T2["static"]["mae_overall"], 1),
    ("static rmse_o d2", "0.259", T2["static"]["rmse_overall"], 1),
    ("static mae_c d2", "0.208", T2["static"]["mae_crisis"], 1),
    ("static rmse_c d2", "0.217", T2["static"]["rmse_crisis"], 1),
    ("r30 mae_o", "0.092", T2["rolling"]["30"]["mae_overall"], 1),
    ("r30 rmse_o", "0.129", T2["rolling"]["30"]["rmse_overall"], 1),
    ("r30 mae_c", "0.037", T2["rolling"]["30"]["mae_crisis"], 1),
    ("r30 rmse_c", "0.048", T2["rolling"]["30"]["rmse_crisis"], 1),
    ("r60 mae_o", "0.069", T2["rolling"]["60"]["mae_overall"], 1),
    ("r60 rmse_o", "0.100", T2["rolling"]["60"]["rmse_overall"], 1),
    ("r60 mae_c", "0.031", T2["rolling"]["60"]["mae_crisis"], 1),
    ("r60 rmse_c", "0.040", T2["rolling"]["60"]["rmse_crisis"], 1),
    ("r120 mae_o", "0.060", T2["rolling"]["120"]["mae_overall"], 1),
    ("r120 rmse_o", "0.087", T2["rolling"]["120"]["rmse_overall"], 1),
    ("r120 mae_c", "0.037", T2["rolling"]["120"]["mae_crisis"], 1),
    ("r120 rmse_c", "0.054", T2["rolling"]["120"]["rmse_crisis"], 1),
    ("r250 mae_o", "0.071", T2["rolling"]["250"]["mae_overall"], 1),
    ("r250 rmse_o", "0.102", T2["rolling"]["250"]["rmse_overall"], 1),
    ("r250 mae_c", "0.074", T2["rolling"]["250"]["mae_crisis"], 1),
    ("r250 rmse_c", "0.101", T2["rolling"]["250"]["rmse_crisis"], 1),
    ("dcc mae_o d2", "0.061", T2["dcc"]["mae_overall"], 1),
    ("dcc rmse_o d2", "0.083", T2["dcc"]["rmse_overall"], 1),
    ("dcc mae_c d2", "0.028", T2["dcc"]["mae_crisis"], 1),
    ("dcc rmse_c d2", "0.039", T2["dcc"]["rmse_crisis"], 1),
    # --- tracking d3 --------------------------------------------------------
    ("dcc mae_c d3", "0.066", T3["dcc"]["mae_crisis"], 1),
    ("dcc a d3", "0.031", T3["dcc"]["a"], 1),
    ("dcc b d3", "0.966", T3["dcc"]["b"], 1),
    ("static mae_c d3", "0.202", T3["static"]["mae_crisis"], 1),
    ("best roll d3", "0.070", T3["best_rolling_crisis_mae"], 1),
    # --- recovery -----------------------------------------------------------
    ("rec a_true d2", "0.05", REC["d2"]["a_true"], 1),
    ("rec b_true d2", "0.90", REC["d2"]["b_true"], 1),
    ("rec a_hat d2", "0.049", REC["d2"]["a_hat"], 1),
    ("rec b_hat d2", "0.914", REC["d2"]["b_hat"], 1),
    ("rec a_err d2", "0.0015", REC["d2"]["a_abs_err"], 1),
    ("rec b_err d2", "0.014", REC["d2"]["b_abs_err"], 1),
    ("rec a_true d3", "0.04", REC["d3"]["a_true"], 1),
    ("rec b_true d3", "0.92", REC["d3"]["b_true"], 1),
    ("rec a_hat d3", "0.040", REC["d3"]["a_hat"], 1),
    ("rec b_hat d3", "0.914", REC["d3"]["b_hat"], 1),
    # --- experiment 2: risk (d2) --------------------------------------------
    ("stat breach_o d2", "0.049", RK2["static"]["breach_overall"], 1),
    ("stat breach_c d2", "0.111", RK2["static"]["breach_crisis"], 1),
    ("stat breach_calm d2", "0.015", RK2["static"]["breach_calm"], 1),
    ("roll breach_o d2", "0.062", RK2["rolling"]["breach_overall"], 1),
    ("roll breach_c d2", "0.069", RK2["rolling"]["breach_crisis"], 1),
    ("roll breach_calm d2", "0.058", RK2["rolling"]["breach_calm"], 1),
    ("dcc breach_o d2", "0.056", RK2["dcc"]["breach_overall"], 1),
    ("dcc breach_c d2", "0.063", RK2["dcc"]["breach_crisis"], 1),
    ("dcc breach_calm d2", "0.052", RK2["dcc"]["breach_calm"], 1),
    ("true breach d2", "0.060", RK2["true"]["breach_crisis"], 1),
    ("static crisis ratio", "2.2", RK2["static_crisis_ratio"], 1),
    # --- risk d3 ------------------------------------------------------------
    ("stat breach_c d3", "0.103", RK3["static"]["breach_crisis"], 1),
    ("dcc breach_c d3", "0.070", RK3["dcc"]["breach_crisis"], 1),
    # --- experiment 3: hedge ------------------------------------------------
    ("beta static", "0.650", HG["beta_static"], 1),
    ("beta true mean", "0.576", HG["beta_true_mean"], 1),
    ("beta dcc mean", "0.575", HG["beta_dcc_mean"], 1),
    ("var red overall", "0.120", HG["var_reduction_overall"], 1),
    ("var red crisis", "0.208", HG["var_reduction_crisis"], 1),
    ("beta mae dcc", "0.067", HG["beta_track_mae_dcc"], 1),
    ("beta mae static", "0.236", HG["beta_track_mae_static"], 1),
    ("beta mae dcc crisis", "0.035", HG["beta_track_mae_dcc_crisis"], 1),
    ("beta mae static crisis", "0.248", HG["beta_track_mae_static_crisis"], 1),
    # --- experiment 4: dimensionality ---------------------------------------
    ("dim d2 vech", "21", DTAB[2]["vech"], 1),
    ("dim d2 bekk", "11", DTAB[2]["bekk"], 1),
    ("dim d2 dcc", "8", DTAB[2]["dcc_total"], 1),
    ("dim d3 vech", "78", DTAB[3]["vech"], 1),
    ("dim d3 bekk", "24", DTAB[3]["bekk"], 1),
    ("dim d3 dcc", "11", DTAB[3]["dcc_total"], 1),  # shares literal w/ d2 bekk
    ("dim d5 vech", "465", DTAB[5]["vech"], 1),
    ("dim d5 bekk", "65", DTAB[5]["bekk"], 1),
    ("dim d5 dcc", "17", DTAB[5]["dcc_total"], 1),
    ("dim d10 vech", "6105", DTAB[10]["vech"], 1),
    ("dim d10 bekk", "255", DTAB[10]["bekk"], 1),
    ("dim d10 dcc", "32", DTAB[10]["dcc_total"], 1),
    ("dim d20 vech", "88410", DTAB[20]["vech"], 1),
    ("dim d20 bekk", "1010", DTAB[20]["bekk"], 1),
    ("dim d20 dcc", "62", DTAB[20]["dcc_total"], 1),
    ("dim d50 vech", "3252525", DTAB[50]["vech"], 1),
    ("dim d50 vech sci", r"3.25\times 10^{6}", DTAB[50]["vech"], 1),
    ("dim d50 bekk", "6275", DTAB[50]["bekk"], 1),
    ("dim d50 dcc", "152", DTAB[50]["dcc_total"], 1),
    ("dims list 10", "10", 10, 1),
    ("dims list 20", "20", 20, 1),
    ("dims list 50", "50", 50, 1),
    # rolling-window lengths (baseline grid + risk window)
    ("roll window 30", "30", M["roll_windows"][0], 1),
    ("roll window 60", "60", M["roll_windows"][1], 1),
    ("roll window 120", "120", M["roll_windows"][2], 1),
    ("roll window 250", "250", M["roll_windows"][3], 1),
    # stability band (max abs errors across d)
    ("stab a band", "0.006", max(s["a_abs_err"] for s in STAB), 1),
    ("stab b band", "0.015", max(s["b_abs_err"] for s in STAB), 1),
]

# structural patterns removed before the reverse sweep (with justification):
ALLOWLIST_PATTERNS = [
    r"GARCH\$\(1,1\)\$",           # model order, math notation
]

SCI_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\\times 10\^\{(-?\d+)\}$")


def _parse_literal(lit):
    m = SCI_RE.match(lit)
    if m:
        mant, expo = m.group(1), int(m.group(2))
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return float(mant) * 10.0 ** expo, (0.5 * 10.0 ** -dec + 1e-12) * 10.0 ** expo
    if re.fullmatch(r"-?\d+\.\d+", lit):
        dec = len(lit.split(".")[1])
        return float(lit), 0.5 * 10.0 ** -dec + 1e-9
    if re.fullmatch(r"-?\d+", lit):
        return float(lit), 1e-9
    return None


def _body(tex):
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    body = m.group(1)
    body = re.sub(
        r"\\(?:eqref|ref|label|pageref|cite[tp]?\*?|bibliographystyle|"
        r"bibliography|href|url)\s*(?:\[[^\]]*\])*\{[^}]*\}", " ", body)
    return body


def main():
    failures = []
    failures += [f"[consistency] {e}" for e in _consistency()]
    failures += [f"[code-const] {e}" for e in _code_constants()]

    with open(TEX) as f:
        body = _body(f.read())

    # forward: count + remove literals, longest first
    literals = sorted({c[1] for c in CLAIMS}, key=len, reverse=True)
    counts = {}
    text = body
    for lit in literals:
        esc = re.escape(lit)
        if lit[0].isdigit() or lit[0] == "-":
            pat = re.compile(r"(?<![\d.])" + esc + r"(?!\d)")
        else:
            pat = re.compile(esc)
        counts[lit] = len(pat.findall(text))
        text = pat.sub(" ", text)

    for label, lit, value, min_count in CLAIMS:
        if counts[lit] < min_count:
            failures.append(f"[presence] {label}: literal '{lit}' found "
                            f"{counts[lit]}x, need >= {min_count}")
        parsed = _parse_literal(lit)
        if parsed is not None:
            num, tol = parsed
            if abs(num - float(value)) > tol:
                failures.append(f"[value] {label}: literal '{lit}' vs "
                                f"results value {value!r} (tol {tol:g})")
        else:
            if lit != str(value):
                failures.append(f"[value] {label}: '{lit}' vs {value!r}")

    # reverse: no unexplained numeric literals may remain
    for pat in ALLOWLIST_PATTERNS:
        text = re.sub(pat, " ", text)
    for m in re.finditer(r"\d+(?:\.\d+)+|\d{2,}", text):
        ctx = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        failures.append(f"[unexplained number] '{m.group(0)}' near: ...{ctx}...")

    if failures:
        print(f"check_paper_numbers: FAIL ({len(failures)} problem(s))")
        for f_ in failures:
            print("  -", f_)
        return 1
    print(f"check_paper_numbers: OK -- {len(CLAIMS)} claims verified against "
          f"results.json; no unexplained numeric literals in main.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
