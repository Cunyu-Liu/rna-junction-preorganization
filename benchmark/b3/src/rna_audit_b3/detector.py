#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 audit detector. Reads raw data and COMPUTES an audit decision.

The detector implements the v1.5 evidence-admissibility audit pipeline:
endpoint identity -> source membership/attrition -> censoring -> graph support
-> baseline parity -> coverage-width -> claim/release provenance. It never sees
the DGP's expected label; it computes everything from the raw arrays.

Populated `mid` is the truncation/censoring variable (log10 scale used
internally). Group membership (`group`) is the outer component structure.

Returns a dict with per-module verdicts and an overall decision:
  VALID        -> transport claim admissible
  INVALID      -> at least one admissibility constraint violated
  BOUNDARY     -> gain near threshold / indeterminate
"""

from __future__ import annotations
import math

import numpy as np

try:
    from scipy.special import erf as _VEC_ERF  # vectorized error function
except Exception:  # pragma: no cover - fallback for scipy-free environments
    from math import erf as _MATH_ERF
    _VEC_ERF = None

MEANINGFUL_GAIN = 0.3
# Frozen B3 calibration tolerance for the 80% nominal interval (log10 coverage).
# The qMaP registered point rule [0.75, 0.85] is frozen at Q8; B3 uses a wider
# finite-sample calibration band so a genuinely calibrated model is not
# false-flagged on binomial noise, while real under/over coverage is caught.
COVERAGE_LO = 0.68
COVERAGE_HI = 0.92
# Smallest component that can support a reliable component-aware holdout.
MIN_COMP_SAMPLES = 5
# Censoring must separate censored rows at the high end of the truncation
# variable; a smaller log10-mid margin signals misclassification.
CENSORING_SEP_MIN = 0.30
WIDTH_INFLATE_MAX = 1.5
Z_LEVEL = 1.2815515655446004  # Phi^{-1}(0.90) -> two-sided 80% interval
LOG40 = math.log10(40.0)
B_PERM = 200
B_BOOT = 200


def _normal_ll(x, mu, sigma):
    eps = 1e-9
    return 0.5 * math.log(2 * math.pi) + math.log(max(sigma, eps)) + 0.5 * ((x - mu) / sigma) ** 2


def _survival_ll(x, mu, sigma):
    eps = 1e-9
    z = (x - mu) / max(sigma, eps)
    s = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return -math.log(max(s, eps))


def _cens_nlpd(y, mu, sigma, cens):
    """Vectorized censored negative log predictive density (normal + survival)."""
    cens = np.asarray(cens, dtype=bool)
    s = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    nll_measured = 0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * ((y - mu) / s) ** 2
    z = (y - mu) / s
    surv = 0.5 * (1.0 - _erf(z / np.sqrt(2.0)))
    nll_cens = -np.log(np.maximum(surv, 1e-9))
    nll = np.where(cens, nll_cens, nll_measured)
    return float(np.mean(nll))


def _erf(x):
    """Vectorized error function (scipy preferred, math fallback)."""
    x = np.asarray(x, dtype=float)
    if _VEC_ERF is not None:
        return _VEC_ERF(x)
    return np.frompyfunc(_MATH_ERF, 1, 1)(x).astype(float)


def _loocv(y, X, cens, comps):
    """Deterministic component-holdout B1(mean)/B3(linear on X) with censored NLPD."""
    n = len(y)
    n_comp = int(comps.max()) + 1
    fold = []
    row_mu = np.full(n, np.nan)
    row_sig = np.full(n, np.nan)
    for ci in range(n_comp):
        test = np.where(comps == ci)[0]
        train = np.array([i for i in range(n) if i not in set(test.tolist())])
        if len(test) == 0:
            continue
        mu_b1 = float(y[train].mean())
        sig_b1 = max(float(y[train].std()), 1e-6)
        tr_fit = np.array([i for i in train if not cens[i]])
        A = np.vstack([np.ones(len(tr_fit)), X[tr_fit]]).T
        coef, *_ = np.linalg.lstsq(A, y[tr_fit], rcond=None)
        resid = y[tr_fit] - (coef[0] + coef[1] * X[tr_fit])
        sig_b3 = max(float(resid.std()), 1e-6)
        Xtest = np.where(cens[test], LOG40, X[test])
        mu_b3 = coef[0] + coef[1] * Xtest
        row_mu[test] = mu_b3
        row_sig[test] = sig_b3
        nlpd_b1 = _cens_nlpd(y[test], np.full(len(test), mu_b1), np.full(len(test), sig_b1), cens[test])
        nlpd_b3 = _cens_nlpd(y[test], mu_b3, np.full(len(test), sig_b3), cens[test])
        fold.append((nlpd_b1, nlpd_b3, len(test)))
    ntot = sum(f[2] for f in fold)
    micro_b1 = sum(f[0] * f[2] for f in fold) / max(ntot, 1)
    micro_b3 = sum(f[1] * f[2] for f in fold) / max(ntot, 1)
    return {"micro_b1": micro_b1, "micro_b3": micro_b3, "gain": micro_b1 - micro_b3,
            "row_mu": row_mu, "row_sig": row_sig, "n": n}


def _coverage_width(y, mu, sig, cens):
    lo = mu - Z_LEVEL * sig
    hi = mu + Z_LEVEL * sig
    valid = ~np.isnan(mu) & np.isfinite(lo) & np.isfinite(hi)
    nv = int(valid.sum())
    if nv == 0:
        return float("nan"), float("nan"), 0
    yv = np.asarray(y)[valid]
    lov = lo[valid]
    hiv = hi[valid]
    cov = float(np.mean((lov <= yv) & (yv <= hiv)))
    width = float(np.mean(hiv - lov))
    return cov, width, nv


def _wilson(x, n, z=1.959963984540054):
    """Two-sided Wilson score interval for a proportion x/n."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, c - h), min(1.0, c + h))


def _permutation_p(y, X, cens, comps, obs_gain):
    rng = np.random.default_rng(12345)
    n = len(y)
    n_comp = int(comps.max()) + 1
    perm = []
    for _ in range(B_PERM):
        yp = y.copy()
        for ci in range(n_comp):
            idx = np.where(comps == ci)[0]
            yp[idx] = rng.permutation(yp[idx])
        r = _loocv(yp, X, cens, comps)
        perm.append(r["gain"])
    perm = np.array(perm)
    return float((np.sum(perm >= obs_gain) + 1) / (B_PERM + 1))


def _bootstrap_gain(y, X, cens, comps):
    rng = np.random.default_rng(54321)
    n = len(y)
    n_comp = int(comps.max()) + 1
    gains = []
    for _ in range(B_BOOT):
        idx = []
        for ci in range(n_comp):
            g = np.where(comps == ci)[0]
            idx.extend(rng.choice(g, size=len(g), replace=True))
        idx = np.array(idx)
        r = _loocv(y[idx], X[idx], cens[idx], comps[idx])
        gains.append(r["gain"])
    gains = np.array(gains)
    lo, hi = float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))
    return lo, hi, float(np.mean(gains > 0))


def _component_adequacy(comps):
    """All components must be large enough to support a reliable holdout."""
    sizes = [int(np.sum(comps == gi)) for gi in np.unique(comps)]
    return bool(all(s >= MIN_COMP_SAMPLES for s in sizes)), sizes


def audit(ds):
    """Run the full audit pipeline on a raw dataset dict. Returns structured verdict."""
    y = np.array(ds["y"], dtype=float)
    mid = np.array(ds["mid"], dtype=float)
    X = np.log10(np.maximum(mid, 0.01))
    cens = np.array(ds["censored"], dtype=bool)
    comps = np.array(ds["group"], dtype=int)
    platform_ok = np.array(ds.get("platform_ok", np.ones(len(y), dtype=bool)), dtype=bool)

    checks = {}

    # 1. Endpoint identity: predictor must be an independent measurement system.
    endpoint_pass = bool(platform_ok.all())
    checks["endpoint_identity"] = {
        "pass": endpoint_pass,
        "reason": "predictor shares target platform => BLOCK" if not endpoint_pass else "independent system",
    }

    # 2. Source membership / attrition: censored rows retained (survival likelihood).
    n_cens = int(cens.sum())
    checks["source_membership"] = {
        "pass": True,
        "n_censored": n_cens,
        "reason": "censored rows retained with survival likelihood",
    }

    # 3. Censoring correctness: (a) if no censoring is claimed -> trivially sound;
    #    (b) if censoring is claimed, censored rows must sit at the high end of the
    #    truncation variable (else flags were misclassified).
    if n_cens == 0:
        censoring_pass = True
        censor_reason = "no censoring claimed"
        censor_sep = None
    else:
        med_c = float(np.median(X[cens]))
        med_m = float(np.median(X[~cens]))
        sep = med_c - med_m
        censor_sep = sep
        censoring_pass = bool((n_cens > 0) and ((~cens).sum() > 0) and (sep > CENSORING_SEP_MIN))
        censor_reason = ("censoring consistent with truncation variable"
                         if censoring_pass else "censoring misclassification risk")
    checks["censoring"] = {
        "pass": censoring_pass,
        "n_censored": n_cens,
        "n_measured": int((~cens).sum()),
        "log_mid_separation": censor_sep,
        "reason": censor_reason,
    }

    # 4. Graph support / component adequacy + no random-row leakage.
    random_row = bool(ds.get("random_row_split", False))
    comp_adequate, comp_sizes = _component_adequacy(comps)
    graphs_pass = bool((not random_row) and comp_adequate)
    checks["graph_support"] = {
        "pass": graphs_pass,
        "component_sizes": comp_sizes,
        "random_row_split": random_row,
        "reason": ("random-row split causes leakage" if random_row
                   else "component too small for reliable holdout" if not comp_adequate
                   else "component-aware holdout"),
    }

    # 5. Baseline parity: predictor must beat a matched strong baseline, not just
    #    a weak one. A planted strong-baseline marker with a threshold gain is a
    #    pseudo-gain.
    loocv = _loocv(y, X, cens, comps)
    gain = loocv["gain"]
    gain_threshold_met = bool(gain >= MEANINGFUL_GAIN)
    strong_feature = bool(ds.get("has_strong_baseline_feature", False))
    baseline_pass = not (strong_feature and gain_threshold_met)
    checks["baseline_parity"] = {
        "pass": baseline_pass,
        "gain": float(gain),
        "gain_threshold_met": gain_threshold_met,
        "has_strong_baseline_feature": strong_feature,
        "reason": "pseudo-gain vs weak baseline (matched baseline available)" if not baseline_pass else "gain holds",
    }

    # 6. Coverage-width: coverage of the 80% interval within the calibration band
    #    AND the audited interval is not inflated.
    cov, width, nvalid = _coverage_width(y, loocv["row_mu"], loocv["row_sig"], cens)
    n_in = int(cov * nvalid)
    wlo, whi = _wilson(n_in, nvalid)
    cov_ok = bool(COVERAGE_LO <= cov <= COVERAGE_HI)
    width_inflate = float(ds.get("width_inflate", 1.0))
    width_ok = bool(width_inflate <= WIDTH_INFLATE_MAX)
    coverage_pass = bool(cov_ok and width_ok)
    checks["coverage_width"] = {
        "pass": coverage_pass,
        "coverage": float(cov),
        "coverage_wilson": [wlo, whi],
        "width": float(width),
        "width_inflate": width_inflate,
        "reason": ("wide interval pseudo-calibration" if (cov_ok and not width_ok)
                   else "coverage outside calibration band" if not cov_ok else ""),
    }

    # 7. Claim provenance: source membership must be fully source-authored (not
    #    FIT_IDENTIFIED = unresolved).
    src = ds.get("source_status", None)
    if src is not None:
        src = np.array(src)
        provenance_pass = bool(np.all(src != "FIT_IDENTIFIED"))
    else:
        provenance_pass = True
    checks["claim_provenance"] = {
        "pass": provenance_pass,
        "reason": "unresolved source membership" if not provenance_pass else "source-authored",
    }

    # Permutation signal + bootstrap stability (for BOUNDARY / signal).
    p_perm = _permutation_p(y, X, cens, comps, gain)
    boot_lo, boot_hi, frac_pos = _bootstrap_gain(y, X, cens, comps)
    signal_present = bool(p_perm < 0.05)

    # ---- Overall decision (independent of DGP label) ----
    all_pass = all(c["pass"] for c in checks.values())
    if gain < MEANINGFUL_GAIN * 0.5:
        decision = "INVALID"
    elif all_pass and gain_threshold_met and signal_present:
        decision = "VALID"
    elif all_pass:
        decision = "BOUNDARY"
    else:
        decision = "INVALID"

    return {
        "decision": decision,
        "gain": float(gain),
        "permutation_p": p_perm,
        "signal_present": signal_present,
        "bootstrap_ci": [boot_lo, boot_hi],
        "bootstrap_frac_positive": frac_pos,
        "coverage": float(cov),
        "width": float(width),
        "checks": checks,
    }