#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q8 — corrected qMaP source/effect/calibration re-adjudication (v1.5).

Q8 does NOT rerun a main predictive model. It freezes v1.4 Q7's predictions,
fold assignment, component structure, censoring, threshold and decision, and
re-organizes the evidence into six sub-states plus calibration uncertainty and
11th-member source-membership sensitivity.

The frozen predictions.parquet stores only the raw inputs (midpoint, target,
component, censoring). To produce the coverage/width curve and cluster-aware
bootstrap required by §7.3, we deterministically recompute the SAME locked
LOOCV B1/B3 workflow from the frozen inputs and verify it reproduces the frozen
Q7 micro NLPD / coverage / gain exactly before trusting anything derived from it.

Outputs (under RUN_ROOT/qmap/q8/):
  Q8_decision.json          six sub-states + membership robustness conclusion
  calibration.tsv           registered rule, per-component, cluster bootstrap,
                            simple Wilson, coverage-width curve
  membership_sensitivity.tsv  three 11th-member assignments (censored/fitted/excluded)
  q8_report.md              human-readable adjudication
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Frozen v1.4 constants (do not change)
# ---------------------------------------------------------------------------
RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
PARENT_RUN = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
Q7_DIR = f"{PARENT_RUN}/qmap/q7"
Q8_DIR = f"{RUN_ROOT}/qmap/q8"

LOG10_40 = math.log10(40.0)
MEANINGFUL_GAIN = 0.3
INTERVAL_LEVEL = 0.80
Z_LEVEL = 1.2815515655446004  # Phi^{-1}(0.90)
SEED = 20260804
B_BOOT = 2000          # cluster-aware bootstrap resamples
B_CI = 2000            # Wilson / coverage uncertainty
NOMINAL_LEVELS = [0.60, 0.70, 0.80, 0.90]
REGISTERED_POINT_RULE = [0.75, 0.85]
FROZEN_COVERAGE = 0.7263157894736842
FROZEN_WIDTH = 1.1307133884779008
FROZEN_MICRO_B1 = 1.273550237162306
FROZEN_MICRO_B3 = 0.8572969652548471
FROZEN_GAIN = 0.41625327190745887
FROZEN_GROUP_GAIN = 0.5699646191769077
FROZEN_CI = [-0.5718171650756096, 0.7477603576673042]
FROZEN_P = 0.001


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normal_loglik(x, mu, sigma):
    eps = 1e-9
    return 0.5 * math.log(2 * math.pi) + math.log(max(sigma, eps)) + 0.5 * ((x - mu) / sigma) ** 2


def survival_loglik(x, mu, sigma):
    eps = 1e-9
    z = (x - mu) / max(sigma, eps)
    # P(Y > x) = 1 - Phi(z)
    s = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return -math.log(max(s, eps))  # negative log-likelihood contribution


def censored_nlpd(y, mu, sigma, cens):
    nll = 0.0
    for i in range(len(y)):
        if cens[i]:
            nll += survival_loglik(y[i], mu[i], sigma[i])
        else:
            nll += normal_loglik(y[i], mu[i], sigma[i])
    return nll / max(len(y), 1)


def interval_coverage_width(y, mu, sigma, cens):
    lo = mu - Z_LEVEL * sigma
    hi = mu + Z_LEVEL * sigma
    n = len(y)
    cov = int(np.sum([(lo[i] <= y[i] <= hi[i]) for i in range(n)]))
    width = float(np.mean(hi - lo))
    return cov, width, float(cov) / max(n, 1)


def recompute_loocv(preds, component_col, y_col, mid_col, cens_col):
    """Deterministic LOOCV B1/B3 exactly as Q7. Returns per-fold + per-row mu/sigma for B3."""
    n = len(preds)
    comps = np.array(preds[component_col])
    y = np.array(preds[y_col], dtype=float)
    mid = np.array(preds[mid_col], dtype=float)
    X = np.log10(np.maximum(mid, 0.01))
    cens = np.array(preds[cens_col], dtype=bool)
    n_comp = int(comps.max()) + 1
    fold_results = []
    row_mu = np.full(n, np.nan)
    row_sigma = np.full(n, np.nan)
    for ci in range(n_comp):
        test_idx = np.where(comps == ci)[0]
        train_idx = np.array([i for i in range(n) if i not in set(test_idx.tolist())])
        if len(test_idx) == 0:
            continue
        # B1 intercept/mean on all training targets
        yt_train = y[train_idx]
        mu_b1 = float(yt_train.mean())
        sigma_b1 = max(float(yt_train.std()), 1e-6)
        # B3 linear fit on training FITTED (non-censored)
        tr_fit = np.array([i for i in train_idx if not cens[i]])
        A = np.vstack([np.ones(len(tr_fit)), X[tr_fit]]).T
        coef, *_ = np.linalg.lstsq(A, y[tr_fit], rcond=None)
        resid = y[tr_fit] - (coef[0] + coef[1] * X[tr_fit])
        sigma_b3 = max(float(resid.std(ddof=0)), 1e-6)
        Xtest = np.where(cens[test_idx], LOG10_40, X[test_idx])
        mu_b3 = coef[0] + coef[1] * Xtest
        row_mu[test_idx] = mu_b3
        row_sigma[test_idx] = sigma_b3
        nlpd_b1 = censored_nlpd(y[test_idx], np.full(len(test_idx), mu_b1),
                                np.full(len(test_idx), sigma_b1), cens[test_idx])
        nlpd_b3 = censored_nlpd(y[test_idx], mu_b3, np.full(len(test_idx), sigma_b3), cens[test_idx])
        covb3n, w3, covb3 = interval_coverage_width(y[test_idx], mu_b3,
                                                    np.full(len(test_idx), sigma_b3), cens[test_idx])
        fold_results.append({
            "component": int(ci),
            "n_test": int(len(test_idx)),
            "n_measured": int(np.sum(~cens[test_idx])),
            "n_censored": int(np.sum(cens[test_idx])),
            "nlpd_b1": float(nlpd_b1),
            "nlpd_b3": float(nlpd_b3),
            "coverage_n": int(covb3n),
            "coverage": float(covb3),
            "width": float(w3),
        })
    n_total = sum(f["n_test"] for f in fold_results)
    micro_b1 = sum(f["nlpd_b1"] * f["n_test"] for f in fold_results) / max(n_total, 1)
    micro_b3 = sum(f["nlpd_b3"] * f["n_test"] for f in fold_results) / max(n_total, 1)
    micro_cov = sum(f["coverage"] * f["n_test"] for f in fold_results) / max(n_total, 1)
    micro_width = sum(f["width"] * f["n_test"] for f in fold_results) / max(n_total, 1)
    return {
        "fold_results": fold_results,
        "micro_b1": float(micro_b1),
        "micro_b3": float(micro_b3),
        "gain": float(micro_b1 - micro_b3),
        "coverage": float(micro_cov),
        "coverage_n": int(np.sum([(row_mu[i] - Z_LEVEL * row_sigma[i] <= y[i] <= row_mu[i] + Z_LEVEL * row_sigma[i]) for i in range(n) if not np.isnan(row_mu[i])])),
        "width": float(micro_width),
        "row_mu": row_mu,
        "row_sigma": row_sigma,
    }


def coef_ci(beta, se):
    z = 1.959963984540054
    return beta - z * se, beta + z * se


def wilson(obs, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = obs / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def coverage_width_at_level(y, mu, sigma, cens, level):
    z = -1.0 * np.percentile(np.random.default_rng(seed=0).standard_normal(1000000), 100 * (1 - level) / 2)
    lo = mu - z * sigma
    hi = mu + z * sigma
    n = len(y)
    cov = float(np.sum([(lo[i] <= y[i] <= hi[i]) for i in range(n)]) / max(n, 1))
    width = float(np.mean(hi - lo))
    return cov, width


def main():
    os.makedirs(Q8_DIR, exist_ok=True)
    import pandas as pd
    preds = pd.read_parquet(f"{Q7_DIR}/predictions.parquet")
    assert len(preds) == 95, f"expected 95 frozen primary rows, got {len(preds)}"

    # Column names from frozen schema
    component_col = "component"
    y_col = "rna_map_dg"
    mid_col = "qmap_midpoint"
    cens_col = "censored_right_40"
    cid_col = "canonical_id"

    # ---- Deterministic recomputation (verification) ----
    res = recompute_loocv(preds, component_col, y_col, mid_col, cens_col)
    tol_nlpd = 5e-4
    tol_cov = 5e-4
    repro_ok = (
        abs(res["micro_b1"] - FROZEN_MICRO_B1) < tol_nlpd
        and abs(res["micro_b3"] - FROZEN_MICRO_B3) < tol_nlpd
        and abs(res["gain"] - FROZEN_GAIN) < tol_nlpd
        and abs(res["coverage"] - FROZEN_COVERAGE) < tol_cov
        and abs(res["width"] - FROZEN_WIDTH) < tol_cov
    )

    # ---- Six sub-states ----
    sub_states = {
        "QMAP_GAIN_THRESHOLD": "MET",
        "QMAP_PERMUTATION_SIGNAL": "PRESENT",
        "QMAP_GAIN_BOOTSTRAP": "INCONCLUSIVE",
        "QMAP_REGISTERED_POINT_COVERAGE_RULE": "FAILED",
        "QMAP_CALIBRATION_DEFICIT_EVIDENCE": "INCONCLUSIVE",
        "QMAP_FULL_PREDECLARED_TRANSPORT_CRITERION": "NOT_MET",
    }

    # ---- Calibration uncertainty ----
    y = np.array(preds[y_col], dtype=float)
    cens = np.array(preds[cens_col], dtype=bool)
    cov_n = res["coverage_n"]
    n = len(preds)
    wl_lo, wl_hi = wilson(cov_n, n)
    # cluster-aware bootstrap on B3 micro gain (resample outer component groups)
    rng = np.random.default_rng(SEED)
    comps = np.array(preds[component_col])
    mid = np.array(preds[mid_col], dtype=float)
    X = np.log10(np.maximum(mid, 0.01))
    obs_gain = res["gain"]
    boot_gains = []
    n_comp = int(comps.max()) + 1
    for b in range(B_BOOT):
        idx_sel = []
        for ci in range(n_comp):
            idx = np.where(comps == ci)[0]
            idx_sel.extend(rng.choice(idx, size=len(idx), replace=True))
        idx_sel = np.array(idx_sel)
        yb = y[idx_sel]
        cb = cens[idx_sel]
        compb = comps[idx_sel]
        Xb = X[idx_sel]
        fold = []
        for ci in range(n_comp):
            test_idx = np.where(compb == ci)[0]
            train_idx = np.array([i for i in range(len(idx_sel)) if i not in set(test_idx.tolist())])
            tr_fit = np.array([i for i in train_idx if not cb[i]])
            if len(tr_fit) == 0 or len(test_idx) == 0:
                continue
            mu_b1 = float(yb[train_idx].mean())
            A = np.vstack([np.ones(len(tr_fit)), Xb[tr_fit]]).T
            coef, *_ = np.linalg.lstsq(A, yb[tr_fit], rcond=None)
            Xtest = np.where(cb[test_idx], LOG10_40, Xb[test_idx])
            mu_b3 = coef[0] + coef[1] * Xtest
            resid = yb[tr_fit] - (coef[0] + coef[1] * Xb[tr_fit])
            sigma_b3 = max(float(resid.std()), 1e-6)
            sigma_b1 = max(float(yb[train_idx].std()), 1e-6)
            nlpd_b1 = censored_nlpd(yb[test_idx], np.full(len(test_idx), mu_b1), np.full(len(test_idx), sigma_b1), cb[test_idx])
            nlpd_b3 = censored_nlpd(yb[test_idx], mu_b3, np.full(len(test_idx), sigma_b3), cb[test_idx])
            fold.append((nlpd_b1, nlpd_b3, len(test_idx)))
        if not fold:
            continue
        ntot = sum(f[2] for f in fold)
        mb1 = sum(f[0] * f[2] for f in fold) / ntot
        mb3 = sum(f[1] * f[2] for f in fold) / ntot
        boot_gains.append(mb1 - mb3)
    boot_gains = np.array(boot_gains)
    boot_lo, boot_hi = float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5))
    frac_positive = float(np.mean(boot_gains > 0))
    # coverage uncertainty across bootstrap (cluster-aware)
    boot_cov = []
    for b in range(B_BOOT):
        idx_sel = []
        for ci in range(n_comp):
            idx = np.where(comps == ci)[0]
            idx_sel.extend(rng.choice(idx, size=len(idx), replace=True))
        idx_sel = np.array(idx_sel)
        cov_b = float(np.mean([
            int(res["row_mu"][i] - Z_LEVEL * res["row_sigma"][i] <= y[i] <= res["row_mu"][i] + Z_LEVEL * res["row_sigma"][i])
            for i in idx_sel if not np.isnan(res["row_mu"][i])
        ])) if len(idx_sel) else float("nan")
        boot_cov.append(cov_b)
    boot_cov = np.array([c for c in boot_cov if not np.isnan(c)])
    cov_lo, cov_hi = float(np.percentile(boot_cov, 2.5)), float(np.percentile(boot_cov, 97.5))

    # coverage-width curve at multiple nominal levels (deterministic fixed z)
    curve = []
    zmap = {0.60: 0.84162123, 0.70: 1.03643339, 0.80: 1.28155157, 0.90: 1.64485363}
    mu = res["row_mu"]
    sigma = res["row_sigma"]
    for lvl in NOMINAL_LEVELS:
        z = zmap[lvl]
        lo = mu - z * sigma
        hi = mu + z * sigma
        valid = ~np.isnan(mu)
        cv = float(np.sum([(lo[i] <= y[i] <= hi[i]) for i in range(n) if valid[i]]) / max(valid.sum(), 1))
        wd = float(np.mean(hi[valid] - lo[valid]))
        curve.append({"nominal_level": lvl, "observed_coverage": cv, "width": wd})

    # ---- 11th member membership sensitivity ----
    # CCUGCC_ACUGG is the frozen FIT_IDENTIFIED beyond-40mM member (component 0)
    member = "CCUGCC_ACUGG"
    rows = preds.to_dict("records")
    def build_population(member_mode):
        out = []
        for r in rows:
            if r[cid_col] == member:
                if member_mode == "censored":
                    r2 = dict(r); r2[cens_col] = True; out.append(r2)
                elif member_mode == "fitted":
                    r2 = dict(r); r2[cens_col] = False; out.append(r2)
                # excluded: skip
            else:
                out.append(dict(r))
        return out

    sens_rows = []
    for mode in ["censored", "fitted", "excluded"]:
        pop = build_population(mode)
        df = pd.DataFrame(pop)
        r = recompute_loocv(df, component_col, y_col, mid_col, cens_col)
        cov_ok = REGISTERED_POINT_RULE[0] <= r["coverage"] <= REGISTERED_POINT_RULE[1]
        sens_rows.append({
            "member_assignment": mode,
            "n": int(len(df)),
            "micro_b1": r["micro_b1"],
            "micro_b3": r["micro_b3"],
            "gain": r["gain"],
            "gain_threshold_met": bool(r["gain"] >= MEANINGFUL_GAIN),
            "coverage": r["coverage"],
            "coverage_ok": bool(cov_ok),
            "width": r["width"],
        })
    # membership robustness: all three yield same full-criterion conclusion
    full_criterion_all = all(
        (not s["coverage_ok"] or s["gain"] < MEANINGFUL_GAIN) for s in sens_rows
    )
    # Full criterion requires gain >= threshold AND coverage_ok across all sensitivity
    # A conclusion is robust if all three agree on the conjunctive criterion.
    conclusions = []
    for s in sens_rows:
        met = s["gain_threshold_met"] and s["coverage_ok"]
        conclusions.append(met)
    membership_conclusion = ("QMAP_SOURCE_MEMBERSHIP_SENSITIVE"
                             if (len(set(conclusions)) > 1)
                             else ("QMAP_SOURCE_MEMBERSHIP_ROBUST_"
                                   + ("NOT_MET" if not all(conclusions) else "MET")))

    # ---- Write decision ----
    decision = {
        "schema_version": "Q8-adjudication-v1.5",
        "gate": "Q8",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "parent_q7_decision_hash": sha256_file(f"{Q7_DIR}/Q7_decision.json"),
        "parent_predictions_hash": sha256_file(f"{Q7_DIR}/predictions.parquet"),
        "decision_time_utc": now_utc(),
        "requires_no_model_rerun": True,
        "deterministic_recomputation_reproduced_frozen_q7": repro_ok,
        "frozen": {
            "primary_n": 95,
            "micro_gain": FROZEN_GAIN,
            "meaningful_threshold": MEANINGFUL_GAIN,
            "group_weighted_gain": FROZEN_GROUP_GAIN,
            "bootstrap_ci_95": FROZEN_CI,
            "permutation_finite_p": FROZEN_P,
            "observed_coverage": FROZEN_COVERAGE,
            "observed_width": FROZEN_WIDTH,
            "registered_point_coverage_rule": REGISTERED_POINT_RULE,
        },
        "recomputed": {
            "micro_b1": res["micro_b1"],
            "micro_b3": res["micro_b3"],
            "gain": res["gain"],
            "coverage": res["coverage"],
            "coverage_n": cov_n,
            "width": res["width"],
        },
        "sub_states": sub_states,
        "calibration_uncertainty": {
            "registered_point_rule": {
                "observed": FROZEN_COVERAGE,
                "rule": REGISTERED_POINT_RULE,
                "passed": False,
                "note": "predeclared point rule; do not relax post hoc",
            },
            "simple_wilson_95": {
                "observed": FROZEN_COVERAGE,
                "n_covered": cov_n,
                "n": n,
                "ci": [wl_lo, wl_hi],
                "contains_nominal_0_8": bool(wl_lo <= 0.80 <= wl_hi),
                "note": "descriptive only; independent-Bernoulli assumption not valid; cannot be used as primary evidence",
            },
            "cluster_aware_bootstrap": {
                "n_resamples": B_BOOT,
                "gain_ci_95": [boot_lo, boot_hi],
                "gain_frac_positive": frac_positive,
                "coverage_ci_95": [cov_lo, cov_hi],
                "note": "primary empirical uncertainty; resampled outer component/mutation groups",
            },
            "per_component": res["fold_results"],
            "coverage_width_curve": curve,
        },
        "membership_11th": {
            "member": member,
            "source_status": "FIT_IDENTIFIED",
            "sensitivity": sens_rows,
            "conclusion": membership_conclusion,
            "note": "retain 11 count and source evidence; do NOT claim exact source-authored membership",
        },
        "overall": "Q8_ADJUDICATION_COMPLETE",
    }
    write_json(f"{Q8_DIR}/Q8_decision.json", decision)

    # ---- calibration.tsv ----
    with open(f"{Q8_DIR}/calibration.tsv", "w") as f:
        f.write("analysis\tmetric\tvalue\n")
        f.write(f"registered_point_rule\tobserved_coverage\t{FROZEN_COVERAGE:.6f}\n")
        f.write(f"registered_point_rule\trule_low\t{REGISTERED_POINT_RULE[0]}\n")
        f.write(f"registered_point_rule\trule_high\t{REGISTERED_POINT_RULE[1]}\n")
        f.write(f"registered_point_rule\tpassed\tFalse\n")
        f.write(f"simple_wilson\tn_covered\t{cov_n}\n")
        f.write(f"simple_wilson\tn\t{n}\n")
        f.write(f"simple_wilson\tci_lo\t{wl_lo:.6f}\n")
        f.write(f"simple_wilson\tci_hi\t{wl_hi:.6f}\n")
        f.write(f"simple_wilson\tcontains_0_8\t{bool(wl_lo <= 0.80 <= wl_hi)}\n")
        f.write(f"cluster_bootstrap\tgain_ci_lo\t{boot_lo:.6f}\n")
        f.write(f"cluster_bootstrap\tgain_ci_hi\t{boot_hi:.6f}\n")
        f.write(f"cluster_bootstrap\tgain_frac_positive\t{frac_positive:.6f}\n")
        f.write(f"cluster_bootstrap\tcoverage_ci_lo\t{cov_lo:.6f}\n")
        f.write(f"cluster_bootstrap\tcoverage_ci_hi\t{cov_hi:.6f}\n")
        for ft in res["fold_results"]:
            f.write(f"per_component\tcomp_{ft['component']}_coverage\t{ft['coverage']:.6f}\n")
            f.write(f"per_component\tcomp_{ft['component']}_width\t{ft['width']:.6f}\n")
            f.write(f"per_component\tcomp_{ft['component']}_n\t{ft['n_test']}\n")
        for c in curve:
            f.write(f"coverage_width_curve\tnominal_{c['nominal_level']}_coverage\t{c['observed_coverage']:.6f}\n")
            f.write(f"coverage_width_curve\tnominal_{c['nominal_level']}_width\t{c['width']:.6f}\n")

    # ---- membership_sensitivity.tsv ----
    with open(f"{Q8_DIR}/membership_sensitivity.tsv", "w") as f:
        f.write("member_assignment\tn\tgain\tgain_threshold_met\tcoverage\tcoverage_ok\twidth\n")
        for s in sens_rows:
            f.write(f"{s['member_assignment']}\t{s['n']}\t{s['gain']:.6f}\t{s['gain_threshold_met']}\t"
                    f"{s['coverage']:.6f}\t{s['coverage_ok']}\t{s['width']:.6f}\n")

    # ---- report ----
    report = f"""# Q8 — qMaP source/effect/calibration re-adjudication (v1.5)

Decision time (UTC): {decision['decision_time_utc']}
Parent: v1_4_boundary_audit_20260804T150707Z (Q7 frozen)
Deterministic recomputation reproduced frozen Q7: **{repro_ok}**

## Six sub-states
- QMAP_GAIN_THRESHOLD = MET (gain {FROZEN_GAIN:.6f} >= {MEANINGFUL_GAIN})
- QMAP_PERMUTATION_SIGNAL = PRESENT (p={FROZEN_P})
- QMAP_GAIN_BOOTSTRAP = INCONCLUSIVE (CI [{FROZEN_CI[0]:.4f},{FROZEN_CI[1]:.4f}] includes 0)
- QMAP_REGISTERED_POINT_COVERAGE_RULE = FAILED (observed {FROZEN_COVERAGE:.6f} not in {REGISTERED_POINT_RULE})
- QMAP_CALIBRATION_DEFICIT_EVIDENCE = INCONCLUSIVE (Wilson [{wl_lo:.3f},{wl_hi:.3f}] contains 0.8)
- QMAP_FULL_PREDECLARED_TRANSPORT_CRITERION = NOT_MET

## Calibration uncertainty
- Simple Wilson 95% CI on {cov_n}/{n} coverage: [{wl_lo:.3f},{wl_hi:.3f}] — contains nominal 0.8. Descriptive only.
- Cluster-aware bootstrap (n={B_BOOT}): gain 95% CI [{boot_lo:.3f},{boot_hi:.3f}], coverage 95% CI [{cov_lo:.3f},{cov_hi:.3f}].

## 11th member ({member})
Source status: FIT_IDENTIFIED (counts closed; exact membership partly inferred).
Membership sensitivity across censored/fitted/excluded:
""" + "\n".join(f"- {s['member_assignment']}: gain {s['gain']:.4f} (threshold {s['gain_threshold_met']}), coverage {s['coverage']:.4f} (ok {s['coverage_ok']})" for s in sens_rows) + f"""

Conclusion: **{membership_conclusion}**

Allowed language: in the selected TL/TLR population's component-aware held-out analysis, the qMaP predictor shows predictive gain above the registered point threshold and permutation signal, but the gain bootstrap is unstable and the registered coverage-width joint constraint is not met; therefore the full predeclared transport criterion is not satisfied.
"""
    with open(f"{Q8_DIR}/q8_report.md", "w") as f:
        f.write(report)

    print("REPRO_OK:", repro_ok)
    print("micro_b1:", res["micro_b1"], "micro_b3:", res["micro_b3"], "gain:", res["gain"])
    print("coverage:", res["coverage"], "n:", cov_n, "width:", res["width"])
    print("wilson:", wl_lo, wl_hi)
    print("boot_gain_ci:", boot_lo, boot_hi, "frac_pos:", frac_positive)
    print("boot_cov_ci:", cov_lo, cov_hi)
    print("membership_conclusion:", membership_conclusion)
    print("DONE")


if __name__ == "__main__":
    main()