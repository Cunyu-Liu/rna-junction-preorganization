"""R4 Track A evidence closure (contract §12.5, 轨 A).

Given D1 = TRACK_A_LOCKED, R4 must prove the negative/boundary conclusion is
robust under the specified data, model class and power.  This module computes
the computable pieces on the current data (no external measurement needed):

  1. Effective independent N  - junction/context/scaffold/symmetry/edit counts
     + ICC-based design effect (contract: "repeated rows inflate support").
  2. Noise ceiling / operator exposure  - per-junction replication spread
     across its 9 operator contexts (measured rows), and model NLL vs a
     junction-mean baseline.
  3. Censoring/dependence sensitivity - matched ablation contrast on the
     measured-only subset (vs full), to show the negative conclusion is not an
     artifact of censoring implementation.
  4. Power analysis - power to detect a scientifically meaningful relative
     gain (>= 0.10) at the current effective N; reports detectable effect size
     and whether the negative result is a power boundary.
  5. Model-class coverage - which independent baseline families were compared
     under the same protocol, and which strong classes (physical prior /
     frozen LM) are NOT_RUN (so conclusions stay scoped to model class).

Outputs into RUN_ROOT/r4/:
  EffectiveN.json
  NoiseCeiling.json
  CensoringSensitivity.json
  PowerAnalysis.json
  ModelCoverage.json
  STATUS.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.statistics import multiway_cluster as mw
from audit.evaluation.metrics import row_nll

TARGET_REL_GAIN = 0.10
ALPHA = 0.05
BETA = 0.20  # power = 1 - beta = 0.80

AXES = ["symmetry_5fold", "edit_5fold", "context_lomo",
        "scaffold_lomo", "edit_x_nested_context"]
JOINT_AXIS = "edit_x_nested_context"


def utc_now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# 1. effective N
# ---------------------------------------------------------------------------
def effective_n(rows):
    """Effective independent units + ICC design effect at the junction level.

    rows: list of {jid, context, scaf, y, cens}.  Each junction appears in
    several contexts; ICC measures within-junction correlation of y on
    measured rows, giving the design effect 1 + (m-1)*rho for the effective
    number of independent junctions."""
    jids = sorted({r["jid"] for r in rows})
    n_j = len(jids)
    n_ctx = len({r["context"] for r in rows})
    n_scaf = len({r["scaf"] for r in rows})
    n_rows = len(rows)
    m = n_rows / n_j if n_j else 0.0

    # ICC via one-way ANOVA on measured rows grouped by junction
    meas = defaultdict(list)
    for r in rows:
        if not r["cens"]:
            meas[r["jid"]].append(float(r["y"]))
    groups = [v for v in meas.values() if len(v) >= 1]
    icc = None
    de = None
    if len(groups) >= 2:
        k = len(groups)
        nj = [len(g) for g in groups]
        N = sum(nj)
        grand = np.mean([x for g in groups for x in g])
        ss_between = sum(n * (np.mean(g) - grand) ** 2 for n, g in zip(nj, groups))
        ss_within = sum((x - np.mean(g)) ** 2 for g in groups for x in g)
        msb = ss_between / (k - 1)
        msw = ss_within / (N - k)
        n0 = (N - sum(n * n for n in nj) / N) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) else None
        if icc is not None and icc >= 0:
            de = 1.0 + (m - 1.0) * icc
    eff_j = (n_j / de) if de else n_j
    return {
        "n_rows": n_rows, "n_junctions": n_j, "n_contexts": n_ctx,
        "n_scaffolds": n_scaf, "mean_rows_per_junction": round(m, 2),
        "icc_junction": icc, "design_effect": de,
        "effective_junctions": round(eff_j, 1),
    }


# ---------------------------------------------------------------------------
# 2. noise ceiling / operator exposure
# ---------------------------------------------------------------------------
def noise_ceiling(rows):
    """Operator-exposure spread + junction-mean baseline NLL on measured rows.

    A junction measured in several operators shows a spread sigma_op; this is
    the operator-exposure sensitivity, not pure measurement noise (we cannot
    separate measurement noise from operator effect without true replicates).
    We also report the NLL of a perfect junction-mean baseline (predict each
    measured row by its junction mean) as an achievable-performance floor, and
    the pooled-junction-macro NLL of the full model for comparison."""
    meas = defaultdict(list)
    for r in rows:
        if not r["cens"]:
            meas[r["jid"]].append(float(r["y"]))
    n_multi = sum(1 for v in meas.values() if len(v) >= 2)
    within = []
    for v in meas.values():
        if len(v) >= 2:
            within.append(np.var(v, ddof=1))
    op_spread = float(np.sqrt(np.mean(within))) if within else None

    # junction-mean baseline NLL on measured rows (pooled-junction-macro)
    baseline_rows = [r for r in rows if not r["cens"]]
    means = {j: float(np.mean(v)) for j, v in meas.items() if v}
    mu = np.array([means[str(r["jid"])] for r in baseline_rows])
    sigma = np.full(len(baseline_rows), op_spread if op_spread else 0.5)
    nll = row_nll([r["y"] for r in baseline_rows],
                  [r["cens"] for r in baseline_rows], mu, sigma)
    # pooled junction macro
    by_jid = defaultdict(list)
    for r, loss in zip(baseline_rows, nll):
        by_jid[str(r["jid"])].append(float(loss))
    jm = float(np.mean([np.mean(v) for v in by_jid.values()]))
    return {
        "n_measured": len(baseline_rows),
        "n_junctions_with_gt1_measurement": n_multi,
        "operator_exposure_spread_sigma_kcal": op_spread,
        "junction_mean_baseline_pooled_junction_macro_nll": jm,
        "note": ("per-junction spread across its operator contexts is the "
                 "operator-exposure effect plus measurement noise; true "
                 "measurement noise cannot be separated without replicate "
                 "(same junction, same context) measurements"),
    }


# ---------------------------------------------------------------------------
# 3. censoring sensitivity (measured-only matched contrast)
# ---------------------------------------------------------------------------
def censoring_sensitivity(run_root, axis):
    """Recompute the matched ablation contrast on measured-only rows."""
    rows = mw.load_axis_rows(run_root, axis)
    measured = [r for r in rows if not r["cens"]]
    res = {"axis": axis, "n_rows_full": len(rows),
           "n_rows_measured_only": len(measured),
           "n_censored_excluded": len(rows) - len(measured)}
    if not measured:
        res["available"] = False
        res["reason"] = "no measured rows"
        return res
    theta, rel, n_j = mw.axis_statistic(measured)
    res.update({"available": True, "measured_only_theta": theta,
                "measured_only_relative_gain": rel,
                "n_junctions": n_j})
    # compare with full (all-rows) contrast
    theta_full, rel_full, _ = mw.axis_statistic(rows)
    res.update({"full_theta": theta_full, "full_relative_gain": rel_full,
                "negative_conclusion_robust": bool(
                    (rel is None or rel < TARGET_REL_GAIN)
                    and (rel_full is None or rel_full < TARGET_REL_GAIN))})
    return res


# ---------------------------------------------------------------------------
# 4. power analysis
# ---------------------------------------------------------------------------
def power_analysis(rows):
    """Power to detect a relative gain >= TARGET_REL_GAIN at the current
    junction-level variance and effective N.  Uses the per-junction delta
    variance (no_sequence minus full NLL) and the junction-macro design."""
    deltas = mw._junction_deltas(rows)
    nll_ns = mw._junction_nll_ns(rows)
    if not deltas:
        return {"available": False}
    vals = np.array([deltas[j] for j in deltas], dtype=float)
    base = float(np.mean(list(nll_ns.values())))
    n = len(vals)
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("inf")
    target_theta = TARGET_REL_GAIN * base
    z = target_theta / se if se else float("inf")
    # power for one-sided 95% test
    from scipy.stats import norm
    power = float(norm.cdf(z - norm.ppf(1 - ALPHA))) if se else None
    detectable_80 = (norm.ppf(1 - ALPHA) + norm.ppf(1 - BETA)) * se
    observed = float(np.mean(vals))
    return {
        "available": True, "n_junctions": n,
        "per_junction_delta_sd": sd, "se_of_mean": se,
        "target_relative_gain": TARGET_REL_GAIN,
        "target_absolute_theta": target_theta,
        "observed_theta": observed,
        "power_at_target_gain": power,
        "detectable_theta_at_80pct_power": detectable_80,
        "detectable_relative_gain_at_80pct_power": (
            detectable_80 / base if base else None),
        "negative_result_interpretation": (
            "POWER_BOUNDARY" if (power is not None and power < 0.80)
            else "ADEQUATE_POWER_TO_EXCLUDE_TARGET"),
    }


# ---------------------------------------------------------------------------
# 5. model coverage
# ---------------------------------------------------------------------------
def model_coverage(run_root):
    """Which independent model families are compared under the same protocol;
    strong classes not run are flagged so conclusions stay model-class scoped."""
    import pandas as pd
    lb = pd.read_csv(run_root / "r1" / "Leaderboard_v2.csv")
    fam = {}
    for mid, group in lb.groupby("model_id"):
        fam[mid] = {
            "family": mid,
            "n_axis_model_cells": int(group["axis"].nunique()),
            "fully_eligible_axes": int(group[group["eligible_full_coverage"]]["axis"].nunique()),
        }
    classes = {
        "global_censor_intercept": "intercept_only",
        "train_only_scaffold": "scaffold_calibration",
        "scaffold_context_hierarchy": "nested_calibration",
        "motif_topology_hierarchy": "nested_calibration",
        "onehot_kmer_ridge": "sequence_linear",
        "position_aware_additive": "sequence_linear",
        "edit_knn": "edit_distance_knn",
        "corrected_v1_31": "latent_operator_sequence",
        "no_sequence_latent_operator": "latent_operator_no_sequence",
    }
    coverage = []
    for mid in sorted(fam):
        coverage.append({
            "model_id": mid, "class": classes.get(mid, "other"),
            **fam[mid], "independent_method_class": True,
        })
    coverage.append({
        "model_id": "physical_ensemble_prior", "class": "physical_prior",
        "status": "NOT_RUN", "independent_method_class": True,
        "note": "RNAMake/Denny-style ensemble prior not executed (needs tooling/license)"
    })
    coverage.append({
        "model_id": "frozen_rna_lm", "class": "frozen_lm_embedding",
        "status": "NOT_RUN", "independent_method_class": True,
        "note": "frozen RNA foundation model embedding not executed; must use same head/search-budget"
    })
    return coverage


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r4"
    out.mkdir(parents=True, exist_ok=True)
    utc = utc_now()

    # representative full universe (joint axis == 11,893 eligible rows)
    joint_rows = mw.load_axis_rows(run_root, JOINT_AXIS)
    eff = effective_n(joint_rows)
    (out / "EffectiveN.json").write_text(
        json.dumps({**eff, "axis": JOINT_AXIS, "run_id": cfg["run_id"],
                    "generated_at_utc": utc}, indent=2, ensure_ascii=False,
                   sort_keys=True) + "\n")

    nc = noise_ceiling(joint_rows)
    (out / "NoiseCeiling.json").write_text(
        json.dumps({**nc, "axis": JOINT_AXIS, "run_id": cfg["run_id"],
                    "generated_at_utc": utc}, indent=2, ensure_ascii=False,
                   sort_keys=True) + "\n")

    # measured-only contrast needs per-row cens info; load_axis_rows keeps y/cens
    sens = [censoring_sensitivity(run_root, ax) for ax in AXES]
    (out / "CensoringSensitivity.json").write_text(
        json.dumps({"axes": sens, "run_id": cfg["run_id"],
                    "generated_at_utc": utc}, indent=2, ensure_ascii=False,
                   sort_keys=True) + "\n")

    pw = power_analysis(joint_rows)
    (out / "PowerAnalysis.json").write_text(
        json.dumps({**pw, "axis": JOINT_AXIS, "run_id": cfg["run_id"],
                    "generated_at_utc": utc}, indent=2, ensure_ascii=False,
                   sort_keys=True) + "\n")

    cov = model_coverage(run_root)
    (out / "ModelCoverage.json").write_text(
        json.dumps({"families": cov, "run_id": cfg["run_id"],
                    "generated_at_utc": utc}, indent=2, ensure_ascii=False,
                   sort_keys=True) + "\n")

    status = {
        "phase": "R4", "track": "A", "state": "TRACK_A_EVIDENCE_DONE",
        "generated_at_utc": utc,
        "effective_n": eff, "noise_ceiling": nc,
        "censoring_sensitivity_robust": bool(
            all(s.get("negative_conclusion_robust", True) for s in sens if s.get("available"))),
        "power": pw,
        "n_model_families_covered": len(cov),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    run(json.loads(Path(__import__("sys").argv[1]).read_text()))
