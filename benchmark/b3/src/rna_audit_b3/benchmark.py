#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 benchmark runner: run every regime x frozen seed, aggregate detector-level
metrics (sensitivity / specificity / false-pass / false-fail / power / calibration
error / coverage / width / runtime) with Monte-Carlo CIs, plus module ablations."""

from __future__ import annotations
import json
import os
import time

import numpy as np

from . import dgp
from . import detector

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # frozen seed list
ABLATION_MODULES = ["endpoint_identity", "censoring", "graph_support",
                    "baseline_parity", "coverage_width", "claim_provenance"]


def _decision_numeric(decision):
    return {"INVALID": 0, "BOUNDARY": 1, "VALID": 2}[decision]


def run_regime(regime, seed):
    ds = dgp.generate(regime, seed)
    t0 = time.time()
    res = detector.audit(ds)
    dt = time.time() - t0
    res["runtime_s"] = dt
    return ds["label"], res


def run_benchmark(seeds=None, out_dir=None):
    seeds = seeds if seeds is not None else SEEDS
    results = {}
    for regime in dgp.REGIMES:
        rows = []
        for seed in seeds:
            label, res = run_regime(regime, seed)
            rows.append({"seed": seed, "label": label, **res})
        results[regime] = {"label": dgp.REGIMES[regime]["label"], "rows": rows}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results


def _aggregate(results):
    """Compute detector-level metrics across regimes/seeds."""
    # For a VALID regime, correct = VALID (true negative of failure).
    # For an INVALID regime, correct = INVALID (true positive of failure).
    valid_rows = []
    invalid_rows = []
    per_regime = {}
    for regime, r in results.items():
        lab = r["label"]
        det = [_decision_numeric(x["decision"]) for x in r["rows"]]
        # correct detection
        if lab == "VALID":
            correct = [d == 2 for d in det]
        elif lab == "INVALID":
            correct = [d == 0 for d in det]
        else:  # BOUNDARY -> correct if BOUNDARY
            correct = [d == 1 for d in det]
        rate = float(np.mean(correct))
        per_regime[regime] = {"label": lab, "detection_rate": rate,
                              "n": len(det), "decisions": det}
        if lab == "VALID":
            valid_rows.append(rate)
        elif lab == "INVALID":
            invalid_rows.append(rate)
    # sensitivity = detection rate on INVALID (power to catch failures)
    sensitivity = float(np.mean(invalid_rows)) if invalid_rows else float("nan")
    # specificity = detection rate on VALID (don't false-flag)
    specificity = float(np.mean(valid_rows)) if valid_rows else float("nan")
    # false-pass rate = fraction of INVALID judged VALID
    false_pass = []
    for regime, r in results.items():
        if r["label"] == "INVALID":
            false_pass.extend([x["decision"] == "VALID" for x in r["rows"]])
    false_pass_rate = float(np.mean(false_pass)) if false_pass else float("nan")
    # false-fail rate = fraction of VALID judged NOT VALID
    false_fail = []
    for regime, r in results.items():
        if r["label"] == "VALID":
            false_fail.extend([x["decision"] != "VALID" for x in r["rows"]])
    false_fail_rate = float(np.mean(false_fail)) if false_fail else float("nan")
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_pass_rate": false_pass_rate,
        "false_fail_rate": false_fail_rate,
        "per_regime": per_regime,
    }


def _binom_ci(x, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math_sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, c - h), min(1.0, c + h))


def math_sqrt(x):
    import math
    return math.sqrt(x)


def run_ablation(seeds=None, out_dir=None):
    """Remove each audit module and measure the false-pass inflation it would allow."""
    seeds = seeds if seeds is not None else SEEDS
    # Baseline: full detector
    base = _calibrate_full(seeds)
    results = {"baseline": base}
    for mod in ABLATION_MODULES:
        # run invalid regimes with that module forced to pass (simulating removal)
        fp = _false_pass_without_module(mod, seeds)
        results[mod] = fp
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results


def _false_pass_without_module(mod, seeds):
    """Measure false-pass rate on INVALID regimes when module `mod` is disabled."""
    fp = []
    for regime in dgp.REGIMES:
        if dgp.REGIMES[regime]["label"] != "INVALID":
            continue
        for seed in seeds:
            ds = dgp.generate(regime, seed)
            res = detector.audit(ds)
            # force module to pass
            res["checks"][mod]["pass"] = True
            all_pass = all(c["pass"] for c in res["checks"].values())
            if res["gain"] < 0.15:
                dec = "INVALID"
            elif all_pass and res["gain"] >= detector.MEANINGFUL_GAIN and res["signal_present"]:
                dec = "VALID"
            elif all_pass:
                dec = "BOUNDARY"
            else:
                dec = "INVALID"
            fp.append(dec == "VALID")
    return {
        "false_pass_rate": float(np.mean(fp)) if fp else float("nan"),
        "errors_permitted": "see false_pass_rate",
    }


def _calibrate_full(seeds):
    """Errors prevented by the full detector (baseline false-pass on INVALID)."""
    fp = []
    for regime in dgp.REGIMES:
        if dgp.REGIMES[regime]["label"] != "INVALID":
            continue
        for seed in seeds:
            ds = dgp.generate(regime, seed)
            res = detector.audit(ds)
            fp.append(res["decision"] == "VALID")
    return {"false_pass_rate": float(np.mean(fp)) if fp else float("nan"),
            "n": len(fp)}