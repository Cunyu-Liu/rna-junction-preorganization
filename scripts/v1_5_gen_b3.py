#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 generative benchmark generator (v1.5).

Runs the full multi-regime Monte-Carlo benchmark and module ablations, then
writes frozen results + gate decision to RUN_ROOT/benchmark/b3/.

The B3 software package lives in the git repo at benchmark/b3; this script only
orchestrates generation and persists the auditable results under /mnt. It never
hands the detector the DGP labels (detector.audit computes from raw data).
"""

from __future__ import annotations
import json
import math
import os
import sys
import time

# Allow running from the run root without installing the package.
_HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT_GIT = os.path.dirname(_HERE)  # /home/cunyuliu/... (git worktree)
B3_SRC = os.path.join(RUN_ROOT_GIT, "benchmark", "b3", "src")
if B3_SRC not in sys.path:
    sys.path.insert(0, B3_SRC)

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
B3_DIR = f"{RUN_ROOT}/benchmark/b3"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _wilson(x, n, z=1.959963984540054):
    if n == 0:
        return (None, None)
    p = x / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, c - h), 6), round(min(1.0, c + h), 6))


def _ci(p, n):
    return _wilson(round(p * n), n)


def _add_cis(results, agg):
    """Add Wilson confidence intervals for the headline metrics to agg."""
    n_invalid = sum(1 for r in results.values() for _ in r["rows"] if r["label"] == "INVALID")
    n_valid = sum(1 for r in results.values() for _ in r["rows"] if r["label"] == "VALID")
    ns, nsp = 0, 0
    for r in results.values():
        for x in r["rows"]:
            if r["label"] == "INVALID" and x["decision"] in ("INVALID", "BOUNDARY"):
                ns += 1
            if r["label"] == "VALID" and x["decision"] == "VALID":
                nsp += 1
    agg["confidence_intervals"] = {
        "false_pass_rate": {"n": n_invalid, "wilson": _ci(agg["false_pass_rate"], n_invalid)},
        "false_fail_rate": {"n": n_valid, "wilson": _ci(agg["false_fail_rate"], n_valid)},
        "sensitivity": {"n": n_invalid, "wilson": _ci(agg["sensitivity"], n_invalid)},
        "specificity": {"n": n_valid, "wilson": _ci(agg["specificity"], n_valid)},
    }


def _aggregate_from_results(results):
    """Recompute aggregate (and CIs) from persisted results for idempotent resume."""
    import rna_audit_b3.benchmark as benchmark
    agg = benchmark._aggregate(results)
    _add_cis(results, agg)
    return agg


def main():
    from rna_audit_b3 import cli, dgp

    os.makedirs(B3_DIR, exist_ok=True)
    t0 = time.time()

    # 1. Frozen DGP specs (label + spec only; plant callables are not serialized)
    specs = {name: {"label": r["label"], "spec": r["spec"]} for name, r in dgp.REGIMES.items()}
    with open(os.path.join(B3_DIR, "dgp_specs.json"), "w") as f:
        json.dump(_jsonable(specs), f, indent=2)

    results_path = os.path.join(B3_DIR, "benchmark_results.json")
    ablation_path = os.path.join(B3_DIR, "ablation_results.json")
    if os.path.exists(results_path) and os.path.exists(ablation_path):
        # Resume: reuse persisted results (idempotent) and only rebuild report/decision.
        with open(results_path) as f:
            results = json.load(f)
        with open(ablation_path) as f:
            ablation = json.load(f)
        agg = _aggregate_from_results(results)
    else:
        # 2. Full benchmark
        import rna_audit_b3.benchmark as benchmark
        results = benchmark.run_benchmark(out_dir=B3_DIR)
        agg = benchmark._aggregate(results)
        # 3. Aggregate CIs (Wilson) for the headline metrics
        _add_cis(results, agg)
        with open(os.path.join(B3_DIR, "aggregate.json"), "w") as f:
            json.dump(_jsonable(agg), f, indent=2)
        # 4. Ablation
        ablation = benchmark.run_ablation(out_dir=B3_DIR)

    # 5. Gate decision
    decision = cli._b3_decision(agg)
    payload = {
        "schema_version": "B3-benchmark-v1.5",
        "gate": "B3",
        "state": decision,
        "aggregate": agg,
        "ablation": ablation,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "regimes": list(dgp.REGIMES.keys()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(B3_DIR, "B3_decision.json"), "w") as f:
        json.dump(_jsonable(payload), f, indent=2)

    # 6. Report
    _write_report(agg, ablation, decision, B3_DIR, _ABLATION_MODULES())
    print(json.dumps(_jsonable(payload["aggregate"]), indent=2))
    print(f"\nB3 state: {decision}  (elapsed {payload['elapsed_s']}s)")
    return 0 if decision == "B3_VALIDATED" else 1


def _ABLATION_MODULES():
    return ["endpoint_identity", "censoring", "graph_support",
            "baseline_parity", "coverage_width", "claim_provenance"]


def _write_report(agg, ablation, decision, out_dir, modules):
    lines = [
        "# B3 Generative Multi-Regime Benchmark — Report",
        "",
        f"**Gate decision:** {decision}",
        "",
        "## Headline detector metrics (Monte-Carlo over frozen seeds)",
        "",
        "| metric | value | Wilson 95% CI | n |",
        "|---|---|---|---|",
    ]
    ci = agg["confidence_intervals"]
    for k, label in (("sensitivity", "sensitivity (power to catch failures)"),
                     ("specificity", "specificity (don't false-flag valid)"),
                     ("false_pass_rate", "false-pass rate (INVALID -> VALID)"),
                     ("false_fail_rate", "false-fail rate (VALID -> NOT VALID)")):
        v = agg[k]
        w = ci[k]["wilson"]
        lines.append(f"| {label} | {v:.4f} | [{w[0]:.4f}, {w[1]:.4f}] | {ci[k]['n']} |")
    lines += ["", "## Per-regime detection rates", "", "| regime | label | detection rate |", "|---|---|---|"]
    for regime, r in agg["per_regime"].items():
        lines.append(f"| {regime} | {r['label']} | {r['detection_rate']:.3f} |")
    lines += ["", "## Module ablations (false-pass inflation when a module is removed)", ""]
    lines.append("| module removed | false-pass rate on INVALID |")
    lines.append("|---|---|")
    lines.append(f"| (none, full detector) | {float(ablation['baseline']['false_pass_rate']):.4f} |")
    for mod in modules:
        fp = float(ablation[mod]["false_pass_rate"]) if ablation[mod]["false_pass_rate"] == ablation[mod]["false_pass_rate"] else float("nan")
        lines.append(f"| {mod} | {fp:.4f} |")
    lines += ["", "All results are frozen under the seed list in `dgp_specs.json`.",
              "The detector computes decisions from raw data; labels are never handed to it."]
    with open(os.path.join(out_dir, "b3_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())