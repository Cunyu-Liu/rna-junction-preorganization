#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 benchmark CLI: from raw registry/spec/evidence hashes to audit decision,
claim card, and machine-readable report in one run. No project absolute paths
are hard-coded (paths come from CLI args / env)."""

from __future__ import annotations
import argparse
import json
import os
import sys

from . import benchmark, dgp


def cmd_run(args):
    results = benchmark.run_benchmark(seeds=args.seeds, out_dir=args.out)
    agg = benchmark._aggregate(results)
    decision = _b3_decision(agg)
    payload = {
        "schema_version": "B3-benchmark-v1.5",
        "state": decision,
        "aggregate": agg,
        "seeds": args.seeds,
        "regimes": list(dgp.REGIMES.keys()),
    }
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "B3_decision.json"), "w") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_ablate(args):
    res = benchmark.run_ablation(seeds=args.seeds, out_dir=args.out)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "ablation_summary.json"), "w") as f:
            json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


def _b3_decision(agg):
    """B3 states: B3_VALIDATED / B3_PARTIAL_REQUIRES_DOWNGRADE / B3_FAILED_STOP_METHODS_CLAIM.
    PASS thresholds are frozen at N1 (here: false-pass <= 0.10 and false-fail <= 0.10)."""
    fp = agg["false_pass_rate"]
    ff = agg["false_fail_rate"]
    if fp is None or ff is None or math_isnan(fp) or math_isnan(ff):
        return "B3_FAILED_STOP_METHODS_CLAIM"
    if fp <= 0.10 and ff <= 0.10:
        return "B3_VALIDATED"
    if fp <= 0.25 and ff <= 0.25:
        return "B3_PARTIAL_REQUIRES_DOWNGRADE"
    return "B3_FAILED_STOP_METHODS_CLAIM"


def math_isnan(v):
    try:
        return v != v
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser(prog="rna-audit-b3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full benchmark")
    r.add_argument("--out", default=None, help="output directory")
    r.add_argument("--seeds", nargs="*", type=int, default=benchmark.SEEDS)
    a = sub.add_parser("ablate", help="run module ablations")
    a.add_argument("--out", default=None)
    a.add_argument("--seeds", nargs="*", type=int, default=benchmark.SEEDS)
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "ablate":
        return cmd_ablate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())