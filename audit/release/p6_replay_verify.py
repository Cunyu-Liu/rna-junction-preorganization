"""Phase 6 gap closure: verify clean-checkout fresh replay reproducibility.

Contract Phase 6 acceptance (rna_junction v1.28-v1.31 strict audit, P6 section):
  - same hash/seed/env row predictions verbatim or <= 1e-10
  - cross-environment metric diff <= 1e-8

This script compares:
  A) sealed p4_final outputs  vs  env1 fresh-replay outputs   (same env, clean checkout)
  B) env1 fresh-replay outputs vs  env2 fresh-replay outputs   (cross environment)

against FinalLeaderboard.csv / BootstrapIntervals.csv / FinalPredictions.parquet.
It does not modify sealed results; it only reads and writes a verification report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SAME_ENV_TOL = 1e-10   # same hash/seed/env -> verbatim or <= 1e-10
CROSS_ENV_TOL = 1e-8   # cross-environment metric diff -> <= 1e-8


def load_leaderboard(p: Path):
    d = pd.read_csv(p)
    return {(r["axis"], r["model_id"]): float(r["mean_supported_nll"]) for _, r in d.iterrows()}


def load_boot(p: Path):
    d = pd.read_csv(p)
    return {r["axis"]: float(r["observed_mean_gain"]) for _, r in d.iterrows()}


def load_preds(p: Path):
    d = pd.read_parquet(p)
    out = {}
    for _, r in d.iterrows():
        out[(r["axis"], int(r["fold"]), str(r["source_row_id"]), str(r["model_id"]))] = (
            float(r["mu"]), float(r["sigma"]))
    return out


def max_abs_diff(a: dict, b: dict):
    keys = set(a) & set(b)
    if not keys:
        return None, 0, set(a), set(b)
    d = 0.0
    for k in keys:
        va = np.asarray(a[k], dtype=float)
        vb = np.asarray(b[k], dtype=float)
        d = max(d, float(np.max(np.abs(va - vb))) if va.size else 0.0)
    return d, len(keys), set(a) - keys, set(b) - keys


def cmp(name, a, b, tol):
    mad, n, only_a, only_b = max_abs_diff(a, b)
    if mad is None:
        ok, detail = False, f"no common keys (only_a={len(only_a)}, only_b={len(only_b)})"
    else:
        ok = mad <= tol
        detail = f"n_common={n} max_abs_diff={mad:.3e} (tol {tol:.0e})"
    return {"comparison": name, "max_abs_diff": mad, "n_common": n,
            "only_in_a": len(only_a), "only_in_b": len(only_b),
            "tol": tol, "pass": bool(ok), "detail": detail}


def main(cfg):
    sealed = Path(cfg["sealed_p4_dir"])
    env1 = Path(cfg["env1_dir"])
    env2 = Path(cfg["env2_dir"])
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # A) same-env clean-checkout replay: env1 vs sealed (<= 1e-10)
    results["A_same_env_clean_checkout"] = {
        "FinalLeaderboard": cmp("leaderboard(axis,model)->mean_supported_nll",
                                load_leaderboard(sealed / "FinalLeaderboard.csv"),
                                load_leaderboard(env1 / "FinalLeaderboard.csv"), SAME_ENV_TOL),
        "BootstrapIntervals": cmp("bootstrap(axis)->observed_mean_gain",
                                  load_boot(sealed / "BootstrapIntervals.csv"),
                                  load_boot(env1 / "BootstrapIntervals.csv"), SAME_ENV_TOL),
        "FinalPredictions": cmp("predictions(axis,fold,sid,model)->mu,sigma",
                                load_preds(sealed / "FinalPredictions.parquet"),
                                load_preds(env1 / "FinalPredictions.parquet"), SAME_ENV_TOL),
    }

    # B) cross-environment: env1 vs env2 (<= 1e-8)
    results["B_cross_environment"] = {
        "FinalLeaderboard": cmp("leaderboard(axis,model)->mean_supported_nll",
                                load_leaderboard(env1 / "FinalLeaderboard.csv"),
                                load_leaderboard(env2 / "FinalLeaderboard.csv"), CROSS_ENV_TOL),
        "BootstrapIntervals": cmp("bootstrap(axis)->observed_mean_gain",
                                  load_boot(env1 / "BootstrapIntervals.csv"),
                                  load_boot(env2 / "BootstrapIntervals.csv"), CROSS_ENV_TOL),
        "FinalPredictions": cmp("predictions(axis,fold,sid,model)->mu,sigma",
                                load_preds(env1 / "FinalPredictions.parquet"),
                                load_preds(env2 / "FinalPredictions.parquet"), CROSS_ENV_TOL),
    }

    all_pass = all(c["pass"]
                   for grp in results.values()
                   for c in grp.values())

    report = {
        "phase": "P6",
        "deliverable": "fresh replay verification (contract P6 acceptance)",
        "clean_checkout_commit": cfg.get("clean_checkout_commit"),
        "env1": cfg.get("env1_env"), "env2": cfg.get("env2_env"),
        "same_env_tol": SAME_ENV_TOL, "cross_env_tol": CROSS_ENV_TOL,
        "results": results,
        "overall_pass": bool(all_pass),
        "sota_status": "SOTA_NOT_ADJUDICATED",
        "scientific_claim_authorized": False,
        "note": "verification only; no re-adjudication, sealed results unchanged",
    }

    (out / "FreshReplayVerification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "STATUS.json").write_text(
        json.dumps({"phase": "P6", "deliverable": "FreshReplayVerification",
                    "state": "PASS" if all_pass else "FAIL",
                    "overall_pass": bool(all_pass)}, indent=2) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text())
    main(cfg)
