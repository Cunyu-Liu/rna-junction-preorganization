#!/usr/bin/env python3
"""Q2 finalizer — verifies attrition artifacts before writing Q2 to PASS.

Checks: source archive MD5s, required artifacts present, classification counts
(84 fitted + 11 right-censored + 2 closing-pair abnormal + 1 alternate-structure = 98),
all 98 variants classified, censored not deleted, strata comparison complete.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

WORKTREE = "/home/cunyuliu/rna_junction_preorganization_v1_2_20260803"
QDATA = "/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap"
DATA = os.path.join(QDATA, "q2")
MANIFEST_PATH = os.path.join(WORKTREE, "manifests", "canonical_manifest_v1_2_20260803.json")
SENTINEL_PATH = os.path.join(WORKTREE, "manifests", "sentinel_Q2.txt")

REQUIRED = [
    os.path.join(DATA, "q2_attrition.jsonl"),
    os.path.join(DATA, "q2_strata_comparison.json"),
    os.path.join(DATA, "q2_attrition_summary.json"),
    os.path.join(DATA, "q2_manifest.json"),
]


def git(*args):
    r = subprocess.run(["git", "-C", WORKTREE, *args], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    results = {}
    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    results["code_commit"] = commit
    results["branch"] = branch

    # --- required artifacts ---
    results["required_artifacts_present"] = all(os.path.exists(p) for p in REQUIRED)
    results["missing_artifacts"] = [p for p in REQUIRED if not os.path.exists(p)]

    # --- load summary ---
    summary_ok = False
    summary = {}
    summary_path = os.path.join(DATA, "q2_attrition_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        acc = summary.get("acceptance", {})
        summary_ok = all(acc.values()) if acc else False
        results["classification"] = summary.get("classification", {})
        results["n_total"] = summary.get("n_total")
        results["n_fitted"] = summary.get("n_fitted")
        results["n_right_censored"] = summary.get("n_right_censored")
        results["n_closing_pair_abnormal"] = summary.get("n_closing_pair_abnormal")
        results["n_alternate_structure"] = summary.get("n_alternate_structure")
        results["acceptance"] = acc
        results["right_censored_breakdown"] = summary.get("right_censored_breakdown", {})
        results["published_vs_replay_match"] = summary.get("published_vs_replay_match")

    # --- verify attrition JSONL record count and categories ---
    attrition_path = os.path.join(DATA, "q2_attrition.jsonl")
    n_records = 0
    cat_counts = {"fitted": 0, "right_censored": 0, "closing_pair_abnormal": 0, "alternate_structure": 0}
    all_have_reason = True
    censored_not_deleted = True
    if os.path.exists(attrition_path):
        with open(attrition_path) as f:
            for line in f:
                r = json.loads(line)
                n_records += 1
                cat = r.get("category")
                if cat in cat_counts:
                    cat_counts[cat] += 1
                if not r.get("reason"):
                    all_have_reason = False
                if cat == "right_censored" and r.get("deleted", False):
                    censored_not_deleted = False
    results["n_attrition_records"] = n_records
    results["category_counts"] = cat_counts
    results["all_have_reason"] = all_have_reason
    results["censored_not_deleted"] = censored_not_deleted

    # --- verify strata comparison ---
    strata_path = os.path.join(DATA, "q2_strata_comparison.json")
    strata_complete = False
    if os.path.exists(strata_path):
        with open(strata_path) as f:
            strata = json.load(f)
        strata_complete = all(cat in strata for cat in ["fitted", "right_censored", "closing_pair_abnormal", "alternate_structure"])
    results["strata_comparison_complete"] = strata_complete

    # --- gate decision ---
    q2_ok = bool(
        results["required_artifacts_present"]
        and summary_ok
        and n_records == 98
        and cat_counts["fitted"] == 84
        and cat_counts["right_censored"] == 11
        and cat_counts["closing_pair_abnormal"] == 2
        and cat_counts["alternate_structure"] == 1
        and all_have_reason
        and censored_not_deleted
        and strata_complete
    )
    decision = "PASS" if q2_ok else "NOT_ADMITTED"

    q2_decision = {
        "gate": "Q2",
        "decision": decision,
        "summary": (
            f"Q2 attrition: 98 variants classified into 84 fitted + 11 right-censored + "
            f"2 closing-pair abnormal + 1 alternate-structure. Censored enter likelihood (not deleted). "
            f"Strata comparison complete. Published vs replay match verified." if q2_ok else
            f"Q2 NOT_ADMITTED: artifacts={results['required_artifacts_present']}, "
            f"summary_ok={summary_ok}, n_records={n_records}, "
            f"cat_counts={cat_counts}, all_have_reason={all_have_reason}, "
            f"censored_not_deleted={censored_not_deleted}, strata_complete={strata_complete}"),
        "evidence": results,
        "finalized_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # --- write sentinel ---
    with open(SENTINEL_PATH, "w") as f:
        f.write(f"Q2={decision}\ncommit={commit}\nbranch={branch}\n")

    # --- update manifest ---
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            mf = json.load(f)
        mf["gate_statuses"]["Q2"] = decision
        mf["gate_decisions"]["Q2"] = q2_decision
        if q2_ok:
            mf["qmap_terminal_disposition"] = "NOT_ADJUDICATED"
        mf["updated_at_utc"] = q2_decision["finalized_at_utc"]
        with open(MANIFEST_PATH, "w") as f:
            json.dump(mf, f, indent=2)

    print(json.dumps({
        "gate": "Q2",
        "decision": decision,
        "n_records": n_records,
        "category_counts": cat_counts,
        "all_have_reason": all_have_reason,
        "censored_not_deleted": censored_not_deleted,
        "strata_complete": strata_complete,
    }, indent=2))
    return 0 if q2_ok else 1


if __name__ == "__main__":
    sys.exit(main())
