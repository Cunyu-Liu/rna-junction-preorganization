"""P0.3 null runner: 1000 outer-train sequence-permutation refit null (joint axis).

Implements NullSpec_v3 (outer_train_sequence_permutation_refit):

  For each permutation p (seed-based, deterministic):
    - For each of the 37 edit_x_nested_context folds:
      - On the OUTER TRAIN rows only, permute the junction->sequence pairing
        (each train junction gets a random other junction's sequence features,
        keeping the same y/cens/scaf/context).
      - Refit corrected_v1_31 on the permuted train, score the untouched test.
      - no_sequence_latent_operator has NO sequence features, so its fit is
        permutation-invariant -> fit once per fold, reused across permutations.
    - Compute the SAME axis-level pooled-junction-macro statistic used for
      genuine:  delta_p = macro over junctions of (NLL_no_seq - NLL_v131_perm)
      (positive means v131 better), exactly matching analyze_p05_rerun.contrast.

  Null statistic distribution (1000 deltas) -> null_975_upper, compared against
  genuine under the same aggregation path (CoreHypothesisDecision_v4).

  Parallelization: fold-level parallelism (each worker fits v131 for one
  (perm, fold) pair).  Checkpointed so a crash never loses completed work.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.legacy_adapters import make_v131_adapter
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.metrics import row_nll
from audit.models.no_sequence_latent_operator import make_no_sequence_adapter
from audit.repair.fold_loader import build_joint_edit_context_folds

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"
R29 = f"{R}/r29_p05_rerun"
CFG_SRC = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"
N_PERM = 1000
SEED = 20260816  # frozen seed for reproducibility
N_WORKERS = 32

# Module-level globals set by pool initializer (avoid re-pickling big data).
_FOLDS = None
_NS_FIT = None


def _init_worker(folds_data, ns_fit):
    global _FOLDS, _NS_FIT
    _FOLDS = folds_data
    _NS_FIT = ns_fit


def _permuted_train(train_rows, perm_map):
    """Copy train rows but replace junction-sequence fields with the permuted
    junction's sequence (keeps y/cens/scaf/context structure)."""
    jid_rows = {str(r["jid"]): r for r in train_rows}
    out = []
    for r in train_rows:
        src = jid_rows[perm_map[str(r["jid"])]]
        nr = dict(r)
        nr["junction_seq"] = src["junction_seq"]
        nr["helix_seq"] = src["helix_seq"]
        nr["motif"] = src["motif"]
        out.append(nr)
    return out


def _work_one(args):
    """Fit v131 on (perm, fold) permuted train; return per-fold junction deltas."""
    perm_idx, fold_idx, seed = args
    spec, train_rows, test_rows, ns_mu, ns_sigma = _FOLDS[fold_idx]
    rng = np.random.default_rng(seed + perm_idx * 1009 + fold_idx)
    jids = sorted({str(r["jid"]) for r in train_rows})
    perm_map = {j: jids[rng.integers(len(jids))] for j in jids}
    ptrain = _permuted_train(train_rows, perm_map)
    try:
        v131_fit, v131_pred = make_v131_adapter()
        model = v131_fit(ptrain)
        mu, sigma, _, _, _ = v131_pred(model, test_rows)
    except Exception as e:  # noqa: BLE001  (fail-closed)
        return {"perm": perm_idx, "fold": fold_idx, "error": f"{type(e).__name__}: {e}"}
    jd = defaultdict(list)
    for i, r in enumerate(test_rows):
        nll_v = float(row_nll([r["y"]], [r["cens"]], [mu[i]], [sigma[i]])[0])
        nll_n = float(row_nll([r["y"]], [r["cens"]], [ns_mu[i]], [ns_sigma[i]])[0])
        jd[str(r["jid"])].append(nll_n - nll_v)
    return {"perm": perm_idx, "fold": fold_idx, "jid_deltas": jd}


def main():
    t0 = time.time()
    max_perm = int(sys.argv[1]) if len(sys.argv) > 1 else N_PERM
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else N_WORKERS

    _, admitted, _, *_ = audit_dataset(Path(CFG_SRC))
    rows = {str(r["source_row_id"]): r for r in admitted}
    specs = build_joint_edit_context_folds(admitted)

    print("Pre-loading folds + no_sequence predictions (fit once)...", file=sys.stderr)
    ns_fit, ns_pred = make_no_sequence_adapter()
    folds_data = []
    for spec in specs:
        spec.validate()
        train_rows = [r for sid, r in rows.items() if sid in spec.train_ids]
        test_rows = [r for sid, r in rows.items() if sid in spec.test_ids]
        ns_model = ns_fit(train_rows)
        ns_mu, ns_sigma, _, _, _ = ns_pred(ns_model, test_rows)
        folds_data.append((spec, train_rows, test_rows, ns_mu, ns_sigma))
    print(f"  {len(folds_data)} folds loaded.", file=sys.stderr)

    # Resumable: load completed (perm, fold) results from checkpoint file
    ck_path = Path(R29) / "null_checkpoint.jsonl"
    done = {}
    if ck_path.exists():
        for line in ck_path.open():
            rec = json.loads(line)
            done[(rec["perm"], rec["fold"])] = rec
        print(f"Resuming: {len(done)} (perm,fold) already done.", file=sys.stderr)

    tasks = [(p, f, SEED) for p in range(max_perm) for f in range(len(folds_data))
             if (p, f) not in done]
    print(f"Total tasks: {len(tasks)} (target {max_perm}x{len(folds_data)})", file=sys.stderr)

    ck_fh = ck_path.open("a")
    pool = Pool(processes=workers, initializer=_init_worker,
                initargs=(folds_data, ns_fit))
    n_failed = 0
    try:
        for i, res in enumerate(pool.imap_unordered(_work_one, tasks, chunksize=1)):
            ck_fh.write(json.dumps(res) + "\n")
            ck_fh.flush()
            if "error" in res:
                n_failed += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                done_n = len(done) + i + 1
                total_n = len(done) + len(tasks)
                rate = done_n / el
                eta = (total_n - done_n) / rate
                print(f"  [{done_n}/{total_n}] {n_failed} failed | "
                      f"{el/60:.1f}min | ETA {eta/60:.1f}min", file=sys.stderr)
    finally:
        pool.close()
        pool.join()
        ck_fh.close()

    # Aggregate per-permutation axis-level deltas
    per_perm = defaultdict(lambda: defaultdict(list))
    errors = 0
    for rec in done.values():
        if "error" in rec:
            errors += 1
            continue
        for jid, d in rec["jid_deltas"].items():
            per_perm[rec["perm"]][jid].extend(d)
    for line in ck_path.open():
        if (line.startswith("{")):
            pass  # already included above via done; guard double-counting below

    # Re-read checkpoint fresh (it now contains all results) and aggregate
    per_perm = defaultdict(lambda: defaultdict(list))
    n_complete = 0
    n_err = 0
    for line in ck_path.open():
        rec = json.loads(line)
        if "error" in rec:
            n_err += 1
            continue
        n_complete += 1
        for jid, d in rec["jid_deltas"].items():
            per_perm[rec["perm"]][jid].extend(d)

    deltas = []
    for p in range(max_perm):
        jd = per_perm[p]
        if not jd:
            continue
        deltas.append(float(np.mean([np.mean(v) for v in jd.values()])))
    deltas = np.asarray(deltas)
    null_975 = float(np.percentile(deltas, 97.5)) if len(deltas) else None

    result = {
        "n_permutations_requested": max_perm,
        "n_permutations_complete": len(deltas),
        "n_fold_tasks_complete": n_complete,
        "n_failed_tasks": n_err,
        "n_fold_tasks_total": max_perm * len(folds_data),
        "statistic": "axis_level_pooled_junction_macro_nll_delta",
        "aggregation_fn": "pooled_junction_macro",
        "contrast": "corrected_v1_31 (permuted train) vs no_sequence_latent_operator",
        "positive_delta_means_a_better": True,
        "null_975_upper": null_975,
        "null_mean": float(np.mean(deltas)) if len(deltas) else None,
        "null_std": float(np.std(deltas)) if len(deltas) else None,
        "null_min": float(np.min(deltas)) if len(deltas) else None,
        "null_max": float(np.max(deltas)) if len(deltas) else None,
        "seed": SEED,
        "elapsed_seconds": round(time.time() - t0, 1),
        "all_deltas": [round(float(d), 6) for d in deltas],
    }
    out_path = Path(R29) / "NullArtifact.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nNullArtifact written to {out_path}", file=sys.stderr)
    print(f"  complete={len(deltas)} failed_tasks={n_err}", file=sys.stderr)
    print(f"  null_mean={result['null_mean']} null_975_upper={null_975}", file=sys.stderr)

    # Update CoreHypothesisDecision_v4 (only when full 1000 complete)
    if len(deltas) >= max_perm and max_perm == N_PERM:
        adj_path = Path(R) / "adjudication_v3" / "CoreHypothesisDecision_v4.json"
        adj = json.loads(adj_path.read_text())
        adj["null_975_upper"] = round(null_975, 6)
        adj["null_n_complete"] = len(deltas)
        adj["null_seed"] = SEED
        adj_path.write_text(json.dumps(adj, indent=2, sort_keys=True) + "\n")
        print(f"Updated CoreHypothesisDecision_v4 null_975_upper={null_975:.6f}",
              file=sys.stderr)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min", file=sys.stderr)


if __name__ == "__main__":
    main()
