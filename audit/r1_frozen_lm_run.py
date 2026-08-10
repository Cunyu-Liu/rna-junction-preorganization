"""R1 supplemental: frozen RNA-FM embedding baseline (contract §9.4 / §12.2).

Runs a frozen RNA foundation-model (RNA-FM via multimolecule) embedding as a
strong baseline on the SAME frozen folds as the other R1 models, using a single
low-capacity linear head fit under the SAME right-censored objective and the
SAME inner-search budget.  The foundation model is frozen (no label exposure);
embeddings are computed once per unique junction sequence and cached.

The head is re-fit per fold ONLY on train rows, so there is no leakage from
test embeddings through the head.  Frozen embeddings themselves are
unsupervised (no label involvement), so caching them across folds is safe.

Outputs:
  r05_frozenlm/Predictions.jsonl        frozen_rnafm row predictions
  r05_frozenlm/LeaderboardDraft.csv
  r05_frozenlm/ConvergenceLedger.parquet
  r05_frozenlm/EmbeddingsCache.npz      {contig seq -> 640-d frozen embedding}
  r05_frozenlm/STATUS.json
  r1/Leaderboard_v2.csv                 updated (10 -> 11 models)
  r1/Predictions_v2.jsonl                updated
  r1/ConvergenceLedger.parquet           updated
  r1/STATUS.json                        updated
Requires GPU (RNA-FM forward under frozen weights).  Fails loudly if CUDA is
unavailable (contract: GPU-only validation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audit.benchmark.frozen_lm import embed_sequences, fit_frozen_head, predict_frozen_head
from audit.data.audit_dataset import audit_dataset
from audit.evaluation.scorer_v2 import full_coverage_score, validate_unique_keys
from audit.r05_run import load_splits
from audit.r1_run import build_joint_edit_context_folds

MODEL_ID = "frozen_rnafm_lm"
HF_MODEL = "multimolecule/rnafm-ss"
LEADERBOARD_FIELDS = ["axis", "fold", "model_id", "coverage",
                      "pooled_junction_macro_nll", "eligible_full_coverage",
                      "n_eligible", "n_abstain_no_fallback", "converged", "error"]


def frozen_lm_already_in_leaderboard(leaderboard_fp):
    if not Path(leaderboard_fp).exists():
        return False
    models = set(pd.read_csv(leaderboard_fp)["model_id"].unique())
    return MODEL_ID in models


def run_fold(train_rows, test_rows, embs):
    try:
        model = fit_frozen_head(train_rows, embs)
        mu, sigma, cp, support, abstain = predict_frozen_head(model, test_rows, embs)
        conv = {"success": bool(model["gate"]["success"]),
                "n_iter": int(model["gate"]["n_iter"]),
                "n_bound_hits": int(model["gate"]["n_bound_hits"]),
                "n_nan_inf_params": int(model["gate"]["n_nan_inf_params"]),
                "final_grad_norm": float(model["gate"]["final_grad_norm"])}
        ok = bool(conv["success"])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}, None, None
    preds_by_rowid = {}
    for i, r in enumerate(test_rows):
        preds_by_rowid[str(r["source_row_id"])] = {
            "mu": float(mu[i]), "sigma": float(sigma[i]),
            "abstain": bool(abstain[i]), "support": bool(support[i]),
            "fallback_type": None,
        }
    metric, elig = full_coverage_score(test_rows, preds_by_rowid)
    return {"convergence": conv, "converged": ok, "metric": metric,
            "eligible": elig["eligible"], "elig_reason": elig["reason"]}, preds_by_rowid, conv


def main(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_frozenlm"
    out.mkdir(parents=True, exist_ok=True)
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    rows = {str(r["source_row_id"]): r for r in admitted}
    protocol = Path(cfg["protocol_dir"])
    axes = cfg["axes"]

    leaderboard_fp = run_root / "r1" / "Leaderboard_v2.csv"
    if frozen_lm_already_in_leaderboard(leaderboard_fp):
        print(f"[skip] {MODEL_ID} already present in R1 leaderboard; remove first to re-run.")
        return {"phase": "R1_FROZEN_LM", "state": "SKIPPED_ALREADY_PRESENT"}

    # ---- load frozen foundation model (GPU only) ----
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; frozen RNA-FM validation must run on GPU. "
                           "Stopping with evidence (no CPU silent fallback).")
    device = "cuda"
    print(f"[frozenlm] CUDA ok: {torch.cuda.get_device_name(0)}")
    from multimolecule import RnaFmModel, RnaTokenizer
    tokenizer = RnaTokenizer.from_pretrained(HF_MODEL)
    model = RnaFmModel.from_pretrained(HF_MODEL)
    model = model.to(device).eval()

    # ---- frozen embeddings of all unique junction sequences (leak-free) ----
    all_seqs = {str(r["junction_seq"]) for r in admitted}
    cache_fp = out / "EmbeddingsCache.npz"
    if cache_fp.exists():
        dat = np.load(cache_fp, allow_pickle=True)
        embs = {str(k): v for k, v in zip(dat["seqs"], dat["embeds"])}
        print(f"[frozenlm] loaded cached embeddings: {len(embs)}")
    else:
        embs = embed_sequences(all_seqs, tokenizer, model, device)
        seqs = sorted(embs)
        np.savez(cache_fp, seqs=np.asarray(seqs, dtype=object),
                 embeds=np.stack([embs[s] for s in seqs]))
        print(f"[frozenlm] embedded {len(embs)} unique junction sequences")

    all_preds = []
    leaderboard = []
    conv_rows = []

    def eval_split(axis, folds):
        for fold, test_ids in folds:
            test_rows = [r for sid, r in rows.items() if sid in test_ids]
            train_rows = [r for sid, r in rows.items() if sid not in test_ids]
            res, preds, conv = run_fold(train_rows, test_rows, embs)
            conv_rows.append({"axis": axis, "fold": str(fold), "model_id": MODEL_ID,
                              **({} if conv is None else conv),
                              "error": res.get("error") if res else "fit_error"})
            if res is None or "error" in res:
                leaderboard.append({"axis": axis, "fold": str(fold),
                                    "model_id": MODEL_ID,
                                    "error": res["error"] if res else "fit_error"})
                continue
            if preds:
                for rid, p in preds.items():
                    r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                    all_preds.append({"axis": axis, "fold": str(fold),
                                      "source_row_id": rid, "jid": r["jid"],
                                      "scaf": int(r["scaf"]),
                                      "context": str(r["helix_seq"]),
                                      "model_id": MODEL_ID, "y": r["y"],
                                      "cens": bool(r["cens"]), "mu": p["mu"],
                                      "sigma": p["sigma"], "abstain": p["abstain"]})
            leaderboard.append({"axis": axis, "fold": str(fold), "model_id": MODEL_ID,
                                "coverage": res["metric"]["coverage"],
                                "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                                "eligible_full_coverage": res["eligible"],
                                "n_eligible": res["metric"]["n_eligible"],
                                "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                                "converged": res["converged"]})

    for axis in axes:
        mp = protocol / f"SplitManifest_{axis}.jsonl"
        if not mp.exists():
            print(f"[skip] no manifest {axis}")
            continue
        _, by_fold = load_splits(mp)
        eval_split(axis, sorted(by_fold.items()))

    jf = build_joint_edit_context_folds(admitted)
    for f in jf:
        if not f["feasible"]:
            continue
        test_rows = [r for sid, r in rows.items() if sid in f["test_ids"]]
        train_rows = [r for sid, r in rows.items() if sid in f["train_ids"]]
        res, preds, conv = run_fold(train_rows, test_rows, embs)
        conv_rows.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                          "model_id": MODEL_ID,
                          **({} if conv is None else conv),
                          "error": res.get("error") if res else "fit_error"})
        if res is None or "error" in res:
            leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                "model_id": MODEL_ID,
                                "error": res["error"] if res else "fit_error"})
            continue
        if preds:
            for rid, p in preds.items():
                r = next(x for x in test_rows if str(x["source_row_id"]) == rid)
                all_preds.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                                  "source_row_id": rid, "jid": r["jid"],
                                  "scaf": int(r["scaf"]), "context": str(r["helix_seq"]),
                                  "model_id": MODEL_ID, "y": r["y"], "cens": bool(r["cens"]),
                                  "mu": p["mu"], "sigma": p["sigma"], "abstain": p["abstain"]})
        leaderboard.append({"axis": "edit_x_nested_context", "fold": f["fold"],
                            "model_id": MODEL_ID,
                            "coverage": res["metric"]["coverage"],
                            "pooled_junction_macro_nll": res["metric"]["pooled_junction_macro_nll"],
                            "eligible_full_coverage": res["eligible"],
                            "n_eligible": res["metric"]["n_eligible"],
                            "n_abstain_no_fallback": res["metric"]["n_abstain_no_fallback"],
                            "converged": res["converged"]})

    # ---- r05_frozenlm artifacts ----
    with (out / "Predictions.jsonl").open("w") as fh:
        for rec in all_preds:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out / "LeaderboardDraft.csv").open("w", newline="") as fh:
        w = pd.DataFrame(leaderboard)
        w.to_csv(fh, index=False)
    pd.DataFrame(conv_rows).to_parquet(out / "ConvergenceLedger.parquet")

    # ---- merge into unified R1 artifacts ----
    r1 = run_root / "r1"
    r1_leaderboard = r1 / "Leaderboard_v2.csv"
    r1_preds = r1 / "Predictions_v2.jsonl"
    r1_conv = r1 / "ConvergenceLedger.parquet"

    lb = pd.read_csv(r1_leaderboard) if r1_leaderboard.exists() else pd.DataFrame(columns=LEADERBOARD_FIELDS)
    pd.concat([lb, pd.DataFrame(leaderboard)], ignore_index=True).to_csv(r1_leaderboard, index=False)

    if r1_preds.exists():
        recs = [json.loads(l) for l in r1_preds.read_text().splitlines() if l.strip()]
    else:
        recs = []
    recs.extend(all_preds)
    with r1_preds.open("w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if r1_conv.exists():
        old = pd.read_parquet(r1_conv)
        pd.concat([old, pd.DataFrame(conv_rows)], ignore_index=True).to_parquet(r1_conv)
    else:
        pd.DataFrame(conv_rows).to_parquet(r1_conv)

    all_merged = [json.loads(l) for l in r1_preds.read_text().splitlines() if l.strip()]
    dups = validate_unique_keys(all_merged)

    n_models = int(pd.read_csv(r1_leaderboard)["model_id"].nunique())
    r1_status_fp = r1 / "STATUS.json"
    if r1_status_fp.exists():
        st = json.loads(r1_status_fp.read_text())
        st["n_models"] = n_models
        st["note"] = (st.get("note", "") + " Added frozen_rnafm_lm (R1 supplemental).").strip()
        r1_status_fp.write_text(json.dumps(st, indent=2, ensure_ascii=False) + "\n")

    status = {
        "phase": "R1_FROZEN_LM", "state": "DONE",
        "model_id": MODEL_ID,
        "foundation_model": HF_MODEL,
        "pretraining_exposure": "RNA-FM: 23M+ ncRNA sequences, MLM (Chen et al. 2022, arXiv:2204.00300)",
        "frozen_weights": True,
        "head": "single linear head, same censored objective + L-BFGS-B (maxiter=2000, gtol=1e-8, ridge=1.0)",
        "n_embeddings_cached": len(embs),
        "n_predictions": len(all_preds),
        "n_leaderboard_rows": len(leaderboard),
        "n_r1_models_now": n_models,
        "duplicate_primary_keys": len(dups),
        "axes": axes + ["edit_x_nested_context"],
    }
    (out / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    conv = pd.DataFrame(conv_rows)
    if not conv.empty:
        print(conv.groupby(["axis", "model_id"])["converged"].agg(["count", "sum"]).to_string())
    return status


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))