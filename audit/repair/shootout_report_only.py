"""Report-only aggregator for the representation shootout (reproducibility + recovery).

Recomputes the authoritative ``ShootoutReport.json`` (and ``STATUS.json``) for a
shootout run from its ALREADY-MATERIALIZED per-fold predictions
(``<out_subdir>/Predictions_v3.jsonl``) plus the canonical admitted records.

Why this exists:
  * The full runner (``shootout_run.py``) both FITS and PREDICTS every model on
    every fold, which is expensive (hours for the RNA-FM hybrid set) and is still
    in flight for r09.  This script lets us aggregate / re-aggregate the report
    from finished predictions without re-fitting, so a partial or completed run
    can always yield a reproducible report.
  * It also serves as a reproducibility check: re-running it on an existing run
    must reproduce the stored report bit-for-bit.

The report fields and contrast semantics are identical to ``shootout_run.py``;
we import the exact helper functions from it so there is a single source of
truth for the estimand.

Usage (from repo root, mirroring the runner invocation):

    python audit/repair/shootout_report_only.py <cfg.json>

where ``<cfg.json>`` carries ``run_root``, ``canonical_source``, ``out_subdir``
(and optionally ``out_report``/``out_status`` to override output paths).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.data.audit_dataset import audit_dataset
from audit.repair.shootout_run import (
    _edit_cluster_ci,
    _pooled_contrast,
    _pooled_nll_by_model,
)


def _load_preds(out_dir: Path):
    preds = []
    with (out_dir / "Predictions_v3.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def build_report(all_preds, admitted):
    pooled = _pooled_nll_by_model(all_preds)

    vienna_vs_noseq = _pooled_contrast(
        all_preds, "vienna_latent_operator", "no_sequence_latent_operator",
        "vienna_latent_operator", "no_sequence_latent_operator",
        "pooled-OOF junction-macro NLL delta (no_sequence - vienna_latent_operator)")
    vienna_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_latent_operator", "no_sequence_latent_operator")
    vienna_vs_63d = _pooled_contrast(
        all_preds, "vienna_latent_operator", "corrected_v1_31",
        "vienna_latent_operator", "corrected_v1_31(63D)",
        "pooled-OOF junction-macro NLL delta (corrected_v1_31 - vienna_latent_operator)")
    kmer_vs_63d = _pooled_contrast(
        all_preds, "kmer_latent_operator", "corrected_v1_31",
        "kmer_latent_operator", "corrected_v1_31(63D)",
        "pooled-OOF junction-macro NLL delta (corrected_v1_31 - kmer_latent_operator)")
    kmer_vs_noseq = _pooled_contrast(
        all_preds, "kmer_latent_operator", "no_sequence_latent_operator",
        "kmer_latent_operator", "no_sequence_latent_operator",
        "pooled-OOF junction-macro NLL delta (no_sequence - kmer_latent_operator)")
    kmer_cluster = _edit_cluster_ci(
        all_preds, admitted, "kmer_latent_operator", "no_sequence_latent_operator")
    vienna_vs_scaffold = _pooled_contrast(
        all_preds, "vienna_latent_operator", "train_only_scaffold",
        "vienna_latent_operator", "train_only_scaffold",
        "pooled-OOF junction-macro NLL delta (train_only_scaffold - vienna_latent_operator)")

    hybrid_vs_nuisance = _pooled_contrast(
        all_preds, "vienna_linear_hybrid", "motif_topology_hierarchy",
        "vienna_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - vienna_linear_hybrid)")
    hybrid_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_linear_hybrid", "motif_topology_hierarchy")

    ext_vs_base = _pooled_contrast(
        all_preds, "vienna_extended_linear_hybrid", "vienna_linear_hybrid",
        "vienna_extended_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - vienna_extended_linear_hybrid)")
    ext_vs_nuisance = _pooled_contrast(
        all_preds, "vienna_extended_linear_hybrid", "motif_topology_hierarchy",
        "vienna_extended_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - vienna_extended_linear_hybrid)")
    ext_cluster = _edit_cluster_ci(
        all_preds, admitted, "vienna_extended_linear_hybrid", "motif_topology_hierarchy")

    rnafm_vs_nuisance = _pooled_contrast(
        all_preds, "rnafm_linear_hybrid", "motif_topology_hierarchy",
        "rnafm_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - rnafm_linear_hybrid)")
    rnafm_vs_vienna = _pooled_contrast(
        all_preds, "rnafm_linear_hybrid", "vienna_linear_hybrid",
        "rnafm_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - rnafm_linear_hybrid)")
    rnafm_cluster = _edit_cluster_ci(
        all_preds, admitted, "rnafm_linear_hybrid", "motif_topology_hierarchy")
    rnafmvienna_vs_vienna = _pooled_contrast(
        all_preds, "rnafm_vienna_linear_hybrid", "vienna_linear_hybrid",
        "rnafm_vienna_linear_hybrid", "vienna_linear_hybrid",
        "pooled-OOF junction-macro NLL delta (vienna_linear_hybrid - rnafm_vienna_linear_hybrid)")
    rnafmvienna_vs_nuisance = _pooled_contrast(
        all_preds, "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy",
        "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy",
        "pooled-OOF junction-macro NLL delta (motif_topology_hierarchy - rnafm_vienna_linear_hybrid)")
    rnafmvienna_cluster = _edit_cluster_ci(
        all_preds, admitted, "rnafm_vienna_linear_hybrid", "motif_topology_hierarchy")

    return {
        "axis": "edit_x_nested_context",
        "purpose": "REPRESENTATION_SHOOTOUT",
        "pooled_junction_macro_nll": {k: round(v, 5) for k, v in sorted(pooled.items(), key=lambda kv: kv[1])},
        "vienna_vs_no_sequence": {
            "note": "positive delta = vienna_latent_operator is BETTER than matched no-sequence.",
            "pooled": vienna_vs_noseq, "edit_cluster": vienna_cluster, "gate_10pct": 0.10,
        },
        "vienna_vs_63d": {
            "note": "positive delta = vienna_latent_operator is BETTER than corrected_v1_31 (63-D).",
            "pooled": vienna_vs_63d,
        },
        "kmer_vs_63d": {
            "note": "positive delta = kmer_latent_operator is BETTER than corrected_v1_31 (63-D).",
            "pooled": kmer_vs_63d,
        },
        "kmer_vs_no_sequence": {
            "note": "positive delta = kmer_latent_operator is BETTER than matched no-sequence.",
            "pooled": kmer_vs_noseq, "edit_cluster": kmer_cluster, "gate_10pct": 0.10,
        },
        "vienna_vs_scaffold": {
            "note": "positive delta = vienna_latent_operator is BETTER than train_only_scaffold.",
            "pooled": vienna_vs_scaffold,
        },
        "hybrid_vs_nuisance": {
            "note": ("positive delta = vienna_linear_hybrid (winning plain-linear head "
                     "+ ViennaRNA sequence block) is BETTER than motif_topology_hierarchy "
                     "(same head, nuisance-only). Decisive test of whether a sequence "
                     "representation adds increment over the strongest simple model."),
            "pooled": hybrid_vs_nuisance, "edit_cluster": hybrid_cluster, "gate_10pct": 0.10,
        },
        "extended_vs_nuisance": {
            "note": ("positive delta = vienna_extended_linear_hybrid (winning head + "
                     "21-D extended ViennaRNA) is BETTER than motif_topology_hierarchy."),
            "pooled": ext_vs_nuisance, "edit_cluster": ext_cluster, "gate_10pct": 0.10,
        },
        "extended_vs_base": {
            "note": ("positive delta = vienna_extended_linear_hybrid is BETTER than "
                     "vienna_linear_hybrid (11-D). Tests if richer folding features "
                     "push the sequence increment further."),
            "pooled": ext_vs_base,
        },
        "rnafm_vs_nuisance": {
            "note": ("positive delta = rnafm_linear_hybrid (winning head + RNA-FM "
                     "frozen 1920-D) is BETTER than motif_topology_hierarchy."),
            "pooled": rnafm_vs_nuisance, "edit_cluster": rnafm_cluster, "gate_10pct": 0.10,
        },
        "rnafm_vs_vienna": {
            "note": ("positive delta = rnafm_linear_hybrid is BETTER than "
                     "vienna_linear_hybrid (11-D folding proxy). Tests whether the "
                     "learned representation beats the folding proxy."),
            "pooled": rnafm_vs_vienna,
        },
        "rnafm_vienna_vs_vienna": {
            "note": ("positive delta = rnafm_vienna_linear_hybrid (winning head + "
                     "RNA-FM + ViennaRNA) is BETTER than vienna_linear_hybrid. Tests "
                     "whether the learned representation is complementary to the "
                     "folding proxy."),
            "pooled": rnafmvienna_vs_vienna,
        },
        "rnafm_vienna_vs_nuisance": {
            "note": ("positive delta = rnafm_vienna_linear_hybrid is BETTER than "
                     "motif_topology_hierarchy."),
            "pooled": rnafmvienna_vs_nuisance, "edit_cluster": rnafmvienna_cluster, "gate_10pct": 0.10,
        },
    }


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    run_root = Path(cfg["run_root"])
    out_dir = run_root / cfg["out_subdir"]
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))

    all_preds = _load_preds(out_dir)
    if not all_preds:
        raise SystemExit(f"no predictions found in {out_dir}/Predictions_v3.jsonl")

    report = build_report(all_preds, admitted)
    out_report = Path(cfg.get("out_report", out_dir / "ShootoutReport.json"))
    out_status = Path(cfg.get("out_status", out_dir / "STATUS.json"))

    n_folds = len({p["fold"] for p in all_preds})
    n_models = len({p["model_id"] for p in all_preds})
    status = {
        "phase": "REPRESENTATION_SHOOTOUT", "state": "DONE_REPORT_ONLY",
        "n_models": n_models, "n_folds": n_folds, "n_predictions": len(all_preds),
        "models": sorted({p["model_id"] for p in all_preds}),
        "note": ("Report aggregated from already-materialized predictions "
                 "(no re-fitting). Field semantics identical to shootout_run.py."),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    out_status.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    main(sys.argv[1])
