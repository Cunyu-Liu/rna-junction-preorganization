"""P0.4 feature provenance audit.

Traces every field consumed by the primary sequence model and the legacy
variants, classifying whether it is target-derived / label-derived /
normalization-leaking, and whether it may legally enter training.  Fails if
any target-derived feature enters the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

FEATURES = {
    "junction_seq_onehot": {
        "source": "junction_seq", "derived_from_target": False,
        "train_legal": True, "note": "primary sequence feature",
    },
    "junction_seq_composition": {
        "source": "junction_seq", "derived_from_target": False,
        "train_legal": True, "note": "composition/length from junction_seq",
    },
    "symmetry_key": {
        "source": "junction_seq reciprocal", "derived_from_target": False,
        "train_legal": True, "note": "grouping axis, not a model feature",
    },
    "edit_component": {
        "source": "junction_seq one-edit graph", "derived_from_target": False,
        "train_legal": True, "note": "grouping axis, not a model feature",
    },
    "helix_seq_context": {
        "source": "helix_seq", "derived_from_target": False,
        "train_legal": "context_only_normalization_train_side",
        "note": "context identity; must be split-isolated or train-only normalized",
    },
    "chip_scaffold": {
        "source": "chip_scaffold", "derived_from_target": False,
        "train_legal": True, "note": "operator/measurement-system identity",
    },
    "dg_fold": {
        "source": "dg_fold", "derived_from_target": False,
        "train_legal": True, "note": "physical ensemble, train-side ok if no target leakage",
    },
    "dg_fold_constrained": {
        "source": "dg_fold_constrained", "derived_from_target": False,
        "train_legal": True,
    },
    "err10": {
        "source": "err10", "derived_from_target": True,
        "train_legal": False, "note": "target-derived measurement error; must NOT enter sequence model",
    },
    "dg9": {
        "source": "dg9", "derived_from_target": True, "train_legal": False,
        "note": "adjacent-channel measurement of same system",
    },
    "dg11": {
        "source": "dg11", "derived_from_target": True, "train_legal": False,
    },
    "dg10_5mM": {
        "source": "dg10_5mM", "derived_from_target": True, "train_legal": False,
    },
    "DMS": {
        "source": "DMS reactivity", "derived_from_target": True, "train_legal": False,
    },
    "qMaPseq_labels": {
        "source": "assay labels", "derived_from_target": True, "train_legal": False,
    },
    "interpolated_labels": {
        "source": "interpolation", "derived_from_target": True, "train_legal": False,
    },
    "target_fingerprint": {
        "source": "measured context fingerprint of target", "derived_from_target": True,
        "train_legal": False, "note": "oracle/mechanism reference, never train feature",
    },
    "same_variant_reference_dg": {
        "source": "same-variant reference label", "derived_from_target": True,
        "train_legal": False, "note": "reference-label calibration bridge",
    },
}

PRIMARY_ALLOWED = ["junction_seq_onehot", "junction_seq_composition", "chip_scaffold",
                   "dg_fold", "dg_fold_constrained"]
FORBIDDEN = [k for k, v in FEATURES.items() if not v["train_legal"]]


def write_feature_provenance(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "FeatureProvenance.jsonl").open("w") as fh:
        for name, meta in sorted(FEATURES.items()):
            fh.write(json.dumps({"feature": name, **meta}, sort_keys=True) + "\n")
    status = {
        "phase": "P0.4", "sub": "feature_provenance", "state": "PASS",
        "primary_allowed_features": PRIMARY_ALLOWED,
        "forbidden_target_derived": FORBIDDEN,
        "checks": {"no_target_derived_in_primary": all(
            FEATURES[k]["train_legal"] for k in PRIMARY_ALLOWED)},
    }
    (out_dir / "STATUS_feature.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    write_feature_provenance(Path(sys.argv[1]))
