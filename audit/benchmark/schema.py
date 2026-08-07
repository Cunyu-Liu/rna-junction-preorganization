"""P0.5 unified prediction schema (contract P0.5).

Every qualified model must emit one row-level record per test observation with
a stable primary key so all summaries can be recomputed from rows.  The schema
carries: row/group/context/operator/fold/model/seed and the prediction
(mu, scale, interval, censor probability, NLL, support, neighbors, abstention,
runtime, hashes).

The single metric used across every model/axis is the junction-macro
right-censored NLL defined in audit.evaluation.metrics (reused verbatim), so
all versions are ranked under the same likelihood and aggregation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREDICTION_SCHEMA = {
    "version": "p0.5_schema_v1",
    "primary_key": ["axis", "fold", "source_row_id", "model_id"],
    "required_fields": [
        "axis", "fold", "source_row_id", "jid", "scaf", "context",
        "model_id", "seed", "y", "cens", "mu", "sigma",
        "censor_prob", "nll", "support", "abstain",
    ],
    "optional_fields": [
        "neighbors", "runtime_s", "fit_hash", "pred_hash",
    ],
    "value_types": {
        "y": "float (observed dg10)",
        "cens": "bool (right-censored at CAP=-7.1)",
        "mu": "float (predicted location)",
        "sigma": "float (predictive scale, >=0.05)",
        "censor_prob": "float in [0,1] = P(Y>=CAP)",
        "nll": "float (row right-censored NLL)",
        "support": "bool|float (has local/train support)",
        "abstain": "bool (model declines prediction)",
    },
    "metric": "right_censored_macro_junction_nll (audit.evaluation.metrics)",
    "sealed_note": "test folds are frozen by P0.4 SplitManifests; "
                   "no test transform/selection allowed.",
}


def row_key(axis, fold, source_row_id, model_id):
    return f"{axis}|{fold}|{source_row_id}|{model_id}"


def record_hash(rec: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: rec[k] for k in PREDICTION_SCHEMA["required_fields"]},
        sort_keys=True).encode()).hexdigest()[:16]


def write_schema(out_dir: Path):
    (out_dir / "PredictionSchema.json").write_text(
        json.dumps(PREDICTION_SCHEMA, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
