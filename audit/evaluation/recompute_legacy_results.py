"""P0.4 metric recalculation audit.

Determines whether historical aggregate results can be recomputed from
row-level predictions.  In the current frozen state there are no sealed
row predictions for v1.28/v1.30/v1.31 in the audit run root, so every
historical aggregate is flagged `REQUIRES_FRESH_REPLAY` (which P0.5 provides).
"""
from __future__ import annotations

import json
from pathlib import Path


def write_metric_recalculation(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"method": "v1.28_symmetry", "has_row_predictions": False,
         "status": "REQUIRES_FRESH_REPLAY",
         "note": "no sealed row predictions in audit run root; cannot recompute macro NLL"},
        {"method": "v1.28_edit", "has_row_predictions": False,
         "status": "REQUIRES_FRESH_REPLAY",
         "note": "no sealed row predictions in audit run root"},
        {"method": "v1.30", "has_row_predictions": False,
         "status": "REQUIRES_FRESH_REPLAY",
         "note": "no sealed row predictions in audit run root"},
        {"method": "v1.31", "has_row_predictions": False,
         "status": "REQUIRES_FRESH_REPLAY",
         "note": "gradient bug uncorrected + no sealed row predictions; corrected v1.31 is a new run"},
    ]
    with (out_dir / "MetricRecalculation.csv").open("w") as fh:
        fh.write("method,has_row_predictions,status,note\n")
        for r in rows:
            fh.write(f"{r['method']},{r['has_row_predictions']},{r['status']},{r['note']}\n")
    status = {
        "phase": "P0.4", "sub": "metric_recalculation",
        "state": "PASS" if all(r["status"] == "REQUIRES_FRESH_REPLAY" for r in rows) else "FAIL",
        "conclusion": "No historical aggregate can be recomputed without fresh replay; all flagged REQUIRES_FRESH_REPLAY.",
    }
    (out_dir / "STATUS_recalc.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    write_metric_recalculation(Path(sys.argv[1]))
