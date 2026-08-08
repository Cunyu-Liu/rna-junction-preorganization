"""Phase 2 effect decomposition (contract Phase 2).

Decomposes the total NLL reduction of the sequence candidate into the marginal
contribution of each factor, using the nested model hierarchy:

  global_censor_intercept  (censor margin / overall location)
  train_only_scaffold      (+ operator/scaffold)
  scaffold_context_hierarchy (+ helix context)
  motif_topology_hierarchy (+ motif / topology structure)
  corrected_v1_31          (+ latent junction sequence functional)

Each layer's marginal gain = NLL(prev) - NLL(this), mean over outer folds.
The sum of marginal gains telescopes to the total gain
  NLL(global) - NLL(corrected_v1_31).
A negative marginal contribution for a layer indicates that adding that factor
did NOT help under the frozen grouped protocol.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = ["global_censor_intercept", "train_only_scaffold", "scaffold_context_hierarchy",
          "motif_topology_hierarchy", "corrected_v1_31"]


def decompose(fold_metrics, axes_spec):
    rows = []
    for axis, n_folds in axes_spec:
        nll = {}
        for model in LAYERS:
            vals = [fold_metrics[(axis, model, f)] for f in range(n_folds)
                    if (axis, model, f) in fold_metrics and fold_metrics[(axis, model, f)] is not None]
            nll[model] = float(np.mean(vals)) if vals else None
        total_gain = (nll["global_censor_intercept"] - nll["corrected_v1_31"]) \
            if all(v is not None for v in [nll["global_censor_intercept"], nll["corrected_v1_31"]]) else None
        marginals = {}
        for i in range(1, len(LAYERS)):
            prev, cur = LAYERS[i - 1], LAYERS[i]
            marginals[cur] = (nll[prev] - nll[cur]) if (nll[prev] is not None and nll[cur] is not None) else None
        rows.append({"axis": axis, **{f"nll_{m}": nll[m] for m in LAYERS},
                     "total_gain": total_gain,
                     "margin_operator": marginals["train_only_scaffold"],
                     "margin_context": marginals["scaffold_context_hierarchy"],
                     "margin_motif_topo": marginals["motif_topology_hierarchy"],
                     "margin_sequence": marginals["corrected_v1_31"]})
    return rows


def write_decomposition(out_dir, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "EffectDecomposition.csv", index=False)
    return out_dir / "EffectDecomposition.csv"
