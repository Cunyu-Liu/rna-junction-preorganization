"""Precompute ViennaRNA physical features for all unique admitted junction
sequences once, so the physical_ensemble_prior head fits see only train-fold
rows (no leakage) and the folding cost is not repeated per fold.

Writes RUN_ROOT/r05_prior/PhysicalFeatureCache.npz {seqs, feats}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.benchmark.physical_prior import build_physical_cache, _contig
from audit.data.audit_dataset import audit_dataset


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    run_root = Path(cfg["run_root"])
    out = run_root / "r05_prior"
    out.mkdir(parents=True, exist_ok=True)
    _, admitted, *_ = audit_dataset(Path(cfg["canonical_source"]))
    seqs = sorted({_contig(r["junction_seq"]) for r in admitted if _contig(r["junction_seq"])})
    cache = build_physical_cache(seqs)
    keys = sorted(cache.keys())
    feats = np.asarray([cache[k] for k in keys], dtype=float)
    np.savez(out / "PhysicalFeatureCache.npz", seqs=np.asarray(keys, dtype=object),
             feats=feats)
    print(json.dumps({"n_unique_seqs": len(keys), "feat_dim": feats.shape[1],
                      "out": str(out / "PhysicalFeatureCache.npz")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
