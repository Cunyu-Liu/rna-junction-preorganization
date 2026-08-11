"""ViennaRNA thermodynamic / secondary-structurefeature builder.

Replaces the 63-D position/composition sequence map with folding-proxy
features that directly encode junction preorganization (how strongly and how
uniquely the junction sequence folds).  The latent-operator head is unchanged:
    q_j ~ N(x_j @ theta, sigma_q^2),  x_j = ViennaRNA features.

Features (all computed on the FULL junction sequence, train-only standardized):
    1  total length
    2  MFE free energy / length
    3  ensemble free energy / length
    4  (ensemble - MFE) / length   (fold determinism / energy gap)
    5  mean base-pair probability
    6  max base-pair probability
    7  mean positional entropy
    8  MFE paired-base fraction
    9  GC content
    10 length of part 1
    11 length of part 2

Standardization is fit on TRAIN junction raw features only and applied to both
train and test, consistent with audit.benchmark.features (no test leakage).
"""
from __future__ import annotations

import math

import numpy as np

from audit.data.audit_dataset import parse_parts

# Optional import: ViennaRNA.  If unavailable, raw_features must still be
# importable so the module can fail loudly at fit time rather than import time.
try:
    import RNA as _RNA
except Exception:  # noqa: BLE001
    _RNA = None


def _fold_props(full: str):
    """Return ViennaRNA thermodynamic/structural scalar stats for one sequence."""
    assert _RNA is not None, "ViennaRNA (import RNA) is required for this feature set"
    fc = _RNA.fold_compound(full)
    try:
        struct, mfe = fc.mfe()
    except Exception:  # noqa: BLE001
        struct, mfe = "", float("nan")
    try:
        pf = _RNA.pf_fold(full)
        ens = float(pf[-1]) if len(pf) > 1 else float("nan")
    except Exception:  # noqa: BLE001
        ens = float("nan")
    try:
        fc.pf()
        bpp = fc.bpp()
        n = len(full)
        flat = [bpp[i][j] for i in range(n) for j in range(n) if i < j]
        mean_bpp = float(np.mean(flat)) if flat else 0.0
        max_bpp = float(np.max(flat)) if flat else 0.0
    except Exception:  # noqa: BLE001
        mean_bpp = float("nan")
        max_bpp = float("nan")
    try:
        _, *ents = fc.positional_entropy()
        mean_entropy = float(np.mean([e for e in ents if e is not None and e > 0]))
    except Exception:  # noqa: BLE001
        mean_entropy = float("nan")
    paired = sum(1 for ch in struct if ch in "()")
    paired_frac = paired / max(len(struct), 1)
    return mfe, ens, mean_bpp, max_bpp, mean_entropy, paired_frac


def raw_features(seq: str) -> np.ndarray:
    parts = parse_parts(seq)
    full = "".join(parts)
    den = max(len(full), 1)
    mfe, ens, mean_bpp, max_bpp, mean_entropy, paired_frac = _fold_props(full)
    gc = (full.count("G") + full.count("C")) / den
    features = [
        len(full),
        (mfe / den) if np.isfinite(mfe) else 0.0,
        (ens / den) if np.isfinite(ens) else 0.0,
        ((ens - mfe) / den) if (np.isfinite(ens) and np.isfinite(mfe)) else 0.0,
        mean_bpp if np.isfinite(mean_bpp) else 0.0,
        max_bpp if np.isfinite(max_bpp) else 0.0,
        mean_entropy if np.isfinite(mean_entropy) else 0.0,
        paired_frac,
        gc,
        len(parts[0]) if parts else 0,
        len(parts[1]) if len(parts) > 1 else 0,
    ]
    return np.asarray(features, dtype=float)


def build_raw_by_jid(rows):
    by_jid = {}
    for r in rows:
        by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
    return by_jid


def fit_scaler(jids, by_jid):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mean, sd


def transform(jids, by_jid, mean, sd):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    return (Xraw - mean) / sd