"""Extended ViennaRNA preorganization feature builder for the winning-linear-head
representation expansion.

The 11-D ViennaRNA summary (MFE, ensemble, bpp mean/max, entropy mean, paired
fraction, GC, lengths) restores the sequence increment on the winning plain-linear
head (+2.7%, group-CI > 0).  This module enriches that representation with
folding features that directly encode junction PREORGANIZATION:

  bpp profile (marginal pairing probability per position):
    - mean / max of per-position marginal pairing probability p_i = sum_j bpp[i,j]
    - fraction of positions with p_i > 0.5 and p_i > 0.8 (structured positions)
  cross-part interaction (the two junction parts folding together):
    - mean bpp between part-1 positions and part-2 positions
    - number of cross-part contacts with bpp > 0.5, normalized by length
  MFE structure topology:
    - number of stems in the MFE structure
    - mean stem length
  positional entropy profile:
    - max positional entropy
    - fraction of positions with entropy below a threshold (fold-determined)

All are computed on the FULL concatenated junction sequence (train-only
standardized), consistent with audit.benchmark.vienna_features.
"""
from __future__ import annotations

import numpy as np

from audit.data.audit_dataset import parse_parts

try:
    import RNA as _RNA
except Exception:  # noqa: BLE001
    _RNA = None

# reuse the base 11-D builder so the enriched set is a strict superset
from audit.benchmark.vienna_features import _fold_props  # noqa: E402


def _parse_structure(struct: str):
    """Return (#stems, mean stem length) from a dot-bracket MFE structure."""
    if not struct:
        return 0, 0.0
    stems = 0
    total_paired = 0
    i = 0
    n = len(struct)
    while i < n:
        if struct[i] == "(":
            j = i
            while j < n and struct[j] == "(":
                j += 1
            stems += 1
            total_paired += (j - i)
            i = j
        else:
            i += 1
    return stems, (total_paired / stems) if stems else 0.0


def _marginal_pairing(fc, n):
    """Per-position marginal pairing probability p_i = sum_j bpp[i,j]."""
    try:
        fc.pf()
        bpp = fc.bpp()
        p = np.zeros(n, dtype=float)
        for i in range(n):
            row = bpp[i]
            p[i] = float(np.sum([row[j] for j in range(n) if j != i]))
        return p
    except Exception:  # noqa: BLE001
        return np.zeros(n, dtype=float) + np.nan


def _cross_part_bpp(fc, n, part1_len):
    """Mean bpp and contact count between part-1 and part-2 positions."""
    try:
        fc.pf()
        bpp = fc.bpp()
        vals = []
        for i in range(min(part1_len, n)):
            for j in range(part1_len, n):
                vals.append(float(bpp[i][j]))
        if not vals:
            return 0.0, 0.0
        mean = float(np.mean(vals))
        contacts = float(np.sum([v > 0.5 for v in vals])) / max(len(vals), 1)
        return mean, contacts
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")


def raw_features(seq: str) -> np.ndarray:
    parts = parse_parts(seq)
    full = "".join(parts)
    den = max(len(full), 1)
    n = len(full)
    part1_len = len(parts[0]) if parts else 0

    # base 11-D (order preserved)
    mfe, ens, mean_bpp, max_bpp, mean_entropy, paired_frac = _fold_props(full)
    gc = (full.count("G") + full.count("C")) / den
    base = [
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

    # extended features
    p = _marginal_pairing(_RNA.fold_compound(full), n)
    p = np.asarray([x for x in p if np.isfinite(x)])
    p_mean = float(np.mean(p)) if len(p) else 0.0
    p_max = float(np.max(p)) if len(p) else 0.0
    frac_hi = float(np.sum(p > 0.5)) / max(len(p), 1) if len(p) else 0.0
    frac_vhi = float(np.sum(p > 0.8)) / max(len(p), 1) if len(p) else 0.0

    cr_mean, cr_contacts = _cross_part_bpp(_RNA.fold_compound(full), n, part1_len)
    cr_mean = cr_mean if np.isfinite(cr_mean) else 0.0
    cr_contacts = cr_contacts if np.isfinite(cr_contacts) else 0.0

    try:
        struct, _ = _RNA.fold_compound(full).mfe()
    except Exception:  # noqa: BLE001
        struct = ""
    n_stems, mean_stem = _parse_structure(struct)

    try:
        _, *ents = _RNA.fold_compound(full).positional_entropy()
        ents = [e for e in ents if e is not None and np.isfinite(e)]
        ent_max = float(np.max(ents)) if ents else 0.0
        frac_det = float(np.sum([e < 0.5 for e in ents])) / max(len(ents), 1) if ents else 0.0
    except Exception:  # noqa: BLE001
        ent_max, frac_det = 0.0, 0.0

    extended = [p_mean, p_max, frac_hi, frac_vhi,
                cr_mean, cr_contacts,
                n_stems, mean_stem,
                ent_max, frac_det]
    return np.asarray(base + extended, dtype=float)


def build_raw_by_jid(rows):
    by_jid = {}
    for r in rows:
        by_jid.setdefault(str(r["jid"]), str(r["junction_seq"]))
    return by_jid


def fit_scaler(jids, by_jid):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    Xraw = np.where(np.isfinite(Xraw), Xraw, 0.0)
    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd = np.where((sd > 1e-8) & np.isfinite(sd), sd, 1.0)
    return mean, sd


def transform(jids, by_jid, mean, sd):
    Xraw = np.asarray([raw_features(by_jid[j]) for j in jids], dtype=float)
    Xraw = np.where(np.isfinite(Xraw), Xraw, 0.0)
    return (Xraw - mean) / sd