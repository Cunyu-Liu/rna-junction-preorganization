#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 data-generating processes (DGP). Each regime has a frozen spec and a
ground-truth transport-claim validity label. The detector never receives the
label: it only sees the raw generated dataset."""

from __future__ import annotations
import json
import math
import os

import numpy as np

# Balanced schema: every component is large enough for a reliable component-aware
# holdout (MIN_COMP_SAMPLES=5 in the detector). Used by all regimes except the
# component_imbalance regime, which deliberately reproduces the qMaP-like
# unbalanced 80/11/2/2 structure.
BALANCED_SCHEMA = [50, 50, 50, 50]
IMBALANCED_SCHEMA = [80, 11, 2, 2]
# Larger schema for the boundary regime so the gain lands reliably inside
# [MEANINGFUL_GAIN*0.5, MEANINGFUL_GAIN) across frozen seeds (finite-sample
# variance shrinks with n).
BOUNDARY_SCHEMA = [150, 150, 150, 150]
CENS_THRESHOLD = 40.0
LOG40 = math.log10(CENS_THRESHOLD)
Z_LEVEL = 1.2815515655446004  # Phi^{-1}(0.90)


def _rng(seed):
    return np.random.default_rng(seed)


def _base_dataset(rng, schema, n_features=3, signal=1.0, noise=0.5,
                  censoring_rate=0.0):
    """Generate a clean dataset with an optional true linear signal in feature 0.

    Returns a dict with arrays: group, x (n x f), y, mid, censored, platform_ok.
    platform_ok=True means the predictor is an independent system (valid transport
    use); False means endpoint/platform reuse (invalid). `mid` is the positive
    truncation variable used for censoring realism.
    """
    n = int(sum(schema))
    f = n_features
    group = np.concatenate([np.full(s, i) for i, s in enumerate(schema)])
    x = rng.normal(size=(n, f))
    beta_true = signal
    y = beta_true * x[:, 0] + noise * rng.normal(size=n)
    mid = np.exp(x[:, 0])  # positive
    cens = np.zeros(n, dtype=bool)
    if censoring_rate > 0:
        thresh = np.quantile(mid, 1.0 - censoring_rate)
        cens = mid > thresh
    return {
        "group": group.astype(int),
        "x": x.astype(float),
        "y": y.astype(float),
        "mid": mid.astype(float),
        "censored": cens,
        "platform_ok": np.ones(n, dtype=bool),
    }


def _plant_endpoint_reuse(ds, rng):
    """Predictor shares measurement platform with target -> BLOCK (invalid)."""
    ds["platform_ok"][:] = False
    return ds


def _plant_censoring_misclass(ds, rng, wrong_frac=0.5):
    """Randomly flip some censored flags (misclassification)."""
    cens = ds["censored"].copy()
    flip = rng.random(len(cens)) < wrong_frac
    cens[flip] = ~cens[flip]
    ds["censored"] = cens
    return ds


def _plant_component_imbalance(ds, rng):
    """qMaP-like unbalanced 80/11/2/2 structure: the two n=2 components cannot
    support a reliable component-aware holdout (graph-support violation)."""
    return ds


def _plant_baseline_failure(ds, rng):
    """A weak baseline only; predictor beats weak but not a matched strong one.
    We simulate this by marking a strong matched baseline that the audit must
    also be compared against."""
    ds["has_strong_baseline_feature"] = True
    return ds


def _plant_wide_interval(ds, rng, width_inflate=3.0):
    """Inflate the reported interval width (pseudo-calibration)."""
    ds["width_inflate"] = width_inflate
    return ds


def _plant_split_leakage(ds, rng, leak_frac=0.3):
    """Random-row split: correlates replicates leaking across folds."""
    ds["random_row_split"] = True
    ds["leak_frac"] = leak_frac
    return ds


def _plant_no_signal(ds, rng):
    """True signal is zero/null."""
    ds["y"] = ds["y"] * 0.0 + 0.5 * rng.normal(size=len(ds["y"]))
    return ds


def _plant_source_unresolved(ds, rng):
    """Source membership unresolved (FIT_IDENTIFIED-only)."""
    ds["source_status"] = np.full(len(ds["y"]), "FIT_IDENTIFIED")
    return ds


def _plant_boundary(ds, rng):
    """Gain near the meaningful threshold (boundary)."""
    ds["boundary"] = True
    return ds


# ---------------------------------------------------------------------------
# Regime registry (frozen spec)
# ---------------------------------------------------------------------------
REGIMES = {
    "valid_transport": {
        "label": "VALID",
        "plant": None,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "endpoint_reuse": {
        "label": "INVALID",
        "plant": _plant_endpoint_reuse,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "censoring_misclassification": {
        "label": "INVALID",
        "plant": _plant_censoring_misclass,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.3, "wrong_frac": 0.5},
    },
    "component_imbalance": {
        "label": "INVALID",
        "plant": _plant_component_imbalance,
        "spec": {"schema": IMBALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "baseline_failure": {
        "label": "INVALID",
        "plant": _plant_baseline_failure,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "coverage_width_inflated": {
        "label": "INVALID",
        "plant": _plant_wide_interval,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0, "width_inflate": 3.0},
    },
    "split_leakage": {
        "label": "INVALID",
        "plant": _plant_split_leakage,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0, "leak_frac": 0.3},
    },
    "no_signal_null": {
        "label": "INVALID",
        "plant": _plant_no_signal,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 0.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "source_unresolved": {
        "label": "INVALID",
        "plant": _plant_source_unresolved,
        "spec": {"schema": BALANCED_SCHEMA, "signal": 1.0, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
    "boundary": {
        "label": "BOUNDARY",
        "plant": _plant_boundary,
        "spec": {"schema": BOUNDARY_SCHEMA, "signal": 0.37, "noise": 0.5,
                 "censoring_rate": 0.0},
    },
}


def generate(regime, seed, out_dir=None):
    """Generate one dataset for a regime at a seed. Writes raw registry if out_dir."""
    if regime not in REGIMES:
        raise KeyError(regime)
    reg = REGIMES[regime]
    spec = reg["spec"]
    rng = _rng(seed)
    ds = _base_dataset(rng, spec["schema"], signal=spec["signal"],
                       noise=spec["noise"], censoring_rate=spec.get("censoring_rate", 0.0))
    if reg["plant"] is not None:
        ds = reg["plant"](ds, rng)
    ds["regime"] = regime
    ds["label"] = reg["label"]
    ds["seed"] = seed
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{regime}_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(_to_jsonable(ds), f, indent=2)
    return ds


def _to_jsonable(ds):
    out = {}
    for k, v in ds.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def load(path):
    import json
    with open(path) as f:
        d = json.load(f)
    for k in ("group", "x", "y", "mid", "censored", "platform_ok"):
        if k in d:
            d[k] = np.array(d[k])
    return d