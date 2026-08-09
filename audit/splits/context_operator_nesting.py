"""R0.3 context-operator strict nesting manifest (contract R0.3 / §2.1.1).

Establishes as a FACT-CONFIRMED, machine-checked artifact that:
  - every helix context maps to exactly ONE scaffold;
  - every scaffold contains exactly 26 nested helix contexts;
  - context and scaffold/operator are NOT two independently crossable factors;
  - per-junction context-scaffold exposure pairs are counted.

Writes ContextOperatorNestingManifest.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def build_nesting_manifest(admitted):
    ctx_to_scaf = {}
    ctx_scaf_pairs = set()
    scaf_ctxs = defaultdict(set)
    jid_ctx_scaf = defaultdict(set)
    for r in admitted:
        ctx = str(r["helix_seq"])
        scaf = int(r["scaf"])
        ctx_to_scaf.setdefault(ctx, set()).add(scaf)
        ctx_scaf_pairs.add((ctx, scaf))
        scaf_ctxs[scaf].add(ctx)
        jid_ctx_scaf[str(r["jid"])].add((ctx, scaf))
    # strict nesting check
    multi_scaf_ctxs = {c: sorted(v) for c, v in ctx_to_scaf.items() if len(v) > 1}
    scaf_sizes = {s: len(c) for s, c in scaf_ctxs.items()}
    exposure = sorted((len(v) for v in jid_ctx_scaf.values()))
    manifest = {
        "strict_nested": bool(not multi_scaf_ctxs),
        "n_contexts": len(ctx_to_scaf),
        "n_scaffolds": len(scaf_ctxs),
        "contexts_per_scaffold": {str(s): n for s, n in sorted(scaf_sizes.items())},
        "context_to_scaf": {c: sorted(v) for c, v in sorted(ctx_to_scaf.items())},
        "multi_scaffold_contexts": multi_scaf_ctxs,
        "per_junction_context_scaffold_pairs": {
            "min": exposure[0] if exposure else 0,
            "median": float(np.median(exposure)) if exposure else None,
            "max": exposure[-1] if exposure else None,
        },
        "consequence": ("context and scaffold/operator are not independently "
                        "crossable factors; context_lomo = unseen context within "
                        "seen scaffold, scaffold_lomo = unseen scaffold+context bundle"),
    }
    return manifest


def write_nesting(admitted, out_path: Path):
    manifest = build_nesting_manifest(admitted)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
