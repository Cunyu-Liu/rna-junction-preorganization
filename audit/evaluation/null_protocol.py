"""P0.4 null protocol freeze.

Frozen (pre-registered) null/control rules that must accompany any positive
claim in P0.5/P2.  These are rules, not runs; the runs happen in P0.5 and P2.
"""
from __future__ import annotations

import json
from pathlib import Path

NULL_PROTOCOL = {
    "version": "P0.4-NULL-001",
    "nulls": {
        "label_permutation": {
            "scope": "permute observed labels within each junction/group-preserving unit",
            "must_be_non_positive": True,
            "min_permutations": 1000,
            "rule": "genuine effect must exceed null 97.5th percentile upper bound",
        },
        "sequence_pairing": {
            "scope": "shuffle target-derived basis after pairing (v1.30 sequence-null analog)",
            "must_be_non_positive": True,
            "rule": "if pairing-null gain >= genuine gain, do not claim sequence-specific mechanism",
        },
        "context_null": {
            "scope": "context-only hierarchy / context baseline",
            "must_be_non_positive": True,
            "rule": "candidate must beat nuisance-only context baseline for blocked-context claim",
        },
        "operator_null": {
            "scope": "scaffold/operator identity baseline",
            "must_be_non_positive": True,
            "rule": "candidate must beat operator-null for operator-transfer claim",
        },
    },
    "bootstrap": {
        "method": "split-unit group bootstrap over outer fold units",
        "n_boot": 2000,
        "criterion": "95% CI lower bound > 0 AND 5/5 outer folds positive",
    },
    "catastrophic_fold": {
        "definition": "any outer fold with relative gain < -10% vs strongest eligible baseline",
        "rule": "no pre-registered catastrophic fold",
    },
    "abstention": {
        "rule": "unsupported/unknown operator predictions must be excluded from scoring and reported separately; never scored as mu=0",
    },
    "fresh_replay": {
        "rule": "all P0.5 numbers are fresh replay on frozen splits/metric/seeds; historical results reported in separate old/fresh columns",
    },
}


def write_null_protocol(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "NullProtocol.json").write_text(json.dumps(NULL_PROTOCOL, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print("NullProtocol written")


if __name__ == "__main__":
    import sys
    write_null_protocol(Path(sys.argv[1]))
