"""Typed FoldSpec: the single row-set contract every model must consume.

The strict audit (2026-08-11) found that corrected v1.31 computed joint
`train_ids` but discarded them, while no-sequence/baselines consumed them.
FoldSpec makes the train/test row sets an explicit, hashable object that ALL
runners (full, no-sequence, every baseline) share for a given fold, so a
matched contrast changes only the representation, never the rows.

Fields
------
axis : str
    symmetry_5fold | edit_5fold | context_lomo | scaffold_lomo |
    edit_x_nested_context
fold : str
    stable fold id (e.g. "e:3")
train_ids : frozenset[str]
test_ids : frozenset[str]
blocked_sequence_groups : frozenset[str]
    sequence-family groups (edit components / symmetry keys) withheld from train
blocked_context_groups : frozenset[str]
    nested-context groups (helix_seq) withheld from train
blocked_operator_groups : frozenset[int]
    scaffolds/operators withheld from train
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import FrozenSet, Iterable, Set


@dataclass(frozen=True)
class FoldSpec:
    axis: str
    fold: str
    train_ids: FrozenSet[str] = field(default_factory=frozenset)
    test_ids: FrozenSet[str] = field(default_factory=frozenset)
    blocked_sequence_groups: FrozenSet[str] = field(default_factory=frozenset)
    blocked_context_groups: FrozenSet[str] = field(default_factory=frozenset)
    blocked_operator_groups: FrozenSet[int] = field(default_factory=frozenset)

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        if self.axis not in {
            "symmetry_5fold", "edit_5fold", "context_lomo",
            "scaffold_lomo", "edit_x_nested_context",
        }:
            raise ValueError(f"unknown axis {self.axis!r}")
        if not self.fold:
            raise ValueError("empty fold id")
        overlap = self.train_ids & self.test_ids
        if overlap:
            raise ValueError(
                f"train/test overlap on {self.axis}/{self.fold}: {len(overlap)} rows")
        if not self.test_ids:
            raise ValueError(f"empty test set on {self.axis}/{self.fold}")

    # -- hashing ---------------------------------------------------------
    def rows_hash(self) -> str:
        """Hash of the exact sorted train+test row IDs (the fit contract)."""
        h = hashlib.sha256()
        for rid in sorted(self.train_ids | self.test_ids):
            h.update(rid.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def to_manifest(self) -> dict:
        return {
            "axis": self.axis,
            "fold": self.fold,
            "train_ids": sorted(self.train_ids),
            "test_ids": sorted(self.test_ids),
            "blocked_sequence_groups": sorted(self.blocked_sequence_groups),
            "blocked_context_groups": sorted(self.blocked_context_groups),
            "blocked_operator_groups": sorted(self.blocked_operator_groups),
            "rows_hash": self.rows_hash(),
        }

    def row_count(self) -> int:
        return len(self.train_ids) + len(self.test_ids)


def fold_to_spec(axis: str, fold: str, test_ids: Set[str],
                 train_ids: Set[str], blocked_seq=None,
                 blocked_ctx=None, blocked_op=None) -> FoldSpec:
    return FoldSpec(
        axis=axis, fold=str(fold),
        train_ids=frozenset(str(x) for x in train_ids),
        test_ids=frozenset(str(x) for x in test_ids),
        blocked_sequence_groups=frozenset(str(x) for x in (blocked_seq or set())),
        blocked_context_groups=frozenset(str(x) for x in (blocked_ctx or set())),
        blocked_operator_groups=frozenset(int(x) for x in (blocked_op or set())),
    )


def spec_from_manifest(manifest: dict) -> FoldSpec:
    return FoldSpec(
        axis=manifest["axis"], fold=manifest["fold"],
        train_ids=frozenset(manifest["train_ids"]),
        test_ids=frozenset(manifest["test_ids"]),
        blocked_sequence_groups=frozenset(manifest.get("blocked_sequence_groups", [])),
        blocked_context_groups=frozenset(manifest.get("blocked_context_groups", [])),
        blocked_operator_groups=frozenset(manifest.get("blocked_operator_groups", [])),
    )
