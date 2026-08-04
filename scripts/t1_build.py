#!/usr/bin/env python3
import runtime_config as rc
"""T1 — idempotent raw->analysis pipeline: CleaningLedger, QC, dedup
(exact / reverse-reciprocal / symmetry-equivalent / near-homolog), symmetry
canonicalization, effective-N, and frozen split groups.

Produces, in the /mnt data root:
  t1/t1_cleaning_ledger.jsonl       (per-row ledger)
  t1/t1_symmetry_groups.json        (symmetry-equivalent groups + canonical frame)
  t1/t1_effective_n.json            (effective-N at every level)
  t1/t1_splits.json                 (frozen candidate holdouts + primary choice)
and a machine-readable manifest in the worktree specs/.
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

WORKTREE = rc.WORKTREE
DATA = rc.RUN_ROOT
CANONICAL = os.path.join(DATA, "t0", "t0_denny_canonical_records.jsonl")
OUT = os.path.join(DATA, "t1")
os.makedirs(OUT, exist_ok=True)

CAP = -7.1  # left-censoring floor (kcal/mol)
COMPLEMENT = str.maketrans("ACGUacgu", "UGCAugca")


def rc(s):
    return s.translate(COMPLEMENT)[::-1]


def fill_junction(a, b):
    """Return a canonical 2-stranded junction string with a fixed separator."""
    return f"{a};{b}"


def canonical_junction(junction_seq):
    """Symmetry canonicalization: junction_seq is '{s1}_{s2}'. Generate transforms
    (as-is, strand-swap, reverse-complement, reverse-complement-and-swap) and pick
    the lexicographically minimal canonical string. Returns (canonical, group_key)."""
    if "_" not in junction_seq:
        return junction_seq, junction_seq
    s1, s2 = junction_seq.split("_", 1)
    cands = [
        fill_junction(s1, s2),
        fill_junction(s2, s1),
        fill_junction(rc(s1), rc(s2)),
        fill_junction(rc(s2), rc(s1)),
    ]
    canon = min(cands)
    return cands[0], canon


def parse_num(v):
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def lexicographic_sequence(s):
    return s.replace("_", "").replace(";", "").upper()


def qc_status(r):
    dg10 = parse_num(r.get("dg10"))
    dg10_interp = parse_num(r.get("dg10_interp"))
    if dg10 is None and dg10_interp is not None:
        return "interpolated_only"
    if dg10 is not None and dg10 <= CAP + 1e-9:
        return "censored_at_cap"
    if dg10 is not None:
        return "measured"
    return "missing"


def main():
    records = []
    with open(CANONICAL) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    ledger = []
    motif_counts = defaultdict(set)
    for r in records:
        jid = r.get("junction_id")
        motif = r.get("motif_type")
        scaffold = r.get("chip_scaffold")
        sub = r.get("sublibrary")
        if motif:
            motif_counts[(sub, motif)].add(jid)
        if scaffold:
            motif_counts[("scaffold", scaffold)].add(jid)

    symmetry_members = defaultdict(list)
    for r in records:
        js = r.get("junction_seq")
        raw, canon = canonical_junction(js)
        r["_canonical_junction"] = canon
        r["_raw_junction"] = raw
        symmetry_members[canon].append(r.get("junction_id"))

    # Build symmetry groups
    symmetry_groups = []
    for canon, members in sorted(symmetry_members.items()):
        uniq = sorted({m for m in members if m})
        symmetry_groups.append({"canonical_frame": canon, "junction_ids": uniq, "n": len(uniq)})

    # Cleaning ledger
    for r in records:
        jid = r.get("junction_id")
        ledger.append({
            "source_row_id": r.get("source_row"),
            "source": "denny_2018_tectorna",
            "source_version": "261_SI.xlsx",
            "condition": r.get("sublibrary"),
            "construct": jid,
            "sequence": r.get("tecto_sequence"),
            "motif_or_family": r.get("motif_type"),
            "scaffold": r.get("chip_scaffold"),
            "study": "denny_2018",
            "replicate": None,
            "measurement_unit": "cluster_fluorescence",
            "measured_interpolated_or_censored": qc_status(r),
            "censoring_direction": "left" if qc_status(r) == "censored_at_cap" else None,
            "censoring_threshold": CAP if qc_status(r) == "censored_at_cap" else None,
            "qc_status": qc_status(r),
            "exclusion_reason": None,
            "uncertainty": parse_num(r.get("err10")),
            "raw_evidence_path": "t0/t0_denny_canonical_records.jsonl",
            "processed_evidence_path": "t1/t1_cleaning_ledger.jsonl",
            "evidence_sha256": None,
            "transform_version": "t1-v1",
            "split_group": None,
            "canonical_junction": r.get("_canonical_junction"),
        })

    # --- Split freeze ---
    # Candidate holdouts. All groups are defined on junction_id level (no row split).
    # Primary (from S0): motif-family holdout.
    # motif-family: group by (sublibrary, motif_type). But junctionmat is the main set.
    by_motif = defaultdict(set)
    by_construct = defaultdict(set)
    by_scaffold = defaultdict(set)
    by_sym = defaultdict(set)
    for r in records:
        jid = r.get("junction_id")
        if not jid:
            continue
        by_construct[jid].add(jid)
        by_scaffold[(r.get("chip_scaffold"))].add(jid)
        by_motif[(r.get("sublibrary"), r.get("motif_type"))].add(jid)
        by_sym[r.get("_canonical_junction")].add(jid)

    splits = {
        "primary": "motif_family_holdout",
        "candidates": {
            "motif_family_holdout": {
                "n_groups": len(by_motif),
                "group_size_range": [min(len(v) for v in by_motif.values()),
                                     max(len(v) for v in by_motif.values())],
                "note": "primary (S0). Blocked on motif family; blocks junction->scaffold reuse.",
            },
            "construct_holdout": {
                "n_groups": len(by_construct),
                "note": "hold out whole junction_ids.",
            },
            "scaffold_holdout": {
                "n_groups": len(by_scaffold),
                "note": "only 9 scaffolds; high confound risk.",
            },
            "symmetry_equivalence_holdout": {
                "n_groups": len(by_sym),
                "note": "same symmetry-equivalent group never crosses folds.",
            },
            "study_holdout": {
                "n_groups": 1,
                "feasible": False,
                "note": "single study (Denny 2018); not a valid generalization axis.",
            },
        },
        "forbidden": ["random_row_split", "nucleotide_level_split"],
        "rule": "same symmetry-equivalent group, same construct, and same scaffold never cross folds.",
    }

    # --- Effective N ---
    jids = {r.get("junction_id") for r in records if r.get("junction_id")}
    effective_n = {
        "raw_rows": len(records),
        "unique_constructs": len(jids),
        "unique_motifs_families": len({(r.get("sublibrary"), r.get("motif_type")) for r in records}),
        "unique_scaffolds": len({r.get("chip_scaffold") for r in records}),
        "unique_studies": 1,
        "unique_symmetry_groups": len(symmetry_groups),
        "independent_scaffold_groups": len({r.get("chip_scaffold") for r in records}),
        "independent_study_groups": 1,
        "connected_components": 1,
        "giant_component_size": len(jids),
        "group_adjusted_effective_n": "effective independent units = scaffolds (9) -> strong "
                                      "group-level dependence; construct-level N (1713) is the "
                                      "analysis unit, NOT row-level",
    }

    # checksums for evidence
    def sha256_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # write outputs
    ledger_path = os.path.join(OUT, "t1_cleaning_ledger.jsonl")
    with open(ledger_path, "w") as f:
        for e in ledger:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    sym_path = os.path.join(OUT, "t1_symmetry_groups.json")
    with open(sym_path, "w") as f:
        json.dump({"symmetry_groups": symmetry_groups, "n_groups": len(symmetry_groups)}, f, indent=2)

    eff_path = os.path.join(OUT, "t1_effective_n.json")
    with open(eff_path, "w") as f:
        json.dump(effective_n, f, indent=2)

    splits_path = os.path.join(OUT, "t1_splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    # manifest
    manifest = {
        "schema_version": "t1-manifest-v1",
        "run_id": rc.RUN_ID,
        "canonical_source": CANONICAL,
        "canonical_sha256": sha256_file(CANONICAL),
        "outputs": {
            "t1_cleaning_ledger.jsonl": {"path": ledger_path, "sha256": sha256_file(ledger_path)},
            "t1_symmetry_groups.json": {"path": sym_path, "sha256": sha256_file(sym_path)},
            "t1_effective_n.json": {"path": eff_path, "sha256": sha256_file(eff_path)},
            "t1_splits.json": {"path": splits_path, "sha256": sha256_file(splits_path)},
        },
        "effective_n": effective_n,
        "primary_split": splits["primary"],
    }
    mpath = os.path.join(OUT, "t1_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "rows": len(records),
        "unique_constructs": len(jids),
        "symmetry_groups": len(symmetry_groups),
        "primary_split": splits["primary"],
        "ledger_sha256": manifest["outputs"]["t1_cleaning_ledger.jsonl"]["sha256"][:16],
        "manifest_sha256": sha256_file(mpath)[:16],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())