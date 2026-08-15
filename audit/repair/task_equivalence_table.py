"""Task-equivalence table (contract P0.4 / §4.4 prior-art comparator list).

Documents how each prior-art work / internal configuration relates to the
current RNA-junction benchmark, whether it can be ranked directly, and the
honest naming.  This is the submission-required TaskEquivalence.csv.

Sources: strict audit 2026-08-11 §4.4 + internal ModelUniverse.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"

ROWS = [
    # prior-art / comparator, actual task, relation, can_rank_directly, naming
    ("Denny et al., Cell 2018",
     ">1000 junctions, multi-scaffold measured; thermodynamic fingerprints + "
     "known-structure junction ensembles explain assembly energetics",
     "same scientific system and main data source; native fingerprint uses "
     "target-junction measured multi-context info",
     "NO",
     "oracle/mechanism reference; requires strict train-only reconstruction"),
    ("Yesselman et al., PNAS 2019 (RNAMake-DDG)",
     "blind prediction of 1,536 tectoRNA helix variants; sequence/length effects",
     "same platform physical-ensemble prior art; changes helix sequence/length "
     "not current junction task",
     "NO",
     "task-equivalence must be reported; cannot cross-compare RMSE vs NLL"),
    ("current corrected_v1_31 (63-D seq map)",
     "63-D sequence map -> scalar latent placeholder",
     "project internal configuration",
     "internal only",
     "not a published Denny reproduction"),
    ("physical_ensemble_prior (ViennaRNA proxy)",
     "ViennaRNA MFE/ensemble energy/defect/GC/length/mean BPP + linear Tobit",
     "secondary-structure proxy",
     "internal only",
     "cannot be called RNAMake tertiary ensemble"),
    ("frozen RNA-FM",
     "640-D frozen embedding + global censored linear head",
     "representation baseline",
     "NO (head not matched)",
     "head not matched; exposure not closed"),
    ("Geng et al., Cell 2026",
     "TAR sequence -> secondary conformational ensemble -> binding/activity",
     "broad sequence->ensemble->function narrative",
     "NO",
     "adjacent mechanism precedent; task not equivalent"),
    ("trRosettaRNA2, NMI 2026",
     "RNA 3D structure/conformer prediction",
     "adjacent representation/structure capability",
     "NO",
     "cannot be ranked against thermodynamic NLL"),
    ("CHANRG 2026",
     "RNA secondary-structure fair split + OOD benchmark",
     "benchmark methodology reference",
     "NO",
     "task not equivalent"),
    ("r10b nonlinear MLP hybrid (this work)",
     "right-censored Student-t nonlinear head on nuisance+ViennaRNA",
     "this work's method core",
     "YES",
     "nonlinear robust head: +17.45% vs nuisance"),
    ("7-member mixed ensemble (this work)",
     "4x GBDT + 3x t7 MLP family-equal mu ensemble",
     "this work's ensemble",
     "YES",
     "ensemble: +21.94% vs nuisance (frozen 0.7)"),
    ("frozen method (7mem + per-scaf x stratum sigma)",
     "ensemble + LOO per-operator measured/censored sigma calibration",
     "this work's frozen submission method",
     "YES",
     "frozen method: 0.7907, +27.57% vs nuisance"),
]


def main():
    out_csv = f"{R}/TaskEquivalence.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["work", "actual_task", "relation", "can_rank_directly", "honest_naming"])
        w.writerows(ROWS)
    print(f"wrote {out_csv} ({len(ROWS)} rows)")
    for r in ROWS:
        print(f"  {r[0]:40s} direct_rank={r[3]}")


if __name__ == "__main__":
    main()
