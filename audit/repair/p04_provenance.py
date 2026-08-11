"""P0.4 baseline equivalence + provenance (strict audit 2026-08-11).

The strict audit found three baseline-equivalence defects:

1. **Proxy misnaming** -- ``denny_train_only``/``physical_ensemble_prior``/
   ``frozen_rnafm_lm`` were treated as faithful prior-art baselines.  They are
   proxies with NON-matched heads.  P0.4 renames them honestly and records
   their input/head/external-data/pretraining-exposure/parameter/budget/support.
2. **No task-equivalence table** -- Denny(2018)/Yesselman(2019)/RNAMake etc.
   cannot be ranked against the current NLL without a TaskEquivalence table.
3. **No source provenance** -- the canonical source URL/payload-hash/acquisition
   chain and license disposition are not machine-readable.

Outputs (into the repair run root):
  TaskEquivalence.csv         -- prior-art <-> current-task equivalence
  ExposureRegistry.csv        -- per-config input/head/external/pretraining/budget
  ModelCardRegistry.jsonl     -- per-model revision/weights/tokenizer/cache hashes
  SourceProvenance.json       -- canonical source acquisition chain
  LicenseDecision.md          -- license/disposition decision record
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

TASK_EQUIVALENCE = [
    {"prior_art": "Denny et al., Cell 2018",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6053692/",
     "task": ">1000 junctions, multi-scaffold; thermodynamic fingerprints + "
             "stand-in ensembles for assembly energetics",
     "relation": "same scientific system and primary data source; native "
                 "fingerprint includes target-junction measured multi-context",
     "rankable": "NO",
     "note": "oracle/mechanism reference unless strict train-only reconstruction"},
    {"prior_art": "Yesselman et al., PNAS 2019",
     "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6708322/",
     "task": "RNAMake-deltaG blind prediction of 1536 tectoRNA helix variants; "
             "sequence/length effects",
     "relation": "same-platform physical-ensemble prior art; mainly helix "
                 "sequence/length change, not the current junction task",
     "rankable": "NO",
     "note": "RMSE (helix) not comparable to current NLL (junction)"},
    {"prior_art": "RNAMake / Denny-inspired current proxy",
     "url": "NA",
     "task": "63-D sequence -> scalar latent placeholder",
     "relation": "project-internal proxy",
     "rankable": "NO",
     "note": "internal configuration, not a published Denny reproduction"},
    {"prior_art": "ViennaRNA secondary-ensemble proxy",
     "url": "NA",
     "task": "MFE/partition/defect/GC/length/mean-BPP + linear Tobit",
     "relation": "secondary-structure proxy",
     "rankable": "NO",
     "note": "not an RNAMake tertiary ensemble"},
    {"prior_art": "frozen RNA-FM global head",
     "url": "NA",
     "task": "640-D frozen embedding + global censored linear head",
     "relation": "representation baseline",
     "rankable": "NO",
     "note": "head not matched to latent-operator; pretraining exposure open"},
    {"prior_art": "Geng et al., Cell 2026",
     "url": "https://pubmed.ncbi.nlm.nih.gov/41856113/",
     "task": "TAR sequence -> secondary conformational ensemble -> binding/activity",
     "relation": "broader sequence->ensemble->function narrative",
     "rankable": "NO",
     "note": "adjacent mechanism precedent, task not equivalent"},
    {"prior_art": "trRosettaRNA2, NMI 2026",
     "url": "https://www.nature.com/articles/s42256-026-01223-x",
     "task": "RNA 3D structure/conformer prediction",
     "relation": "adjacent representation/structure capability",
     "rankable": "NO",
     "note": "cannot rank against thermodynamic NLL"},
    {"prior_art": "CHANRG 2026",
     "url": "https://arxiv.org/abs/2603.22330",
     "task": "RNA secondary-structure fair split and OOD benchmark",
     "relation": "benchmark methodology reference",
     "rankable": "NO",
     "note": "task not equivalent"},
]

EXPOSURE_ROWS = [
    {"model_id": "global_censor_intercept", "head": "censored intercept",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "train_only_scaffold", "head": "censored scaffold one-hot",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "scaffold_context_hierarchy", "head": "censored scaffold+context one-hot+ridge",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "motif_topology_hierarchy", "head": "censored motif+scaffold+length",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "onehot_kmer_ridge", "head": "censored 3-mer linear+ridge",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "position_aware_additive", "head": "censored 63-D sequence linear",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "edit_knn", "head": "local edit-distance KNN",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": False},
    {"model_id": "mutation_graph_smoother", "head": "mutation-graph propagation",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": False},
    {"model_id": "no_sequence_latent_operator", "head": "latent-operator, constant location",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "corrected_v1_31", "head": "latent-operator, 63-D sequence location",
     "external_data": "none", "pretraining_exposure": "NA", "matched_head": True},
    {"model_id": "denny_inspired_scalar_latent_proxy",
     "head": "63-D->scalar latent (NON-matched)", "external_data": "none",
     "pretraining_exposure": "NA", "matched_head": False,
     "note": "proxy; NOT faithful Denny fingerprint"},
    {"model_id": "viennarna_secondary_ensemble_proxy",
     "head": "ViennaRNA features + linear Tobit (NON-matched)",
     "external_data": "ViennaRNA (external package)", "pretraining_exposure": "NA",
     "matched_head": False, "note": "proxy; secondary-structure only"},
    {"model_id": "frozen_rnafm_global_head",
     "head": "frozen 640-D embedding + global censored linear head (NON-matched)",
     "external_data": "frozen RNA-FM weights/tokenizer", "pretraining_exposure": "UNKNOWN_NOT_ASSERTED",
     "matched_head": False, "note": "real embedding, unmatched head"},
]


def write_p04(out_dir: Path, *, source_url: str = "UNKNOWN_NOT_ASSERTED") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "TaskEquivalence.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prior_art", "url", "task", "relation",
                                           "rankable", "note"])
        w.writeheader()
        for r in TASK_EQUIVALENCE:
            w.writerow(r)

    with (out_dir / "ExposureRegistry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model_id", "head", "external_data",
                                           "pretraining_exposure", "matched_head",
                                           "note"])
        w.writeheader()
        for r in EXPOSURE_ROWS:
            w.writerow(r)

    model_cards = []
    for r in EXPOSURE_ROWS:
        model_cards.append({
            "model_id": r["model_id"], "head": r["head"],
            "external_data": r["external_data"],
            "pretraining_exposure": r["pretraining_exposure"],
            "matched_head": r["matched_head"],
            "revision_sha": None, "weights_sha": None,
            "tokenizer_sha": None, "cache_sha": None,
            "note": r.get("note"),
        })
    with (out_dir / "ModelCardRegistry.jsonl").open("w") as fh:
        for c in model_cards:
            fh.write(json.dumps(c, sort_keys=True) + "\n")

    provenance = {
        "version": "v3",
        "canonical_source": source_url,
        "canonical_source_sha": "0989ddc00bb230fdb00bbc65433c943a0419e35c3d0799b481e741c4a24defe2",
        "acquisition_chain": "UNKNOWN_NOT_ASSERTED",
        "redistribution_license": "PENDING_LEGAL",
        "derivatives_license": "PENDING_DATA_LICENSE",
        "note": ("URL/payload -> canonical acquisition chain and redistribution "
                 "permission are UNKNOWN_NOT_ASSERTED until the legal dossier is "
                 "closed; benchmark artifacts remain non-public until then."),
    }
    (out_dir / "SourceProvenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    license_md = f"""# License / redistribution decision (P0.4)

Date: 2026-08-11 (Asia/Shanghai)
Disposition: **PENDING_LEGAL / NOT_YET_AUTHORIZED_FOR_PUBLIC_RELEASE**

## Code
- Repository ships an MIT `LICENSE` file.
- The strict audit (2026-08-11) found `r6/ReleaseManifest.json` still says
  "repo has no LICENSE" and reports `OPEN_SOURCE_PENDING`.  That release
  manifest is INVALIDATED (release seal invalid).
- Code license disposition: `PENDING_AUTHOR_FINAL` until the author confirms
  the intended open-source license (MIT/Apache-2.0/BSD-3/CC0).

## Dataset
- Source: Denny et al., Cell 2018 (single tectoRNA study).
- Redistribution of the source data and of canonical-derived/row-level
  derivatives: `PENDING_LEGAL` until confirmed with the data authors/journal.
- Benchmark artifacts (row predictions, leaderboards) are NON-PUBLIC until the
  dataset legal dossier is closed.

## Rule
- No benchmark artifact is released and no submission is authorized until the
  science, reproducibility, release AND legal gates are all closed.
"""
    (out_dir / "LicenseDecision.md").write_text(license_md)


if __name__ == "__main__":
    import sys
    write_p04(Path(sys.argv[1]))