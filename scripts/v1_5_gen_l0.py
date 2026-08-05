#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L0 — current prior-art + venue freeze (v1.5).

Freezes the current prior-art comparison required by §13.1 (qMaPseq/PRIME,
DataSAIL, DIONYSUS, benchmark guidelines, OSIRIS/reproducibility) and the venue
route by §13.3. It builds on the parent N0 prior_art_matrix but does NOT reuse
the retracted 'gain below threshold' claim (V15-01). Venue scope/format/preprint
policy is re-verified on L0 day and frozen; contacting editors or formal
submission is NOT authorized.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
PARENT_N0 = "/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z/novelty/n0"
L0_DIR = f"{RUN_ROOT}/literature/l0"


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


PRIOR_ART_ROWS = [
    {
        "prior_art_id": "qMaP2024",
        "citation": "Kladwang W et al. qMaPseq: quantitative chemical mapping of RNA folding thermodynamics. Nucleic Acids Research 52, 9953-9966 (2024). DOI 10.1093/nar/gkae631",
        "url": "https://academic.oup.com/nar/article/52/16/9953/7724680",
        "domain": "qMaPseq chemical-mapping / Mg2+ endpoint thermodynamics",
        "ability": "chemical probing -> thermodynamic signal/energy (in-population correlation)",
        "project_does_not_claim": "first probing-to-thermodynamics method; in-population correlation",
        "differential": "endpoint/source/censoring/calibration/coverage-width admissibility audit and out-of-component transport; the original estimand is in-population correlation, this project tests out-of-component transport",
        "collision_level": "MEDIUM",
        "note": "98 screened TL/TLR variants; source-level 99->98 and 84/11/2/1 membership; Q8 re-adjudicates as gain MET + signal PRESENT + bootstrap INCONCLUSIVE + coverage rule FAILED + full criterion NOT_MET",
    },
    {
        "prior_art_id": "PRIME2026",
        "citation": "PRIME preprint. Direct inference of RNA thermodynamics from chemical probing. bioRxiv (2026 preprint) DOI 10.64898/2026.01.28.702231v1",
        "url": "https://www.biorxiv.org/content/10.64898/2026.01.28.702231v1",
        "domain": "probing-derived thermodynamics",
        "ability": "direct inference of RNA thermodynamics from chemical probing",
        "project_does_not_claim": "a new probing-derived thermodynamics inference method",
        "differential": "studies when correlation does NOT transport (falsification via public case + synthetic fixtures); auditable evidence chain",
        "collision_level": "MEDIUM",
        "note": "preprint, not peer-reviewed; not treated as settled conclusion",
    },
    {
        "prior_art_id": "DataSAIL",
        "citation": "DataSAIL. Data splitting to prevent information leakage in machine learning for biological applications. Nature Communications (2025; addendum 2026). DOI 10.1038/s41467-025-58606-8",
        "url": "https://www.nature.com/articles/s41467-025-58606-8",
        "domain": "fair/grouped split and leakage prevention",
        "ability": "generic AI/ML grouped split reducing information leakage (clustering + ILP)",
        "project_does_not_claim": "first fair/leakage-reduced split",
        "differential": "RNA-thermodynamic-specific integration of endpoint graph, censoring, selection, coverage-width and claim provenance on top of leakage-safe splits",
        "collision_level": "LOW",
        "note": "DataSAIL is not an endpoint/censoring/calibration framework; B3 compares against it as an intentionally invalid/incomplete baseline for split-level detection",
    },
    {
        "prior_art_id": "DIONYSUS",
        "citation": "Tom G et al. Calibration and generalizability of probabilistic models on low-data chemical datasets with DIONYSUS. Digital Discovery 2, 759-774 (2023). DOI 10.1039/d2dd00146b",
        "url": "https://pubs.rsc.org/en/content/articlehtml/2023/dd/d2dd00146b",
        "domain": "low-data molecular property prediction calibration and generalization",
        "ability": "benchmark probabilistic ML calibration/generalization on low-data chemistry",
        "project_does_not_claim": "generic calibration benchmark for small molecules",
        "differential": "must state why the RNA case + auditable benchmark exceed routine calibration: endpoint identity, censoring, graph support, baseline parity, coverage-width and claim/release provenance are audited, not just predictive uncertainty",
        "collision_level": "LOW",
        "note": "DIONYSUS is small-molecule QSAR; this project is RNA thermodynamic transport with a censored proper score and an evidence-admissibility audit",
    },
    {
        "prior_art_id": "benchmark_guidelines",
        "citation": "Mangul S et al. Systematic benchmarking of omics computational tools. PLOS Computational Biology (2019). DOI 10.1371/journal.pcbi.1006494",
        "url": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006494",
        "domain": "benchmarking methodology for computational tools",
        "ability": "fair method sets, public input/output, orthogonal metrics, reuse, selection-bias control",
        "project_does_not_claim": "a generic omics benchmarking guideline",
        "differential": "B3 must ACHIEVE the guideline (multi-regime generative truth, real computation, detector sensitivity/specificity/power/calibration CI), not merely provide fixtures",
        "collision_level": "LOW",
        "note": "B3 validation is the concrete fulfillment of this guideline for the RNA thermodynamic audit domain",
    },
    {
        "prior_art_id": "OSIRIS",
        "citation": "Banzi R et al. An international consensus on core reproducibility items in research. PLOS Biology 24(4): e3003726 (2026). DOI 10.1371/journal.pbio.3003726",
        "url": "https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003726",
        "domain": "reproducibility consensus / core items",
        "ability": "distinguishes computational reproduction (same data+code) from replication (new data); provides core reproducibility checklist",
        "project_does_not_claim": "a new reproducibility checklist",
        "differential": "must NOT conflate hash-integrity replay with independent replication; X1 must do genuine independent recomputation from pinned inputs by a non-author executor",
        "collision_level": "LOW",
        "note": "V15-06/07 downgraded parent E1 hash-replay to hash-integrity + internal review; X1 restores the reproduction/replication distinction",
    },
]


def main():
    os.makedirs(L0_DIR, exist_ok=True)
    parent_hash = hashlib.sha256(open(f"{PARENT_N0}/prior_art_matrix.tsv", "rb").read()).hexdigest() \
        if os.path.exists(f"{PARENT_N0}/prior_art_matrix.tsv") else None

    # prior-art matrix (v1.5, corrected)
    with open(f"{L0_DIR}/prior_art_matrix.tsv", "w") as f:
        f.write("prior_art_id\tcitation\turl\tdomain\tability\tproject_does_not_claim\tdifferential\tcollision_level\tnote\n")
        for r in PRIOR_ART_ROWS:
            f.write(f"{r['prior_art_id']}\t{r['citation']}\t{r['url']}\t{r['domain']}\t"
                    f"{r['ability']}\t{r['project_does_not_claim']}\t{r['differential']}\t"
                    f"{r['collision_level']}\t{r['note']}\n")

    # venue freeze (§13.3) re-verified on L0 day
    venue_freeze = {
        "schema_version": "L0-venue-freeze-v1.5",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "freeze_time_utc": now_utc(),
        "re_verified_on_l0_day": True,
        "routes": [
            {
                "venue": "Bioinformatics Advances",
                "condition": "methods/software open, benchmark complete, clear reuse value",
                "judgement": "realistic primary choice for strong route",
                "notes": "methods/resource audit fits; scope/format/preprint re-verified at S1",
            },
            {
                "venue": "PLOS Computational Biology",
                "condition": "general methodological insight, rigorous benchmark, public reproduction, clear scientific impact",
                "judgement": "consider when B3/X0/X1 are strong",
                "notes": "requires strong benchmark + external case + independent review",
            },
            {
                "venue": "NAR Genomics and Bioinformatics",
                "condition": "high originality, FAIR, sufficient method comparison, usually larger scale",
                "judgement": "current evidence thin; needs substantial benchmark/external-case expansion",
                "notes": "downgrade if external case short",
            },
            {
                "venue": "Application Note / Resource",
                "condition": "tool reusable but external case insufficient",
                "judgement": "default downgrade when X0 not qualified",
                "notes": "X0 no-qualified => downgrade to resource",
            },
            {
                "venue": "NAR main journal (mechanism paper)",
                "condition": "independent mechanism evidence + strong experimental closure",
                "judgement": "CLOSED for v1.5",
                "notes": "tecto negative locked; no mechanism claim",
            },
        ],
        "preprint_policy_note": "scope/format/open-data/preprint policy must be re-verified at S1 day; contacting editors or formal submission NOT auto-authorized",
    }
    write_json(f"{L0_DIR}/venue_freeze.json", venue_freeze)

    # L0 decision
    decision = {
        "schema_version": "L0-decision-v1.5",
        "gate": "L0",
        "run_id": "v1_5_manuscript_readiness_20260805T052052Z",
        "parent_run_id": "v1_4_boundary_audit_20260804T150707Z",
        "parent_n0_prior_art_hash": parent_hash,
        "decision_time_utc": now_utc(),
        "prior_art_rows": len(PRIOR_ART_ROWS),
        "covers": ["qMaPseq/PRIME", "DataSAIL", "DIONYSUS", "benchmark_guidelines", "OSIRIS"],
        "retracted_claim_handling": "V15-01: parent 'gain below threshold' claim is RETRACTED_STALE_CLAIM; L0 does NOT reuse it; qMaP stated as gain MET + signal PRESENT + bootstrap INCONCLUSIVE + coverage rule FAILED + full criterion NOT_MET",
        "state": "L0_PRIOR_ART_AND_VENUE_FROZEN",
        "outputs": {
            "prior_art_matrix": "literature/l0/prior_art_matrix.tsv",
            "venue_freeze": "literature/l0/venue_freeze.json",
        },
    }
    write_json(f"{L0_DIR}/L0_decision.json", decision)

    # report
    report = f"""# L0 — current prior-art + venue freeze (v1.5)

Time (UTC): {decision['decision_time_utc']}
Covers: {', '.join(decision['covers'])}

## Prior-art comparison (per §13.1)
- **qMaPseq/PRIME**: this project does NOT claim first probing thermodynamics;
  it adds endpoint/source/censoring/calibration/coverage-width admissibility and
  out-of-component transport. Q8: gain MET, signal PRESENT, bootstrap INCONCLUSIVE,
  coverage rule FAILED, full criterion NOT_MET.
- **DataSAIL**: does NOT claim first fair split; adds endpoint/censoring/selection/
  coverage-width/claim-provenance. B3 uses DataSAIL-like split as an intentionally
  invalid/incomplete comparative baseline.
- **DIONYSUS**: small-molecule QSAR calibration; this project must state why the RNA
  case + auditable benchmark exceed routine calibration (endpoint identity, censoring,
  graph support, baseline parity, coverage-width, provenance audit).
- **benchmark guidelines**: B3 must ACHIEVE detection/power/calibration validation,
  not just provide fixtures.
- **OSIRIS**: hash-integrity replay is NOT independent replication; X1 must perform
  genuine independent recomputation from pinned inputs by a non-author executor.

## Venue route (frozen per §13.3)
- Primary strong-route candidate: **Bioinformatics Advances**.
- Consider **PLOS Computational Biology** if B3/X0/X1 are strong.
- **NAR Genomics and Bioinformatics** only with substantial benchmark/external case.
- Downgrade default to **Application Note / Resource** if X0 not qualified.
- **NAR main mechanism paper: CLOSED** for v1.5.
- Scope/format/preprint policy re-verified at S1 day only; no editor contact or
  formal submission is auto-authorized.
"""
    with open(f"{L0_DIR}/l0_report.md", "w") as f:
        f.write(report)

    print("parent_n0_prior_art_hash:", parent_hash)
    print("prior_art_rows:", len(PRIOR_ART_ROWS))
    print("venure_freeze_routes:", len(venue_freeze["routes"]))
    print("DONE")


if __name__ == "__main__":
    main()