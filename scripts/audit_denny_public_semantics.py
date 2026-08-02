#!/usr/bin/env python3
"""Bind Denny et al. public-method semantics to an immutable BioC snapshot.

This audit records only passage indexes, hashes, and semantic decisions. It does
not emit raw sequences, row-level workbook values, or primary labels. A
successful result establishes the published censor/interpolation semantics as a
Phase-0 sub-evidence component; it does not establish a raw-to-processed
crosswalk and cannot unlock modeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "censor_upper_bound": (
        r"upper bound of the measurable range of affinity.*?7[.]1",
        r"nonbinders.*?upper bound.*?7[.]1",
    ),
    "nonbinder_direction": (
        r"ΔG\s*>\s*[−-]7[.]1",
        r"ΔG values.*?7[.]1.*?nonbind",
    ),
    "same_context_median_interpolation": (
        r"missing data were interpolated.*?median affinity measured for other motifs"
        r".*?same chip-scaffold/flow-piece context",
    ),
    "nearest_fingerprint_interpolation": (
        r"20 most similar thermodynamic fingerprints.*?0[.]2 kcal/mol MAD"
        r".*?based only on measured contexts",
    ),
    "interpolation_only_for_comparative_analysis": (
        r"clustering was carried out on data sets with interpolated data"
        r".*?individual data points.*?only uninterpolated data",
    ),
    "nine_scaffold_context": (
        r"across the nine scaffolds",
    ),
    "nine_to_eleven_bp_context": (
        r"9 bp flow piece.*?10 bp chip",
        r"8 to 12 bp",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_passages(payload: Any) -> Iterable[dict[str, Any]]:
    roots = payload if isinstance(payload, list) else [payload]
    for root_index, root in enumerate(roots):
        for document_index, document in enumerate(root.get("documents", [])):
            for passage_index, passage in enumerate(document.get("passages", [])):
                text = passage.get("text") or ""
                yield {
                    "root_index": root_index,
                    "document_index": document_index,
                    "passage_index": passage_index,
                    "text": " ".join(text.split()),
                    "infons": passage.get("infons", {}),
                }


def find_evidence(passages: list[dict[str, Any]], patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for passage in passages:
        text = passage["text"]
        matched = [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]
        if not matched:
            continue
        evidence.append(
            {
                "root_index": passage["root_index"],
                "document_index": passage["document_index"],
                "passage_index": passage["passage_index"],
                "passage_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "passage_length": len(text),
                "matched_pattern_count": len(matched),
                "infons": {
                    key: value
                    for key, value in passage["infons"].items()
                    if key in {"section", "type", "id", "offset", "sentences"}
                },
            }
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema_version": "phase0-denny-public-semantics-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "expected_source_sha256": args.expected_source_sha256,
        "contract_sha256": args.contract_sha256,
        "raw_sequence_content_emitted": False,
        "raw_workbook_values_emitted": False,
        "primary_labels_admitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    if not args.source.is_file():
        result.update({"status": "BLOCKED_SOURCE_SNAPSHOT_MISSING", "observed_source_sha256": None})
    else:
        observed = sha256_file(args.source)
        result["observed_source_sha256"] = observed
        result["source_size_bytes"] = args.source.stat().st_size
        if observed != args.expected_source_sha256:
            result["status"] = "BLOCKED_SOURCE_HASH_MISMATCH"
        else:
            try:
                payload = json.loads(args.source.read_text(encoding="utf-8"))
                passages = list(iter_passages(payload))
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError, AttributeError) as exc:
                result.update({"status": "BLOCKED_SOURCE_PARSE_ERROR", "error_type": type(exc).__name__})
            else:
                fact_evidence = {
                    name: find_evidence(passages, patterns)
                    for name, patterns in FACT_PATTERNS.items()
                }
                missing = [name for name, evidence in fact_evidence.items() if not evidence]
                result.update(
                    {
                        "passage_count": len(passages),
                        "fact_evidence": fact_evidence,
                        "missing_facts": missing,
                        "semantic_decisions": {
                            "censor_direction": "UPPER_BOUND_CAP_AT_DELTA_G_MINUS_7_1"
                            if not {"censor_upper_bound", "nonbinder_direction"} & set(missing)
                            else "NOT_ESTABLISHED",
                            "same_context_missing_value_rule": "MEDIAN_OF_OTHER_MOTIFS_IN_SAME_CHIP_SCAFFOLD_FLOW_PIECE_CONTEXT"
                            if "same_context_median_interpolation" not in missing
                            else "NOT_ESTABLISHED",
                            "dissimilar_context_missing_value_rule": "MEDIAN_OF_20_NEAREST_FINGERPRINTS_WITHIN_0_2_KCAL_MOL_MAD_USING_MEASURED_CONTEXTS"
                            if "nearest_fingerprint_interpolation" not in missing
                            else "NOT_ESTABLISHED",
                            "interpolated_values_admitted_to_individual_data_point_plots": False,
                        },
                    }
                )
                result["status"] = (
                    "PUBLIC_SEMANTICS_BOUND_NO_CROSSWALK_UNLOCK"
                    if not missing
                    else "PUBLIC_SEMANTICS_INCOMPLETE_NO_CROSSWALK_UNLOCK"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "missing_facts": result.get("missing_facts"),
                "primary_labels_admitted": False,
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PUBLIC_SEMANTICS_BOUND_NO_CROSSWALK_UNLOCK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
