#!/usr/bin/env python3
"""Fetch and audit small NCBI SRA full-XML metadata records.

Only provenance metadata is emitted: accession/alias fields, source-material
attributes, original FASTQ file metadata, and hashes. No FASTQ sequence content,
processed labels, or private credentials are emitted. Source-material numbers
are treated as candidates only; this script never promotes them to processed
library labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


DEFAULT_RUNS = (
    "SRR31402663",
    "SRR31402664",
    "SRR35766784",
    "SRR35766785",
    "SRR38259812",
)
PROCESSED_TOKEN = re.compile(rb"\bpdb_library_[^\s,;<>]+", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(parent: ET.Element | None, tag: str) -> str | None:
    if parent is None:
        return None
    for child in parent.iter():
        if local_name(child.tag) == tag and child.text:
            return " ".join(child.text.split())
    return None


def first_element(root: ET.Element, tag: str) -> ET.Element | None:
    return next((item for item in root.iter() if local_name(item.tag) == tag), None)


def parse_xml(run: str, data: bytes, raw_path: str) -> dict[str, object]:
    root = ET.fromstring(data)
    experiment = first_element(root, "EXPERIMENT")
    sample = first_element(root, "SAMPLE")
    run_element = first_element(root, "RUN")
    library = first_element(root, "LIBRARY_DESCRIPTOR")
    sample_attributes: dict[str, str] = {}
    for attribute in root.iter():
        if local_name(attribute.tag) != "SAMPLE_ATTRIBUTE":
            continue
        tag = child_text(attribute, "TAG")
        value = child_text(attribute, "VALUE")
        if tag and value:
            sample_attributes[tag] = value
    original_files = []
    for file_element in root.iter():
        if local_name(file_element.tag) != "SRAFile":
            continue
        if file_element.attrib.get("semantic_name") != "fastq":
            continue
        original_files.append(
            {
                "filename": file_element.attrib.get("filename"),
                "size_bytes": int(file_element.attrib["size"])
                if file_element.attrib.get("size", "").isdigit()
                else None,
                "md5": file_element.attrib.get("md5"),
                "version": file_element.attrib.get("version"),
            }
        )
    return {
        "run_accession": run,
        "raw_xml_path": raw_path,
        "experiment_accession": experiment.attrib.get("accession") if experiment is not None else None,
        "experiment_alias": experiment.attrib.get("alias") if experiment is not None else None,
        "sample_accession": sample.attrib.get("accession") if sample is not None else None,
        "sample_alias": sample.attrib.get("alias") if sample is not None else None,
        "biosample_accession": next(
            (
                external.attrib.get("accession")
                for external in root.iter()
                if local_name(external.tag) == "EXTERNAL_ID"
                and external.attrib.get("namespace") == "BioSample"
            ),
            None,
        ),
        "library_name": child_text(library, "LIBRARY_NAME"),
        "library_strategy": child_text(library, "LIBRARY_STRATEGY"),
        "library_source": child_text(library, "LIBRARY_SOURCE"),
        "library_selection": child_text(library, "LIBRARY_SELECTION"),
        "source_material_id": sample_attributes.get("source_material_id"),
        "collection_date": sample_attributes.get("collection_date"),
        "sample_attributes": {
            key: sample_attributes[key]
            for key in sorted(sample_attributes)
            if key in {"source_material_id", "collection_date", "isolation_source", "host", "geo_loc_name"}
        },
        "original_fastq_files": original_files,
        "processed_namespace_tokens_in_xml": sorted(
            {match.decode("utf-8", errors="replace") for match in PROCESSED_TOKEN.findall(data)}
        ),
        "primary_labels_admitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--run", action="append", dest="runs", default=None)
    parser.add_argument("--contract-sha256", required=True)
    args = parser.parse_args()
    runs = tuple(args.runs or DEFAULT_RUNS)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "phase0-sra-full-xml-metadata-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": args.contract_sha256,
        "endpoint_template": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id={run}&rettype=full&retmode=xml",
        "runs": [],
        "primary_labels_admitted": False,
        "raw_sequence_content_emitted": False,
        "scientific_gate_effect": "NO_PHASE_0_PASS",
    }
    failures: list[dict[str, str]] = []
    for index, run in enumerate(runs):
        raw_path = args.raw_dir / f"{run}.xml"
        try:
            if raw_path.exists():
                data = raw_path.read_bytes()
            else:
                query = urllib.parse.urlencode(
                    {"db": "sra", "id": run, "rettype": "full", "retmode": "xml"}
                )
                request = urllib.request.Request(
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{query}",
                    headers={"User-Agent": "rna-junction-preorganization/phase0-audit"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                raw_path.write_bytes(data)
            record = parse_xml(run, data, str(raw_path))
            record["raw_xml_sha256"] = sha256_bytes(data)
            record["raw_xml_size_bytes"] = len(data)
            result["runs"].append(record)  # type: ignore[union-attr]
        except (OSError, ET.ParseError, ValueError, urllib.error.URLError) as exc:
            failures.append({"run": run, "error_type": type(exc).__name__})
        if index + 1 < len(runs):
            time.sleep(1)
    result["failures"] = failures
    result["status"] = "SRA_FULL_XML_METADATA_AUDIT_COMPLETE" if not failures else "SRA_FULL_XML_METADATA_AUDIT_PARTIAL"
    result["crosswalk_interpretation"] = {
        "source_material_id_is_candidate_only": True,
        "processed_library_binding_found_in_xml": any(
            record.get("processed_namespace_tokens_in_xml")
            for record in result["runs"]  # type: ignore[union-attr]
        ),
        "required_next_step": "Bind source-material/replicate metadata to official processed namespace using author-defined evidence; do not infer pdb_library labels from integer equality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_count": len(result["runs"]),  # type: ignore[arg-type]
                "failure_count": len(failures),
                "primary_labels_admitted": False,
                "scientific_gate_effect": "NO_PHASE_0_PASS",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
