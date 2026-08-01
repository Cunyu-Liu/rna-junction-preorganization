#!/usr/bin/env python3
"""Derive a new fail-closed DMS dependency ledger from a route re-probe.

The previous ledger and route audit remain immutable.  This utility only
creates a new ledger when the new official route probe still has no 2xx
response and no payload was downloaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def atomic_dump(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-ledger", required=True, type=Path)
    parser.add_argument("--route-audit", required=True, type=Path)
    parser.add_argument("--range-audit", type=Path)
    parser.add_argument("--metadata-audit", type=Path)
    parser.add_argument("--provenance-audit", type=Path)
    parser.add_argument("--doi-provenance-audit", type=Path)
    parser.add_argument("--v8-audit", type=Path)
    parser.add_argument("--oai-format-audit", type=Path)
    parser.add_argument("--github-public-audit", type=Path)
    parser.add_argument("--fastq-batch-audit", type=Path)
    parser.add_argument("--raw-fastq-range-probe", type=Path)
    parser.add_argument("--downloader-control-audit", type=Path)
    parser.add_argument("--chunked-fastq-audit", type=Path)
    parser.add_argument("--partial-size-audit", type=Path)
    parser.add_argument("--final-fastq-audit", type=Path)
    parser.add_argument("--additional-range-audit", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    previous = args.previous_ledger.resolve()
    route_path = args.route_audit.resolve()
    range_path = args.range_audit.resolve() if args.range_audit else None
    metadata_path = args.metadata_audit.resolve() if args.metadata_audit else None
    provenance_path = args.provenance_audit.resolve() if args.provenance_audit else None
    doi_provenance_path = args.doi_provenance_audit.resolve() if args.doi_provenance_audit else None
    v8_path = args.v8_audit.resolve() if args.v8_audit else None
    oai_format_path = args.oai_format_audit.resolve() if args.oai_format_audit else None
    github_public_path = args.github_public_audit.resolve() if args.github_public_audit else None
    fastq_batch_path = args.fastq_batch_audit.resolve() if args.fastq_batch_audit else None
    raw_fastq_range_path = args.raw_fastq_range_probe.resolve() if args.raw_fastq_range_probe else None
    downloader_control_path = args.downloader_control_audit.resolve() if args.downloader_control_audit else None
    chunked_fastq_path = args.chunked_fastq_audit.resolve() if args.chunked_fastq_audit else None
    partial_size_path = args.partial_size_audit.resolve() if args.partial_size_audit else None
    final_fastq_path = args.final_fastq_audit.resolve() if args.final_fastq_audit else None
    additional_range_paths = [path.resolve() for path in args.additional_range_audit]
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing ledger: {output}")
    if not previous.is_file() or not route_path.is_file():
        raise SystemExit("previous ledger and route audit must exist")

    old = load_json(previous)
    route = load_json(route_path)
    if old.get("status") != "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE":
        raise SystemExit("previous ledger is not fail-closed")
    if route.get("status") != "ROUTE_REPROBE_BLOCKED_NO_2XX":
        raise SystemExit("route audit is not a blocked no-2xx result")
    if route.get("payload_downloaded") is not False:
        raise SystemExit("route audit does not prove payload_downloaded=false")
    if route.get("access_control_bypassed") is not False:
        raise SystemExit("route audit does not prove access_control_bypassed=false")
    range_audit = None
    if range_path is not None:
        range_audit = load_json(range_path)
        if range_audit.get("status") != "BLOCKED_HTTP_403_RANGE_PROBE" or range_audit.get("payload_downloaded") is not False:
            raise SystemExit("range audit is not a blocked no-payload result")
        if range_audit.get("observed_body_bytes") != 0:
            raise SystemExit("range audit does not prove zero observed body bytes")
    metadata_audit = None
    if metadata_path is not None:
        metadata_audit = load_json(metadata_path)
        if metadata_audit.get("status") != "BLOCKED_HTTP_403_FILE_METADATA_PROBE" or metadata_audit.get("payload_downloaded") is not False:
            raise SystemExit("metadata audit is not a blocked no-payload result")
        if metadata_audit.get("observed_body_bytes") != 0:
            raise SystemExit("metadata audit does not prove zero observed body bytes")
    provenance_audit = None
    if provenance_path is not None:
        provenance_audit = load_json(provenance_path)
        allowed_statuses = {
            "BLOCKED_NO_2XX_PROVENANCE_ROUTE",
            "PROVENANCE_HEAD_ONLY_METADATA_BLOCKED",
        }
        if provenance_audit.get("status") not in allowed_statuses:
            raise SystemExit("provenance audit is not a blocked no-payload result")
        if provenance_audit.get("payload_downloaded") is not False or provenance_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("provenance audit does not prove that no processed payload was admitted")
    doi_provenance_audit = None
    if doi_provenance_path is not None:
        doi_provenance_audit = load_json(doi_provenance_path)
        if doi_provenance_audit.get("status") not in {"DOI_OAI_PROVENANCE_METADATA_AVAILABLE", "BLOCKED_NO_2XX_DOI_OAI_PROVENANCE_ROUTE"}:
            raise SystemExit("DOI/OAI provenance audit has an unexpected status")
        if doi_provenance_audit.get("payload_downloaded") is not False or doi_provenance_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("DOI/OAI provenance audit does not prove that no processed payload was admitted")
    v8_audit = None
    if v8_path is not None:
        v8_audit = load_json(v8_path)
        if v8_audit.get("status") not in {"FIGSHARE_V8_ROUTE_METADATA_AVAILABLE", "BLOCKED_NO_2XX_FIGSHARE_V8_ROUTE"}:
            raise SystemExit("Figshare v8 audit has an unexpected status")
        if v8_audit.get("payload_downloaded") is not False or v8_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("Figshare v8 audit does not prove that no processed payload was admitted")
    oai_format_audit = None
    if oai_format_path is not None:
        oai_format_audit = load_json(oai_format_path)
        if oai_format_audit.get("status") not in {
            "FIGSHARE_OAI_FORMAT_METADATA_AVAILABLE",
            "BLOCKED_NO_2XX_FIGSHARE_OAI_FORMAT_ROUTE",
        }:
            raise SystemExit("Figshare OAI format audit has an unexpected status")
        if oai_format_audit.get("payload_downloaded") is not False or oai_format_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("Figshare OAI format audit does not prove that no processed payload was admitted")
    github_public_audit = None
    if github_public_path is not None:
        github_public_audit = load_json(github_public_path)
        if github_public_audit.get("status") not in {
            "GITHUB_PUBLIC_METADATA_AVAILABLE",
            "BLOCKED_GITHUB_PUBLIC_METADATA_ROUTE",
        }:
            raise SystemExit("GitHub public metadata audit has an unexpected status")
        if github_public_audit.get("payload_downloaded") is not False or github_public_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("GitHub public metadata audit does not prove that no processed payload was admitted")
    fastq_batch_audit = None
    if fastq_batch_path is not None:
        fastq_batch_audit = load_json(fastq_batch_path)
        if fastq_batch_audit.get("status") not in {
            "BATCH_COMPLETE",
            "BATCH_PARTIAL_PENDING_OR_BLOCKED",
        }:
            raise SystemExit("FASTQ batch audit has an unexpected status")
        if fastq_batch_audit.get("scientific_gate_effect") != "NO_PHASE_0_PASS":
            raise SystemExit("FASTQ batch audit does not preserve the scientific stop rule")
        if fastq_batch_audit.get("raw_sequence_content_emitted") is not False or fastq_batch_audit.get("scientific_labels_admitted") is not False:
            raise SystemExit("FASTQ batch audit is not fail-closed")
    raw_fastq_range_probe = None
    if raw_fastq_range_path is not None:
        raw_fastq_range_probe = load_json(raw_fastq_range_path)
        if raw_fastq_range_probe.get("status") not in {
            "RANGE_PROBE_206_EXACT",
            "BLOCKED_HTTP_403_RANGE_PROBE",
            "BLOCKED_NONEXACT_RANGE_RESPONSE",
        }:
            raise SystemExit("raw FASTQ range probe has an unexpected status")
        if raw_fastq_range_probe.get("processed_payload_admitted") is not False:
            raise SystemExit("raw FASTQ range probe is not fail-closed")
    downloader_control_audit = None
    if downloader_control_path is not None:
        downloader_control_audit = load_json(downloader_control_path)
        if downloader_control_audit.get("status") not in {"SIGSTOP_SENT", "SIGCONT_SENT"}:
            raise SystemExit("downloader control audit has an unexpected status")
        if downloader_control_audit.get("signal_sent") is not True:
            raise SystemExit("downloader control audit does not prove that its signal was sent")
        for field in ("new_process_started", "final_files_overwritten", "partial_files_deleted", "raw_sequence_content_emitted"):
            if downloader_control_audit.get(field) is not False:
                raise SystemExit(f"downloader control audit is not fail-closed: {field}")
    chunked_fastq_audit = None
    if chunked_fastq_path is not None:
        chunked_fastq_audit = load_json(chunked_fastq_path)
        if not str(chunked_fastq_audit.get("status", "")).startswith(("CHUNKED_DOWNLOAD_", "CHUNKED_MERGE_")):
            raise SystemExit("chunked FASTQ audit has an unexpected status")
        if chunked_fastq_audit.get("scientific_labels_admitted") is not False or chunked_fastq_audit.get("raw_sequence_content_emitted") is not False:
            raise SystemExit("chunked FASTQ audit is not fail-closed")
    partial_size_audit = None
    if partial_size_path is not None:
        partial_size_audit = load_json(partial_size_path)
        if partial_size_audit.get("status") not in {"PARTIAL_SIZE_REGRESSION_AFTER_SAFE_CONTINUE_FROZEN", "PARTIAL_SIZE_OBSERVATION_FROZEN"}:
            raise SystemExit("partial-size audit has an unexpected status")
        for field in ("partial_file_deleted", "final_file_overwritten", "raw_sequence_content_emitted", "scientific_labels_admitted"):
            if partial_size_audit.get(field) is not False:
                raise SystemExit(f"partial-size audit is not fail-closed: {field}")
    final_fastq_audit = None
    if final_fastq_path is not None:
        final_fastq_audit = load_json(final_fastq_path)
        if final_fastq_audit.get("status") not in {
            "FASTQ_PAYLOAD_AUDIT_COMPLETE",
            "BLOCKED_HASH_MISMATCH",
            "BLOCKED_MALFORMED_FASTQ",
            "BLOCKED_PAIRED_ID_MISMATCH",
        }:
            raise SystemExit("final FASTQ audit has an unexpected status")
        if final_fastq_audit.get("scientific_gate_effect") != "NO_PHASE_0_PASS":
            raise SystemExit("final FASTQ audit does not preserve the scientific stop rule")
        if final_fastq_audit.get("raw_sequence_content_emitted") is not False or final_fastq_audit.get("scientific_labels_admitted") is not False:
            raise SystemExit("final FASTQ audit is not fail-closed")
    additional_range_audits = []
    for additional_range_path in additional_range_paths:
        additional_range = load_json(additional_range_path)
        if additional_range.get("status") != "BLOCKED_HTTP_403_RANGE_PROBE" or additional_range.get("payload_downloaded") is not False or additional_range.get("observed_body_bytes") != 0:
            raise SystemExit(f"additional range audit is not a blocked zero-byte result: {additional_range_path}")
        if additional_range.get("processed_payload_admitted") is not False:
            raise SystemExit(f"additional range audit is not fail-closed: {additional_range_path}")
        additional_range_audits.append(additional_range)

    ledger = copy.deepcopy(old)
    ledger["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["derived_from_ledger"] = str(previous)
    ledger["derived_from_ledger_sha256"] = sha256(previous)
    ledger["latest_route_reprobe"] = file_record(route_path)
    ledger["latest_route_reprobe_status"] = route.get("status")
    ledger["latest_route_reprobe_payload_downloaded"] = False
    ledger["latest_route_reprobe_access_control_bypassed"] = False

    route_evidence = ledger.setdefault("route_evidence", {})
    prior_latest = route_evidence.get("latest")
    prior_status = route_evidence.get("latest_status")
    if prior_latest is not None:
        route_evidence["previous"] = prior_latest
        route_evidence["previous_status"] = prior_status
    route_evidence["latest"] = file_record(route_path)
    route_evidence["latest_status"] = route.get("status")
    route_evidence["latest_route_count"] = len(route.get("routes", [])) if isinstance(route.get("routes"), list) else None
    route_evidence["latest_all_http_codes"] = sorted(
        {item.get("http_code") for item in route.get("routes", []) if isinstance(item, dict)}
    )
    route_evidence["latest_payload_downloaded"] = False
    route_evidence["access_control_bypassed"] = False
    if range_path is not None and range_audit is not None:
        route_evidence["latest_range_probe"] = file_record(range_path)
        route_evidence["latest_range_probe_status"] = range_audit.get("status")
        route_evidence["latest_range_probe_observed_bytes"] = range_audit.get("observed_body_bytes")
        ledger["latest_range_probe"] = file_record(range_path)
        ledger["latest_range_probe_status"] = range_audit.get("status")
        ledger["latest_range_probe_payload_downloaded"] = False
    if metadata_path is not None and metadata_audit is not None:
        route_evidence["latest_metadata_probe"] = file_record(metadata_path)
        route_evidence["latest_metadata_probe_status"] = metadata_audit.get("status")
        route_evidence["latest_metadata_probe_observed_bytes"] = metadata_audit.get("observed_body_bytes")
        ledger["latest_metadata_probe"] = file_record(metadata_path)
        ledger["latest_metadata_probe_status"] = metadata_audit.get("status")
        ledger["latest_metadata_probe_payload_downloaded"] = False
    if provenance_path is not None and provenance_audit is not None:
        route_evidence["latest_provenance_audit"] = file_record(provenance_path)
        route_evidence["latest_provenance_audit_status"] = provenance_audit.get("status")
        route_evidence["latest_provenance_successful_route_count"] = provenance_audit.get("successful_route_count")
        route_evidence["latest_provenance_successful_metadata_route_count"] = provenance_audit.get("successful_metadata_route_count")
        route_evidence["latest_provenance_payload_downloaded"] = False
        ledger["latest_provenance_audit"] = file_record(provenance_path)
        ledger["latest_provenance_audit_status"] = provenance_audit.get("status")
        ledger["latest_provenance_payload_downloaded"] = False
    if doi_provenance_path is not None and doi_provenance_audit is not None:
        route_evidence["latest_doi_oai_provenance_audit"] = file_record(doi_provenance_path)
        route_evidence["latest_doi_oai_provenance_audit_status"] = doi_provenance_audit.get("status")
        route_evidence["latest_doi_oai_successful_route_count"] = doi_provenance_audit.get("successful_route_count")
        route_evidence["latest_doi_oai_payload_downloaded"] = False
        ledger["latest_doi_oai_provenance_audit"] = file_record(doi_provenance_path)
        ledger["latest_doi_oai_provenance_audit_status"] = doi_provenance_audit.get("status")
        ledger["latest_doi_oai_payload_downloaded"] = False
    if v8_path is not None and v8_audit is not None:
        route_evidence["latest_figshare_v8_audit"] = file_record(v8_path)
        route_evidence["latest_figshare_v8_audit_status"] = v8_audit.get("status")
        route_evidence["latest_figshare_v8_successful_route_count"] = v8_audit.get("successful_route_count")
        route_evidence["latest_figshare_v8_payload_downloaded"] = False
        ledger["latest_figshare_v8_audit"] = file_record(v8_path)
        ledger["latest_figshare_v8_audit_status"] = v8_audit.get("status")
        ledger["latest_figshare_v8_payload_downloaded"] = False
    if oai_format_path is not None and oai_format_audit is not None:
        route_evidence["latest_figshare_oai_format_audit"] = file_record(oai_format_path)
        route_evidence["latest_figshare_oai_format_audit_status"] = oai_format_audit.get("status")
        route_evidence["latest_figshare_oai_format_successful_route_count"] = oai_format_audit.get("successful_route_count")
        route_evidence["latest_figshare_oai_format_payload_downloaded"] = False
        ledger["latest_figshare_oai_format_audit"] = file_record(oai_format_path)
        ledger["latest_figshare_oai_format_audit_status"] = oai_format_audit.get("status")
        ledger["latest_figshare_oai_format_payload_downloaded"] = False
    if github_public_path is not None and github_public_audit is not None:
        route_evidence["latest_github_public_metadata_audit"] = file_record(github_public_path)
        route_evidence["latest_github_public_metadata_audit_status"] = github_public_audit.get("status")
        route_evidence["latest_github_public_metadata_successful_route_count"] = github_public_audit.get("successful_route_count")
        route_evidence["latest_github_public_metadata_payload_downloaded"] = False
        ledger["latest_github_public_metadata_audit"] = file_record(github_public_path)
        ledger["latest_github_public_metadata_audit_status"] = github_public_audit.get("status")
        ledger["latest_github_public_metadata_payload_downloaded"] = False
    if fastq_batch_path is not None and fastq_batch_audit is not None:
        route_evidence["latest_fastq_batch_audit"] = file_record(fastq_batch_path)
        route_evidence["latest_fastq_batch_audit_status"] = fastq_batch_audit.get("status")
        route_evidence["latest_fastq_batch_failed_run_count"] = fastq_batch_audit.get("failed_run_count")
        route_evidence["latest_fastq_batch_pending_run_count"] = fastq_batch_audit.get("pending_run_count")
        route_evidence["latest_fastq_batch_scientific_gate_effect"] = fastq_batch_audit.get("scientific_gate_effect")
        ledger["latest_fastq_batch_audit"] = file_record(fastq_batch_path)
        ledger["latest_fastq_batch_audit_status"] = fastq_batch_audit.get("status")
        ledger["latest_fastq_batch_scientific_gate_effect"] = fastq_batch_audit.get("scientific_gate_effect")
    if raw_fastq_range_path is not None and raw_fastq_range_probe is not None:
        route_evidence["latest_raw_fastq_range_probe"] = file_record(raw_fastq_range_path)
        route_evidence["latest_raw_fastq_range_probe_status"] = raw_fastq_range_probe.get("status")
        route_evidence["latest_raw_fastq_range_probe_observed_bytes"] = raw_fastq_range_probe.get("observed_body_bytes")
        route_evidence["latest_raw_fastq_range_probe_payload_downloaded"] = raw_fastq_range_probe.get("payload_downloaded")
        ledger["latest_raw_fastq_range_probe"] = file_record(raw_fastq_range_path)
        ledger["latest_raw_fastq_range_probe_status"] = raw_fastq_range_probe.get("status")
        ledger["latest_raw_fastq_range_probe_payload_downloaded"] = raw_fastq_range_probe.get("payload_downloaded")
    if downloader_control_path is not None and downloader_control_audit is not None:
        route_evidence["latest_downloader_control_audit"] = file_record(downloader_control_path)
        route_evidence["latest_downloader_control_audit_status"] = downloader_control_audit.get("status")
        route_evidence["latest_downloader_control_signal"] = downloader_control_audit.get("signal")
        ledger["latest_downloader_control_audit"] = file_record(downloader_control_path)
        ledger["latest_downloader_control_audit_status"] = downloader_control_audit.get("status")
        ledger["latest_downloader_control_signal"] = downloader_control_audit.get("signal")
    if chunked_fastq_path is not None and chunked_fastq_audit is not None:
        route_evidence["latest_chunked_fastq_audit"] = file_record(chunked_fastq_path)
        route_evidence["latest_chunked_fastq_audit_status"] = chunked_fastq_audit.get("status")
        route_evidence["latest_chunked_fastq_chunk_bytes"] = chunked_fastq_audit.get("chunk_bytes")
        ledger["latest_chunked_fastq_audit"] = file_record(chunked_fastq_path)
        ledger["latest_chunked_fastq_audit_status"] = chunked_fastq_audit.get("status")
        ledger["latest_chunked_fastq_chunk_bytes"] = chunked_fastq_audit.get("chunk_bytes")
    if partial_size_path is not None and partial_size_audit is not None:
        route_evidence["latest_partial_size_audit"] = file_record(partial_size_path)
        route_evidence["latest_partial_size_audit_status"] = partial_size_audit.get("status")
        route_evidence["latest_partial_observed_bytes"] = partial_size_audit.get("observed_compressed_bytes")
        route_evidence["latest_partial_prior_observed_bytes"] = partial_size_audit.get("prior_observation", {}).get("observed_compressed_bytes")
        ledger["latest_partial_size_audit"] = file_record(partial_size_path)
        ledger["latest_partial_size_audit_status"] = partial_size_audit.get("status")
        ledger["latest_partial_observed_bytes"] = partial_size_audit.get("observed_compressed_bytes")
    if final_fastq_path is not None and final_fastq_audit is not None:
        route_evidence["latest_final_fastq_audit"] = file_record(final_fastq_path)
        route_evidence["latest_final_fastq_audit_status"] = final_fastq_audit.get("status")
        route_evidence["latest_final_fastq_payload_count"] = len(final_fastq_audit.get("payloads", [])) if isinstance(final_fastq_audit.get("payloads"), list) else None
        ledger["latest_final_fastq_audit"] = file_record(final_fastq_path)
        ledger["latest_final_fastq_audit_status"] = final_fastq_audit.get("status")
    if additional_range_paths:
        route_evidence["latest_additional_range_probes"] = [file_record(path) for path in additional_range_paths]
        route_evidence["latest_additional_range_probe_statuses"] = [audit.get("status") for audit in additional_range_audits]
        route_evidence["latest_additional_range_probe_payload_downloaded"] = False
        ledger["latest_additional_range_probes"] = [file_record(path) for path in additional_range_paths]
        ledger["latest_additional_range_probe_statuses"] = [audit.get("status") for audit in additional_range_audits]
        ledger["latest_additional_range_probe_payload_downloaded"] = False

    for requirement in ledger.get("required_evidence", []):
        if isinstance(requirement, dict) and requirement.get("requirement") == "official processed-DMS payload or verified public route":
            requirement["status"] = "BLOCKED_HTTP_403_NO_2XX"
            requirement["evidence"] = str(route_path)
            requirement["scientific_gate_effect"] = "NO_PHASE_0_PASS"
            if range_path is not None and range_audit is not None:
                requirement["status"] = "BLOCKED_HTTP_403_RANGE_PROBE"
                requirement["evidence"] = str(range_path)
            if provenance_path is not None and provenance_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(provenance_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(provenance_path))
            if doi_provenance_path is not None and doi_provenance_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(doi_provenance_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(doi_provenance_path))
            if v8_path is not None and v8_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(v8_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(v8_path))
            if oai_format_path is not None and oai_format_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(oai_format_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(oai_format_path))
            if github_public_path is not None and github_public_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(github_public_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(github_public_path))
            if fastq_batch_path is not None and fastq_batch_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(fastq_batch_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(fastq_batch_path))
            for extra_path in (raw_fastq_range_path, downloader_control_path, chunked_fastq_path, final_fastq_path):
                if extra_path is not None:
                    requirement.setdefault("additional_evidence", [])
                    if str(extra_path) not in requirement["additional_evidence"]:
                        requirement["additional_evidence"].append(str(extra_path))
            if partial_size_path is not None and partial_size_audit is not None:
                requirement.setdefault("additional_evidence", [])
                if str(partial_size_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(partial_size_path))
            for additional_range_path in additional_range_paths:
                requirement.setdefault("additional_evidence", [])
                if str(additional_range_path) not in requirement["additional_evidence"]:
                    requirement["additional_evidence"].append(str(additional_range_path))

    ledger["status"] = "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE"
    ledger["primary_labels_admitted"] = False
    ledger["modeling_authorized"] = False
    ledger["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_dump(output, ledger)
    checked = load_json(output)
    print(json.dumps({
        "status": checked["status"],
        "latest_route_status": checked["latest_route_reprobe_status"],
        "output": str(output),
        "output_sha256": sha256(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
