#!/usr/bin/env python3
"""Refresh Phase 0 governance manifests from append-only evidence artifacts.

The script backs up each current manifest before updating it. It never changes
the contract, matching rows, scientific labels, or phase-unlock state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTRACT_SHA256 = "218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel_artifact(artifact_root: Path, relative_path: str, **extra: object) -> dict:
    path = artifact_root / relative_path
    result: dict[str, object] = {"relative_path": relative_path, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}
    if path.is_file() and path.stat().st_size < 20 * 1024 * 1024:
        result["sha256"] = sha256(path)
    result.update(extra)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--download-pid", required=True)
    parser.add_argument("--download-log", required=True)
    args = parser.parse_args()

    manifests = args.code_root / "manifests"
    history = manifests / "history"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("phase0_payload_inventory.json", "data_registry.json", "acceptance_phase0.json", "phase_status.json"):
        current = manifests / name
        backup = history / f"{name.removesuffix('.json')}_{args.run_id}.json"
        if not backup.exists():
            shutil.copy2(current, backup)

    inventory_path = manifests / "phase0_payload_inventory.json"
    inventory = load(inventory_path)
    inventory["inventory_id"] = f"PHASE0_PAYLOAD_INVENTORY_{args.run_id}"
    inventory["status"] = "IN_PROGRESS_PUBLIC_FASTQ_PAYLOAD_DOWNLOAD_AUDIT"
    inventory["last_refresh_utc"] = datetime.now(timezone.utc).isoformat()
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["primary_labels_admitted"] = False
    inventory.setdefault("artifacts", [])
    download_failures: list[dict[str, object]] = []
    try:
        for raw_line in Path(args.download_log).read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_line.startswith("{"):
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("status") in {"DOWNLOAD_FAILED_PARTIAL_PRESERVED", "DOWNLOAD_SIZE_MISMATCH_PARTIAL_PRESERVED"}:
                download_failures.append(
                    {
                        "run": event.get("run"),
                        "path": event.get("path"),
                        "expected_bytes": event.get("expected_bytes"),
                        "observed_bytes": event.get("observed_bytes"),
                        "returncode": event.get("returncode"),
                        "status": event.get("status"),
                    }
                )
    except OSError:
        download_failures = []
    inventory["download_failure_events"] = download_failures
    if download_failures:
        inventory["status"] = "IN_PROGRESS_PUBLIC_FASTQ_PAYLOAD_DOWNLOAD_WITH_PRESERVED_FAILURES"
        inventory.setdefault("required_next_evidence", [])
        inventory["required_next_evidence"] = sorted(
            set(inventory["required_next_evidence"])
            | {
                "safe resume of every preserved partial download, followed by final size/hash/gzip/pair audit"
            }
        )

    def append_unique(entry: dict) -> None:
        key = (entry.get("relative_path"), entry.get("kind"), entry.get("source_id"))
        if not any((item.get("relative_path"), item.get("kind"), item.get("source_id")) == key for item in inventory["artifacts"]):
            inventory["artifacts"].append(entry)

    append_unique({"source_id": "denny_2018_tectorna", "kind": "subset_mapping_aggregate_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_subset_mapping_20260801T072000Z.json", log_path="phase0/audits/denny_subset_mapping_20260801T072000Z.log", status="AGGREGATE_COMPLETE_NO_ACCEPTED_COUNT_MAPPING", scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "PMC_BioC_article_metadata", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_bioc_PMC6053692.json", status="PUBLIC_ARTICLE_METADATA_COMPLETE")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "PMC_semantics_term_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_pmc_semantics_20260801T082000Z.json", log_path="phase0/audits/denny_pmc_semantics_20260801T082000Z.log", status="TERM_AUDIT_COMPLETE_CENSOR_DIRECTION_NOT_ESTABLISHED", scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "XLSX_OOXML_structure_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_xlsx_ooxml_structure_20260801T170000Z.json", status="STRUCTURAL_AUDIT_COMPLETE_SEMANTICS_UNRESOLVED", raw_cell_values_read=False, raw_values_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "XLSX_semantic_evidence_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json", status="SEMANTIC_EVIDENCE_EXTRACTED_UPPER_BOUND_CONSISTENT_NOT_PROOF_REQUIRES_MANUAL_ACCEPTANCE", raw_values_emitted=False, sequence_values_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "ENA_file_report", **rel_artifact(args.artifact_root, "phase0/source_metadata/ena_filereport_PRJNA1188187.tsv", status="PUBLIC_FILE_LEVEL_METADATA_COMPLETE")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "ENA_FASTQ_manifest", **rel_artifact(args.artifact_root, "phase0/source_metadata/ena_fastq_manifest_PRJNA1188187.tsv", status="PUBLIC_FILE_LEVEL_MANIFEST_COMPLETE")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "ENA_FASTQ_inventory", **rel_artifact(args.artifact_root, "phase0/source_metadata/ena_fastq_inventory_20260801.json", status="PUBLIC_FASTQ_INVENTORY_COMPLETE", compressed_file_count=30, compressed_total_bytes=95123388656)})
    append_unique({"source_id": "phase0_source_registry", "kind": "license_public_access_registry", **rel_artifact(args.artifact_root, "phase0/source_metadata/license_registry_20260801.json", status="SOURCE_LICENSE_AND_PUBLIC_ACCESS_AUDIT_REGISTERED")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_BioC_article_metadata", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_bioc_PMC11601540.json", status="PUBLIC_ARTICLE_METADATA_COMPLETE")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_OA_manifest", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_oa_manifest_PMC11601540.xml", status="PUBLIC_OA_MANIFEST_REGISTERED")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_OA_package_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_oa_package_PMC11601540.probe", status="BLOCKED_HTTP_404_NO_SUPPLEMENT_PAYLOAD", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "public_processing_source_code_registry", **rel_artifact(args.artifact_root, "phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json", status="PUBLIC_SOURCE_CODE_SEMANTICS_REGISTERED_PRIMARY_PAYLOAD_NOT_ADMITTED", raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "public_processing_source_code_commit_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/dms_github_main_commit_20260801T190000Z.txt", status="PUBLIC_SOURCE_CODE_COMMIT_PINNED", raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Figshare_direct_GET_headers_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.headers", status="BLOCKED_HTTP_403", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Figshare_direct_GET_body_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.probe", status="BLOCKED_HTTP_403", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_article_page", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc11601540_article_page_20260801T111500Z.html", status="PUBLIC_ARTICLE_PAGE_METADATA_COMPLETE", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_supplement_link_inventory", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc11601540_supplement_links_20260801T111500Z.tsv", status="PUBLIC_SUPPLEMENT_LINK_INVENTORY_COMPLETE", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_supplement_download_body_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_media-1_20260801T113000Z.body", status="BLOCKED_HTML_POW_CHALLENGE_NOT_DOCX", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_supplement_download_headers_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_media-1_20260801T113000Z.headers", status="BLOCKED_HTTP_200_HTML_CHALLENGE", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "PMC_supplement_download_status_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_media-1_20260801T113000Z.status", status="BLOCKED_HTTP_200_HTML_CHALLENGE", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Zenodo_code_archive_route_body_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.body", status="BLOCKED_CONNECTION_REFUSED_HTTP_000", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Zenodo_code_archive_route_headers_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.headers", status="BLOCKED_CONNECTION_REFUSED_HTTP_000", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Zenodo_code_archive_route_status_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.status", status="BLOCKED_CONNECTION_REFUSED_HTTP_000", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Zenodo_code_archive_route_stderr_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.stderr", status="BLOCKED_CONNECTION_REFUSED_HTTP_000", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_payload_audit", **rel_artifact(args.artifact_root, "phase0/audits/junction_design_1_fastq_20260801T064900Z.json", log_path="phase0/audits/junction_design_1_fastq_20260801T064900Z.log", status="COMPLETE_ONE_PUBLIC_PAIRED_RUN", raw_sequence_content_emitted=False, scientific_labels_admitted=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_payload_audit_complete_selected_run", **rel_artifact(args.artifact_root, "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.json", log_path="phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.log", status="COMPLETE_SELECTED_RUN_PAIR_AUDIT", raw_sequence_content_emitted=False, scientific_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_task", "status": "IN_PROGRESS", "download_pid": args.download_pid, "download_log": args.download_log, "output_root": "/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_sra/main_library", "selected_runs": ["SRR31402664", "SRR31402663", "SRR35766784", "SRR35766785", "SRR38259812"], "raw_sequence_content_emitted": False, "scientific_gate_effect": "NO_PHASE_0_PASS"})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_resume_wrapper_incident", **rel_artifact(args.artifact_root, "phase0/audits/resume_wrapper_incident_20260801T161500Z.json", status="CORRECTED_PRESERVED_PARTIAL_REQUIRES_INTEGRITY_REAUDIT", final_files_overwritten=False, partial_files_deleted=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_partial_size_anomaly", **rel_artifact(args.artifact_root, "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json", status="OBSERVED_DURING_ORIGINAL_TASK_REQUIRES_FINAL_INTEGRITY_AUDIT", partial_files_deleted=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    inventory["required_next_evidence"] = sorted(set(inventory.get("required_next_evidence", [])) | {"complete selected ENA FASTQ download and batch hash/gzip/pair audit", "reconcile raw FASTQ to construct-level DMS counts, background, read-depth and QC semantics", "establish source-defined DMS treatment/background hierarchy before any matching"})
    dump(inventory_path, inventory)

    registry_path = manifests / "data_registry.json"
    registry = load(registry_path)
    registry["status"] = "PHASE_0_PUBLIC_FASTQ_PAYLOAD_AUDIT_IN_PROGRESS"
    registry["metadata_audit_status"] = "PARTIAL_METADATA_AND_PUBLIC_FASTQ_INVENTORY_NOT_PHASE0_PASS"
    registry.setdefault("phase0_evidence_updates", []).append({"run_id": args.run_id, "ena_inventory": "phase0/source_metadata/ena_fastq_inventory_20260801.json", "small_fastq_audit": "phase0/audits/junction_design_1_fastq_20260801T064900Z.json", "denny_subset_audit": "phase0/audits/denny_subset_mapping_20260801T072000Z.json", "scientific_gate_effect": "NO_PHASE_0_PASS"})
    for source in registry.get("sources", []):
        if source.get("source_id") == "deenalattha_2026_dms":
            source["public_raw_fastq_route"] = "ENA_filereport_and_FASTQ_mirror"
            source["public_raw_fastq_inventory"] = "phase0/source_metadata/ena_fastq_inventory_20260801.json"
            source["raw_fastq_status"] = "PUBLIC_PAYLOAD_DOWNLOAD_IN_PROGRESS_CONSTRUCT_LEVEL_RECONCILIATION_PENDING"
            source["figshare_status"] = "BLOCKED_HTTP_403_PRESERVED"
            source["license_status"] = "ARTICLE_LICENSE_REGISTERED_RAW_FASTQ_TERMS_AND_DATA_PAYLOAD_TERMS_NOT_YET_VERIFIED"
            source["download_failure_status"] = "PRESERVED_PARTIAL_FAILURES" if download_failures else "NO_PRESERVED_FAILURES_OBSERVED"
            source["download_failure_events"] = download_failures
            source["resume_wrapper_incident"] = "phase0/audits/resume_wrapper_incident_20260801T161500Z.json"
            source["resume_wrapper_status"] = "CORRECTED_FAIL_CLOSED_TESTED_WHILE_ORIGINAL_DOWNLOADER_ACTIVE"
            source["download_partial_size_anomaly_path"] = "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json"
            source["download_partial_size_anomaly_status"] = "OBSERVED_REQUIRES_FINAL_INTEGRITY_AUDIT"
        if source.get("source_id") == "denny_2018_tectorna":
            source["semantic_evidence_path"] = "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json"
            source["semantic_evidence_status"] = "EXTRACTED_24073_VARIANT_SUM_1713_CARDINALITY_MINUS7_1_UPPER_BOUND_CONSISTENT_NOT_PROOF"
    dump(registry_path, registry)

    acceptance_path = manifests / "acceptance_phase0.json"
    acceptance = load(acceptance_path)
    acceptance["status"] = "IN_PROGRESS_PUBLIC_FASTQ_PAYLOAD_AUDIT"
    acceptance["pass"] = False
    acceptance.setdefault("evidence_paths", [])
    for path in ("phase0/audits/denny_subset_mapping_20260801T072000Z.json", "phase0/audits/denny_subset_mapping_20260801T072000Z.log", "phase0/source_metadata/pmc_bioc_PMC6053692.json", "phase0/audits/denny_pmc_semantics_20260801T082000Z.json", "phase0/audits/denny_pmc_semantics_20260801T082000Z.log", "phase0/audits/denny_xlsx_ooxml_structure_20260801T170000Z.json", "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json", "phase0/source_metadata/ena_filereport_PRJNA1188187.tsv", "phase0/source_metadata/ena_fastq_manifest_PRJNA1188187.tsv", "phase0/source_metadata/ena_fastq_inventory_20260801.json", "phase0/source_metadata/license_registry_20260801.json", "phase0/source_metadata/pmc_bioc_PMC11601540.json", "phase0/source_metadata/pmc_oa_manifest_PMC11601540.xml", "phase0/source_metadata/pmc_oa_package_PMC11601540.probe", "phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json", "phase0/source_metadata/dms_github_main_commit_20260801T190000Z.txt", "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.headers", "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.probe", "phase0/source_metadata/pmc11601540_article_page_20260801T111500Z.html", "phase0/source_metadata/pmc11601540_supplement_links_20260801T111500Z.tsv", "phase0/source_metadata/pmc_media-1_20260801T113000Z.body", "phase0/source_metadata/pmc_media-1_20260801T113000Z.headers", "phase0/source_metadata/pmc_media-1_20260801T113000Z.status", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.body", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.headers", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.status", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.stderr", "phase0/audits/junction_design_1_fastq_20260801T064900Z.json", "phase0/audits/junction_design_1_fastq_20260801T064900Z.log", "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.json", "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.log", "phase0/audits/resume_wrapper_incident_20260801T161500Z.json", "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json"):
        if path not in acceptance["evidence_paths"]:
            acceptance["evidence_paths"].append(path)
    acceptance["note"] = "Public ENA file-level metadata and one complete paired FASTQ run are now audited. The main DMS payload download and construct-level raw/background/read-depth reconciliation remain incomplete; Phase 0 stays fail-closed."
    if download_failures:
        acceptance["note"] += " At least one selected ENA transfer has a preserved partial failure; safe resume and re-audit are required before payload completion."
    acceptance["note"] += " A wrapper process-detection incident was preserved and corrected; the corrected wrapper was tested to block while the original downloader remained active."
    acceptance["note"] += " Denny semantic evidence extraction found an explicit 24,073 variant-count sum, a 1,713 numeric cardinality candidate, and -7.1 behavior consistent with an upper-bound cap in seven ΔG columns; this is not directional proof, and exact 1,687/1,713/1,636 subset mapping and censor/raw-interpolated acceptance remain unresolved."
    acceptance["note"] += " A read-only size anomaly was observed for an active SRR31402664 partial; no repair was attempted and final size/hash/gzip/pair audit remains required."
    dump(acceptance_path, acceptance)

    phase_path = manifests / "phase_status.json"
    phase = load(phase_path)
    phase["last_transition"] = datetime.now(timezone.utc).isoformat()
    phase["transition_evidence"] = "manifests/phase0_payload_inventory.json"
    blockers = phase.setdefault("blocking_conditions", [])
    additions = [
        "Public ENA file-level metadata is registered and one small paired FASTQ run passes integrity audit, but the selected main DMS FASTQ payload remains in progress.",
        "Raw FASTQ availability does not by itself establish construct-level DMS counts, treated/background hierarchy, read-depth QC, or primary labels.",
        "A PMC OA manifest was available, but the advertised OA package URL returned HTTP 404; no supplement was substituted or bypassed.",
        "The Denny subset audit confirms aggregate counts only and explicitly does not accept a 1687/1713/1636 count mapping.",
        "The official PMC media-1.docx link returned HTTP 200 text/html with a Preparing to download/POW challenge rather than a DOCX; no challenge was bypassed.",
        "The official Zenodo code-archive route probe exited 7 with HTTP status 000 due connection refusal; record contents remain unverified and this is not treated as evidence that data are absent.",
    ]
    for blocker in additions:
        if blocker not in blockers:
            blockers.append(blocker)
    if download_failures:
        blocker = "A selected ENA transfer logged DOWNLOAD_FAILED_PARTIAL_PRESERVED; the partial file was retained and requires safe resume plus final hash/gzip/pair audit."
        if blocker not in blockers:
            blockers.append(blocker)
    blocker = "A preserved wrapper incident affected one partial-transfer provenance; the partial remains unverified until a post-download isolated integrity re-audit passes."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "Denny semantic workbook evidence extracted a 24,073 variant-count sum, a 1,713 numeric cardinality candidate, and -7.1 distributions consistent with an upper-bound cap in seven ΔG columns, but this is not directional proof and does not establish the contract's 1,687/1,713/1,636 mapping or censor/raw-interpolated semantics."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "A read-only metadata anomaly showed materially different observed sizes for the active SRR31402664 partial; its final payload must pass size/hash/gzip/pair audit before any acceptance."
    if blocker not in blockers:
        blockers.append(blocker)
    dump(phase_path, phase)

    report = args.code_root / "reports" / f"phase0_payload_inventory_{args.run_id}.md"
    failure_note = (
        "- Download failure evidence: "
        + "; ".join(
            f"{item.get('run')} returncode={item.get('returncode')} partial preserved"
            for item in download_failures
        )
        + ". Safe resume is required.\n"
        if download_failures
        else ""
    )
    report.write_text(
        f"# Phase 0 payload inventory refresh ({args.run_id})\n\n"
        "This refresh is governance evidence only. It does not unlock Phase 0 or admit primary labels.\n\n"
        "- Public ENA file-level inventory: 15 runs, 30 paired FASTQ files, 95,123,388,656 compressed bytes.\n"
        "- One small paired run passed hash/gzip/record/pair-ID audit; sequence content was not emitted.\n"
        "- Main DMS/nomod/denature/37C download is still in progress under the recorded PID/log.\n"
        "- Denny subset audit found an explicit variant-count sum of 24,073 and a candidate field with 1,713 distinct values; the contract's 1,687/1,713/1,636 mapping is unresolved.\n"
        "- Figshare HTTP 403, OUP access challenge, and PMC package 404 are preserved as access evidence; no access control was bypassed.\n\n"
        "- The official PMC media-1.docx route returned HTTP 200 text/html with a POW challenge rather than a DOCX; the Zenodo route probe returned curl exit 7/HTTP 000 and remains unverified.\n\n"
        "- A dependency-free Denny XLSX OOXML structure audit completed without decoding any cell values; semantic count/censor/matching evidence remains unresolved.\n\n"
        "- Official DMS processing source code was pinned to a public Git commit and its field semantics were registered; no source-code field was admitted as a primary label.\n\n"
        "- One selected main-library paired FASTQ run passed hash/gzip/record/pair-ID audit; this is file-integrity evidence only and does not establish construct-level DMS labels or QC hierarchy.\n\n"
        "- A resume-wrapper process-detection incident was preserved with no final-file overwrite or partial deletion; the corrected wrapper now blocks while the original downloader is active. The affected partial remains unverified.\n\n"
        "- Dependency-free Denny semantic evidence extracted a 24,073 variant-count sum, a 1,713 numeric cardinality candidate, measured/interpolated 9/10/11-bp ΔG headers, and -7.1 distributions consistent with an upper-bound cap in seven ΔG columns; this is not directional proof, and exact subset mapping, censor direction, and accepted raw/interpolated semantics remain unresolved.\n\n"
        "- A read-only metadata anomaly observed materially different sizes for an active SRR31402664 partial; no repair or deletion was attempted, and final size/hash/gzip/pair audit remains mandatory.\n\n"
        + failure_note
        + "## Gate\n\n"
        "`PHASE_0 = IN_PROGRESS`; `scientific_gate_effect = NO_PHASE_0_PASS`; `primary_labels_admitted = false`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PHASE0_MANIFESTS_REFRESHED_FAIL_CLOSED", "run_id": args.run_id, "inventory": str(inventory_path), "report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
