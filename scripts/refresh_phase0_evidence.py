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
    parser.add_argument("--fastq-batch-audit", type=Path)
    args = parser.parse_args()

    batch_audit_data: dict[str, object] | None = None
    batch_audit_relative: str | None = None
    if args.fastq_batch_audit:
        batch_audit_path = args.fastq_batch_audit.resolve()
        artifact_root_resolved = args.artifact_root.resolve()
        try:
            batch_audit_relative = str(batch_audit_path.relative_to(artifact_root_resolved))
        except ValueError as exc:
            parser.error("--fastq-batch-audit must be under --artifact-root")
        if not batch_audit_path.is_file():
            parser.error(f"FASTQ batch audit does not exist: {batch_audit_path}")
        try:
            loaded_batch_audit = load(batch_audit_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            parser.error(f"FASTQ batch audit is not valid JSON: {batch_audit_path}")
        if not isinstance(loaded_batch_audit, dict):
            parser.error(f"FASTQ batch audit must be a JSON object: {batch_audit_path}")
        batch_audit_data = loaded_batch_audit

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

    if batch_audit_data is not None and batch_audit_relative is not None:
        append_unique(
            {
                "source_id": "deenalattha_2026_dms",
                "kind": "FASTQ_batch_payload_audit",
                **rel_artifact(
                    args.artifact_root,
                    batch_audit_relative,
                    status=batch_audit_data.get("status"),
                    selected_runs=batch_audit_data.get("selected_runs", []),
                    pending_run_count=batch_audit_data.get("pending_run_count"),
                    failed_run_count=batch_audit_data.get("failed_run_count"),
                    raw_sequence_content_emitted=False,
                    scientific_labels_admitted=False,
                    scientific_gate_effect="NO_PHASE_0_PASS",
                ),
            }
        )

    append_unique({"source_id": "denny_2018_tectorna", "kind": "subset_mapping_aggregate_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_subset_mapping_20260801T072000Z.json", log_path="phase0/audits/denny_subset_mapping_20260801T072000Z.log", status="AGGREGATE_COMPLETE_NO_ACCEPTED_COUNT_MAPPING", scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "PMC_BioC_article_metadata", **rel_artifact(args.artifact_root, "phase0/source_metadata/pmc_bioc_PMC6053692.json", status="PUBLIC_ARTICLE_METADATA_COMPLETE")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "PMC_semantics_term_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_pmc_semantics_20260801T082000Z.json", log_path="phase0/audits/denny_pmc_semantics_20260801T082000Z.log", status="TERM_AUDIT_COMPLETE_CENSOR_DIRECTION_NOT_ESTABLISHED", scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "XLSX_OOXML_structure_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_xlsx_ooxml_structure_20260801T170000Z.json", status="STRUCTURAL_AUDIT_COMPLETE_SEMANTICS_UNRESOLVED", raw_cell_values_read=False, raw_values_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "denny_2018_tectorna", "kind": "XLSX_semantic_evidence_audit", **rel_artifact(args.artifact_root, "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json", status="SEMANTIC_EVIDENCE_EXTRACTED_UPPER_BOUND_CONSISTENT_NOT_PROOF_REQUIRES_MANUAL_ACCEPTANCE", raw_values_emitted=False, sequence_values_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "phase0_matching", "kind": "manual_matching_acceptance_component", **rel_artifact(args.artifact_root, "phase0/audits/manual_matching_acceptance.json", raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
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
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Figshare_ndownloader_HEAD_headers_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.headers", status="BLOCKED_HTTP_403_PUBLIC_ROUTE_PRESERVED", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Figshare_ndownloader_HEAD_status_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.status", status="BLOCKED_HTTP_403_PUBLIC_ROUTE_PRESERVED", payload_downloaded=False)})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "Figshare_ndownloader_HEAD_stderr_probe", **rel_artifact(args.artifact_root, "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.stderr", status="BLOCKED_HTTP_403_PUBLIC_ROUTE_PRESERVED", payload_downloaded=False)})
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
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_payload_audit_complete_selected_run", **rel_artifact(args.artifact_root, "phase0/audits/SRR35766785_fastq_audit_20260801T095400Z.json", status="COMPLETE_SELECTED_RUN_PAIR_AUDIT", raw_sequence_content_emitted=False, scientific_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_paired_audit_run_provenance", **rel_artifact(args.artifact_root, "phase0/audits/SRR35766785_fastq_audit_20260801T095400Z.run.json", status="COMPLETE_POSTHOC_PROVENANCE_START_TIME_APPROXIMATE", raw_sequence_content_emitted=False, scientific_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_single_file_audit_run_provenance", **rel_artifact(args.artifact_root, "phase0/audits/SRR31402664_2_fastq_audit_20260801T092500Z.run.json", status="IN_PROGRESS_STABLE_PAYLOAD_INTEGRITY_AUDIT", raw_sequence_content_emitted=False, scientific_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_single_file_payload_audit", **rel_artifact(args.artifact_root, "phase0/audits/SRR31402664_2_fastq_audit_20260801T092500Z.json", status="COMPLETE_SINGLE_FILE_INTEGRITY_AUDIT_PAIRED_AUDIT_PENDING", raw_sequence_content_emitted=False, scientific_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_task", "status": "IN_PROGRESS", "download_pid": args.download_pid, "download_log": args.download_log, "output_root": "/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_sra/main_library", "selected_runs": ["SRR31402664", "SRR31402663", "SRR35766784", "SRR35766785", "SRR38259812"], "raw_sequence_content_emitted": False, "scientific_gate_effect": "NO_PHASE_0_PASS"})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_resume_wrapper_incident", **rel_artifact(args.artifact_root, "phase0/audits/resume_wrapper_incident_20260801T161500Z.json", status="CORRECTED_PRESERVED_PARTIAL_REQUIRES_INTEGRITY_REAUDIT", final_files_overwritten=False, partial_files_deleted=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_partial_size_anomaly", **rel_artifact(args.artifact_root, "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json", status="OBSERVED_DURING_ORIGINAL_TASK_REQUIRES_FINAL_INTEGRITY_AUDIT", partial_files_deleted=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_partial_size_regression", **rel_artifact(args.artifact_root, "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T111109Z.json", status="UNRESOLVED_PARTIAL_SIZE_REGRESSION_REQUIRES_FINAL_INTEGRITY_AUDIT", partial_files_deleted=False, final_files_overwritten=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_partial_size_regression_reobserved", **rel_artifact(args.artifact_root, "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T113210Z.json", status="REOBSERVED_UNRESOLVED_PARTIAL_SIZE_REGRESSION_REQUIRES_FINAL_INTEGRITY_AUDIT", partial_files_deleted=False, final_files_overwritten=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_resume_diagnostic", **rel_artifact(args.artifact_root, "phase0/audits/download_resume_diagnostic_SRR38259812_2_20260801T114255Z.json", status="ACTIVE_DOWNLOAD_TRANSPORT_INSTABILITY_DIAGNOSTIC_UNRESOLVED", partial_files_deleted=False, final_files_overwritten=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    append_unique({"source_id": "deenalattha_2026_dms", "kind": "FASTQ_download_partial_size_regression_third_observation", **rel_artifact(args.artifact_root, "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T115127Z.json", status="THIRD_PARTIAL_SIZE_REGRESSION_FROZEN_FOR_DIAGNOSTICS", partial_files_deleted=False, final_files_overwritten=False, raw_sequence_content_emitted=False, scientific_gate_effect="NO_PHASE_0_PASS")})
    inventory["required_next_evidence"] = sorted(set(inventory.get("required_next_evidence", [])) | {"complete selected ENA FASTQ download and batch hash/gzip/pair audit", "complete and record the stable SRR31402664_2 single-file integrity audit", "reconcile raw FASTQ to construct-level DMS counts, background, read-depth and QC semantics", "establish source-defined DMS treatment/background hierarchy before any matching", "produce and manually adjudicate an evidence-linked opaque matching table with at least 50 matched and 30 rejected-or-ambiguous cases"})
    dump(inventory_path, inventory)

    registry_path = manifests / "data_registry.json"
    registry = load(registry_path)
    registry["status"] = "PHASE_0_PUBLIC_FASTQ_PAYLOAD_AUDIT_IN_PROGRESS"
    registry["metadata_audit_status"] = "PARTIAL_METADATA_AND_PUBLIC_FASTQ_INVENTORY_NOT_PHASE0_PASS"
    registry.setdefault("phase0_evidence_updates", []).append({"run_id": args.run_id, "ena_inventory": "phase0/source_metadata/ena_fastq_inventory_20260801.json", "small_fastq_audit": "phase0/audits/junction_design_1_fastq_20260801T064900Z.json", "denny_subset_audit": "phase0/audits/denny_subset_mapping_20260801T072000Z.json", "scientific_gate_effect": "NO_PHASE_0_PASS"})
    if batch_audit_data is not None and batch_audit_relative is not None:
        registry["phase0_evidence_updates"][-1]["fastq_batch_audit"] = batch_audit_relative
    for source in registry.get("sources", []):
        if source.get("source_id") == "deenalattha_2026_dms":
            source["public_raw_fastq_route"] = "ENA_filereport_and_FASTQ_mirror"
            source["public_raw_fastq_inventory"] = "phase0/source_metadata/ena_fastq_inventory_20260801.json"
            source["raw_fastq_status"] = "PUBLIC_PAYLOAD_DOWNLOAD_IN_PROGRESS_CONSTRUCT_LEVEL_RECONCILIATION_PENDING"
            source["figshare_status"] = "BLOCKED_HTTP_403_DIRECT_AND_NDOWNLOADER_ROUTES_PRESERVED"
            source["figshare_ndownloader_probe"] = "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.status"
            source["license_status"] = "ARTICLE_LICENSE_REGISTERED_RAW_FASTQ_TERMS_AND_DATA_PAYLOAD_TERMS_NOT_YET_VERIFIED"
            source["download_failure_status"] = "PRESERVED_PARTIAL_FAILURES" if download_failures else "NO_PRESERVED_FAILURES_OBSERVED"
            source["download_failure_events"] = download_failures
            source["resume_wrapper_incident"] = "phase0/audits/resume_wrapper_incident_20260801T161500Z.json"
            source["resume_wrapper_status"] = "CORRECTED_FAIL_CLOSED_TESTED_WHILE_ORIGINAL_DOWNLOADER_ACTIVE"
            source["download_partial_size_anomaly_path"] = "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json"
            source["download_partial_size_anomaly_status"] = "OBSERVED_REQUIRES_FINAL_INTEGRITY_AUDIT"
            source["download_partial_size_regression_reobserved_path"] = "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T113210Z.json"
            source["download_partial_size_regression_reobserved_status"] = "REOBSERVED_UNRESOLVED_REQUIRES_FINAL_INTEGRITY_AUDIT"
            source["download_resume_diagnostic_path"] = "phase0/audits/download_resume_diagnostic_SRR38259812_2_20260801T114255Z.json"
            source["download_resume_diagnostic_status"] = "ACTIVE_TRANSPORT_INSTABILITY_UNRESOLVED"
            source["download_partial_size_regression_third_path"] = "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T115127Z.json"
            source["download_partial_size_regression_third_status"] = "THIRD_REGRESSION_FROZEN_FOR_DIAGNOSTICS"
            if batch_audit_data is not None and batch_audit_relative is not None:
                source["fastq_batch_audit_path"] = batch_audit_relative
                source["fastq_batch_audit_status"] = batch_audit_data.get("status")
        if source.get("source_id") == "denny_2018_tectorna":
            source["semantic_evidence_path"] = "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json"
            source["semantic_evidence_status"] = "EXTRACTED_24073_VARIANT_SUM_1713_CARDINALITY_MINUS7_1_UPPER_BOUND_CONSISTENT_NOT_PROOF"
    dump(registry_path, registry)

    acceptance_path = manifests / "acceptance_phase0.json"
    acceptance = load(acceptance_path)
    acceptance["status"] = "IN_PROGRESS_PUBLIC_FASTQ_PAYLOAD_AUDIT"
    acceptance["pass"] = False
    acceptance.setdefault("evidence_paths", [])
    for path in ("phase0/audits/denny_subset_mapping_20260801T072000Z.json", "phase0/audits/denny_subset_mapping_20260801T072000Z.log", "phase0/source_metadata/pmc_bioc_PMC6053692.json", "phase0/audits/denny_pmc_semantics_20260801T082000Z.json", "phase0/audits/denny_pmc_semantics_20260801T082000Z.log", "phase0/audits/denny_xlsx_ooxml_structure_20260801T170000Z.json", "phase0/audits/denny_xlsx_semantic_evidence_20260801T090500Z.json", "phase0/audits/manual_matching_acceptance.json", "phase0/source_metadata/ena_filereport_PRJNA1188187.tsv", "phase0/source_metadata/ena_fastq_manifest_PRJNA1188187.tsv", "phase0/source_metadata/ena_fastq_inventory_20260801.json", "phase0/source_metadata/license_registry_20260801.json", "phase0/source_metadata/pmc_bioc_PMC11601540.json", "phase0/source_metadata/pmc_oa_manifest_PMC11601540.xml", "phase0/source_metadata/pmc_oa_package_PMC11601540.probe", "phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json", "phase0/source_metadata/dms_github_main_commit_20260801T190000Z.txt", "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.headers", "phase0/source_metadata/figshare_data_direct_get_20260801T091500Z.probe", "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.headers", "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.status", "phase0/source_metadata/figshare_ndownloader_data_20260801T091600Z.stderr", "phase0/audits/SRR31402664_2_fastq_audit_20260801T092500Z.run.json", "phase0/audits/SRR31402664_2_fastq_audit_20260801T092500Z.json", "phase0/source_metadata/pmc11601540_article_page_20260801T111500Z.html", "phase0/source_metadata/pmc11601540_supplement_links_20260801T111500Z.tsv", "phase0/source_metadata/pmc_media-1_20260801T113000Z.body", "phase0/source_metadata/pmc_media-1_20260801T113000Z.headers", "phase0/source_metadata/pmc_media-1_20260801T113000Z.status", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.body", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.headers", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.status", "phase0/source_metadata/zenodo_16884332_probe_20260801T122000Z.stderr", "phase0/audits/junction_design_1_fastq_20260801T064900Z.json", "phase0/audits/junction_design_1_fastq_20260801T064900Z.log", "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.json", "phase0/audits/SRR35766784_fastq_audit_20260801T130000Z.log", "phase0/audits/SRR35766785_fastq_audit_20260801T095400Z.json", "phase0/audits/SRR35766785_fastq_audit_20260801T095400Z.run.json", "phase0/audits/resume_wrapper_incident_20260801T161500Z.json", "phase0/audits/download_partial_size_anomaly_20260801T084716Z.json", "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T111109Z.json"):
        if path not in acceptance["evidence_paths"]:
            acceptance["evidence_paths"].append(path)
    if batch_audit_relative is not None and batch_audit_relative not in acceptance["evidence_paths"]:
        acceptance["evidence_paths"].append(batch_audit_relative)
    if "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T113210Z.json" not in acceptance["evidence_paths"]:
        acceptance["evidence_paths"].append("phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T113210Z.json")
    if "phase0/audits/download_resume_diagnostic_SRR38259812_2_20260801T114255Z.json" not in acceptance["evidence_paths"]:
        acceptance["evidence_paths"].append("phase0/audits/download_resume_diagnostic_SRR38259812_2_20260801T114255Z.json")
    if "phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T115127Z.json" not in acceptance["evidence_paths"]:
        acceptance["evidence_paths"].append("phase0/audits/download_partial_size_regression_SRR38259812_2_20260801T115127Z.json")
    acceptance["note"] = "Public ENA file-level metadata and one complete paired FASTQ run are now audited. The main DMS payload download and construct-level raw/background/read-depth reconciliation remain incomplete; Phase 0 stays fail-closed."
    if download_failures:
        acceptance["note"] += " At least one selected ENA transfer has a preserved partial failure; safe resume and re-audit are required before payload completion."
    acceptance["note"] += " A wrapper process-detection incident was preserved and corrected; the corrected wrapper was tested to block while the original downloader remained active."
    acceptance["note"] += " Denny semantic evidence extraction found an explicit 24,073 variant-count sum, a 1,713 numeric cardinality candidate, and -7.1 behavior consistent with an upper-bound cap in seven ΔG columns; this is not directional proof, and exact 1,687/1,713/1,636 subset mapping and censor/raw-interpolated acceptance remain unresolved."
    acceptance["note"] += " A read-only size anomaly was observed for an active SRR31402664 partial; no repair was attempted and final size/hash/gzip/pair audit remains required."
    acceptance["note"] += " The manual matching acceptance component is explicitly tracked fail-closed; no opaque matching table or manually adjudicated rows are admitted until evidence-linked records are available."
    acceptance["note"] += " The official processing README Figshare ndownloader HEAD probe returned HTTP 403; no data payload was downloaded and no access control was bypassed."
    acceptance["note"] += " A stable SRR31402664_2 single-file integrity audit was started with immutable run provenance; its final result is recorded separately and does not unlock Phase 0."
    acceptance["note"] += " The stable SRR31402664_2 single-file audit completed with size/hash/gzip/FASTQ structural checks and zero malformed records; no paired-read audit was performed for this single mate, so Phase 0 remains fail-closed."
    acceptance["note"] += " An unresolved SRR38259812_2 partial-size regression was observed after an earlier larger read-only observation; the partial remains preserved and requires final size/hash/gzip/pair audit, with no inference of cause."
    acceptance["note"] += " A second read-only SRR38259812_2 observation found the active partial materially smaller than its prior observation; the new anomaly is preserved independently, with no cause, repair, deletion, overwrite, or scientific interpretation inferred."
    acceptance["note"] += " Read-only inspection also preserved the active curl resume command and repeated transport return code 56 events; this is transport evidence only, with no cause or scientific interpretation inferred."
    acceptance["note"] += " A third partial-size regression was frozen after a guarded SIGSTOP of the two project download processes; no process was killed, no partial was deleted, and no cause or scientific interpretation is inferred."
    acceptance["note"] += " SRR35766785 is audited separately as a complete paired run when its batch artifact is present; this remains file-integrity evidence only and does not establish construct-level DMS labels or unlock Phase 0."
    if batch_audit_data is not None:
        acceptance["note"] += f" The selected-run FASTQ batch audit was registered with status {batch_audit_data.get('status')}; it is file-integrity evidence only and cannot unlock Phase 0 without construct-level DMS reconciliation and manual matching acceptance."
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
    blocker = "The manual matching acceptance component is missing or blocked; no candidate correspondence may be treated as accepted until an evidence-linked opaque table has at least 50 manually matched and 30 rejected-or-ambiguous cases with agreement at least 0.95."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "The official processing README Figshare ndownloader route returned HTTP 403; the data.zip payload remains unavailable through the recorded public route and no bypass was attempted."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "SRR31402664_2 passed a single-file size/hash/gzip/FASTQ structural audit with zero malformed records, but it is one mate only; paired-read and all-selected-run audits remain pending, so this evidence cannot unlock Phase 0."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "SRR38259812_2 showed an unresolved partial-size regression between read-only observations while its downloader remained active; the preserved partial requires final size/hash/gzip/pair audit, and no cause or scientific meaning may be inferred."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "A second read-only SRR38259812_2 observation found the active partial materially smaller than the prior observation; the new anomaly is preserved independently and requires final size/hash/gzip/pair audit, with no cause or scientific meaning inferred."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "The active SRR38259812_2 curl transfer showed repeated transport return code 56 events while no terminal event was recorded; this diagnostic is unresolved and requires final terminal-state integrity audit, with no cause or scientific meaning inferred."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "A third SRR38259812_2 partial-size regression was frozen after a guarded SIGSTOP of the two project download processes; the original partial remains evidence-only and requires a separately audited recovery path."
    if blocker not in blockers:
        blockers.append(blocker)
    blocker = "SRR35766785 paired FASTQ audit, when present, covers only one selected run; all selected payloads, raw/background/read-depth reconciliation, and manual matching remain required before Phase 0 can pass."
    if blocker not in blockers:
        blockers.append(blocker)
    if batch_audit_data is not None:
        batch_status = batch_audit_data.get("status")
        if batch_status == "BATCH_COMPLETE":
            blocker = "The selected-run FASTQ batch audit completed file-integrity checks, but this evidence does not establish construct-level DMS semantics, source-defined background/read-depth hierarchy, or manual matching acceptance; Phase 0 remains locked."
        else:
            blocker = f"The selected-run FASTQ batch audit is {batch_status}; pending or failed payload integrity must be resolved before any Phase 0 consideration, and this audit cannot unlock the gate."
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
    batch_note = (
        f"- Selected-run FASTQ batch audit: `{batch_audit_data.get('status')}`; this is file-integrity evidence only and does not unlock Phase 0.\n\n"
        if batch_audit_data is not None
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
        "- Manual matching acceptance is explicitly fail-closed and tracked as a separate component; no evidence-linked opaque audit table has been admitted.\n\n"
        "- The official processing README Figshare ndownloader HEAD probe returned HTTP 403; the data.zip payload was not downloaded and no access control was bypassed.\n\n"
        "- SRR31402664_2 passed a single-file size/hash/gzip/FASTQ structural audit with zero malformed records; paired-read and all-selected-run audits remain pending.\n\n"
        "- An unresolved SRR38259812_2 partial-size regression was preserved as a separate anomaly artifact; no cause or scientific meaning is inferred, and final integrity audit remains mandatory.\n\n"
        "- A second SRR38259812_2 partial-size regression was independently preserved after a later read-only observation; no cause, repair, deletion, overwrite, or scientific meaning is inferred.\n\n"
        "- Read-only transport diagnostics preserved the active curl resume options and repeated return code 56 events; no cause or scientific meaning is inferred, and terminal-state integrity audit remains mandatory.\n\n"
        "- A third partial-size regression was frozen after a guarded SIGSTOP of the two project download processes; the original partial remains preserved and a separately audited chunked recovery path is required.\n\n"
        "- SRR35766785 is independently audited as a paired run when its batch artifact is present; this remains file-integrity evidence only and does not establish construct-level DMS labels.\n\n"
        + batch_note
        + failure_note
        + "## Gate\n\n"
        "`PHASE_0 = IN_PROGRESS`; `scientific_gate_effect = NO_PHASE_0_PASS`; `primary_labels_admitted = false`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PHASE0_MANIFESTS_REFRESHED_FAIL_CLOSED", "run_id": args.run_id, "inventory": str(inventory_path), "report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
