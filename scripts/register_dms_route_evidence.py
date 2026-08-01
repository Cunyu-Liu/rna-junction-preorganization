#!/usr/bin/env python3
"""Register a new blocked DMS route audit and ledger in Phase 0 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def dump_atomic(path: Path, value: dict) -> None:
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


def relative_artifact(artifact_root: Path, path: Path, **extra: object) -> dict:
    relative = str(path.resolve().relative_to(artifact_root.resolve()))
    result: dict[str, object] = {
        "relative_path": relative,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }
    result.update(extra)
    return result


def append_unique(items: list[dict], entry: dict) -> None:
    key = (entry.get("relative_path"), entry.get("kind"), entry.get("source_id"))
    if not any((item.get("relative_path"), item.get("kind"), item.get("source_id")) == key for item in items):
        items.append(entry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--route-audit", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--github-tree", type=Path)
    parser.add_argument("--zenodo-audit", type=Path)
    parser.add_argument("--range-audit", type=Path)
    parser.add_argument("--metadata-audit", type=Path)
    parser.add_argument("--provenance-audit", type=Path)
    parser.add_argument("--doi-provenance-audit", type=Path)
    parser.add_argument("--v8-audit", type=Path)
    parser.add_argument("--oai-format-audit", type=Path)
    parser.add_argument("--github-public-audit", type=Path)
    parser.add_argument("--fastq-batch-audit", type=Path)
    parser.add_argument("--additional-range-audit", action="append", type=Path, default=[])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    artifact_root = args.artifact_root.resolve()
    route_path = args.route_audit.resolve()
    ledger_path = args.ledger.resolve()
    github_tree_path = args.github_tree.resolve() if args.github_tree else None
    zenodo_audit_path = args.zenodo_audit.resolve() if args.zenodo_audit else None
    range_audit_path = args.range_audit.resolve() if args.range_audit else None
    metadata_audit_path = args.metadata_audit.resolve() if args.metadata_audit else None
    provenance_audit_path = args.provenance_audit.resolve() if args.provenance_audit else None
    doi_provenance_audit_path = args.doi_provenance_audit.resolve() if args.doi_provenance_audit else None
    v8_audit_path = args.v8_audit.resolve() if args.v8_audit else None
    oai_format_audit_path = args.oai_format_audit.resolve() if args.oai_format_audit else None
    github_public_audit_path = args.github_public_audit.resolve() if args.github_public_audit else None
    fastq_batch_audit_path = args.fastq_batch_audit.resolve() if args.fastq_batch_audit else None
    additional_range_audit_paths = [path.resolve() for path in args.additional_range_audit]
    route = load(route_path)
    ledger = load(ledger_path)
    if route.get("status") != "ROUTE_REPROBE_BLOCKED_NO_2XX" or route.get("payload_downloaded") is not False:
        raise SystemExit("route audit is not a blocked no-payload result")
    if ledger.get("status") != "BLOCKED_PHASE0_DMS_PAYLOAD_UNAVAILABLE" or ledger.get("primary_labels_admitted") is not False:
        raise SystemExit("ledger is not fail-closed")
    github_tree = None
    if github_tree_path is not None:
        github_tree = load(github_tree_path)
        if github_tree.get("truncated") is not False or not isinstance(github_tree.get("tree"), list):
            raise SystemExit("GitHub tree metadata is incomplete or malformed")
    zenodo_audit = None
    if zenodo_audit_path is not None:
        zenodo_audit = load(zenodo_audit_path)
        if zenodo_audit.get("status") != "BLOCKED_ZENODO_API_CONNECTION_REFUSED_HTTP_000" or zenodo_audit.get("payload_downloaded") is not False:
            raise SystemExit("Zenodo audit is not a blocked no-payload result")
    range_audit = None
    if range_audit_path is not None:
        range_audit = load(range_audit_path)
        if range_audit.get("status") != "BLOCKED_HTTP_403_RANGE_PROBE" or range_audit.get("payload_downloaded") is not False or range_audit.get("observed_body_bytes") != 0:
            raise SystemExit("range audit is not a blocked zero-byte result")
    metadata_audit = None
    if metadata_audit_path is not None:
        metadata_audit = load(metadata_audit_path)
        if metadata_audit.get("status") != "BLOCKED_HTTP_403_FILE_METADATA_PROBE" or metadata_audit.get("payload_downloaded") is not False or metadata_audit.get("observed_body_bytes") != 0:
            raise SystemExit("metadata audit is not a blocked zero-byte result")
    provenance_audit = None
    if provenance_audit_path is not None:
        provenance_audit = load(provenance_audit_path)
        if provenance_audit.get("status") not in {"BLOCKED_NO_2XX_PROVENANCE_ROUTE", "PROVENANCE_HEAD_ONLY_METADATA_BLOCKED"}:
            raise SystemExit("provenance audit is not a blocked no-payload result")
        if provenance_audit.get("payload_downloaded") is not False or provenance_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("provenance audit is not fail-closed")
    doi_provenance_audit = None
    if doi_provenance_audit_path is not None:
        doi_provenance_audit = load(doi_provenance_audit_path)
        if doi_provenance_audit.get("status") not in {"DOI_OAI_PROVENANCE_METADATA_AVAILABLE", "BLOCKED_NO_2XX_DOI_OAI_PROVENANCE_ROUTE"}:
            raise SystemExit("DOI/OAI provenance audit has an unexpected status")
        if doi_provenance_audit.get("payload_downloaded") is not False or doi_provenance_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("DOI/OAI provenance audit is not fail-closed")
    v8_audit = None
    if v8_audit_path is not None:
        v8_audit = load(v8_audit_path)
        if v8_audit.get("status") not in {"FIGSHARE_V8_ROUTE_METADATA_AVAILABLE", "BLOCKED_NO_2XX_FIGSHARE_V8_ROUTE"}:
            raise SystemExit("Figshare v8 audit has an unexpected status")
        if v8_audit.get("payload_downloaded") is not False or v8_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("Figshare v8 audit is not fail-closed")
    oai_format_audit = None
    if oai_format_audit_path is not None:
        oai_format_audit = load(oai_format_audit_path)
        if oai_format_audit.get("status") not in {
            "FIGSHARE_OAI_FORMAT_METADATA_AVAILABLE",
            "BLOCKED_NO_2XX_FIGSHARE_OAI_FORMAT_ROUTE",
        }:
            raise SystemExit("Figshare OAI format audit has an unexpected status")
        if oai_format_audit.get("payload_downloaded") is not False or oai_format_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("Figshare OAI format audit is not fail-closed")
    github_public_audit = None
    if github_public_audit_path is not None:
        github_public_audit = load(github_public_audit_path)
        if github_public_audit.get("status") not in {
            "GITHUB_PUBLIC_METADATA_AVAILABLE",
            "BLOCKED_GITHUB_PUBLIC_METADATA_ROUTE",
        }:
            raise SystemExit("GitHub public metadata audit has an unexpected status")
        if github_public_audit.get("payload_downloaded") is not False or github_public_audit.get("processed_payload_admitted") is not False:
            raise SystemExit("GitHub public metadata audit is not fail-closed")
    fastq_batch_audit = None
    if fastq_batch_audit_path is not None:
        fastq_batch_audit = load(fastq_batch_audit_path)
        if fastq_batch_audit.get("status") not in {"BATCH_COMPLETE", "BATCH_PARTIAL_PENDING_OR_BLOCKED"}:
            raise SystemExit("FASTQ batch audit has an unexpected status")
        if fastq_batch_audit.get("scientific_gate_effect") != "NO_PHASE_0_PASS":
            raise SystemExit("FASTQ batch audit does not preserve the scientific stop rule")
        if fastq_batch_audit.get("raw_sequence_content_emitted") is not False or fastq_batch_audit.get("scientific_labels_admitted") is not False:
            raise SystemExit("FASTQ batch audit is not fail-closed")
    additional_range_audits = []
    for additional_range_audit_path in additional_range_audit_paths:
        additional_range_audit = load(additional_range_audit_path)
        if additional_range_audit.get("status") != "BLOCKED_HTTP_403_RANGE_PROBE" or additional_range_audit.get("payload_downloaded") is not False or additional_range_audit.get("observed_body_bytes") != 0:
            raise SystemExit(f"additional range audit is not a blocked zero-byte result: {additional_range_audit_path}")
        if additional_range_audit.get("processed_payload_admitted") is not False:
            raise SystemExit(f"additional range audit is not fail-closed: {additional_range_audit_path}")
        additional_range_audits.append(additional_range_audit)

    manifests = code_root / "manifests"
    history = manifests / "history"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("phase0_payload_inventory.json", "data_registry.json", "acceptance_phase0.json", "phase_status.json"):
        current = manifests / name
        backup = history / f"{name.removesuffix('.json')}_{args.run_id}.json"
        if backup.exists():
            raise SystemExit(f"refusing to overwrite existing manifest backup: {backup}")
        shutil.copy2(current, backup)

    route_rel = str(route_path.relative_to(artifact_root))
    route_tsv_rel = str((artifact_root / "phase0/source_metadata" / f"figshare_readme_reprobe_{args.run_id}.tsv").relative_to(artifact_root))
    ledger_rel = str(ledger_path.relative_to(artifact_root))
    tree_rel = str(github_tree_path.relative_to(artifact_root)) if github_tree_path is not None else None
    tree_audit_rel = f"phase0/audits/dms_github_tree_payload_audit_{args.run_id}.json"
    zenodo_rel = str(zenodo_audit_path.relative_to(artifact_root)) if zenodo_audit_path is not None else None
    range_rel = str(range_audit_path.relative_to(artifact_root)) if range_audit_path is not None else None
    metadata_rel = str(metadata_audit_path.relative_to(artifact_root)) if metadata_audit_path is not None else None
    provenance_rel = str(provenance_audit_path.relative_to(artifact_root)) if provenance_audit_path is not None else None
    doi_provenance_rel = str(doi_provenance_audit_path.relative_to(artifact_root)) if doi_provenance_audit_path is not None else None
    v8_rel = str(v8_audit_path.relative_to(artifact_root)) if v8_audit_path is not None else None
    oai_format_rel = str(oai_format_audit_path.relative_to(artifact_root)) if oai_format_audit_path is not None else None
    github_public_rel = str(github_public_audit_path.relative_to(artifact_root)) if github_public_audit_path is not None else None
    fastq_batch_rel = str(fastq_batch_audit_path.relative_to(artifact_root)) if fastq_batch_audit_path is not None else None
    additional_range_rels = [str(path.relative_to(artifact_root)) for path in additional_range_audit_paths]
    now = datetime.now(timezone.utc).isoformat()

    inventory_path = manifests / "phase0_payload_inventory.json"
    inventory = load(inventory_path)
    inventory["last_refresh_utc"] = now
    inventory["scientific_gate_effect"] = "NO_PHASE_0_PASS"
    inventory["primary_labels_admitted"] = False
    inventory.setdefault("artifacts", [])
    append_unique(inventory["artifacts"], {
        "source_id": "deenalattha_2026_dms",
        "kind": "processed_dms_payload_current_readme_route_reprobe",
        **relative_artifact(artifact_root, route_path, route_probe_tsv=route_tsv_rel, status=route.get("status"), request_referer=route.get("request_referer"), payload_downloaded=False, access_control_bypassed=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
    })
    append_unique(inventory["artifacts"], {
        "source_id": "deenalattha_2026_dms",
        "kind": "DMS_phase0_dependency_ledger_current",
        **relative_artifact(artifact_root, ledger_path, status=ledger.get("status"), payload_downloaded=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
    })
    if github_tree_path is not None and github_tree is not None:
        tree_paths = [item.get("path") for item in github_tree["tree"] if isinstance(item, dict)]
        tree_audit_path = artifact_root / tree_audit_rel
        if tree_audit_path.exists():
            raise SystemExit(f"refusing to overwrite existing GitHub tree audit: {tree_audit_path}")
        tree_audit = {
            "schema_version": "phase0-public-source-tree-audit-v1",
            "status": "PUBLIC_SOURCE_REPOSITORY_TREE_METADATA_COMPLETE_NO_PROCESSED_PAYLOAD_ADMITTED",
            "created_at_utc": now,
            "source_tree": relative_artifact(artifact_root, github_tree_path),
            "tree_truncated": False,
            "tree_entry_count": len(github_tree["tree"]),
            "file_path_count": len(tree_paths),
            "data_zip_path_present": "data.zip" in tree_paths,
            "data_directory_path_present": any(str(path).startswith("data/") for path in tree_paths),
            "processed_payload_admitted": False,
            "raw_sequence_content_emitted": False,
            "primary_labels_admitted": False,
            "scientific_gate_effect": "NO_PHASE_0_PASS",
            "interpretation_boundary": "Repository tree metadata inventories public source paths only; absence of a data path is not proof that the official external payload is absent.",
        }
        dump_atomic(tree_audit_path, tree_audit)
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "public_source_repository_tree_audit",
            **relative_artifact(artifact_root, tree_audit_path, status=tree_audit["status"], source_tree=tree_rel, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "public_source_repository_tree_metadata",
            **relative_artifact(artifact_root, github_tree_path, status="GITHUB_TREE_METADATA_COMPLETE", tree_truncated=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if zenodo_audit_path is not None and zenodo_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "public_zenodo_code_route_audit_current",
            **relative_artifact(artifact_root, zenodo_audit_path, status=zenodo_audit["status"], payload_downloaded=False, access_control_bypassed=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if range_audit_path is not None and range_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_range_probe_current",
            **relative_artifact(artifact_root, range_audit_path, status=range_audit["status"], requested_range=range_audit.get("requested_range"), observed_body_bytes=range_audit.get("observed_body_bytes"), payload_downloaded=False, access_control_bypassed=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if metadata_audit_path is not None and metadata_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_file_metadata_probe_current",
            **relative_artifact(artifact_root, metadata_audit_path, status=metadata_audit["status"], observed_body_bytes=metadata_audit.get("observed_body_bytes"), payload_downloaded=False, access_control_bypassed=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if provenance_audit_path is not None and provenance_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_provenance_route_audit_current",
            **relative_artifact(artifact_root, provenance_audit_path, status=provenance_audit["status"], successful_route_count=provenance_audit.get("successful_route_count"), successful_metadata_route_count=provenance_audit.get("successful_metadata_route_count"), payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if doi_provenance_audit_path is not None and doi_provenance_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_doi_oai_provenance_audit_current",
            **relative_artifact(artifact_root, doi_provenance_audit_path, status=doi_provenance_audit["status"], successful_route_count=doi_provenance_audit.get("successful_route_count"), payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if v8_audit_path is not None and v8_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_figshare_v8_route_audit_current",
            **relative_artifact(artifact_root, v8_audit_path, status=v8_audit["status"], successful_route_count=v8_audit.get("successful_route_count"), payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if oai_format_audit_path is not None and oai_format_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_oai_format_audit_current",
            **relative_artifact(artifact_root, oai_format_audit_path, status=oai_format_audit["status"], successful_route_count=oai_format_audit.get("successful_route_count"), metadata_only=True, payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if github_public_audit_path is not None and github_public_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "public_github_release_tree_metadata_audit_current",
            **relative_artifact(artifact_root, github_public_audit_path, status=github_public_audit["status"], successful_route_count=github_public_audit.get("successful_route_count"), release_asset_count=github_public_audit.get("release_asset_count"), tree_payload_like_path_count=github_public_audit.get("tree_payload_like_path_count"), metadata_only=True, payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    if fastq_batch_audit_path is not None and fastq_batch_audit is not None:
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "public_raw_fastq_partial_state_audit_current",
            **relative_artifact(artifact_root, fastq_batch_audit_path, status=fastq_batch_audit["status"], selected_run_count=len(fastq_batch_audit.get("selected_runs", [])), failed_run_count=fastq_batch_audit.get("failed_run_count"), pending_run_count=fastq_batch_audit.get("pending_run_count"), raw_sequence_content_emitted=False, scientific_labels_admitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    for additional_range_audit_path, additional_range_audit in zip(additional_range_audit_paths, additional_range_audits):
        append_unique(inventory["artifacts"], {
            "source_id": "deenalattha_2026_dms",
            "kind": "processed_dms_payload_additional_range_probe_current",
            **relative_artifact(artifact_root, additional_range_audit_path, status=additional_range_audit["status"], requested_range=additional_range_audit.get("requested_range"), observed_body_bytes=additional_range_audit.get("observed_body_bytes"), observed_total_bytes=additional_range_audit.get("observed_total_bytes"), payload_downloaded=False, processed_payload_admitted=False, raw_sequence_content_emitted=False, primary_labels_admitted=False, scientific_gate_effect="NO_PHASE_0_PASS"),
        })
    inventory.setdefault("required_next_evidence", [])
    inventory["required_next_evidence"] = sorted(set(inventory["required_next_evidence"]) | {
        "verified official processed-DMS 2xx route with expected payload size, then 128 MiB range download and hash/gzip/provenance audit",
    })
    dump_atomic(inventory_path, inventory)

    registry_path = manifests / "data_registry.json"
    registry = load(registry_path)
    registry["metadata_audit_status"] = "PARTIAL_METADATA_AND_PUBLIC_FASTQ_INVENTORY_NOT_PHASE0_PASS"
    registry.setdefault("phase0_evidence_updates", []).append({"run_id": args.run_id, "processed_dms_route_reprobe": route_rel, "dms_dependency_ledger": ledger_rel, "scientific_gate_effect": "NO_PHASE_0_PASS"})
    for source in registry.get("sources", []):
        if source.get("source_id") == "deenalattha_2026_dms":
            source["processed_dms_payload_latest_readme_route_reprobe_current"] = route_rel
            source["processed_dms_payload_latest_readme_route_reprobe_current_status"] = route.get("status")
            source["dms_phase0_dependency_ledger_current"] = ledger_rel
            source["dms_phase0_dependency_ledger_current_status"] = ledger.get("status")
            if github_tree_path is not None and github_tree is not None:
                source["public_source_repository_tree"] = tree_rel
                source["public_source_repository_tree_status"] = "GITHUB_TREE_METADATA_COMPLETE"
                source["public_source_repository_tree_audit"] = tree_audit_rel
                source["public_source_repository_tree_audit_status"] = "PUBLIC_SOURCE_REPOSITORY_TREE_METADATA_COMPLETE_NO_PROCESSED_PAYLOAD_ADMITTED"
            if zenodo_audit_path is not None and zenodo_audit is not None:
                source["zenodo_code_record_route_audit_current"] = zenodo_rel
                source["zenodo_code_record_route_audit_current_status"] = zenodo_audit["status"]
            if range_audit_path is not None and range_audit is not None:
                source["processed_dms_payload_range_probe_current"] = range_rel
                source["processed_dms_payload_range_probe_current_status"] = range_audit["status"]
            if metadata_audit_path is not None and metadata_audit is not None:
                source["processed_dms_payload_file_metadata_probe_current"] = metadata_rel
                source["processed_dms_payload_file_metadata_probe_current_status"] = metadata_audit["status"]
            if provenance_audit_path is not None and provenance_audit is not None:
                source["processed_dms_payload_provenance_route_audit_current"] = provenance_rel
                source["processed_dms_payload_provenance_route_audit_current_status"] = provenance_audit["status"]
            if doi_provenance_audit_path is not None and doi_provenance_audit is not None:
                source["processed_dms_payload_doi_oai_provenance_audit_current"] = doi_provenance_rel
                source["processed_dms_payload_doi_oai_provenance_audit_current_status"] = doi_provenance_audit["status"]
            if v8_audit_path is not None and v8_audit is not None:
                source["processed_dms_payload_figshare_v8_route_audit_current"] = v8_rel
                source["processed_dms_payload_figshare_v8_route_audit_current_status"] = v8_audit["status"]
            if oai_format_audit_path is not None and oai_format_audit is not None:
                source["processed_dms_payload_oai_format_audit_current"] = oai_format_rel
                source["processed_dms_payload_oai_format_audit_current_status"] = oai_format_audit["status"]
            if github_public_audit_path is not None and github_public_audit is not None:
                source["public_github_release_tree_metadata_audit_current"] = github_public_rel
                source["public_github_release_tree_metadata_audit_current_status"] = github_public_audit["status"]
            if fastq_batch_audit_path is not None and fastq_batch_audit is not None:
                source["fastq_batch_audit_path"] = fastq_batch_rel
                source["fastq_batch_audit_status"] = fastq_batch_audit["status"]
                source["raw_fastq_status"] = "PUBLIC_RAW_FASTQ_PARTIALS_PRESENT_PENDING_RECOVERY_AND_FINAL_INTEGRITY_AUDIT"
            if additional_range_audit_paths:
                source["processed_dms_payload_additional_range_probes_current"] = additional_range_rels
                source["processed_dms_payload_additional_range_probes_current_status"] = [audit["status"] for audit in additional_range_audits]
            source["processed_dms_payload_status"] = "BLOCKED_ALL_TESTED_PUBLIC_FIGSHARE_ROUTES_HTTP_403"
            source["dms_reconstruction_status"] = "BLOCKED_RECONSTRUCTION_INPUTS_MISSING"
    dump_atomic(registry_path, registry)

    acceptance_path = manifests / "acceptance_phase0.json"
    acceptance = load(acceptance_path)
    acceptance["status"] = "IN_PROGRESS_PUBLIC_FASTQ_PAYLOAD_AUDIT"
    acceptance["pass"] = False
    evidence = acceptance.setdefault("evidence_paths", [])
    for value in (route_rel, route_tsv_rel, ledger_rel):
        if value not in evidence:
            evidence.append(value)
    if github_tree_path is not None and github_tree is not None:
        for value in (tree_rel, tree_audit_rel):
            if value not in evidence:
                evidence.append(value)
    if zenodo_audit_path is not None and zenodo_audit is not None and zenodo_rel not in evidence:
        evidence.append(zenodo_rel)
    if range_audit_path is not None and range_audit is not None and range_rel not in evidence:
        evidence.append(range_rel)
    if metadata_audit_path is not None and metadata_audit is not None and metadata_rel not in evidence:
        evidence.append(metadata_rel)
    if provenance_audit_path is not None and provenance_audit is not None and provenance_rel not in evidence:
        evidence.append(provenance_rel)
    if doi_provenance_audit_path is not None and doi_provenance_audit is not None and doi_provenance_rel not in evidence:
        evidence.append(doi_provenance_rel)
    if v8_audit_path is not None and v8_audit is not None and v8_rel not in evidence:
        evidence.append(v8_rel)
    if oai_format_audit_path is not None and oai_format_audit is not None and oai_format_rel not in evidence:
        evidence.append(oai_format_rel)
    if github_public_audit_path is not None and github_public_audit is not None and github_public_rel not in evidence:
        evidence.append(github_public_rel)
    if fastq_batch_audit_path is not None and fastq_batch_audit is not None and fastq_batch_rel not in evidence:
        evidence.append(fastq_batch_rel)
    for additional_range_rel in additional_range_rels:
        if additional_range_rel not in evidence:
            evidence.append(additional_range_rel)
    acceptance["note"] = str(acceptance.get("note", "")) + f" A low-frequency official processed-DMS route re-probe at {args.run_id} returned {route.get('status')} for {len(route.get('routes', [])) if isinstance(route.get('routes'), list) else 'unknown'} routes; no payload was downloaded, no access control was bypassed, and the updated dependency ledger remains fail-closed."
    if github_tree_path is not None and github_tree is not None:
        acceptance["note"] += f" Public GitHub v1.0.0 tree metadata was recorded with {len(github_tree['tree'])} entries; it inventories source paths only and does not establish that the external official data payload is absent or available."
    if zenodo_audit_path is not None and zenodo_audit is not None:
        acceptance["note"] += f" The official Zenodo API route for code record {zenodo_audit.get('zenodo_record_id')} returned connection refused/HTTP 000 at {args.run_id}; this is network-route evidence only and does not establish data absence or admit the code archive as processed-DMS payload."
    if range_audit_path is not None and range_audit is not None:
        acceptance["note"] += f" A real 128 MiB Range GET probe at {args.run_id} returned HTTP 403 with zero body bytes and no Content-Range; no chunk was downloaded and the 128 MiB transfer gate remains closed."
    if metadata_audit_path is not None and metadata_audit is not None:
        acceptance["note"] += f" The official Figshare file metadata endpoint at {args.run_id} returned HTTP 403 with zero body bytes; expected payload size could not be established through that official metadata route."
    if provenance_audit_path is not None and provenance_audit is not None:
        acceptance["note"] += f" A new metadata-only Figshare provenance audit at {args.run_id} returned {provenance_audit.get('status')} across {len(provenance_audit.get('routes', [])) if isinstance(provenance_audit.get('routes'), list) else 'unknown'} DOI/landing-page/article/files/versions routes; no processed payload was downloaded or admitted."
    if doi_provenance_audit_path is not None and doi_provenance_audit is not None:
        acceptance["note"] += f" DataCite DOI and Figshare OAI-PMH metadata at {args.run_id} returned {doi_provenance_audit.get('status')}; metadata exposed provenance relationships only, and no processed payload was admitted."
    if v8_audit_path is not None and v8_audit is not None:
        acceptance["note"] += f" The newly identified Figshare v8 landing/download/API routes at {args.run_id} returned {v8_audit.get('status')}; no payload was downloaded or admitted."
    if oai_format_audit_path is not None and oai_format_audit is not None:
        acceptance["note"] += f" Figshare OAI-PMH METS/QDC/RDF/CERIF metadata routes returned {oai_format_audit.get('status')} at {args.run_id}; METS exposed file sizes and FLocat URLs, but no processed payload was downloaded or admitted and the 128 MiB transfer gate remains closed."
    if github_public_audit_path is not None and github_public_audit is not None:
        acceptance["note"] += f" Official GitHub release/tree metadata returned {github_public_audit.get('status')} at {args.run_id}; release assets and repository payload-like paths were inventoried without downloading repository files, and this metadata cannot substitute for the external processed-DMS payload."
    if fastq_batch_audit_path is not None and fastq_batch_audit is not None:
        acceptance["note"] += f" A current fail-closed FASTQ partial-state audit at {args.run_id} returned {fastq_batch_audit.get('status')} for {len(fastq_batch_audit.get('selected_runs', []))} explicitly scoped runs; {fastq_batch_audit.get('failed_run_count')} run(s) still contain preserved partial files, so recovery and final integrity audit remain pending."
    if additional_range_audit_paths:
        acceptance["note"] += f" Newly discovered Figshare file IDs were each subjected to one exact 128 MiB Range probe at {args.run_id}; all recorded results remain fail-closed and no complete payload was admitted."
    dump_atomic(acceptance_path, acceptance)

    phase_path = manifests / "phase_status.json"
    phase = load(phase_path)
    phase["last_transition"] = now
    phase["transition_evidence"] = "manifests/phase0_payload_inventory.json"
    phase["scientific_gate_effect"] = "NO_UNLOCK"
    blockers = phase.setdefault("blocking_conditions", [])
    blocker = f"The latest official processed-DMS route re-probe ({args.run_id}) returned no 2xx response; the payload remains unavailable and the 128 MiB download stop rule is still active."
    if blocker not in blockers:
        blockers.append(blocker)
    if github_tree_path is not None and github_tree is not None:
        blocker = f"Public GitHub v1.0.0 tree metadata was audited at {args.run_id}; it is source-path evidence only and cannot substitute for the external processed-DMS payload or primary labels."
        if blocker not in blockers:
            blockers.append(blocker)
    if zenodo_audit_path is not None and zenodo_audit is not None:
        blocker = f"The official Zenodo API code-record route returned connection refused/HTTP 000 at {args.run_id}; this route failure does not establish data absence and does not admit the code archive as processed-DMS payload."
        if blocker not in blockers:
            blockers.append(blocker)
    if range_audit_path is not None and range_audit is not None:
        blocker = f"A real 128 MiB Range GET probe at {args.run_id} returned HTTP 403 with zero body bytes and no Content-Range; no processed-DMS chunk is admitted and the payload download gate remains closed."
        if blocker not in blockers:
            blockers.append(blocker)
    if metadata_audit_path is not None and metadata_audit is not None:
        blocker = f"The official Figshare file metadata endpoint returned HTTP 403 at {args.run_id}; expected processed-DMS payload size remains unverified."
        if blocker not in blockers:
            blockers.append(blocker)
    if provenance_audit_path is not None and provenance_audit is not None:
        blocker = f"The metadata-only Figshare provenance audit returned {provenance_audit.get('status')} at {args.run_id}; no official 2xx provenance route or candidate payload metadata was available."
        if blocker not in blockers:
            blockers.append(blocker)
    if doi_provenance_audit_path is not None and doi_provenance_audit is not None:
        blocker = f"The DataCite/OAI provenance audit returned {doi_provenance_audit.get('status')} at {args.run_id}; metadata relationships do not substitute for a verified processed-DMS payload."
        if blocker not in blockers:
            blockers.append(blocker)
    if v8_audit_path is not None and v8_audit is not None:
        blocker = f"The Figshare v8 route audit returned {v8_audit.get('status')} at {args.run_id}; no v8-specific 2xx payload route is available."
        if blocker not in blockers:
            blockers.append(blocker)
    if oai_format_audit_path is not None and oai_format_audit is not None:
        blocker = f"Figshare OAI-PMH METS/QDC/RDF/CERIF metadata was available at {args.run_id}, but metadata-only evidence cannot substitute for a verified processed-DMS payload or unlock the 128 MiB transfer gate."
        if blocker not in blockers:
            blockers.append(blocker)
    if github_public_audit_path is not None and github_public_audit is not None:
        blocker = f"Official GitHub release/tree metadata returned {github_public_audit.get('status')} at {args.run_id}; code-repository metadata cannot substitute for the external processed-DMS payload or primary labels."
        if blocker not in blockers:
            blockers.append(blocker)
    if fastq_batch_audit_path is not None and fastq_batch_audit is not None:
        blocker = f"The current FASTQ partial-state audit returned {fastq_batch_audit.get('status')} at {args.run_id}; preserved partial files remain and no recovery or final integrity PASS is accepted while the guarded downloader process exists."
        if blocker not in blockers:
            blockers.append(blocker)
    for additional_range_audit_path, additional_range_audit in zip(additional_range_audit_paths, additional_range_audits):
        blocker = f"Additional Figshare range probe {additional_range_audit_path.name} returned {additional_range_audit.get('status')} at {args.run_id}; no complete processed-DMS payload is admitted."
        if blocker not in blockers:
            blockers.append(blocker)
    dump_atomic(phase_path, phase)
    print(json.dumps({"status": "PHASE0_DMS_EVIDENCE_REGISTERED_FAIL_CLOSED", "route": route_rel, "ledger": ledger_rel, "run_id": args.run_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
