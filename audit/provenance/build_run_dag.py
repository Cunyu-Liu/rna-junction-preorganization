"""Build the run DAG from historical run manifests (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def walk_manifests(outputs_root: Path):
    """Yield (run_id, parent_run_id, contract_sha, manifest_relpath)."""
    nodes = {}
    for m in outputs_root.rglob("*.json"):
        try:
            data = json.loads(m.read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = data.get("run_id")
        if not rid:
            continue
        parent = data.get("parent_run_id")
        csha = None
        if isinstance(data.get("contract"), dict):
            csha = data["contract"].get("sha256")
        if isinstance(data.get("contract_artifacts"), dict):
            csha = data["contract_artifacts"].get("sha256", csha)
        nodes[rid] = {
            "run_id": rid,
            "parent_run_id": parent,
            "contract_sha256": csha,
            "manifest_path": str(m.relative_to(outputs_root)),
            "manifest_sha256": __import__("hashlib").sha256(m.read_bytes()).hexdigest(),
            "state": data.get("current_operational_state") or data.get("disposition"),
        }
    return nodes


def build_dag(outputs_root: Path) -> dict:
    nodes = walk_manifests(outputs_root)
    edges = []
    for rid, node in nodes.items():
        p = node["parent_run_id"]
        if p and p in nodes:
            edges.append({"from": p, "to": rid})
        elif p:
            edges.append({"from": p, "to": rid, "parent_present": False})
    # Detect parent referenced but absent in this snapshot (run-DAG gap).
    dangling = [e for e in edges if e.get("parent_present") is False]
    return {
        "n_nodes": len(nodes),
        "nodes": nodes,
        "edges": edges,
        "dangling_parent_edges": dangling,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    dag = build_dag(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dag, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "n_nodes": dag["n_nodes"],
        "n_edges": len(dag["edges"]),
        "dangling": len(dag["dangling_parent_edges"]),
    }, indent=2))
