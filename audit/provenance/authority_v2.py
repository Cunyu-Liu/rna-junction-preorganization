"""R0.1 fail-closed authority binding (contract R0.1, §13.2).

Replaces the old non-fail-closed P0 gate.  Every authority condition must be
TRUE for the gate to PASS; any FALSE yields a concrete FAIL with evidence.
Conditions checked:
  - strict Markdown contract present and hashes to the authorized SHA;
  - current git commit is a real immutable SHA (never the literal string "HEAD");
  - run DAG has NO dangling parent edges and NO cycles (real detection);
  - canonical source hash matches the frozen durable source;
  - dirty/untracked code state is reported (does not auto-fail but is recorded).

Writes CanonicalStateManifest_v2.json, RunDAG_v2.json, GateStatus_v2.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

# The strict execution contract for the R0-R6 audit is
# rna_junction_post_execution_strict_audit_2026-08-09.md (SHA-256 06dd81cc...).
# Do NOT revert to the older v1.28-v1.31 audit doc's SHA (0be00f01...) — that
# document is a prior review authority, not the current execution contract.
AUTHORIZED_CONTRACT_SHA = "06dd81ccd610f91aab0d07c7980c4e898791bfa9f5c2bf0e12b60c9db3b82496"
CANONICAL_SOURCE_SHA = "0989ddc00bb230fdb00bbc65433c943a0419e35c3d0799b481e741c4a24defe2"


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cmd, cwd):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def current_commit(worktree: Path):
    r = git(["git", "rev-parse", "HEAD"], worktree)
    return r.stdout.strip() if r.returncode == 0 else None


def build_run_dag_v2(nodes, edges):
    """nodes: list of ids; edges: list of (parent, child).  Detects cycles and
    dangling parent edges (parent not in nodes).  Returns (dag, problems)."""
    adj = {n: [] for n in nodes}
    dangling = []
    for p, c in edges:
        if p not in nodes:
            dangling.append((p, c))
            continue
        if c not in nodes:
            continue
        adj[p].append(c)
    # cycle detection (DFS)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    cycle = []

    def dfs(u, stack):
        color[u] = GREY
        stack.append(u)
        for v in adj[u]:
            if color[v] == GREY:
                cycle.append((stack + [v])[stack.index(v):])
                return
            if color[v] == WHITE:
                dfs(v, stack)
        stack.pop()
        color[u] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n, [])
    return {"nodes": nodes, "edges": [list(e) for e in edges],
            "dangling_parent_edges": [list(e) for e in dangling],
            "cycles": [list(c) for c in cycle]}, (dangling, cycle)


def run_authority_gate(worktree: Path, strict_contract_path: Path,
                       canonical_source_path: Path, node_ids, edges):
    """Return (status_dict, all_pass_bool).  Fail-closed: any FALSE -> FAIL."""
    checks = {}
    # 1. contract hash
    if strict_contract_path.exists():
        checks["contract_present"] = True
        checks["contract_hash_matches"] = sha256(strict_contract_path) == AUTHORIZED_CONTRACT_SHA
    else:
        checks["contract_present"] = False
        checks["contract_hash_matches"] = False
    # 2. commit is real SHA (not literal HEAD)
    commit = current_commit(worktree)
    checks["commit_is_real_sha"] = bool(commit and len(commit) == 40
                                        and all(c in "0123456789abcdef" for c in commit))
    checks["current_commit"] = commit
    # 3. run DAG integrity
    dag, (dangling, cycle) = build_run_dag_v2(node_ids, edges)
    checks["run_dag_no_dangling"] = len(dangling) == 0
    checks["run_dag_no_cycles"] = len(cycle) == 0
    # 4. canonical source hash
    if canonical_source_path.exists():
        checks["canonical_source_present"] = True
        checks["canonical_source_hash_matches"] = sha256(canonical_source_path) == CANONICAL_SOURCE_SHA
    else:
        checks["canonical_source_present"] = False
        checks["canonical_source_hash_matches"] = False
    # 5. git status dirty (recorded, not auto-fail)
    st = git(["git", "status", "--porcelain"], worktree)
    checks["worktree_clean"] = bool(st.stdout.strip() == "")
    checks["dirty_note"] = "recorded; dirty code state is a release blocker not an authority gate"

    required = [k for k in checks if k in (
        "contract_present", "contract_hash_matches", "commit_is_real_sha",
        "run_dag_no_dangling", "run_dag_no_cycles",
        "canonical_source_present", "canonical_source_hash_matches")]
    all_pass = all(bool(checks[k]) for k in required)
    status = {
        "phase": "R0.1", "state": "PASS" if all_pass else "FAIL",
        "all_authority_checks_pass": all_pass,
        "checks": checks, "run_dag": dag,
    }
    return status, all_pass


def write_authority_v2(worktree: Path, out_dir: Path, strict_contract_path: Path,
                       canonical_source_path: Path, node_ids, edges, run_id, utc):
    out_dir.mkdir(parents=True, exist_ok=True)
    status, all_pass = run_authority_gate(
        worktree, strict_contract_path, canonical_source_path, node_ids, edges)
    status["run_id"] = run_id
    status["generated_at_utc"] = utc
    status["worktree"] = str(worktree)
    (out_dir / "GateStatus_v2.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    # RunDAG_v2
    (out_dir / "RunDAG_v2.json").write_text(
        json.dumps(status["run_dag"], indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    # CanonicalStateManifest_v2 (authority-bound)
    manifest = {
        "run_id": run_id, "utc": utc, "worktree": str(worktree),
        "strict_contract_sha": AUTHORIZED_CONTRACT_SHA,
        "canonical_source_sha": CANONICAL_SOURCE_SHA,
        "current_commit": status["checks"].get("current_commit"),
        "authority_state": status["state"],
    }
    (out_dir / "CanonicalStateManifest_v2.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return status
