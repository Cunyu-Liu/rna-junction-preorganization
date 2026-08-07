"""P0.6 phase sealer (contract P0.6).

Consumes the P0.1-P0.5 STATUS files, the gate matrix and claim matrix, and
seals P0 with a fail-closed decision.  Emits:
  P0Decision.md    : human-readable fail-closed P0 decision
  checksums.sha256 : hashes of every evidence file referenced
  STATUS.json      : final P0.6 STATUS (only allowed states)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ALLOWED_STATES = {"P0_PASS_COMPARISON_ELIGIBLE", "P0_PASS_FRESH_ONLY", "BLOCKED_WITH_EVIDENCE"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_checksums(run_root: Path, gate_matrix: dict):
    sums = {}
    for g in gate_matrix["gates"]:
        ep = run_root / g["evidence_path"]
        if ep.exists():
            sums[g["evidence_path"]] = sha256(ep)
    return sums


def seal(cfg):
    run_root = Path(cfg["run_root"])
    adj = run_root / "adjudication"
    gmx = json.loads((adj / "P0GateMatrix.json").read_text())
    clm = json.loads((adj / "ClaimMatrix.json").read_text())
    overall = gmx["overall_state"]
    if overall not in ALLOWED_STATES:
        overall = "BLOCKED_WITH_EVIDENCE"  # fail-closed: any illegal state blocks

    # Eligible model IDs for Phase 1: only those that passed numerical gates.
    eligible = []
    if overall == "P0_PASS_COMPARISON_ELIGIBLE" or overall == "P0_PASS_FRESH_ONLY":
        # corrected v1.31 passed P0.3; v1.28/v1.30 not numerically clean.
        # Eligibility is per-method and decided in P0.5 leaderboard + P0.6 verdict.
        pass

    checksums = build_checksums(run_root, gmx)
    (adj / "checksums.sha256").write_text(
        "".join(f"{h}  {p}\n" for p, h in sorted(checksums.items())))

    status = {
        "phase": "P0.6", "state": overall,
        "run_id": gmx.get("run_id") or cfg.get("run_id"),
        "n_gates": gmx["n_gates"], "n_pass": gmx["n_pass"],
        "n_fail": gmx["n_fail"], "n_not_run": gmx["n_not_run"],
        "hard_fail_gates": gmx["hard_fail_gates"],
        "not_run_gates": gmx["not_run_gates"],
        "allowed_states": sorted(ALLOWED_STATES),
        "sealed_note": ("P0 sealed fail-closed. No SOTA/mechanism/submission claim "
                        "may be made from P0 evidence. P0 only decides gate to P1."),
    }
    (adj / "STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")

    # P0Decision.md
    fail_lines = "\n".join(f"- FAIL: {g}" for g in gmx["hard_fail_gates"]) or "- none"
    notrun_lines = "\n".join(f"- NOT_RUN: {g}" for g in gmx["not_run_gates"]) or "- none"
    md = f"""# P0 科学资格裁定 (P0 Decision)

Run ID: `{status.get('run_id')}`
日期: 2026-08-07 (Asia/Shanghai)
裁定: **{overall}**

## 允许的最终状态
仅允许以下三者之一：`P0_PASS_COMPARISON_ELIGIBLE` / `P0_PASS_FRESH_ONLY` / `BLOCKED_WITH_EVIDENCE`。
当前 gate matrix 判定：`{overall}`。

## 硬失败 gates（fail-closed，不得掩盖）
{fail_lines}

## 未运行 gates
{notrun_lines}

## Gate 汇总
- 总 gates: {gmx['n_gates']}
- PASS: {gmx['n_pass']}
- FAIL: {gmx['n_fail']}
- NOT_RUN: {gmx['n_not_run']}

## Claim 裁定摘要
见 `ClaimMatrix.csv` / `ClaimMatrix.json`。P0 证据不得自动产生 SOTA、
mechanism 或 submission 主张。

## Phase 1 门控
- 仅当 `P0_PASS_COMPARISON_ELIGIBLE` 或 `P0_PASS_FRESH_ONLY` 时，才允许进入
  Phase 1 科学比较；P0 本身不授予 SOTA/submission。
- 每个科学主张必须链接到具体数据/代码/结果/图（见 ClaimMatrix）。

## 封存说明
P0 已 fail-closed 封存。任何后续修改都必须重新走对应 gate 并提供新证据。
"""
    (adj / "P0Decision.md").write_text(md)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    seal(json.loads(Path(sys.argv[1]).read_text()))
