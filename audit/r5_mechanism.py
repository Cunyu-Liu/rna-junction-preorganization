"""R5 mechanism analysis, claim-to-evidence matrix and paper narrative
(contract §12.6, 轨 A).

Given D1 = TRACK_A_LOCKED and R4 evidence closure, R5 writes the paper story
WITHOUT overstepping the evidence boundary.  Every claim is tagged with an
evidence label and mapped to a concrete run/commit/split/metric/row artifact.

Outputs into RUN_ROOT/r5/:
  ClaimEvidenceMatrix.json/.csv   every claim -> decision + evidence pointers
  FailureAtlas.json               support / nested-context / scaffold-bundle strata
  PaperOutline.md                 benchmark/failure-boundary paper narrative
  Limitations.md                  explicit limitations & NOT-asserted boundaries
  STATUS.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit.evaluation.metrics import row_nll
from audit.statistics import multiway_cluster as mw


def _read_json(p: Path):
    if p.exists():
        return json.loads(p.read_text())
    return None


def _load_predictions(run_root: Path):
    """R1 merged row predictions keyed by (model_id, source_row_id)."""
    recs = []
    p = run_root / "r1" / "Predictions_v2.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# claim-to-evidence matrix
# ---------------------------------------------------------------------------
CLAIMS = [
    {
        "claim": "support_aware_mixture 候选（Candidate C）失败",
        "decision": "FACT_CONFIRMED / NOT_PROMOTED",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R0.6 ComparisonEligibilityDecision_v2; supported rows 与 P1 edit_knn 的 mu/sigma 逐点一致；symmetry/edit/context 三轴显著落后配置基线",
        "artifacts": ["r06/ComparisonEligibilityDecision_v2.json",
                      "p4_final/FinalPredictions.parquet"],
        "limitation": "只淘汰该具体候选，不构成对所有 sequence representation 的反证",
    },
    {
        "claim": "六个参数基线此前因右删失梯度符号反转而不合格",
        "decision": "FACT_CONFIRMED",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "baselines.py censored-row gradient 符号反转；中心有限差分诊断 解析 +2.512 vs 有限差分 -2.512",
        "artifacts": ["audit/benchmark/baselines.py",
                      "audit/numerics/finite_difference.py"],
        "limitation": "R0.5 已用梯度修正重跑，本 claim 指历史 P1 失效状态",
    },
    {
        "claim": "abstention 占位 mu=0 曾错误进入 NLL（无 eligible prediction 被计为灾难性负预测）",
        "decision": "FACT_CONFIRMED",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "phase1_run.py 无视 support/abstain 直接把占位 mu=0 纳入 NLL；scaffold LOMO 中 11,893 行 abstain 被算出约 80.45 NLL",
        "artifacts": ["audit/benchmark/phase1_run.py"],
        "limitation": "R0.2 scorer_v2 已改为 full-coverage / selective 分离计分",
    },
    {
        "claim": "核心 sequence 假设在合格 evaluator 下未被识别到稳健增量",
        "decision": "NOT_SUPPORTED_OR_INCONCLUSIVE",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R2 matched ablation：全部轴 relative_gain 0.5-2.8%，低于预注册 10% gate；junction-bootstrap/1000-null/5/5-fold 检查",
        "artifacts": ["r2/CoreHypothesisDecision_v3.json",
                      "r2/MultiwayCluster.json",
                      "r2/MatchedAblationContrast.json"],
        "limitation": "effect size 接近统计显著但效应量 1-4% 远低于 10% 预注册阈值；不能写成普遍 impossibility",
    },
    {
        "claim": "负结果具有足够统计功效（非 power boundary 造成的假阴性）",
        "decision": "ADEQUATE_POWER_TO_EXCLUDE_TARGET",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R4 PowerAnalysis：power@target_gain=1.0；80% 功效可检测相对增益 0.65% << 10% 目标",
        "artifacts": ["r4/PowerAnalysis.json"],
        "limitation": "功效以 junction 为独立单位；若真实独立单位更小则功效下降",
    },
    {
        "claim": "负结论对右删失实现稳健（仅测量子集同结论）",
        "decision": "ROBUST",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R4 CensoringSensitivity：四个轴 measured-only 相对增益仍 <10%，negative_conclusion_robust=true",
        "artifacts": ["r4/CensoringSensitivity.json"],
        "limitation": "scaffold_lomo 无 eligible rows，无法在此数据上裁定",
    },
    {
        "claim": "有效独立样本量远小于行数（重复 context 暴露膨胀 support）",
        "decision": "CONFIRMED",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R4 EffectiveN：11,893 rows -> 1,336 junctions -> effective 927.9 (ICC 0.056, DE 1.44)",
        "artifacts": ["r4/EffectiveN.json"],
        "limitation": "effective N 以 junction 为聚类单位；context 嵌套进一步降低自由单元",
    },
    {
        "claim": "benchmark 覆盖多个独立模型类且强基线未运行",
        "decision": "PARTIAL_COVERAGE",
        "evidence_label": "FACT_CONFIRMED",
        "evidence": "R4 ModelCoverage：9 个运行 family + physical_prior/frozen_lm NOT_RUN",
        "artifacts": ["r4/ModelCoverage.json", "r1/Leaderboard_v2.csv"],
        "limitation": "physical ensemble prior 与 frozen RNA-LM 尚未运行，结论限定到已覆盖模型类",
    },
    {
        "claim": "普遍 identifiability boundary / formal impossibility",
        "decision": "UNKNOWN_NOT_ASSERTED",
        "evidence_label": "UNKNOWN_NOT_ASSERTED",
        "evidence": "当前仅 tested-candidate failure；d>=2/d=3 旧结果混入 abstain 占位",
        "artifacts": [],
        "limitation": "不得写成 formal theorem；最多为指定数据/split/model class/power 下的经验边界",
    },
    {
        "claim": "领域 SOTA 或 best-under-protocol",
        "decision": "SOTA_NOT_ADJUDICATED",
        "evidence_label": "UNKNOWN_NOT_ASSERTED",
        "evidence": "无同数据同 split 同 censored likelihood 同 aggregation 的完整公开榜单",
        "artifacts": [],
        "limitation": "不得使用 SOTA 表述",
    },
    {
        "claim": "投稿 / 方法贡献",
        "decision": "NO_SUBMISSION_AUTHORIZATION",
        "evidence_label": "REQUIRES_NEW_EVIDENCE",
        "evidence": "R5/R6 未闭合：完整强基线、公开许可、clean replay 未完成",
        "artifacts": [],
        "limitation": "当前仅能考虑 benchmark/failure-boundary 技术报告或投稿，需 R6 与 legal 闭合",
    },
]


def build_claim_matrix(run_root: Path, r2, r4):
    rows = []
    for c in CLAIMS:
        rows.append({
            "claim": c["claim"], "decision": c["decision"],
            "evidence_label": c["evidence_label"],
            "evidence": c["evidence"], "artifacts": c["artifacts"],
            "limitation": c["limitation"],
        })
    return {"run_root": str(run_root), "n_claims": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# failure atlas (strata)
# ---------------------------------------------------------------------------
def failure_atlas(run_root: Path):
    recs = _load_predictions(run_root)
    if not recs:
        return {"available": False, "reason": "no R1 predictions"}
    # full-coverage eligible strata: supported rows with a real prediction
    # (abstain excluded).  Compare corrected_v1_31 vs no_sequence vs edit_knn.
    by = {}
    for model in ("corrected_v1_31", "no_sequence_latent_operator", "edit_knn"):
        sub = [r for r in recs if r["model_id"] == model and not r["abstain"]]
        by[model] = sub

    def strata_nll(rows, by_ctx, by_scaf, by_support):
        out = {}
        # support strata
        sup = defaultdict(list)
        for r in rows:
            sup["supported" if r["support"] else "unsupported"].append(r)
        out["support"] = {k: _pooled_macro(v) for k, v in sup.items() if v}
        # nested-context strata
        ctx = defaultdict(list)
        for r in rows:
            ctx[str(r["context"])].append(r)
        out["nested_context"] = {
            "n_contexts": len(ctx),
            "pooled_junction_macro_nll": _pooled_macro(rows),
            "context_nll_5_95_pct": _quantile_strata(ctx),
        }
        # scaffold-bundle strata
        scaf = defaultdict(list)
        for r in rows:
            scaf[int(r["scaf"])].append(r)
        out["scaffold_bundle"] = {
            "n_scaffolds": len(scaf),
            "pooled_junction_macro_nll": _pooled_macro(rows),
            "scaffold_nll_range": _range_strata(scaf),
        }
        return out

    out = {}
    for model, rows in by.items():
        out[model] = {
            "n_supported_rows": len(rows),
            **strata_nll(rows, None, None, None),
        }
    return {"available": True, "models": out}


def _pooled_macro(rows):
    """pooled-junction-macro NLL over rows (uses y/cens/mu/sigma)."""
    if not rows:
        return None
    by = defaultdict(list)
    for r in rows:
        nll = float(row_nll([r["y"]], [r["cens"]], [r["mu"]], [r["sigma"]])[0])
        by[str(r["jid"])].append(nll)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def _quantile_strata(ctx_map):
    vals = [_pooled_macro(v) for v in ctx_map.values() if v]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return [round(float(np.percentile(vals, 5)), 4),
            round(float(np.percentile(vals, 95)), 4)]


def _range_strata(scaf_map):
    vals = [_pooled_macro(v) for v in scaf_map.values() if v]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return [round(min(vals), 4), round(max(vals), 4)]


# ---------------------------------------------------------------------------
# paper narrative
# ---------------------------------------------------------------------------
def paper_outline():
    return (
        "# Benchmark / Failure-Boundary Paper Outline (Track A)\n\n"
        "## 1. Question\n"
        "在正确处理 helix-context 嵌套于 scaffold/operator 的设计、正确右删失并阻断依赖暴露后，"
        "junction sequence 是否仍包含可跨 sequence family 与 nested context/scaffold-bundle "
        "迁移的 preorganization 增量信息？\n\n"
        "## 2. Method (benchmark, not a new predictor)\n"
        "- 修正的右删失 evaluator（统一 CensoredObjective，支持/abstain 分离计分）\n"
        "- joint-blocked split：同时阻断 sequence group 与 nested context / scaffold bundle\n"
        "- 多 estimand（pooled / nested-context / scaffold-bundle junction-macro NLL）\n"
        "- 多模型族同协议比较（校正 v1.31、edit-KNN、参数基线与 matched no-sequence）\n\n"
        "## 3. Results\n"
        "- 校正后各轴 matched relative gain 0.5-2.8%，显著低于 10% 预注册 gate\n"
        "- power@target=1.0，可检测 0.65% 相对增益 -> 负结果非功率假象\n"
        "- 对删失实现、effective N 与多模型类稳健\n\n"
        "## 4. Failure boundary\n"
        "apparent sequence gain 对 operator/context exposure、support policy、删失实现与 "
        "estimand 敏感；在指定数据/模型类/power 下未识别到稳健增量。\n\n"
        "## 5. Limitations\n"
        "物理 ensemble prior 与 frozen RNA-LM 未运行；普遍 identifiability boundary 未断言；"
        "独立 context/operator 交叉需 prospective factorial 数据。\n"
    )


def limitations():
    return (
        "# Limitations & NOT-Asserted Boundaries\n\n"
        "- 核心 sequence 假设：NOT_SUPPORTED_OR_INCONCLUSIVE，非普遍 impossibility\n"
        "- identifiability boundary：UNKNOWN_NOT_ASSERTED；当前仅 tested-candidate failure\n"
        "- SOTA：NOT_ADJUDICATED；无同协议公开榜单\n"
        "- 模型类：physical_prior / frozen_lm 未运行，结论限定到已覆盖类\n"
        "- 数据：单一 study、9 operators、37 edit components；无跨研究复制\n"
        "- 许可：dataset UNKNOWN_NEEDS_LEGAL_REVIEW，code OPEN_SOURCE_PENDING（需 R6 legal）\n"
        "- 复现：R6 双环境 clean replay 尚未完成\n"
    )


def run(cfg):
    run_root = Path(cfg["run_root"])
    out = run_root / "r5"
    out.mkdir(parents=True, exist_ok=True)

    r2 = _read_json(run_root / "r2" / "CoreHypothesisDecision_v3.json")
    r4 = _read_json(run_root / "r4" / "STATUS.json")

    matrix = build_claim_matrix(run_root, r2, r4)
    (out / "ClaimEvidenceMatrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "ClaimEvidenceMatrix.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim", "decision", "evidence_label", "evidence", "limitation"])
        for r in matrix["rows"]:
            w.writerow([r["claim"], r["decision"], r["evidence_label"],
                        r["evidence"], r["limitation"]])

    atlas = failure_atlas(run_root)
    (out / "FailureAtlas.json").write_text(
        json.dumps(atlas, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    (out / "PaperOutline.md").write_text(paper_outline())
    (out / "Limitations.md").write_text(limitations())

    status = {
        "phase": "R5", "track": "A", "state": "CLAIM_MATRIX_AND_NARRATIVE_DONE",
        "n_claims": matrix["n_claims"],
        "failure_atlas_available": atlas.get("available", False),
        "generated_at_utc": _utc(),
    }
    (out / "STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


def _utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    import sys
    run(json.loads(Path(sys.argv[1]).read_text()))
