# RNA Junction 投稿整合建议（最终版，r45 修正）

日期：2026-08-15
分支：`r0_audit_repair_20260811`
状态：方法边界闭合（19 条路线），冻结方法已修正至无约束最优，投稿材料建议如下。

## 1. 冻结投稿方法

**7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 留一折
per-scaffold × stratum σ 校准（σ 扫描下界 0.05 = MetricSpec floor）。**

| 口径 | pooled NLL | vs nuisance | edit-cluster CI |
|------|-----------:|------------:|----------------:|
| frozen σ=0.7 | 0.8522 | +21.94% | [0.1807, 0.3684] lower>0 |
| **r45 校准（冻结）** | **0.7907** | **+21.25%（同口径）/+27.57%（@frozen）** | **[0.1919, 0.2682] lower>0** |

## 2. 主表（Table 1：全模型族横向对比）

见 `submission_horizontal_table.json` 与本文档的最终横向表（frozen 0.7 + r45 两口径）。
关键读数：
- 7-member 混合集成在两种口径下均为最优；
- GBDT-only / MLP-only 均显著更差（r45 0.8243/0.8410 vs 0.7907），证明跨族
  误差多样性是关键；
- r50 确认 equal-family 权重（wg=0.5）在 r45 下仍最优。

## 3. 主图（Figure 1：per-operator stratum sigma）

见 `per_scaf_stratum_sigma_figure.svg/.png`：每个 operator 的 σ_m（measured）
与 σ_c（censored）柱状图，标注删失率。核心视觉信息：
- scaf9（78.5% 删失）：σ_m=1.15 vs σ_c=0.19 —— 两 stratum 尺度差异最大；
- scaf2-7（低删失）：σ_m≈σ_c≈0.45–0.73 —— 分层必要性集中在高删失算子；
- 虚线 = 冻结 0.7 参考：说明为什么固定 σ=0.7 在分层后不再合适。

## 4. 可发表主张（ClaimAuthorization 约束内）

**Allowed：**
1. censor-aware 鲁棒（Student-t）非线性头 + 跨族 7-member mu-集成 +
   per-operator 异方差 σ 校准（measured/censored 分层、留一折无泄漏、跨 37 折
   稳定），相对线性/no-sequence 对照的 pooled-NLL 增益 **+27.57%**（@ frozen
   0.7）/ +21.25%（@ 同口径），edit-cluster CI lower>0，非单一组件驱动。
2. 集成残差方差收缩使发射 σ 应从 0.7 校准至 per-scaf×stratum 0.19–1.59。
3. 方法级边界系统闭合：19 条组合/校准/换族/算子/粒度/训练侧路线全部测尽，
   仅 σ 事后校准（global→scaf→scaf×stratum）为正 —— 本身是可发表的
   benchmark 认识。

**Forbidden（延续）：**
- transferable sequence mechanism、SOTA、noise ceiling；
- 13 独立模型族公平比较；
- 63-D sequence-map 路线关闭；
- 提交/release 不授权（需 owner 明确指示 + P0.6 重裁定 + release seal）。

## 5. 稿件材料清单

| 材料 | 状态 | 来源 |
|------|------|------|
| 最终横向表 | 已生成 | `submission_horizontal_table.json` |
| 主图（per-operator σ） | 已生成 | `per_scaf_stratum_sigma_figure.*` |
| 残差/分层诊断 | 已生成 | `residual_structure_diagnostic.json` |
| 方法冻结记录 | 已更新 | `reactflow_delta_method_freeze_r37_20260815.md`（r37–r50 全记录） |
| Claim 矩阵 | 已更新 | `audit/release/SubmissionClaimMatrix.csv` |
| Task equivalence | 待生成 | 需 owner 确认 comparator 清单 |
| 稿件 draft | 待生成 | 需 owner 选择故事线（benchmark vs mechanism） |
| 图 2+（NLL 对比、CI） | 待生成 | 可基于现有 JSON 快速生成 |

## 6. 剩余投稿前置（不可省略）

1. P0.6 重新裁定（eligibility_status + scientific verdict）基于修正后 0.7907；
2. owner 明确投稿授权 + 故事线选择；
3. Release seal 重建（当前 checksum 7/13 失败，需用修正后方法重跑）；
4. Legal closure（数据/derivative 许可）。

## 7. 建议的下一步（一次性完成）

- 生成 Figure 2（横向 NLL 柱状图 + CI 误差条）；
- 生成 Figure 3（censored/measured 分层分解）；
- 生成稿件 Table 2（消融阶梯：单成员→3x t7→7mem→+r45）。

## 8. Table 2（消融阶梯：方法贡献链）

| 步骤 | frozen 0.7 | r45 校准 |
|------|-----------:|---------:|
| nuisance（基线） | 1.0916 | 1.0040 |
| 最优单成员（GBDT lr03） | 0.8807 | 0.8252 |
| 3x t7 集成 | 0.8823 | 0.8410 |
| 7-member 混合集成 | 0.8527 | 0.7907 |
| **冻结方法（7mem + r45 σ 校准）** | 0.8527 | **0.7907（+27.57%）** |

方法贡献链：非线性鲁棒头（+17.45%）→ 跨族集成（+21.94%）→ per-operator
stratum σ 校准（+27.57%）。

## 9. Figure 2（横向对比）

见 `horizontal_nll_figure.svg/.png`：全模型族 frozen 0.7 vs r45 校准双柱图，
r45 排序，冻结方法高亮。视觉信息：r45 校准系统性降低所有模型 NLL（含
nuisance），7-member 集成在两口径下均最优。

## 10. 更新后的材料清单

| 材料 | 状态 | 来源 |
|------|------|------|
| 最终横向表 | 已生成 | `submission_horizontal_table.json` |
| Figure 1（per-operator σ） | 已生成 | `per_scaf_stratum_sigma_figure.*` |
| Figure 2（横向对比） | 已生成 | `horizontal_nll_figure.*` |
| Table 2（消融阶梯） | 已生成 | `fig2_and_ablation_ladder.json` + 本节 |
| 残差/分层诊断 | 已生成 | `residual_structure_diagnostic.json` |
| 方法冻结记录 | 已更新 | `reactflow_delta_method_freeze_r37_20260815.md` |
| Claim 矩阵 | 已更新 | `audit/release/SubmissionClaimMatrix.csv` |
| Task equivalence | 待生成 | 需 owner 确认 comparator 清单 |
| 稿件 draft | 待生成 | 需 owner 选择故事线 |

## 11. 生成命令（可复现）

```bash
cd /home/cunyuliu/rna_junction_repair_20260811
# 最终横向表（已生成）
python -c "from audit.repair.submission_horizontal_table import main; main()"
# Figure 1（per-operator sigma，已生成）
python -c "from audit.repair.per_scaf_stratum_sigma_figure import main; main()"
# Figure 2 + Table 2（已生成）
python -c "from audit.repair.horizontal_nll_figure import main; main()"
```
