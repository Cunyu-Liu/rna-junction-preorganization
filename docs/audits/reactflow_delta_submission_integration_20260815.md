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

## 12. P0.6 重新裁定（修正后 0.7907）

基于修正后的 37 折双门 eligible 预测，以 VALID eligibility 重新裁定核心假设
（sequence map 相对 matched no-sequence 的 true joint 增量）：

| 字段 | 值 |
|------|-----|
| eligibility_status | **VALID** |
| corrected_v1_31（full）r45 NLL | 1.3989 |
| no_sequence（matched）r45 NLL | 1.1126 |
| sequence 相对增益 | **−25.73%（full 劣于 no-sequence）** |
| scientific_verdict | **NOT_SUPPORTED_AT_PRE_REGISTERED_GATE** |
| D1 decision | **TRACK_A_LOCKED**（benchmark 轨） |

artifacts：`adjudication_v3/ComparisonEligibilityDecision_v3.json`、
`CoreHypothesisDecision_v4.json`、`DecisionGateD1_v2.json`、
`ClaimAuthorization.json`。

**含义**：63-D sequence-map 路线永久关闭；项目的可发表贡献是 **benchmark 轨**
（censor-aware 评估 + 方法边界闭合 + per-operator stratum σ 校准），不是
transferable mechanism。冻结方法（7mem + r45 = 0.7907，+27.57% vs nuisance）
是 benchmark 轨内的方法贡献。

## 13. TaskEquivalence.csv（P0.4）

见 `TaskEquivalence.csv`（11 行）：Denny/RNAMake/frozen-LM/Geng/trRosettaRNA2/
CHANRG 均不能直接排名（任务不等价或 head 不匹配）；本工作 3 个变体（r10b
非线性头、7mem 集成、冻结方法）为可直接排名的内部配置。禁止把 proxy 命名成
published baseline。

## 14. 投稿材料最终清单

| 材料 | 状态 | 路径 |
|------|------|------|
| Table 1（横向表） | ✅ | `submission_horizontal_table.json` |
| Figure 1（per-op σ） | ✅ | `per_scaf_stratum_sigma_figure.*` |
| Figure 2（横向 NLL） | ✅ | `horizontal_nll_figure.*` |
| Table 2（消融阶梯） | ✅ | `fig2_and_ablation_ladder.json` |
| P0.6 裁定 | ✅ | `adjudication_v3/`（VALID, TRACK_A_LOCKED） |
| Task equivalence | ✅ | `TaskEquivalence.csv` |
| Claim 矩阵 | ✅ | `audit/release/SubmissionClaimMatrix.csv` |
| 方法冻结记录 | ✅ | `reactflow_delta_method_freeze_r37_20260815.md` |
| 稿件 draft | ⏳ | 需 owner 选择故事线（benchmark 轨） |

## 15. 剩余投稿前置（需 owner）

1. 稿件 draft（benchmark 轨故事线）；
2. Release seal 重建（当前 checksum 7/13 失败，需以修正后 0.7907 为准重跑）；
3. Legal closure（数据/derivative 许可）；
4. 最终综合审查。

---

## 16. r51–r53 方法边界补测（2026-08-16）：joint mu-affine + σ 重扫

### 16.1 遗留瓶颈诊断

`residual_structure_diagnostic.json` 显示 r45 后**最大残余误差是 measured 层
系统性偏差**（scaf9 measured bias −0.996、scaf1 −0.445、overall measured
resid mean −0.106、measured RMSE 0.6116）。r46/r47 的 mu 修正均**阴性**，但
事后发现根因是**陈旧 σ**：r47 只在未修正 mu 上扫描 σ_m，且使用旧 grid floor
0.4 —— 修正 mu 后未重扫 σ，等价于用错误尺度评分。

### 16.2 r51：JOINT（measured-only affine mu + σ_m 在修正后 mu 上重扫）LOO 校准

- 变体：global / per-scaf affine / ridge / **per-scaf EB**（最优）；
- grid floor 0.05（MetricSpec floor）+ LOO 无泄漏；censored 行 mu **严格不变**、
  σ_c 保持 r45，censored 侧 NLL 完全不受影响（0.1974→0.1990 仅浮点舍入）；
- 结果：measured 层偏置被消除（scaf9 −0.996→−0.01、scaf1 −0.445→−0.008）；
  per-scaf EB σ_m 重扫后 scaf9 1.22→0.35。

**关键数字（7-member 集成，equal-family wg=0.5）**

| 口径 | pooled NLL | 相对 r45 | 相对 nuisance（同口径） | edit-cluster CI（vs nuisance） |
|------|-----------:|---------:|------------------------:|------------------------------:|
| r45（旧冻结） | 0.7907 | — | +21.25% | [0.1919, 0.2682] |
| **r51（新冻结）** | **0.7815** | **−0.0092（−1.16%）** | **+19.84%** | **[0.1463, 0.2481] lower>0** |

诚实解读（必须写入稿件）：r51 改善**绝对** pooled NLL（主估计量），但
measured-only affine 修正对**所有模型**（含 nuisance 基线）都降低偏置，
故同口径相对增益从 21.25% 微降至 19.84% —— 这是 r47「同口径相对增益降」的
再确认，不是方法退步。**10% 提升门在同口径下仍通过（19.84%>10%）**。

### 16.3 r52 / r53：per-scaf family weight 阴性 + 权重重扫确认

- **r52（per-scaf GBDT/MLP family weight，LOO 无泄漏）**：0.7841（r51 口径），
  **差于** equal-family 0.7815 —— per-scaf 权重过拟合，路线阴性闭合；
- **r53（r51 下 family-weight 重扫）**：最优 wg=0.4 → 0.7803，但 wg=0.5
  （equal-family，0.7815）与其仅差 0.0012（噪声内）。**r50 的结论在 r51 下
  依然成立**：equal-family wg=0.5 是稳健、无调参的冻结选择。

### 16.4 结论与投稿影响

- **方法边界再闭合 2 条**（r52 阴性、r53 确认），冻结方法更新为
  **7-member 集成（wg=0.5）+ r51 joint 校准 = 0.7815**；
- 主估计量（pooled-OOF NLL）较旧冻结 0.7907 **再降 1.16%**，并消除
  measured 层系统性偏置 —— 这是 r45 之后唯一的正向校准级方法增益；
- 建议以 `submission_horizontal_table_v2.json` 作为**新的 definitive 投稿主表**
  （含 frozen 0.7 / r45 / r51 三口径），Figure 2 / Table 2 相应更新；
- P0.6 科学裁定不受影响（sequence map 依然 −25.73%，TRACK_A_LOCKED 不变）；
  Claim 矩阵需补充 r51 为方法贡献链第 4 级（σ 校准之后）。

