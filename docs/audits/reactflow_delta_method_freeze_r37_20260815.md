# RNA Junction 方法级优化：集成组合与校准扫描（r37 决策记录）

日期：2026-08-15
分支：`r0_audit_repair_20260811`
数据：Denny tectoRNA two-way-junction 冻结 benchmark（11,893 admitted rows，
1,336 junctions，37 edit components，16.2% 右删失）
评估：37 个 blocked edit×nested-context joint folds，right-censored Gaussian NLL
（MetricSpec_v3 pooled-junction-macro PRIMARY；σ 为模型发射参数，未被 MetricSpec
冻结为 0.7），仅消费 optimizer+full-coverage 双门通过的 fold（fail-closed）。

## 1. 背景：此前方法冻结状态

| 方法 | pooled NLL | vs nuisance(1.0916) |
|------|-----------:|--------------------:|
| 3x t7 MLP 集成（seed 0/99/2026） | 0.8823 | +19.17% |
| 4-member 混合（XGB + 3x t7） | 0.8599 | +21.23% |
| **7-member 混合（4x GBDT + 3x t7，family-equal）** | **0.8522** | **+21.93%** |
| 3-family（+ kernel RBF） | 0.8681 | +20.47%（阴性） |

既有结论：方法贡献主要来自右删失鲁棒（Student-t）非线性头（+17.45%），
ViennaRNA 序列表征增量仅 +2.09%；kernel（r36）、异方差 sigma（r17）、
RNA-FM/localctx/deep 等方向全部失败。

## 2. 本次方法级扫描（同一 37-fold OOF predictions，无泄漏）

### 2.1 学到的集成权重（censoring-aware stacking，leave-one-fold-out）

非负线性组合，在 held-out 折叠上优化精确右删失 NLL：

- `stacked_7mem_nll = 0.8655`，**劣于** equal-weight 0.8527（+0.0128）。
- 原因：7 成员质量已足够均衡（误差相关 0.84–0.95），逐 fold 学习权重过拟合
  小折叠，且留一折训练集信息量下降。
- **结论：NEGATIVE**。此前 analyze_mixed_gbdt_t7.py 注释所称"r34/r35 weight
  sweep validated family-equal as optimal"并无独立脚本支撑，本次为首次真实检验；
  family-equal 确实是该数据上的局部最优，无需改为学习式权重。

### 2.2 校准集成 sigma（per-row：sqrt(0.7² + var(mus))）

- `calibrated_sigma_7mem_nll = 0.8592`，劣于 0.8527。
- 原因：把成员间方差当作额外不确定度会使 sigma 过大（max 1.62），NLL 变差；
  成员高度相关，成员 spread 不是独立噪声。
- **结论：NEGATIVE**（per-row 形式）。

### 2.3 残差结构诊断（关键新证据）

7-member equal-weight 集成的 OOF 残差：

- measured RMSE = **0.6116**（明显 < 冻结 σ=0.7），resid mean = −0.106；
- censored mu−CAP 均值 = +0.809，mu 高于阈值占比 88.9%；
- **最优全局 σ（OOF 扫描）= 0.62，NLL 0.8419**（vs 0.8527 at 0.7）；
- per-fold 最优 σ 稳定在 0.45–0.89（36 折），均值 0.617；
- per-scaffold measured RMSE 高度异质：scaf9=1.146、scaf1=0.760、scaf2=0.454；
- per-scaffold bias：scaf9 −0.996、scaf1 −0.445（均未被集成修正）；
- scaffold 9 删失率 78.5%，scaffold 1 删失率 59.2%——高删失 scaffold 的系统偏差
  是残差主要来源。

### 2.4 留一折 σ-only 校准（诚实、无泄漏、仅改 σ 不改 mu）

对每个 held-out fold：σ 仅在**其他 36 折的 OOF 行**上拟合（1-D 扫描最小化 pooled
NLL），再应用到 held-out fold。mu 保持 equal-weight 集成不变。

- `sigma_only_loo_nll = 0.8460`（**−0.0067 vs 0.8527**，即 +22.50% vs nuisance）；
- per-fold σ 均值 0.622（0.61–0.65），高度稳定；
- **机理确认**：纯 σ-only 对单个成员与 nuisance 几乎无增益（nuisance 1.0914
  vs 1.0916、成员 t7_s99 0.8858 vs 0.8839 略差），**只有 7-member 集成受益**——
  因为 mu-平均使残差方差（0.61）低于单模型的 0.7，所以集成应发射校准后的 σ。
- edit-cluster CI（OEC 全变体 vs nuisance）[0.0935, 0.3209]，lower>0；
  leave-one-largest 0.2441，非单一组件驱动。

### 2.5 算子（scaffold）加性截距校准（OEC alpha）

per-scaffold 加性截距 α（在其余 36 折 OOF 上拟合），再连同逐折 σ 应用：

- `oec_7mem_nll = 0.9149`，`alpha_only_loo_nll = 0.9187`，**均显著劣化**。
- 原因：α 把高删失 scaffold（1、9）的 mu 下移，虽降低 measured 残差，却把大量
  censored 行（Y≥−7.1）的 mu 推离右尾，censored NLL 恶化远超 measured 增益。
- **结论：NEGATIVE**。算子偏差应通过更优 mu（保留删失信息）而非加性修正处理。

### 2.6 Student-t GBDT 族（r34 已注册未全跑）

`xgboost_censored_hybrid_t7*`（df=7.0，3 seeds）在 r34 t7 smoke（2 折）中
数值不稳定：t7=0.834、t7_s99=1.397、t7_s2026=1.205（后两者灾难性折叠）。
Student-t 负 hessian 使 XGBoost Newton 步不稳定，鲁棒头对树族不成立。
**结论：DEAD END**，不进入全量 37 折。

## 3. 方法冻结建议

**冻结方法：7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 
留一折 σ 校准（σ≈0.62，非冻结 0.7）。**

| 口径 | pooled NLL | vs nuisance |
|------|-----------:|------------:|
| frozen σ=0.7（此前冻结） | 0.8522 | +21.93% |
| **σ-only LOO 校准（本次）** | **0.8460** | **+22.50%** |

### 可发表主张（ClaimAuthorization 约束内）

- **Allowed**：censor-aware 鲁棒（Student-t）非线性头 + 跨族 7-member mu-集成，
  相对线性/no-sequence 对照的 pooled NLL 增益（frozen σ +21.93%；校准 σ +22.50%，
  edit-cluster CI lower>0，非单一组件驱动）；**集成残差方差收缩使发射 σ 应从 0.7
  校准至 ~0.62**（诚实、留一折、无泄漏）——这是本数据上唯一新确认的方法级改进。
- **Forbidden**（延续）：transferable sequence mechanism、SOTA、noise ceiling、
  13 独立模型族公平比较等；63-D sequence-map 路线保持关闭；提交/release 仍不授权。

### 方法级瓶颈的诚实定位

本次 5 条方法级组合/校准/换族路线中，4 条阴性（stacking、per-row σ、OEC α、
Student-t GBDT），1 条小正（σ-only LOO，−0.0067）。这精确界定了集成贡献的性质：
**7-member equal-weight 集成已是该数据上的局部最优**，残差中的主要不可约部分是
高删失 scaffold（1、9）的算子系统偏差与测量噪声，而非集成权重或 σ 形态可再压缩。
要在绝对 NLL 上显著前移，需新的删失感知算子表征或 prospective 数据，而非继续调
组合层。

## 4. 新增/修改文件

- `audit/repair/analyze_stacked_ensemble.py`（新增）：censoring-aware LOO stacking
- `audit/repair/residual_structure_diagnostic.py`（新增）：残差结构/σ 扫描/CRPS
- `audit/repair/operator_calibrated_ensemble.py`（新增）：σ-only 与 OEC α 的
  leave-one-fold-out 校准
- artifacts：`stacked_ensemble_analysis.json`、`residual_structure_diagnostic.json`、
  `operator_calibrated_ensemble.json`（run root）
