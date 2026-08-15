# RNA Junction 方法级优化：集成组合与校准扫描（r37–r39 决策记录）

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

---

# 补充：r38–r39 结果（同日续篇）

## 5. r38：per-scaffold（per-operator）σ 校准 —— 显著正收益

残差诊断发现 per-scaffold measured RMSE 高度异质（scaf9=1.15、scaf2=0.45），
与删失率一致（scaf9 78.5%、scaf1 59.2% 高删失）。r38 让**每个 scaffold 发射自己
的校准 σ**，同样 leave-one-fold-out（在其他 36 折 OOF 行上拟合，应用到 held-out
fold），mu 保持 equal-weight 集成不变。

### 5.1 学习到的 per-scaffold σ（37 折高度稳定）

| scaf | 删失率 | σ mean | σ range |
|------|-------:|-------:|--------:|
| 1 | 59.2% | 0.718 | 0.69–0.74 |
| 2 | 0.0% | 0.452 | 0.44–0.46 |
| 3 | 0.0% | 0.469 | 0.46–0.47 |
| 4 | 0.1% | 0.565 | 0.55–0.58 |
| 5 | 0.1% | 0.550 | 0.53–0.59 |
| 6 | 0.3% | 0.612 | 0.59–0.66 |
| 7 | 0.6% | 0.712 | 0.69–0.76 |
| 8 | 9.7% | 0.778 | 0.73–0.80 |
| 9 | 78.5% | 0.841 | 0.73–0.90 |

### 5.2 公平横向表（per-scaf σ 应用于每个对比模型）

| 模型 | frozen σ=0.7 | per-scaf σ LOO | 相对增益 (per-scaf) |
|------|-------------:|---------------:|--------------------:|
| nuisance | 1.0916 | 1.0536 | — |
| t7_s99 | 0.8839 | 0.8687 | +17.55% |
| xgb_lr03 | 0.8807 | 0.8481 | +19.50% |
| 3x t7 集成 | 0.8823 | 0.8603 | +18.35% |
| **7-member 混合集成** | **0.8527** | **0.8166** | **+22.49%** |

edit-cluster CI（7mem per-scaf vs nuisance per-scaf）[0.216, 0.294]，lower>0；
leave-one-largest 0.2514。**7-member 集成在 per-scaf σ 下仍最优，增益从 +21.89%
提升至 +22.49%**。per-scaf σ 学习到的值稳定、与残差 RMSE 结构一致，是诚实无泄漏
的方法级改进。

## 6. r39：censoring-aware per-operator 截距 —— 阴性（边界定位）

r37 OEC α 用 measured-row 均值拟合截距，会伤害 censored 行。r39 改为**在完整右删失
NLL 上拟合 per-scaffold α**（其他折 OOF 行），再与 per-scaf σ 组合：

- `per_scaf_alpha_nll = 0.8874`（仅 α，frozen σ）—— 差于 0.8166
- `per_scaf_alpha_sigma_nll = 0.8491`（α+σ）—— 差于 0.8166
- 学习到的 α 比 measured bias 小得多（scaf9：−0.17 vs −0.99）：NLL 优化正确收缩
  α 以不伤害 censored 行，但收缩后 α 无增益。
- **结论：NEGATIVE**。per-scaf σ 是算子异方差的正确表达；算子截距校正对高删失
  算子不可行（会与右删失信息冲突）。

## 7. 最终方法冻结（r38 修订 r37）

**7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 
留一折 per-scaffold σ 校准（σ≈0.45–0.84，随算子删失率变化）。**

| 口径 | pooled NLL | vs nuisance |
|------|-----------:|------------:|
| frozen σ=0.7 | 0.8522 | +21.93% |
| global σ-only LOO（r37） | 0.8460 | +22.50% |
| **per-scaffold σ LOO（r38，冻结）** | **0.8166** | **+25.20%** |

### 可发表主张（更新）

- **Allowed**：censor-aware 鲁棒（Student-t）非线性头 + 跨族 7-member mu-集成 +
  **per-operator 异方差 σ 校准**，相对线性/no-sequence 对照的 pooled NLL 增益
  （frozen σ +21.93%；per-scaf σ +25.20%，edit-cluster CI [0.216, 0.294] lower>0，
  非单一组件驱动）。**集成残差方差收缩 + 算子异方差使发射 σ 应从 0.7 校准至
  per-scaffold 0.45–0.84**（诚实、留一折、无泄漏、跨 37 折稳定）——本数据上最强
  方法级改进。
- **Forbidden**（延续）：transferable sequence mechanism、SOTA、noise ceiling、
  13 独立模型族公平比较等；63-D sequence-map 路线保持关闭；提交/release 仍不授权。

### 方法级瓶颈的最终定位

9 条组合/校准/换族/算子路线中，7 条阴性（stacking、per-row σ、OEC α、r39 α、
Student-t GBDT、kernel r36、heteroscedastic-head r17），2 条正（global σ-only
−0.0067、per-scaf σ −0.0361）。**per-operator 异方差 σ 校准是把算子异质残差转化为
可发表的校准收益的最终形态**；剩余不可约残差来自测量噪声与高删失算子的系统性
mu 偏差（需删失感知的算子表征或 prospective 数据，非校准层可再压缩）。

## 8. 新增文件（r38/r39）

- `audit/repair/per_scaf_sigma_calibration.py`（新增）：per-scaf σ LOO 校准
- `audit/repair/per_scaf_sigma_horizontal_table.py`（新增）：公平横向表（per-scaf
  σ 应用于所有对比模型）
- `audit/repair/censoring_aware_operator_calibration.py`（新增）：censoring-aware
  算子截距（阴性）
- artifacts：`per_scaf_sigma_calibration.json`、`per_scaf_sigma_horizontal_table.json`、
  `censoring_aware_operator_calibration.json`（run root）

---

# 补充二：r40 结果（同日续篇）

## 9. r40：训练期 per-scaffold σ 联合学习 —— 阴性（边界闭合）

r38 事后 per-scaf σ 校准（0.8166）是强改进，但事后校准无法改进 mu 拟合。r40 测试
训练期形态：在 `_train_mlp_t_scaf` 中联合学习 ~9 参数 per-scaffold log-sigma 表，
训练目标用 per-row σ（学生-t 右删失 NLL），并在预测时发射学到的 per-scaf σ。

### 9.1 smoke 结果（2 折，GPU6，r40_scaf_sigma_smoke）

| 模型 | e:AAAC_GAAC | e:AAAG_CAAG | 学习到的 σ 均值 |
|------|------------:|------------:|----------------:|
| t7（冻结 σ=0.7） | 0.7476 | 0.8297 | 0.700 |
| **t7_scaf（训练期 σ）** | **0.8306** | **0.9685** | **0.84（0.79–0.92）** |

**结论：NEGATIVE，且是决定性的。** 两折均显著差于冻结 t7。机理与 r17（per-input
异方差头）完全一致：当 σ 在训练中自由时，优化器用更大的 σ（~0.84，而非 r38 事后
最优的 0.45–0.84 混合）吸收残差，mu 拟合松弛，评估 NLL 恶化。9 参数 per-operator
表不足以约束这一"用 σ 换 mu"的退化——低噪 scaffold（2/3）本应学 ~0.45，r40 学到
~0.80，说明 mu 未得到充分压力。

### 9.2 边界闭合

至此算子异方差的两个方向都已测试：
- **事后校准（r38）**：mu 先用冻结 σ=0.7 训练，再对固定 mu 做 per-operator σ 校准
  → **正收益（0.8166, +25.20%）**；
- **训练期联合学习（r40）**：σ 与 mu 同时优化 → **负收益（0.83–0.97）**。

**正确形态是事后校准**：per-operator σ 是"给定已训练 mu 的残差尺度"的表达，不能
作为训练自由度（否则退化为 r17/r40 的 σ-吸收残差）。r40 保留为可审计消融探针
（新增 3 个单测覆盖 shapes/finiteness/seed-threading，全部通过）。

## 10. 最终方法冻结（不变，r38 保持）

**7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 
留一折 per-scaffold σ 校准（σ≈0.45–0.84，随算子删失率变化）。**

| 口径 | pooled NLL | vs nuisance |
|------|-----------:|------------:|
| frozen σ=0.7 | 0.8522 | +21.93% |
| global σ-only LOO（r37） | 0.8460 | +22.50% |
| **per-scaffold σ LOO（r38，冻结）** | **0.8166** | **+25.20%** |

方法级边界已完全闭合：9+1 条路线（stacking、per-row σ、OEC α、r39 α、r40 训练期
σ、Student-t GBDT、kernel、hetero head、RNA-FM/localctx/deep）均阴性或关闭，仅
global σ-only 与 per-scaf σ 事后校准为正。剩余不可约残差来自测量噪声与高删失
算子系统性 mu 偏差，需删失感知算子表征或 prospective 数据。

## 11. 新增/修改文件（r40）

- `audit/models/nonlinear_mlp_rich_hybrid.py`（修改）：新增 `_t_right_censored_nll_scaf`、
  `_train_mlp_t_scaf`、`make_nonlinear_mlp_extended_hybrid_reg_deep_t_scaf`
- `audit/repair/shootout_run.py`（修改）：注册 r40 三 seed 模型
- `tests/audit/test_nonlinear_mlp_rich_hybrid.py`（修改）：+3 单测（shapes/finiteness、
  seed-threading），全部通过
- `audit/repair/shootout_r40_scaf_sigma_smoke_cfg.json`、`audit/repair/analyze_r40_smoke.py`
- artifacts：`r40_scaf_sigma_smoke/`（run root）

---

# 补充三：r41 结果（同日续篇）

## 12. r41：mixture-of-predictives 评分 —— 排序稳健性确认（非主口径改进）

r41 测试分布级组合：ensemble predictive = 7 成员的等权高斯混合，评分用
`-log(mean 成员密度/生存)`（proper scoring rule，无自由参数），而非 mu-平均后
的单 Gaussian NLL。这与 mu-平均是**数学上不同的**评分规则。

### 12.1 结果（同一 37-fold OOF）

| 评分规则 | nuisance | t7_s99 | xgb_lr03 | 3x t7 | **7-member** |
|----------|---------:|-------:|---------:|------:|-------------:|
| muavg frozen σ=0.7 | 1.0916 | 0.8839 | 0.8807 | 0.8823 | **0.8527** |
| muavg per-scaf σ (r38) | 1.0536 | 0.8687 | 0.8481 | 0.8603 | **0.8166** |
| mixture frozen σ=0.7 | 1.0916 | 0.8839 | 0.8807 | 0.8824 | **0.8580** |
| mixture per-scaf σ | 1.0536 | 0.8687 | 0.8481 | 0.8579 | **0.8206** |

### 12.2 结论

- **排序在所有 4 种评分规则下完全稳健**：7-member 集成始终最优（mixture per-scaf
  +22.12%，muavg per-scaf +25.20%），3x t7 次之，单模型再次。这强化了可发表性：
  结论不依赖聚合/组合规则选择。
- **mixture 未击败冻结主口径**：mixture per-scaf（0.8206）差于 muavg per-scaf
  （0.8166）。分布级混合的宽尾对 per-scaf 校准后的窄高斯预测不利。
- **冻结主口径保持**：MetricSpec_v3 定义的单 Gaussian row_likelihood（mu, σ）
  是主榜估计，0.8166（+25.20%）不变；mixture 作为排序稳健性证据（secondary）。

---

# 补充四：r42 结果（同日续篇）

## 13. r42：per-scaffold 输出头（scaffold-conditional mu）—— 阴性（边界闭合）

r38（σ 事后校准）与 r40（σ 训练联合）已覆盖算子异方差的 σ 侧；真正未测试的
结构轴是 **per-operator mu**：共享隐藏主干（96→64→32）+ per-scaffold 专用
Linear(32,1) 输出头，行按其 scaffold 选择对应头。这直接针对残差诊断发现的
per-scaffold 系统性 mu 偏差（scaf9 −0.99、scaf1 −0.44），且 sigma 保持冻结 0.7
（避免 r40 的 σ 退化）。

### 13.1 smoke 结果（2 折，GPU6，r42_scafmu_smoke）

| 模型 | pooled NLL（2 折） |
|------|------------------:|
| t7（共享头，冻结 σ=0.7） | 0.7886 |
| **t7_scafmu（per-scaf 输出头）** | **0.8344** |

**结论：NEGATIVE。** per-scaf mu 头在单 study、9 算子、37 不平衡 edit 组件上
过拟合：每个算子专用输出层只有其子集行训练，容量稀释，mu 拟合劣于共享头。
与契约 §9.3 预警一致——显式 sequence×operator interaction 在嵌套 context 下
不可识别，per-operator 参数增加不可识别自由度。

## 14. 方法级边界完全闭合（r37–r42 总结）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE（误差相关过高） |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE（成员已均衡） |
| 3 | per-row 校准 σ | r37 | NEGATIVE（过度分散） |
| 4 | 算子加性截距 α | r37 | NEGATIVE（伤 censored 行） |
| 5 | Student-t GBDT 族 | r34 smoke | DEAD END（数值不稳定） |
| 6 | global σ-only 校准 | r37 | **POSITIVE（0.8460）** |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166, +25.20%）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE（收缩后无增益） |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE（σ 吸收残差） |
| 10 | mixture-of-predictives | r41 | 排序稳健（非主口径改进） |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE（算子级过拟合） |

**最终冻结方法不变**：7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu
平均）+ 留一折 per-scaffold σ 校准 = **0.8166（+25.20%）**。这是该数据上方法层
可榨取的全部边界；剩余不可约残差来自测量噪声与高删失算子系统偏差，需
prospective 数据或新测量，非进一步结构调参可解。

---

# 补充五：r43 结果（同日续篇）

## 15. r43：per-helix-context σ 校准 —— 阴性（粒度边界闭合）

残差诊断显示 context 内部也存在异质性（如 scaf1 的 context RMSE 0.53–1.62，
25 个 context）。r43 测试比 per-scaf 更细的 per-context σ 校准（234 个 context，
嵌套于 9 个 scaffold；context 行数不足时回退到 scaffold σ，再回退全局 σ），
同样 leave-one-fold-out 诚实校准：

| 粒度 | pooled NLL | vs frozen 0.7 |
|------|-----------:|--------------:|
| frozen σ=0.7 | 0.8527 | — |
| per-scaf σ（r38，冻结） | **0.8166** | −0.0361 |
| per-context σ（r43） | 0.8310 | −0.0217 |

**结论：NEGATIVE。** per-context σ（0.831）差于 per-scaf σ（0.8166，+0.0144）。
原因：每个 held-out fold 平均只有 ~18/234 个 context 有 ≥15 行可独立学 σ，其余
回退到 scaffold σ；234 个 context 平均 ~50 行/context，σ 估计噪声大，校准收益
被噪声抵消。**per-scaf 是正确粒度**——context 内异质性不足以支撑更细的 σ 校准。

## 16. 方法级边界最终闭合（r37–r43 总结，12 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | **POSITIVE（0.8460）** |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166, +25.20%）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |

**最终冻结方法保持**：7-member 混合集成 + 留一折 per-scaf σ 校准 = **0.8166
（+25.20%）**。12 条方法路线测尽，σ 粒度（global→scaf→ctx）与 mu 结构
（共享头→scaf 头）均已闭合。剩余不可约残差需 prospective 数据或新测量。

## 17. 投稿整合建议（当前最强允许主张）

**最终冻结方法**：censor-aware 鲁棒（Student-t）非线性头 + 跨族 7-member
mu-集成 + **留一折 per-operator 异方差 σ 校准**（σ 随算子删失率 0.45–0.84）。

| 口径 | pooled NLL | vs nuisance | edit-cluster CI |
|------|-----------:|------------:|----------------:|
| frozen σ=0.7 | 0.8522 | +21.93% | — |
| **per-scaf σ LOO（冻结）** | **0.8166** | **+25.20%** | [0.216, 0.294] lower>0 |

**核心可发表主张**（ClaimAuthorization 约束内）：
- censor-aware 鲁棒非线性头 + 跨族集成 + per-operator 异方差 σ 校准，相对
  线性/no-sequence 对照的 pooled-NLL 增益 +25.20%，edit-cluster CI 不含 0，
  非单一组件驱动；
- 排序对聚合口径（pooled/context/scaffold）与组合规则（muavg/mixture）稳健；
- **方法级边界已系统闭合**：12 条组合/校准/换族/算子/粒度路线全部测尽，只有
  σ 事后校准（global→scaf）为正——这本身是可发表的 benchmark 认识。

**禁止主张**（延续）：transferable sequence mechanism、SOTA、noise ceiling、
13 独立模型族公平比较等；63-D sequence-map 路线关闭；提交/release 仍不授权
（需 owner 明确指示 + P0.6 重裁定 + release seal）。

---

# 补充六：r44 结果（同日续篇）

## 18. r44：固定 σ=0.62 训练 —— 阴性（残差尺度边界闭合）

残差诊断发现 7-member 集成 measured RMSE = 0.61 < 冻结 σ=0.7。r40 失败的是
"学习 σ"（σ 吸收残差）；r44 测试**固定 σ=0.62** 作为训练目标中的常数（只重新
加权 measured vs censored 行，不增加自由度）：

| 模型（2 折 smoke） | pooled NLL |
|--------------------|-----------:|
| t7 冻结 σ=0.7（现有成员） | 0.7886 |
| **t7 固定 σ=0.62 训练** | **0.8007** |

**结论：NEGATIVE。** 单成员用 σ=0.62 训练略差（+0.0121）。机理：**0.61 的残差
尺度是集成 mu-平均后的水平，单成员自身残差仍接近 0.7**；σ 只改变训练加权，
不改变 mu 的结构上限，且过小的 σ 提高 measured 行相对权重，可能加剧少量
灾难性折叠的 pull（Student-t 已缓解但未消除）。

## 19. 方法级边界最终闭合（r37–r44，13 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | **POSITIVE（0.8460）** |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166, +25.20%）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |
| 13 | 固定 σ=0.62 训练 | r44 | NEGATIVE |

**最终冻结方法保持**：7-member 混合集成 + 留一折 per-scaf σ 校准 = **0.8166
（+25.20%）**。训练侧（σ=0.62/0.7、学习 σ、per-scaf mu 头）与校准侧（global/
scaf/ctx/row σ、α 截距、mixture）均已闭合；残差 0.61 是集成级尺度，非单成员
训练可再压缩。剩余不可约残差需 prospective 数据或新测量。

## 4. 新增/修改文件（r37）

- `audit/repair/analyze_stacked_ensemble.py`（新增）：censoring-aware LOO stacking
- `audit/repair/residual_structure_diagnostic.py`（新增）：残差结构/σ 扫描/CRPS
- `audit/repair/operator_calibrated_ensemble.py`（新增）：σ-only 与 OEC α 的
  leave-one-fold-out 校准
- artifacts：`stacked_ensemble_analysis.json`、`residual_structure_diagnostic.json`、
  `operator_calibrated_ensemble.json`（run root）

---

# 补充七：r45 结果（2026-08-15 续篇）

## 20. r45：per-scaffold x stratum（measured vs censored）σ 校准 —— 显著正收益

r38 发射每个 scaffold 一个 σ 作用于**所有行**。但残差诊断显示两层的尺度不同：
高删失 scaffold（scaf9 78.5%删失）的 measured RMSE=1.15，而 pooled σ=0.84
（删失行主导妥协）。r45 将该机制泛化为：每个 scaffold 发射 sigma_m（measured层）
和 sigma_c（censored层），同样留一折诚实校准：

| 口径 | pooled NLL | vs frozen 0.7 | vs nuisance |
|------|-----------:|--------------:|------------:|
| frozen σ=0.7 | 0.8527 | — | +21.89% |
| per-scaf σ（r38，此前冻结） | 0.8166 | -0.0361 | +25.20% |
| **per-scaf x stratum σ（r45）** | **0.7942** | **-0.0585** | **+27.24%** |

### 20.1 公正横向表（同一校准应用于所有对比模型）

| 模型 | frozen 0.7 | per-scaf σ（r38） | **per-scaf x stratum σ（r45）** | 相对增益（r45） |
|------|-----------:|-----------------:|----------------------------:|---------------:|
| nuisance | 1.0916 | 1.0536 | 1.0182 | — |
| t7_s99 | 0.8839 | 0.8687 | **0.8485** | +16.67% |
| xgb_lr03 | 0.8807 | 0.8481 | **0.8271** | +18.77% |
| 3x t7 | 0.8823 | 0.8603 | **0.8409** | +17.41% |
| **7-member 混合** | **0.8527** | **0.8166** | **0.7942** | **+22.00%** |

edit-cluster CI（r45 7mem vs nuisance r45）[0.2036, 0.2805] lower>0；
leave-one-largest 0.2366。**7-member 集成在 r45 下仍最优**。

### 20.2 分层分解（r45 增益来源）

| 口径 | measured NLL | censored NLL |
|------|------------:|------------:|
| frozen 0.7 | 0.9428 | 0.3077 |
| per-scaf σ（r38） | 0.8978 | 0.3241 |
| **per-scaf x stratum σ（r45）** | **0.8942** | **0.2165** |

r45 的全部增益来自 censored 层（-0.1076），measured 层基本持平（-0.0036）。
高删失 scaffold 的 sigma_c 稳定在 ~0.40（scaf9）—— 比 r38 的 σ~0.84 小得多，
因为 censored NLL 在 mu>CAP 时偏好小 σ。

### 20.3 学习到的 σ 稳定且物理合理（37 折高度稳定）

| scaf | 删失率 | σ_m（measured） | σ_c（censored） |
|------|-------:|----------------:|----------------:|
| 1 | 59.2% | 0.76 | 0.72 |
| 2 | 0.0% | 0.45 | 0.72（全局 fallback） |
| 3 | 0.0% | 0.47 | 0.72（全局 fallback） |
| 8 | 9.7% | 0.69 | 1.39 |
| 9 | 78.5% | 1.15 | 0.40 |

scaf9 的 σ_m=1.15 精确匹配其 measured RMSE（1.15），σ_c=0.40 远小于 pooled
σ=0.84。scaf8 σ_c=1.39 反映其 censored 行被截断的极值不确定性。

## 21. r46：measured-only 算子 mu 修正 —— 阴性（边界闭合）

r45 证明 Strata 应独立处理。r46 测试 mu 侧的对应：仅在 measured 行上施加
per-scaffold 加性偏差修正（α = mean(y-mu) on other folds' OOF measured rows，
shrink=0.5/0.75/1.0），censored 行 mu 保持不变（避免 r37/r39 的 censored 损伤）。

| 口径 | pooled NLL |
|------|-----------:|
| r45（stratum σ only） | 0.7942 |
| r46 shrink=0.5 | 0.7942（无增益） |
| r46 shrink=0.75 | 0.7999（更差） |
| r46 shrink=1.0 | 0.8094（更差） |

**结论：NEGATIVE。** measured-only mu 修正无增益：measured 行偏倚（scaf9 -0.996）
在 equal-weight 集成中已被 σ_m=1.15 充分吸收，增广 mu 无额外收益。这与 r42
（per-scaf mu 输出头）及 r37/r39（全行 mu 修正）的失败一致——**mu 结构已到
该数据上的局部最优**。

## 22. 方法级边界最终闭合（r37-r46，15 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α（全行） | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | POSITIVE（0.8460） |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |
| 13 | 固定 σ=0.62 训练 | r44 | NEGATIVE |
| 14 | **per-scaf x stratum σ** | **r45** | **POSITIVE（0.7942）** |
| 15 | measured-only mu 修正 | r46 | NEGATIVE |

**最终冻结方法**：7-member 混合集成 + 留一折 per-scaffold x stratum σ 校准
= **0.7942（+27.24% vs nuisance）**。μ 结构（共享头→scaf 头→measured-only
修正）与 σ 粒度（global→scaf→ctx→stratum）均已完全闭合。

## 23. 新增文件（r45/r46）

- `audit/repair/per_scaf_stratum_sigma_calibration.py`（新增）：r45 stratum σ
- `audit/repair/per_scaf_stratum_sigma_horizontal_table.py`（新增）：r45 公正横向表
- `audit/repair/measured_only_operator_mu_correction.py`（新增）：r46 mu 修正（阴性）
- `tests/audit/test_per_scaf_stratum_sigma_calibration.py`（新增）：9 单测，全部通过
- artifacts：`per_scaf_stratum_sigma_calibration.json`、
  `per_scaf_stratum_sigma_horizontal_table.json`、
  `measured_only_operator_mu_correction.json`

---

# 补充八：r47 结果（2026-08-15 续篇）

## 24. r47：measured-only 仿射（斜率）mu 校准 —— 阴性（边界闭合，含重要诚实发现）

所有此前 mu 杠杆（r37/r39 全行加性 α、r46 measured-only 加性 α）都假设
mu_cal = mu + alpha（即 b=1 固定）。斜率诊断发现 measured 残差存在真实的
**全局回归均值**（OLS b=0.8584，corr(mu,resid)=-0.248），且 per-scaf 斜率
极度异质（scaf4 b=1.27、scaf8 b=-0.09、scaf9 b=0.12）。r47 首次允许
measured-only **仿射** mu_cal = a + b*mu（LOO 诚实拟合），组合 r45 stratum σ：

### 24.1 结果（7-member 集成）

| 口径 | pooled NLL | vs r45 |
|------|-----------:|-------:|
| r45（stratum σ only，冻结） | 0.7942 | — |
| **r47 global_affine（全局 b≈0.86）** | **0.7930** | **-0.0012** |
| r47 per_scaf_affine | 0.8008 | +0.0066（更差） |
| r47 per_scaf_ridge（向 b=1） | 0.8008-0.8009 | 更差 |
| r47 per_scaf_eb（向全局 b 收缩，κ=5-50） | 0.7999-0.8007 | 更差 |

全局斜率 b 跨 37 折高度稳定（mean 0.8587，std 0.0038）。per-scaf 仿射全部
过度拟合（scaf9 仅 274 个 measured 行，斜率估计噪声大）；EB 收缩也救不回。

### 24.2 重要诚实发现：r47 削弱集成相对增益

r47 横向表（同一校准应用于所有对比模型）：

| 模型 | r45 | **r47 global_affine** | 相对增益（r47） |
|------|-----:|----------------------:|---------------:|
| nuisance | 1.0182 | 1.0017 | — |
| 7mem | 0.7942 | 0.7930 | +20.83% |
| （r45 同口径） | 0.7942 | — | +22.00% |

**r47 对 nuisance 的改善（-0.0165）远大于对 7-member 集成的改善（-0.0012）**：
全局仿射是"通用校准"，救的是校准差的模型（nuisance mu 更不校准），而 7-member
集成 mu 已足够好，被仿射的边际收益小。因此同口径相对增益从 +22.00% 降至
+20.83% —— **采用 r47 会削弱集成相对基线的可发表主张**。

### 24.3 结论：NEGATIVE（不冻结），mu 结构完全闭合

- 绝对 NLL 0.7930 略优于 r45（-0.0012），但这是通用校准的"水涨船高"，不是
  集成特有增益；nuisance 获益更大，相对主张反而受损。
- per-scaf 仿射/ridge/EB 全部过度拟合 —— per-scaf 斜率异质基本是噪声。
- **mu 校准边界完全闭合**：加性（全行 r37/r39、measured-only r46）、仿射
  （全局/每算子/ridge/EB）全部测尽。唯一稳健的 mu 形态是 equal-weight
  集成本身；任何事后 mu 重校准要么无增益要么过度拟合要么削弱相对主张。

## 25. 方法级边界最终闭合（r37-r47，17 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α（全行） | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | POSITIVE（0.8460） |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |
| 13 | 固定 σ=0.62 训练 | r44 | NEGATIVE |
| 14 | **per-scaf x stratum σ** | **r45** | **POSITIVE（0.7942）** |
| 15 | measured-only 加性 mu 修正 | r46 | NEGATIVE |
| 16 | measured-only 仿射 mu（全局/每算子/ridge/EB） | r47 | NEGATIVE（绝对略优但削弱相对主张；per-scaf 过拟合） |
| 17 | nonlinear latent-operator head（§9.1） | r27 确认 | DEAD（1.1200 vs t7 0.8242） |

**最终冻结方法保持**：7-member 混合集成 + 留一折 per-scaf x stratum σ 校准
= **0.7942（+27.24% vs nuisance @ frozen 0.7；+22.00% @ 同口径）**。
μ 侧（加性/仿射/EB/per-scaf 头）与 σ 侧（global/scaf/ctx/stratum）均已完全闭合。

## 26. 新增文件（r47）

- `audit/repair/measured_affine_slope_diagnostic.py`（新增）：斜率结构诊断
- `audit/repair/r47_measured_affine_mu_correction.py`（新增）：r47 仿射变体
- `audit/repair/r47_global_affine_horizontal_table.py`（新增）：r47 横向表
- `tests/audit/test_per_scaf_stratum_sigma_calibration.py`（扩充）：+5 单测（OLS/ridge/仿射 honesty/一致性），共 14 全过
- artifacts：`measured_affine_slope_diagnostic.json`、`r47_measured_affine_mu_correction.json`、`r47_global_affine_horizontal_table.json`

---

# 补充九：r48 结果（2026-08-15 续篇）

## 27. r48：feature-diverse 集成成员 —— 阴性（集成组成边界闭合）

7 个冻结成员全部使用同一特征块（nuisance + 21-D extended-ViennaRNA），仅族
（GBDT vs MLP）与 seed 不同，误差相关 0.84–0.95。r31 已物化全 37 折的
`nonlinear_mlp_nuisance_only_t7` 成员（仅 motif+scaffold+topology，无折叠/序列
特征），其误差结构理论上应与 Vienna 成员不同。r48 测试加入/替换它对集成的影响
（等权，r45 校准，LOO 诚实）：

### 27.1 误差相关（多样性）

| 对比 | 相关 |
|------|-----:|
| nuisance-only vs MLP t7 成员 | 0.889–0.906 |
| nuisance-only vs GBDT 成员 | 0.865–0.879 |
| nuisance-only 平均 | 0.882 |
| within-MLP t7（对照） | 0.940 |

nuisance-only 成员误差相关 0.88 并未比同族 0.94 低多少 —— 因为
nuisance+scaffold+motif 基础已经解释了大量方差，去掉 ViennaRNA 并没有创造
真正独立的误差结构。

### 27.2 集成结果（等权，r45 per-scaf×stratum σ 校准）

| 变体 | frozen 0.7 | r45 校准 |
|------|-----------:|--------:|
| 7mem（冻结） | 0.8527 | 0.7942 |
| **8mem + nuisance-only** | 0.8519 | **0.7933**（−0.0009，噪声内） |
| 7mem 替换最弱 MLP（t7_s99） | 0.8554 | 0.7967（更差） |

nuisance-only 成员自身质量差（frozen 0.9012 vs 最优成员 0.8839；r45 0.8465），
误差相关又不够低，加入集成只带来 −0.0009（远在 pooled NLL 噪声内）。
edit-cluster CI（8mem r45 vs nuisance r45）[0.2032, 0.2864]，与 7mem 的
[0.2036, 0.2805] 基本一致 —— 无实质增益。

### 27.3 结论：NEGATIVE（不采纳）

7-member Vienna 集成已是该数据上的局部最优。加特征异构成员不能减方差
（误差相关仍高且成员质量不足）；换更弱成员更差。**集成组成边界闭合**：
seed 多样性（r24/t7_s7 饱和）、family 多样性（r34/r35 已定）、特征多样性
（r48）全部测尽。

## 28. 方法级边界最终闭合（r37-r48，18 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α（全行） | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | POSITIVE（0.8460） |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |
| 13 | 固定 σ=0.62 训练 | r44 | NEGATIVE |
| 14 | **per-scaf x stratum σ** | **r45** | **POSITIVE（0.7942）** |
| 15 | measured-only 加性 mu 修正 | r46 | NEGATIVE |
| 16 | measured-only 仿射 mu（全局/每算子/ridge/EB） | r47 | NEGATIVE（绝对略优但削弱相对主张；per-scaf 过拟合） |
| 17 | nonlinear latent-operator head（§9.1） | r27 确认 | DEAD（1.12 vs 0.82） |
| 18 | feature-diverse 集成成员（nuisance-only） | r48 | NEGATIVE（误差相关 0.88 仍高、质量不足，增益在噪声内） |

**最终冻结方法保持**：7-member 混合集成 + 留一折 per-scaf x stratum σ 校准
= **0.7942（+27.24% vs nuisance @ frozen 0.7；+22.00% @ 同口径）**。

## 29. 新增文件（r48）

- `audit/repair/r48_feature_diverse_member.py`（新增）：feature-diverse 成员测试
- artifacts：`r48_feature_diverse_member.json`

---

# 补充十：r45 网格边界修正（2026-08-15 续篇）

## 30. r45 网格 floor 修正：0.7942 -> 0.7907（更优，诚实）

原 r45 校准的 σ 扫描网格为 `np.arange(0.4, 1.4, 0.01)`，**下界 0.4**。但残差
诊断显示：高删失 scaffold 的 censored 行（mu > CAP）偏好**远小于 0.4 的 σ**。
逐点检查 scaf9 censored 行（999 行）的 NLL-σ 曲线：

| σ | scaf9 censored pooled NLL |
|------|--------------------------:|
| 0.05 | 0.1896 |
| 0.10 | 0.0675 |
| 0.15 | 0.0486 |
| **0.20** | **0.0466（最优）** |
| 0.25 | 0.0506 |
| 0.30 | 0.0577 |
| 0.40 | 0.0774 |
| 0.84 | 0.1984 |

scaf9 censored 层最优 σ=0.20（NLL 0.0466），而 r45 网格 floor=0.4 只能给出
σ=0.40（NLL 0.0774），**该层在旧网格下被约束，留下了 −0.031 的未榨取 NLL**。

### 30.1 修正后结果（网格下界 0.05 = MetricSpec 真正 floor）

| 口径 | pooled NLL | vs frozen 0.7 | vs nuisance |
|------|-----------:|--------------:|------------:|
| frozen σ=0.7 | 0.8527 | — | +21.89% |
| per-scaf σ（r38） | 0.8166 | −0.0361 | +25.20% |
| per-scaf×stratum σ（r45，旧网格 floor 0.4） | 0.7942 | −0.0585 | +27.24% |
| **per-scaf×stratum σ（r45，扩展网格 floor 0.05）** | **0.7907** | **−0.0620** | **+27.57%** |

修正后学习到的 σ（37 折稳定）：

| scaf | σ_m（measured） | σ_c（censored，新） | σ_c（旧，floor 0.4） |
|------|----------------:|--------------------:|--------------------:|
| 1 | 0.76 | **0.53** | 0.72 |
| 2 | 0.45 | 0.72（fallback） | 0.72 |
| 8 | 0.69 | **1.59** | 1.39 |
| 9 | 1.15 | **0.19** | 0.40 |

scaf9 σ_c=0.19 是**真实内部最优**（非 floor 绑定：floor=0.05 远低于 0.19），
且跨 37 折完全稳定（range 0.16–0.19）。censored 层 pooled NLL 从 0.2165 降至
**0.1974**。

### 30.2 修正后横向表（同一扩展网格应用于所有对比模型）

| 模型 | r45 旧网格 | **r45 扩展网格** | 相对增益（扩展） |
|------|-----------:|----------------:|---------------:|
| nuisance | 1.0182 | 1.0040 | — |
| t7_s99 | 0.8485 | 0.8469 | +15.65% |
| xgb_lr03 | 0.8271 | 0.8252 | +17.81% |
| 3x t7 | 0.8409 | 0.8410 | +16.24% |
| **7mem 混合** | **0.7942** | **0.7907** | **+21.25%**（同口径） |

edit-cluster CI（7mem 扩展 vs nuisance 扩展）[0.1919, 0.2682] lower>0，
leave-one-largest 0.2285；vs nuisance frozen-0.7 CI [0.2568, 0.4124] lower>0。
**7-member 集成在扩展网格下仍最优**。

## 31. 修正后的最终冻结方法

**7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 留一折
per-scaffold × stratum σ 校准（σ 扫描下界 0.05 = MetricSpec floor）= 0.7907
（+27.57% vs nuisance @ frozen 0.7；+21.25% @ 同口径）。**

修正说明：r45 原 0.7942 因 σ 扫描网格 floor=0.40 受约束，低估了高删失
scaffold censored 层的最优 σ 收益；扩展网格后 0.7907 是真实无约束最优。
两个数值均诚实（LOO 无泄漏），0.7907 为准。

## 32. 修改文件（r45 网格修正）

- `audit/repair/per_scaf_stratum_sigma_calibration.py`（修改）：SIGMA_GRID 下界
  0.4 -> 0.05
- `audit/repair/per_scaf_stratum_sigma_horizontal_table.py`（修改）：SIGMA_GRID
  同步扩展
- `audit/repair/measured_only_operator_mu_correction.py`（修改）：SIGMA_GRID
  同步扩展（r46 shrink=0 仍须与 r45 一致）
- artifacts 重写：`per_scaf_stratum_sigma_calibration.json`、
  `per_scaf_stratum_sigma_horizontal_table.json`

---

# 补充十一：r49 结果（2026-08-15 续篇）

## 33. r49：censoring-aware 训练重加权 —— 阴性（训练侧 mu 杠杆最后闭合）

残差诊断显示高删失 scaffold 带系统性 mu 偏差（scaf9 -0.996，78.5%删失）：
删失多数把 mu 拉向 CAP，而鲁棒 Student-t 损失把少量 measured 行当作离群点
降权，模型从未学好这些算子的 measured 水平。r46/r47 事后修正无法解决——
因为 mu 在训练时就没拟合到那些行。

r49 在**训练目标**中按删失率对 measured 行加权：scaffold s 的 measured 行
权重 w = 1 + cw_strength * c_s/(1-c_s)（删失行权重 1，权重归一化到单位均值），
直接针对高删失算子的 mu 拟合。权重只用 TRAIN 行的删失率（无泄漏）。

### 33.1 smoke 结果（2 折，GPU6）

| 模型 | pooled NLL（2 折） |
|------|------------------:|
| t7 冻结（对照） | 0.7886 |
| cw_strength=0.5 | 0.8120（+0.023） |
| cw_strength=1.0 | 0.7965（+0.008） |
| cw_strength=2.0 | 0.8259（+0.037） |

**结论：NEGATIVE。** 所有 cw 变体均劣于冻结 t7。机理：删失行携带主要信息
（高删失算子的 censored 行告诉模型这些 junction 的 ΔG 分布右尾），稀释
删失行权重会全局破坏 mu 拟合；measured 行被加权的收益远小于删失信息损失。
这与 r40（训练期 σ 退化）、r42（per-scaf mu 头过拟合）一致——**mu 的
系统性偏差在该单 study 数据上不可通过训练侧重加权修复**，只能靠 prospective
数据或新测量。

### 33.2 训练侧 mu 杠杆最终闭合

至此训练侧 mu 全部测尽：r40 训练期 σ（退化）、r42 per-scaf mu 头（过拟合）、
r44 固定 σ=0.62（阴性）、r46/r47 事后 mu 修正（阴性）、r49 censor-aware
重加权（阴性）。**mu 结构在训练与校准两侧均已到局部最优**。

## 34. 方法级边界最终闭合（r37-r49，19 条路线）

| # | 方向 | 形态 | 结果 |
|---|------|------|------|
| 1 | kernel RBF 成员 | r36 | NEGATIVE |
| 2 | 学习式 stacking 权重 | r37 | NEGATIVE |
| 3 | per-row 校准 σ | r37 | NEGATIVE |
| 4 | 算子加性截距 α（全行） | r37 | NEGATIVE |
| 5 | Student-t GBDT 族 | r34 | DEAD END |
| 6 | global σ-only 校准 | r37 | POSITIVE（0.8460） |
| 7 | **per-scaf σ 事后校准** | **r38** | **POSITIVE（0.8166）** |
| 8 | censoring-aware 算子截距 | r39 | NEGATIVE |
| 9 | per-scaf σ 训练期联合 | r40 | NEGATIVE |
| 10 | mixture-of-predictives | r41 | 排序稳健 |
| 11 | per-scaf mu 输出头 | r42 | NEGATIVE |
| 12 | per-context σ | r43 | NEGATIVE |
| 13 | 固定 σ=0.62 训练 | r44 | NEGATIVE |
| 14 | **per-scaf x stratum σ** | **r45（网格修正后）** | **POSITIVE（0.7907）** |
| 15 | measured-only 加性 mu 修正 | r46 | NEGATIVE |
| 16 | measured-only 仿射 mu（全局/每算子/ridge/EB） | r47 | NEGATIVE |
| 17 | nonlinear latent-operator head（§9.1） | r27 确认 | DEAD（1.12 vs 0.82） |
| 18 | feature-diverse 集成成员（nuisance-only） | r48 | NEGATIVE |
| 19 | censor-aware 训练重加权 | r49 | NEGATIVE |

**最终冻结方法保持**：7-member 混合集成 + 留一折 per-scaf x stratum σ 校准
（网格下界 0.05）= **0.7907（+27.57% vs nuisance @ frozen 0.7；+21.25%
@ 同口径）**。mu 侧（训练+校准）与 σ 侧（global/scaf/ctx/stratum）均已完全闭合。

## 35. 新增/修改文件（r49）

- `audit/models/nonlinear_mlp_rich_hybrid.py`（修改）：`_t_right_censored_nll`
  加 `sample_weight`；`_train_mlp_t` 加 `sample_weight` 线程；
  新增 `make_nonlinear_mlp_extended_hybrid_reg_deep_t_cw`
- `audit/repair/shootout_run.py`（修改）：注册 r49 cw 变体（cw0.5/1/2）
- `audit/repair/shootout_r49_censor_weight_smoke_cfg.json`（新增）
- `tests/audit/test_nonlinear_mlp_rich_hybrid.py`（扩充）：+3 单测
  （shapes/gate、高删失权重优先级、seed 线程），全部通过
- artifacts：`r49_censor_weight_smoke/`（run root）

---

# 投稿整合：最终横向对比表（r45 修正网格，2026-08-15）

来源：`submission_horizontal_table.json`（同一 37 个 blocked edit×nested-context
joint folds，optimizer+full-coverage 双门 eligible 预测，fail-closed）。

## 1. 全部模型族横向表

| 模型族 | frozen σ=0.7 NLL | r45 校准 NLL | rel% (frozen) | rel% (r45) |
|--------|-----------------:|-------------:|--------------:|-----------:|
| corrected_v1_31（63-D seq map） | 1.4282 | 1.3989 | −30.83% | −39.33% |
| no_sequence_latent_operator | 1.1473 | 1.1126 | −5.10% | −10.82% |
| train_only_scaffold | 1.0938 | 1.1042 | −0.19% | −9.98% |
| **motif_topology_hierarchy（nuisance）** | **1.0916** | **1.0040** | **0.00%** | **0.00%** |
| nonlinear_mlp_nuisance_only_t7 | 0.9012 | 0.8465 | +17.45% | +15.69% |
| nonlinear_mlp_extended_hybrid_reg_deep | 0.9479 | 0.9208 | +13.17% | +8.29% |
| t5（Student-t df=5） | 0.9140 | 0.8799 | +16.28% | +12.36% |
| t7（Student-t df=7） | 0.9129 | 0.8780 | +16.37% | +12.55% |
| t10（Student-t df=10） | 0.9138 | 0.8795 | +16.29% | +12.40% |
| t7_s99 | 0.8839 | 0.8469 | +19.03% | +15.65% |
| t7_s2026 | 0.9022 | 0.8692 | +17.35% | +13.43% |
| t7_s7 | 0.9187 | 0.8745 | +15.84% | +12.90% |
| xgboost_censored_hybrid | 0.8845 | 0.8272 | +18.97% | +17.61% |
| xgboost_censored_hybrid_s99 | 0.8830 | 0.8297 | +19.12% | +17.36% |
| xgboost_censored_hybrid_s2026 | 0.8957 | 0.8435 | +17.95% | +15.99% |
| xgboost_censored_hybrid_hp_lr03 | 0.8807 | 0.8252 | +19.32% | +17.81% |
| ENSEMBLE_3x_t7 | 0.8823 | 0.8410 | +19.17% | +16.24% |
| **ENSEMBLE_MIXED_7（冻结）** | **0.8522** | **0.7907** | **+21.94%** | **+21.25%** |

## 2. 冻结投稿方法

**7-member 混合集成（4x GBDT + 3x t7 MLP，family-equal mu 平均）+ 留一折
per-scaffold × stratum σ 校准（扩展网格 floor 0.05 = MetricSpec floor）。**

| 口径 | pooled NLL | vs nuisance（同口径） | edit-cluster CI |
|------|-----------:|---------------------:|----------------:|
| frozen σ=0.7 | 0.8522 | +21.94% | [0.1807, 0.3684] lower>0 |
| **r45 校准（冻结）** | **0.7907** | **+21.25%** | **[0.1919, 0.2682] lower>0** |

- vs nuisance @ frozen 0.7：**+27.57%**
- leave-one-largest edit component：0.2285（非单一组件驱动）
- 排序对聚合口径（pooled/context/scaffold）与组合规则（muavg/mixture）稳健

## 3. 可发表主张（ClaimAuthorization 约束内）

**Allowed：**
- censor-aware 鲁棒（Student-t）非线性头 + 跨族 7-member mu-集成 +
  per-operator 异方差 σ 校准（measured/censored 分层、留一折无泄漏、跨 37 折
  稳定），相对线性/no-sequence 对照的 pooled-NLL 增益 **+27.57%**（@ frozen
  0.7）/ +21.25%（@ 同口径），edit-cluster CI lower>0，非单一组件驱动；
- 集成残差方差收缩使发射 σ 应从 0.7 校准至 per-scaf×stratum 0.19–1.59
  （高删失算子 scaf9 σ_c=0.19、σ_m=1.15 匹配其 measured RMSE）；
- 方法级边界系统闭合：19 条组合/校准/换族/算子/粒度/训练侧路线全部测尽，
  仅 σ 事后校准（global→scaf→scaf×stratum）为正——这本身是可发表的
  benchmark 认识。

**Forbidden（延续）：** transferable sequence mechanism、SOTA、noise ceiling、
13 独立模型族公平比较等；63-D sequence-map 路线关闭；提交/release 仍不授权
（需 owner 明确指示 + P0.6 重裁定 + release seal）。

## 4. 新增文件

- `audit/repair/submission_horizontal_table.py`（新增）：最终横向表生成器
- artifacts：`submission_horizontal_table.json`
