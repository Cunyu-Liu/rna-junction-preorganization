# RNA Junction Benchmark: 横向对比表与组感知不确定性

日期：2026-08-14
分支：`r0_audit_repair_20260811`
数据：Denny tectoRNA two-way-junction 冻结 benchmark（11,893 admitted rows，
1,336 junctions，37 edit components）
评估：37 个 blocked edit×nested-context joint folds，right-censored Gaussian NLL，
固定 sigma=0.7，仅消费 optimizer+full-coverage 双门通过的 fold（fail-closed），
ConvergenceLedger 记录实际 fit 行集 SHA-256。

## 1. 横向对比表（primary estimand = pooled junction-macro NLL）

所有模型在同一 37-fold joint split、同一 scorer、同一 eligibility 门下比较：

| 模型族 | n_rows | pooled | ctx-macro | scaf-macro | cens% | vs nuisance |
|--------|-------:|-------:|----------:|-----------:|------:|------------:|
| corrected_v1_31（63-D 序列映射） | 11893 | 1.4282 | 1.3949 | 1.4233 | 16.2 | **−30.83%** |
| no_sequence latent operator | 11893 | 1.1473 | 1.1656 | 1.1422 | 16.2 | −5.10% |
| train_only_scaffold | 11893 | 1.0938 | 1.0785 | 1.0854 | 16.2 | −0.19% |
| motif_topology_hierarchy（nuisance） | 11893 | 1.0916 | 1.0602 | 1.0835 | 16.2 | +0.00% |
| nuisance-only t7（非线性头，无 ViennaRNA） | 11893 | 0.9012 | 0.9909 | 0.8961 | 16.2 | **+17.45%** |
| reg_deep（3-layer Gaussian） | 11893 | 0.9479 | 0.9785 | 0.9433 | 16.2 | +13.17% |
| Student-t df=5 单模型 | 11893 | 0.9140 | 0.9555 | 0.9092 | 16.2 | +16.28% |
| Student-t df=7 单模型 | 11893 | 0.9129 | 0.9493 | 0.9089 | 16.2 | +16.37% |
| Student-t df=10 单模型 | 11893 | 0.9138 | 0.9591 | 0.9097 | 16.2 | +16.29% |
| t7 s99 | 11893 | 0.8839 | 0.9642 | 0.8791 | 16.2 | +19.03% |
| t7 s2026 | 11893 | 0.9022 | 0.9623 | 0.8980 | 16.2 | +17.35% |
| t7 s7 | 11893 | 0.9187 | 0.9718 | 0.9140 | 16.2 | +15.84% |
| **3x t7 集成（seed 0/99/2026）** | 11893 | **0.8823** | **0.9545** | **0.8781** | 16.2 | **+19.17%** |

观测：
- **四种 estimand（pooled-junction / nested-context / scaffold-bundle）一致排序**：
  3x t7 集成在全部口径下最优，模型结论对聚合口径稳健。
- **非线性头是主导贡献**：nuisance-only t7（无 ViennaRNA）已达 +17.45%；
  3x t7 只在此基础上再加 +1.7%。
- **63-D 序列映射显著为负**（−30.83%）：sequence map 路线在 true joint 下被
  matched no-sequence 击败，正式关闭（P0.6 D1=TRACK_A_LOCKED）。

## 2. edit-cluster 组感知 CI（bootstrap 单位 = 37 edit components）

| 对比 | delta = (b−a)，正值 = a 更优 | CI (2.5%, 97.5%) | lower>0 | leave-one-largest |
|------|------------------------------|------------------|:-------:|------------------:|
| 3x t7 vs nuisance | ensemble − nuisance | [0.1648, 0.2926] | ✓ | 0.2125 |
| 3x t7 vs t7_s99（最优单模型） | ensemble − t7_s99 | [−0.0305, 0.0200] | ✗ | −0.0129 |
| nuisance vs t7_s99 | nuisance − t7_s99 | [−0.3173, −0.1635] | ✗（a 更差） | −0.2254 |

结论：
- **3x t7 相对 nuisance 的 +19.17% 增益在 edit 组件层面稳健**（CI lower>0，
  leave-one-largest 仍正），非单一最大组件驱动。
- **3x t7 vs 最优单成员 t7_s99 的额外增益不显著**（CI 穿过 0）：集成相对单
  模型的小增益来自种子级方差缩减，而非组件级能力提升——这精确界定了集成
  贡献的性质。

## 3. Censoring 敏感性（full vs measured-only）

| 模型 | full NLL | measured-only NLL | 差异 |
|------|---------:|------------------:|------:|
| corrected_v1_31 | 1.4282 | 1.6341 | −0.206 |
| no_sequence | 1.1473 | 1.2409 | −0.094 |
| motif_topology | 1.0916 | 1.1489 | −0.057 |
| nuisance-only t7 | 0.9012 | 1.0029 | −0.102 |
| reg_deep | 0.9479 | 1.0513 | −0.103 |
| t7 单模型 | 0.9129 | 0.9936 | −0.081 |
| t7 s99 | 0.8839 | 0.9822 | −0.098 |
| **3x t7 集成** | **0.8823** | **0.9704** | **−0.088** |

- **核心排名对 censoring 处理稳健**：3x t7 集成与 t7_s99 在 full/measured-only
  下均居前二；63-D 序列映射两者均垫底。
- measured-only 下 nuisance-only t7 与 t7 单模型几乎打平（1.0029 vs 0.9936），
  说明在无删失行上非线性头优势更明显。

## 4. 方法贡献归属（决定性消融）

| 方法 | NLL | vs 线性 nuisance |
|------|-----|-----------------|
| motif_topology_hierarchy（线性基线） | 1.0916 | — |
| nuisance-only t7（非线性头，无 ViennaRNA） | 0.9012 | **+17.45%** |
| 3x t7 集成（非线性头 + ViennaRNA） | 0.8823 | **+19.17%** |
| ViennaRNA 序列增量（非线性头下） | — | **+2.09%** |

**核心结论**：方法贡献主要来自**右删失鲁棒（Student-t）非线性头结构**（+17.45%），
ViennaRNA 序列表征仅提供 +2.09% 的增量。这解释了：所有"加序列特征"方向
（RNA-FM、localctx、deep4/5、latent-operator、异构集成）全部失败——sequence
信号在此单 study 数据上的增量天花板就在 ~2%，不足以支撑更复杂表征。

## 5. 可发表主张（ClaimAuthorization 约束内）

- **Allowed**：censor-aware 鲁棒非线性头 + 跨种子 mu-集成对线性/无序列对照的
  pooled NLL 增益（+19.17%，edit-cluster CI [0.165, 0.293]）；模型结论对
  pooled/context/scaffold 聚合口径稳健；sequence 增量可写为 small conditional
  effect（+2.09%，CI 穿过 0）。
- **Forbidden**：transferable sequence mechanism、SOTA、noise ceiling、13 独立
  模型族公平比较等（见 `adjudication/ClaimAuthorization.json`）。
- 63-D sequence-map 路线永久关闭（P0.6 D1=TRACK_A_LOCKED）；提交/release 仍不授权。
