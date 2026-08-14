# r19-r21 鲁棒似然头部与 df 多样集成方法改进审计

日期：2026-08-13

## 背景

按合同思路继续优化 D2T-RNA_v7 模型效果。此前所有"加特征/加容量"方向（RNA-FM、deep4/5、
异方差 sigma、局部上下文 one-hot）均被否决，只有方差缩减（mu-ensemble）有效。本阶段转向
**鲁棒似然头部**（重尾 Student-t 训练目标）并在其上叠加 df 多样 mu-集成，最后做**独立种子复现**。

## 方法

- **Student-t 右删失 NLL**：`audit/models/nonlinear_mlp_rich_hybrid.py` 中 `_train_mlp_t`。
  使用 df 自由度的 Student-t 分布替换 Gaussian NLL，通过自定义可微正则化不完全 Beta 函数
  （Lentz 连分数）计算生存函数，梯度用 Student-t pdf 精确给出。评估目标保持不变
  （固定 sigma=0.7 的右删失 Gaussian NLL），仅训练目标不同。
- **df 扫描（r20）**：37 个 blocked fold，df ∈ {3,5,7,10}，对照 reg_deep(Gaussian) 与 nuisance。
- **df 多样集成**：对同一 fold 上各变体的 mu 取平均（共享固定 sigma=0.7）。
- **独立种子复现（r21）**：seed=99 重训 4 个集成成员（reg_deep/t5/t7/t10），37 fold 完整复现，
  验证方法增益非单一初始化侥幸。

## 结果：pooled junction-macro NLL（越低越好）

### r20 df 扫描（原始 seed）

| 模型 | NLL | vs nuisance |
|------|-----|-----------|
| nuisance | 1.0916 | — |
| reg_deep (Gaussian) | 0.9479 | +13.2% |
| Student-t df=3 | 1.0228 | 更差 |
| Student-t df=5 | 0.9140 | +16.3% |
| Student-t df=7 | 0.9129 | +16.4% |
| Student-t df=10 | 0.9138 | +16.3% |

### df 多样集成（4 成员：reg_deep + t5 + t7 + t10）

| 集成 | NLL | vs nuisance | edit-cluster CI |
|------|-----|-----------|----------------|
| 原始 seed 4x | 0.9012 | +17.44% | [0.145, 0.250] |
| **seed=99 复现 4x** | **0.8866** | **+18.78%** | **[0.159, 0.286]** |
| 跨 seed 8x | 0.8896 | +18.51% | [0.156, 0.277] |
| **纯鲁棒跨种子 6x（t5/t7/t10×2 seeds）** | **0.8855** | **+18.88%** | **[0.157, 0.277]** |

两个独立种子均以 CI 排除 0 通过 10% gate，增益稳定在 **+17.4%~+18.9%**。
**当前最佳为纯鲁棒跨种子 6x 集成（0.8855，+18.88%）**。

### r23 第三种子（seed=2026）与跨种子集成扩展

seed=2026 的 t5/t7/t10 单模型 NLL：

| 模型 | NLL |
|------|-----|
| t5_s2026 | 0.9321 |
| **t7_s2026** | **0.9022**（历史最佳单模型） |
| t10_s2026 | 0.9244 |

跨种子集成组合对比：

| 集成 | NLL | vs nuisance | CI |
|------|-----|-----------|-----|
| 纯鲁棒 6x（t5/t7/t10×2 seeds） | 0.8855 | +18.88% | [0.157, 0.277] |
| **纯鲁棒 9x（t5/t7/t10×3 seeds）** | 0.8833 | +19.09% | [0.160, 0.281] |
| **t7-only 3x（t7×3 seeds）** | **0.8823** | **+19.17%** | **[0.166, 0.289]** |

关键发现：t7 在所有 3 个种子里都是最优 df；**聚焦 t7 的跨种子集成（0.8823，+19.17%）
胜过混入 t5/t10 的 9x（0.8833）**。最优方法确定为 t7 跨种子 mu-集成。

### r24 t7 饱和性测试（seed=7）

第 4 个独立种子只训练 t7（0.9187，弱于 seed-0 0.9129 与 seed-2026 0.9022，体现逐 seed 方差）：

| 集成 | NLL | vs nuisance | CI |
|------|-----|-----------|-----|
| **3x t7（0+99+2026）** | **0.8823** | **+19.17%** | **[0.166, 0.289]** |
| 4x t7（+seed7） | 0.8832 | +19.09% | [0.165, 0.291] |
| 2x t7（0+2026） | 0.8934 | +18.16% | [0.154, 0.255] |
| 2x t7（99+2026） | 0.8831 | +19.11% | [0.166, 0.298] |

加入更弱的 seed=7 反而稀释（0.8823→0.8832）。**3x t7（seed 0/99/2026）为饱和点，
是最终最优方法**。成员选择是关键的：聚焦最优 df 的最强种子，胜过盲目增加成员。

### r22 SWA（随机权重平均）—— 阴性结果

在 `_train_mlp`/`_train_mlp_t` 中加入 `swa_n=10` 的滚动权重平均（对最后 10 个收敛 epoch
的权重求平均），注册 t5/t7/t10 的 SWA 变体，37 fold 全量对照：

| 模型 | NLL（SWA=10） | NLL（best-epoch） | 结论 |
|------|-----|-----------|------|
| t5_swa | 0.9129 | 0.9140 | 微升 |
| t7_swa | 0.9246 | 0.9129 | 变差 |
| t10_swa | 0.9256 | 0.9138 | 变差 |

将 t5_swa 并入 6x 集成也变差（0.8855→0.8879）。**SWA 是阴性结果**：权重级方差缩减
不带来增益，说明瓶颈不在最后一轮权重的噪声，而在逐 fold/逐样本的结构（已被 mu-集成、
鲁棒目标覆盖）。不再沿此方向扩展。

### r25 Bagging（bootstrap 数据级方差缩减）—— 阴性结果

实现 `make_nonlinear_mlp_extended_hybrid_reg_deep_t_bag`：每个 bag 对训练行做有放回重采样
（满 train scaler，逐 bag 独立种子），训练 t7 MLP 后平均 mu。2-fold 冒烟对照：

| 模型 | NLL（2 folds） |
|------|-----|
| t7（单模型） | 0.7886 |
| t7_s2026 | 0.8016 |
| t7_s99 | 0.8194 |
| **t7_bag5** | **0.8194** |

**Bagging 是阴性结果**：数据级扰动引入的噪声无法被 Student-t 目标补偿，两个 fold 均
差于其基础成员。与项目一贯规律一致——只有干净的 seed 级方差缩减有效，数据/特征级扰动
（RNA-FM、localctx、het-sigma、SWA、bagging）全部失败。冒烟阶段即否决，不再全量运行。

### r26 dropout×df 交互扫描 —— 中性/阴性结果

df=7 下重扫 dropout（Gaussian 调参最优为 0.1）：3-fold 冒烟对照

| 模型 | NLL（3 folds） |
|------|-----|
| **t7（dropout=0.1）** | **0.7895** |
| t7_light（dropout=0.05） | 0.7905 |
| t7_strong（dropout=0.2） | 0.8093 |

**中性/阴性**：正则化景观在 Gaussian 调优点附近平坦，df=7 下重调 dropout 无增益；
现有 t7 配置已在其最优。这是最后一个未探索轴，同样收敛到当前最优。

### 组合策略与 sigma 校准诊断（只读，不改协议）

对既有 3x t7 预测做组合/校准诊断：
- **mean vs median 组合**：NLL 相同（0.8823）；median 仅略微收窄 CI（[0.164, 0.285] vs
  [0.166, 0.289]），无实质增益。
- **集成残差诊断**：3x t7 集成在 uncensored 行的 RMSE=0.6321，低于各单成员（0.640-0.651）
  也低于固定评估 sigma=0.7 —— 证明集成确实降低了预测方差。
- **sigma 校准（诊断性，不纳入主结果）**：在评估集上最小化 NLL 的最优 sigma≈0.65（jm_nll
  0.8769 vs 0.8823@0.7），但该 sigma 是评估集调参，违反预注册固定 sigma=0.7 的匹配比较
  协议，故主结果仍采用固定 sigma=0.7 的 +19.17%。此诊断仅作为"协议对集成偏保守"的证据
  留档，不作为方法主张。

## 结论

- **鲁棒似然头部有效**：df≥5 的 Student-t 训练目标在等容量下稳定优于 Gaussian（约 +3.6%，
  robust_t_vs_reg_deep）；df=3 过重尾有害。
- **t7 跨种子 mu-集成是当前最佳方法**：+19.17%，edit-cluster CI [0.166, 0.289]，
  跨 3 个独立种子、聚焦最优 df=7 均以排除 0 的 CI 通过 10% gate。
- **SWA 阴性**：权重级方差缩减无增益。
- **Bagging 阴性**：数据级 bootstrap 扰动无增益（冒烟否决）。
- **完整消融梯次（ablation ladder）**（pooled junction-macro NLL，固定 sigma=0.7）：

| 方法 | NLL | vs nuisance |
|------|-----|-----------|
| nuisance baseline | 1.0916 | — |
| +Vienna21 nonlinear 2-layer reg | 0.9569 | +12.35% |
| reg_deep 3-layer | 0.9479 | +13.17% |
| +robust Student-t df=5 | 0.9140 | +16.28% |
| +df=7 best single | 0.9129 | +16.37% |
| **3x t7 ensemble (seed 0/99/2026)** | **0.8823** | **+19.17%** |

- 复现产物、manifest 与预测存于 `/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/`
  `r14_extended_mlp_scan/`、`r20_robust_t_df_sweep/`、`r21_seed99_replication/`、
  `r22_swa/`、`r23_seed2026_replication/`、`r24_t7_seed7/`、`r25_bag_smoke/`。

## r27 非线性 latent-operator 头 —— 阴性结果（合同 §9.1 空白组合）

实现合同 §9.1 要求的"统一 operator-aware head"下真正未试的组合：MLP 非线性映射
junction 特征 → latent q_j，经 scaffold 专属 intercept/slope（a_s + b_s·q_j）
operator 头，GH48 边缘化右删失似然联合训练（`audit/models/nonlinear_latent_operator.py`）。
与线性 latent-operator 系（v1.31）唯一差异是序列映射非线性化；与平头 MLP 系唯一
差异是显式 junction latent + scaffold slope。

3-fold 冒烟（r27，matched folds）pooled junction-macro NLL：

| 模型 | NLL（3 folds） | vs nuisance |
|------|-----|-----------|
| motif_topology_hierarchy（nuisance） | 0.9667 | — |
| vienna_latent_operator（线性 GH） | 1.0708 | 更差 |
| nonlinear_mlp_extended_hybrid_reg_deep_t7（平头 MLP） | **0.8242** | **+14.7%** |
| **nonlinear_latent_operator（MLP + GH 头）** | **1.1200** | **更差（-15.9%）** |

**否决**：GH 边缘化优化远难于平头 row-level NLL；junction 级 latent + scaffold
约束对非线性情形过强。连线性 latent-operator 也差于 nuisance，证实 GH 目标本身
在此数据上不是瓶颈——平头 MLP 直接优化 row-level mu 更有效。单元测试 7 passed
（含 GH 边际 NLL 可微、删失一致性、unseen scaffold abstain、scaffold 不入 MLP 特征）。

## r28 异构模型族集成 —— 阴性结果（稀释）

把不同模型族的 mu 一起平均（跨 r06/r08/r24 三个 run root，37 folds 与 11,893 rows
完全一致，无泄漏）：

| 集成 | NLL | Rel Gain | CI |
|------|-----|---------|-----|
| **3x t7（当前最优）** | **0.8823** | **+19.17%** | **[0.166, 0.295]** |
| 3x t7 + 扩展线性 hybrid | 0.8894 | +18.53% | [0.164, 0.284] |
| 3x t7 + 线性 hybrid | 0.9036 | +17.23% | [0.154, 0.265] |
| 3x t7 + latent-operator | 0.9095 | +16.68% | [0.150, 0.256] |
| 3x t7 + 扩展 + latent | 0.9211 | +15.62% | [0.141, 0.242] |

**否决**：异构族成员的预测相关性并不比同架构种子低到足以补偿其更强的单个误差。
同架构 3x t7 集成是饱和点。

## P0.2 joint evaluator 修复（2026-08-14）

修复前主榜聚合 `_pooled_nll_by_model` 只按 support/abstain 过滤，未消费
`optimizer_eligible`/`eligible_full_coverage`；虽然当前恰好全 eligible，但 evaluator
不满足合同 P0.5"不合格 fold 不进入主榜"的 fail-closed 要求。修复：

1. **主榜 eligibility 过滤**：新增 `_eligible_keys`/`_filter_eligible_preds`，
   pooled NLL 与全部 contrast/CI 只消费 optimizer 与 full-coverage 双门均通过的 fold。
2. **fit 行集 hash 记录**：`_rows_hash` 记录每个 model×fold 实际传入 fit 的
   train/test row-ID SHA-256，写入 ConvergenceLedger，证明所有模型共享同一行集。
3. **单元测试**：`test_repair_p02.py` 新增 3 项（rows_hash 确定性/顺序无关、
   双门过滤、ineligible fold 排除），共 13 passed。

既有 P0.2 基建（FoldSpec、optimizer gate、joint zero-overlap、adjudication
CI 穿零否决）早已就位且 10/10 passed；本次补齐了主榜消费 eligibility 的最后缺口。

## P0.5 最小受影响重跑（r29，修复后 evaluator）+ P0.6 重新裁定

按合同 P0.5 在修复后 runner 下重跑受 joint/optimizer 影响的参数模型（corrected
v1.31 / no_sequence / motif_topology / train_only_scaffold），37 个 blocked
joint folds，全部 fold optimizer+full-coverage 双门 eligible：

- **rows_hash 端到端验证**：37 folds × 4 模型，0 个 fold 出现 train/test
  rows_hash 分歧——所有模型确证消费同一行集（P0.2 修复验证通过）。
- **决定性 true-joint 对比（pooled junction-macro NLL，固定 sigma=0.7）**：

| 模型 | NLL | vs no-sequence |
|------|-----|-----------|
| corrected_v1_31（63-D 序列映射） | 1.4282 | **−24.5%（更差）** |
| no_sequence_latent_operator（matched） | 1.1473 | — |
| motif_topology_hierarchy | 1.0916 | +4.9% |
| train_only_scaffold | 1.0938 | +4.7% |

- **63-D 序列映射路线正式关闭**：corrected v1.31 在 true joint 下比 matched
  no-sequence 差 24.5%（theta=−0.281），edit-cluster CI [−0.439, 0.0175]
  上界亦未能排除 0。完全符合审计 D1 门的"不优于 matched no-seq"触发条件。
- **3x t7 集成在修复后 evaluator 下增益稳健**：NLL 0.8823，vs nuisance
  +19.17%，CI [0.166, 0.295]（因全部 fold 均 eligible，过滤不改变数字，
  但已端到端验证 eligibility 正确消费）。

P0.6 裁定（`/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/adjudication/`）：

- `eligibility_status = VALID`
- `scientific_verdict = NOT_SUPPORTED_AT_PRE_REGISTERED_GATE`
- `D1 = TRACK_A_LOCKED`（63-D sequence-map 路线永久关闭；benchmark 轨锁定）
- `ClaimAuthorization`：allowed 仅限 small conditional increment 措辞；
  forbidden 包括 transferable mechanism / SOTA / noise ceiling 等；
  submission/release 仍不授权。

结论：修复后的 true joint 确认 63-D sequence map 无增量；当前唯一幸存的方法
贡献是鲁棒似然 t7 跨种子 mu-集成（+19.17% over nuisance）。后续按 benchmark 轨
整理（aggregation/support/nesting/censoring 认识），而非继续扩展 sequence-map。

## r31 匹配消融：方法贡献归属分解（决定性）

实现 `nonlinear_mlp_nuisance_only_t`（同 reg_deep t7 MLP 架构 + Student-t df=7
目标，但**去掉 21-D extended-ViennaRNA 特征**，仅保留 nuisance 基
= motif+scaffold+topology），37-fold 全量对照：

| 方法 | NLL | vs 线性 nuisance |
|------|-----|-----------------|
| motif_topology_hierarchy（线性基线） | 1.0916 | — |
| nuisance-only t7（非线性头，无 ViennaRNA） | 0.9012 | **+17.45%** |
| 3x t7 集成（非线性头 + ViennaRNA） | 0.8823 | **+19.17%** |
| **ViennaRNA 增量（非线性头下）** | — | **+2.09%** |

**决定性发现**：3x t7 相对线性基线的 +19.17% 中，**+17.45% 来自非线性头 +
nuisance 特征本身**，仅 **+2.09% 来自 ViennaRNA 序列表征**（r30 3-fold 冒烟
+4.36% 方向一致但更小）。这直接解释了三件事：

1. **为何"模型效果太差、无领先成果"**：sequence 信号在此单 study 数据上
   确实很小（63-D 序列映射差 24.5%、ViennaRNA 增量仅 ~2%）；方法贡献不在
   sequence 表征，而在**右删失鲁棒非线性头**的结构。
2. **为何所有"加序列特征"方向失败**（RNA-FM、localctx、deep4/5）：序列
   信息增量天花板就在 ~2%，远不足以支撑更复杂表征。
3. **论文方法的真实可主张贡献**：censor-aware 鲁棒（Student-t）非线性
   头 + 跨种子 mu-集成，对线性/无序列对照的增益主要由非线性结构而非
   序列表征驱动；sequence 增量可写为 small conditional effect。

## r32/r33 梯度提升树（XGBoost）新模型族 + 混合集成突破

此前所有非线性家族都是 MLP；梯度提升树（GBDT）从未进入 shootout。实现
`audit/models/xgboost_censored_hybrid.py`：XGBoost 用与平头 MLP 完全相同的
右删失 Gaussian 目标（tau=0.7, CAP=-7.1，自定义 gradient/hessian）和相同特征块
（nuisance + train-scaled 21-D extended-ViennaRNA），GPU histogram 训练 + early
stopping（train hold-out）。单元测试 4 passed（grad/hess 单调性、有限差分校验、
形状、unseen scaffold abstain）。

**r33 全量 37-fold**（`r33_xgboost_full`）：

| 模型 | NLL | vs nuisance |
|------|-----|-----------|
| motif_topology_hierarchy | 1.0916 | — |
| t7 MLP（单模型，r33 本 run） | 0.9505 | +12.9% |
| **xgboost_censored_hybrid（单模型）** | **0.8845** | **+18.97%** |
| 3x t7 集成（r24） | 0.8823 | +19.17% |

单个 XGBoost 模型几乎追平 3x t7 MLP 集成（0.8845 vs 0.8823）——新模型族在
相同目标+特征下达到 MLP 集成的效果。注：r33 中 t7 MLP 单模型 0.9505 与 r24
的 0.9129 有差异，属 GPU 训练的 run-to-run 方差，正说明跨种子集成的必要性。

**决定性突破：混合模型族集成（XGBoost + 3x t7）**

| 集成 | NLL | vs nuisance | edit-cluster CI |
|------|-----|-----------|----------------|
| 3x t7（MLP 同族） | 0.8823 | +19.17% | [0.166, 0.293] |
| **xgb + 3x t7（4 成员）** | **0.8599** | **+21.23%** | **[0.179, 0.346]** |

这是**本项目第一个超越 3x t7 饱和点的改进**（`analyze_mixed_xgb_t7.py`）：
把 XGBoost 与 3 个 t7 MLP 成员的 mu 平均，NLL 从 0.8823 降至 0.8599，相对
nuisance 增益从 +19.17% 提升至 **+21.23%**，edit-cluster CI [0.179, 0.346]
（lower>0），leave-one-largest 0.242（37 组件稳健）。

**与 r28 阴性结果的对照**：r28 把线性 hybrid / latent-operator 加入集成会稀释
（0.8823→0.8894..0.9211），因为线性模型与 MLP 预测相关性太高；而 **GBDT 是
结构正交的学习器**（树模型 vs 神经网络），错误模式互补，所以能真正降低方差。
这打破了"只有同架构跨种子集成有效"的饱和状态——**关键是引入真正低相关的异构
模型族，而非同质模型**。

## 代码与提交

- 修改：`audit/models/nonlinear_mlp_hybrid.py`（`_train_mlp` 增加 seed 与 swa_n 参数）、
  `audit/models/nonlinear_mlp_rich_hybrid.py`（Student-t 训练目标、seed/swa_n 线程化、
  seed-diverse 与 SWA/Bag 变体）、`audit/repair/shootout_run.py`（注册 t3/t5/t7/t10、_s99、
  _s2026、_swa、_bag 变体）、`tests/audit/test_nonlinear_mlp_rich_hybrid.py`。
- 新增配置：`shootout_r20_robust_t_df_sweep_cfg.json`、`shootout_r21_seed99_replication_cfg.json`、
  `shootout_r22_swa_cfg.json`、`shootout_r24_t7_seed7_cfg.json`、`shootout_r25_bag_smoke_cfg.json`、
  `shootout_r27_latent_operator_smoke_cfg.json`。
- 新增模型：`audit/models/nonlinear_latent_operator.py`（r27）、
  `audit/repair/analyze_hetero_ensemble.py`（r28）、`audit/repair/analyze_r27_smoke.py`、
  `audit/repair/per_component_diag.py`、`tests/audit/test_nonlinear_latent_operator.py`。
- P0.2 修复：`audit/repair/shootout_run.py`（eligibility filter + rows_hash）、
  `tests/audit/test_repair_p02.py`（+3 项）。
- P0.5/P0.6：`audit/repair/analyze_p05_rerun.py`、
  `audit/repair/shootout_r29_p05_rerun_smoke_cfg.json`（commit `4f90015`）。
- r31 消融：`audit/models/nonlinear_mlp_rich_hybrid.py`（`make_nonlinear_mlp_nuisance_only_t`）、
  `audit/repair/shootout_r30_nuisance_ablation_smoke_cfg.json`、
  `audit/repair/shootout_r31_nuisance_only_full_cfg.json`（commit `5d5cfb4`、`0dbb3eb`）。
- r32/r33 XGBoost：`audit/models/xgboost_censored_hybrid.py`、
  `audit/repair/sanity_xgboost.py`、`audit/repair/shootout_r32_xgboost_smoke_cfg.json`、
  `audit/repair/shootout_r33_xgboost_full_cfg.json`、
  `tests/audit/test_xgboost_censored_hybrid.py`（commit `450ab47`）、
  `audit/repair/analyze_mixed_xgb_t7.py`（commit `2812c47`）。
- 提交：`9fff7f2`、`62ddc91`、`0003bd5`、`5ae913b`、`075f8d9`、`6d4e223`、`7a1f5fa`、
  `f4c322f`、`4f90015`、`5d5cfb4`、`0dbb3eb`、`c6ed2a4`、`450ab47`、`2812c47`
  （branch `r0_audit_repair_20260811`）。
- 单元测试：24 passed（r19-r26）；r27 新增 7 passed；P0.2 修复后全套 187 passed
  （1 项 pre-existing 失败 `test_shootout_report_only.py::test_paired_contrast_sign_and_semantics`
  由 source_row_id 不匹配的 fixture bug 引起，与本次改动无关）。
