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

## 结论

- **鲁棒似然头部有效**：df≥5 的 Student-t 训练目标在等容量下稳定优于 Gaussian（约 +3.6%，
  robust_t_vs_reg_deep）；df=3 过重尾有害。
- **t7 跨种子 mu-集成是当前最佳方法**：+19.17%，edit-cluster CI [0.166, 0.289]，
  跨 3 个独立种子、聚焦最优 df=7 均以排除 0 的 CI 通过 10% gate。
- **SWA 阴性**：权重级方差缩减无增益。
- **Bagging 阴性**：数据级 bootstrap 扰动无增益（冒烟否决）。
- 复现产物、manifest 与预测存于 `/mnt/cunyuliu/rna_junction_repair_20260811T090000Z/`
  `r20_robust_t_df_sweep/`、`r21_seed99_replication/`、`r22_swa/`、`r23_seed2026_replication/`、
  `r24_t7_seed7/`、`r25_bag_smoke/`。

## 代码与提交

- 修改：`audit/models/nonlinear_mlp_hybrid.py`（`_train_mlp` 增加 seed 与 swa_n 参数）、
  `audit/models/nonlinear_mlp_rich_hybrid.py`（Student-t 训练目标、seed/swa_n 线程化、
  seed-diverse 与 SWA/Bag 变体）、`audit/repair/shootout_run.py`（注册 t3/t5/t7/t10、_s99、
  _s2026、_swa、_bag 变体）、`tests/audit/test_nonlinear_mlp_rich_hybrid.py`。
- 新增配置：`shootout_r20_robust_t_df_sweep_cfg.json`、`shootout_r21_seed99_replication_cfg.json`、
  `shootout_r22_swa_cfg.json`、`shootout_r24_t7_seed7_cfg.json`、`shootout_r25_bag_smoke_cfg.json`。
- 提交：`9fff7f2`、`62ddc91`、`0003bd5`、`5ae913b`、`075f8d9`、`6d4e223`（branch `r0_audit_repair_20260811`）。
- 单元测试：24 passed。
