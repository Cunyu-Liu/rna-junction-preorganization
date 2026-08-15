# RNA Junction 模型级改进清点审计（r51–r53 之后）

日期：2026-08-16
分支：`r0_audit_repair_20260811`
目的：在 r51（joint mu-affine + σ 重扫，冻结方法 0.7815）之后，系统性清点
所有已测/未测的**模型级**改进方向，判断方法边界是否真正闭合，并给出
剩余可尝试的杠杆。

## 1. 结论先行

**模型级方法边界已系统性闭合。** 在 base 预测器层面，所有已测的
representation / architecture / training / ensemble 杠杆均为阴性或持平；
唯一正向方向是**事后校准**（global→scaf→scaf×stratum σ，再升级到
r51 joint mu-affine + σ 重扫）。继续增加 base 模型复杂度无法带来增益，
反而破坏当前最优。

## 2. 模型级杠杆证据表（全部已跑）

所有 NLL 均为对应 run root 的 pooled-OOF junction-macro 口径；比较基准为
当前最优 base 成员 `reg_deep_t7`（MLP）与混合集成。

### 2.1 Representation（输入表征）

| 杠杆 | run | 结果 | 判定 |
|------|-----|------|------|
| 63-D seq map | r29 (full) | sequence 相对增益 −25.73% | NEGATIVE（P0.6 TRACK_A_LOCKED）|
| RNA-FM embedding | r15 (smoke) | 1.57/2.57 vs reg_deep 0.74/0.89 | NEGATIVE |
| local-context one-hot (24-D) | r18 (smoke) | 0.886/1.202 vs 0.741/0.920 | NEGATIVE |
| Vienna21 增量 | r07-r14 | ~+2% | 小（已冻结进特征集）|
| RBF kernel 混合 | r36 (full) | 0.799 vs reg_deep 0.741 | NEGATIVE |

### 2.2 Architecture（网络结构）

| 杠杆 | run | 结果 | 判定 |
|------|-----|------|------|
| 更深 MLP（4/5 层） | r16 | 0.780/0.953 vs 0.741 | NEGATIVE |
| in-network 异方差 σ head | r17 (smoke) | σ≈0.96, NLL 0.937 vs 0.830 | NEGATIVE |
| latent-operator head | r27 (smoke) | 1.09/1.15 vs 0.74/0.90 | NEGATIVE |

### 2.3 Training（训练侧）

| 杠杆 | run | 结果 | 判定 |
|------|-----|------|------|
| Student-t df 扫描 | r20 | df=7/10 最优 | 已冻结 t7 |
| SWA | r22 (smoke) | fold1 改善 fold2 恶化 | MIXED（不采用）|
| bagging (bootstrap) | r25 (smoke) | 0.758/0.881 vs 0.748/0.830 | NEGATIVE |
| censor-aware 重加权 | r49 (smoke) | 0.7965-0.826 vs 0.7886 | NEGATIVE |
| reg 强度/宽度变体 | r14 | reg_deep 最优 | 已冻结 |

### 2.4 Ensemble（集成侧）

| 杠杆 | run | 结果 | 判定 |
|------|-----|------|------|
| GBDT+MLP 混合 | r33-r35 | 误差多样性 = 关键 | 已冻结 7-member |
| family 权重 | r50/r53 | wg=0.5 equal-family 最优 | 已冻结 |
| per-scaf family 权重 | r52 | 0.7841 > 0.7815 | NEGATIVE |
| feature-diverse 成员 | r48 | error corr 0.88，增益 −0.0009 | NEGATIVE |
| mixture-of-predictives | r41 | 0.858 vs mu-avg 0.853 | NEGATIVE |

### 2.5 Calibration（事后校准 —— 唯一正向族）

| 杠杆 | run | 结果 | 判定 |
|------|-----|------|------|
| global σ 扫描 | — | 0.8419 (σ=0.62) | 正向起点 |
| per-scaf σ | r38 | 0.8166 | 正向 |
| per-scaf×stratum σ | r45 | 0.7907（修正 grid floor 0.05）| 正向 |
| per-context σ | r43 | 0.8310 > per-scaf | NEGATIVE |
| measured-only mu 修正 | r46 | 0.7942（陈旧 σ）| 无增益 |
| affine mu 修正 | r47 | 0.7930（陈旧 σ + 旧 grid）| 无增益 |
| **joint mu-affine + σ 重扫** | **r51** | **0.7815（= 新冻结）** | **正向（唯一新增益）** |

## 3. 为何 base 模型已到局部最优

- sequence 信号在此单 study 数据上确实很小（P0.6 已裁：63-D map −25.73%、
  Vienna 增量 ~2%），base 预测器已把这些信号榨干；
- 所有替代表征（RNA-FM、localctx、kernel、deeper）都在拟合噪声或破坏
  现有信号，无一超过 `reg_deep_t7` + Vienna21 特征集；
- 集成侧误差多样性已充分（r48 确认 7mem 是局部最优），per-scaf/feature-
  diverse 成员均过拟合；
- 剩余误差主要由 irreducible 测量噪声主导（measured RMSE 0.61 在消除偏置后
  已接近数据本身的噪声底），不是模型容量不足。

## 4. 仍可尝试但预计收益很小的方向（如需）

1. **per-fold σ 结构**：fold 是 leave-out 独立单位，无法在不泄漏前提下
   per-fold 校准 σ（scaf×stratum 是共享单位，已用满）—— 死路。
2. **r51 网格/EB 超参微调**：κ=20 已接近最优，收益 <0.001 —— 噪声内。
3. **t10 成员替换 t7**：r20 全 fold t7 vs t10 几乎持平，无净增益。
4. **更细粒度 sigma（context×stratum）**：r43 已证明 context 粒度更差。
5. **calibrate-then-ensemble 组合顺序**：先逐成员 r51 校准再平均，
   0.8673 vs 冻结（先平均后校准）0.7815 —— **差 0.0858**，NEGATIVE。
   证明校准必须作用于集成后的 mu（成员级噪声被平均后再校正才有效）。

这些方向的预期收益全部在 edit-cluster CI 宽度（±0.05-0.1）以内，无法
产生统计上可检测的改善。

## 5. 最终判断与建议

- **冻结方法**：7-member 混合集成（equal-family wg=0.5）+ **r51 joint
  校准 = 0.7815**（绝对 NLL 较 r45 0.7907 再降 1.16%，measured 层偏置消除）。
- **模型级改进已到数据允许的极限**：继续在 base 模型上加复杂度不产生增益；
  要获得数量级提升只能走数据侧（prospective 多 operator 数据，需 owner 授权）。
- **建议转向 benchmark 轨稿件**：以 r51 为冻结方法更新 Figure 2/Table 2/
  Claim 矩阵，撰写 benchmark 轨故事线（censor-aware 评估 + 方法边界闭合 +
  joint mu+σ 校准），P0.6 裁定不变（TRACK_A_LOCKED）。
