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
6. **r54 per-context EB 收缩 σ**（r43 硬分桶的修复版，2026-08-16 已跑）：
   κ∈{2..50} 扫描全部劣于 r51（最佳 κ=50 → 0.7823，+0.0008），CI
   [-0.0164, 0.013] 穿过 0；censored 层过拟合（0.2154 vs r51 0.1990，
   多数 context 仅 1-2 条删失行）。κ 单调递增时 NLL 单调趋近 r51
   （κ→∞ 才等于 r51）→ **context-level σ 结构不存在或不可检测**，
   r43 的硬分桶失败不是稀疏性而是真实无结构。NEGATIVE 闭合。
7. **r55 edit-motif 回归 per-fold σ**（2026-08-16 已跑）：fold σ 变化
   0.25–1.07，但 motif 特征（donor/acceptor 长度、GC、总长）与 σ 相关
   极弱（总长 +0.185、donor GC −0.10、acceptor GC −0.01）。LOO 线性
   回归预测 held-out fold σ = **0.8306 vs r51 0.7815（+0.0491）**——
   motif 特征对 σ 无预测力，per-fold σ 无法无泄漏恢复。NEGATIVE 闭合。

### 4.x 重大突破：r56/r56b per-context EB mu 修正（2026-08-16）

**这是自 r45 之后第一个统计显著的模型级正向增益。**

诊断发现：r51 只修正了 **scaf 层** mu 偏置（scaf9 −0.996→−0.01），但
**context 层** mu 偏置从未测过。r51 后 186 个 context（≥10 measured 行）
的 mean(y−mu) sd = 0.341，split-half（fold 奇偶）相关 **+0.986** ——
真实、稳定、可利用的信号。

- **r56**（per-context EB mu，kappa=10）：**0.7410**（−0.0405 vs r51）；
  measured 层 0.890→0.838；CI vs r51 下界 >0。
- **r56b**（r56 + min_meas=3 floor + kappa=2 强收缩）：**0.7314**
  （−0.0501 vs r51，−6.41% 相对），**edit-cluster CI vs r51 =
  [0.0126, 0.0862] 下界 >0**，leave_one_largest=0.0409，24/37 折正向。
  measured 层 → 0.8265，censored 严格不变（0.199）。
- 机理验证：r56b 后 scaf5/6/7 measured 残差 sd 显著下降（scaf6
  0.691→0.561、scaf7 0.796→0.694）—— 真实的 context 偏置消除，非 σ 重扫
  伪效应。
- 诚实 caveat（必须写入稿件）：r56b 的 context 修正对 nuisance 基线**不适用**
  （nuisance 无 context 特征），nuisance 在 r56b 下反而恶化
  （0.9749→1.1479），故同口径相对增益升至 +36.28% 是"方法普惠性不对称"
  而非纯绝对增益；必须同时报告绝对 NLL（0.7314）与同口径相对增益，
  并以 vs-r45-nuisance 口径（+27.15%）为保守读数。
- 最差折 CUCAG_CUGAG（−0.58）是 n_meas=2 的稀疏 context 噪声，r56b 的
  min_meas floor 已部分缓解（leave_one_worst=0.0492 仍显著）。
- **新冻结方法 = 7-member 集成（wg=0.5）+ r56b = 0.7314**
  （`submission_horizontal_table_v3.json`，四口径 definitive 表）。

### 4.y r57 收敛诊断（2026-08-16）：r56b 已到 context mu 修正极限

确认 r56b 是收敛冻结方法，残余方向全部闭合：

1. **jid 层随机效应 = 泄漏，禁止**：1336/1336 个 junction 严格单 fold
   （每个 junction 只出现在一个 edit component 中），jid 级修正会看到
   held-out 自身 —— 死路。
2. **残余 context bias 不可再提取**：r56b 后 context bias sd 0.341→0.313
   （mean|b| 0.266→0.216），split-half 残余相关 +0.622 但仅 n=18 稳定
   context（不稳定）。更细超参扫描：mm3_kappa0.5=0.7308 vs r56b 0.7314，
   paired CI [-0.0014, 0.0021] **穿 0** —— 无显著差异，r56b 保持冻结。
3. **迭代修正崩溃**：n_iter=2 → 0.8245、n_iter=3 → 1.0127 —— 二次修正
   把噪声当信号，强烈过拟合，证实残余 +0.622 相关是稀疏 context 噪声而非
   可提取结构。
4. **H1 扫描**：mm5/8/10/12 全部 ≥0.742（更差），min_meas=3 + kappa=2
   是唯一最优区。
5. **fold 层偏置**（sd 0.223）不可修正 —— fold 是 leave-out 独立单位，
   修正 fold 自身即泄漏。

**结论：r56b（mm3, κ=2）= 0.7314 是 context mu 校准的收敛点**；
   kappa0.5 vs r51 的 CI [0.0096, 0.0871] 下界 > 0 确认该家族显著正向。

这些方向的预期收益全部在 edit-cluster CI 宽度（±0.05-0.1）以内，无法
产生统计上可检测的改善。

## 5. 最终判断与建议

- **冻结方法**：7-member 混合集成（equal-family wg=0.5）+ **r56b per-context
  EB mu + σ 重扫 = 0.7314**（绝对 NLL 较 r45 0.7907 降 7.5%、较 r51 0.7815
  降 6.41%，measured 层偏置在 scaf 与 context 两层均消除，CI 下界 >0）。
- **模型级边界更新**：σ 粒度（context EB/motif per-fold）、mu 粒度（context
  EB）全部测尽；**per-context mu 是唯一新发现的真实信号**（r56b 正向），
  其余 22+ 条路线 NEGATIVE/持平。
- **建议转向 benchmark 轨稿件**：以 r56b 为冻结方法更新 Figure 2/Table 2/
  Claim 矩阵，撰写 benchmark 轨故事线（censor-aware 评估 + scaf→context 双层
  mu/σ 联合校准 + 方法边界闭合），P0.6 裁定不变（TRACK_A_LOCKED）。
