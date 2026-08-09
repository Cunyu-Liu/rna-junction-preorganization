# Phase 2 验收报告：核心科学假设审查（sequence 可迁移增量信息）

run root: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z`
audit worktree: `/home/cunyuliu/rna_junction_audit_20260807T090244Z`
提交: `e5d0167`（p2 pipeline + tests）
日期: 2026-08-09（全量 scientific run 结束 11:18 CST）
合同: `rna_junction_v1_28_v1_31_strict_audit_2026-08-07.md` §Phase 2 / §12 验收标准

## 1. 裁定总览

**VERDICT = FAIL（H0_OR_INCONCLUSIVE）**

Phase 2 合同验收标准（§12, line 512：gain CI 下界>0；5/5 外层 fold 为正；genuine 最小效应>null 97.5% 上界；blocked context 与 edit 轴为正；无 catastrophic fold）未全部满足。按合同 §14.2，本项目应转 **benchmark / identifiability boundary** 论文方向，不得把负结果包装成方法 SOTA。

## 2. 运行配置（p2_full_cfg.json）

- candidate: `corrected_v1_31`（latent-operator Tobit，序列 functional）
- reference baseline: `train_only_scaffold`
- gain 定义: mean over folds [NLL(ref) − NLL(candidate)]
- axes: symmetry_5fold / edit_5fold / context_lomo / scaffold_lomo
- bootstrap: split-unit outer-fold，2000 次
- permutations: symmetry/edit 各 1000（label+sequence）；context_lomo/scaffold_lomo 各 100（每 perm 已平均 234/9 个 outer folds，null 紧）

## 3. 主对比结果（junction-macro 右删失 NLL gain）

| axis | mean gain | CI 下界 | 5/5 正 | p_positive | 通过 A1/A2 |
|---|---|---|---|---|---|
| symmetry_5fold | +0.091 | +0.068 | 5/5 True | 1.0 | ✓ |
| edit_5fold | +0.088 | −0.102 | 3/5 False | 0.85 | ✗ (CI<0) |
| context_lomo | +0.266 | +0.147 | 104/234 False | 1.0 | ✗ (非 5/5，catastrophic) |
| scaffold_lomo | 0.000 | 0.0 | 0/9 False | 0.0 | ✗ (无 operator transfer) |

## 4. Null 检验（genuine > null 97.5% 上界？）

| axis | genuine | sequence-null p975 | 过？ | label-null p975 | 过？ |
|---|---|---|---|---|---|
| symmetry_5fold | 0.091 | 0.061 | ✓ | 0.522 | ✗ |
| edit_5fold | 0.088 | 0.073 | ✓ | 0.507 | ✗ |
| context_lomo | 0.266 | 0.251 | ✓(勉强) | 0.457 | ✗ |
| scaffold_lomo | 0.000 | 0.0 | ✗ | 0.0 | ✗ |

- **sequence-pairing null**：symmetry/edit/context 三轴 genuine 超过 null 97.5% 上界 → 已知 operator 宇宙内 sequence 信号未被 sequence-pairing 噪声复制。
- **label-permutation null**：全轴 genuine << label null p975 → 在 junction 内置换 label 后模型仍可获得 ≥ genuine 的 gain。该 null 未破坏驱动 gain 的每组结构，说明大部分 gain 不能稳健归因于“真实 label↔sequence 配对”。此为最关键的负向证据。

## 5. Effect decomposition（各层边际贡献）

| axis | margin_operator | margin_context | margin_motif_topo | margin_sequence |
|---|---|---|---|---|
| symmetry_5fold | +1.71 | −0.58 | −1.76 | +2.43 |
| edit_5fold | +1.55 | −0.51 | −2.02 | +2.62 |
| context_lomo | +1.30 | −0.30 | −1.24 | +1.80 |
| scaffold_lomo | −77.3 | +76.4 | +0.80 | −77.2 |

- margin_sequence（motif→sequence）大且为正：序列 functional 提供大量 NLL 降低。
- margin_context 为负：scaffold/context hierarchy 反而恶化（比 train_only_scaffold 更差）。
- scaffold_lomo 全轴崩塌（train_only_scaffold 与 corrected_v1_31 均回到 NLL≈80 的退化/拒答口径，gain=0）→ 真正未见 operator 时无任何方法具备预测能力（operator 泛化边界）。

## 6. 科学结论

1. 已知 operator 宇宙（symmetry/edit/context）内，序列 functional 确实降低 NLL 并超过 sequence-pairing null —— 这是**条件性**的已知-context 信号。
2. 但 label-permutation null 全轴超过 genuine，edit 轴 CI 下界<0（3/5），context_lomo 大量 catastrophic fold，且 scaffold_lomo 零迁移。严格合同标准 **未通过**。
3. 结论限定：不能主张“sequence 编码可跨 context/operator 迁移的 preorganization mechanism”；只能主张“已知 operator 宇宙内、sequence-pairing null 之下存在的条件信号，且其可迁移边界在未见图（operator）处为零”。
4. 触发 §14.2 → 论文方向转为 **严格删失/分组 benchmark：揭示常规 sequence split 如何把 context/operator 校准误写成泛化，并量化可识别边界**。负结果完整保留，不包装为方法 SOTA。

## 7. 验收记录

- 输出物（`/mnt/.../p2_full/`）：HypothesisRegistry.json、NullProtocol.json、NullResults_{axis}_{type}.parquet（8 个）、SupportLedger.parquet、EffectDecomposition.csv、BootstrapIntervals.csv/json、CoreHypothesisDecision.json、STATUS.json ✓
- 单元测试：`tests/audit/test_p2.py` **7 passed** ✓
- STATUS.json: state=FAIL, verdict=H0_OR_INCONCLUSIVE, operator_transfer_boundary=true ✓
