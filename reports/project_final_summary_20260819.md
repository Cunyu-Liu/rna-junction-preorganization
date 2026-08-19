# RNA Junction 预组织基准测试 — 项目最终状态总结

- 收尾日期：2026-08-19（UTC）
- 分支 / commit：`r0_audit_repair_20260811` @ `9300a9a`
- 定位：右删失感知评估方法建立 + 序列信号边界检验（benchmark / identifiability-boundary 轨）
- 后续状态：项目不再继续推进；本文件为最终交付摘要

## 1. 科学问题

RNA 三级结构的形成依赖 helix junction 的构象预组织。核心问题：**junction 序列是否携带 operator/scaffold 身份之外的可迁移 ΔG 预组织信号？**

## 2. 数据（单研究）

- 来源：Denny et al., Cell 2018，tectoRNA 双路 junction
- 规模：11,893 行 × 1,336 junction × 9 scaffold/operator × 234 context
- 删失：16.25% 右删失（y ≥ −7.1 kcal/mol）
- 结构：panel（每 junction 4–9 context，行非独立）→ 必须按 junction 分组评估
- 许可：受限再分发，不随仓库分发

## 3. 方法

- 统一右删失 Tobit 似然（measured / censored 统一 NLL）
- 37 个 edit×nested-context blocked folds（泄漏控制）
- fail-closed 双资格门（optimizer 收敛 + full-coverage）
- 冻结方法：7 成员混合集成（4 GBDT + 3 t7 MLP，family-equal wg=0.5）+ scaf→context 双层 EB mu 修正 + 解耦 per-scaf×stratum σ 校准
- 主估计量：pooled junction-macro 右删失 NLL（lower better）

## 4. 关键指标（最终冻结）

| 指标 | 值 |
|---|---|
| 冻结 pooled NLL | **0.7243** |
| 相对线性基线 | **+27.86%** |
| edit-cluster CI | [0.2416, 0.3794]（lower>0） |
| 37 折稳定性 | 稳定（leave-one-largest = 0.3077） |
| 双环境重放 NLL | 0.724302 / 0.724302（cross-env ≤ 1e-8） |
| 方法贡献链 | 1.0916 → 0.7243（累计降 33.7%） |

## 5. 方法边界（75+ 条路线闭合）

- 6 大类：输入表征 / 网络结构 / 训练侧 / 集成侧 / 事后校准 / 校准闭合
- 唯一正向族：事后校准（global σ → r38 → r45 → r51 → r56b → r62）
- 63-D 序列映射：frozen / r45 口径 −25.73%（NEGATIVE）→ P0.6 TRACK_A_LOCKED
- Candidate C（support_aware_mixture）：永久淘汰

## 6. Null 与统计验证

- 1000 次外训序列置换 null refit（0 失败，seed 20260816）
- `null_975_upper = 0.008501`；`null_mean = −0.01448`
- 与 genuine（true joint 增益 −0.2573）对比：null 门不通过（符合负结果预期）

## 7. P0.6 最终裁定

- `eligibility_status = VALID`
- `scientific_verdict = NOT_SUPPORTED_AT_PRE_REGISTERED_GATE`
- D1 = `TRACK_A_LOCKED`（锁 benchmark 轨，停止序列映射扩展）

## 8. 状态声明（诚实边界）

- SOTA：`SOTA_NOT_ADJUDICATED`
- 投稿 / 发布授权：`NO_SUBMISSION_AUTHORIZATION`
- `scientific_claim_authorized = false`
- 可发表贡献：右删失感知评估方法 + 边界闭合的 benchmark 认识（非 base-model 突破）

## 9. Seal（release 收尾）

- 12 个 artifact checksum 全过（含 `NullArtifact.json`）
- ReleaseManifest / STATUS 绑定 `9300a9a`（r62 冻结谱系）
- 数据 / 许可：受限，需 owner 授权方可再分发
