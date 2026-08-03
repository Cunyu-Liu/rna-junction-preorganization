# Contract Issue Register — v1.2

**Purpose**: 记录合同执行过程中影响科学含义或执行基准的不一致、缺失与冲突，采用 fail-closed 解释，不自行选择最有利解释。

## Issue-001 — clean 1.2.docx 与 redline 缺失

| 字段 | 值 |
|---|---|
| 章节 | §4.1 local contract verification |
| 日期 | 2026-08-03 |
| 冲突内容 | 执行提示词要求以 `1.2.docx`（预期 SHA-256 `3ad0c9997cde...`）为唯一 clean 合同，并核验其哈希；但该文件与 `1.1_to_1.2_redline.docx`（预期 `c77d647d...`）在本地 `outputs/` 与远端 `contract/` 均未找到。仅存在 `v1.2_decision_and_claim_log.md`（哈希已核验 `372e37a1...`）。 |
| 影响 | 无法对 clean DOCX 做 §4.1 要求的逐字节哈希核验；无法证明执行所用规则与 clean DOCX 逐字一致。 |
| 建议方案 | 以执行提示词 `rna 三级.md`（含 v1.2 全部科学/工程规则）为权威科学合同，以决策日志为决策记录；在未找到 DOCX 前不宣称已核验 clean DOCX 哈希。若后续获得 DOCX，需重新核验哈希并更新本 register。 |
| 采用解释 | **fail-closed**：不把缺失 DOCX 当作已核验；不宣称已满足 §4.1 的完整 DOCX 哈希核验；T0/Q0 只读与数据发现工作不受影响，可继续。 |
| 受影响依赖阶段 | 任何依赖"已核验 clean 1.2.docx 哈希"的 Gate 声明；本 register 记录为文件级缺口，不降级任何科学 Gate。 |

## Issue-002 — Git `-C` 在远端 zsh 异常

| 字段 | 值 |
|---|---|
| 章节 | §4.2 只读 preflight |
| 冲突内容 | 远端默认 shell (zsh) 下 `git -C <path>` 报 `cannot change to <path>: No such file or directory`，但 `cd <path> && git ...` 正常。 |
| 影响 | 预检脚本若用 `git -C` 会误报。 |
| 建议方案 | 远端统一使用 `cd <repo> && git ...` 方式。 |
| 采用解释 | 工具层面规避，不影响科学含义。 |

## Issue-003 — SSH 偶发回落到本地执行

| 字段 | 值 |
|---|---|
| 章节 | §4.2 只读 preflight |
| 冲突内容 | 部分 `ssh A100 '...'` 调用未进入远端（输出缺少 "SSH warring" banner，且 `which python` 返回本地 macOS 路径），在本地执行了命令。 |
| 影响 | 读取远端状态可能读到本地快照，造成误判。 |
| 建议方案 | 使用 SSH ControlMaster 持久连接（`ControlPath=/tmp/sshctl_a100`），并在每次远端命令开头校验 `hostname` 是否为 `bms-18937653-012`。 |
| 采用解释 | 工程规避；所有远端结论均以实际远端主机名输出为准。 |## Issue-004 — T0 1687 集合的 crystal 分量归属需论文补充方法

| 字段 | 值 |
|---|---|
| 章节 | 九、T0 数据准入（重建 1,687/1,713/1,636 三集合） |
| 日期 | 2026-08-03 |
| 冲突内容 | 工作簿 `junction_conformations_pdb` 子库含 377 个 distinct junction_id；论文正文报告 359 个 crystal junction（N = 359，Denny et al. 2018, Cell）。377 − 359 = 18 个 crystal junction_id 的具体归属无法仅由工作簿字段（junctionmat 标志、scaffold 完整性、censoring 状态）唯一确定，因此 1687 = 1328 designed junctionmat + 359 crystal 的 crystal 分量是"论文报告值"，而非工作簿可独立重建的计数。 |
| 影响 | 1687 集合的 designed 分量（1328）与工作簿完全一致（可独立重建）；crystal 分量（359）需论文补充方法（STAR Methods / 补充表）才能确定 18 个排除项。在获得该补充材料前，1687 的重建带有"论文报告值"的限定。 |
| 建议方案 | 获取 Denny 论文补充方法与 crystal junction 完整清单，逐行核对 18 个排除项；在此期间以"1328 designed junctionmat（工作簿可重建）+ 359 crystal（论文报告）"为 1687 的带限定重建，不把 26 个排除项（8 designed wc1 + 18 crystal）的全部逐行证据写成已闭环。 |
| 采用解释 | **fail-closed**：8 个 designed wc1 排除项已给出逐行证据（见 `manifests/t0_denny_semantics_manifest.json` 的 `set_mapping.designed_exclusion_evidence`）；18 个 crystal 排除项在补充材料到位前标记为 `RESIDUAL_EVIDENCE_GAP`，不降低 T0 其他科学 Gate，但 1687 的 crystal 归属不宣称已独立重建。 |
| 受影响依赖阶段 | T0 的 1687 集合重建声明；S0/T1 若依赖 1687 精确成员名单，需先补齐 crystal 排除项证据。 |
## Issue-005 — Censoring 语义：S0 表述 vs T0 操作规则

| 字段 | 值 |
|---|---|
| 章节 | 八、S0 EstimandSpec identification_assumptions / Phase 2 (M0) |
| 日期 | 2026-08-03 |
| 冲突内容 | S0 estimand 表述"censoring at -7.1 kcal/mol is left-censoring (values at/more negative than floor are not point-measurable)"字面会把"比 -7.1 更负"的行也视为 censored。但 T0 数据准入（已 PASS）的操作规则是 `censored = (dg10 == -7.1)`：28222 行 `dg10 != -7.1`（更负，如 -15.03）作为点测量，5865 行 `dg10 == -7.1` 作为 cap 行进入 censored likelihood。若按 S0 字面把所有 `<= -7.1` 行都 censored，则全部数据在单一阈值处 censored，family 均值不可点识别，T2 反向求解不可行。 |
| 影响 | 若按 S0 字面解释，T2 非可识别（whole-data 全删失）；若按 T0 操作规则，22,292 行点测量 + 5,865 行删失，可识别。 |
| 建议方案 | 以 T0 已 PASS 的操作规则为准：`censored = (dg10 == -7.1)`（cap 行），cap 行进入 left-censored likelihood（真值 <= -7.1），更负行作为点值。禁用任意"把更负点值改判为 censored"的灵活性。 |
| 采用解释 | **fail-closed + 只能前进**：cap 行（== -7.1）删除失（两解释一致）；更负行保留为点测量（T0 已锁定，避免整个数据不可识别导致 T2 无法前进）。S0 的"more negative"措辞记为表述偏差，不推翻 T0 操作规则。 |
| 受影响依赖阶段 | T2 的 censoring likelihood 方向；M0 已按此语义验证（cap 行 = 真值 <= CAP）。 |
