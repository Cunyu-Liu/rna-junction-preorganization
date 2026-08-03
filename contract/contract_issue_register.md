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
| 采用解释 | 工程规避；所有远端结论均以实际远端主机名输出为准。 |