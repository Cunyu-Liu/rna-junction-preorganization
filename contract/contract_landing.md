# RNA Junction Preorganization v1.2 — Contract Landing

**Date (UTC)**: 2026-08-03
**Contract version**: v1.2
**Primary agent**: 本项目首席科学家 / 数据负责人 / ML 工程负责人 / 可复现性审计负责人

## 1. 合同哈希与权威顺序

| 文件 | 状态 | SHA-256 |
|---|---|---|
| clean `1.2.docx` | **缺失**（本地与远端均未找到） | 预期 `3ad0c999...`（未核验） |
| `1.1_to_1.2_redline.docx` | **缺失**（本地与远端均未找到） | 预期 `c77d647d...`（未核验） |
| `v1.2_decision_and_claim_log.md` | 存在，已核验 | `372e37a159195f6d5b21d57aca32cf1d679b4408d6ce9b43b079974a87adcf92` |
| 执行提示词 `rna 三级.md` | 存在（本次执行依据） | 见读取快照 |
| `1.1.docx`（远端 `contract/1.1.docx`） | 存在（678,936 bytes） | 历史 1.1，不作为 v1.2 权威 |

**权威顺序**（本执行约束）：
```
执行提示词 `rna 三级.md`（科学/工程规则全文）
  > v1.2_decision_and_claim_log.md（决策日志）
  > gate-specific frozen specifications
  > 历史 1.1、旧 registry、旧 report、旧 replay
```

**重要记录**：clean `1.2.docx` 与 redline `.docx` 在本次 landing 时缺失，无法做 §4.1 要求的逐字节哈希核验。因此本次执行以执行提示词 `rna 三级.md` 的完整正文作为权威科学合同，并以 `v1.2_decision_and_claim_log.md` 作为其决策记录。缺失 DOCX 已记入 `contract_issue_register.md`，不阻塞 T0/Q0 只读与数据发现工作。

## 2. 远端只读 preflight 快照（2026-08-03 UTC）

- 连接目标：`ssh -p 22 cunyuliu@36.137.135.49`（alias `A100`）
- 主机名：`bms-18937653-012`（Linux，Ubuntu 5.15.0-173）
- 仓库实际 Git root：`/home/cunyuliu/rna_junction_preorganization_v1_1_20260801`
- 分支：`main`；HEAD：`5aa0da6`（`audit: close engineering replay handoff`）
- remote：`origin git@github.com:Cunyu-Liu/rna-junction-preorganization.git`
- working tree dirty：是（5 个未跟踪文件，均在 `manifests/history/` 与 `reports/`，属 v1.1 历史产物）
- 已有 worktrees：1（主 checkout）
- 运行进程：存在多个其他用户进程（yihaozh / af3_bosun / shenxin / chenyuj），**本任务不触碰**；未发现本任务专属活跃 job
- GPU：8× NVIDIA A100-PCIE-40GB；空闲显存（MiB）：GPU6=38753、GPU1=22118、GPU4=26342、GPU3=21383、GPU7=19358、GPU0=5379、GPU5=25449、GPU2=11151
- GPU 约束：`GPU4` 按项目记忆保留（calibrate），避免占用；其余 GPU 按显存空闲度选择
- 磁盘：`/home` 7.0T（5.3T avail）；`/mnt` 18T（13T avail）
- Python：系统 `python3` 3.10.12；conda env `rna_junction_preorganization_v1_1` = Python 3.10.20, torch 2.9.0+cu126（CUDA 可用，8 devices）, numpy 2.2.6, pandas 2.3.3, scipy 1.15.2, biopython 1.87；缺 openpyxl/xlrd
- 已有数据：`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/`（denny + dms_sra + dms_pmc + dms_processed + dms_figshare + nar_gkag672 + author_release）
- 历史 auditors：v1.1 完成大量 Phase 0 crosswalk 恢复审计，结论为 crosswalk 永久不可得（见 reports/phase0_handoff_20260803.md）

## 3. 初始冻结状态（v1.2 起点）

```
CURRENT_OPERATIONAL_STATE = BLOCKED_AT_TECTO_DATA_ADMISSION
CURRENT_SCIENTIFIC_DISPOSITION = CONDITIONAL_CANDIDATE
CURRENT_DMS_CROSSWALK = ASSUMED_PERMANENTLY_UNAVAILABLE_V1_2
CURRENT_DMS_PRIMARY_LABELS = NOT_ADMITTED_FINAL_V1_2
CURRENT_DMS_REPLAY = ENGINEERING_EVIDENCE_ONLY
CURRENT_DMS_JOINT_TRANSPORT = CLOSED_NOT_AUTHORIZED
QMAPSEQ_ROLE = MANDATORY_COMPLETION_GATE_FOR_STRONG_MANUSCRIPT
SCIENTIFIC_UNLOCK = NO_UNLOCK
```

任何正式 Gate 通过前：`admitted tecto labels = 0`，`scientific unlock = NO_UNLOCK`，`model-stage authorization = NOT_GRANTED`，`publication claim = NOT_ADJUDICATED`。

## 4. 隔离执行环境

- 唯一 `run_id`：`v1_2_tecto_qmap_20260803`
- `parent_run_id`：`v1_1_phase0_20260801`（历史分支，仅作 parent 记录）
- 隔离 worktree：`/home/cunyuliu/rna_junction_preorganization_v1_2_20260803`
- 分支：`v1.2/tecto-qmap`（从 HEAD `5aa0da6` 创建）
- 数据/运行根：`/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/`
- 大体积 raw/intermediate 数据放 `/mnt`；代码放隔离 worktree；raw 只读、不可覆盖
- 未修改主 checkout、未修改历史 run/manifest/registry/evidence、未 push

## 5. 计划并行推进的 DAG

- **tecto 主线**：T0 data admission → S0 estimand/operator/symmetry freeze → T1 cleaning/QC/effective-N/split freeze → M0 synthetic/operator-identification → T2 tecto-only inference → T3 target-specific functional →（可选）sequence-only deployment → manuscript adjudication
- **qMaPseq 第二系统**：Q0 integrity/license freeze → Q1 98-variant registry → Q2 attrition/censoring reconstruction → Q3 endpoint replay → Q4 selection/split/analysis freeze → Q5 locked transfer test → claim adjudication
- 硬依赖：T0–T3 不依赖 current DMS 与 qMaPseq；Q0–Q3 可与 T0/S0/T1 并行；Q4/Q5 不得反向修改已冻结的 tecto 规格

## 6. 当前没有解锁的内容

- 真实 tecto 标签建模（需 T0、S0、T1、M0 依次 PASS）
- current DMS 任何准入（永久 `NOT_ADMITTED_FINAL_V1_2`）
- qMaPseq Q4/Q5 outcome-adaptive 分析（需先完成 Q0–Q3）
- 任何 publication claim（`NOT_ADJUDICATED`）