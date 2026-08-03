# RNA Junction Preorganization v1.1：Phase 0 交接文档

**交接日期：** 2026-08-03（Asia/Shanghai）
**执行合同：** `1.1.docx`
**合同 SHA-256：** `218dec34037487fae14c50eef2eab28b79292fe428bd4917a9da1f36687aa0e9`
**GitHub：** `git@github.com:Cunyu-Liu/rna-junction-preorganization.git`

## 1. 一句话结论

当前没有“程序跑不动”的主阻塞：完整 RNA-MAP 工程回放已经自然结束，输出和资源结果已留下；但 Phase 0 仍然 **BLOCKED / NO_UNLOCK**，因为缺少把原始 accession/sample 与处理条件、处理构建/文库和证据哈希逐行绑定的权威 provenance crosswalk，且该 crosswalk 尚未完成合同要求的自动—人工复核门。

因此，当前结果只能表述为：**工程回放完成、科学标签未准入、未进行 GPU 训练或 GPU 科学验证**。不能把回放相关性、比对率、smoke/proxy 指标或工程产物写成最终科学结论。

## 2. 当前卡点及需要解决的问题

### 唯一科学阻塞

现有公开材料可以证明论文方法、公开代码语义和 `data.zip` 的内容，但不能证明每个处理后的标签/文库来自哪一个原始样本和 accession。当前没有发现满足以下字段的逐行、可追溯、可复核材料：

`source accession / sample → processed condition → processed construct or library → evidence path or source reference → evidence SHA-256 → reviewer → review time`

目前注册表明确保持：

- `status=PUBLIC_SOURCE_CODE_SEMANTICS_REGISTERED_PRIMARY_PAYLOAD_NOT_ADMITTED`
- `source_code_semantics_only=true`
- `primary_labels_admitted=false`
- `pdb_library_1`、`pdb_library_2` 的 trial 对应关系仍是候选关系，未准入；`pdb_library_3` 尚未解决。

这不是下载速度、GPU 显存或 RNA-MAP 运行时故障。即使工程回放全部成功，只要这条 provenance 链没有闭合，就不能进入正式科学 gate。

### 需要用户/数据提供方补给的最小材料

请提供一份权威 row-level crosswalk，或提供能无歧义推出该 crosswalk 的原始材料。每一行至少应包含：

1. 原始 accession、run/sample 标识及其来源（SRA/ENA/作者补充材料等）；
2. 处理条件（例如修改类型、时间、温度、对照/实验条件）；
3. 处理后的 construct/library 名称，必须能与当前数据中的标签一一对应；
4. 支持该映射的文件路径、表格行号、网页/补充材料位置或稳定 URL；
5. 对证据文件记录 SHA-256；
6. reviewer、reviewed-at 时间和备注；
7. 对无法匹配、冲突或证据不足的行显式标记 `rejected` 或 `ambiguous`，不能靠推断补齐。

合同门槛保持不变：至少 50 条明确 matched、至少 30 条 rejected/ambiguous、自动与人工判断 agreement ≥ 0.95，并且每一行都有证据引用和哈希。若达不到，应如实报告 `BLOCKED`，不能通过改标签、删行或放宽口径制造 PASS。

## 3. 已完成并可复核的工程工作

### 项目位置

- 代码：`/home/cunyuliu/rna_junction_preorganization_v1_1_20260801`
- 大型数据与 artifacts：`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801`
- 合同副本及其哈希已登记；本项目没有把大型数据或权重放入 Git。

### 输入数据与来源审计

- Figshare archive：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_payloads/dms_figshare/data.zip`
- 大小：`643,502,044` bytes
- SHA-256：`241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`
- ZIP CRC：通过；成员数：`3220`
- 已确认包含 `library_sequences.csv`、官方 construct JSON 以及 6 个 mutation histogram pickle；这些内容本身不能替代 accession/sample 到 processed label 的 provenance crosswalk。

### 完整 RNA-MAP 工程回放

- run root：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/runs/rna_map_full_replay_SRR35766784_retry_envpath_20260802`
- source accession：`SRR35766784`
- reference：`7,500` 条 library reference；reference SHA-256：
  `e24bcc3253a44d8c71d4a0443eafc307ef91490445328bbe0f5a494324a8f044`
- Bowtie2：`92,296,217` paired reads；`98.85% overall alignment rate`
- 回放 worker 已自然结束；未杀死、未重启，也没有覆盖既有结果。
- `cuda_visible_devices` 为空；本回放属于工程重放，不是 GPU 训练或 GPU 科学验证。

最终主要输出及 SHA-256：

| 产物 | SHA-256 |
|---|---|
| `output/BitVector_Files/mutation_histos.json` | `438ccf3af1f34d70f789906570a4595b83dd05bfcb3493cf40059d1314990afb` |
| `output/BitVector_Files/mutation_histos.p` | `349c04e701b744ab1a5994c082d6bb27b1dd4105d94f822a2038c6de8a6d68ba` |
| `output/BitVector_Files/rejected_bvs.csv` | `fcb6f3607ae5b558c484da5bc5dc01fc52b994199de2821e7de59f6ef396b3ba` |
| `output/BitVector_Files/summary.csv` | `21781972bce9cc3fb0be969736e414617df8edf7c7587576ba4ae3e7c886ff02` |
| `output/Mapping_Files/SRR35766784_1_val_2.fq.gz` | `0031f4a4c9c0a4ecfdde5f8f6bd9cab99ec019636ea363006aa21a5ff0893520` |
| `output/Mapping_Files/SRR35766784_2_val_1.fq.gz` | `b2b5c046b944a4d32db60dc3acec5d0ea4cdb9efecc13d45800ae0950fdafd6a` |
| `output/Mapping_Files/aligned.sam` | `07c73a84d72576155e6360377821d1ba7cdc9c1cc608dd0d932ff19774449939` |
| `run_manifest.json` | `b8e52c91db69c5981c42cd5a6137044cccfd5117e1687a1194d901a5d5e83bec` |

回放中的 count-vector comparison 仍然是 `CANDIDATE_ONLY`，没有任何标签准入。记录的候选相关性如下，仅作工程诊断，不作科学结论：

| 候选标签 | cosine | Pearson | Spearman |
|---|---:|---:|---:|
| `pdb_library_1` | 0.9409892374 | 0.8138540170 | 0.8672390058 |
| `pdb_library_2` | 0.9627691671 | 0.8802906350 | 0.8947818783 |
| `pdb_library_3` | 0.9545419931 | 0.8625226184 | 0.8883293300 |
| `37C_2min` | 0.3651815754 | 0.2513511539 | 0.6407242351 |
| `denature` | 0.9110113423 | 0.6708040475 | 0.7361559767 |
| `nomod` | 0.9772488736 | 0.8983170513 | 0.8980185439 |

上述数值不构成标签正确性证明，也不解除 crosswalk gate。

## 4. 审计与状态说明

原始 `run_state.json` 是启动态记录，仍保留 `status=RUNNING`。这是有意保留的历史证据，没有直接覆盖它，以免改变启动态哈希或抹掉 launch provenance。回放完成状态应通过单独的 `completion_state.json` 和最终 replay audit 记录，而不是篡改原始状态文件。

审计脚本已修正两个工程审计问题：

1. 当 run 使用 `run_manifest.json` 而不是旧的 `inputs_manifest.json` 时，审计器现在能识别输入清单；
2. 当后台 launcher 没有写出显式 exit marker、但日志已经到达 `MUTATION SUMMARY:` 且主要输出完整时，审计器会记录 `REPLAY_COMPLETED_ENGINEERING_ONLY`，不会再把它误报为未完成，也不会改变任何科学 gate。

审计刷新期间只运行一个审计进程；曾经误启动的两个重复审计进程已终止，原因和过程均保留在本次执行记录中，RNA-MAP 输出未被删除或覆盖。

最终 replay audit 已于 `2026-08-03T05:18:57.255737+00:00` 完成，SHA-256 为：
`22034d33204b933b6ff01fe4131db68f1e008007bb8002b2a1d76d5ec4481fc`。

独立 completion state 已生成，用于记录 worker/launcher/watcher 已退出、旧 `run_state.json` 已保留，以及最终 audit 的哈希和状态：

`/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/rna_map_full_replay_SRR35766784_retry_envpath_20260802_completion_state.json`

## 5. 保护边界

以下用户任务在本轮保持原样，没有杀死、重启或修改：

- PID `713052`：curl，`SRR38259812_2.fastq.gz.partial`
- PID `1468666`：`download_ena_fastq.py`，选定 ENA runs

远端 Git 工作区中原有的 5 个 untracked 历史文件也未纳入本次 focused commit、未删除、未覆盖：

- `manifests/history/acceptance_phase0_20260801T164700Z.json`
- `manifests/history/data_registry_20260801T164700Z.json`
- `manifests/history/phase0_payload_inventory_20260801T164700Z.json`
- `manifests/history/phase_status_20260801T164700Z.json`
- `reports/phase0_payload_inventory_20260801T164700Z.md`

## 6. 解锁后的严格恢复顺序

1. 对收到的 crosswalk 做 schema、文件存在性、SHA-256 和重复/冲突检查；把原始材料放在 `/mnt/cunyuliu/.../phase0/source_payloads/`，不把大文件放入 Git。
2. 运行现有 `scripts/audit_manual_matching.py`，生成逐行自动判定、人工判定、证据引用、reviewer/time 和 checksum ledger。
3. 只有在 50 matched、30 rejected/ambiguous、agreement ≥ 0.95 且每行证据闭合后，才可登记 `PASS_READY`；不满足则保留 `BLOCKED` 并记录失败证据。
4. 先做 CPU/source-bound 最小验证，再做真实 CUDA 验证。CUDA 阶段必须记录模型、输入和 forward 的 device、GPU UUID、显存、fallback count；任何 CUDA 不可用或 CPU 静默降级都必须停止。
5. 完成 finalizer、sentinel、completion/freeze manifests 和 checksum ledger 后，才允许更新阶段状态；工程测试、proxy 指标和训练集结果不得写成最终科学结论。
6. 对本次新增代码和文档做 focused test、`git diff --check`、提交并推送到 GitHub；不改写失败 ancestor，不复用失败 run 作为新的科学证据。

## 7. 接手者首先查看的文件

- 合同：远端项目登记的 `1.1.docx` 副本及合同 SHA-256
- 处理来源登记：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/source_metadata/dms_processing_source_registry_20260801T190000Z.json`
- Figshare 内容审计：
  `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/reports/phase0_archive_manifest_content_audit_20260802.md`
- RNA-MAP bitvector 尾部审计：
  `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/reports/phase0_rna_map_bitvector_tail_audit_20260802.md`
- full replay manifest：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/runs/rna_map_full_replay_SRR35766784_retry_envpath_20260802/run_manifest.json`
- full replay log：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/runs/rna_map_full_replay_SRR35766784_retry_envpath_20260802/logs/rna_map_full.log`
- replay audit：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/rna_map_full_replay_SRR35766784_retry_envpath_20260802.json`
- audit refresh log：
  `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/audits/rna_map_full_replay_SRR35766784_retry_envpath_20260802_audit_refresh.log`

## 8. 当前交接状态

| 项目 | 状态 |
|---|---|
| 合同与项目隔离 | 已完成 |
| 只读 preflight、Git/进程/GPU/磁盘检查 | 已完成 |
| Figshare archive 与内容审计 | 已完成 |
| RNA-MAP 完整工程回放 | 已完成 |
| replay 结果哈希与最终审计封口 | 已完成；final audit SHA 已登记 |
| row-level provenance crosswalk | **BLOCKED：待补给** |
| manual matching acceptance gate | 未开始，依赖 crosswalk |
| GPU-only scientific validation | 未开始，禁止提前开始 |
| scientific unlock | **NO_UNLOCK** |

**接手规则：** 在收到并验证权威 crosswalk 之前，不要训练、不要做 GPU 科学结论、不要把候选 trial 映射写成正式标签，也不要把当前工程回放写成研究成功。
