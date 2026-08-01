# GPU preflight record

Date: 2026-08-01 13:29:25 Asia/Shanghai (2026-08-01T05:29:25Z)

## Scope

This record covers only the hardware/runtime precondition for future GPU
validation. It is not a model result, a smoke-test qualification, a Phase 0
data result, or a scientific acceptance gate.

## Protected execution

Immediately before the probe, a read-only `nvidia-smi` check recorded GPU 0 at
4 MiB used, 40,438 MiB free, 0% utilization, and no compute application. The
other GPUs were left untouched because they had active jobs, substantial
allocation, or both. No existing process was stopped or reconfigured.

## Probe

- Project code root: `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801`
- Probe: `scripts/cuda_probe.py`
- Environment: `/home/cunyuliu/miniconda3/envs/editflow/bin/python`
- Device selection: `CUDA_VISIBLE_DEVICES=0` (logical device `cuda:0`)
- Data/model access: none
- Training: none
- Timeout: 30 seconds
- Result: `CUDA_PROBE_PASS`, exit code 0
- Device: NVIDIA A100-PCIE-40GB
- PyTorch: `2.5.1+cu121`; CUDA build: `12.1`
- Probe tensor device: `cuda:0`
- Log: `/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/logs/cuda_probe_20260801T052925Z.log`
- Log SHA-256: `65e037ee8c53d5f1073fd1f1a3d4bb8086501d296a77627620d914e1b63410bd`

## Interpretation

The selected environment can allocate and synchronize a CUDA tensor without
CPU fallback. This clears only the GPU runtime precondition for a future,
contract-authorized validation stage. It does not establish the Phase 0 data
semantics/provenance gate, does not deploy the contract source to the remote
project, and does not unlock Phase 0.5 or any later stage.

The project remains fail-closed with status
`REMOTE_CONTRACT_SOURCE_NOT_DEPLOYED_EGRESS_BLOCKED`; the exact contract hash
remains `218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`.
