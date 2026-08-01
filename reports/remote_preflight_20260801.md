# Remote read-only preflight — 2026-08-01

Connection: `ssh -p 22 cunyuliu@36.137.135.49`

Observed host: `bms-18937653-012`

Observed user: `cunyuliu`

Observed remote time: `2026-08-01T13:06:28+08:00`

## Safety findings

- Existing user work is active in `/home/cunyuliu/d2t-rna` and
  `/home/cunyuliu/reactflow_delta_goal_20260729`; neither is used as this
  project's worktree.
- Existing repositories have dirty/untracked changes. No existing repository,
  result, download, checkpoint, or process was changed.
- The current user's D2T-RNA acceptance run and ReactFlow-Delta GPU pilot are
  active. The latter is using `CUDA_VISIBLE_DEVICES=2` and must not be
  disturbed.
- The host has eight NVIDIA A100 40 GB GPUs. All GPUs had compute processes at
  preflight time. No new GPU task was started.
- `/home` had approximately 5.4 TB available and `/mnt` approximately 13 TB
  available at preflight time.
- The requested code and artifact target directories were absent before
  initialization.

## Gate consequence

This preflight is infrastructure evidence only. It is not Phase 0 data
acceptance, Phase 0.5 specification acceptance, a GPU validation result, or a
scientific claim. Existing activities remain protected; this project proceeds
with Phase 0 metadata/provenance work only.

## Evidence limitations

The first read-only scan did not read controlled sequence/label/effect-value
content. A shallow public-data name scan and source/license/hash registry are
still required before Phase 0 can pass. The remote preflight log generated in
the new project is the authoritative command-level record for this snapshot.
