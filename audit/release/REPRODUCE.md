# Reproduction Guide (r62 frozen lineage)

## Scope
Reproduces the grouped, right-censor-aware benchmark and the frozen
r62 method: 7-member ensemble (wg=0.5) + r62 calibration = 0.7243 pooled
NLL (+27.86% vs nuisance, edit-cluster CI [0.2416,0.3794]).
The core transferable-sequence-mechanism hypothesis is NOT_SUPPORTED at
the pre-registered gate (P0.6 TRACK_A_LOCKED); the benchmark track is
the surviving contribution. NO submission authorization.

## Requirements
- Run root `/mnt/cunyuliu/rna_junction_repair_20260811T090000Z` with
  r24_t7_seed7/, r33_xgboost_full/, r34_gbdt_seeds_full/, r35_gbdt_hp_full/
  member predictions and the r62 calibration product.
- Conda env `rna_junction_preorganization_v1_1` (see environment.lock);
  cross-env verification uses `pc_cng` (scipy 1.15.3).
- Code: git repo `rna-junction-preorganization` at commit `fa5649a113626efb1e3372767192843217620bf8` (audit/ tree).

## Steps (raw -> final replay)
1. `conda activate rna_junction_preorganization_v1_1`
2. Reproduce the frozen NLL from raw member predictions:
   `python audit/repair/r62_decoupled_frozen.py`  -> writes r62_decoupled_sigma.json (best=0.7243)
3. Verify artifacts against checksums.sha256.
4. Read ReleaseManifest.json (frozen_method + replay_verification) for
   the sealed statuses.  p6_r62_replay_verify.json holds the dual-env
   raw->final replay result (same-env <= 1e-10, cross-env <= 1e-8).

## Expected outcome
The frozen method reproduces 0.7243 in both environments; the benchmark
narrative (censor-aware evaluation + calibration chain + boundary
closure) is the contribution; sequence-mechanism claims stay locked.
