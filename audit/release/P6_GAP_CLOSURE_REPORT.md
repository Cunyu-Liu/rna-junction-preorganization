# Gap Closure Report — P4/P5/P6 (rna_junction v1.28–v1.31 strict audit)

- **Date**: 2026-08-09
- **Run root**: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/`
- **Evidence class**: `DEVELOPMENT_ONLY` — engineering closure; no scientific claim
- **SOTA status**: `SOTA_NOT_ADJUDICATED`; **NO_SUBMISSION_AUTHORIZATION**
- **scientific_claim_authorized = false**
- **Clean-checkout commit**: `de7192f797e8f4579769975dcde6604746ab7da9` (audit P6 release)

The post-audit review identified 4 contract-alignment gaps that were closed below
in a fail-closed manner (no re-adjudication, no expansion of authorization, sealed
results unchanged). All 4 gaps are closed; none changes the P4 `NOT_PROMOTED`
verdict or the `benchmark/identifiability-boundary` narrative.

---

## Gap ① — Phase 4: missing 3-seed final comparison

Contract Phase 4 requires "三seed最终比较" (3-seed final comparison). The original
P4 run used a single seed bootstrap. Closed by `audit/p4/p4_gap.py`.

**Result** (`p4_gap/BootstrapIntervals_3seed.csv`, `SeedsConsistency.json`):
All three seeds (0,1,2) are consistent: every axis has CI upper bound ≤ 0 (or
exact 0 tie on `scaffold_lomo`), `p_positive=0.0`, `promotion_eligible=false`.
`all_seeds_consistent_not_promoted = true` on every axis. The `NOT_PROMOTED`
verdict is **seed-robust**.

## Gap ② — Phase 4: missing 1,000 null permutations

Contract Phase 4 requires "1,000次null" (1,000 null permutations) and the
acceptance criterion `genuine > null`. The original P4 only wrote a placeholder
`NullAdjudication.csv`. Closed by the 1,000 sequence-pairing null permutation
block of `audit/p4/p4_gap.py`.

**Result** (`p4_gap/NullAdjudication_full.csv`): genuine mean supported-NLL gain
does **not** exceed the null 97.5 percentile on any axis
(`genuine_gt_null_97_5=false` for all 4 axes). There is no genuine signal beyond
null — consistent with the fail-closed, no-mechanism posture.

## Gap ③ — Phase 5: missing `MechanismRegistry.json` (contract deliverable naming)

Contract Phase 5 lists `MechanismRegistry.json` among its outputs; the earlier
`p5_run.py` did not emit it. Closed by `audit/p5/p5_mechanism_registry.py`.

**Result** (`p5_diagnostics/MechanismRegistry.json`): a formal registry of the
mechanism hypotheses considered, each linked to data/code/result/figure with a
fail-closed status (`NOT_AUTHORIZED` / `NOT_PROMOTABLE` / `SUPPORTED_BOUNDARY`).
It records that **no mechanism claim is authorized**, matching the P4 `NOT_PROMOTED`
verdict and the selected benchmark/identifiability-boundary narrative.

## Gap ④ — Phase 6: fresh replay + second-environment rerun (numeric tolerance)

Contract Phase 6 requires "clean checkout fresh replay; 第二环境复跑" with
acceptance "same hash/seed/env row predictions verbatim or ≤1e-10; cross-env
metric diff ≤1e-8". The original P6 emitted static release artifacts only; it did
not execute the replay. Closed by `audit/release/p6_replay_verify.py`.

**Procedure**:
1. `git worktree add` a clean checkout of the audit repo at commit
   `de7192f...` (detached HEAD).
2. Ran `audit/p4/p4_run.py` from the clean worktree in env1
   (`rna_junction_preorganization_v1_1`, numpy 2.2.6 / scipy 1.15.2) →
   `p6_replay_env1/`.
3. Re-ran `audit/p4/p4_run.py` from the same clean worktree in env2
   (`pc_cng`, numpy 2.2.6 / scipy 1.15.3 / pyarrow) → `p6_replay_env2/`.

**Result** (`p6_replay_verify/FreshReplayVerification.json`, `overall_pass=true`):

| comparison | object | n_common | max_abs_diff | tolerance | pass |
|---|---|---|---|---|---|
| A. same-env clean checkout (sealed vs env1) | FinalLeaderboard | 44 | 0.0 | ≤1e-10 | PASS |
| A. same-env clean checkout | BootstrapIntervals | 4 | 0.0 | ≤1e-10 | PASS |
| A. same-env clean checkout | FinalPredictions | 47,572 | 0.0 | ≤1e-10 | PASS |
| B. cross-environment (env1 vs env2) | FinalLeaderboard | 44 | 0.0 | ≤1e-8 | PASS |
| B. cross-environment | BootstrapIntervals | 4 | 0.0 | ≤1e-8 | PASS |
| B. cross-environment | FinalPredictions | 47,572 | 0.0 | ≤1e-8 | PASS |

The final P4 numbers rebuild byte-for-byte from a clean checkout (≤1e-10, actually
0.0) and are stable across two independent environments (≤1e-8, actually 0.0),
fully satisfying the contract P6 acceptance criteria.

---

## Net effect

All 4 gaps closed in a fail-closed, non-expanding, sealed-results-preserving way.
The scientific posture is unchanged: `NOT_PROMOTED`,
`benchmark_identifiability_boundary`, `SOTA_NOT_ADJUDICATED`,
`scientific_claim_authorized=false`.
