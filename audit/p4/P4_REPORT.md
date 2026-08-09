# Phase 4 Report — Final Outer-Test Comparison & Promotion Adjudication

- **Phase**: P4 (final outer-test comparison)
- **Date**: 2026-08-09
- **Candidate**: `support_aware_mixture` (Candidate C, Phase 3)
- **Run root**: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/p4_final/`
- **Evidence class**: `DEVELOPMENT_ONLY` — no scientific claim, no submission gate
- **Promotion verdict**: `NOT_PROMOTED`
- **SOTA status**: `SOTA_NOT_ADJUDICATED`
- **Claim scope**: `KNOWN_OPERATOR_CONDITIONAL_ONLY`

---

## 1. Objective

Phase 4 adjudicates whether Candidate C (the support-aware gated mixture with
abstention from Phase 3) should be **promoted** as a scientific model relative to
the strongest eligible baseline, on each of the four evaluation axes, under a
**coverage-matched supported-NLL** comparison.

The comparison is deliberately fair and leakage-controlled:

- **Baselines are reused from sealed Phase 1 predictions** — no refit on outer
  test, no test-selection. Only baselines that produced outer-test predictions in
  P1 are eligible.
- **Gates are frozen from Phase 3 nested-CV** — the per-axis/per-fold `d_thresh`
  was selected on outer-train only (leave-one-inner-fold-out CV, K=8 grouped by
  the axis group key) and applied unchanged to the held-out fold.
- **Coverage matching** — candidate and baseline supported-NLL are computed on
  the *same* supported subset of each fold, so a model that merely abstains from
  hard junctions cannot inflate its score.

## 2. Pre-registered acceptance criterion (Phase 4 promotion gate)

For a given axis, promotion requires **all** of:

1. bootstrap CI lower bound on mean supported-NLL gain `> 0`;
2. all folds positive (`all_folds_positive`);
3. mean relative gain `>= 0.10`.

A candidate that meets these on any axis is `promotion_eligible` on that axis;
`overall_promotion = PROMOTED` iff at least one axis is eligible.

## 3. Configuration

- Axes: `symmetry_5fold`, `edit_5fold`, `context_lomo`, `scaffold_lomo`
- Strong baseline per axis:
  - symmetry/edit: `corrected_v1_31`
  - context_lomo: `train_only_scaffold`
  - scaffold_lomo: `edit_knn`
- Bootstrap: 2000 resamples, seed fixed
- Baselines available: `global_censor_intercept`, `train_only_scaffold`,
  `scaffold_context_hierarchy`, `motif_topology_hierarchy`, `onehot_kmer_ridge`,
  `position_aware_additive`, `edit_knn`, `mutation_graph_smoother`, `small_mlp`,
  `corrected_v1_31`

## 4. Results

### 4.1 Bootstrap intervals (candidate − strongest baseline, supported-NLL gain)

| axis | n_folds | strong baseline | mean gain | CI low | CI high | p_positive | folds>0 | all positive | mean rel gain |
|---|---|---|---|---|---|---|---|---|---|
| symmetry_5fold | 5 | corrected_v1_31 | −1.5525 | −1.6240 | −1.4636 | 0.0 | 0 | False | −1.4255 |
| edit_5fold | 5 | corrected_v1_31 | −1.4019 | −1.6215 | −1.1566 | 0.0 | 0 | False | −1.2286 |
| context_lomo | 234 | train_only_scaffold | −1.3131 | −1.5656 | −1.0734 | 0.0 | 31 | False | −1.1499 |
| scaffold_lomo | 9 | edit_knn | 0.0 | 0.0 | 0.0 | 0.0 | 0 | False | 0.0 |

Candidate supported-NLL is **worse** than the strongest baseline on all
known-operator axes (large negative gains, CI entirely below 0, 0% positive
bootstrap draws) and **exactly ties** it on scaffold_lomo (gain 0.0).

### 4.2 Leaderboard (mean supported-NLL, lower is better)

| axis | candidate | strongest baseline | other notable |
|---|---|---|---|
| symmetry_5fold | 2.6437 (edit_knn tie) | **1.0912** (corrected_v1_31) | train_only_scaffold 1.1819 |
| edit_5fold | 2.5530 (edit_knn tie) | **1.1511** (corrected_v1_31) | train_only_scaffold 1.2159 |
| context_lomo | 2.7195 (edit_knn tie) | **1.4064** (train_only_scaffold) | scaffold_context 1.7015 |
| scaffold_lomo | 3.1047 (edit_knn tie) | **3.1047** (edit_knn) | corrected_v1_31 80.4494 (catastrophic) |

On symmetry/edit/context, `corrected_v1_31` (or train-only scaffold) is far
better than the candidate. On scaffold_lomo the candidate **ties** `edit_knn`
exactly (3.1047); it rescues the catastrophic 80.45 of `corrected_v1_31`, but
`edit_knn` already achieves the same supported-NLL, so there is **no incremental
gain** to promote.

### 4.3 Null adjudication

No permutation nulls were required: the candidate was not promoted on any axis,
so no mechanism/sequence claim is being made. All axes recorded
`nulls not required because candidate not promoted; no mechanism claim`.

### 4.4 Ablations (gate effect)

| axis | no_abstention NLL | fixed_gate NLL |
|---|---|---|
| symmetry_5fold (fold 0..4) | 2.72 / 2.51 / 2.63 / 2.69 / 2.67 | identical |
| edit_5fold (fold 0..1) | 2.76 / 2.13 | identical |

`no_abstention` and `fixed_gate` produce identical supported-NLL on the axes
tested, i.e. the abstention gate has **no measurable effect** where it was
exercised — it neither helps nor hurts on known-operator axes.

### 4.5 Generalization matrix

The candidate wins the supported-NLL comparison on only a tiny minority of
context_lomo folds and loses on essentially all symmetry/edit folds (negative
relative gains everywhere). No axis shows consistent wins.

## 5. Interpretation

1. **Candidate C is NOT promoted.** The support-aware gated mixture does not beat
   the strongest eligible baseline under coverage-matched supported-NLL on any of
   the four axes. On the three known-operator axes it is substantially worse
   (CI entirely below 0, 0/5 and 31/234 positive folds); on the operator-holdout
   axis it ties `edit_knn` with zero gain.

2. **The scaffold_lomo rescue is not an incremental contribution.** Phase 3 showed
   the candidate rescues the operator-holdout axis from `corrected_v1_31`'s
   catastrophic 80.45. Phase 4 shows `edit_knn` (a Phase 1 strong simple baseline)
   already reaches the same supported-NLL (3.1047). Candidate C therefore adds no
   supported-NLL improvement beyond an existing baseline on its best axis, and
   underperforms on every other axis.

3. **The abstention gate is inert where it matters.** The ablation shows
   `no_abstention == fixed_gate`; the gate neither rescues nor degrades known-operator
   axes. Its only role in P3 was avoiding the catastrophic scaffold calibration,
   which `edit_knn` already avoids by construction.

4. **No mechanism claim is authorized.** Because the candidate is not promoted and
   no prospective constructs exist, the contract's Phase 4 failure handling forbids
   any broad mechanism / operator-transfer / cross-system claim. The
   `KNOWN_OPERATOR_CONDITIONAL_ONLY` claim scope from Phase 2/3 remains the ceiling.

## 6. Decision

- `overall_promotion = NOT_PROMOTED`; `promoted_axes = []`.
- Candidate C (`support_aware_mixture`) is **retired** as a promotable scientific
  model under the coverage-matched outer-test contract.
- Evidence class `DEVELOPMENT_ONLY`; `SOTA_NOT_ADJUDICATED`; no submission gate.
- No prospective protocol is available (`NO_PROSPECTIVE_CONSTRUCTS_AVAILABLE`);
  broad generalization claims remain blocked.

## 7. Artifacts (run root `p4_final/`)

- `FinalLeaderboard.csv`
- `FinalPredictions.parquet`
- `BootstrapIntervals.csv`
- `NullAdjudication.csv`
- `GeneralizationMatrix.csv`
- `AblationTable.csv`
- `FairnessLedger.jsonl`
- `ProspectiveProtocol.json`
- `CandidatePromotionDecision.json`
- `STATUS.json`
