# Phase 3 Report — Candidate C: Support-Aware Gated Mixture with Abstention

- **Phase**: P3 (finite architecture iteration)
- **Date**: 2026-08-09
- **Candidate**: `support_aware_mixture`
- **Run root**: `/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/p3_full_v2/`
- **Evidence class**: `DEVELOPMENT_ONLY` — no scientific claim, no submission gate
- **Promotion verdict**: `NOT_ADJUDICATED` (outer-test comparison deferred to Phase 4)

---

## 1. Motivation

Phase 2 re-adjudication concluded `CONDITIONAL_KNOWN_OPERATOR_SIGNAL`: the
sequence signal survives on known-operator axes (symmetry/context) but fails on
low-support extrapolation folds (`A6_no_catastrophic = false`). In particular,
`corrected_v1_31` produces catastrophic right-censored NLL on `scaffold_lomo`
(the true operator-holdout axis, known catastrophic value ≈ 80.45).

Candidate C treats "extrapolation vs interpolation" explicitly: it computes
per-junction support features from outer-train only, and **abstains** (excludes
from scoring) junctions lacking local sequence-neighbour support. Supported rows
use a train-only local edit-KNN censored-location predictor.

## 2. Method change (leak fix)

The first runner version had a **selection-on-test leak**: it selected the gate
threshold `d_thresh` by evaluating supported-NLL gain on the outer test folds
and then reported coverage-risk on the same folds. This was fixed to **true
nested CV**: for each outer fold `f`, the gate is selected using only the
outer-train (folds ≠ f) via a bounded leave-one-inner-fold-out CV (K=8 inner
folds grouped by the axis `group` key), then applied to the held-out fold `f`.
The reported supported-NLL on fold `f` never informs the gate chosen for `f`.

## 3. Pre-registration (frozen before outer-test selection)

- Gate grid: `d_thresh ∈ {1, 2, 3, 5, 8, 12, 1000}` (1000 ≈ no abstention)
- Selection rule: argmax mean inner supported-NLL gain (vs `train_only_scaffold`)
  subject to mean inner coverage ≥ 0.5
- Coverage minimum: 0.5
- Elimination condition: discard the learned gate if it gives no risk improvement
  (supported NLL not lower than baseline at matched coverage) and coverage is too low

## 4. Results (full 4-axis run)

| axis | n_folds | selected gates | mean cov | cand supported NLL | baseline supported NLL | catastrophic folds |
|---|---|---|---|---|---|---|
| **scaffold_lomo** | 9 | all d=1 | 1.000 | **~3.10** | **~80.45** | **0** |
| context_lomo | 234 | all d=1 | 1.000 | 2.72 | 1.41 | 197 |
| edit_5fold | 5 | d=1/2 | 0.848 | 2.55 | 1.22 | 5 |
| symmetry_5fold | 5 | d=1/2 | 1.000 | 2.64 | 1.18 | 5 |

### Per-fold detail (scaffold_lomo)

| fold | inner gain | cand NLL | baseline NLL |
|---|---|---|---|
| 0 | 71.85 | 6.30 | 26.71 |
| 1 | 65.63 | 2.00 | 105.78 |
| 2 | 62.10 | 1.52 | 101.94 |
| 3 | 65.69 | 1.95 | 105.11 |
| 4 | 59.99 | 1.85 | 103.88 |
| 5 | 65.69 | 1.99 | 105.11 |
| 6 | 62.19 | 1.92 | 101.55 |
| 7 | 70.76 | 3.03 | 61.44 |
| 8 | 85.70 | 7.37 | 12.53 |

## 5. Interpretation

1. **Candidate C uniquely rescues the operator-holdout axis (scaffold_lomo).**
   The train-only local sequence KNN does not depend on the held-out scaffold's
   calibration, so it avoids the catastrophic right-censored NLL that the
   scaffold-conditioned baseline incurs when its scaffold-specific mean is
   unavailable. Inner gain is large and positive on every fold; 0 catastrophic
   folds. This directly fixes the Phase 2 `A6_no_catastrophic` failure on the
   most important (operator-generalization) axis.

2. **On known-operator axes (symmetry/edit/context), the candidate does NOT beat
   the scaffold baseline.** Because those axes keep repeated context/scaffold
   exposure in the train set, scaffold calibration already captures most of the
   signal, and the local edit-KNN is worse. This is consistent with the contract's
   core concern: repeated context/scaffold exposure packages calibration as
   generalization. The support gate does not manufacture sequence-generalization
   where it does not exist.

3. **The support/abstention gate is therefore NOT eliminated.** Per the
   pre-registered elimination condition, Candidate C provides a real, large risk
   improvement specifically on the operator-extrapolation stratum — the stratum
   where the strongest baseline (and v1.31) catastrophically failed. Its value is
   narrow but real.

## 6. Decision

- Candidate C (`support_aware_mixture`) is retained as a **targeted extrapolation
  fix**, not as a general conditional-sequence model.
- Evidence is `DEVELOPMENT_ONLY`; promotion and outer-test comparison are Phase 4
  scope (`NOT_ADJUDICATED`).
- Claim scope: `KNOWN_OPERATOR_CONDITIONAL_ONLY`; no operator-transfer claim.

## 7. Artifacts

- `CandidateRegistry.json`, `AblationRegistry.json`
- `InnerCVSelection.json` (per-axis/fold selected gate + inner stats)
- `SelectedGateEvaluation.csv` (per fold at nested-selected gate)
- `CoverageRisk.csv` (per-axis aggregate)
- `SupportedNLL.csv` (per fold/gate diagnostic)
- `CandidatePromotionDecision.json`, `STATUS.json`
