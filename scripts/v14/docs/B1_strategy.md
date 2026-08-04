# B1 — Synthetic failure-mode validation: strategy & expected results

Run: `v1_4_boundary_audit_20260804T150707Z`
Goal: prove the audit procedure behaves as intended on fixtures with **known truth**. B1 PASS does **not** prove that the tecto or qMaP biological models are correct — it proves the audit detects the failure modes they are designed to detect.

---

## 0. Principle

Every fixture is a controlled experiment with a planted failure. The audit procedure runs on the corrupted data and must either (a) reject the invalid claim, or (b) quantify the induced bias. The result is compared against the fixture's known ground truth. We track **false-pass** (audit fails to catch a planted failure) and **false-fail** (audit rejects a valid experiment) as first-class metrics.

Not a dataset expansion: all synthetic samples/reads/titration points are fixtures for the audit procedure, never summed into biological N.

---

## 1. Fixture: endpoint_reuse

**Manipulation**: disguise the same-platform old estimate as an external predictor for `rna_map_dg`.

| Case | Setup | Expected audit behavior |
|------|-------|--------------------------|
| E1 leakage | old_dg (same-platform) injected as if external | **Block** false transfer PASS (`BLOCK_TRANSPORT_PASS`) |
| E2 genuine external | a real independent predictor | **Preserve** (not falsely killed) |

**Quantification**: leakage flag = overlap between predictor lineage and target lineage. If `lineage(predictor) ∩ lineage(target) ≠ ∅` and predictor derivation is not independent, the predictor is marked ineligible for cross-measurement transport.

**Expected result**: E1 → `BLOCK_TRANSPORT_PASS`; E2 → `PRESERVE_REAL_EXTERNAL`.

---

## 2. Fixture: censoring_misclassification

**Manipulation**: handle out-of-range (right-censored, beyond 40 mM) samples by dropping, exactifying, or wrong-direction.

| Handling | Effect | Expected audit output |
|----------|--------|------------------------|
| correct likelihood | baseline | score preserved |
| complete-case (drop) | n shrinks, bias in effect | **bias quantified** |
| exactify (treat censored as exact) | variance/score distortion | **bias quantified** |
| wrong-direction | sign distortion | **bias quantified** |

**Quantification**: compare censored proper score vs. each corrupted handling; report delta in score, calibration and effect. Bias is "detectable" when the corrupted handling produces a materially different (worse or sign-flipped) result vs. the correct likelihood.

**Expected result**: `BIAS_QUANTIFIED` for all three corruptions; correct likelihood preserved.

---

## 3. Fixture: component_imbalance

**Manipulation**: build the 83/11/2/2 graph and a balanced graph.

**Quantification**: report micro, component-weighted (macro), and target-policy estimates separately. The estimand difference is "captured" when the three estimates differ under imbalance and converge under balance.

**Expected result**: `ESTIMAND_DIFFERENCE_CAPTURED`; macro/micro/policy diverge for 83/11/2/2, converge for balanced.

---

## 4. Fixture: baseline_failure

**Manipulation**: a complex model that only learns the motif/group mean.

**Quantification**: run a matched simple baseline (motif_mean) against the complex model. A "pseudo-gain" is revealed when the complex model's apparent gain vanishes or inverts once the matched baseline is included.

**Expected result**: `PSEUDO_GAIN_REVEALED` — matched simple baseline exposes the phantom gain.

---

## 5. Fixture: coverage_width_tradeoff

**Manipulation**: inflate interval width to maximise coverage.

**Quantification**: nominal coverage rises with width, but the joint criterion (coverage + width/usefulness band) must reject the useless infinite-width interval.

**Expected result**: `USELESS_UNCERTAINTY_REJECTED` — coverage alone does not pass; width constraint enforces the rejection.

---

## 6. Acceptance criteria (B1 PASS)

Each fixture must match its expected audit outcome. Report:
- false-pass rate (audit missed a planted failure),
- false-fail rate (audit rejected a valid experiment),
- boundary conditions (where the audit flips between detect/not-detect),
- software versions.

B1 PASS = all five fixtures produce the expected outcome with no false-pass and no false-fail on the planted cases.

---

## 7. Expected result summary table

| Fixture | Expected audit outcome | Verdict |
|---------|------------------------|---------|
| endpoint_reuse (E1/E2) | BLOCK_TRANSPORT_PASS / PRESERVE_REAL_EXTERNAL | PASS |
| censoring_misclassification | BIAS_QUANTIFIED | PASS |
| component_imbalance | ESTIMAND_DIFFERENCE_CAPTURED | PASS |
| baseline_failure | PSEUDO_GAIN_REVEALED | PASS |
| coverage_width_tradeoff | USELESS_UNCERTAINTY_REJECTED | PASS |