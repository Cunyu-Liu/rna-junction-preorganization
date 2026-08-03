# RNA Junction Preorganization v1.2 — Claim Adjudication

**Run**: `v1_2_tecto_qmap_20260803`
**Contract**: `rna 三级.md` (SHA-256 `32d09729638b7681b6efcfdf8b2addc3c7f83060e37ce5ef3dd5c5a051702252`)
**Finalized**: 2026-08-04 UTC
**Adjudicated claim class**: `TECTO_REANALYSIS_AND_PARTIAL_ID_CASE_STUDY`

---

## 1. Result combination (per contract §24)

| Branch | Engineering gate | Scientific disposition | qMaP terminal disposition |
|---|---|---|---|
| tecto main line | T0–T3 **PASS** | `INCONCLUSIVE_FOR_1_KCAL_PRECISION` | — |
| qMaPseq second system | Q0 **NOT_ADMITTED** | — | `QMAP_NOT_ADMITTED` |

Matched contract row: **Tecto PASS + QMAP_NOT_ADMITTED** → allowed max claim is:

> **tecto-specific reanalysis, benchmark, or partial-ID case study.**

## 2. Core scientific results (measured, not proxy)

- **Identified set / interval**: median interval width for the target-specific functional = **1.35 kcal/mol** (p90 = 1.66); only **11.2%** of junction intervals ≤ 1.0 kcal/mol. Operator sensitivity shows `dg11` and `dg10_5mM` tighten the interval (median 0.55 / 0.50 kcal/mol), but the frozen primary operator is 10 mM Mg²⁺.
- **Coverage / calibration**: interval coverage on held-out = **0.957**, validated on synthetic fixtures in M0 (coverage in [0.9, 1.0]). Units are honest; midpoint is **not** reported as a biological truth.
- **Signal detection** (T2 permutation control): real between-junction SD = **0.384** vs permuted = **0.279**, `signal_detected = true`, `calibration_drift` recovered (planted 0.5 → recovered 0.495).
- **Model vs matched baseline** (T3, motif-family holdout, n=392 rows): hierarchical (motif+scaffold) model proper score = **41.8**, strongest simple baseline (`motif_mean`) = **27.0**, relative gain = **−0.55**, `t3_beats_baseline = false`. Group bootstrap gain positive fraction = **0.0**. Held-out ranking τ = 0.06, ρ = 0.07.
- **Extrapolation boundary**: functional reported only for 1,336 scaffold-identified junctions; 0 out-of-range junctions get point estimates; 392 holdout junctions.

## 3. Adjudication vs. claim caps

| Claimed claim | Status |
|---|---|
| Partial-ID / identified-set functional for tectoRNA junction ΔG (10mM Mg²⁺), with coverage & width | **ALLOWED** (supported, INCONCLUSIVE) |
| Tecto-specific reanalysis / benchmark / case study | **ALLOWED** |
| Operator-robustness comparison (5 mM / 11 mM / 9 mM sensitivity) | **ALLOWED** as sensitivity |
| 1 kcal/mol precision thermodynamic functional | **NOT SUPPORTED** (width 1.35 > 1.0) |
| Hierarchical model beats simple baseline | **NOT SUPPORTED** (negative gain) |
| Cross-measurement-system (qMaPseq) transfer | **NOT ADMITTED** (QMAP_NOT_ADMITTED) |
| Biological mechanism / junction-preorganization mechanism paper | **NOT SUPPORTED** → decline mechanism position |

## 4. Permanently prohibited claims (checklist, contract §24)

- [ ] current 7,500-construct DMS validated tectoRNA
- [ ] DMS universally equivalent to thermodynamic ΔG
- [ ] qMaPseq independently reproduced junction preorganization
- [ ] method proven to generalize across all two-way junctions
- [ ] Bonilla, Shin, Yesselman are three independent external validations
- [ ] a high correlation coefficient proves the same estimand
- [ ] synthetic recovery proves biological mechanism
- [ ] negative result itself guarantees publication
- [ ] contract completion guarantees publication
- [ ] frozen RNA LM constitutes core architectural innovation

## 5. Completion status

```
IMPLEMENTATION_COMPLETE = true
SCIENTIFIC_SUCCESS = inconclusive
PUBLICATION_ROUTE = TECTO_REANALYSIS_AND_PARTIAL_ID_CASE_STUDY
```

## 6. Recommended route

A **tecto-specific partial-identification case study / reanalysis** paper centered on:
1. Exhaustive, auditable data census & provenance (T0/S0/T1).
2. Synthetic operator-identification validation (M0) establishing coverage/calibration of the censored-likelihood identified-set estimator.
3. Censored-likelihood partial-ID inference on the tectoRNA platform (T2) with honest intervals, coverage, and negative controls (T2: permutation, calibration drift, homolog leakage, out-of-range).
4. Honest model-vs-baseline comparison (T3) reporting that the hierarchical model does **not** beat a simple motif-mean baseline on the frozen holdout — framing the interval/coverage methodology and the platform limitation as the contribution, not a mechanism win.

Mechanism title and cross-system claims are **not** supported and must be dropped.