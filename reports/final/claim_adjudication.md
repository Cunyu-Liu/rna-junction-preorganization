# RNA Junction Preorganization v1.2 — Claim Adjudication

**Run**: `v1_2_tecto_qmap_20260803`
**Contract**: `rna 三级.md` (SHA-256 `32d09729638b7681b6efcfdf8b2addc3c7f83060e37ce5ef3dd5c5a051702252`)
**Finalized**: 2026-08-04 UTC
**Adjudicated claim class**: `STRONG_CROSS_SYSTEM_RESTRICTED`
**qMaP terminal state**: `QMAP_TRANSFER_SUPPORTED`

---

## 1. Result combination (per contract §24)

| Branch | Engineering gate | Scientific disposition | qMaP terminal disposition |
|---|---|---|---|
| tecto main line | T0–T3 **PASS** | `INCONCLUSIVE_FOR_1_KCAL_PRECISION` (tecto-only) | — |
| qMaPseq second system | Q0–Q5 **PASS** | cross-system transfer supported (restricted) | `QMAP_TRANSFER_SUPPORTED` |

Matched contract row: **Tecto PASS + QMAP_TRANSFER_SUPPORTED** → allowed max claim:

> **Restricted cross measurement-system transfer**: the tecto model (old_dg) with isotonic calibration supports a restricted cross measurement-system transfer claim on held-out RNA-MaP reference ΔG, within the 98-variant mttr6 TTR mutant set.

## 2. Core scientific results (measured, not proxy)

### Tecto main line (T0–T3, platform context)
- **Identified set / interval**: median interval width = **1.35 kcal/mol** (p90 = 1.66); 11.2% of junction intervals ≤ 1.0 kcal/mol. Operator sensitivity: `dg11`/`dg10_5mM` tighten to 0.55/0.50 kcal/mol; frozen primary operator = 10 mM Mg²⁺.
- **Coverage / calibration**: interval coverage on held-out = **0.957**; M0 synthetic coverage in [0.9, 1.0].
- **Signal detection** (T2 permutation): real between-junction SD = 0.384 vs permuted 0.279, `signal_detected = true`.
- **Model vs baseline** (T3, motif-family holdout, n=392): hierarchical proper score 41.8 vs motif_mean 27.0, `t3_beats_baseline = false`. Honest negative on the tecto-only platform.

### qMaPseq second system (Q0–Q5, the cross-system evidence)
- **Q0 provenance**: ENA PRJNA1086549 (8 runs / 16 FASTQ, SHA-256 verified), YesselmanLab/rna_map @ 2d7337d (Apache-2.0), Figshare 10.6084/m9.figshare.25331758 (md5=7a080dc7...), Zenodo 10.5281/zenodo.11672684 (md5=48da131a...). All admitted.
- **Q1 registry**: 99 variants (1 reference + 98 mutants), cross-referenced Zenodo rna_map_dg.csv + Figshare mttr6_data_full.json.
- **Q2 attrition**: 84 fitted + 11 right-censored + 2 closing-pair abnormal + 1 alternate-structure = 98. Censored enter likelihood, not deleted.
- **Q3 endpoint replay**: 98 variants × 8 endpoints, tolerances frozen BEFORE run. **1600 PASS + 66 NOT_APPLICABLE + 0 FAIL**. No trend-only pass; categorical exact match; 11 censored exempt from E2/E3 numerical (authoritative = E7).
- **Q4 freeze**: 98 mttr6 TTR mutants; mutation graph 193 edges, 4 components (sizes [83, 11, 2, 2]); K=4 folds, 0 leakage. All 12 freeze items locked before viewing transfer outcome.
- **Q5 locked transfer test**: B4 (tecto old_dg + isotonic calibration) RMSE = **0.1951 kcal/mol**; gain over B1 = **0.511** (95%CI [0.403, 0.618], p=0.0006355); 68% coverage = 0.682; label permutation p = 0.0; Spearman = 0.947. Mutation-class bootstrap 95%CI = [0.359, 0.630].

## 3. Adjudication criteria (Q5, all PASS)

| Criterion | Threshold | Result |
|---|---|---|
| (a) B4 RMSE < 1.0 kcal/mol | < 1.0 | 0.1951 **PASS** |
| (b) Gain over B1 > 0.3, 95%CI excludes 0 | > 0.3, CI excl 0 | 0.511, CI [0.403,0.618] **PASS** |
| (c) 68% coverage in [0.55, 0.80] | [0.55,0.80] | 0.682 **PASS** |
| (d) Label permutation p < 0.05 | < 0.05 | 0.0 **PASS** |

→ **QMAP_TRANSFER_SUPPORTED**

## 4. Adjudication vs. claim caps

| Claim | Status |
|---|---|
| Restricted cross measurement-system transfer (tecto→RNA-MaP ΔG, 98 mttr6 TTR mutants, 4-fold mutation-graph CV) | **ALLOWED_SUPPORTED** |
| Partial-ID / identified-set functional for tectoRNA junction ΔG (10mM Mg²⁺) | **ALLOWED** (tecto-only context) |
| Tecto-specific reanalysis / benchmark / case study | **ALLOWED** |
| Operator-robustness comparison (5/11/9 mM sensitivity) | **ALLOWED_SENSITIVITY** |
| 1 kcal/mol precision thermodynamic functional (tecto-only) | **NOT SUPPORTED** (width 1.35 > 1.0) |
| Hierarchical model beats simple baseline (tecto-only T3) | **NOT SUPPORTED** (negative gain) |
| Cross-system transfer to arbitrary TLR families | **NOT SUPPORTED** (restricted to 98 mttr6 TTR mutants) |
| Biological mechanism / junction-preorganization mechanism paper | **NOT SUPPORTED** → decline mechanism position |

## 5. Restrictions on the cross-system claim
1. Restricted to 98 mttr6 TTR mutants — no extrapolation to arbitrary TLR families.
2. 4-fold CV with unbalanced fold sizes (83/11/2/2) due to mutation graph structure.
3. B4 uses tecto old_dg as input — requires a pre-existing tecto model.
4. 11 censored variants enter likelihood but their mg_1_2 is unreliable.
5. B2 (mg_1_2 univariate) performs poorly due to censored variants with extreme mg_1_2.

## 6. Permanently prohibited claims (contract §24)
- [x] current 7,500-construct DMS validated tectoRNA
- [x] DMS universally equivalent to thermodynamic ΔG
- [x] qMaPseq independently reproduced junction preorganization (transfer ≠ reproduction)
- [x] method proven to generalize across all two-way junctions
- [x] Bonilla, Shin, Yesselman are three independent external validations
- [x] a high correlation coefficient proves the same estimand
- [x] synthetic recovery proves biological mechanism
- [x] negative result itself guarantees publication
- [x] contract completion guarantees publication
- [x] frozen RNA LM constitutes core architectural innovation

## 7. Completion status

```
IMPLEMENTATION_COMPLETE = true
ALL_12_GATES_PASS = true
QMAP_TERMINAL_STATE = QMAP_TRANSFER_SUPPORTED
MAX_ALLOWABLE_CLAIM = STRONG_CROSS_SYSTEM_RESTRICTED
```
