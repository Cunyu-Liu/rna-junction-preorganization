# rna-audit-b3

RNA thermodynamic evidence admissibility audit — B3 generative multi-regime
benchmark software.

## What this is

A reusable, empirically validated benchmark for the RNA thermodynamic evidence
admissibility audit. It replaces the v1.4 B0/B1 prototype (hard-coded booleans,
schema-parse-only) with a generative, multi-regime, Monte-Carlo benchmark in
which each regime has a frozen data-generating process (DGP), a ground-truth
transport-claim validity label, a frozen seed list, and a replicate count.

The audit detector reads the raw generated registry and *computes* an audit
decision (endpoint identity -> source membership -> censoring -> graph support ->
baseline parity -> coverage-width -> claim provenance). It never hands itself the
expected status.

## Install

```bash
pip install -e .
```

Requires Python >= 3.9 and numpy.

## Quickstart

```bash
# Run the full benchmark across all regimes and frozen seeds
rna-audit-b3 run --out ./results

# Run module ablations (which errors each audit module prevents)
rna-audit-b3 ablate --out ./results
```

## What it reports

- Detector-level sensitivity / specificity / false-pass / false-fail / power
  with Monte-Carlo CIs across frozen seeds and regimes.
- Per-regime detection rates.
- Module ablations quantifying the false-pass inflation allowed when a single
  audit module (endpoint, censoring, graph, baseline, coverage-width,
  provenance) is removed.

## Regimes

- valid_transport (VALID)
- endpoint_reuse (INVALID)
- censoring_misclassification (INVALID)
- component_imbalance (INVALID)
- baseline_failure (INVALID)
- coverage_width_inflated (INVALID)
- split_leakage (INVALID)
- no_signal_null (INVALID)
- source_unresolved (INVALID)
- boundary (BOUNDARY)

## Detector checks (computed from raw data, never from the expected label)

Each audit payload carries per-module verdicts. Decisions are *computed* from
the raw arrays:

- **endpoint identity** — predictor must be an independent measurement system
  (`platform_ok`), else BLOCK.
- **censoring** — if censoring is claimed, censored rows must sit at the high
  end of the truncation variable (log10-mid separation); if no censoring is
  claimed the check trivially passes. Misclassification is caught directly.
- **graph support / component adequacy** — component-aware holdout is required,
  and no component may be too small to support a reliable holdout
  (`MIN_COMP_SAMPLES`). This is what flags the qMaP-like `80/11/2/2`
  `component_imbalance` regime.
- **baseline parity** — a threshold gain must not be a pseudo-gain against a
  matched strong baseline.
- **coverage-width** — the 80% interval coverage must fall inside the frozen
  calibration band (with a Wilson descriptive check) and the interval must not
  be inflated.
- **claim provenance** — source membership must be fully source-authored (not
  unresolved `FIT_IDENTIFIED`).

## DGP schemas

- `valid_transport` and all specific-defect regimes use a balanced schema in
  which every component is large enough for a reliable holdout.
- `component_imbalance` deliberately reproduces the unbalanced `80/11/2/2`
  structure so graph-support catches the small components.
- `boundary` uses a larger schema so the gain lands reliably inside
  `[MEANINGFUL_GAIN*0.5, MEANINGFUL_GAIN)` across frozen seeds.

## License, data, and reproducibility

- Code license: MIT (see LICENSE).
- Data: synthetic, generated from frozen DGP specs and seeds; no external data
  required.
- See CITATION.cff for citation metadata.

## Notes

- The B3 PASS threshold is frozen at N1. A decision of `B3_VALIDATED`,
  `B3_PARTIAL_REQUIRES_DOWNGRADE`, or `B3_FAILED_STOP_METHODS_CLAIM` is produced
  by the CLI from actual computed metrics, never from hard-coded expected status.