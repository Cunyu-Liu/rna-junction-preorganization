#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNA thermodynamic evidence admissibility audit — B3 generative benchmark.

B3 is the hardened, empirically validated benchmark for the audit procedure
(§9). It replaces the v1.4 B0/B1 prototype (hard-coded booleans, schema-parse-only)
with a generative, multi-regime, Monte-Carlo benchmark:

  - Each regime has a frozen data-generating process (DGP), a ground-truth
    transport-claim validity label, a seed list, and a replicate count.
  - The audit detector reads the raw generated registry and *computes* an audit
    decision (it is never handed the expected status).
  - We report detector sensitivity / specificity / false-pass / false-fail /
    power / calibration error / coverage / width / runtime with Monte-Carlo CIs
    across frozen seeds.
  - We run ablations that remove single audit modules (endpoint, censoring,
    graph, baseline, coverage-width, provenance) to quantify the errors each
    module prevents.

This package is the concrete fulfilment of the omics benchmarking guideline
(Mangul 2019) for the RNA-thermodynamic audit domain.
"""

__version__ = "0.1.0"