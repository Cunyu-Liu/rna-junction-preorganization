# 《1.1》执行摘要（只读提取，原件为唯一权威）

## Scientific center

The contract asks which RNA two-way junction preorganization-related
functionals remain robustly identifiable when the target estimand, cross-assay
transport, probe operator, and symmetry coordinates are uncertain. The public
data route must build a censored/conditional/correlation-aware generative
inverse model, report probe-conditioned or target-specific functionals and
their identified set over the allowed transport/operator sets, and test
whether DMS provides attributable incremental information in a paired nested
split. A sequence predictor is a later deployment layer, not the starting
point.

The scoped object is normalized 1x1 and 2x2 RNA two-way junctions. The project
does not assume that DMS and tectoRNA observe the same physical ensemble. A
sequence match is an anchor only; a registered transport model is required.

## Hard gates

The following are immediate blockers:

1. If `Omega*` or `U*` is not frozen, do not report intrinsic preorganization
   free energy; report only probe-conditioned functionals.
2. If `AssayTransportSpec` has no calibratable transport family, do not use a
   joint shared-latent likelihood; downgrade DMS to weak prior/external
   covariate or a transport-failure result.
3. If `OperatorUncertaintySpec` has no stable functional or mode over the
   allowed operator set, do not claim pose, stiffness, or tomography; report
   an identified set and probe-design recommendation.
4. If `SymmetryFrameSpec` fails round-trip, strand-swap, or boundary/index
   tests, pause cross-modal matching, geometry labels, and latent-parameter
   training.

## Phase order and acceptance

### Phase 0 — data, provenance, and semantic audit

Complete the public-source registry, Denny field/schema and filtering audit,
the -7.1 kcal/mol censoring direction, 9/11-bp raw versus interpolated naming
and generation logic, DMS JSON/count/background/read-depth hierarchy, and a
matched/ambiguous/rejected motif table. Manually audit at least 50 matched
cases and 30 rejected/ambiguous cases. Pass requires reproducible or
row-explained counts, no fatal field ambiguity, candidate matching accuracy at
least 95%, and traceable primary labels. Failure stops modeling.

### Phase 0.5 — freeze the four scientific specifications

Freeze `EstimandSpec` (normalized partition function, E1/E2/E3, units, gauge,
and any candidate `Omega*`/`U*`), `AssayTransportSpec` (environment variables,
bridge families, calibration evidence, failure label),
`OperatorUncertaintySpec` (operator source, ensemble/range, contact/electrostatic
reference), and `SymmetryFrameSpec` (frame, group action, parameter transport,
index map, round-trip tests). Run bridge-power and operator-robustness
simulation, plus normalized-forward, identity-bridge, and incorrect-strand-
swap counterexample tests. Pass requires executable non-contradictory specs,
all sanity/round-trip checks, and at least one testable E1/E3 candidate.

### Later gates

Phase 1 freezes registry/QC/benchmark and nested group splits. Phase 2 must
demonstrate synthetic coverage and controlled global alias behavior over the
operator ensemble. Phase 3 is tectoRNA-only real-data inversion. Phase 4
tests transport and DMS increment with paired nested splits, permutations,
negative controls, and capacity-matched baselines. Phase 5 requires a frozen
target kernel and calibrated useful-width target functional. Phase 6 sequence
deployment is conditional on at least L3 evidence. Phase 7 external/second
system validation and Phase 8 reproducibility/evidence-ledger freeze are
conditional extensions.

## Evaluation and evidence boundaries

- The statistical unit is the independent motif group, not a Delta-G unit,
  DMS nucleotide, read, scaffold row, or flow-piece cell.
- Use nested group splits over canonical motif, symmetry, family, construct,
  scaffold, study, and connected components as applicable.
- Primary metrics include censored NLL, held-out probe prediction,
  identified-set diameter/coverage/width, robust sign stability, calibration,
  and group-level uncertainty. Do not use R2 or ordinary MSE on censored
  observations as the sole evidence.
- Global alias search, profile likelihood/posterior, SBC, misspecification
  recovery, operator/transport union, and permutation/negative controls are
  required before strong mechanism language.
- Primary claims must be mapped to an evidence ledger L0-L7 with scope labels
  such as `conditional`, `robust`, `partial`, or
  `transport-model-conditional`.

## Prohibited shortcuts and claim limits

No random row split, no interpolated primary test values, no test-posterior
label regression, no PDB single structure/MD/inferred posterior as error-free
truth, no un-audited symmetry/transport concatenation, no private-data core
claim, no conditional result written as robust, and no partial identification
written as point estimation. “Orthogonal”, “tomography”, “complete energy
landscape”, “accurate dynamics”, “intrinsic preorganization” without a frozen
target, and “general RNA 3D prediction” are not default claims.

## Resource and reproducibility policy

The Git code/contract root is under `/home/cunyuliu`. Large public data,
downloads, caches, checkpoints, weights, and run artifacts are under the
separate `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801` root.
Each run must register experiment ID, Git commit, data/contract/spec hashes,
split version, seed, environment, hardware/software, equation/prior,
transport/operator sets, target kernel, thresholds, stop rule, failures, and
result hashes. Existing jobs and dirty repositories are outside this scope
and must not be modified.
