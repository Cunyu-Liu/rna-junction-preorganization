"""P0 repair machinery (strict audit 2026-08-11).

The decisive joint-runner defect: corrected v1.31 (full) computed joint
`train_ids` but never consumed them, so the full model trained on test
contexts while no-sequence and all baselines correctly blocked them.  This
package provides the typed FoldSpec, a shared fold loader, the optimizer
gate, frozen v3 specs and the re-adjudication verdicts so every model
consumes the SAME row set and gate.
"""
