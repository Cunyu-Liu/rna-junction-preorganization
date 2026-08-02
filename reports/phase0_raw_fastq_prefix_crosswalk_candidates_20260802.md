# Phase 0 raw FASTQ construct-prefix crosswalk candidates — 2026-08-02

## Scope and method

This audit uses only the two public main-library trial R2 FASTQ files already
present on the remote host. It reads the first 250,000 gzip records from each
file, removes the official 5′ primer `GGGCTTCGGCCC`, assigns an exact 60-nt
construct prefix against the 7,500-row `library_sequences.csv`, and compares
the resulting sampled count vector with each processed construct JSON's
`num_reads` vector. The script does not run `rna-map`, does not scan the full
FASTQ files, and does not admit an accession/condition binding.

The processed archive is the current Figshare `data.zip`, SHA-256
`241d15141298ce78471b360f598fd981c7870aab5ba19b9716f64b057bdfd681`.
The prefix table has 7,500 unique prefixes under the source-defined
`seq_N → constructN` and U→T rules.

## Results

| raw run | sampled records | primer-prefix records | unique assigned constructs | `pdb_library_1` Pearson | `pdb_library_2` Pearson | `pdb_library_3` Pearson |
|---|---:|---:|---:|---:|---:|
| `SRR31402664` / `rna_library_trial1` | 250,000 | 134,790 | 7,445 | 0.946595891 | 0.828633160 | 0.904793694 |
| `SRR31402663` / `rna_library_trial2` | 250,000 | 137,557 | 7,449 | 0.851873000 | 0.941647693 | 0.908331503 |

The corresponding Spearman correlations are:

| raw run | `pdb_library_1` | `pdb_library_2` | `pdb_library_3` |
|---|---:|---:|---:|
| `SRR31402664` / `trial1` | 0.918805431 | 0.845099482 | 0.889054772 |
| `SRR31402663` / `trial2` | 0.873612020 | 0.922100156 | 0.904095468 |

The sample therefore supports the candidate ordering
`SRR31402664 → pdb_library_1` and `SRR31402663 → pdb_library_2` more strongly
than the cross-order alternatives. It does not identify the source of
`pdb_library_3`: its correlations are intermediate for both runs. The two raw
sample vectors themselves have Pearson correlation about 0.805, so the
intermediate signal is not evidence of a unique third accession.

## Full SRA scope check

The public runinfo snapshot contains 15 rows. Ten are named
`junction_design_1` through `junction_design_10`, with approximately
0.23–0.48 million spots each. The current processed archive contains no
`junction_design` member and the public methods describe the target payload as
a 7,500-construct DMS library. Those ten runs are therefore retained as
out-of-scope candidates pending provenance; they are not silently assigned to
any processed library.

## Interpretation and gate effect

This is stronger candidate evidence than archive-level read depth, but it is
still not a formal crosswalk. Exact prefix assignment in a bounded early-file
sample is not accession-preserving provenance, and the processed `num_reads`
vector is not the output of a replayed `rna-map` run. `pdb_library_3` remains
unresolved, and no manual review row is created from this audit.

- `raw_processed_crosswalk_gate_effect`: `NO_CHANGE`
- `primary_labels_admitted`: `false`
- `phase0_gate_effect`: `NO_PHASE_0_PASS`
- `scientific_gate_effect`: `NO_UNLOCK`
- `training_started`: `false`
