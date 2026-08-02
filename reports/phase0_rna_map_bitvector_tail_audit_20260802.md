# Phase 0 RNA-MAP bit-vector tail audit — 2026-08-02

## Scope

This report explains the observed long tail in the isolated RNA-MAP replay. It
is an engineering/runtime audit only. It does not change the replay, replace
the official implementation, or create a scientific result.

## Runtime evidence

| Item | Evidence |
|---|---|
| Run | `rna_map_full_replay_SRR35766784_retry_envpath_20260802` |
| Run root | `/mnt/cunyuliu/rna_junction_preorganization_v1_1_20260801/phase0/runs/rna_map_full_replay_SRR35766784_retry_envpath_20260802` |
| Reference set | 7,500 valid reference sequences, recorded in `logs/rna_map_full.log` |
| Paired reads | 92,296,217 read pairs, recorded by bowtie2 in the same log |
| Alignment output | `output/Mapping_Files/aligned.sam`, 57,720,931,811 bytes at the latest inventory |
| Alignment | 98.85% overall alignment rate |
| Runtime state | Worker PID `3274097`, state `R`, CPU approximately 46%, RSS approximately 201 MB at the latest check |
| Current phase | `INFO - rna-map.BIT_VECTOR - starting bitvector generation` |
| Current gate | `NO_PHASE_0_PASS`; `NO_UNLOCK` |

The worker's SAM file descriptor was observed advancing to approximately
10.71 GB, and `/proc/3274097/io` showed increasing `read_bytes`. This is
positive liveness evidence, not completion evidence.

## Source-level explanation

The inspected environment is:

```text
/home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1
rna-map 0.4.1
```

The relevant installed source files are:

```text
/home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/bit_vector.py
/home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/sam.py
```

The official implementation performs the following work:

1. `BitVectorGenerator.__generate_all_bit_vectors` iterates over every item
   from `BitVectorIterator`.
2. `PairedSamIterator.__next__` reads and parses two SAM lines per iteration.
3. `__record_bit_vector` calls `__update_mut_histo` for accepted reads.
4. `__update_mut_histo` loops over `mh.get_nuc_coords()` for the whole
   reference coordinate set and performs Python dictionary lookups and counter
   updates for each coordinate.
5. `summary_output_only` disables per-read bit-vector file writing, but it does
   not disable SAM parsing or mutation-histogram updates.

This explains why the mapping stage can finish quickly while the bit-vector
stage remains CPU-bound for hours. As a scale intuition only, if the typical
reference length is about 143–144 nucleotides, the unoptimized upper-bound
work is on the order of 92 million paired-read iterations times roughly 143
coordinate checks. This is an implementation-level workload estimate, not a
measured scientific metric.

## Safety decision

The current process is live and reading forward. No process was killed, paused,
reniced, patched, or restarted. No alternate parser was run against the live
SAM, because doing so would create a second unsealed processing path and could
compete for I/O with the official replay.

The low-frequency finalization watcher remains responsible for running the
replay audit after the worker exits naturally. Until then, the replay is
engineering evidence in progress and cannot be promoted to a Phase 0 or
scientific acceptance result.

## Reproducible read-only source probes

```text
grep -nEi "rejected|SAM|sam|bit|vector|for .*line|readline" \
  /home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/bit_vector.py \
  /home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/sam.py
sed -n '403,482p' \
  /home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/bit_vector.py
sed -n '63,130p' \
  /home/cunyuliu/miniconda3/envs/rna_junction_preorganization_v1_1/lib/python3.10/site-packages/rna_map/sam.py
```
