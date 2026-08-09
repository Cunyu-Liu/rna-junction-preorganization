# Data Card

- Source: Denny 2018 tectoRNA canonical records (canonical source persisted to
  run root; SHA-256 `0989ddc0...` in authority/).
- Admitted universe: 11,893 junction x scaffold/context rows; 1,336 junctions;
  234 admitted helix contexts; 9 scaffolds/operators; study = 1.
- Right-censored fraction: 16.25% (y >= -7.1 kcal/mol recorded as censored).
- Panel structure: each junction observed in 4-9 scaffold/contexts (median 9);
  rows are NOT independent biological samples.
- Cleaning ledger: data/CleaningLedger.jsonl (per-row layer/reason).
- Leakage control: mmseqs/grouped splits, frozen SplitManifests, overlap audit.
