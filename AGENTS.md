# Execution contract

1. Read and hash `contract/1.1.docx` before any substantive change.
2. Preserve all user changes and all unrelated processes/runs.
3. Advance phases only after the preceding acceptance record is complete and
   internally consistent.
4. Record every run ID, absolute path, input/code/contract hash, environment,
   seed, command, result, failure label, and stop rule.
5. Do not disclose or copy controlled sequence/label/effect-value content into
   ordinary status updates. Report only the metadata needed for audit.
6. GPU work is fail-closed: require an explicit CUDA device probe and stop if
   CUDA is unavailable or execution silently falls back to CPU.
7. Do not use a shared/occupied GPU in a way that can affect an existing job.
8. Commit only focused changes in this repository. Do not push until a
   concrete GitHub remote/repository is known and the commit scope is verified.

The contract's failure taxonomy is authoritative. In particular, transport
may be `NO_BRIDGE_SUPPORT`, `MODEL_CONDITIONAL`, or `ROBUST`; operator status
may be `OPERATOR_SENSITIVE`, `PARTIAL_ONLY`, or `ROBUST`.
