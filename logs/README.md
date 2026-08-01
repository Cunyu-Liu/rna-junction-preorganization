# Logs

Every execution log must use a unique UTC run ID and record the exact command,
working directory, contract/spec/data/code hashes, environment, hardware,
stdout/stderr, exit code, terminal status, and failure label when applicable.

Logs are evidence. Do not delete or overwrite a failed run; use a new run ID.
Large mutable logs belong in the external artifact root when they are not
needed for code review. This directory retains small governance and gate
records.
