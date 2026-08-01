# Contract source transfer status

The local source document was read in full and hashed before remote setup:

`218dec34037487fae14c50eef2aeb28b79292fe428bd4917a9da1f36687aa0e9`

The remote project currently contains the hash record and the execution
summary, but not the original `1.1.docx`. A security review blocked copying
the non-public source document to the remote host because the current request
did not provide payload-and-destination-specific egress authorization.

This is intentionally fail-closed. The project must not claim that the remote
contract bytes were verified until either:

1. the user explicitly authorizes sending this exact document to
   `cunyuliu@36.137.135.49:/home/cunyuliu/rna_junction_preorganization_v1_1_20260801/contract/1.1.docx`,
   after which the remote SHA-256 must match exactly; or
2. the user supplies an already-authorized remote copy whose SHA-256 matches
   the recorded value.

Until then, Phase 0/0.5 execution remains blocked at the contract-source
deployment check. No data, weights, run artifacts, existing repositories, or
processes were modified by this condition.
