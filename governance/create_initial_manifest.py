#!/usr/bin/env python
"""Create the initial v1.2 CanonicalStateManifest at the frozen start state."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canonical_manifest import CanonicalStateManifest, validate_schema, sha256_file  # noqa: E402

EXPECTED_CONTRACT_SHA256 = "3ad0c9997cdea8e510f80424c4b011062f0f95a8bf8879a4659a847adcab22a0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contract-path", required=True)
    ap.add_argument("--contract-version", default="v1.2")
    ap.add_argument("--contract-sha256", default=EXPECTED_CONTRACT_SHA256)
    ap.add_argument("--code-commit", default="")
    ap.add_argument("--run-id", default="v1_2_tecto_qmap_unbound")
    ap.add_argument("--parent-run-id", default="v1_2_tecto_qmap_20260803")
    ap.add_argument("--host", default="")
    ap.add_argument("--env-lock-hash", default="")
    args = ap.parse_args()

    actual_contract_sha256 = sha256_file(args.contract_path)
    if actual_contract_sha256 != EXPECTED_CONTRACT_SHA256 or actual_contract_sha256 != args.contract_sha256:
        print(f"CONTRACT_HASH_ERROR expected={EXPECTED_CONTRACT_SHA256} actual={actual_contract_sha256}")
        return 2

    m = CanonicalStateManifest.new(
        contract_version=args.contract_version,
        contract_sha256=args.contract_sha256,
        code_commit=args.code_commit,
        worktree_path=args.worktree,
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        host=args.host,
        environment_lock_hash=args.env_lock_hash,
    )
    # Validate schema before writing
    errors = validate_schema(m.data)
    if errors:
        print("SCHEMA_ERRORS:", "; ".join(errors))
        return 2
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    h = m.save(args.out)
    print(f"WROTE {args.out}")
    print(f"SHA256 {h}")
    print("STATE ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())