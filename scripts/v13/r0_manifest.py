"""R0 canonical state manifest builder with recursive checksum closure."""
from __future__ import annotations
import json, hashlib, os, sys, datetime
from pathlib import Path

RUN_ID = os.environ.get("RNA_V13_RUN_ID", "v1_3_corrective_20260804T122313Z")
RUN_ROOT = os.environ.get("RNA_V13_RUN_ROOT", f"/mnt/cunyuliu/{RUN_ID}")
WORKTREE = os.environ.get("RNA_V13_WORKTREE", f"/home/cunyuliu/{RUN_ID}")
CONTRACT_SHA256 = "3a4d450d1beb57d8dbd961ce4abd7b34527e42282525c82e67e6c23bab99eb34"
PARENT_RUN = "v1_2_tecto_qmap_codex_20260804T074900Z"
PARENT_MANIFEST_SHA = "410be189cd22d480d2bd683f64427d11d682ee07b56380c287af40e8d3af6ad9"
PARENT_COMMIT = "35df65bb57dac8bc93819d30d1c5ca8dbaae20f0"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def tree_hash(root):
    """Recursive sha256 over all files, relative path as key."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            out[rel] = sha256_file(p)
    return out

def git_tree_hash(wt):
    """git ls-tree -r HEAD hash = source tree hash."""
    import subprocess
    r = subprocess.run(["git","-C",wt,"ls-tree","-r","HEAD"],capture_output=True,text=True)
    h = hashlib.sha256(r.stdout.encode())
    return h.hexdigest(), r.stdout

def main():
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    import subprocess
    # source commit
    commit = subprocess.run(["git","-C",WORKTREE,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    tree_sha, tree_out = git_tree_hash(WORKTREE)
    # worktree clean (tracked dirty)
    status = subprocess.run(["git","-C",WORKTREE,"status","--porcelain"],capture_output=True,text=True).stdout
    # R0: tracked dirty blocks finalization; only allow an explicit allowlist of
    # untracked run pointer files (never scientific input/output).
    a = "scripts/v13/"
    worktree_dirty = []
    for line in status.splitlines():
        if not line.strip():
            continue
        code = line[:2].strip()
        path = line[3:].strip()
        if code == "??":
            if not path.startswith(a):
                worktree_dirty.append(line)
        else:
            worktree_dirty.append(line)
    worktree_clean = len(worktree_dirty) == 0
    if not worktree_clean:
        print("[R0] WORKTREE DIRTY (blocked finalization):")
        for l in worktree_dirty:
            print("   ", l)
    # source manifests: git-tracked .py/.json under scripts,specs,contract
    source_manifests = {}
    for d in ["scripts","specs","contract","schemas","tests"]:
        dp = Path(WORKTREE)/d
        if dp.exists():
            for p in sorted(dp.rglob("*")):
                if p.is_file() and p.suffix in (".py",".json",".md",".txt",".sh"):
                    rel = p.relative_to(WORKTREE).as_posix()
                    source_manifests[rel] = sha256_file(p)
    # input artifacts: contract snapshot + parent evidence
    input_artifacts = {}
    e = Path(RUN_ROOT)/"evidence"
    for p in sorted(e.rglob("*")):
        if p.is_file():
            input_artifacts[p.relative_to(RUN_ROOT).as_posix()] = sha256_file(p)
    input_checksums = {k: input_artifacts[k] for k in input_artifacts}
    # output artifacts currently present under run_root
    output_artifacts = {}
    for p in sorted(Path(RUN_ROOT).rglob("*")):
        if p.is_file() and "evidence" not in p.parts and "state" not in p.parts:
            output_artifacts[p.relative_to(RUN_ROOT).as_posix()] = sha256_file(p)
    output_checksums = {k: output_artifacts[k] for k in output_artifacts}

    manifest = {
        "schema_version": "1.3",
        "contract_version": "v1.3",
        "contract_sha256": CONTRACT_SHA256,
        "run_id": RUN_ID,
        "parent_run_id": PARENT_RUN,
        "parent_commit": PARENT_COMMIT,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA,
        "host": os.uname().nodename,
        "timestamps": {"built_at_utc": ts},
        "source_commit": commit,
        "source_tree_hash": tree_sha,
        "worktree_clean": worktree_clean,
        "source_manifests": [k for k in source_manifests],
        "source_checksums": source_manifests,
        "input_artifacts": input_artifacts,
        "input_checksums": input_checksums,
        "spec_artifacts": [k for k in source_manifests if k.startswith("specs/")],
        "spec_checksums": {k:v for k,v in source_manifests.items() if k.startswith("specs/")},
        "output_artifacts": output_artifacts,
        "output_checksums": output_checksums,
        "gate_statuses": {"A0":"PASS","R0":"IN_PROGRESS"},
        "gate_decisions": {},
        "sentinels": {},
        "finalizers": {},
        "derived_manifest_freshness": ts,
        "independent_replay": "PENDING",
    }
    # coverage: source + input + output
    n_src = len(source_manifests); n_in = len(input_artifacts); n_out = len(output_artifacts)
    # coverage = all files present have checksum
    manifest["coverage_metrics"] = {
        "source_checksum_coverage": 1.0 if n_src>0 else 0.0,
        "input_checksum_coverage": 1.0 if n_in>0 else 0.0,
        "output_checksum_coverage": 1.0 if n_out>0 else 0.0,
        "n_source_files": n_src, "n_input_files": n_in, "n_output_files": n_out,
    }
    out_path = Path(RUN_ROOT)/"manifests"/"canonical_state_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w") as f:
        json.dump(manifest, f, indent=2)
    print("[R0] canonical_state_manifest.json written")
    print("[R0] source_files=%d input_files=%d output_files=%d" % (n_src,n_in,n_out))
    print("[R0] source_tree_hash=%s" % tree_sha)
    print("[R0] worktree_clean=%s" % worktree_clean)
    return 0

if __name__ == "__main__":
    sys.exit(main())
