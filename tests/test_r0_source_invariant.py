"""R0 source-tree invariant tests (v1.3).

Verifies that tracked source scripts (q5_build.py, finalize_q5.py) do NOT
self-write or modify the tracked source tree at runtime, and that the
r0_manifest builder correctly flags tracked-dirty worktrees.
"""
import hashlib
import json
import os
import subprocess
import unittest

WT = os.environ.get("RNA_V13_WORKTREE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestSourceTreeInvariant(unittest.TestCase):
    def test_no_self_write_q5_build(self):
        """q5_build.py must not write_text to its own tracked path."""
        src = open(os.path.join(WT, "scripts", "q5_build.py")).read()
        # The dangerous self-overwrite pattern must be gone.
        self.assertNotIn("build_script_path.write_text", src)
        self.assertNotIn("(WT / \"scripts\" / \"finalize_q5.py\").write_text", src)
        self.assertNotIn("(WT / \"scripts\" / \"q5_build.py\").write_text", src)

    def test_no_self_write_finalize_q5(self):
        """finalize_q5.py must not write its own source."""
        src = open(os.path.join(WT, "scripts", "finalize_q5.py")).read()
        self.assertNotIn("write_text", src)
        self.assertNotIn("finalize_q5.py", src.split("os.path", 1)[0] if "os.path" in src else "finalize_q5.py")

    def test_source_tree_clean_after_commit(self):
        """Worktree must be clean (no tracked dirty) after R0 commit."""
        r = subprocess.run(["git", "-C", WT, "status", "--porcelain"], capture_output=True, text=True)
        dirty = [l for l in r.stdout.splitlines() if l.strip() and not l.strip().startswith("??")]
        self.assertEqual(dirty, [], f"tracked dirty files block finalization: {dirty}")

    def test_r0_manifest_has_full_coverage(self):
        """Generated canonical manifest must report full checksum coverage."""
        rr = os.environ.get("RNA_V13_RUN_ROOT", "/mnt/cunyuliu/v1_3_corrective_20260804T122313Z")
        with open(os.path.join(rr, "manifests", "canonical_state_manifest.json")) as f:
            m = json.load(f)
        cov = m["coverage_metrics"]
        self.assertEqual(cov["source_checksum_coverage"], 1.0)
        self.assertEqual(cov["input_checksum_coverage"], 1.0)
        self.assertEqual(cov["output_checksum_coverage"], 1.0)
        self.assertGreater(cov["n_source_files"], 0)


if __name__ == "__main__":
    unittest.main()