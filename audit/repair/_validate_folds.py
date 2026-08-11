import sys
sys.path.insert(0, "/home/cunyuliu/rna_junction_repair_20260811")
from pathlib import Path
from audit.data.audit_dataset import audit_dataset
from audit.repair.fold_loader import build_joint_edit_context_folds

_, admitted, profile, *_ = audit_dataset(
    Path("/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"))
sp = build_joint_edit_context_folds(admitted)
print("n_folds", len(sp))
m = sp[0].to_manifest()
print("manifest keys", list(m.keys()))
print("first fold", {k: m.get(k) for k in
                     ("axis", "fold", "blocked_sequence_groups", "blocked_context_groups")})
# zero-overlap sanity on the first fold
tr = set(m["train_ids"]); te = set(m["test_ids"])
print("train_test_overlap", len(tr & te))
print("n_train", len(tr), "n_test", len(te))
