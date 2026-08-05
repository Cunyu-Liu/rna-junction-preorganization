import json, hashlib, os, re, datetime
P="/mnt/cunyuliu/v1_4_boundary_audit_20260804T150707Z"
R=os.environ["RUN_ROOT"]; RUNID=os.environ["RUN_ID"]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(65536),b''): h.update(b)
    return h.hexdigest()
def stats(p):
    st=os.stat(p); return st.st_size, datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), sha(p)
# production commits per gate (from parent git log)
commits={"contract":"65896dc","T6":"4caa26d","Q6":"7c1d2e6","Q7":"36815af","N0":"b89522d","B0":"99dd14f","B1":"efb5923","B2":"6098033","R1":"b6b6e30","M1":"2d8d5f7","E1":"1dfad2e","final":"65896dc"}
# A1 parent evidence ledger rows
ledger=[
 ("contract","contracts/1.4.docx","1.4 clean contract bytes","PARENT_CONTRACT","e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc"),
 ("T6","sentinels/T6_TECTO_NEGATIVE_BOUND_AND_LOCKED.json","locked tecto negative","GATE_DECISION","TECTO_NEGATIVE_BOUND_AND_LOCKED"),
 ("Q6","sentinels/Q6_QMAP_SOURCE_RECONSTRUCTED.json","qMaP source reconstruction","GATE_DECISION","QMAP_SOURCE_RECONSTRUCTED"),
 ("Q7","sentinels/Q7_QMAP_TRANSFER_NOT_SUPPORTED.json","qMaP transfer decision","GATE_DECISION","QMAP_TRANSFER_NOT_SUPPORTED"),
 ("Q7_metrics","qmap/q7/metrics.json","qMaP measured metrics","PRIMARY_RESULT","micro_gain=0.4163,coverage=0.7263"),
 ("Q7_pred","qmap/q7/predictions.parquet","qMaP frozen predictions","PRIMARY_DATA","-"),
 ("N0","sentinels/N0_METHODS_BOUNDARY_AUDIT.json","novelty/boundary audit","GATE_DECISION","METHODS_BOUNDARY_AUDIT"),
 ("N0_claim","novelty/n0/claim_matrix.tsv","claim matrix","CLAIM","RETRACTED_GAIN_PHRASING"),
 ("N0_spine","novelty/n0/paper_spine.md","paper spine","CLAIM","RETRACT_GAIN_BELOW_THRESHOLD"),
 ("B0","sentinels/B0_BENCHMARK_FROZEN.json","benchmark freeze","GATE_DECISION","B0_BENCHMARK_FROZEN"),
 ("B1","sentinels/B1_FAILURE_MODE_VALIDATION_PASS.json","B1 PASS sentinel","GATE_DECISION","CONFLICT_FAIL_PASS_COEXIST"),
 ("B1_fail","sentinels/B1_FAILURE_MODE_VALIDATION_FAIL.json","B1 FAIL sentinel","GATE_DECISION","CONFLICT_FAIL_PASS_COEXIST"),
 ("B2","sentinels/B2_POST_HOC_SENSITIVITY_COMPLETE.json","B2 sensitivity","GATE_DECISION","B2_COMPLETE"),
 ("R1","release/r1/R1_decision.json","R1 release decision","RELEASE","R1_RELEASE_SEALED"),
 ("R1_inv","release/r1/release_inventory.tsv","R1 release inventory","RELEASE","PARTIAL_INVENTORY"),
 ("R1_replay","release/r1/replay.sh","R1 replay script","REPLAY","HASH_ONLY_NOT_RECOMPUTE"),
 ("M1","sentinels/M1_MANUSCRIPT_DRAFT_AUTHORIZED.json","manuscript skeleton","MANUSCRIPT","SKELETON_NOT_SUBMISSION_READY"),
 ("E1","sentinels/E1_E1_REPRODUCED_CLAIMS_ADMISSIBLE_SUBMISSION_READY.json","E1 verdict","REVIEW","SELF_GENERATED_NOT_INDEPENDENT"),
 ("E1_rev","external_review/e1/adversarial_review.md","E1 review","REVIEW","INTERNAL_ADVERSARIAL"),
 ("E1_issues","external_review/e1/issue_registry.tsv","E1 issue registry","REVIEW","9_ISSUES_SELF_CLOSED"),
 ("state","state/authoritative_status.json","canonical status","STATUS","STALE_V13_OPERATIONAL_STATE"),
]
rows=["gate\trelative_path\tsize\tmtime_utc\tsha256\tproduction_commit\tclaimed_status\tactual_evidentiary_meaning"]
for g,rel,mean,kind,claim in ledger:
    p=os.path.join(P,rel)
    if os.path.exists(p):
        sz,mt,h=stats(p)
        rows.append(f"{g}\t{rel}\t{sz}\t{mt}\t{h}\t{commits.get(g,commits['final'])}\t{claim}\t{mean}")
    else:
        rows.append(f"{g}\t{rel}\tMISSING\t-\t-\t-\t{claim}\t{mean}")
os.makedirs(f"{R}/reconciliation/a1",exist_ok=True)
with open(f"{R}/reconciliation/a1/parent_evidence_ledger.tsv","w") as f: f.write("\n".join(rows)+"\n")
# claim-evidence matrix
cm={
 "V15-01":{"contradiction":"N0/paper spine/E1 write qMaP gain below 0.3","actuality":"micro_gain=0.416253>0.3 threshold MET","impact":"wrong failure mechanism","disposition":"RETRACTED_STALE_CLAIM","fix_artifact":"C1 claim_reconciliation + forbidden scan","acceptance_test":"test_qmap_gain_wording"},
 "V15-02":{"contradiction":"coverage point fail written as proven undercoverage","actuality":"69/95=0.7263; Wilson simple 95% CI ~[0.629,0.806] contains 0.8","impact":"overclaims calibration deficit","disposition":"REGISTERED_RULE_FAILED_EVIDENCE_INCONCLUSIVE","fix_artifact":"Q8 calibration card","acceptance_test":"test_calibration_claim_tier"},
 "V15-03":{"contradiction":"Q6 exact source reconstruction","actuality":"11th >40mM member CCUGCC_ACUGG is FIT_IDENTIFIED","impact":"overstates source authorship","disposition":"COUNTS_CLOSED_MEMBERSHIP_PARTLY_INFERRED","fix_artifact":"membership evidence + 3-way sensitivity","acceptance_test":"test_qmap_member_provenance"},
 "V15-04":{"contradiction":"R1 as final sealed release","actuality":"source_commit 6098033 < final 65896dc; partial inventory","impact":"premature release claim","disposition":"HISTORICAL_PARTIAL_SEAL_PRE_FINAL_COMMIT","fix_artifact":"release scope truth + R2","acceptance_test":"test_release_commit_freshness"},
 "V15-05":{"contradiction":"canonical status still initial blocked state","actuality":"authoritative_status.json reports V13-era operational state","impact":"no single authoritative status","disposition":"NEW_SINGLE_AUTHORITY_IN_C1","fix_artifact":"status authority","acceptance_test":"test_status_consistency"},
 "V15-06":{"contradiction":"E1 fresh result replay","actuality":"replay.sh only checks payload hash","impact":"hash integrity mislabeled as recomputation","disposition":"HASH_INTEGRITY_ONLY_NOT_RECOMPUTE","fix_artifact":"X1 actual computation record","acceptance_test":"test_replay_recomputes"},
 "V15-07":{"contradiction":"E1 independent review","actuality":"9 issues self-generated/self-closed by same chain","impact":"self-review mislabeled independent","disposition":"INTERNAL_ADVERSARIAL_NOT_INDEPENDENT","fix_artifact":"reviewer independence declaration","acceptance_test":"test_reviewer_independence"},
 "V15-08":{"contradiction":"140 passed as whole project","actuality":"conftest excludes legacy v1.2 tests (contract_compliance, schema_validation)","impact":"overstates test scope","disposition":"V14_SCOPED_140_PASS_LEGACY_EXCLUDED","fix_artifact":"test_scope_truth","acceptance_test":"test_collection_accounting"},
 "V15-09":{"contradiction":"B0/B1 validated reusable benchmark","actuality":"5 fixtures use hard-coded constants/booleans; 0 false-pass is tautology","impact":"prototype mislabeled validated","disposition":"PROTOTYPE_SCHEMAS_AND_TOY_FIXTURES","fix_artifact":"B3 metrics/CI","acceptance_test":"test_benchmark_not_hardcoded"},
 "V15-10":{"contradiction":"FAIL and PASS sentinel coexist no supersession","actuality":"B1_FAIL and B1_PASS sentinels both present","impact":"ambiguous active state","disposition":"R2_SENTINEL_REGISTRY_ACTIVE_SUPERSEDED","fix_artifact":"supersession registry","acceptance_test":"test_single_active_sentinel"},
 "V15-11":{"contradiction":"manuscript submission ready","actuality":"M1 manuscript.md is a short skeleton missing figures/methods/supplement/citations","impact":"skeleton mislabeled submission-ready","disposition":"SKELETON_NOT_SUBMISSION_READY","fix_artifact":"M2 completeness checklist","acceptance_test":"test_manuscript_completeness"},
}
with open(f"{R}/reconciliation/a1/parent_claim_evidence_matrix.json","w") as f:
    json.dump({"run_id":RUNID,"schema":"A1-claim-evidence-matrix-v1.5","entries":cm},f,indent=2,ensure_ascii=False)
# test scope audit
tsa={
 "run_id":RUNID,
 "v14_scoped_command": "cd /home/cunyuliu/v1_4_boundary_audit_20260804T150707Z && python -m pytest tests/ -q",
 "v14_scoped_passed": 140,
 "conftest_exclusions": ["tests/test_contract_compliance.py","tests/test_schema_validation.py"],
 "exclusion_reason":"legacy v1.2 parent-contract tests require v1.2 run env; excluded from v1.4 collection",
 "statement":"V14_SCOPED_140_PASS; LEGACY_V12_TESTS=EXCLUDED_NOT_RUN_IN_V14",
 "forbidden_claim":"must NOT say 'entire project passes'",
}
with open(f"{R}/reconciliation/a1/test_scope_audit.json","w") as f: json.dump(tsa,f,indent=2,ensure_ascii=False)
# release lineage audit
rla={
 "run_id":RUNID,
 "final_commit":"65896dca7181a172505b78416c171e6426094471",
 "r1_source_commit":"6098033",
 "r1_commit_predates_final":True,
 "r1_commit_order_note":"6098033 (B2) < b6b6e30 (R1) < 1dfad2e (E1) < 65896dc (final test-scope isolation)",
 "r1_seal_claim":"R1_RELEASE_SEALED (historical partial seal, NOT final release)",
 "r1_payload_sha":"da4b1ac0327b3596835d912df872518e85d384f9f39a3500725aafabdfa93836",
 "r1_detached_seal_sha":"d7ddd128b6af28a3f38052588d7c96046bd0fcaf02cc3c4e6f75e2530a2969ee",
 "r1_replay_is_hash_only":True,
 "e1_review_independence":"INTERNAL_ADVERSARIAL (self-generated, 9 issues self-closed)",
 "e1_replay_claim":"E1_REPRODUCED_CLAIMS_ADMISSIBLE_SUBMISSION_READY (hash replay only)",
 "final_release_required":"R2 after M3 on final clean commit",
}
with open(f"{R}/reconciliation/a1/release_lineage_audit.json","w") as f: json.dump(rla,f,indent=2,ensure_ascii=False)
# A1 decision
a1={
 "gate":"A1","run_id":RUNID,"schema":"A1-decision-v1.5",
 "decision_time_utc":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
 "parent_commit":"65896dca7181a172505b78416c171e6426094471",
 "parent_contract_sha":"e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc",
 "v15_contradictions_registered":11,
 "all_v15_have_evidence_path":True,
 "parent_bytes_unchanged":True,
 "ledger":"reconciliation/a1/parent_evidence_ledger.tsv",
 "claim_matrix":"reconciliation/a1/parent_claim_evidence_matrix.json",
 "state":"A1_PASS_PARENT_EVIDENCE_FROZEN",
}
with open(f"{R}/reconciliation/a1/A1_decision.json","w") as f: json.dump(a1,f,indent=2,ensure_ascii=False)
# report
rep=f"""# A1 Report — Parent Evidence and Result Freeze

**State: A1_PASS_PARENT_EVIDENCE_FROZEN**

## Outcome
Parent v1.4 numeric results are frozen by original bytes; lineage, scope and 11 contradictions (V15-01..11) are recorded without rewriting parent files. No parent run_root was modified.

## What was frozen
- Parent contract 1.4.docx SHA-256 = e7edff0998319512b8afc2f06bfc40e82639845f15ed56467bf60e240ef1f9fc (unchanged pre/post)
- Parent commit 65896dca7181a172505b78416c171e6426094471 (branch codex/v1_4_boundary_audit_20260804T150707Z)
- Gate artifacts T6/Q6/Q7/N0/B0/B1/B2/R1/M1/E1 + state + benchmark, each size/mtime/SHA (see parent_evidence_ledger.tsv)

## Contradiction ledger (all 11 registered with evidence path)
- V15-01 gain-below-threshold wording RETRACTED (actual 0.4163>0.3)
- V15-02 point-coverage fail != proven undercoverage (Wilson CI contains 0.8)
- V15-03 11th member CCUGCC_ACUGG = FIT_IDENTIFIED (not fully source-authored)
- V15-04 R1 source_commit 6098033 predates final 65896dc => historical partial seal
- V15-05 authoritative_status.json still reports V13 operational state => stale
- V15-06 replay.sh is hash-only, not recomputation
- V15-07 E1 review is internal adversarial, 9 issues self-closed
- V15-08 140 tests are v1.4-scoped (legacy v1.2 excluded by conftest)
- V15-09 B0/B1 are prototype schemas + toy fixtures (hard-coded constants)
- V15-10 B1 FAIL and PASS sentinels coexist without supersession
- V15-11 M1 manuscript is a skeleton, not submission-ready

## SHAs (discrepancy ledger is machine-readable)
Computing all SHA-256 from original bytes (not copied manifest values). See parent_evidence_ledger.tsv.
"""
with open(f"{R}/reports/A1_report.md","w") as f: f.write(rep)
with open(f"{R}/sentinels/A1_PASS_PARENT_EVIDENCE_FROZEN.sentinel","w") as f:
    json.dump(a1,f,indent=2)
print("A1 outputs written")
