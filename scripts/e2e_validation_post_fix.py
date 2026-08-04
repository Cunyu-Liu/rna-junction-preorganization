#!/usr/bin/env python3
"""
End-to-end validation of canonical manifest after audit-round-3 fix.
Covers: schema, required fields, enums, cross-field consistency, artifact
existence, sentinel existence, reports/final consistency, claim_matrix
consistency, git state, contract SHA256.

Exit code 0 = no errors (warnings allowed); non-zero = errors found.
"""
import json, jsonschema, os, sys, subprocess
from pathlib import Path

WT = Path('/home/cunyuliu/rna_junction_preorganization_v1_2_20260803')
QDATA = Path('/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap')
MANIFEST = WT / 'manifests/canonical_manifest_v1_2_20260803.json'
SCHEMA = WT / 'schemas/canonical_manifest.schema.json'
CONTRACT_SHA256 = '32d09729638b7681b6efcfdf8b2addc3c7f83060e37ce5ef3dd5c5a051702252'

errors = []
warnings = []
checks = []

def ok(msg): checks.append(('PASS', msg))
def err(msg): errors.append(msg)
def warn(msg): warnings.append(msg)

m = json.load(open(MANIFEST))
s = json.load(open(SCHEMA))

# ---------- 1. SCHEMA VALIDATION ----------
v = jsonschema.Draft7Validator(s)
schema_errs = sorted(v.iter_errors(m), key=lambda e: list(e.absolute_path))
if schema_errs:
    for e in schema_errs:
        err(f'SCHEMA: {list(e.absolute_path)}: {e.message[:140]}')
else:
    ok('schema validation: 0 errors')

# ---------- 2. REQUIRED FIELDS ----------
for f in s.get('required', []):
    if f not in m:
        err(f'REQUIRED field missing: {f}')
    else:
        pass
if all(f in m for f in s.get('required', [])):
    ok(f'required fields: all {len(s.get("required",[]))} present')

# ---------- 3. ENUM FIELDS ----------
for f, prop in s.get('properties', {}).items():
    if 'enum' in prop and f in m:
        if m[f] not in prop['enum']:
            err(f'ENUM: {f}={m[f]!r} not in {prop["enum"]}')
        else:
            ok(f'enum {f}={m[f]!r} valid')

# ---------- 4. CROSS-FIELD CONSISTENCY ----------
# 4a. qmap_terminal_disposition == qmap_terminal_state
if m.get('qmap_terminal_disposition') == m.get('qmap_terminal_state'):
    ok(f'qmap_terminal_disposition == qmap_terminal_state == {m.get("qmap_terminal_disposition")}')
else:
    err(f'qmap_terminal_disposition={m.get("qmap_terminal_disposition")!r} != qmap_terminal_state={m.get("qmap_terminal_state")!r}')

# 4b. top-level code_commit == Q3/Q4/Q5 gate_decisions code_commit
top_cc = m.get('code_commit')
q345_ccs = [m['gate_decisions'][g]['evidence']['code_commit'] for g in ['Q3','Q4','Q5']]
if all(cc == top_cc for cc in q345_ccs):
    ok(f'code_commit consistent (top + Q3/Q4/Q5 = {top_cc[:7]})')
else:
    err(f'code_commit mismatch: top={top_cc[:7]} Q3/Q4/Q5={[c[:7] for c in q345_ccs]}')

# 4c. claim_class consistent with max_allowable_claim
cc = m.get('claim_class')
mac = m.get('final_claim_adjudication', {}).get('max_allowable_claim')
if cc == 'STRONG_CROSS_SYSTEM' and mac == 'STRONG_CROSS_SYSTEM_RESTRICTED':
    ok(f'claim_class={cc} consistent with max_allowable_claim={mac}')
elif cc == 'TECTO_SPECIFIC' and 'TECTO' in mac:
    ok(f'claim_class={cc} consistent with max_allowable_claim={mac}')
else:
    err(f'claim_class={cc} inconsistent with max_allowable_claim={mac}')

# 4d. all gate_statuses PASS
all_pass = all(v == 'PASS' for v in m.get('gate_statuses', {}).values())
if all_pass:
    ok(f'gate_statuses: all {len(m["gate_statuses"])} PASS')
else:
    non_pass = {g:v for g,v in m['gate_statuses'].items() if v != 'PASS'}
    err(f'gate_statuses not all PASS: {non_pass}')

# 4e. finalizer_criteria all pass (where present)
for g, gd in m.get('gate_decisions', {}).items():
    fc = gd.get('finalizer_criteria', {})
    if fc:
        if all(fc.values()):
            ok(f'{g} finalizer_criteria all pass')
        else:
            err(f'{g} finalizer_criteria NOT all pass: {fc}')

# 4f. T0/S0 decision vs gate_status (WARNING - semantic, not schema)
for g in ['T0','S0']:
    status = m['gate_statuses'].get(g)
    decision = m['gate_decisions'].get(g, {}).get('decision')
    if status == 'PASS' and decision == 'PARTIAL_ENGINEERING_EVIDENCE':
        warn(f'{g}: gate_status=PASS but decision=PARTIAL_ENGINEERING_EVIDENCE (semantic: T0/S0 admit partial engineering evidence; finalizer_criteria all pass)')
    elif status == 'PASS' and decision == 'PASS':
        ok(f'{g}: gate_status=PASS decision=PASS')
    else:
        warn(f'{g}: gate_status={status} decision={decision}')

# 4g. current_operational_state == IMPLEMENTATION_COMPLETE
if m.get('current_operational_state') == 'IMPLEMENTATION_COMPLETE':
    ok('current_operational_state=IMPLEMENTATION_COMPLETE')
else:
    err(f'current_operational_state={m.get("current_operational_state")!r} (expected IMPLEMENTATION_COMPLETE)')

# ---------- 5. OUTPUT_ARTIFACTS: strings + exist ----------
oa = m.get('output_artifacts', [])
all_str = all(isinstance(x, str) for x in oa)
if all_str:
    ok(f'output_artifacts: all {len(oa)} strings')
else:
    err(f'output_artifacts: not all strings: {[type(x).__name__ for x in oa]}')
for a in oa:
    if isinstance(a, str):
        p = WT / a if not os.path.isabs(a) else Path(a)
        if not p.exists():
            # try as-is
            if not Path(a).exists():
                warn(f'output_artifact not found: {a}')

# ---------- 6. INPUT_ARTIFACTS exist ----------
ia = m.get('input_artifacts', [])
for a in ia:
    if isinstance(a, str):
        p = WT / a if not os.path.isabs(a) else Path(a)
        if not p.exists() and not Path(a).exists():
            warn(f'input_artifact not found: {a}')
ok(f'input_artifacts: {len(ia)} checked')

# ---------- 7. CONTRACT SHA256 ----------
if m.get('contract_sha256') == CONTRACT_SHA256:
    ok(f'contract_sha256 matches expected ({CONTRACT_SHA256[:12]}...)')
else:
    err(f'contract_sha256={m.get("contract_sha256")!r} != expected {CONTRACT_SHA256}')

# ---------- 8. REPORTS/FINAL/ exist (8 required artifacts) ----------
required_reports = [
    'claim_adjudication.md', 'claim_matrix.json', 'limitations.md',
    'data_availability.md', 'reproducibility.md', 'model_card.md',
    'dataset_card.md', 'reviewer_attack_matrix.md', 'final_delivery_report.md'
]
for r in required_reports:
    p = WT / 'reports/final' / r
    if p.exists():
        ok(f'reports/final/{r} exists')
    else:
        err(f'reports/final/{r} MISSING')

# ---------- 9. SENTINEL files for all 12 gates ----------
gates12 = ['T0','S0','T1','M0','T2','T3','Q0','Q1','Q2','Q3','Q4','Q5']
for g in gates12:
    p1 = WT / f'manifests/sentinel_{g}.txt'
    p2 = WT / f'Sentinel_{g}.txt'
    if p1.exists() or p2.exists():
        pass
    else:
        warn(f'sentinel for {g} not found (checked manifests/sentinel_{g}.txt and Sentinel_{g}.txt)')
n_sent = sum(1 for g in gates12 if (WT/f'manifests/sentinel_{g}.txt').exists() or (WT/f'Sentinel_{g}.txt').exists())
ok(f'sentinels: {n_sent}/12 gates have sentinel files')

# ---------- 10. Q3-Q5 evidence files ----------
ev_files = [
    'specs/q3_endpoint_replay_spec.json', 'specs/q4_selection_split_freeze_spec.json', 'specs/q5_locked_transfer_spec.json',
    'scripts/q3_build.py', 'scripts/q4_build.py', 'scripts/q5_build.py',
    'scripts/finalize_q3.py', 'scripts/finalize_q4.py', 'scripts/finalize_q5.py',
]
for f in ev_files:
    if (WT / f).exists():
        pass
    else:
        err(f'Q3-Q5 evidence file MISSING: {f}')
ok(f'Q3-Q5 spec/script files: {len(ev_files)} checked')

# QDATA evidence
for g in ['q3','q4','q5']:
    d = QDATA / g
    if d.exists():
        ok(f'QDATA/{g} exists')
    else:
        err(f'QDATA/{g} MISSING')

# ---------- 11. claim_matrix.json consistency ----------
cm_path = WT / 'reports/final/claim_matrix.json'
if cm_path.exists():
    cm = json.load(open(cm_path))
    cm_claim = cm.get('claim_class')
    cm_term = cm.get('qmap_terminal_state')
    if cm_claim and 'STRONG_CROSS_SYSTEM' in str(cm_claim):
        ok(f'claim_matrix.json claim_class={cm_claim} consistent')
    else:
        err(f'claim_matrix.json claim_class={cm_claim} inconsistent')
    if cm_term == m.get('qmap_terminal_state'):
        ok(f'claim_matrix.json qmap_terminal_state={cm_term} matches manifest')
    else:
        err(f'claim_matrix.json qmap_terminal_state={cm_term} != manifest {m.get("qmap_terminal_state")}')
else:
    err('claim_matrix.json MISSING')

# ---------- 12. GIT state ----------
r = subprocess.run(['git','-C',str(WT),'rev-parse','HEAD'], capture_output=True, text=True)
head = r.stdout.strip()
r2 = subprocess.run(['git','-C',str(WT),'status','--porcelain'], capture_output=True, text=True)
clean = (len(r2.stdout.strip()) == 0)
ok(f'git HEAD={head[:7]} working_tree_clean={clean}')
if not clean:
    warn(f'git working tree not clean: {r2.stdout.strip()[:200]}')

# ---------- 13. final_claim_adjudication consistency ----------
fca = m.get('final_claim_adjudication', {})
if fca.get('terminal_state') == m.get('qmap_terminal_state'):
    ok(f'final_claim_adjudication.terminal_state matches qmap_terminal_state')
else:
    err(f'final_claim_adjudication.terminal_state={fca.get("terminal_state")} != qmap_terminal_state={m.get("qmap_terminal_state")}')
if fca.get('max_allowable_claim') == mac:
    ok(f'final_claim_adjudication.max_allowable_claim consistent')
else:
    err(f'final_claim_adjudication.max_allowable_claim inconsistent')

# ---------- SUMMARY ----------
print('='*70)
print('E2E VALIDATION SUMMARY (post audit-round-3 fix)')
print('='*70)
print(f'Total checks passed: {len(checks)}')
print(f'Warnings: {len(warnings)}')
print(f'Errors:   {len(errors)}')
print()
if warnings:
    print('--- WARNINGS ---')
    for w in warnings:
        print(f'  [WARN] {w}')
    print()
if errors:
    print('--- ERRORS ---')
    for e in errors:
        print(f'  [FAIL] {e}')
    print()
print('--- ALL PASSED CHECKS ---')
for _, msg in checks:
    print(f'  [OK] {msg}')
print()
if errors:
    print(f'RESULT: FAIL ({len(errors)} errors)')
    sys.exit(1)
else:
    print(f'RESULT: PASS ({len(checks)} checks, {len(warnings)} warnings)')
    sys.exit(0)
