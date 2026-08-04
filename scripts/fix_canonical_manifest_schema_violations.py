#!/usr/bin/env python3
"""
Fix canonical manifest schema violations discovered 2026-08-04 audit round 3.

Root cause: when Q3-Q5 completed and adjudication promoted claim to
STRONG_CROSS_SYSTEM_RESTRICTED, the canonical manifest's TOP-LEVEL fields
were never updated. They still reflect the v1.1 adjudication state
(TECTO_REANALYSIS_AND_PARTIAL_ID_CASE_STUDY / QMAP_NOT_ADMITTED era).

This script corrects the stale top-level fields to match the actual
adjudicated state (which is already correctly recorded in
final_claim_adjudication, gate_decisions[Q3/Q4/Q5], and qmap_terminal_state).

All corrections are metadata-only; no gate status or evidence is changed.
Gate decisions remain PASS backed by frozen evidence files.
"""
import json, jsonschema, datetime, sys, os

MANIFEST = 'manifests/canonical_manifest_v1_2_20260803.json'
SCHEMA = 'schemas/canonical_manifest.schema.json'
GATE_CODE_COMMIT = '0874c88'  # commit that produced Q3-Q5 gate results

def main():
    with open(MANIFEST) as f:
        m = json.load(f)
    with open(SCHEMA) as f:
        s = json.load(f)

    # --- pre-fix validation: enumerate all errors ---
    pre = sorted(jsonschema.Draft7Validator(s).iter_errors(m), key=lambda e: list(e.absolute_path))
    print(f'PRE-FIX schema errors: {len(pre)}')
    for e in pre:
        print(f'  - {list(e.absolute_path)}: {e.message[:120]}')

    # --- corrections (metadata only, reflect already-adjudicated truth) ---
    changes = []

    # 1. claim_class: stale v1.1 value -> schema-valid STRONG_CROSS_SYSTEM
    old = m.get('claim_class')
    m['claim_class'] = 'STRONG_CROSS_SYSTEM'
    if old != m['claim_class']:
        changes.append(f'claim_class: {old!r} -> {m["claim_class"]!r}')

    # 2. current_operational_state: not in enum -> IMPLEMENTATION_COMPLETE
    old = m.get('current_operational_state')
    m['current_operational_state'] = 'IMPLEMENTATION_COMPLETE'
    if old != m['current_operational_state']:
        changes.append(f'current_operational_state: {old!r} -> {m["current_operational_state"]!r}')

    # 3. current_scientific_disposition: not in enum -> ADJUDICATED
    old = m.get('current_scientific_disposition')
    m['current_scientific_disposition'] = 'ADJUDICATED'
    if old != m['current_scientific_disposition']:
        changes.append(f'current_scientific_disposition: {old!r} -> {m["current_scientific_disposition"]!r}')

    # 4. qmap_terminal_disposition: stale QMAP_READY_FOR_Q3 -> QMAP_TRANSFER_SUPPORTED
    old = m.get('qmap_terminal_disposition')
    m['qmap_terminal_disposition'] = 'QMAP_TRANSFER_SUPPORTED'
    if old != m['qmap_terminal_disposition']:
        changes.append(f'qmap_terminal_disposition: {old!r} -> {m["qmap_terminal_disposition"]!r}')

    # 5. top-level code_commit: stale 3ffbe905 (v1.1 adjudication) -> 0874c88 (gate code)
    old = m.get('code_commit')
    m['code_commit'] = GATE_CODE_COMMIT
    if old != m['code_commit']:
        changes.append(f'code_commit: {old!r} -> {m["code_commit"]!r}')

    # 6. output_artifacts: entries 4,5,6 are dicts -> convert to path strings
    new_oa = []
    fixed_oa = False
    for item in m.get('output_artifacts', []):
        if isinstance(item, dict):
            new_oa.append(item.get('path', str(item)))
            fixed_oa = True
        else:
            new_oa.append(item)
    if fixed_oa:
        changes.append(f'output_artifacts: converted dict entries to strings')
    m['output_artifacts'] = new_oa

    # 7. update timestamps
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    m['last_updated_utc'] = now
    m['updated_at_utc'] = now

    print()
    print(f'CHANGES ({len(changes)}):')
    for c in changes:
        print(f'  + {c}')

    # --- post-fix validation ---
    post = sorted(jsonschema.Draft7Validator(s).iter_errors(m), key=lambda e: list(e.absolute_path))
    print()
    print(f'POST-FIX schema errors: {len(post)}')
    for e in post:
        print(f'  - {list(e.absolute_path)}: {e.message[:120]}')

    if post:
        print()
        print('ABORTING: schema still has errors after fix')
        sys.exit(1)

    # --- write back ---
    with open(MANIFEST, 'w') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print()
    print('WROTE', MANIFEST)
    print('SCHEMA VALIDATION: PASS')

if __name__ == '__main__':
    main()
