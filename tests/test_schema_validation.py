#!/usr/bin/env python3
"""Schema validation test for canonical manifest.

This is the minimal test that would have caught the 7 schema violations
found in audit round 3. Run with: python -m pytest tests/test_schema_validation.py
"""
import json
import os
import jsonschema
from pathlib import Path

WT = Path(__file__).resolve().parent.parent
MANIFEST = Path(os.environ.get('RNA_V12_MANIFEST_PATH', str(WT / 'manifests' / 'canonical_manifest_v1_2_unbound.json')))
SCHEMA = WT / 'schemas/canonical_manifest.schema.json'


def test_manifest_passes_schema_validation():
    """Canonical manifest must pass its own schema with 0 errors."""
    m = json.load(open(MANIFEST))
    s = json.load(open(SCHEMA))
    errors = list(jsonschema.Draft7Validator(s).iter_errors(m))
    assert errors == [], f'{len(errors)} schema violations:\n' + '\n'.join(
        f'  - {list(e.absolute_path)}: {e.message[:120]}' for e in errors
    )


def test_all_enum_fields_valid():
    """All enum-typed fields must have values in their allowed enum."""
    m = json.load(open(MANIFEST))
    s = json.load(open(SCHEMA))
    for field, prop in s.get('properties', {}).items():
        if 'enum' in prop and field in m:
            assert m[field] in prop['enum'],                 f'{field}={m[field]!r} not in enum {prop["enum"]}'


def test_output_artifacts_are_strings():
    """output_artifacts must be array of strings per schema."""
    m = json.load(open(MANIFEST))
    oa = m.get('output_artifacts', [])
    assert all(isinstance(x, str) for x in oa),         f'output_artifacts contains non-string: {[type(x).__name__ for x in oa]}'


def test_cross_field_consistency():
    """Key cross-field invariants that must hold."""
    m = json.load(open(MANIFEST))
    assert m['qmap_terminal_disposition'] in {'NOT_STARTED', 'NOT_ADJUDICATED', 'QMAP_TRANSFER_SUPPORTED', 'QMAP_TRANSFER_NOT_SUPPORTED', 'QMAP_INCONCLUSIVE', 'QMAP_NOT_ADMITTED'}
    assert m['scientific_unlock'] == 'NO_UNLOCK' or m['current_scientific_disposition'] == 'ADJUDICATED'
    top_cc = m['code_commit']
    for g in ['Q3', 'Q4', 'Q5']:
        gate_cc = m['gate_decisions'][g]['evidence']['code_commit']
        assert gate_cc == top_cc, f'code_commit mismatch: top={top_cc[:7]} {g}={gate_cc[:7]}'
    assert all(v in {'NOT_STARTED', 'RUNNING', 'PASS', 'FAIL', 'BLOCKED', 'CLOSED', 'NOT_APPLICABLE', 'STALE_NOT_AUTHORITATIVE'} for v in m['gate_statuses'].values())


if __name__ == '__main__':
    test_manifest_passes_schema_validation()
    test_all_enum_fields_valid()
    test_output_artifacts_are_strings()
    test_cross_field_consistency()
    print('All schema tests PASS')
