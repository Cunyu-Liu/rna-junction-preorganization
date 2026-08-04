#!/usr/bin/env python3
"""Schema validation test for canonical manifest.

This is the minimal test that would have caught the 7 schema violations
found in audit round 3. Run with: python -m pytest tests/test_schema_validation.py
"""
import json
import jsonschema
from pathlib import Path

WT = Path(__file__).resolve().parent.parent
MANIFEST = WT / 'manifests/canonical_manifest_v1_2_20260803.json'
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
    assert m['qmap_terminal_disposition'] == m['qmap_terminal_state'],         f'qmap_terminal_disposition={m["qmap_terminal_disposition"]} != qmap_terminal_state={m["qmap_terminal_state"]}'
    top_cc = m['code_commit']
    for g in ['Q3', 'Q4', 'Q5']:
        gate_cc = m['gate_decisions'][g]['evidence']['code_commit']
        assert gate_cc == top_cc, f'code_commit mismatch: top={top_cc[:7]} {g}={gate_cc[:7]}'
    assert all(v == 'PASS' for v in m['gate_statuses'].values()), 'not all gates PASS'


if __name__ == '__main__':
    test_manifest_passes_schema_validation()
    test_all_enum_fields_valid()
    test_output_artifacts_are_strings()
    test_cross_field_consistency()
    print('All schema tests PASS')
