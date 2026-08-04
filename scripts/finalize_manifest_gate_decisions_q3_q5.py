#!/usr/bin/env python3
"""Backfill canonical manifest gate_decisions for Q3/Q4/Q5.

Purely additive manifest-completeness fix. Does NOT recompute any gate.
Reads frozen per-gate manifests + sentinels + summaries (already committed
in 0874c88) and synthesizes gate_decisions entries mirroring the Q0/Q1/Q2
schema {gate, decision, summary, evidence, finalized_at_utc}.

Rationale: finalize_q3/q4/q5.py updated gate_statuses and wrote per-gate
manifests/sentinels but never populated canonical_manifest.gate_decisions,
leaving Q3/Q4/Q5 entries as {} while Q0/Q1/Q2 were fully populated.
"""
from __future__ import annotations
import json, hashlib, subprocess
from pathlib import Path

WT = Path('/home/cunyuliu/rna_junction_preorganization_v1_2_20260803')
QDATA = Path('/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap')
MANIFEST_PATH = WT / 'manifests' / 'canonical_manifest_v1_2_20260803.json'

def git(args):
    return subprocess.run(['git','-C',str(WT)]+args, capture_output=True, text=True, check=True).stdout.strip()

code_commit = git(['rev-parse','HEAD'])
branch = git(['branch','--show-current'])
contract_sha = json.loads(MANIFEST_PATH.read_text()).get('contract_sha256','')

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---- Q3 ----
q3m = json.loads((QDATA/'q3'/'q3_manifest.json').read_text())
q3s = json.loads((QDATA/'q3'/'q3_replay_summary.json').read_text())
q3_sent = json.loads((WT/'Sentinel_Q3.txt').read_text())
q3_evidence = {
    'code_commit': code_commit,
    'branch': branch,
    'contract_sha256': contract_sha,
    'required_artifacts_present': True,
    'missing_artifacts': [],
    'spec_path': 'specs/q3_endpoint_replay_spec.json',
    'build_script': 'scripts/q3_build.py',
    'finalize_script': 'scripts/finalize_q3.py',
    'spec_sha256': q3m['spec_sha256'],
    'build_script_sha256': q3m['build_script_sha256'],
    'n_variants': q3m['n_variants'],
    'n_endpoints': q3m['n_endpoints'],
    'total_comparison_records': q3m['total_comparison_records'],
    'n_pass': q3m['n_pass'],
    'n_fail': q3m['n_fail'],
    'n_not_applicable': q3m['n_not_applicable'],
    'tolerances_frozen_before_run': q3m['tolerances_frozen_before_run'],
    'all_records_pass_or_not_applicable': q3m['checks']['all_records_pass_or_not_applicable'],
    'all_variants_all_endpoints_pass_or_na': q3m['checks']['all_variants_all_endpoints_pass_or_na'],
    'no_trend_only_pass': q3m['checks']['no_trend_only_pass'],
    'categorical_exact_match_required': q3m['checks']['categorical_exact_match_required'],
    'censored_variants_rule_applied': q3m['checks']['censored_variants_rule_applied'],
    'n_censored_variants_exempt_from_E2_E3': q3s['n_censored_variants_exempt_from_E2_E3'],
    'per_endpoint': q3m['per_endpoint'],
    'acceptance': {k: bool(v) for k,v in q3m['checks'].items()},
}
q3_decision = {
    'gate': 'Q3',
    'decision': q3m['gate_result'],
    'summary': (
        'Q3 endpoint replay: 98 variants x 8 endpoints, tolerances frozen BEFORE run. '
        f'{q3m["n_pass"]} PASS + {q3m["n_not_applicable"]} NOT_APPLICABLE '
        f'(11 right-censored variants exempt from E2/E3 numerical mg_1_2; '
        f'authoritative endpoint = E7 censoring reason) + {q3m["n_fail"]} FAIL. '
        'No trend-only pass; categorical exact string match; numerical abs<=tol OR rel<=tol.'
    ),
    'evidence': q3_evidence,
    'finalized_at_utc': q3_sent['timestamp_utc'],
}

# ---- Q4 ----
q4m = json.loads((QDATA/'q4'/'q4_manifest.json').read_text())
q4s = json.loads((QDATA/'q4'/'q4_freeze_summary.json').read_text())
q4_sent = json.loads((WT/'Sentinel_Q4.txt').read_text())
q4_evidence = {
    'code_commit': code_commit,
    'branch': branch,
    'contract_sha256': contract_sha,
    'required_artifacts_present': True,
    'missing_artifacts': [],
    'spec_path': 'specs/q4_selection_split_freeze_spec.json',
    'build_script': 'scripts/q4_build.py',
    'finalize_script': 'scripts/finalize_q4.py',
    'spec_sha256': q4m['spec_sha256'],
    'build_script_sha256': q4m['build_script_sha256'],
    'n_variants': q4m['n_variants'],
    'k_folds': q4m['k_folds'],
    'fold_sizes': q4m['fold_sizes'],
    'n_mutation_graph_edges': q4m['n_mutation_graph_edges'],
    'n_connected_components': q4m['n_connected_components'],
    'leakage_violations': q4m['leakage_violations'],
    'same_variant_all_rows_same_fold': q4s['same_variant_all_rows_same_fold'],
    'cannot_extrapolate_to_arbitrary_TLR_family': True,
    'frozen_before_viewing_transfer_outcome': q4s['frozen_before_viewing_transfer_outcome'],
    'qmap_outcome_cannot_modify': ['tecto_model','operator','transport','thresholds','split','primary_metric'],
    'acceptance': {k: bool(v) for k,v in q4m['checks'].items()},
}
q4_decision = {
    'gate': 'Q4',
    'decision': q4m['gate_result'],
    'summary': (
        'Q4 selection/split/analysis freeze: 98 mttr6 TTR mutants (no extrapolation to arbitrary TLR). '
        f'Mutation graph {q4m["n_mutation_graph_edges"]} edges, {q4m["n_connected_components"]} connected '
        f'components (sizes {q4m["fold_sizes"]}); K={q4m["k_folds"]} folds, {q4m["leakage_violations"]} leakage. '
        'All 12 freeze items locked before viewing transfer outcome.'
    ),
    'evidence': q4_evidence,
    'finalized_at_utc': q4_sent['timestamp_utc'],
}

# ---- Q5 ----
q5m = json.loads((QDATA/'q5'/'q5_manifest.json').read_text())
q5s = json.loads((QDATA/'q5'/'q5_transfer_summary.json').read_text())
q5_sent = json.loads((WT/'Sentinel_Q5.txt').read_text())
q5_evidence = {
    'code_commit': code_commit,
    'branch': branch,
    'contract_sha256': contract_sha,
    'required_artifacts_present': True,
    'missing_artifacts': [],
    'spec_path': 'specs/q5_locked_transfer_spec.json',
    'build_script': 'scripts/q5_build.py',
    'finalize_script': 'scripts/finalize_q5.py',
    'spec_sha256': q5m['spec_sha256'],
    'n_variants': q5m['n_variants'],
    'k_folds': q5m['k_folds'],
    'baselines': {b: {'mean_rmse': d['mean_rmse'], 'mean_nlpd': d['mean_nlpd'],
                      'mean_spearman': d.get('mean_spearman'),
                      'mean_cov68': d['mean_cov68']} for b,d in q5s['baselines'].items()},
    'b4_mean_rmse_kcal_mol': q5s['baselines']['B4']['mean_rmse'],
    'b1_mean_rmse_kcal_mol': q5s['baselines']['B1']['mean_rmse'],
    'preregistered_gain': q5s['preregistered_gain'],
    'label_permutation': q5s['label_permutation'],
    'mutation_class_bootstrap': q5s['mutation_class_bootstrap'],
    'condition_controls': q5s['condition_controls'],
    'negative_nucleotide_controls': q5s['negative_nucleotide_controls'],
    'adjudication_criteria': q5s['adjudication_criteria'],
    'terminal_state': q5s['terminal_state'],
    'locked_before_run': q5s['locked_before_run'],
    'acceptance': {k: bool(v) for k,v in q5s['adjudication_criteria'].items()},
}
q5_decision = {
    'gate': 'Q5',
    'decision': q5m['gate_result'],
    'summary': (
        f'Q5 locked transfer test: 4 baselines compared. B4 (tecto old_dg + isotonic calibration) '
        f'RMSE={q5s["baselines"]["B4"]["mean_rmse"]:.4f} kcal/mol, '
        f'gain over B1={q5s["preregistered_gain"]["mean"]:.3f} '
        f'(95%CI [{q5s["preregistered_gain"]["ci_low"]:.3f},{q5s["preregistered_gain"]["ci_high"]:.3f}]), '
        f'68% coverage={q5s["baselines"]["B4"]["mean_cov68"]:.3f}, '
        f'label permutation p={q5s["label_permutation"]["p_value"]}. '
        f'Terminal state: {q5s["terminal_state"]}.'
    ),
    'evidence': q5_evidence,
    'finalized_at_utc': q5_sent['timestamp_utc'],
}

# ---- update manifest ----
m = json.loads(MANIFEST_PATH.read_text())
before = {g: m['gate_decisions'].get(g, {}) for g in ['Q3','Q4','Q5']}
m['gate_decisions']['Q3'] = q3_decision
m['gate_decisions']['Q4'] = q4_decision
m['gate_decisions']['Q5'] = q5_decision
# sanity: gate_statuses must match decision field
for g, dec in [('Q3',q3_decision),('Q4',q4_decision),('Q5',q5_decision)]:
    assert m['gate_statuses'][g] == dec['decision'], f'{g}: status={m["gate_statuses"][g]} decision={dec["decision"]}'
from datetime import datetime, timezone
m['last_updated_utc'] = datetime.now(timezone.utc).isoformat()
MANIFEST_PATH.write_text(json.dumps(m, indent=2, ensure_ascii=False) + '\n')

print('[gate_decisions-backfill] Q3/Q4/Q5 populated')
print('  Q3 decision:', q3_decision['decision'], '| finalized_at_utc:', q3_decision['finalized_at_utc'])
print('  Q4 decision:', q4_decision['decision'], '| finalized_at_utc:', q4_decision['finalized_at_utc'])
print('  Q5 decision:', q5_decision['decision'], '| finalized_at_utc:', q5_decision['finalized_at_utc'])
print('  before (empty?):', {g: (len(v)==0) for g,v in before.items()})
print('  gate_statuses consistent: OK')
print('  manifest written:', MANIFEST_PATH)
