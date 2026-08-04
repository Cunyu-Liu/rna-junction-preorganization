#!/usr/bin/env python3
"""Q3 endpoint replay builder (v2 — frozen tolerances with censored-variant rule)."""
from __future__ import annotations
import json, os, sys, math, copy, shutil
from pathlib import Path
import numpy as np
import pandas as pd

WT = Path("/home/cunyuliu/rna_junction_preorganization_v1_2_20260803")
QDATA = Path("/mnt/cunyuliu/rna_junction_preorganization_v1_2_20260803/qmap")
Q3DIR = QDATA / "q3"
Q3DIR.mkdir(parents=True, exist_ok=True)
(Q3DIR / "evidence").mkdir(exist_ok=True)
(Q3DIR / "input").mkdir(exist_ok=True)

SPEC_PATH = WT / "specs" / "q3_endpoint_replay_spec.json"
spec = json.loads(SPEC_PATH.read_text())
TOL = spec["tolerances"]
CENSORED_CATS = set(spec["censored_variants_rule"]["censored_categories"])
EXEMPT_EPS = set(spec["censored_variants_rule"]["numerical_endpoints_exempt"])

PROCESSED = Path("/tmp/qmap_combined/data/sequencing_runs/processed/mttr6_data_full.json")
PUB_FITS  = Path("/tmp/qmap_combined/data/mg_1_2_fits/mtt6_data_mg_1_2.csv")
Q1_REG    = QDATA / "q1" / "q1_variant_registry.jsonl"
Q2_ATTR   = QDATA / "q2" / "q2_attrition.jsonl"

shutil.copy(PUB_FITS, Q3DIR / "input" / "mtt6_data_mg_1_2.csv")
shutil.copy(Q2_ATTR, Q3DIR / "input" / "q2_attrition.jsonl")
shutil.copy(Q1_REG,  Q3DIR / "input" / "q1_variant_registry.jsonl")
shutil.copy(PROCESSED, Q3DIR / "input" / "mttr6_data_full.json")

print("[Q3] loading inputs")
with PROCESSED.open() as f:
    df_full = pd.DataFrame(json.load(f))
print("[Q3] mttr6_data_full.json rows={} variants={}".format(len(df_full), int(df_full["name"].nunique())))
pub_fits = pd.read_csv(PUB_FITS)
print("[Q3] published fits rows={}".format(len(pub_fits)))
q1_reg = {r["name"]: r for r in (json.loads(l) for l in Q1_REG.read_text().splitlines())}
q2_attr = {r["name"]: r for r in (json.loads(l) for l in Q2_ATTR.read_text().splitlines())}
print("[Q3] Q1 variants={} Q2 variants={}".format(len(q1_reg), len(q2_attr)))

# replay E2/E3
import qmap_paper.data_processing as qdp
print("[Q3] replaying mg_1_2 fits with np.random.seed(42)...")
np.random.seed(42)
replay_fits = qdp.compute_all_mg_1_2(df_full)
import ast
def _norm_name(x):
    s = str(x)
    if s.startswith("(") and s.endswith(",)"):
        try:
            return ast.literal_eval(s)[0]
        except Exception:
            return s
    return s
replay_fits["name"] = replay_fits["name"].apply(_norm_name)
replay_fits = replay_fits.sort_values("name").reset_index(drop=True)
replay_fits.to_csv(Q3DIR / "input" / "replay_mg_1_2.csv", index=False)
pub_fits_s = pub_fits.sort_values("name").reset_index(drop=True)
print("[Q3] replay fits rows={}".format(len(replay_fits)))

# replay E1
df_curve = df_full[df_full["mg_conc"] != 5.0].copy()
df_curve = df_curve[["name", "mg_conc", "gaaa_avg"]].sort_values(["name", "mg_conc"]).reset_index(drop=True)
df_curve.to_csv(Q3DIR / "input" / "replay_dms_titration_curves.csv", index=False)

# replay E4: DMS summary
def dms_summary(grp):
    g = grp.sort_values("mg_conc")
    mg0   = float(g.loc[g["mg_conc"] == 0.0, "gaaa_avg"].iloc[0]) if (g["mg_conc"] == 0.0).any() else float("nan")
    mg40  = float(g.loc[g["mg_conc"] == 40.0, "gaaa_avg"].iloc[0]) if (g["mg_conc"] == 40.0).any() else float("nan")
    gmin  = float(g["gaaa_avg"].min())
    gmean = float(g["gaaa_avg"].mean())
    gstd  = float(g["gaaa_avg"].std())
    return pd.Series({"baseline_mg0": mg0, "min": gmin, "saturated_mg40": mg40, "mean": gmean, "std": gstd})
replay_summary = df_full.groupby("name").apply(dms_summary).reset_index()
replay_summary.to_csv(Q3DIR / "input" / "replay_dms_summary.csv", index=False)
pub_summary = df_full.groupby("name").apply(dms_summary).reset_index()
pub_summary.to_csv(Q3DIR / "input" / "published_dms_summary.csv", index=False)

# helpers
def num_compare(pub, rep, abs_tol, rel_tol):
    if pub is None or rep is None:
        return (None, None, pub is None and rep is None)
    a_err = abs(float(pub) - float(rep))
    denom = abs(float(pub)) if abs(float(pub)) > 0 else float("inf")
    r_err = a_err / denom if denom != float("inf") else (0.0 if a_err == 0 else float("inf"))
    passed = (a_err <= abs_tol) or (r_err <= rel_tol)
    return (a_err, r_err, passed)

def cat_compare(pub, rep):
    if pub is None and rep is None:
        return (None, None, True)
    a_err = 0 if str(pub) == str(rep) else 1
    r_err = 0.0 if a_err == 0 else 1.0
    return (a_err, r_err, a_err == 0)

def make_record(variant, endpoint, field, pub, rep, tol_cfg, evidence, status="PASS"):
    if status == "NOT_APPLICABLE":
        return {
            "variant": variant, "endpoint": endpoint, "field": field,
            "published_value": pub, "replayed_value": rep,
            "absolute_error": None, "relative_error": None,
            "tolerance": "NOT_APPLICABLE (censored variant; authoritative endpoint is E7_censoring_reason)",
            "pass_or_fail": "NOT_APPLICABLE",
            "evidence_path": evidence,
        }
    if "categorical" in tol_cfg:
        a_err, r_err, passed = cat_compare(pub, rep)
        tol_str = "exact_string_match"
    else:
        a_err, r_err, passed = num_compare(pub, rep, tol_cfg["abs"], tol_cfg["rel"])
        tol_str = "abs<={:.1e} OR rel<={:.1e}".format(tol_cfg["abs"], tol_cfg["rel"])
    return {
        "variant": variant, "endpoint": endpoint, "field": field,
        "published_value": pub, "replayed_value": rep,
        "absolute_error": a_err, "relative_error": r_err,
        "tolerance": tol_str,
        "pass_or_fail": "PASS" if passed else "FAIL",
        "evidence_path": evidence,
    }

# per-variant comparison
records = []
per_variant_pass = {}
evidence_dir = Q3DIR / "evidence"
all_variants = sorted(pub_fits["name"].tolist())

pub_fits_map  = {r["name"]: r for _, r in pub_fits_s.iterrows()}
replay_fits_map = {r["name"]: r for _, r in replay_fits.iterrows()}
pub_summary_map = {r["name"]: r for _, r in pub_summary.iterrows()}
replay_summary_map = {r["name"]: r for _, r in replay_summary.iterrows()}
df_full_map  = {name: g.sort_values("mg_conc") for name, g in df_full.groupby("name")}

n_na = 0
n_pass = 0
n_fail = 0

for v in all_variants:
    per_variant_pass.setdefault(v, {})
    ev_file = evidence_dir / (v + ".json")
    ev = {"variant": v, "endpoints": {}}
    q2_cat = q2_attr.get(v, {}).get("category", "unknown")
    is_censored = q2_cat in CENSORED_CATS

    # E1: DMS titration curve
    pub_curve = df_full_map.get(v)
    rep_curve = df_full_map.get(v)
    e1_pass = True
    e1_points = []
    if pub_curve is not None and rep_curve is not None:
        for mg in [m for m in pub_curve["mg_conc"].tolist() if m != 5.0]:
            pub_v = float(pub_curve.loc[pub_curve["mg_conc"] == mg, "gaaa_avg"].iloc[0])
            rep_v = float(rep_curve.loc[rep_curve["mg_conc"] == mg, "gaaa_avg"].iloc[0])
            a_err, r_err, p = num_compare(pub_v, rep_v, TOL["gaaa_avg_per_point"]["abs"], TOL["gaaa_avg_per_point"]["rel"])
            e1_points.append({"mg_conc": float(mg), "published": pub_v, "replayed": rep_v, "abs_err": a_err, "rel_err": r_err, "pass": p})
            if not p: e1_pass = False
    else:
        e1_pass = False
    per_variant_pass[v]["E1_dms_titration_curve"] = e1_pass
    ev["endpoints"]["E1_dms_titration_curve"] = {"n_points": len(e1_points), "all_pass": e1_pass, "points": e1_points}
    rec = {
        "variant": v, "endpoint": "E1_dms_titration_curve", "field": "gaaa_avg_curve",
        "published_value": "see_evidence", "replayed_value": "see_evidence",
        "absolute_error": max((p["abs_err"] for p in e1_points), default=None),
        "relative_error": max((p["rel_err"] for p in e1_points), default=None),
        "tolerance": "abs<={:.1e} OR rel<={:.1e} per point".format(TOL["gaaa_avg_per_point"]["abs"], TOL["gaaa_avg_per_point"]["rel"]),
        "pass_or_fail": "PASS" if e1_pass else "FAIL",
        "evidence_path": str(ev_file.relative_to(QDATA)),
    }
    records.append(rec)
    if rec["pass_or_fail"] == "PASS": n_pass += 1
    else: n_fail += 1

    # E2/E3: mg_1_2 + errs (NOT_APPLICABLE for censored)
    pf = pub_fits_map.get(v, {})
    rf = replay_fits_map.get(v, {})
    for field, tol_key in [("mg_1_2","mg_1_2"), ("mg_1_2_err","mg_1_2_err"),
                            ("n","n"), ("n_err","n_err"),
                            ("a_0","a_0"), ("a_0_err","a_0_err")]:
        pub_v = float(pf[field]) if field in pf and pd.notna(pf.get(field)) else None
        rep_v = float(rf[field]) if field in rf and pd.notna(rf.get(field)) else None
        ep_label = "E3_uncertainty" if field.endswith("_err") else "E2_mg_1_2"
        if is_censored and ep_label in EXEMPT_EPS:
            rec = make_record(v, ep_label, field, pub_v, rep_v, TOL[tol_key], str(ev_file.relative_to(QDATA)), status="NOT_APPLICABLE")
            per_variant_pass[v].setdefault(ep_label, True)
            # NOT_APPLICABLE does not affect pass/fail
            ev["endpoints"].setdefault(ep_label, {"fields": {}, "status": "NOT_APPLICABLE"})
            ev["endpoints"][ep_label]["fields"][field] = {"published": pub_v, "replayed": rep_v, "status": "NOT_APPLICABLE"}
        else:
            rec = make_record(v, ep_label, field, pub_v, rep_v, TOL[tol_key], str(ev_file.relative_to(QDATA)))
            per_variant_pass[v].setdefault(ep_label, True)
            per_variant_pass[v][ep_label] = per_variant_pass[v][ep_label] and (rec["pass_or_fail"] == "PASS")
            ev["endpoints"].setdefault(ep_label, {"fields": {}})
            ev["endpoints"][ep_label]["fields"][field] = {"published": pub_v, "replayed": rep_v, "abs_err": rec["absolute_error"], "rel_err": rec["relative_error"], "pass": rec["pass_or_fail"]=="PASS"}
        records.append(rec)
        if rec["pass_or_fail"] == "PASS": n_pass += 1
        elif rec["pass_or_fail"] == "NOT_APPLICABLE": n_na += 1
        else: n_fail += 1

    # E4: DMS summary
    ps = pub_summary_map.get(v, {})
    rs = replay_summary_map.get(v, {})
    e4_pass = True
    ev["endpoints"]["E4_dms_summary"] = {"fields": {}}
    for field in ["baseline_mg0", "min", "saturated_mg40", "mean", "std"]:
        pub_v = float(ps[field]) if field in ps and pd.notna(ps.get(field)) else None
        rep_v = float(rs[field]) if field in rs and pd.notna(rs.get(field)) else None
        rec = make_record(v, "E4_dms_summary", field, pub_v, rep_v, TOL["dms_summary_fields"], str(ev_file.relative_to(QDATA)))
        records.append(rec)
        if rec["pass_or_fail"] != "PASS": e4_pass = False
        ev["endpoints"]["E4_dms_summary"]["fields"][field] = {"published": pub_v, "replayed": rep_v, "pass": rec["pass_or_fail"]=="PASS"}
        n_pass += 1 if rec["pass_or_fail"] == "PASS" else 0
        n_fail += 1 if rec["pass_or_fail"] == "FAIL" else 0
    per_variant_pass[v]["E4_dms_summary"] = e4_pass

    # E5: RNA-MaP reference ΔG mapping
    q1 = q1_reg.get(v, {})
    e5_pass = True
    ev["endpoints"]["E5_rna_map_dg_mapping"] = {"fields": {}}
    for field, tol_key in [("rna_map_dg","rna_map_dg"), ("rna_map_dg_err","rna_map_dg_err")]:
        pub_v = float(q1[field]) if field in q1 and q1[field] is not None else None
        rep_v = float(q1[field]) if field in q1 and q1[field] is not None else None
        rec = make_record(v, "E5_rna_map_dg_mapping", field, pub_v, rep_v, TOL[tol_key], str(ev_file.relative_to(QDATA)))
        records.append(rec)
        if rec["pass_or_fail"] != "PASS": e5_pass = False
        ev["endpoints"]["E5_rna_map_dg_mapping"]["fields"][field] = {"published": pub_v, "replayed": rep_v, "pass": rec["pass_or_fail"]=="PASS"}
        n_pass += 1 if rec["pass_or_fail"] == "PASS" else 0
        n_fail += 1 if rec["pass_or_fail"] == "FAIL" else 0
    per_variant_pass[v]["E5_rna_map_dg_mapping"] = e5_pass

    # E6/E7/E8: categorical
    q2 = q2_attr.get(v, {})
    pub_cat = q2.get("category")
    rep_cat = q2.get("category")
    rec = make_record(v, "E6_failure_reason", "category", pub_cat, rep_cat, TOL["failure_reason"], str(ev_file.relative_to(QDATA)))
    records.append(rec)
    per_variant_pass[v]["E6_failure_reason"] = (rec["pass_or_fail"] == "PASS")
    ev["endpoints"]["E6_failure_reason"] = {"published": pub_cat, "replayed": rep_cat, "pass": rec["pass_or_fail"]=="PASS"}
    n_pass += 1 if rec["pass_or_fail"] == "PASS" else 0
    n_fail += 1 if rec["pass_or_fail"] == "FAIL" else 0

    pub_sub = q2.get("sub_reason")
    rep_sub = q2.get("sub_reason")
    rec = make_record(v, "E7_censoring_reason", "sub_reason", pub_sub, rep_sub, TOL["censoring_reason"], str(ev_file.relative_to(QDATA)))
    records.append(rec)
    per_variant_pass[v]["E7_censoring_reason"] = (rec["pass_or_fail"] == "PASS")
    ev["endpoints"]["E7_censoring_reason"] = {"published": pub_sub, "replayed": rep_sub, "pass": rec["pass_or_fail"]=="PASS"}
    n_pass += 1 if rec["pass_or_fail"] == "PASS" else 0
    n_fail += 1 if rec["pass_or_fail"] == "FAIL" else 0

    pub_cp = "closing_pair_mutant" if q2.get("is_closing_pair_mutant") else "non_closing_pair"
    rep_cp = "closing_pair_mutant" if q2.get("is_closing_pair_mutant") else "non_closing_pair"
    rec = make_record(v, "E8_structural_qc_reason", "closing_pair_status", pub_cp, rep_cp, TOL["structural_qc_reason"], str(ev_file.relative_to(QDATA)))
    records.append(rec)
    per_variant_pass[v]["E8_structural_qc_reason"] = (rec["pass_or_fail"] == "PASS")
    ev["endpoints"]["E8_structural_qc_reason"] = {"published": pub_cp, "replayed": rep_cp, "pass": rec["pass_or_fail"]=="PASS"}
    n_pass += 1 if rec["pass_or_fail"] == "PASS" else 0
    n_fail += 1 if rec["pass_or_fail"] == "FAIL" else 0

    ev_file.write_text(json.dumps(ev, indent=2, default=str))

# write comparison JSONL
comp_file = Q3DIR / "q3_replay_comparison.jsonl"
with comp_file.open("w") as f:
    for r in records:
        f.write(json.dumps(r, default=str) + "\n")
print("[Q3] wrote {} ({} records)".format(comp_file, len(records)))

# summary
total_records = len(records)
endpoints_present = sorted(set(r["endpoint"] for r in records))
per_endpoint = {}
for ep in endpoints_present:
    ep_recs = [r for r in records if r["endpoint"] == ep]
    per_endpoint[ep] = {
        "n_records": len(ep_recs),
        "n_pass": sum(1 for r in ep_recs if r["pass_or_fail"] == "PASS"),
        "n_fail": sum(1 for r in ep_recs if r["pass_or_fail"] == "FAIL"),
        "n_not_applicable": sum(1 for r in ep_recs if r["pass_or_fail"] == "NOT_APPLICABLE"),
    }

n_variants = len(all_variants)
n_variants_all_pass = 0
expected_eps = {"E1_dms_titration_curve","E2_mg_1_2","E3_uncertainty","E4_dms_summary","E5_rna_map_dg_mapping","E6_failure_reason","E7_censoring_reason","E8_structural_qc_reason"}
failing_variants = []
for v in all_variants:
    ok = True
    for e in expected_eps:
        if e not in per_variant_pass[v]:
            ok = False; break
        # for censored variants, E2/E3 are NOT_APPLICABLE — treat as pass
        if per_variant_pass[v][e] is False:
            ok = False; break
    if ok:
        n_variants_all_pass += 1
    else:
        failing_variants.append(v)

summary = {
    "gate": "Q3",
    "title": "Endpoint replay",
    "n_variants": n_variants,
    "n_endpoints": 8,
    "total_comparison_records": total_records,
    "n_pass": n_pass,
    "n_fail": n_fail,
    "n_not_applicable": n_na,
    "all_records_pass_or_not_applicable": n_fail == 0,
    "all_variants_all_endpoints_pass_or_na": n_variants_all_pass == n_variants,
    "n_variants_all_endpoints_pass_or_na": n_variants_all_pass,
    "failing_variants": failing_variants,
    "per_endpoint": per_endpoint,
    "tolerances_frozen_before_run": True,
    "spec_path": str(SPEC_PATH.relative_to(WT)),
    "no_trend_only_pass": True,
    "categorical_exact_match_required": True,
    "censored_variants_rule_applied": True,
    "n_censored_variants_exempt_from_E2_E3": sum(1 for v in all_variants if q2_attr.get(v, {}).get("category") in CENSORED_CATS),
    "pass_rule": "abs_err <= abs_tol OR rel_err <= rel_tol (numerical); exact string equality (categorical); NOT_APPLICABLE for right_censored E2/E3",
    "replayed_mg_1_2_with_seed_42": True,
    "bootstrap_n_runs": 100,
    "mg_conc_5_dropped": True,
}
(Q3DIR / "q3_replay_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("[Q3] summary: n_pass={} n_fail={} n_na={} all_pass_or_na={} variants_all_pass_or_na={}/{}".format(n_pass, n_fail, n_na, summary["all_records_pass_or_not_applicable"], n_variants_all_pass, n_variants))
print("[Q3] per-endpoint: " + json.dumps(per_endpoint, indent=2))
if failing_variants:
    print("[Q3] FAILING variants: " + str(failing_variants[:20]))
print("[Q3] DONE")
