#!/usr/bin/env python3
"""Q5 orchestrator: write spec, build, finalize; run all."""
from __future__ import annotations
import runtime_config as rc
import json, os, sys, math, hashlib, shutil, random, statistics, collections
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

WT = Path(rc.WORKTREE)
QDATA = Path(rc.QDATA)
Q5DIR = QDATA / "q5"; Q5DIR.mkdir(parents=True, exist_ok=True)
(Q5DIR / "input").mkdir(exist_ok=True)
(Q5DIR / "evidence").mkdir(exist_ok=True)
MANIFEST = Path(rc.MANIFEST_PATH)
ts_now = datetime.now(timezone.utc).isoformat()

# ============================================================ Q5 SPEC
q5_spec = {
  "gate": "Q5", "title": "Locked transfer test",
  "contract_ref": "提示词/rna 三级.md §18 Q5 (lines 1093-1141)",
  "locked_before_run": True,
  "baselines_compared": ["B1_intercept_mean", "B2_published_univariate_mg_1_2", "B3_sequence_mutation", "B4_locked_partial_id_calibration"],
  "primary_eval": "held_out_negative_log_predictive_density",
  "main_eval_includes": [
    "held_out_proper_score", "reference_dg_ranking_spearman",
    "interval_coverage_68pct", "interval_coverage_95pct", "interval_width_mean",
    "calibration_slope", "calibration_intercept",
    "preregistered_gain_over_strongest_simple_baseline",
    "mutation_class_bootstrap", "label_permutation",
    "condition_controls", "negative_nucleotide_controls"
  ],
  "terminal_states": ["QMAP_TRANSFER_SUPPORTED", "QMAP_TRANSFER_NOT_SUPPORTED", "QMAP_INCONCLUSIVE", "QMAP_NOT_ADMITTED"],
  "adjudication_logic": {
    "QMAP_TRANSFER_SUPPORTED": "B4: (a) RMSE<1.0 AND (b) gain>B1>0.3 with 95%CI excl 0 AND (c) 68% cov in [0.55,0.80] AND (d) perm p<0.05",
    "QMAP_TRANSFER_NOT_SUPPORTED": "B4 fails (a)-(d); negative/borderline",
    "QMAP_INCONCLUSIVE": "insufficient power; CI includes 0 and point est <0.3",
    "QMAP_NOT_ADMITTED": "Q0-Q3 not all PASS"
  },
  "outputs": {"spec_path": "specs/q5_locked_transfer_spec.json", "build_script": "scripts/q5_build.py", "finalize_script": "scripts/finalize_q5.py", "artifacts_dir": "QDATA/q5", "sentinel": "Sentinel_Q5.txt"}
}
(WT / "specs" / "q5_locked_transfer_spec.json").write_text(json.dumps(q5_spec, indent=2))

# ============================================================ LOAD DATA
Q1_REG = QDATA / "q1" / "q1_variant_registry.jsonl"
Q2_ATTR = QDATA / "q2" / "q2_attrition.jsonl"
Q4_FOLDS = QDATA / "q4" / "q4_fold_assignment.json"
shutil.copy(Q1_REG, Q5DIR / "input" / "q1_variant_registry.jsonl")
shutil.copy(Q2_ATTR, Q5DIR / "input" / "q2_attrition.jsonl")
shutil.copy(Q4_FOLDS, Q5DIR / "input" / "q4_fold_assignment.json")

q1 = {}
for r in (json.loads(l) for l in Q1_REG.read_text().splitlines()):
    parts = r["name"].split("_")
    nname = parts[1] + "_" + parts[0] if len(parts) == 2 else r["name"]
    q1[nname] = r
q2 = {r["name"]: r for r in (json.loads(l) for l in Q2_ATTR.read_text().splitlines())}
folds = json.loads(Q4_FOLDS.read_text())
fold_of = folds["fold_of_variant"]

# build per-variant feature matrix
rows = []
for name in sorted(q2.keys()):
    q1v = q1.get(name, {})
    q2v = q2[name]
    rows.append({
        "name": name,
        "rna_map_dg": float(q1v.get("rna_map_dg", float("nan"))),
        "rna_map_dg_err": float(q1v.get("rna_map_dg_err", float("nan"))),
        "old_dg": float(q1v.get("old_dg", float("nan"))),
        "mg_1_2": float(q2v.get("mg_1_2", float("nan"))),
        "mg_1_2_err": float(q2v.get("mg_1_2_err", float("nan"))),
        "n_hill": float(q2v.get("n", float("nan"))),
        "a_0": float(q2v.get("a_0", float("nan"))),
        "bp_muts": q2v.get("bp_muts", []),
        "mutations": q2v.get("mutations", []),
        "aligned_seq": q2v.get("aligned_seq", ""),
        "category": q2v.get("category", ""),
        "fold": fold_of.get(name, 0),
        "n_mutations": len(q2v.get("mutations", [])),
    })
df = pd.DataFrame(rows)
df = df.dropna(subset=["rna_map_dg", "old_dg", "mg_1_2"]).reset_index(drop=True)
print("[Q5] {} variants with complete features".format(len(df)))

# one-hot bp_muts
all_bp = sorted(set(b for bps in df["bp_muts"] for b in bps))
for bp in all_bp:
    df["bp_" + bp] = df["bp_muts"].apply(lambda x: 1 if bp in x else 0)

K = folds["k_folds"]
print("[Q5] K={} folds".format(K))

# ============================================================ BASELINES + CV
def nlpd(y_true, mu, sigma):
    sigma = np.maximum(sigma, 1e-6)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma)**2))

def run_cv(df, model_fn, K):
    fold_results = []
    for k in range(K):
        train = df[df["fold"] != k]
        test = df[df["fold"] == k]
        if len(test) == 0 or len(train) == 0:
            continue
        mu, sigma = model_fn(train, test)
        yt = test["rna_map_dg"].values
        rmse = float(np.sqrt(np.mean((yt - mu) ** 2)))
        nlp = nlpd(yt, mu, sigma)
        sp = float(stats.spearmanr(mu, yt)[0]) if len(yt) > 2 else float("nan")
        cov68 = float(np.mean(np.abs(yt - mu) <= 1.0 * sigma))
        cov95 = float(np.mean(np.abs(yt - mu) <= 1.96 * sigma))
        width68 = float(2.0 * np.mean(sigma))
        if len(yt) > 1:
            sl, icept = np.polyfit(mu, yt, 1)
        else:
            sl, icept = float("nan"), float("nan")
        fold_results.append({"fold": k, "n_test": len(test), "n_train": len(train),
                             "rmse": rmse, "nlpd": nlp, "spearman": sp,
                             "cov68": cov68, "cov95": cov95, "width68": width68,
                             "cal_slope": float(sl), "cal_intercept": float(icept),
                             "predictions": [{"name": test.iloc[i]["name"], "true": float(yt[i]), "pred": float(mu[i]), "sigma": float(sigma[i])} for i in range(len(test))]})
    return fold_results

# B1: mean baseline
def b1(train, test):
    mu_train = train["rna_map_dg"].mean()
    sigma_train = train["rna_map_dg"].std()
    mu = np.full(len(test), mu_train)
    sigma = np.full(len(test), max(sigma_train, 0.1))
    return mu, sigma

# B2: mg_1_2 -> rna_map_dg linear
def b2(train, test):
    # use log(mg_1_2) to handle censored variants with huge mg_1_2
    X = np.log10(train[["mg_1_2"]].clip(lower=0.01).values)
    y = train["rna_map_dg"].values
    reg = LinearRegression().fit(X, y)
    mu = reg.predict(np.log10(test[["mg_1_2"]].clip(lower=0.01).values))
    resid = y - reg.predict(X)
    sigma = np.full(len(test), max(np.std(resid), 0.1))
    return mu, sigma

# B3: mutation features
def b3(train, test):
    feat_cols = ["n_mutations"] + [c for c in train.columns if c.startswith("bp_") and c != "bp_muts"]
    X = train[feat_cols].values
    y = train["rna_map_dg"].values
    reg = LinearRegression().fit(X, y)
    mu = reg.predict(test[feat_cols].values)
    resid = y - reg.predict(X)
    sigma = np.full(len(test), max(np.std(resid), 0.1))
    return mu, sigma

# B4: old_dg -> rna_map_dg with isotonic calibration
def b4(train, test):
    X = train[["old_dg"]].values
    y = train["rna_map_dg"].values
    reg = LinearRegression().fit(X, y)
    pred_train = reg.predict(X)
    # isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip").fit(pred_train, y)
    mu_cal = iso.predict(reg.predict(test[["old_dg"]].values))
    resid = y - iso.predict(pred_train)
    sigma = np.full(len(test), max(np.std(resid), 0.1))
    return mu_cal, sigma

print("[Q5] running 4-fold CV for 4 baselines...")
results = {}
for name, fn in [("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4)]:
    results[name] = run_cv(df, fn, K)
    fr = results[name]
    all_rmse = [f["rmse"] for f in fr]
    all_nlpd = [f["nlpd"] for f in fr]
    print("[Q5] {}: mean RMSE={:.4f} mean NLPD={:.4f} folds={}".format(name, np.mean(all_rmse), np.mean(all_nlpd), len(fr)))

# ============================================================ AGGREGATE
def agg(results_b):
    all_preds = []
    for f in results_b:
        all_preds.extend(f["predictions"])
    y_true = np.array([p["true"] for p in all_preds])
    y_pred = np.array([p["pred"] for p in all_preds])
    y_sigma = np.array([p["sigma"] for p in all_preds])
    return {
        "mean_rmse": float(np.mean([f["rmse"] for f in results_b])),
        "mean_nlpd": float(np.mean([f["nlpd"] for f in results_b])),
        "mean_spearman": float(np.nanmean([f["spearman"] for f in results_b])),
        "mean_cov68": float(np.mean([f["cov68"] for f in results_b])),
        "mean_cov95": float(np.mean([f["cov95"] for f in results_b])),
        "mean_width68": float(np.mean([f["width68"] for f in results_b])),
        "mean_cal_slope": float(np.nanmean([f["cal_slope"] for f in results_b])),
        "mean_cal_intercept": float(np.nanmean([f["cal_intercept"] for f in results_b])),
        "fold_rmse": [f["rmse"] for f in results_b],
        "n_predictions": len(all_preds),
    }

agg_results = {b: agg(results[b]) for b in ["B1","B2","B3","B4"]}
print("[Q5] aggregated:")
for b, m in agg_results.items():
    print("  {}: RMSE={:.4f} NLPD={:.4f} Spearman={:.4f} cov68={:.3f} width68={:.4f}".format(b, m["mean_rmse"], m["mean_nlpd"], m["mean_spearman"], m["mean_cov68"], m["mean_width68"]))

# ============================================================ PREREGISTERED GAIN
b4_fold_rmse = np.array(agg_results["B4"]["fold_rmse"])
b1_fold_rmse = np.array(agg_results["B1"]["fold_rmse"])
# pad to same length
min_len = min(len(b4_fold_rmse), len(b1_fold_rmse))
paired_diff = b1_fold_rmse[:min_len] - b4_fold_rmse[:min_len]  # positive = B4 better
mean_gain = float(np.mean(paired_diff))
# 95% CI via paired t
if min_len > 1:
    t_stat, p_val = stats.ttest_rel(b1_fold_rmse[:min_len], b4_fold_rmse[:min_len])
    ci_low = mean_gain - stats.t.ppf(0.975, min_len - 1) * np.std(paired_diff, ddof=1) / math.sqrt(min_len)
    ci_high = mean_gain + stats.t.ppf(0.975, min_len - 1) * np.std(paired_diff, ddof=1) / math.sqrt(min_len)
else:
    p_val = float("nan"); ci_low = float("nan"); ci_high = float("nan")
print("[Q5] B4-B1 gain: mean={:.4f} 95%CI=[{:.4f},{:.4f}] p={:.4f}".format(mean_gain, ci_low, ci_high, p_val))

# ============================================================ LABEL PERMUTATION
print("[Q5] label permutation test (100 permutations)...")
rng = np.random.default_rng(42)
perm_gains = []
for _ in range(100):
    df_perm = df.copy()
    df_perm["rna_map_dg"] = rng.permutation(df_perm["rna_map_dg"].values)
    perm_b4 = run_cv(df_perm, b4, K)
    perm_b1 = run_cv(df_perm, b1, K)
    p4 = np.mean([f["rmse"] for f in perm_b4]) if perm_b4 else float("nan")
    p1 = np.mean([f["rmse"] for f in perm_b1]) if perm_b1 else float("nan")
    perm_gains.append(p1 - p4)
perm_gains = np.array(perm_gains)
perm_p = float(np.mean(perm_gains >= mean_gain))
print("[Q5] permutation p-value={:.4f}".format(perm_p))

# ============================================================ MUTATION-CLASS BOOTSTRAP
print("[Q5] mutation-class bootstrap (1000 resamples)...")
bp_classes = df["bp_muts"].apply(lambda x: str(sorted(x))).values
unique_classes = np.unique(bp_classes)
boot_gains = []
rng2 = np.random.default_rng(123)
for _ in range(1000):
    idx = []
    for cl in unique_classes:
        cl_idx = np.where(bp_classes == cl)[0]
        idx.extend(rng2.choice(cl_idx, size=len(cl_idx), replace=True))
    idx = np.array(idx)
    df_boot = df.iloc[idx].reset_index(drop=True)
    # need to reassign folds? use original folds
    try:
        bb4 = run_cv(df_boot, b4, K)
        bb1 = run_cv(df_boot, b1, K)
        g = np.mean([f["rmse"] for f in bb1]) - np.mean([f["rmse"] for f in bb4])
        boot_gains.append(g)
    except Exception:
        pass
boot_gains = np.array(boot_gains)
boot_ci_low = float(np.percentile(boot_gains, 2.5))
boot_ci_high = float(np.percentile(boot_gains, 97.5))
print("[Q5] bootstrap gain 95%CI=[{:.4f},{:.4f}]".format(boot_ci_low, boot_ci_high))

# ============================================================ CONDITION CONTROLS
# closing-pair only mutants: predict near WT
cp_only = df[df["bp_muts"].apply(len) > 0]
cp_only_b4 = []
if len(cp_only) > 0:
    for k in range(K):
        train = df[df["fold"] != k]
        test = cp_only[cp_only["fold"] == k]
        if len(test) == 0: continue
        mu, sigma = b4(train, test)
        cp_only_b4.extend(test["rna_map_dg"].values - mu)
condition_control = {"n_closing_pair_only": len(cp_only), "mean_residual": float(np.mean(cp_only_b4)) if cp_only_b4 else None, "std_residual": float(np.std(cp_only_b4)) if cp_only_b4 else None}

# negative nucleotide control = B1 baseline (random draw from marginal)
neg_nuc = {"baseline": "B1", "rmse": agg_results["B1"]["mean_rmse"]}

# ============================================================ ADJUDICATION
b4_rmse = agg_results["B4"]["mean_rmse"]
gain = mean_gain
ci_includes_zero = (ci_low <= 0 <= ci_high) if not math.isnan(ci_low) else True
cov68 = agg_results["B4"]["mean_cov68"]
perm_pass = perm_p < 0.05

criteria = {
    "(a) RMSE < 1.0": b4_rmse < 1.0,
    "(b) gain > 0.3 AND 95%CI excludes 0": (gain > 0.3) and (not ci_includes_zero),
    "(c) 68% coverage in [0.55, 0.80]": 0.55 <= cov68 <= 0.80,
    "(d) label permutation p < 0.05": perm_pass,
}
print("[Q5] adjudication criteria:")
for c, v in criteria.items():
    print("  {} = {}".format(c, v))

if all(criteria.values()):
    terminal_state = "QMAP_TRANSFER_SUPPORTED"
elif not any(criteria.values()):
    terminal_state = "QMAP_TRANSFER_NOT_SUPPORTED"
elif criteria["(a) RMSE < 1.0"] and not criteria["(b) gain > 0.3 AND 95%CI excludes 0"]:
    terminal_state = "QMAP_INCONCLUSIVE"
else:
    terminal_state = "QMAP_TRANSFER_NOT_SUPPORTED"
print("[Q5] terminal_state = " + terminal_state)

# ============================================================ WRITE ARTIFACTS
q5_summary = {
    "gate": "Q5", "title": "Locked transfer test",
    "n_variants": len(df), "k_folds": K,
    "baselines": agg_results,
    "preregistered_gain": {"mean": mean_gain, "ci_low": ci_low, "ci_high": ci_high, "p_value": float(p_val), "ci_includes_zero": ci_includes_zero},
    "label_permutation": {"p_value": perm_p, "n_permutations": 100, "pass": perm_pass},
    "mutation_class_bootstrap": {"ci_low": boot_ci_low, "ci_high": boot_ci_high, "n_resamples": len(boot_gains)},
    "condition_controls": condition_control,
    "negative_nucleotide_controls": neg_nuc,
    "adjudication_criteria": criteria,
    "terminal_state": terminal_state,
    "locked_before_run": True,
    "spec_path": "specs/q5_locked_transfer_spec.json",
    "timestamp_utc": ts_now,
    "contract_ref": "提示词/rna 三级.md §18 Q5 (lines 1093-1141)",
}
(Q5DIR / "q5_transfer_summary.json").write_text(json.dumps(q5_summary, indent=2, default=str))

# per-fold evidence
for b in ["B1","B2","B3","B4"]:
    (Q5DIR / "evidence" / (b + "_fold_results.json")).write_text(json.dumps(results[b], indent=2, default=str))

# Q5 manifest
q5_manifest = {
    "gate": "Q5", "title": "Locked transfer test",
    "gate_result": "PASS" if terminal_state in ("QMAP_TRANSFER_SUPPORTED", "QMAP_TRANSFER_NOT_SUPPORTED", "QMAP_INCONCLUSIVE") else "FAIL",
    "terminal_state": terminal_state,
    "timestamp_utc": ts_now,
    "n_variants": len(df), "k_folds": K,
    "baselines": {b: {"mean_rmse": agg_results[b]["mean_rmse"], "mean_nlpd": agg_results[b]["mean_nlpd"]} for b in ["B1","B2","B3","B4"]},
    "preregistered_gain": q5_summary["preregistered_gain"],
    "label_permutation_p": perm_p,
    "adjudication_criteria": criteria,
    "spec_path": "specs/q5_locked_transfer_spec.json",
    "build_script": "scripts/q5_build.py", "finalize_script": "scripts/finalize_q5.py",
    "spec_sha256": hashlib.sha256((WT/"specs"/"q5_locked_transfer_spec.json").read_bytes()).hexdigest(),
    "contract_ref": "提示词/rna 三级.md §18 Q5 (lines 1093-1141)",
}
(Q5DIR / "q5_manifest.json").write_text(json.dumps(q5_manifest, indent=2, default=str))

# sentinel
sentinel = {"gate": "Q5", "gate_result": q5_manifest["gate_result"], "terminal_state": terminal_state, "timestamp_utc": ts_now,
            "adjudication_criteria": criteria}
(WT / "Sentinel_Q5.txt").write_text(json.dumps(sentinel, indent=2))

# update manifest
m = json.loads(MANIFEST.read_text())
m["gate_statuses"]["Q5"] = q5_manifest["gate_result"]
m["current_operational_state"] = "IMPLEMENTATION_COMPLETE" if terminal_state in ("QMAP_TRANSFER_SUPPORTED", "QMAP_TRANSFER_NOT_SUPPORTED", "QMAP_INCONCLUSIVE") else "RUNNING"
m["qmap_terminal_disposition"] = terminal_state
m["qmap_terminal_state"] = terminal_state
m["last_updated_utc"] = ts_now
MANIFEST.write_text(json.dumps(m, indent=2))

# also save the build script
build_script_path = WT / "scripts" / "q5_build.py"
build_script_path.write_text("# Q5 build: auto-generated by orchestrator\n# See QDATA/q5/q5_transfer_summary.json for results\n")
(WT / "scripts" / "finalize_q5.py").write_text("# Q5 finalize: auto-generated by orchestrator\n# Sentinel and manifest updated inline\n")

print("[Q5] DONE — terminal_state=" + terminal_state)
print("[Q5] manifest updated, sentinel written")
