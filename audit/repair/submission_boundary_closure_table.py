"""Definitive method-boundary closure table (submission artifact).

Consolidates the ENTIRE model-level lever search (r15-r81) into one
submission-ready table + JSON.  This is the scientific evidence that the frozen
method (7-member ensemble + r62 = 0.7243) is at the train-legal data limit on
this single-study dataset, and that the publishable contribution is the
censor-aware benchmark methodology + calibration chain + boundary closure.

Sections:
  1. Base predictor levers (representation / architecture / training) -- all NEGATIVE
  2. Calibration levers (mu / sigma / ensemble) -- only r45->r51->r56b->r62 positive
  3. Noise-floor decomposition (err10, context visibility) -- model at data limit
  4. Honest caveats (label-derived err10 illegal, sigma_c ceiling unbounded hedge)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"

# ---- 1. Base predictor levers (from audit docs / runs) ----
base_predictor = {
    "representation": [
        {"lever": "63-D seq map (r29 full)", "result": "seq gain -25.73%",
         "verdict": "NEGATIVE (P0.6 TRACK_A_LOCKED)"},
        {"lever": "RNA-FM embedding (r15 smoke)", "result": "1.57 vs 0.74",
         "verdict": "NEGATIVE"},
        {"lever": "local-context one-hot 24-D (r18 smoke)", "result": "0.886 vs 0.741",
         "verdict": "NEGATIVE"},
        {"lever": "Vienna21 increment (r07-r14)", "result": "~+2%",
         "verdict": "small (frozen into features)"},
        {"lever": "RBF kernel mix (r36 full)", "result": "0.799 vs 0.741",
         "verdict": "NEGATIVE"},
    ],
    "architecture": [
        {"lever": "deeper MLP 4/5-layer (r16)", "result": "0.780/0.953 vs 0.741",
         "verdict": "NEGATIVE"},
        {"lever": "in-network heteroscedastic sigma head (r17 smoke)", "result": "0.937 vs 0.830",
         "verdict": "NEGATIVE"},
        {"lever": "latent-operator head (r27 smoke)", "result": "1.09/1.15 vs 0.74/0.90",
         "verdict": "NEGATIVE"},
    ],
    "training": [
        {"lever": "Student-t df sweep (r20)", "result": "df=7/10 optimal", "verdict": "frozen t7"},
        {"lever": "SWA (r22 smoke)", "result": "fold1 improves fold2 worsens", "verdict": "MIXED (not used)"},
        {"lever": "bagging bootstrap (r25 smoke)", "result": "0.758 vs 0.748", "verdict": "NEGATIVE"},
        {"lever": "censor-aware reweighting (r49 smoke)", "result": "0.797-0.826 vs 0.789", "verdict": "NEGATIVE"},
        {"lever": "per-scaf sigma training (r81 smoke)", "result": "t7 sig-train worse +0.088/+0.070",
         "verdict": "NEGATIVE: train/eval sigma mismatch does not cost mu quality"},
    ],
    "ensemble": [
        {"lever": "GBDT+MLP mix (r33-r35)", "result": "error diversity = key", "verdict": "frozen 7-member"},
        {"lever": "family weight (r50/r53/r75)", "result": "wg=0.5 optimal (re-verified under r62)",
         "verdict": "frozen"},
        {"lever": "per-scaf family weight (r52)", "result": "0.7841 > 0.7815", "verdict": "NEGATIVE"},
        {"lever": "feature-diverse member (r48)", "result": "error corr 0.88, gain -0.0009",
         "verdict": "NEGATIVE"},
        {"lever": "mixture-of-predictives (r41)", "result": "0.858 vs 0.853", "verdict": "NEGATIVE"},
    ],
}

# ---- 2. Calibration levers ----
calibration = {
    "positive_chain": [
        {"lever": "global sigma scan", "result": "0.8419 (sigma=0.62)", "verdict": "baseline"},
        {"lever": "per-scaf sigma (r38)", "result": "0.8166", "verdict": "positive"},
        {"lever": "per-scaf x stratum sigma (r45)", "result": "0.7907", "verdict": "positive"},
        {"lever": "joint mu-affine + sigma rescan (r51)", "result": "0.7815", "verdict": "positive"},
        {"lever": "per-context EB mu (r56b)", "result": "0.7314", "verdict": "MAJOR positive (CI lower>0)"},
        {"lever": "decoupled sigma re-scan (r62)", "result": "0.7243 (FROZEN)", "verdict": "positive (impl correction)"},
    ],
    "closed_levers": [
        {"lever": "per-context sigma on r56b (r60)", "result": "best 0.7282 > 0.7314", "verdict": "NEGATIVE"},
        {"lever": "r62 + per-context EB sigma (r66)", "result": ">= 0.7275 > 0.7243", "verdict": "NEGATIVE"},
        {"lever": "context bias feature-predictable (r67)", "result": "OOD R2 = -0.31..-0.42",
         "verdict": "NEGATIVE: irreducible per-context random effect"},
        {"lever": "err10-driven per-row sigma (r70/r72)", "result": "0.7195 (-0.0048)",
         "verdict": "ILLEGAL: err10 is label-derived (train_legal=False)"},
        {"lever": "row-level sigma from legal features (r73)", "result": "0.819-0.827 > 0.7243",
         "verdict": "NEGATIVE"},
        {"lever": "joint per-context (mu,sigma) EB (r74)", "result": "0.8520 > 0.7243",
         "verdict": "NEGATIVE: sequential structure is correct"},
        {"lever": "nonlinear monotone mu (r76)", "result": "0.739-0.745 > 0.7243",
         "verdict": "NEGATIVE"},
        {"lever": "per-context slope (r77)", "result": "split-half corr 0.527 (unstable)",
         "verdict": "NEGATIVE: only intercept is stable structure"},
        {"lever": "sigma_c ceiling extension (r78/r79)", "result": "0.7190 but UNBOUNDED hedge",
         "verdict": "REJECTED: monotone to log(2) abstention, no finite optimum"},
        {"lever": "context EB mu on censored rows (r80)", "result": "0.7299 > 0.7243",
         "verdict": "NEGATIVE"},
        {"lever": "per-scaf sigma base retraining (r81)", "result": "worse on 2 folds",
         "verdict": "NEGATIVE"},
    ],
}

# ---- 3. Noise-floor decomposition ----
noise_floor = {
    "residual_sd_measured": 0.548,
    "err10_rms": 0.248,
    "ratio_resid_over_err": 2.21,
    "ood_context_resid_sd": 0.694,
    "train_visible_context_resid_sd": 0.525,
    "measured_layer_nll_frozen": 0.8182,
    "gaussian_floor_at_sigma_0_54": 0.803,
    "interpretation": (
        "measured layer NLL 0.8182 sits at the Gaussian info floor for its emitted "
        "sigma_m (0.42-0.68).  The residual gap vs err10 is dominated by OOD-context "
        "random effects (r67: not feature-predictable; r69: OOD 0.694 vs train-visible "
        "0.525).  The only recoverable part (train-visible context bias) is fully "
        "extracted by r56b/r62 (cumulative -0.0568)."),
}

# ---- 4. Honest caveats ----
honest_caveats = [
    "err10 (dg10 measurement SE) is label-derived: feature_provenance.py marks "
    "derived_from_target=True, train_legal=False.  Its -0.0048 sigma gain is NOT a "
    "legitimate model improvement -- it is label leakage.",
    "sigma_c ceiling extension (r78/r79) lowers NLL but via unbounded monotone "
    "abstention toward log(2)=0.693 (P(censored)=0.5).  No finite optimum; the "
    "frozen ceiling 1.6 is a principled anti-degeneracy choice.",
    "r56b context correction does not apply to the nuisance baseline (no context "
    "feature) -> same-caliber relative gain (vs r45-nuisance +27.15%) is the "
    "conservative headline; absolute NLL 0.7243 must be reported alongside.",
    "r62 vs r56b pooled gain -0.0071 but edit-cluster CI crosses 0 (implementation "
    "correction, not new lever); pooled estimand is primary.",
]

out = {
    "frozen_method": "7-member ensemble (wg=0.5) + r62 (r56b per-context EB mu + "
                     "decoupled per-scaf x stratum sigma, kappa=1, mm3)",
    "frozen_nll": 0.7243,
    "method_chain": {"nuisance": 1.0916, "r45": 0.7907, "r51": 0.7815,
                     "r56b": 0.7314, "r62": 0.7243},
    "relative_gain_vs_nuisance_pct": 33.7,
    "relative_gain_vs_nuisance_r45_pct": 27.86,
    "base_predictor_levers": base_predictor,
    "calibration_levers": calibration,
    "noise_floor": noise_floor,
    "honest_caveats": honest_caveats,
    "conclusion": (
        "Model-level method improvement is exhausted within train-legal constraints "
        "on this single-study dataset.  The frozen method is at the Gaussian "
        "information floor (measured layer) with the remaining gap being irreducible "
        "OOD-context random effects + measurement noise.  The publishable "
        "contribution is the censor-aware benchmark methodology (evaluation + "
        "scaf->context double-layer mu EB + decoupled sigma calibration + boundary "
        "closure), not a base-model breakthrough."),
}

Path(f"{R}/submission_boundary_closure_table.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n")

# ---- markdown rendering ----
md = []
md.append("# Method Boundary Closure Table (submission artifact)\n")
md.append(f"**Frozen method**: {out['frozen_method']} = **0.7243**\n")
md.append("| chain | NLL |")
md.append("|-------|-----|")
for k, v in out["method_chain"].items():
    md.append(f"| {k} | {v} |")
md.append("")
md.append(f"Relative gain vs nuisance: **{out['relative_gain_vs_nuisance_pct']}%** "
          f"(absolute); vs r45-nuisance same-caliber: **{out['relative_gain_vs_nuisance_r45_pct']}%**\n")
for sec_name, sec in (("Base predictor levers", base_predictor),
                      ("Calibration positive chain", {"positive": calibration["positive_chain"]}),
                      ("Calibration closed levers", {"closed": calibration["closed_levers"]})):
    md.append(f"## {sec_name}\n")
    for cat, rows in sec.items():
        md.append(f"### {cat}\n")
        md.append("| lever | result | verdict |")
        md.append("|-------|--------|---------|")
        for row in rows:
            md.append(f"| {row['lever']} | {row['result']} | {row['verdict']} |")
        md.append("")
md.append("## Noise floor\n")
for k, v in noise_floor.items():
    if isinstance(v, str):
        md.append(f"- **{k}**: {v}")
    else:
        md.append(f"- **{k}**: {v}")
md.append("")
md.append("## Honest caveats (must appear in manuscript)\n")
for i, c in enumerate(honest_caveats, 1):
    md.append(f"{i}. {c}")
md.append("")
md.append(out["conclusion"])

Path(f"{R}/submission_boundary_closure_table.md").write_text("\n".join(md) + "\n")
print("\n".join(md[-30:]))
print("\nWrote submission_boundary_closure_table.{json,md}")
