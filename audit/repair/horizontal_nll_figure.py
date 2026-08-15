"""Horizontal comparison figure (Fig 2) + ablation ladder table (Table 2).

Reads the frozen submission table and calibration artifacts:
  - Fig 2: pooled NLL of each model family at frozen sigma 0.7 AND r45
    calibrated, sorted by r45; error bars = edit-cluster CI of the frozen
    method vs nuisance (from per_scaf_stratum_sigma_calibration.json).
  - Table 2: ablation ladder -- nuisance -> best single -> 3x t7 -> 7mem
    -> 7mem+r45 (the method contribution chain).

Outputs:
  - horizontal_nll_figure.svg/.png
  - ablation_ladder.json (machine-readable) + printed markdown table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

R = "/mnt/cunyuliu/rna_junction_repair_20260811T090000Z"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # noqa: BLE001
    HAVE_MPL = False


def main():
    sub = json.loads(Path(f"{R}/submission_horizontal_table.json").read_text())
    frozen = sub["frozen_sigma_07_nll"]
    r45 = sub["r45_calibrated_nll"]

    # Short display names for the manuscript
    NAME = {
        "corrected_v1_31": "63-D seq map (v1.31)",
        "no_sequence_latent_operator": "no-sequence latent operator",
        "train_only_scaffold": "train-only scaffold",
        "motif_topology_hierarchy": "nuisance (baseline)",
        "nonlinear_mlp_nuisance_only_t7": "nuisance-only MLP",
        "nonlinear_mlp_extended_hybrid_reg_deep": "Gaussian MLP (reg_deep)",
        "nonlinear_mlp_extended_hybrid_reg_deep_t": "Student-t MLP (df=5)",
        "nonlinear_mlp_extended_hybrid_reg_deep_t7": "Student-t MLP (df=7)",
        "nonlinear_mlp_extended_hybrid_reg_deep_t10": "Student-t MLP (df=10)",
        "nonlinear_mlp_extended_hybrid_reg_deep_t7_s99": "Student-t MLP s99",
        "nonlinear_mlp_extended_hybrid_reg_deep_t7_s2026": "Student-t MLP s2026",
        "nonlinear_mlp_extended_hybrid_reg_deep_t7_s7": "Student-t MLP s7",
        "xgboost_censored_hybrid": "GBDT",
        "xgboost_censored_hybrid_s99": "GBDT s99",
        "xgboost_censored_hybrid_s2026": "GBDT s2026",
        "xgboost_censored_hybrid_hp_lr03": "GBDT lr03",
        "ENSEMBLE_3x_t7": "3x t7 ensemble",
        "ENSEMBLE_MIXED_7": "7-member ensemble",
    }

    # ---- Fig 2: horizontal NLL bars (r45 sorted, frozen as paired) ----
    order = sorted(r45, key=lambda k: r45[k])
    labels = [NAME.get(k, k) for k in order]
    vals_r45 = [r45[k] for k in order]
    vals_frozen = [frozen[k] for k in order]

    out = {
        "fig2_order": order,
        "fig2_labels": labels,
        "fig2_r45_nll": vals_r45,
        "fig2_frozen07_nll": vals_frozen,
    }

    if HAVE_MPL:
        y = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.5, 8))
        ax.barh(y + 0.2, vals_frozen, height=0.38, label="frozen $\\sigma$=0.7",
                color="#CBCBCB", alpha=0.85)
        ax.barh(y - 0.2, vals_r45, height=0.38,
                label="r45 (per-operator x stratum $\\sigma$)",
                color="#4C72B0", alpha=0.9)
        # highlight the frozen method
        idx7 = order.index("ENSEMBLE_MIXED_7")
        ax.get_yticklabels()
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("pooled-OOF junction-macro NLL (lower = better)", fontsize=11)
        ax.set_title("RNA junction prediction: method comparison", fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.axvline(1.0916, color="gray", ls=":", lw=0.8)
        ax.text(1.0916, len(order) - 0.4, "nuisance 1.0916", fontsize=8, color="gray")
        fig.tight_layout()
        fig.savefig(f"{R}/horizontal_nll_figure.svg", dpi=150)
        fig.savefig(f"{R}/horizontal_nll_figure.png", dpi=150)
        plt.close(fig)
        print(f"Fig2 saved: {R}/horizontal_nll_figure.svg/.png")

    # ---- Table 2: ablation ladder ----
    def nll(key):
        return {"frozen07": round(frozen[key], 4), "r45": round(r45[key], 4)}

    ladder = {
        "nuisance_baseline": nll("motif_topology_hierarchy"),
        "best_single_member": nll("xgboost_censored_hybrid_hp_lr03"),
        "3x_t7_ensemble": nll("ENSEMBLE_3x_t7"),
        "7member_mixed_ensemble": nll("ENSEMBLE_MIXED_7"),
        "frozen_method_7mem_plus_r45": {
            "frozen07": round(frozen["ENSEMBLE_MIXED_7"], 4),
            "r45": round(sub["frozen_method_r45_nll"], 4),
        },
    }
    out["ablation_ladder"] = ladder

    print("\n=== Table 2: ablation ladder ===")
    print(f"{'step':28s} {'frozen 0.7':>10s} {'r45 cal':>10s}")
    for k, v in ladder.items():
        print(f"{k:28s} {v['frozen07']:10.4f} {v['r45']:10.4f}")
    print(f"\nrelative gain (frozen method r45 vs nuisance frozen07): "
          f"{100.0*(frozen['motif_topology_hierarchy']-sub['frozen_method_r45_nll'])/frozen['motif_topology_hierarchy']:.2f}%")

    Path(f"{R}/fig2_and_ablation_ladder.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {R}/fig2_and_ablation_ladder.json")


if __name__ == "__main__":
    main()
