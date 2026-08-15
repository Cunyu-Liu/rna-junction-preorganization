"""Per-scaffold stratum sigma figure for the manuscript (r45 corrected).

Generates a grouped bar plot: per-scaffold sigma_m (measured) and sigma_c
(censored) from the corrected r45 calibration (extended grid floor 0.05),
with the censoring rate annotated on the top axis.  This is the key method
figure showing that the measured and censored strata require fundamentally
different sigma scales, and that the high-censoring operators (scaf9 78.5%,
scaf1 59.2%) drive the largest sigma asymmetry.

Output: per_scaf_stratum_sigma_figure.svg (PDF/vector).
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
    cal = json.loads(Path(f"{R}/per_scaf_stratum_sigma_calibration.json").read_text())
    # Corrected r45: per-scaf stratum sigma from the fit_log_folds
    fl = cal["fit_log_folds"]
    first_fold = list(fl.values())[0]
    strat = first_fold["stratum_sigma"]
    scafs = sorted(strat, key=int)
    sigma_m = [strat[s]["sigma_m"] for s in scafs]
    sigma_c = [strat[s]["sigma_c"] for s in scafs]

    if not HAVE_MPL:
        print("matplotlib not available; skipping figure")
        print(f"{'scaf':>5s} {'sigma_m':>8s} {'sigma_c':>8s}")
        for s in scafs:
            print(f"{int(s):5d} {sigma_m[scafs.index(s)]:8.3f} {sigma_c[scafs.index(s)]:8.3f}")
        return

    # Censoring rates from the frozen data profile (documented values)
    cens_rates = {1: 59.2, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.1, 6: 0.3, 7: 0.6, 8: 9.7, 9: 78.5}
    labels = [f"scaf{s}\n({cens_rates[int(s)]:.0f}%)" for s in scafs]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scafs))
    w = 0.35
    bars_m = ax.bar(x - w / 2, sigma_m, w, label=r"$\sigma_m$ (measured)", color="#4C72B0", alpha=0.85)
    bars_c = ax.bar(x + w / 2, sigma_c, w, label=r"$\sigma_c$ (censored)", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Emitted sigma (kcal/mol)", fontsize=12)
    ax.set_title("Per-operator stratum-specific sigma (r45 corrected)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(sigma_m + sigma_c) * 1.15)
    ax.axhline(0.7, color="gray", ls="--", lw=0.8, alpha=0.6, label="Frozen 0.7 (reference)")
    ax.legend(fontsize=10)

    # Annotate bar values
    for bar, val in zip(bars_m, sigma_m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_c, sigma_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{R}/per_scaf_stratum_sigma_figure.svg", dpi=150)
    fig.savefig(f"{R}/per_scaf_stratum_sigma_figure.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {R}/per_scaf_stratum_sigma_figure.svg / .png")


if __name__ == "__main__":
    main()