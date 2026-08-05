"""F0 figure generation tests (v1.5)."""
import json
import os

RUN_ROOT = "/mnt/cunyuliu/v1_5_manuscript_readiness_20260805T052052Z"
F0 = os.path.join(RUN_ROOT, "figures", "f0")


def test_decision_state():
    with open(os.path.join(F0, "F0_decision.json")) as f:
        d = json.load(f)
    assert d["state"] == "F0_FIGURES_GENERATED"
    assert d["main_figures"] == 8


def test_all_main_figures_exist():
    fig_ids = [f"fig{i}" for i in range(1, 9)]
    for fid in fig_ids:
        assert os.path.exists(os.path.join(F0, "main", f"{fid}.png"))
        assert os.path.exists(os.path.join(F0, "main", f"{fid}.pdf"))


def test_manifest_has_checksums_and_sources():
    with open(os.path.join(F0, "figure_manifest.json")) as f:
        m = json.load(f)
    for fid in [f"fig{i}" for i in range(1, 9)]:
        assert fid in m
        assert "checksums" in m[fid] and "png" in m[fid]["checksums"]
        assert "source_data" in m[fid]


def test_source_data_present():
    expected = ["fig2_tecto.tsv", "fig3_qmap.tsv", "fig4_membership.tsv",
                "fig5_b3.tsv", "fig6_x0.tsv", "fig7_claims.tsv", "fig8_ablation.tsv"]
    for e in expected:
        assert os.path.exists(os.path.join(F0, "source_data", e))


def test_figure_plan_present():
    assert os.path.exists(os.path.join(F0, "figure_plan.tsv"))