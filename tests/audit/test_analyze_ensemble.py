"""Unit tests for analyze_ensemble.py (mu-ensemble of nonlinear MLP variants)."""
import json
import tempfile
from pathlib import Path

import numpy as np

from audit.repair.analyze_ensemble import (
    ENSEMBLE_ID,
    build_ensemble_rows,
    pooled_nll_by_model,
)


def _row(rid, model_id, mu, jid="1", cens=False, y=-7.0, sigma=0.7):
    return {"axis": "edit_x_nested_context", "fold": "e:AAAC_GAAC",
            "source_row_id": rid, "jid": jid, "scaf": 1, "context": "AG_CG",
            "model_id": model_id, "y": y, "cens": cens, "mu": mu,
            "sigma": sigma, "abstain": False, "support": True,
            "fallback_type": None}


def test_build_ensemble_averages_mu_and_coverage_matches():
    m1 = ["a", "b", "c"]
    rows = []
    # rows present in all three members
    for rid in ("R1", "R2", "R3"):
        for m in m1:
            rows.append(_row(rid, m, mu=float(rid[1]) * 1.0))
    # R4 present only in 'a' and 'b' -> excluded from ensemble (not full coverage)
    rows.append(_row("R4", "a", mu=4.0))
    rows.append(_row("R4", "b", mu=4.0))

    ens = build_ensemble_rows(rows, m1)
    rids = {r["source_row_id"] for r in ens}
    assert rids == {"R1", "R2", "R3"}
    assert all(r["model_id"] == ENSEMBLE_ID for r in ens)
    assert all(r["sigma"] == 0.7 for r in ens)
    for r in ens:
        rid = int(r["source_row_id"][1])
        assert np.isclose(r["mu"], rid)


def test_build_ensemble_single_row_mean():
    rows = [_row("R1", "a", mu=-8.0), _row("R1", "b", mu=-6.0)]
    ens = build_ensemble_rows(rows, ["a", "b"])
    assert len(ens) == 1
    assert np.isclose(ens[0]["mu"], -7.0)


def test_pooled_nll_by_model_matches_hand_computed():
    # two supported rows, same junction, uncensored: NLL = 0.5 log(2pi) + log(sigma) + 0.5 z^2
    rows = [_row("R1", "m", mu=-7.0, y=-7.0, cens=False),
            _row("R2", "m", mu=-7.0, y=-6.0, cens=False)]
    pooled = pooled_nll_by_model(rows)
    import math
    const = 0.5 * math.log(2 * math.pi) + math.log(0.7)
    nll_r1 = const + 0.5 * (0.0 / 0.7) ** 2
    nll_r2 = const + 0.5 * (1.0 / 0.7) ** 2
    # single junction -> pooled = mean of its two row NLLs
    expected = (nll_r1 + nll_r2) / 2.0
    assert np.isclose(pooled["m"], expected)


def test_main_writes_json(tmp_path):
    from audit.repair.analyze_ensemble import main
    import sys
    preds = []
    for m in ("a", "b", "motif_topology_hierarchy"):
        for rid in ("R1", "R2"):
            preds.append(_row(rid, m, mu=-7.0))
    pfile = tmp_path / "p.jsonl"
    with open(pfile, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    outfile = tmp_path / "out" / "ens.json"
    sys.argv = ["analyze_ensemble", "--preds", str(pfile), "--out", str(outfile),
                "--members", "a", "b", "--base", "motif_topology_hierarchy"]
    main()
    res = json.loads(Path(outfile).read_text())
    assert res["n_ensemble_rows"] == 2
    assert ENSEMBLE_ID in res["pooled_junction_macro_nll"]
    assert f"{ENSEMBLE_ID}_vs_motif_topology_hierarchy" in res["contrasts"]


if __name__ == "__main__":
    tests = [test_build_ensemble_averages_mu_and_coverage_matches,
             test_build_ensemble_single_row_mean,
             test_pooled_nll_by_model_matches_hand_computed,
             test_main_writes_json]
    import sys as _sys
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        failed = 0
        for t in tests:
            try:
                t(Path(_d))
                print(f"PASS {t.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {t.__name__}: {e}")
        print("analyze_ensemble tests", "PASS" if failed == 0 else f"{failed} FAILURES")
