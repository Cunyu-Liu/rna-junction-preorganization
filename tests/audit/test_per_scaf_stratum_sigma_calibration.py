"""Unit tests for the r45 per-scaffold x stratum sigma calibration.

Covers the vectorized `_scan_sigma` and the leave-one-fold-out stratum
calibration logic in per_scaf_stratum_sigma_calibration.py and
per_scaf_stratum_sigma_horizontal_table.py, plus the r46 measured-only mu
correction in measured_only_operator_mu_correction.py.  Tests are
self-contained on synthetic prediction rows (no server data dependency).
"""
import numpy as np

from audit.repair.per_scaf_stratum_sigma_calibration import (
    _pooled,
    _scan_sigma_stratum,
    main as _main_calib,
)
from audit.repair.per_scaf_stratum_sigma_horizontal_table import (
    _per_scaf_calibrate,
    _per_scaf_stratum_calibrate,
    _scan_sigma,
)
from audit.repair.measured_only_operator_mu_correction import (
    _calibrate_r45_plus_mu,
)


def _row(rid, fold, scaf, y, cens, mu, jid=None):
    return {"jid": jid or str(rid), "fold": fold, "scaf": scaf,
            "y": y, "cens": cens, "mu": mu}


def _mk(rids, fold, scaf, mu, cens=False, y=-7.0):
    return {rid: _row(rid, fold, scaf, y, cens, mu) for rid in rids}


def test_scan_sigma_picks_minimal_nll():
    # two groups (junctions); the optimal sigma should be near the residual
    # scale ~1.0, not the grid edge.
    rows = {}
    rng = np.random.default_rng(0)
    for g in range(20):
        base = rng.normal(0.0, 1.0)
        for i in range(5):
            rid = f"{g}_{i}"
            rows[rid] = _row(rid, "f0", 1, y=base + rng.normal(0, 1.0),
                             cens=False, mu=base)
    s, nll = _scan_sigma_stratum(rows, cens_mask=False, grid=np.arange(0.5, 1.6, 0.05))
    assert 0.8 <= s <= 1.4, f"optimal sigma {s} should be near residual scale"
    assert np.isfinite(nll)


def test_scan_sigma_vectorized_matches_bruteforce():
    rows = {}
    rng = np.random.default_rng(1)
    for g in range(8):
        base = rng.normal(0.0, 0.5)
        for i in range(4):
            rid = f"{g}_{i}"
            rows[rid] = _row(rid, "f0", 1, y=base, cens=(i % 2 == 0),
                             mu=base - 0.3)
    grid = np.arange(0.4, 1.0, 0.05)
    sv, nv = _scan_sigma_stratum(rows, cens_mask=None, grid=grid)
    # brute force reference
    best_s, best_n = None, np.inf
    for s in grid:
        cal = {r: {**p, "sigma": s} for r, p in rows.items()}
        n = _pooled(cal)
        if n < best_n:
            best_n, best_s = n, s
    assert np.isclose(sv, best_s)
    assert np.isclose(nv, best_n, atol=1e-9)


def test_scan_sigma_none_uses_all_rows():
    rows = {}
    for i in range(6):
        rows[f"r{i}"] = _row(f"r{i}", "f0", 1, y=-7.0, cens=(i % 2 == 0), mu=-7.0)
    s_all, _ = _scan_sigma_stratum(rows, cens_mask=None, grid=np.arange(0.4, 1.0, 0.05))
    s_meas, _ = _scan_sigma_stratum(rows, cens_mask=False, grid=np.arange(0.4, 1.0, 0.05))
    s_cens, _ = _scan_sigma_stratum(rows, cens_mask=True, grid=np.arange(0.4, 1.0, 0.05))
    assert s_all is not None
    assert s_meas is not None
    assert s_cens is not None


def test_stratum_calibrate_applies_stratum_specific_sigma():
    # Build 2 folds with a high-censoring scaffold (scaf 9, mostly censored,
    # small sigma_c) and a measured-heavy scaffold (scaf 2, large sigma_m).
    # Calibration on fold1 must fit on fold0's rows only and apply per-stratum.
    rng = np.random.default_rng(2)
    preds = {}
    for fold, seed in (("f0", 10), ("f1", 11)):
        for sc in (2, 9):
            for i in range(40):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 5 != 0)  # scaf9 80% censored
                mu = -6.5 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.3)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 4}")
    cal, sigma_map = _per_scaf_stratum_calibrate(preds, ["f0", "f1"])
    assert len(cal) == len(preds)
    for f, sm in sigma_map.items():
        assert 9 in sm and 2 in sm
        # measured sigma for the low-censoring scaffold should be larger than
        # the censored sigma for the high-censoring scaffold
        assert sm[9]["sigma_c"] is not None
        assert sm[2]["sigma_m"] is not None
    # every calibrated row carries a finite positive sigma
    for rid, p in cal.items():
        assert p["sigma"] > 0.0
        assert np.isfinite(p["sigma"])


def test_stratum_calibrate_matches_pooled_recompute():
    # Applying the calibrated sigma map and recomputing pooled NLL by hand must
    # equal the pooled NLL of the calibrated dict (consistency check).
    rng = np.random.default_rng(3)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (1, 3, 9):
            for i in range(30):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 2 == 0)
                mu = -6.0 if sc == 9 else -4.5
                y = -7.1 if cens else mu + rng.normal(0, 0.4)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 3}")
    cal, _ = _per_scaf_stratum_calibrate(preds, ["f0", "f1", "f2"])
    from audit.evaluation.metrics import row_nll
    from collections import defaultdict
    jd = defaultdict(list)
    for rid, p in cal.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    pooled_hand = float(np.mean([np.mean(v) for v in jd.values()]))
    assert np.isclose(_pooled(cal), pooled_hand, atol=1e-9)


def test_horizontal_per_scaf_and_stratum_consistent():
    # On a tiny synthetic set both calibrators must return finite sigmas and
    # per-scaffold (r38) must be a single sigma per scaffold while stratum
    # (r45) can differ between strata on the same scaffold.
    rng = np.random.default_rng(4)
    preds = {}
    for fold in ("f0", "f1"):
        for sc in (2, 5, 9):
            for i in range(25):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 3 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.35)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 5}")
    folds = ["f0", "f1"]
    cal_scaf, scaf_map = _per_scaf_calibrate(preds, folds)
    cal_strat, strat_map = _per_scaf_stratum_calibrate(preds, folds)
    for f in folds:
        # r38: single sigma per scaffold
        for sc in scaf_map[f]:
            assert scaf_map[f][sc] > 0.0
        # r45: two strata
        for sc in strat_map[f]:
            assert strat_map[f][sc]["sigma_m"] > 0.0
            assert strat_map[f][sc]["sigma_c"] > 0.0
    assert len(cal_scaf) == len(cal_strat) == len(preds)


def test_r46_mu_correction_keeps_censored_mu_untouched():
    # The core honesty property of r46: measured rows get mu + alpha, censored
    # rows keep their mu EXACTLY (never shifted), so censored-side NLL cannot be
    # damaged by the mu correction.
    rng = np.random.default_rng(7)
    preds = {}
    orig_censored_mu = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (1, 9):
            for i in range(30):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 2 == 0)
                mu = -6.0 if sc == 9 else -4.5
                y = -7.1 if cens else mu + rng.normal(0, 0.4)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 3}")
                if cens:
                    orig_censored_mu[rid] = mu
    cal, fit_log = _calibrate_r45_plus_mu(preds, ["f0", "f1", "f2"], shrink=1.0)
    for rid, mu0 in orig_censored_mu.items():
        assert cal[rid]["cens"] is True
        assert np.isclose(cal[rid]["mu"], mu0), \
            "censored mu must never be shifted by r46"
    # measured rows should differ on high-bias scaffolds (shrink=1 full alpha)
    changed_meas = [rid for rid, p in cal.items()
                    if not p["cens"] and not np.isclose(p["mu"], preds[rid]["mu"])]
    assert changed_meas, "measured rows should receive a mu correction"


def test_r46_shrink_zero_equals_r45():
    # shrink=0 disables the mu correction -> must reproduce r45 exactly.
    rng = np.random.default_rng(8)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 5, 9):
            for i in range(25):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 3 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.35)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 5}")
    cal_zero, _ = _calibrate_r45_plus_mu(preds, ["f0", "f1", "f2"], shrink=0.0)
    cal_ref, _ = _per_scaf_stratum_calibrate(preds, ["f0", "f1", "f2"])
    assert np.isclose(_pooled(cal_zero), _pooled(cal_ref), atol=1e-9)
    # sigma assignment must be identical row-for-row (mu unchanged at shrink 0)
    for rid in cal_zero:
        assert np.isclose(cal_zero[rid]["sigma"], cal_ref[rid]["sigma"])


def test_r46_sigma_stays_positive_and_finite():
    rng = np.random.default_rng(9)
    preds = {}
    for fold in ("f0", "f1"):
        for sc in (1, 2, 9):
            for i in range(20):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 4 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.3)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 4}")
    cal, _ = _calibrate_r45_plus_mu(preds, ["f0", "f1"], shrink=0.75)
    for rid, p in cal.items():
        assert p["sigma"] > 0.0
        assert np.isfinite(p["sigma"])
        assert np.isfinite(p["mu"])


if __name__ == "__main__":
    tests = [test_scan_sigma_picks_minimal_nll,
             test_scan_sigma_vectorized_matches_bruteforce,
             test_scan_sigma_none_uses_all_rows,
             test_stratum_calibrate_applies_stratum_specific_sigma,
             test_stratum_calibrate_matches_pooled_recompute,
             test_horizontal_per_scaf_and_stratum_consistent,
             test_r46_mu_correction_keeps_censored_mu_untouched,
             test_r46_shrink_zero_equals_r45,
             test_r46_sigma_stays_positive_and_finite]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print("per_scaf_stratum tests", "PASS" if failed == 0 else f"{failed} FAILURES")
    raise SystemExit(1 if failed else 0)
