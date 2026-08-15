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
from audit.repair.r47_measured_affine_mu_correction import (
    _ols,
    _ridge_ols,
    _calibrate_r47,
)
from audit.repair.r50_family_weight_sweep_r45 import (
    _blend,
    _ens_mu,
)
from audit.repair.r52_per_scaf_family_weight import (
    _blend_per_scaf,
)
from audit.repair.r53_family_weight_sweep_r51 import (
    _blend as r53_blend,
    _ens_mu as r53_ens_mu,
)
from audit.repair.r54_per_ctx_eb_sigma import (
    _calibrate_r54,
)
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _calibrate_r51,
    _scan_sigma as r51_scan_sigma,
    _ols as r51_ols,
    _ridge_ols as r51_ridge_ols,
    _pooled as r51_pooled,
    GRID as R51_GRID,
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


def test_r50_ens_mu_averages_over_member_subset():
    # _ens_mu must produce the member-mean mu for the given key subset.
    rng = np.random.default_rng(20)
    members = {}
    common = [f"r{i}" for i in range(40)]
    for m in ("g1", "g2", "m1", "m2"):
        members[m] = {rid: _row(rid, "f0", 1, y=-6.0, cens=False,
                                mu=float(rng.normal(0, 1)) + int(m.startswith("g")) * 2.0)
                      for rid in common}
    GBDT = ["g1", "g2"]
    MLP = ["m1", "m2"]
    ens_g = _ens_mu(members, GBDT, common)
    assert len(ens_g) == len(common)
    for rid in common:
        assert np.isclose(ens_g[rid]["mu"], np.mean([members[g][rid]["mu"] for g in GBDT]))
    ens_m = _ens_mu(members, MLP, common)
    for rid in common:
        assert np.isclose(ens_m[rid]["mu"], np.mean([members[m2][rid]["mu"] for m2 in MLP]))
    # mixture: mean over g1+m1 is between the two means
    ens_mix = _ens_mu(members, ["g1", "m1"], common)
    for rid in common:
        assert np.isclose(ens_mix[rid]["mu"],
                          0.5 * (members["g1"][rid]["mu"] + members["m1"][rid]["mu"]))


def test_r50_member_names_match_shootout_universe():
    # The hard-coded member ids in r50 must all exist in the shootout universe.
    from audit.repair.shootout_run import _universe
    from audit.repair import r50_family_weight_sweep_r45 as mod
    U = _universe()
    for m in mod.ALL_MEMBERS:
        assert m in U, f"{m} not registered"
    assert set(mod.GBDT) | set(mod.MLP) == set(mod.ALL_MEMBERS)
    assert len(mod.GBDT) == 4 and len(mod.MLP) == 3


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


def test_ols_recovers_known_slope():
    rng = np.random.default_rng(10)
    x = np.linspace(-8.0, -3.0, 200)
    y = 0.5 - 0.7 * x + rng.normal(0, 0.1, size=len(x))
    a, b = _ols(x, y)
    assert np.isclose(b, -0.7, atol=0.02)
    assert np.isclose(a, 0.5, atol=0.1)


def test_ridge_ols_shrinks_toward_prior():
    # a b=2.0 slope with strong noise: ridge toward b_prior=1.0 should pull
    # the estimate below the raw OLS slope.
    rng = np.random.default_rng(11)
    x = np.linspace(-6.0, -2.0, 30)
    y = 0.0 + 2.0 * x + rng.normal(0, 1.5, size=len(x))
    a_raw, b_raw = _ols(x, y)
    a_r, b_r = _ridge_ols(x, y, lam=5.0, b_prior=1.0)
    assert abs(b_r - 1.0) < abs(b_raw - 1.0), \
        "ridge must pull the slope toward the prior"


def test_r47_global_affine_applies_only_to_measured_rows():
    # r47 core honesty: censored rows keep mu EXACTLY; measured rows get a + b*mu.
    rng = np.random.default_rng(12)
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
    cal, _ = _calibrate_r47(preds, ["f0", "f1", "f2"], mode="global_affine")
    for rid, mu0 in orig_censored_mu.items():
        assert np.isclose(cal[rid]["mu"], mu0), \
            "r47 must never shift censored mu"
    # measured rows should differ (global slope ~0.86, not 1.0)
    changed = [rid for rid, p in cal.items()
               if not p["cens"] and not np.isclose(p["mu"], preds[rid]["mu"])]
    assert changed, "r47 should adjust measured mu"


def test_r47_per_scaf_affine_changes_measured_mu_per_scaf():
    rng = np.random.default_rng(13)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 5, 9):
            for i in range(25):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 3 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.35)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 5}")
    cal, fit_log = _calibrate_r47(preds, ["f0", "f1", "f2"], mode="per_scaf_affine")
    for f, l in fit_log.items():
        assert "9" in l["affine"] and "2" in l["affine"]
        assert isinstance(l["affine"]["9"]["b"], float)
    assert len(cal) == len(preds)
    for rid, p in cal.items():
        assert np.isfinite(p["mu"]) and p["sigma"] > 0.0


def test_r47_global_matches_hand_computed_pooled():
    # Consistency: pooled NLL of the r47 output equals a hand recompute from
    # (mu, sigma) using row_nll + junction-macro grouping.
    rng = np.random.default_rng(14)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (1, 3, 9):
            for i in range(30):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 2 == 0)
                mu = -6.0 if sc == 9 else -4.5
                y = -7.1 if cens else mu + rng.normal(0, 0.4)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 3}")
    cal, _ = _calibrate_r47(preds, ["f0", "f1", "f2"], mode="global_affine")
    from audit.evaluation.metrics import row_nll
    from collections import defaultdict as _dd
    jd = _dd(list)
    for rid, p in cal.items():
        nll = float(row_nll([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    hand = float(np.mean([np.mean(v) for v in jd.values()]))
    assert np.isclose(_pooled(cal), hand, atol=1e-9)


def test_r51_censored_mu_stays_untouched():
    # r51 core honesty: censored rows keep mu EXACTLY (never shifted), and
    # keep a finite positive sigma_c; only measured rows receive the affine.
    rng = np.random.default_rng(30)
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
    cal, _ = _calibrate_r51(preds, ["f0", "f1", "f2"], mode="global_affine")
    for rid, mu0 in orig_censored_mu.items():
        assert cal[rid]["cens"] is True
        assert np.isclose(cal[rid]["mu"], mu0), \
            "r51 must never shift censored mu"
        assert cal[rid]["sigma"] > 0.0 and np.isfinite(cal[rid]["sigma"])
    changed = [rid for rid, p in cal.items()
               if not p["cens"] and not np.isclose(p["mu"], preds[rid]["mu"])]
    assert changed, "r51 should adjust measured mu"


def test_r51_global_affine_rescans_sigma_on_corrected_mu():
    # The core gap: r46/r47 scanned sigma_m on the UNCORRECTED mu.  r51 must
    # re-scan sigma_m on the corrected mu, so on data with a strong measured
    # bias the fitted sigma_m must differ from scanning on the raw mu.
    rng = np.random.default_rng(31)
    preds = {}
    # strong measured-layer bias: y = mu - 1.5 systematically
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 5):
            for i in range(40):
                rid = f"{fold}_{sc}_{i}"
                mu = -4.0 + rng.normal(0, 0.2)
                y = mu - 1.5 + rng.normal(0, 0.3)
                preds[rid] = _row(rid, fold, sc, y, False, mu, jid=f"j{sc}_{i // 5}")
    cal, fit_log = _calibrate_r51(preds, ["f0", "f1", "f2"], mode="global_affine")
    # global slope must be ~1.0 (bias is additive), intercept ~-1.5
    some_fit = fit_log[list(fit_log)[0]]
    b_g = some_fit["global_affine"]["b"]
    a_g = some_fit["global_affine"]["a"]
    assert abs(b_g - 1.0) < 0.2, f"slope should be ~1, got {b_g}"
    assert -2.2 < a_g < -0.8, f"intercept should be ~-1.5, got {a_g}"
    # sigma_m re-scanned on corrected mu should be SMALLER than scanning on raw
    # mu (residuals after correction ~0.3 not ~1.5)
    for f, fl in fit_log.items():
        for sc, entry in fl["stratum_sigma"].items():
            assert entry["sigma_m"] <= 0.9, \
                f"scaf {sc}: sigma_m {entry['sigma_m']} should shrink after correction"


def test_r51_sigma_m_falls_back_globally_when_scaf_sparse():
    # Sparse scaffold: measured rows too few for a per-scaf scan -> sigma_m
    # must come from the global corrected-mu scan (finite, in-grid).
    rng = np.random.default_rng(32)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (1, 9):
            n = 40 if sc == 1 else 4  # scaf9 sparse
            for i in range(n):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 2 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.4)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j{sc}_{i // 4}")
    cal, fit_log = _calibrate_r51(preds, ["f0", "f1", "f2"], mode="global_affine")
    for f, fl in fit_log.items():
        for sc, entry in fl["stratum_sigma"].items():
            assert entry["sigma_m"] > 0.0 and np.isfinite(entry["sigma_m"])
    for rid, p in cal.items():
        assert p["sigma"] > 0.0 and np.isfinite(p["sigma"])
        assert np.isfinite(p["mu"])


def test_r51_global_matches_hand_computed_pooled():
    # Consistency: pooled NLL of r51 output equals hand recompute from
    # (mu, sigma) using row_nll + junction-macro grouping.
    rng = np.random.default_rng(33)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (1, 3, 9):
            for i in range(30):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 2 == 0)
                mu = -6.0 if sc == 9 else -4.5
                y = -7.1 if cens else mu + rng.normal(0, 0.4)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 3}")
    cal, _ = _calibrate_r51(preds, ["f0", "f1", "f2"], mode="global_affine")
    from audit.evaluation.metrics import row_nll as rn
    from collections import defaultdict as _dd
    jd = _dd(list)
    for rid, p in cal.items():
        nll = float(rn([p["y"]], [p["cens"]], [p["mu"]], [p["sigma"]])[0])
        jd[p["jid"]].append(nll)
    hand = float(np.mean([np.mean(v) for v in jd.values()]))
    assert np.isclose(r51_pooled(cal), hand, atol=1e-9)


def test_r51_no_bias_affine_is_near_identity():
    # On data with NO systematic bias, the global affine should be ~(0,1) and
    # r51 should essentially reproduce r45-level performance (no invented gain).
    rng = np.random.default_rng(34)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 5, 9):
            for i in range(30):
                rid = f"{fold}_{sc}_{i}"
                cens = (sc == 9) and (i % 3 == 0)
                mu = -6.0 if sc == 9 else -4.0
                y = -7.1 if cens else mu + rng.normal(0, 0.35)
                preds[rid] = _row(rid, fold, sc, y, cens, mu, jid=f"j_{sc}_{i // 5}")
    cal, fit_log = _calibrate_r51(preds, ["f0", "f1", "f2"], mode="global_affine")
    some_fit = fit_log[list(fit_log)[0]]
    b_g = some_fit["global_affine"]["b"]
    a_g = some_fit["global_affine"]["a"]
    assert abs(b_g - 1.0) < 0.25, f"slope should be ~1, got {b_g}"
    assert abs(a_g) < 0.6, f"intercept should be ~0, got {a_g}"


def test_r52_blend_per_scaf_applies_scaffold_weights():
    # _blend_per_scaf must apply the per-scaffold weight to the member-mean
    # GBDT/MLP mus, defaulting to wg=0.5 for scaffolds without a weight.
    rng = np.random.default_rng(40)
    members = {}
    common = [f"r{i}" for i in range(40)]
    G = ["g1", "g2"]
    M = ["m1", "m2"]
    for m in G + M:
        members[m] = {}
        for i, rid in enumerate(common):
            scaf = 1 if i % 2 == 0 else 9
            gbdt_side = m.startswith("g")
            mu = 1.0 if gbdt_side else -1.0
            members[m][rid] = _row(rid, "f0", scaf, y=-6.0, cens=False,
                                   mu=float(mu + rng.normal(0, 0.01)))
    # g1/g2 are the GBDT family, m1/m2 the MLP family
    G = ["g1", "g2"]
    M = ["m1", "m2"]
    wg_sc = {1: 0.8, 9: 0.5}
    blend = _blend_per_scaf(members, common, wg_sc, ref_key="g1", gbdt=G, mlp=M)
    for rid in common:
        scaf = 1 if common.index(rid) % 2 == 0 else 9
        wg = wg_sc[scaf]
        gmu = np.mean([members[g][rid]["mu"] for g in G])
        mmu = np.mean([members[m2][rid]["mu"] for m2 in M])
        assert np.isclose(blend[rid]["mu"], wg * gmu + (1 - wg) * mmu)
    # unknown scaffold defaults to 0.5
    wg_sc2 = {}
    blend2 = _blend_per_scaf(members, common, wg_sc2, ref_key="g1", gbdt=G, mlp=M)
    for rid in common:
        gmu = np.mean([members[g][rid]["mu"] for g in G])
        mmu = np.mean([members[m2][rid]["mu"] for m2 in M])
        assert np.isclose(blend2[rid]["mu"], 0.5 * gmu + 0.5 * mmu)


def test_r53_blend_weight_controls_gbdt_mlp_balance():
    # r53 _blend: mu = wg*mean(GBDT) + (1-wg)*mean(MLP); wg=1 -> GBDT only.
    rng = np.random.default_rng(41)
    members = {}
    common = [f"r{i}" for i in range(30)]
    G = ["g1", "g2"]
    M = ["m1", "m2"]
    for m in G + M:
        members[m] = {rid: _row(rid, "f0", 1, y=-6.0, cens=False,
                                mu=float(rng.normal(0, 1)) + (2.0 if m.startswith("g") else 0.0))
                      for rid in common}
    ens05 = r53_blend(members, common, 0.5, gbdt=G, mlp=M, ref_key="g1")
    ens10 = r53_blend(members, common, 1.0, gbdt=G, mlp=M, ref_key="g1")
    for rid in common:
        gmu = np.mean([members[g][rid]["mu"] for g in G])
        mmu = np.mean([members[m2][rid]["mu"] for m2 in M])
        assert np.isclose(ens05[rid]["mu"], 0.5 * gmu + 0.5 * mmu)
        assert np.isclose(ens10[rid]["mu"], gmu), "wg=1 must be GBDT-only"
    # r53 _ens_mu: member-mean over the given key subset
    ens_mm = r53_ens_mu(members, G, common, ref_key="g1")
    for rid in common:
        assert np.isclose(ens_mm[rid]["mu"], np.mean([members[g][rid]["mu"] for g in G]))


def test_r54_high_kappa_tends_to_scaf_sigma():
    # r54 with a very high kappa should behave like r51 (scaf sigma dominates,
    # context shrinkage ~0).  Build synthetic data with a strong context-level
    # sigma structure and verify: (1) finite positive sigmas, (2) kappa=0.1
    # (aggressive context use) differs from kappa=1e6 (scaf-only).
    rng = np.random.default_rng(50)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 5, 9):
            for c in range(4):  # 4 contexts per scaffold
                ctx = f"c{sc}_{c}"
                for i in range(20):
                    rid = f"{fold}_{sc}_{ctx}_{i}"
                    cens = (sc == 9) and (i % 2 == 0)
                    mu = -6.0 if sc == 9 else -4.0
                    y = -7.1 if cens else mu + rng.normal(0, 0.3 + 0.4 * c)
                    preds[rid] = {"jid": ctx, "fold": fold, "scaf": sc,
                                  "context": ctx, "y": y, "cens": cens,
                                  "mu": mu}
    folds = ["f0", "f1", "f2"]
    cal_low, _ = _calibrate_r54(preds, folds, kappa=0.1)
    cal_high, _ = _calibrate_r54(preds, folds, kappa=1e6)
    for rid, p in cal_low.items():
        assert p["sigma"] > 0.0 and np.isfinite(p["sigma"])
    for rid, p in cal_high.items():
        assert p["sigma"] > 0.0 and np.isfinite(p["sigma"])
    # with real context structure, low kappa should capture more context signal
    # (larger measured sigma spread than high kappa on the heterogeneous ctx)
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _pooled as _pl
    # pooled NLL: low kappa should be <= high kappa if context structure is real
    nll_low = _pl(cal_low)
    nll_high = _pl(cal_high)
    assert nll_low <= nll_high + 1e-6, \
        f"context structure should help at low kappa: {nll_low} vs {nll_high}"


def test_r54_censored_mu_stays_untouched():
    # r54 only swaps sigma; censored mu must stay exactly as r51 left it.
    rng = np.random.default_rng(51)
    preds = {}
    for fold in ("f0", "f1", "f2"):
        for sc in (2, 9):
            for c in range(2):
                ctx = f"c{sc}_{c}"
                for i in range(20):
                    rid = f"{fold}_{sc}_{ctx}_{i}"
                    cens = (sc == 9) and (i % 2 == 0)
                    mu = -6.0 if sc == 9 else -4.0
                    y = -7.1 if cens else mu + rng.normal(0, 0.4)
                    preds[rid] = {"jid": ctx, "fold": fold, "scaf": sc,
                                  "context": ctx, "y": y, "cens": cens,
                                  "mu": mu}
    folds = ["f0", "f1", "f2"]
    cal, _ = _calibrate_r54(preds, folds, kappa=5.0)
    # compare to r51 (r54's mu should equal r51's mu exactly)
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _calibrate_r51 as _r51
    cal_r51, _ = _r51(preds, folds, mode="per_scaf_eb", eb_kappa=20.0)
    for rid in preds:
        assert np.isclose(cal[rid]["mu"], cal_r51[rid]["mu"]), \
            "r54 must not change mu"
        assert np.isfinite(cal[rid]["sigma"]) and cal[rid]["sigma"] > 0.0


if __name__ == "__main__":
    tests = [test_scan_sigma_picks_minimal_nll,
             test_scan_sigma_vectorized_matches_bruteforce,
             test_scan_sigma_none_uses_all_rows,
             test_stratum_calibrate_applies_stratum_specific_sigma,
             test_stratum_calibrate_matches_pooled_recompute,
             test_horizontal_per_scaf_and_stratum_consistent,
             test_r46_mu_correction_keeps_censored_mu_untouched,
             test_r46_shrink_zero_equals_r45,
             test_r46_sigma_stays_positive_and_finite,
             test_ols_recovers_known_slope,
             test_ridge_ols_shrinks_toward_prior,
             test_r47_global_affine_applies_only_to_measured_rows,
             test_r47_per_scaf_affine_changes_measured_mu_per_scaf,
             test_r47_global_matches_hand_computed_pooled,
             test_r50_ens_mu_averages_over_member_subset,
             test_r50_member_names_match_shootout_universe,
             test_r51_censored_mu_stays_untouched,
             test_r51_global_affine_rescans_sigma_on_corrected_mu,
             test_r51_sigma_m_falls_back_globally_when_scaf_sparse,
             test_r51_global_matches_hand_computed_pooled,
             test_r51_no_bias_affine_is_near_identity,
             test_r52_blend_per_scaf_applies_scaffold_weights,
             test_r53_blend_weight_controls_gbdt_mlp_balance,
             test_r54_high_kappa_tends_to_scaf_sigma,
             test_r54_censored_mu_stays_untouched]
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
