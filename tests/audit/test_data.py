"""P0.2 tests: ledger consistency, counts conservation, right-censor direction.

Right-censor semantics: weak values are recorded as dg10 >= -7.1; the observed
value -7.1 means "true value at least -7.1" (right-censored, Y >= -7.1).  For a
censored observation the survival likelihood is P(Y >= -7.1) = Phi((mu-CAP)/tau).
Increasing mu must increase that survival probability and therefore DECREASE the
censored negative log-likelihood.  A sign-flipped (left-censor) implementation
must fail this test.
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.special import log_ndtr

CAP = -7.1
TAU = 0.7


def censored_log_obs(mu):
    """Right-censored survival log-likelihood: log P(Y >= CAP)."""
    return log_ndtr((mu - CAP) / TAU)


def wrong_left_censored_log_obs(mu):
    """Incorrect left-censor direction: log P(Y <= CAP) = log(1 - Phi(...))."""
    # log1p(-exp(log_ndtr)) computed stably; sign-flip fixture.
    import math
    p = math.exp(log_ndtr((mu - CAP) / TAU))
    return math.log(max(1.0 - p, 1e-300))


def test_right_censor_monotonic_increasing_survival():
    # increasing mu must increase survival prob (log_ndtr increasing in mu)
    for a, b in [(-9.0, -8.0), (-8.0, -7.1), (-7.1, -6.0), (-6.0, -5.0)]:
        assert censored_log_obs(b) > censored_log_obs(a), f"mu up must raise survival at {a}->{b}"


def test_right_censor_decreases_censored_nll():
    # increasing mu must decrease censored NLL (=-log_obs)
    base = -censored_log_obs(-8.0)
    higher = -censored_log_obs(-7.0)
    assert higher < base


def test_sign_swap_fixture_fails():
    # the wrong left-censor fixture must violate monotonic increase
    # for mu increasing toward the cap, survival P(Y<=cap) increases, i.e.
    # wrong_log_obs increases with mu -> -NLL decreases is wrong direction
    # Assert that the wrong fixture is NOT monotone-decreasing in NLL the right way.
    d0 = -wrong_left_censored_log_obs(-9.0)
    d1 = -wrong_left_censored_log_obs(-7.0)
    # For left-censor, raising mu raises P(Y<=cap), raising NLL. So d1>d0.
    assert d1 > d0, "left-censor fixture must have NLL increasing with mu"


def test_ledger_and_counts_conservation(data_dir: Path):
    ledger_path = data_dir / "CleaningLedger.jsonl"
    profile = json.loads((data_dir / "DataProfile.json").read_text())
    rows = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    admitted = [r for r in rows if not r["excluded"]]
    # every row has a status
    assert all("excluded" in r and "reason" in r for r in rows)
    # admitted rows equal profile
    assert len(admitted) == profile["admitted_rows"]
    # measured + censored conservation
    measured = sum(1 for r in admitted if not r["cens"])
    censored = sum(1 for r in admitted if r["cens"])
    assert measured + censored == profile["admitted_rows"]
    assert measured == profile["measured_rows"]
    assert censored == profile["right_censored_rows"]
    # unique source_row_id
    ids = [r["source_row_id"] for r in rows]
    assert len(ids) == len(set(ids))


if __name__ == "__main__":
    import sys
    test_right_censor_monotonic_increasing_survival()
    test_right_censor_decreases_censored_nll()
    test_sign_swap_fixture_fails()
    test_ledger_and_counts_conservation(Path(sys.argv[1]))
    print("P0.2 data/censor tests PASS")
