"""Sanity check for xgboost_censored_hybrid on tiny synthetic rows (no full run)."""
import numpy as np

from audit.models.xgboost_censored_hybrid import (
    make_xgboost_censored_hybrid,
    _censored_nll_grad_hess,
    HAVE_XGB,
)

if not HAVE_XGB:
    print("XGB MISSING - skip")
    raise SystemExit(0)

rows = []
seqs = {1: "CUAG_CUAAG", 2: "CGAC_CGAC", 3: "AUGC_GCUA", 4: "UACG_ACGU",
        5: "GCUA_AUCG", 6: "AUCG_UAGC"}
motifs = {1: "0x1", 2: "0x2", 3: "0x3", 4: "0x1", 5: "0x2", 6: "0x3"}
r0 = 0
for k, s in seqs.items():
    scaf = (k % 3) + 1
    for n in range(4):
        cens = (n == 3)
        y = -8.5 + 0.2 * (r0 % 5) if not cens else -7.1
        rows.append({"source_row_id": f"R{r0:05d}", "jid": f"j{k}",
                     "motif": motifs[k], "scaf": scaf, "y": y, "cens": cens,
                     "junction_seq": s, "helix_seq": f"h{k}_{n}",
                     "symmetry_key": "_".join(reversed(s.split("_")))})
        r0 += 1

# grad/hess sanity: censored row with low mu (way below cap) => larger gradient
mu = np.array([-3.0, 3.0])
y = np.array([-7.1, -7.1])
cens = np.array([True, True])
g, h = _censored_nll_grad_hess(mu, y, cens)
print("censored grad:", g, "hess:", h)
assert g[0] < g[1], "higher mu (further above cap) should have smaller magnitude gradient"

fit, predict = make_xgboost_censored_hybrid(n_estimators=100, max_depth=3)
model = fit(rows)
print("model kind:", model["kind"], "best_iter:", model["best_iteration"])
te = [{"source_row_id": "R999", "jid": "j1", "motif": "0x1", "scaf": 1,
       "y": -6.0, "cens": 0, "junction_seq": "CUAG_CUAAG",
       "helix_seq": "h1_0", "symmetry_key": "CUAAG_CUAG"}]
mu_o, sigma, cp, support, abstain = predict(model, te)
print("predict mu:", mu_o, "sigma:", sigma, "cp:", cp, "support:", support, "abstain:", abstain)
assert np.all(np.isfinite(mu_o)) and np.allclose(sigma, 0.7)
assert cp.min() >= 0.0 and cp.max() <= 1.0
assert support.dtype == bool and abstain.dtype == bool

# unseen scaffold abstains
te2 = [{"source_row_id": "R998", "jid": "j99", "motif": "0x1", "scaf": 99,
        "y": -6.0, "cens": 0, "junction_seq": "AAAA_BBBB",
        "helix_seq": "h99", "symmetry_key": "AAAA_BBBB"}]
mu2, s2, cp2, su2, ab2 = predict(model, te2)
assert bool(ab2[0]) is True and bool(su2[0]) is False
print("ALL SANITY CHECKS PASSED")
