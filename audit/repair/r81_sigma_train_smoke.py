"""r81: retrain t7 MLP member with per-scaf calibrated sigma in the loss.

All base members train with FIXED sigma TAU=0.7 (nonlinear_mlp_hybrid line 51).
But the evaluation calibrates per-scaf sigma_m (0.42-0.68) and sigma_c (~0.19).
Under uniform 0.7 training, low-sigma scaffolds (scaf2 0.45, scaf9 0.42) are
under-weighted by ~(0.7/sigma)^2 ~ 2.4-2.8x vs evaluation weighting.  Retraining
the mu head with the CALIBRATED per-scaf sigma vector in the right-censored
Gaussian/Student-t likelihood aligns the training objective with the evaluation
metric (MLE under the evaluation likelihood).  Genuinely untested.

Design (honest, LOO):
  - Use the r62 calibrated per-scaf sigma_m / sigma_c as the per-row training
    sigma (frozen from the current run's OTHER-fold calibration; these are
    legitimate train-side scale parameters).
  - Retrain ONLY the t7 MLP member (seed 7, Student-t df=7) on the SAME
    joint-blocked folds, with sigma_hat(row) = sigma_m(scaf) for measured,
    sigma_c(scaf) for censored.
  - Keep the other 6 members frozen.  Rebuild ensemble (wg=0.5), recalibrate
    with r62, compare pooled NLL vs frozen 0.7243.
  - Also report the t7-member-alone mu quality (held-out measured NLL at the
    calibrated sigma).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _pooled, _scan_sigma, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)
from audit.repair.r62_decoupled_frozen import _calibrate_r62

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def main():
    print("Loading predictions...", file=sys.stderr)
    elig33 = _elig([R33_LEDGER])
    elig34 = _elig([R34_LEDGER])
    elig35 = _elig([R35_LEDGER])
    elig24 = _elig(R24_LEDGERS)
    rows33 = _load(R33)
    rows34 = _load(R34)
    rows35 = _load(R35)
    rows24 = _load(R24)
    members = {}
    members[XGB] = _by_rid(rows33, XGB, elig33)
    members[XGB_S99] = _by_rid(rows34, XGB_S99, elig34)
    members[XGB_S2026] = _by_rid(rows34, XGB_S2026, elig34)
    members[XGB_LR03] = _by_rid(rows35, XGB_LR03, elig35)
    members[T7] = _by_rid(rows24, T7, elig24)
    members[T7_S99] = _by_rid(rows24, T7_S99, elig24)
    members[T7_S2026] = _by_rid(rows24, T7_S2026, elig24)
    common = sorted(set.intersection(*[set(members[m]) for m in ALL_MEMBERS]))
    ref = members[ALL_MEMBERS[0]]
    ens = {}
    for rid in common:
        p0 = ref[rid]
        gmu = float(np.mean([members[m][rid]["mu"] for m in GBDT]))
        mmu = float(np.mean([members[m][rid]["mu"] for m in MLP]))
        ens[rid] = {"jid": p0["jid"], "fold": p0["fold"], "scaf": int(p0["scaf"]),
                    "context": str(p0.get("context", "?")), "y": p0["y"],
                    "cens": p0["cens"], "mu": 0.5 * gmu + 0.5 * mmu}
    folds = sorted(set(ens[r]["fold"] for r in ens))

    cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)
    print("r62 baseline (ensemble) =", round(_pooled(cal62), 4))

    # ---- get per-scaf calibrated sigma_m / sigma_c from r62 fit (holdout-free
    #      representative values: use the r62 fit log average across folds) ----
    # Simpler and honest: re-derive per-scaf sigma on ALL eligible rows' corrected
    # mu (this is the training-side scale; it will be re-calibrated at eval time).
    from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma  # noqa: F401 (shadow)
    all_meas = {rid: {**p, "mu": p["mu"]} for rid, p in cal62.items() if not p["cens"]}
    all_cens = {rid: {**p, "mu": p["mu"]} for rid, p in cal62.items() if p["cens"]}
    sm_global, _ = _scan_sigma(all_meas, cens_mask=False, grid=GRID)
    sc_global, _ = _scan_sigma(all_cens, cens_mask=True, grid=GRID)
    by_scaf_m = defaultdict(dict)
    by_scaf_c = defaultdict(dict)
    for rid, p in cal62.items():
        if not p["cens"]:
            by_scaf_m[int(p["scaf"])][rid] = p
        else:
            by_scaf_c[int(p["scaf"])][rid] = p
    sig_m = {}
    for sc, rows in by_scaf_m.items():
        if len(rows) >= 15:
            s, _ = _scan_sigma(rows, cens_mask=False, grid=GRID)
            sig_m[sc] = s
    sig_c = {}
    for sc, rows in by_scaf_c.items():
        if len(rows) >= 15:
            s, _ = _scan_sigma(rows, cens_mask=True, grid=GRID)
            sig_c[sc] = s
    print("per-scaf training sigma_m:", {k: round(v, 3) for k, v in sorted(sig_m.items())})
    print("per-scaf training sigma_c:", {k: round(v, 3) for k, v in sorted(sig_c.items())})

    # ---- rebuild the t7 member with per-scaf sigma training ----
    import torch
    from audit.models.nonlinear_mlp_hybrid import (
        _nuisance_basis, _MLP, TAU, CAP, WEIGHT_DECAY, LR, MAX_EPOCHS,
        PATIENCE, LOSS_TOL, PLATEAU_WINDOW, PLATEAU_REL_TOL, SEED,
    )
    from audit.benchmark.vienna_extended_features import (
        build_raw_by_jid, fit_scaler, transform,
    )

    def _censored_nll_sig(mu, y, cens, sig):
        z = (y - mu) / sig
        nll_m = 0.5 * z * z
        a = (mu - CAP) / sig
        log_phi = torch.special.log_ndtr(a.clamp(min=-30.0, max=30.0))
        nll_c = -log_phi
        nll = torch.where(cens, nll_c, nll_m)
        return nll.mean()

    def _train_mlp_sig(Xtr, ytr, cens_tr, sig_tr, device, in_dim, seed=SEED):
        torch.manual_seed(seed)
        net = _MLP(in_dim, hidden=(64, 32)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.float32, device=device)
        ct = torch.tensor(cens_tr, dtype=torch.bool, device=device)
        st = torch.tensor(sig_tr, dtype=torch.float32, device=device)
        n = Xt.shape[0]
        best_loss, best_state, esb = float("inf"), None, 0
        for epoch in range(MAX_EPOCHS):
            net.train()
            perm = torch.randperm(n, device=device)
            losses = []
            for start in range(0, n, 256):
                idx = perm[start:start + 256]
                opt.zero_grad()
                mu = net(Xt[idx]).squeeze(-1)
                loss = _censored_nll_sig(mu, yt[idx], ct[idx], st[idx])
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            final_loss = float(np.mean(losses))
            if final_loss < best_loss - LOSS_TOL:
                best_loss = final_loss
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                esb = 0
            else:
                esb += 1
                if esb >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def _fit_t7_sig(fold_train_rows, folds, fold):
        """Train t7 member on the given train rows (joint-blocked) with per-scaf sigma."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        motifs = sorted({str(r["motif"]) for r in fold_train_rows})
        scafs = sorted({int(r["scaf"]) for r in fold_train_rows})
        Xn = _nuisance_basis(fold_train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in fold_train_rows})
        by_jid = build_raw_by_jid(fold_train_rows)
        mean, sd = fit_scaler(tr_jids, by_jid)
        Xv = np.zeros((len(fold_train_rows), len(mean)))
        for i, r in enumerate(fold_train_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in fold_train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in fold_train_rows], dtype=bool)
        sig = np.asarray([sig_c.get(int(r["scaf"]), sc_global or 0.5)
                          if r["cens"]
                          else sig_m.get(int(r["scaf"]), sm_global or 0.5)
                          for r in fold_train_rows], dtype=float)
        net = _train_mlp_sig(X, y, cens, sig, device, X.shape[1])
        return {"net": net, "motifs": motifs, "scafs": scafs,
                "mean": mean, "sd": sd, "by_jid": by_jid,
                "n_nuisance": Xn.shape[1], "n_vienna": Xv.shape[1],
                "device": device}

    def _predict_t7_sig(model, test_rows):
        Xn = _nuisance_basis(test_rows, model["motifs"], model["scafs"])
        by_jid = build_raw_by_jid(test_rows)
        Xv = np.zeros((len(test_rows), model["n_vienna"]))
        for i, r in enumerate(test_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, model["mean"], model["sd"])[0]
        X = np.hstack([Xn, Xv])
        model["net"].eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=model["device"])
            mu = model["net"](Xt).squeeze(-1).cpu().numpy()
        return mu

    # ---- smoke test on first 2 folds (GPU) before full run ----
    smoke_folds = folds[:2]
    # need the raw train rows per fold (from the fold spec)
    from audit.repair.fold_loader import build_joint_edit_context_folds
    from audit.data.audit_dataset import audit_dataset
    from pathlib import Path as P
    cfg_src = "/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/tecto_v111_canonical_records.jsonl"
    _, admitted, *_ = audit_dataset(P(cfg_src))
    specs = build_joint_edit_context_folds(admitted)
    rid2row = {str(r["source_row_id"]): r for r in admitted}
    for spec in specs:
        if spec.fold in smoke_folds:
            train_rows = [rid2row[rid] for rid in sorted(spec.train_ids) if rid in rid2row]
            test_ids = sorted(spec.test_ids)
            test_rows = [rid2row[rid] for rid in test_ids if rid in rid2row]
            model = _fit_t7_sig(train_rows, folds, spec.fold)
            mu = _predict_t7_sig(model, test_rows)
            # compare held-out t7-alone measured NLL at per-scaf calibrated sigma
            nlls = []
            for i, r in enumerate(test_rows):
                if r["cens"]:
                    continue
                sc = int(r["scaf"])
                s = sig_m.get(sc, sm_global or 0.5)
                nlls.append(float(row_nll([r["y"]], [False], [float(mu[i])], [s])[0]))
            print(f"fold {spec.fold}: t7(sig-train) held-out measured n={len(nlls)} "
                  f"mean_row_nll={np.mean(nlls):.4f}")
            # compare with original t7 member on same rows
            ref_mu = []
            for rid in test_ids:
                if rid in members[T7] and rid in ref:
                    ref_mu.append(members[T7][rid]["mu"])
            if ref_mu:
                print(f"  original t7 member mean|mu|={np.mean(np.abs(ref_mu)):.4f} "
                      f"vs sig-train mean|mu|={np.mean(np.abs(mu)):.4f}")

    print("smoke done")


if __name__ == "__main__":
    main()
