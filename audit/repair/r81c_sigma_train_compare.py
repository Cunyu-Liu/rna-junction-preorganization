"""r81c: proper side-by-side held-out comparison (junction-macro measured NLL)
between ORIGINAL t7 and per-scaf-sigma-trained t7 on the same smoke folds.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audit.evaluation.metrics import row_nll
from audit.repair.r51_joint_mu_affine_sigma_rescan import (
    _load, _elig, _by_rid, _scan_sigma, GRID,
    R33, R34, R35, R24, R33_LEDGER, R34_LEDGER, R35_LEDGER, R24_LEDGERS,
    XGB, XGB_S99, XGB_S2026, XGB_LR03, T7, T7_S99, T7_S2026, ALL_MEMBERS,
)
from audit.repair.r62_decoupled_frozen import _calibrate_r62

GBDT = [XGB, XGB_LR03, XGB_S99, XGB_S2026]
MLP = [T7, T7_S99, T7_S2026]


def _jm_meas(mus_by_rid, sig_m, rids):
    by_jid = defaultdict(list)
    for rid in rids:
        p = mus_by_rid[rid]
        if p["cens"]:
            continue
        sc = int(p["scaf"])
        nll = float(row_nll([p["y"]], [False], [p["mu"]],
                            [sig_m.get(sc, 0.55)])[0])
        by_jid[str(p["jid"])].append(nll)
    return float(np.mean([np.mean(v) for v in by_jid.values()])) if by_jid else None


def main():
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
    from audit.repair.r62_decoupled_frozen import _calibrate_r62
    cal62, _ = _calibrate_r62(ens, folds, kappa=1.0, min_meas=3)
    by_scaf_m = defaultdict(dict)
    for rid, p in cal62.items():
        if not p["cens"]:
            by_scaf_m[int(p["scaf"])][rid] = p
    sig_m = {}
    for sc, rows in by_scaf_m.items():
        if len(rows) >= 15:
            s, _ = _scan_sigma(rows, cens_mask=False, grid=GRID)
            sig_m[sc] = s

    smoke_folds = ["e:AAAC_GAAC", "e:AAAG_CAAG"]
    for f in smoke_folds:
        rids = [r for r in ens if ens[r]["fold"] == f]
        orig = _jm_meas(members[T7], sig_m, rids)
        # new sig-trained mu: need to retrain here (r81 didn't save); retrain inline
        import torch
        from audit.models.nonlinear_mlp_hybrid import (
            _nuisance_basis, _MLP, CAP, WEIGHT_DECAY, LR, MAX_EPOCHS,
            PATIENCE, LOSS_TOL, SEED,
        )
        from audit.benchmark.vienna_extended_features import (
            build_raw_by_jid, fit_scaler, transform,
        )
        from audit.repair.fold_loader import build_joint_edit_context_folds
        from audit.data.audit_dataset import audit_dataset
        from pathlib import Path as P
        cfg_src = ("/mnt/cunyuliu/rna_junction_audit_20260807T090244Z/source/"
                   "tecto_v111_canonical_records.jsonl")
        _, admitted, *_ = audit_dataset(P(cfg_src))
        rid2row = {str(r["source_row_id"]): r for r in admitted}
        spec = [s for s in build_joint_edit_context_folds(admitted) if s.fold == f][0]
        train_rows = [rid2row[x] for x in sorted(spec.train_ids) if x in rid2row]

        def _censored_nll_sig(mu, y, cens, sig):
            z = (y - mu) / sig
            nll_m = 0.5 * z * z
            a = (mu - CAP) / sig
            log_phi = torch.special.log_ndtr(a.clamp(min=-30.0, max=30.0))
            nll_c = -log_phi
            return torch.where(cens, nll_c, nll_m).mean()

        device = "cuda:0"
        torch.manual_seed(SEED)
        motifs = sorted({str(r["motif"]) for r in train_rows})
        scafs = sorted({int(r["scaf"]) for r in train_rows})
        Xn = _nuisance_basis(train_rows, motifs, scafs)
        tr_jids = sorted({str(r["jid"]) for r in train_rows})
        by_jid = build_raw_by_jid(train_rows)
        mean, sd = fit_scaler(tr_jids, by_jid)
        Xv = np.zeros((len(train_rows), len(mean)))
        for i, r in enumerate(train_rows):
            Xv[i] = transform([str(r["jid"])], by_jid, mean, sd)[0]
        X = np.hstack([Xn, Xv])
        y = np.asarray([r["y"] for r in train_rows], dtype=float)
        cens = np.asarray([r["cens"] for r in train_rows], dtype=bool)
        # per-scaf sigma vector
        from audit.repair.r51_joint_mu_affine_sigma_rescan import _scan_sigma as _ss2
        _ = _ss2
        sig_v = np.asarray([
            sig_m.get(int(r["scaf"]), 0.55) if not r["cens"] else 0.5
            for r in train_rows], dtype=float)
        net = _MLP(X.shape[1], hidden=(64, 32)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        yt = torch.tensor(y, dtype=torch.float32, device=device)
        ct = torch.tensor(cens, dtype=torch.bool, device=device)
        st = torch.tensor(sig_v, dtype=torch.float32, device=device)
        best_loss, best_state, esb = float("inf"), None, 0
        n = Xt.shape[0]
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
            fl = float(np.mean(losses))
            if fl < best_loss - LOSS_TOL:
                best_loss, best_state, esb = fl, {k: v.detach().cpu().clone()
                                                   for k, v in net.state_dict().items()}, 0
            else:
                esb += 1
                if esb >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        # predict test
        test_ids = sorted(spec.test_ids)
        test_rows = [rid2row[x] for x in test_ids if x in rid2row]
        Xn2 = _nuisance_basis(test_rows, motifs, scafs)
        by_jid2 = build_raw_by_jid(test_rows)
        Xv2 = np.zeros((len(test_rows), len(mean)))
        for i, r in enumerate(test_rows):
            Xv2[i] = transform([str(r["jid"])], by_jid2, mean, sd)[0]
        X2 = np.hstack([Xn2, Xv2])
        with torch.no_grad():
            mu2 = net(torch.tensor(X2, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
        # score new member junction-macro
        by_jid_new = defaultdict(list)
        for i, r in enumerate(test_rows):
            if r["cens"]:
                continue
            sc = int(r["scaf"])
            nll = float(row_nll([r["y"]], [False], [float(mu2[i])],
                                [sig_m.get(sc, 0.55)])[0])
            by_jid_new[str(r["jid"])].append(nll)
        new = float(np.mean([np.mean(v) for v in by_jid_new.values()]))
        print(f"fold {f}: ORIG t7={orig:.4f}  SIG-train t7={new:.4f}  "
              f"delta={new-orig:+.4f}")


if __name__ == "__main__":
    main()
