#!/usr/bin/env python3
"""Controlled v1.31 probe: joint latent-operator Tobit model.

This is deliberately a small, sequence-only model.  Unlike the historical
panel-factor/EM probes it does not turn censored observations into pseudo
labels and does not fit a junction residual in a first stage.  It integrates
one latent junction functional over a right-censored likelihood with a
fixed-size Gauss--Hermite rule, then reports both a common-tau score and the
latent-predictive score.  The file is exploratory until a parent-linked
contract run closes all gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

import hetero_censored_benchmark as hetero
import sequence_fingerprint_factor_benchmark as seq

CAP = -7.1
TAU = 0.7
SIGMA_Q = 1.0
SEED = 20260807
GH_NODES = 24
RIDGE = 100.0
SLOPE_RIDGE = 5.0


def split_rows(rows, field, held):
    held = {str(x) for x in held}
    return [r for r in rows if str(r[field]) not in held], [r for r in rows if str(r[field]) in held]


def outer_blocks(rows, split, seed):
    if split == "symmetry_5fold":
        field = "symmetry_key"
        groups = sorted({str(r[field]) for r in rows})
    elif split == "edit_component_5fold":
        seq.edit_component_labels(rows, max_distance=1)
        field = "edit_component"
        groups = sorted({str(r[field]) for r in rows})
    elif split == "motif_lomo":
        field = "motif"
        groups = sorted({str(r[field]) for r in rows})
        return [{g} for g in groups], field
    elif split == "scaffold_lomo":
        field = "scaf"
        groups = sorted({str(r[field]) for r in rows})
        return [{g} for g in groups], field
    else:
        raise ValueError(split)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    return [set(x.tolist()) for x in np.array_split(np.asarray(groups, dtype=object), 5) if len(x)], field


def feature_raw(row):
    # Only junction sequence features are used in the primary candidate.  The
    # physical/fold fields are scaffold-dependent in this source and therefore
    # are retained for a later pre-registered sensitivity, not smuggled into
    # the latent junction function.
    return seq.seq_features(row["junction_seq"])


def feature_state(rows):
    by = {}
    for r in rows:
        by.setdefault(str(r["jid"]), str(r["junction_seq"]))
    ids = sorted(by)
    x = np.asarray([feature_raw({"junction_seq": by[j]}) for j in ids], dtype=float)
    mean = np.mean(x, axis=0) if len(x) else np.zeros(0)
    sd = np.std(x, axis=0) if len(x) else np.ones(0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    return {"ids": ids, "by_jid": by, "mean": mean, "sd": sd}


def transform(rows, state):
    x = np.asarray([feature_raw(r) for r in rows], dtype=float)
    return (x - state["mean"]) / state["sd"]


def log_obs(y, cens, mu, sigma):
    y = np.asarray(y, dtype=float)
    cens = np.asarray(cens, dtype=bool)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 0.05)
    out = np.zeros(len(y), dtype=float)
    measured = ~cens
    if measured.any():
        z = (y[measured] - mu[measured]) / sigma[measured]
        out[measured] = -0.5 * math.log(2.0 * math.pi) - np.log(sigma[measured]) - 0.5 * z * z
    if cens.any():
        out[cens] = log_ndtr((mu[cens] - CAP) / sigma[cens])
    return out


def build_panel(train):
    jids = sorted({str(r["jid"]) for r in train})
    scaffolds = sorted({int(r["scaf"]) for r in train})
    ji = {j: i for i, j in enumerate(jids)}
    si = {s: i for i, s in enumerate(scaffolds)}
    rows = [[] for _ in jids]
    for r in train:
        rows[ji[str(r["jid"])]].append(r)
    # Array form is used by the objective so finite-difference optimization
    # does not repeatedly walk Python dictionaries for every GH node.
    flat_j = []
    flat_s = []
    flat_y = []
    flat_c = []
    for j, group in enumerate(rows):
        for r in group:
            flat_j.append(j)
            flat_s.append(si[int(r["scaf"])])
            flat_y.append(float(r["y"]))
            flat_c.append(bool(r["cens"]))
    return {
        "jids": jids, "scaffolds": scaffolds, "si": si, "rows": rows,
        "flat_j": np.asarray(flat_j, dtype=int),
        "flat_s": np.asarray(flat_s, dtype=int),
        "flat_y": np.asarray(flat_y, dtype=float),
        "flat_c": np.asarray(flat_c, dtype=bool),
    }


def unpack(params, n_features, n_scaffolds, ref_index):
    theta = np.asarray(params[:n_features], dtype=float)
    a = np.asarray(params[n_features:n_features + n_scaffolds], dtype=float)
    log_b = np.zeros(n_scaffolds, dtype=float)
    cursor = n_features + n_scaffolds
    for s in range(n_scaffolds):
        if s == ref_index:
            log_b[s] = 0.0
        else:
            log_b[s] = float(params[cursor])
            cursor += 1
    return theta, a, np.exp(np.clip(log_b, -2.0, 2.0))


def initial_params(panel, x_by_jid, ref_index):
    n_features = x_by_jid.shape[1]
    n_scaffolds = len(panel["scaffolds"])
    a = np.zeros(n_scaffolds, dtype=float)
    all_measured = []
    for s in range(n_scaffolds):
        vals = [r["y"] for group in panel["rows"] for r in group if int(r["scaf"]) == panel["scaffolds"][s] and not r["cens"]]
        a[s] = float(np.mean(vals)) if vals else (float(np.mean(all_measured)) if all_measured else -9.0)
        all_measured.extend(vals)
    theta = np.zeros(n_features, dtype=float)
    free = np.zeros(max(n_scaffolds - 1, 0), dtype=float)
    return np.r_[theta, a, free]


def fit_joint(train, ridge=RIDGE, slope_ridge=SLOPE_RIDGE, gh_nodes=GH_NODES,
              maxiter=180, seed=SEED):
    panel = build_panel(train)
    state = feature_state(train)
    x_by_jid = np.asarray([(state["by_jid"][j] and transform([{"junction_seq": state["by_jid"][j]}], state)[0]) for j in panel["jids"]], dtype=float)
    nodes, weights = np.polynomial.hermite.hermgauss(int(gh_nodes))
    log_weights = np.log(np.maximum(weights, 1e-300)) - 0.5 * math.log(math.pi)
    n_features = int(x_by_jid.shape[1])
    n_scaffolds = len(panel["scaffolds"])
    ref_index = panel["scaffolds"].index(2) if 2 in panel["scaffolds"] else 0
    p0 = initial_params(panel, x_by_jid, ref_index)
    lo = np.r_[np.full(n_features, -4.0), np.full(n_scaffolds, -18.0), np.full(max(n_scaffolds - 1, 0), -1.5)]
    hi = np.r_[np.full(n_features, 4.0), np.full(n_scaffolds, -5.0), np.full(max(n_scaffolds - 1, 0), 1.5)]
    bounds = list(zip(lo, hi))

    def objective_and_grad(params):
        theta, a, b = unpack(params, n_features, n_scaffolds, ref_index)
        f = np.asarray(x_by_jid @ theta, dtype=float)
        q = f[:, None] + math.sqrt(2.0 * SIGMA_Q) * nodes[None, :]
        mu = a[panel["flat_s"]][:, None] + b[panel["flat_s"]][:, None] * q[panel["flat_j"]]
        y = panel["flat_y"][:, None]
        c = panel["flat_c"][:, None]
        z = (y - mu) / TAU
        ll = np.where(c,
                      log_ndtr((mu - CAP) / TAU),
                      -0.5 * math.log(2.0 * math.pi) - math.log(TAU) - 0.5 * z * z)
        # Sum each row's likelihood into a junction-by-node matrix.
        grouped = np.zeros((len(panel["jids"]), len(nodes)), dtype=float)
        for k in range(len(nodes)):
            np.add.at(grouped[:, k], panel["flat_j"], ll[:, k])
        log_marginal = logsumexp(grouped + log_weights[None, :], axis=1)
        total = -float(np.sum(log_marginal))
        posterior = np.exp(grouped + log_weights[None, :] - log_marginal[:, None])
        # Analytic derivative of the marginal log likelihood.  This is the
        # main numerical guard: finite-difference optimization over 63 sequence
        # coefficients was too slow and could stop before a stationary point.
        score_mu = np.where(
            c,
            np.exp(-0.5 * ((mu - CAP) / TAU) ** 2 - 0.5 * math.log(2.0 * math.pi)
                   - log_ndtr((mu - CAP) / TAU)) / TAU,
            (y - mu) / (TAU * TAU),
        )
        n_groups = len(panel["jids"])
        score_jk = np.zeros_like(grouped)
        score_b_jk = np.zeros_like(grouped)
        score_a = np.zeros(n_scaffolds, dtype=float)
        score_logb = np.zeros(n_scaffolds, dtype=float)
        for k in range(len(nodes)):
            np.add.at(score_jk[:, k], panel["flat_j"], score_mu[:, k] * b[panel["flat_s"]])
            np.add.at(score_b_jk[:, k], panel["flat_j"], score_mu[:, k])
            np.add.at(score_a, panel["flat_s"], posterior[panel["flat_j"], k] * score_mu[:, k])
            np.add.at(score_logb, panel["flat_s"], posterior[panel["flat_j"], k]
                      * score_mu[:, k] * b[panel["flat_s"]] * q[panel["flat_j"], k])
        # score_jk already contains b * dlogL/dmu; posterior integrates it.
        grad_f = np.sum(posterior * score_jk, axis=1)
        grad_theta = -(x_by_jid.T @ grad_f) / max(n_groups, 1)
        grad_a = -score_a / max(n_groups, 1)
        grad_logb = -score_logb / max(n_groups, 1)
        # Shrink sequence function and scaffold slopes.  The reference slope
        # is fixed to one, making the latent scale identifiable.
        total += 0.5 * float(ridge) * float(np.dot(theta, theta))
        free_slopes = [i for i in range(n_scaffolds) if i != ref_index]
        total += 0.5 * float(slope_ridge) * sum(float(math.log(b[i]) ** 2) for i in free_slopes)
        grad_theta += float(ridge) * theta
        grad_a += 0.0
        grad_free = []
        for i in range(n_scaffolds):
            if i == ref_index:
                continue
            grad_free.append(float(grad_logb[i] + slope_ridge * math.log(b[i])))
        grad = np.r_[grad_theta, grad_a, np.asarray(grad_free, dtype=float)]
        return total / max(n_groups, 1), grad / max(n_groups, 1)

    def objective(params):
        return objective_and_grad(params)[0]

    def jac(params):
        return objective_and_grad(params)[1]

    result = minimize(objective, p0, jac=jac, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-6, "maxls": 30})
    theta, a, b = unpack(result.x, n_features, n_scaffolds, ref_index)
    return {
        "panel": panel, "state": state, "theta": theta, "a": a, "b": b,
        "ref_scaffold": int(panel["scaffolds"][ref_index]),
        "sigma_q": float(SIGMA_Q), "tau": float(TAU), "ridge": float(ridge),
        "slope_ridge": float(slope_ridge), "gh_nodes": int(gh_nodes),
        "success": bool(result.success), "message": str(result.message),
        "objective": float(result.fun), "nit": int(result.nit),
        "n_features": n_features, "n_scaffolds": n_scaffolds,
    }


def predict(model, rows):
    x = transform(rows, model["state"])
    f = np.asarray(x @ model["theta"], dtype=float)
    si = model["panel"]["si"]
    mu = np.zeros(len(rows), dtype=float)
    sigma = np.full(len(rows), TAU, dtype=float)
    support = {"operator_supported": 0, "operator_abstain": 0, "n_rows": int(len(rows))}
    for i, r in enumerate(rows):
        s = si.get(int(r["scaf"]))
        if s is None:
            support["operator_abstain"] += 1
            continue
        support["operator_supported"] += 1
        mu[i] = float(model["a"][s] + model["b"][s] * f[i])
        # Marginal predictive uncertainty is reported separately.  The common
        # tau vector is retained as the primary matched-score counterfactual.
        sigma[i] = float(math.sqrt(TAU * TAU + model["b"][s] ** 2 * SIGMA_Q ** 2))
    return mu, sigma, support


def group_nll(rows, mu, sigma):
    losses = -log_obs([r["y"] for r in rows], [r["cens"] for r in rows], mu, sigma)
    by = defaultdict(list)
    for r, loss in zip(rows, losses):
        by[str(r["jid"])].append(float(loss))
    return {k: float(np.mean(v)) for k, v in by.items()}


def paired(base, cand, seed, n_boot=2000):
    keys = sorted(set(base) & set(cand))
    if not keys:
        return {"n_groups": 0, "mean_gain": None, "ci95": [None, None], "positive_fraction": None}
    b = np.asarray([base[k] for k in keys], dtype=float)
    c = np.asarray([cand[k] for k in keys], dtype=float)
    gain = (b - c) / np.maximum(np.abs(b), 1e-12)
    rng = np.random.default_rng(int(seed))
    boot = gain[rng.integers(0, len(gain), size=(int(n_boot), len(gain)))].mean(axis=1)
    return {"n_groups": int(len(gain)), "mean_gain": float(np.mean(gain)),
            "median_gain": float(np.median(gain)), "positive_fraction": float(np.mean(gain > 0)),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def evaluate(train, test, seed, maxiter=180):
    model = fit_joint(train, maxiter=maxiter, seed=seed)
    mu, predictive_sigma, support = predict(model, test)
    baseline_model = hetero.fit_beta(train, mode="none", tau=TAU, ridge=1.0, balanced=True, use_err=False)
    base_mu = np.asarray(seq.predict_m0_fast(baseline_model, test), dtype=float)
    base_j = group_nll(test, base_mu, np.full(len(test), TAU))
    cand_common = group_nll(test, mu, np.full(len(test), TAU))
    cand_predictive = group_nll(test, mu, predictive_sigma)
    measured = [not r["cens"] for r in test]
    censored = [bool(r["cens"]) for r in test]
    fit_meta = {k: v for k, v in model.items() if k not in {"panel", "state", "theta"}}
    for key in ("a", "b"):
        if key in fit_meta:
            fit_meta[key] = np.asarray(fit_meta[key], dtype=float).tolist()
    out = {
        "fit": fit_meta,
        "baseline_macro_junction_nll": float(np.mean(list(base_j.values()))) if base_j else None,
        "candidate_common_tau_macro_junction_nll": float(np.mean(list(cand_common.values()))) if cand_common else None,
        "candidate_predictive_macro_junction_nll": float(np.mean(list(cand_predictive.values()))) if cand_predictive else None,
        "common_tau_gain_vs_scaffold": paired(base_j, cand_common, int(seed) + 100),
        "predictive_gain_vs_scaffold": paired(base_j, cand_predictive, int(seed) + 200),
        "measured_common_tau_gain": paired(
            group_nll([r for r, keep in zip(test, measured) if keep], base_mu[np.asarray(measured)], np.full(sum(measured), TAU)),
            group_nll([r for r, keep in zip(test, measured) if keep], mu[np.asarray(measured)], np.full(sum(measured), TAU)), int(seed) + 300),
        "censored_common_tau_gain": paired(
            group_nll([r for r, keep in zip(test, censored) if keep], base_mu[np.asarray(censored)], np.full(sum(censored), TAU)),
            group_nll([r for r, keep in zip(test, censored) if keep], mu[np.asarray(censored)], np.full(sum(censored), TAU)), int(seed) + 400),
        "support": support,
        "n_test_rows": int(len(test)), "n_test_junctions": int(len({str(r["jid"]) for r in test})),
        "n_test_censored": int(sum(censored)),
    }
    return out


def permute_labels(train, seed):
    rng = np.random.default_rng(int(seed))
    out = [dict(r) for r in train]
    by = defaultdict(list)
    for i, r in enumerate(out):
        by[int(r["scaf"])].append(i)
    for inds in by.values():
        values = [(out[i]["y"], out[i]["cens"]) for i in inds]
        rng.shuffle(values)
        for i, (y, c) in zip(inds, values):
            out[i]["y"] = float(y); out[i]["cens"] = bool(c)
    return out


def permute_sequences(train, seed):
    rng = np.random.default_rng(int(seed))
    out = [dict(r) for r in train]
    by = {}
    for r in out: by.setdefault(str(r["jid"]), str(r["junction_seq"]))
    ids = sorted(by); vals = [by[j] for j in ids]; rng.shuffle(vals)
    mapping = dict(zip(ids, vals))
    for r in out: r["junction_seq"] = mapping[str(r["jid"])]
    return out


def run(args):
    rows, fallback = seq.load_records(args.records)
    seq.edit_component_labels(rows, max_distance=1)
    blocks, field = outer_blocks(rows, args.split, args.seed)
    folds = []
    blocks = blocks[:int(args.max_folds)] if args.max_folds is not None else blocks
    for i, held in enumerate(blocks):
        train, test = split_rows(rows, field, held)
        genuine = evaluate(train, test, int(args.seed) + i * 1000, maxiter=args.maxiter)
        controls = {}
        if not args.no_controls:
            controls = {
                "label_within_scaffold_null": evaluate(permute_labels(train, args.seed + 20000 + i), test, args.seed + 21000 + i, maxiter=args.maxiter),
                "sequence_pairing_null": evaluate(permute_sequences(train, args.seed + 30000 + i), test, args.seed + 31000 + i, maxiter=args.maxiter),
            }
        folds.append({"fold": i, "held_groups": sorted(held), "group_field": field,
                      "n_train": len(train), "n_test": len(test), "genuine": genuine, "controls": controls})
    gains = [f["genuine"]["common_tau_gain_vs_scaffold"]["mean_gain"] for f in folds]
    nulls = {}
    for name in ("label_within_scaffold_null", "sequence_pairing_null"):
        if all(name in f["controls"] for f in folds):
            vals = [f["controls"][name]["common_tau_gain_vs_scaffold"]["mean_gain"] for f in folds]
            nulls[name] = {"fold_gains": vals, "mean": float(np.mean(vals)), "positive_fraction": float(np.mean(np.asarray(vals) > 0))}
        else:
            nulls[name] = {"status": "NOT_RUN"}
    out = {
        "schema_version": "tecto-v1.31-joint-latent-operator-tobit-probe-v1",
        "status": "EXPLORATORY_NOT_AUTHORITATIVE",
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": args.split, "group_field": field, "seed": int(args.seed),
        "n_rows": len(rows), "n_measured": int(sum(not r["cens"] for r in rows)),
        "n_censored": int(sum(bool(r["cens"]) for r in rows)),
        "n_junctions": len({str(r["jid"]) for r in rows}),
        "n_symmetry_keys": len({str(r["symmetry_key"]) for r in rows}),
        "n_edit_components": len({str(r["edit_component"]) for r in rows}),
        "n_scaffolds": len({int(r["scaf"]) for r in rows}),
        "err10_fallback": fallback,
        "model": "q_j~Normal(f_theta(sequence), sigma_q=1); Y_js|q_j~Normal(a_s+b_s*q_j,tau=0.7); exact marginal right-censored likelihood by Gauss-Hermite",
        "quadrature_nodes": GH_NODES, "ridge": RIDGE, "slope_ridge": SLOPE_RIDGE,
        "summary": {"fold_gains_common_tau_vs_scaffold": gains, "mean_gain": float(np.mean(gains)),
                    "positive_fold_fraction": float(np.mean(np.asarray(gains) > 0)), "nulls": nulls},
        "forbidden": ["current_DMS", "qMaPseq_labels", "dg9", "dg11", "dg10_5mM", "interpolated_labels", "outer_test_labels", "foundation_model"],
        "claim_boundary": "operator-aware latent-functional likelihood probe; no universal thermodynamics or independent assay transfer",
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "summary": out["summary"]}, ensure_ascii=False, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", default="symmetry_5fold")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--maxiter", type=int, default=180)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--no-controls", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
