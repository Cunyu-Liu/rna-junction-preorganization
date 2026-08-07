"""v1.31 corrected objective/Jacobian.

Model (unchanged from legacy):
    q_j ~ N(f_theta(sequence), sigma_q^2),  f = X_j @ theta
    Y_js | q_j ~ N(a_s + b_s q_j, tau^2)
    right-censored marginal likelihood via Gauss-Hermite
    objective = (D + R) / n_groups
      D = -sum_j log_marginal_j        (data term, sum over junctions)
      R = 0.5*ridge*||theta||^2 + 0.5*slope_ridge*sum_{s!=ref} (log b_s)^2
    correct gradient = (grad D + grad R) / n_groups

The ONLY change versus the legacy v1.31 implementation is the gradient scaling:
the legacy code divided the data gradient by n_groups at each component and
then divided the whole gradient by n_groups again (so data part -> n_groups^2),
while the regularization gradient was divided once.  Here the data gradient is
computed unscaled, the regularization gradient is computed unscaled, and both
are divided once by n_groups, matching the objective.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr, logsumexp

CAP = -7.1
TAU = 0.7
SIGMA_Q = 1.0


def hermite(nodes):
    """Return (nodes, log_weights) with the standard Gaussian-weight convention."""
    x, w = np.polynomial.hermite.hermgauss(int(nodes))
    log_w = np.log(np.maximum(w, 1e-300)) - 0.5 * math.log(math.pi)
    return x, log_w


def unpack(params, n_features, n_scaffolds, ref_index):
    theta = np.asarray(params[:n_features], dtype=float)
    a = np.asarray(params[n_features:n_features + n_scaffolds], dtype=float)
    log_b = np.zeros(n_scaffolds, dtype=float)
    cursor = n_features + n_scaffolds
    for s in range(n_scaffolds):
        if s != ref_index:
            log_b[s] = float(params[cursor])
            cursor += 1
    return theta, a, np.exp(np.clip(log_b, -2.0, 2.0))


def pack(theta, a, log_b, ref_index):
    out = list(np.asarray(theta, dtype=float))
    out.extend(float(x) for x in a)
    for s in range(len(log_b)):
        if s != ref_index:
            out.append(float(log_b[s]))
    return np.asarray(out, dtype=float)


def bounds(n_features, n_scaffolds, ref_index):
    """Parameter bounds matching the legacy v1.31 constraint set."""
    lo = list(np.full(n_features, -4.0))
    hi = list(np.full(n_features, 4.0))
    lo += [-18.0] * n_scaffolds
    hi += [-5.0] * n_scaffolds
    for s in range(n_scaffolds):
        if s != ref_index:
            lo.append(-1.5)
            hi.append(1.5)
    return list(zip(lo, hi))


def _score_and_grad(panel, x_by_jid, nodes, log_weights, theta, a, b,
                    ridge, slope_ridge, ref_index):
    """Return (objective, correct_gradient) with gradient scaled once by n."""
    n_groups = len(panel["jids"])
    n_scaffolds = len(panel["scaffolds"])
    f = np.asarray(x_by_jid @ theta, dtype=float)
    q = f[:, None] + math.sqrt(2.0 * SIGMA_Q) * nodes[None, :]
    mu = a[panel["flat_s"]][:, None] + b[panel["flat_s"]][:, None] * q[panel["flat_j"]]
    y = panel["flat_y"][:, None]
    c = panel["flat_c"][:, None]
    z = (y - mu) / TAU
    ll = np.where(c,
                  log_ndtr((mu - CAP) / TAU),
                  -0.5 * math.log(2.0 * math.pi) - math.log(TAU) - 0.5 * z * z)
    grouped = np.zeros((n_groups, len(nodes)), dtype=float)
    for k in range(len(nodes)):
        np.add.at(grouped[:, k], panel["flat_j"], ll[:, k])
    log_marginal = logsumexp(grouped + log_weights[None, :], axis=1)
    D = -float(np.sum(log_marginal))
    posterior = np.exp(grouped + log_weights[None, :] - log_marginal[:, None])

    score_mu = np.where(
        c,
        np.exp(-0.5 * ((mu - CAP) / TAU) ** 2 - 0.5 * math.log(2.0 * math.pi)
               - log_ndtr((mu - CAP) / TAU)) / TAU,
        (y - mu) / (TAU * TAU),
    )
    score_jk = np.zeros_like(grouped)
    score_a = np.zeros(n_scaffolds, dtype=float)
    score_logb = np.zeros(n_scaffolds, dtype=float)
    for k in range(len(nodes)):
        np.add.at(score_jk[:, k], panel["flat_j"], score_mu[:, k] * b[panel["flat_s"]])
        np.add.at(score_a, panel["flat_s"], posterior[panel["flat_j"], k] * score_mu[:, k])
        np.add.at(score_logb, panel["flat_s"], posterior[panel["flat_j"], k]
                  * score_mu[:, k] * b[panel["flat_s"]] * q[panel["flat_j"], k])
    grad_f = np.sum(posterior * score_jk, axis=1)  # dD / d f_j
    # unscaled data gradients
    grad_theta_data = -(x_by_jid.T @ grad_f)
    grad_a_data = -score_a
    grad_logb_data = -score_logb

    # regularization (unscaled)
    R = 0.5 * float(ridge) * float(np.dot(theta, theta))
    free = [s for s in range(n_scaffolds) if s != ref_index]
    R += 0.5 * float(slope_ridge) * sum(float(math.log(b[s]) ** 2) for s in free)
    grad_theta_reg = float(ridge) * theta
    grad_free_reg = [float(slope_ridge * math.log(b[s])) for s in free]

    total = D + R
    grad_theta = (grad_theta_data + grad_theta_reg) / max(n_groups, 1)
    grad_a = grad_a_data / max(n_groups, 1)
    grad_free = [(grad_logb_data[s] + reg) / max(n_groups, 1)
                 for s, reg in zip(free, grad_free_reg)]
    grad = np.r_[grad_theta, grad_a, np.asarray(grad_free, dtype=float)]
    return total / max(n_groups, 1), grad


def corrected_objective_and_grad(params, panel, x_by_jid, nodes, log_weights,
                                 ridge, slope_ridge, ref_index):
    n_features = x_by_jid.shape[1]
    n_scaffolds = len(panel["scaffolds"])
    theta, a, b = unpack(params, n_features, n_scaffolds, ref_index)
    return _score_and_grad(panel, x_by_jid, nodes, log_weights, theta, a, b,
                           ridge, slope_ridge, ref_index)


def legacy_objective_and_grad(params, panel, x_by_jid, nodes, log_weights,
                              ridge, slope_ridge, ref_index):
    """Faithful reproduction of the v1.31 bug: data grad /n inside and /n again."""
    n_features = x_by_jid.shape[1]
    n_scaffolds = len(panel["scaffolds"])
    theta, a, b = unpack(params, n_features, n_scaffolds, ref_index)
    n_groups = len(panel["jids"])
    f = np.asarray(x_by_jid @ theta, dtype=float)
    q = f[:, None] + math.sqrt(2.0 * SIGMA_Q) * nodes[None, :]
    mu = a[panel["flat_s"]][:, None] + b[panel["flat_s"]][:, None] * q[panel["flat_j"]]
    y = panel["flat_y"][:, None]
    c = panel["flat_c"][:, None]
    z = (y - mu) / TAU
    ll = np.where(c, log_ndtr((mu - CAP) / TAU),
                  -0.5 * math.log(2.0 * math.pi) - math.log(TAU) - 0.5 * z * z)
    grouped = np.zeros((n_groups, len(nodes)), dtype=float)
    for k in range(len(nodes)):
        np.add.at(grouped[:, k], panel["flat_j"], ll[:, k])
    log_marginal = logsumexp(grouped + log_weights[None, :], axis=1)
    total = -float(np.sum(log_marginal))
    posterior = np.exp(grouped + log_weights[None, :] - log_marginal[:, None])
    score_mu = np.where(
        c,
        np.exp(-0.5 * ((mu - CAP) / TAU) ** 2 - 0.5 * math.log(2.0 * math.pi)
               - log_ndtr((mu - CAP) / TAU)) / TAU,
        (y - mu) / (TAU * TAU),
    )
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
    grad_f = np.sum(posterior * score_jk, axis=1)
    grad_theta = -(x_by_jid.T @ grad_f) / max(n_groups, 1)
    grad_a = -score_a / max(n_groups, 1)
    grad_logb = -score_logb / max(n_groups, 1)
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
