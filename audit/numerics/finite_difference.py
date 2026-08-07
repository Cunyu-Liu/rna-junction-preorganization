"""Central finite-difference gradient check for the corrected v1.31 objective."""
from __future__ import annotations

import numpy as np


def central_fd(fun, x, eps=1e-6):
    """Central finite-difference gradient of scalar fun at x."""
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (fun(xp) - fun(xm)) / (2.0 * eps)
    return g


def relative_grad_error(analytic, fd, scale=1e-8):
    """max |analytic-fd| / (scale + |fd|)."""
    a = np.asarray(analytic, dtype=float)
    f = np.asarray(fd, dtype=float)
    denom = np.maximum(np.abs(f), scale)
    return float(np.max(np.abs(a - f) / denom))


def max_abs_error(analytic, fd):
    return float(np.max(np.abs(np.asarray(analytic) - np.asarray(fd))))
