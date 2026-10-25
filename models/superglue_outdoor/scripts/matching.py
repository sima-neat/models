"""The application's CPU side of SuperGlue: keypoint normalization, the Sinkhorn iterations and match extraction.

NumPy only, so the conversion check, the compile gate and the DevKit accuracy stage all use the same reference.
"""

import numpy as np

SINKHORN_ITERATIONS = 100
# Mutual matches scoring at or below this are dropped, as in Magic Leap's demo.
MATCH_THRESHOLD = 0.2


def normalize_keypoints(keypoints, width, height):
    """Map pixel (x, y) keypoints to SuperGlue's frame: centred on the image, divided by 0.7 times its longer side."""
    size = np.array([width, height], dtype=np.float64)
    return ((np.asarray(keypoints, dtype=np.float64) - size / 2) / (size.max() * 0.7)).astype(np.float32)


def log_sinkhorn(couplings, iterations=SINKHORN_ITERATIONS):
    """Return the log optimal-transport matrix of [1, M+1, N+1] couplings, as SuperGlue's log_optimal_transport."""
    rows, columns = couplings.shape[-2] - 1, couplings.shape[-1] - 1
    normalization = -np.log(rows + columns)
    log_mu = np.concatenate([np.full(rows, normalization), [np.log(columns) + normalization]])
    log_nu = np.concatenate([np.full(columns, normalization), [np.log(rows) + normalization]])
    cost = np.asarray(couplings, dtype=np.float64).reshape(rows + 1, columns + 1)
    # The same iterations in the linear domain: one exponential, then matrix-vector products. Shifting by the
    # maximum keeps exp() finite; the scalings absorb the shift. Fall back to log space if a row underflows.
    kernel = np.exp(cost - cost.max())
    mu, nu = np.exp(log_mu), np.exp(log_nu)
    u, v = np.ones_like(mu), np.ones_like(nu)
    for _ in range(iterations):
        u = mu / (kernel @ v)
        v = nu / (kernel.T @ u)
    if np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(u > 0) and np.all(v > 0):
        return (cost - cost.max()) + np.log(u)[:, None] + np.log(v)[None, :] - normalization
    u, v = np.zeros_like(log_mu), np.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - _logsumexp(cost + v[None, :], axis=1)
        v = log_nu - _logsumexp(cost + u[:, None], axis=0)
    return cost + u[:, None] + v[None, :] - normalization


def _logsumexp(values, axis):
    top = values.max(axis=axis, keepdims=True)
    return (top + np.log(np.exp(values - top).sum(axis=axis, keepdims=True))).squeeze(axis)


def extract_matches(log_transport, threshold=0.0):
    """Return, for each image-0 keypoint, its mutual nearest image-1 keypoint (-1 when unmatched) and match score."""
    inner = log_transport[:-1, :-1]
    best0, best1 = inner.argmax(1), inner.argmax(0)
    mutual = best1[best0] == np.arange(len(best0))
    scores = np.where(mutual, np.exp(inner.max(1)), 0.0)
    valid = mutual & (scores > threshold)
    return np.where(valid, best0, -1), np.where(valid, scores, 0.0)
