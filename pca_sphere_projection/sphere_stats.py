"""
Spherical statistics for PCA-on-sphere embeddings.

These routines operate on (N, 3) arrays of unit vectors (the same coordinates
that ``apply_euler_rotation`` in ``core.py`` produces). They quantify
structure on S^2 with proper spherical geometry, replacing several places in
the existing pipeline where Euclidean or (lon, lat) statistics implicitly
distort area or distance.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _as_unit(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    n = np.linalg.norm(coords, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("coords contains zero-magnitude rows; cannot project to S^2")
    return coords / n


def _fibonacci_sphere(m: int) -> np.ndarray:
    """Quasi-uniform M points on S^2 (golden-spiral). Used for KDE eval grids."""
    i = np.arange(m, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / m)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)])


def stripe_strength_score(coords: np.ndarray,
                          n_bins: int = 36,
                          axis: np.ndarray | None = None) -> dict:
    """
    Quantifies how strongly points concentrate into longitudinal stripes
    around a chosen axis.

    Args:
        coords: (N, 3) unit vectors on S^2.
        n_bins: number of longitudinal bins. Default 36 (10-degree bins).
        axis: (3,) pole vector. If None, uses +z (matching the convention
            after ``align_to_north_pole``).

    Returns:
        dict with keys:
            score: 1 - H_obs / H_uniform, where H is Shannon entropy in nats.
                0 = uniform; 1 = all points in one bin.
            entropy_observed: H_obs in nats.
            entropy_uniform: H_uniform in nats (= log(n_bins)).
            bin_counts: (n_bins,) integer counts.

    Algorithm:
        Rotate coords so ``axis`` -> +z, take longitude lambda = atan2(y, x),
        bin into ``n_bins`` equal longitudinal slices, compute Shannon
        entropy. Compare against the uniform null on the *same* longitudinal
        partition (which is uniform in lambda regardless of latitude weight),
        so this score is unbiased by the equirectangular Jacobian.
    """
    coords = _as_unit(coords)
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    R = _rotation_to_pole(axis)
    rotated = coords @ R.T
    lon = np.arctan2(rotated[:, 1], rotated[:, 0])
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(lon, bins=bins)
    p = counts / counts.sum()
    nz = p > 0
    h_obs = -np.sum(p[nz] * np.log(p[nz]))
    h_uni = np.log(n_bins)
    return {
        "score": float(1.0 - h_obs / h_uni),
        "entropy_observed": float(h_obs),
        "entropy_uniform": float(h_uni),
        "bin_counts": counts,
    }


def spherical_kde(coords: np.ndarray,
                  kappa: float = 50.0,
                  eval_points: np.ndarray | None = None,
                  m_grid: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum-of-vMF kernel density estimate on S^2.

    Args:
        coords: (N, 3) unit vectors.
        kappa: vMF concentration. Larger = sharper kernels.
        eval_points: optional (M, 3) grid; default Fibonacci grid of size m_grid.
        m_grid: grid size if eval_points is None.

    Returns:
        density: (M,) density values, normalised so the discrete sum on the
            Fibonacci grid approximates a probability density.
        eval_points: (M, 3) the grid used.

    Notes:
        f(x) = (kappa / (4 pi sinh(kappa))) * mean_i exp(kappa * x . x_i)
        Scaled by 4 pi so integral over S^2 ~ 1.
    """
    coords = _as_unit(coords)
    if eval_points is None:
        eval_points = _fibonacci_sphere(m_grid)
    else:
        eval_points = _as_unit(eval_points)

    # vMF normalisation in 3D: C_3(kappa) = kappa / (4 pi sinh(kappa))
    # For numerical stability when kappa is large, use log-space.
    log_C = np.log(kappa) - np.log(4 * np.pi) - kappa - np.log1p(-np.exp(-2 * kappa)) + np.log(2)
    inner = eval_points @ coords.T  # (M, N)
    log_kernel = log_C + kappa * inner
    # mean over N kernels -> logsumexp - log(N)
    mx = log_kernel.max(axis=1, keepdims=True)
    log_density = mx.squeeze(1) + np.log(np.exp(log_kernel - mx).mean(axis=1))
    return np.exp(log_density), eval_points


def geodesic_gradient(coords: np.ndarray,
                      scalar_field: np.ndarray,
                      k: int = 15) -> np.ndarray:
    """
    Tangent-space gradient of a scalar field on S^2, estimated by local
    weighted least squares among k geodesic neighbours.

    Args:
        coords: (N, 3) unit vectors.
        scalar_field: (N,) per-cell scalar (pseudotime, gene expression, ...).
        k: neighbours per cell.

    Returns:
        grads: (N, 3) gradient vectors. Each row lies in the tangent plane
        at coords[i] (i.e. orthogonal to coords[i]).
    """
    coords = _as_unit(coords)
    scalar_field = np.asarray(scalar_field, dtype=float)
    if scalar_field.shape[0] != coords.shape[0]:
        raise ValueError("scalar_field length must match coords")

    n = coords.shape[0]
    cosd = np.clip(coords @ coords.T, -1.0, 1.0)
    geo = np.arccos(cosd)  # geodesic distances (radians)
    np.fill_diagonal(geo, np.inf)

    grads = np.zeros_like(coords)
    for i in range(n):
        nb = np.argpartition(geo[i], k)[:k]
        x_i = coords[i]
        # Tangent-plane projection of neighbour displacements
        disp = coords[nb] - x_i
        disp = disp - (disp @ x_i)[:, None] * x_i
        dy = scalar_field[nb] - scalar_field[i]
        # weights: inverse geodesic distance, capped
        w = 1.0 / (geo[i, nb] + 1e-6)
        # Weighted least squares: minimise sum w_j (dy_j - disp_j . g)^2 with
        # constraint that g lies in the tangent plane (already true since disp does).
        WX = disp * w[:, None]
        Wy = dy * w
        try:
            g, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
        except np.linalg.LinAlgError:
            g = np.zeros(3)
        # Re-project g into the tangent plane (numerical safety)
        g = g - (g @ x_i) * x_i
        grads[i] = g
    return grads


def fit_great_circle(coords: np.ndarray,
                     weights: np.ndarray | None = None) -> dict:
    """
    Best-fit great circle through the origin.

    The great circle is the intersection of S^2 with the plane through the
    origin whose normal n minimises sum w_i (n . x_i)^2 subject to ||n||=1.
    Solution: eigenvector of smallest eigenvalue of sum w_i x_i x_i^T.

    Returns:
        dict with:
            normal: (3,) unit normal of the best-fit plane.
            mean_residual_radians: mean angular deviation from the plane.
            r_squared: 1 - sum w_i (n . x_i)^2 / sum w_i (1 - mean term).
            eigenvalues: ascending eigenvalues of the inertia matrix.

    This replaces manual ``apply_euler_rotation`` tuning: the rotation that
    aligns ``normal`` to (0, 0, 1) puts the dataset's principal great circle
    on the equator, so longitude becomes a natural pseudotime axis.
    """
    coords = _as_unit(coords)
    if weights is None:
        weights = np.ones(coords.shape[0])
    weights = np.asarray(weights, dtype=float)
    W = (coords * weights[:, None]).T @ coords / weights.sum()
    eigvals, eigvecs = np.linalg.eigh(W)
    normal = eigvecs[:, 0]
    inner = coords @ normal
    mean_resid = float(np.mean(np.abs(np.arcsin(np.clip(inner, -1.0, 1.0)))))
    # variance explained by the great-circle plane vs total
    r_squared = float(1.0 - eigvals[0] / eigvals.sum())
    return {
        "normal": normal,
        "mean_residual_radians": mean_resid,
        "r_squared": r_squared,
        "eigenvalues": eigvals,
    }


def spherical_anisotropy(coords: np.ndarray) -> dict:
    """
    Shape descriptors of a spherical point cloud, from the inertia tensor
    (1/N) sum x_i x_i^T (eigenvalues sum to 1 since ||x_i||=1).

    Returns:
        dict with eigenvalues lambda1 >= lambda2 >= lambda3 and:
            linear: (lambda1 - lambda2) / lambda1 -- arc-like
            planar: (lambda2 - lambda3) / lambda1 -- great-circle ring-like
            spherical: lambda3 / lambda1 -- uniform on S^2

    Useful as a *prerequisite* check before claiming "stripes": a dataset
    with high `spherical` score is not stripey at all and downstream
    interpretations will be unreliable.
    """
    coords = _as_unit(coords)
    M = coords.T @ coords / coords.shape[0]
    eigvals = np.linalg.eigvalsh(M)[::-1]  # descending
    l1, l2, l3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    return {
        "lambda1": l1, "lambda2": l2, "lambda3": l3,
        "linear": (l1 - l2) / l1,
        "planar": (l2 - l3) / l1,
        "spherical": l3 / l1,
    }


def horseshoe_null_test(expression_hvg: np.ndarray,
                        n_perm: int = 200,
                        n_pcs: int = 3,
                        score_fn=None,
                        rng: np.random.Generator | None = None) -> dict:
    """
    Permutation null for spherical structure.

    Permutes each HVG independently across cells (preserves marginal
    expression distribution; destroys cell-cell covariance), re-runs PCA on
    the permuted matrix, projects to S^2, scores with score_fn, and compares
    against the un-permuted score.

    Args:
        expression_hvg: (N, G) HVG-filtered, log-normalised expression.
        n_perm: number of permutations.
        n_pcs: number of PCs to project (default 3 -> S^2).
        score_fn: callable (coords) -> float. Default: stripe_strength_score
            then take ['score'].
        rng: optional np.random.Generator.

    Returns:
        dict with observed, null_distribution (n_perm,), p_value, effect_size_z.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if score_fn is None:
        def score_fn(c):
            return stripe_strength_score(c)["score"]

    X = np.asarray(expression_hvg, dtype=float)
    if n_pcs != 3:
        raise NotImplementedError("Spherical projection assumes 3 PCs")

    def _project(X_):
        Xc = X_ - X_.mean(axis=0, keepdims=True)
        # Use SVD; truncated to 3 components.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        pcs = Xc @ Vt[:n_pcs].T
        return _as_unit(pcs)

    observed = float(score_fn(_project(X)))
    null = np.empty(n_perm)
    for p in range(n_perm):
        Xp = X.copy()
        for g in range(Xp.shape[1]):
            rng.shuffle(Xp[:, g])
        null[p] = float(score_fn(_project(Xp)))

    p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    z = float((observed - null.mean()) / (null.std(ddof=1) + 1e-12))
    return {
        "observed": observed,
        "null_distribution": null,
        "p_value": p_value,
        "effect_size_z": z,
    }


def _rotation_to_pole(axis: np.ndarray) -> np.ndarray:
    """3x3 rotation that takes ``axis`` to (0, 0, 1) (Rodrigues, no-op when aligned)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    s = np.linalg.norm(v)
    c = float(a @ z)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))
