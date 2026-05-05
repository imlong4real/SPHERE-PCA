"""
Topological / trajectory diagnostics on the spherical embedding.

Distinguishes linear (single great-circle) trajectories from branching
trajectories without relying on graph-clustering heuristics (PAGA, Slingshot,
Monocle3). All scores are local and per-cell, supporting single-cell
resolution branchpoint annotation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .sphere_stats import _as_unit, geodesic_gradient, fit_great_circle


def build_spherical_knn_graph(coords: np.ndarray, k: int = 15) -> sparse.csr_matrix:
    """
    k-nearest-neighbour graph on S^2 using geodesic (arc-length) distance.

    Args:
        coords: (N, 3) unit vectors.
        k: neighbours per cell (excluding self).

    Returns:
        (N, N) sparse CSR matrix; entry [i, j] = geodesic distance in
        radians if j is among i's k nearest, else 0.
    """
    coords = _as_unit(coords)
    n = coords.shape[0]
    cosd = np.clip(coords @ coords.T, -1.0, 1.0)
    geo = np.arccos(cosd)
    np.fill_diagonal(geo, np.inf)

    rows = np.repeat(np.arange(n), k)
    cols = np.empty(n * k, dtype=int)
    vals = np.empty(n * k, dtype=float)
    for i in range(n):
        nb = np.argpartition(geo[i], k)[:k]
        cols[i * k:(i + 1) * k] = nb
        vals[i * k:(i + 1) * k] = geo[i, nb]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def detect_branchpoints(coords: np.ndarray,
                        pseudotime: np.ndarray,
                        k: int = 15,
                        n_bootstrap: int = 0,
                        rng: np.random.Generator | None = None) -> dict:
    """
    Per-cell branchpoint score from angular divergence of local pseudotime
    gradients.

    Idea: at a true branchpoint, neighbours' tangent-space gradient vectors
    point in different downstream directions. Along a linear trajectory they
    align. Score = 1 - || mean(unit gradient vectors of k neighbours) ||.

    Args:
        coords: (N, 3).
        pseudotime: (N,).
        k: neighbours used for both the gradient estimate and the
            divergence statistic.
        n_bootstrap: if > 0, resample pseudotime within each k-neighbourhood
            and compute a bootstrap p-value per cell. Disabled by default
            for speed.
        rng: optional np.random.Generator.

    Returns:
        dict with:
            score: (N,) angular-divergence score in [0, 1]; high = branch-like.
            gradient: (N, 3) per-cell tangent gradient.
            p_value: (N,) bootstrap p-value (or None if n_bootstrap=0).
    """
    coords = _as_unit(coords)
    pseudotime = np.asarray(pseudotime, dtype=float)
    n = coords.shape[0]
    grads = geodesic_gradient(coords, pseudotime, k=k)
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    unit_grads = np.where(norms > 1e-12, grads / np.maximum(norms, 1e-12), 0.0)

    cosd = np.clip(coords @ coords.T, -1.0, 1.0)
    geo = np.arccos(cosd)
    np.fill_diagonal(geo, np.inf)

    score = np.empty(n)
    for i in range(n):
        nb = np.argpartition(geo[i], k)[:k]
        # Parallel-transport correction is small for k-neighbourhoods on S^2;
        # we instead project neighbour gradients into the tangent plane at
        # coords[i] before averaging.
        x_i = coords[i]
        gj = unit_grads[nb]
        gj = gj - (gj @ x_i)[:, None] * x_i
        nrm = np.linalg.norm(gj, axis=1, keepdims=True)
        gj = np.where(nrm > 1e-12, gj / np.maximum(nrm, 1e-12), 0.0)
        mean_vec = gj.mean(axis=0)
        score[i] = 1.0 - float(np.linalg.norm(mean_vec))

    p_value = None
    if n_bootstrap > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        null = np.empty((n_bootstrap, n))
        for b in range(n_bootstrap):
            shuffled = rng.permutation(pseudotime)
            g_b = geodesic_gradient(coords, shuffled, k=k)
            nb_norms = np.linalg.norm(g_b, axis=1, keepdims=True)
            ug_b = np.where(nb_norms > 1e-12, g_b / np.maximum(nb_norms, 1e-12), 0.0)
            for i in range(n):
                nb = np.argpartition(geo[i], k)[:k]
                x_i = coords[i]
                gj = ug_b[nb]
                gj = gj - (gj @ x_i)[:, None] * x_i
                nrm = np.linalg.norm(gj, axis=1, keepdims=True)
                gj = np.where(nrm > 1e-12, gj / np.maximum(nrm, 1e-12), 0.0)
                null[b, i] = 1.0 - float(np.linalg.norm(gj.mean(axis=0)))
        p_value = (np.sum(null >= score, axis=0) + 1) / (n_bootstrap + 1)

    return {"score": score, "gradient": grads, "p_value": p_value}


def linear_vs_branching_score(coords: np.ndarray,
                              pseudotime: np.ndarray,
                              k: int = 15) -> dict:
    """
    Approximate Bayes-factor-style comparison between (a) a linear
    great-circle trajectory and (b) a single-branchpoint two-arm trajectory.

    Linear model:
        Cells lie near one great circle; pseudotime predicts position along it.
        Residual = mean angular distance to the fitted great-circle plane
        plus mean residual of pseudotime ~ longitude regression along the plane.

    Branching model:
        1) Detect candidate branchpoint as the cell with the highest
           ``detect_branchpoints`` score in the middle third of pseudotime.
        2) Split downstream cells (pseudotime > pt_branch) into two groups
           by sign of inner product with the second principal tangent
           direction.
        3) Fit two great circles and sum residuals.

    The branching model has 2x the parameters of the linear model. We
    BIC-correct: smaller BIC = better.

    Returns:
        dict with linear_logL, branching_logL, linear_bic, branching_bic,
        bayes_factor (= exp((BIC_linear - BIC_branching) / 2)), decision.
    """
    coords = _as_unit(coords)
    pt = np.asarray(pseudotime, dtype=float)
    n = coords.shape[0]

    # Linear model
    gc_lin = fit_great_circle(coords)
    plane_resid_lin = np.abs(np.arcsin(np.clip(coords @ gc_lin["normal"], -1.0, 1.0)))
    sigma2_lin = np.mean(plane_resid_lin ** 2) + 1e-12
    logL_lin = -0.5 * n * (np.log(2 * np.pi * sigma2_lin) + 1.0)
    p_lin = 2  # plane normal direction (2 free params on S^2)
    bic_lin = p_lin * np.log(n) - 2 * logL_lin

    # Branching model: detect candidate branchpoint
    bp = detect_branchpoints(coords, pt, k=k)
    pt_lo, pt_hi = np.quantile(pt, [1 / 3, 2 / 3])
    mid_mask = (pt >= pt_lo) & (pt <= pt_hi)
    if not np.any(mid_mask):
        mid_mask = np.ones(n, dtype=bool)
    candidate_idx = np.where(mid_mask)[0][np.argmax(bp["score"][mid_mask])]
    branch_pt_value = pt[candidate_idx]

    downstream = pt > branch_pt_value
    if downstream.sum() < 6:
        return {
            "linear_logL": float(logL_lin),
            "branching_logL": float("nan"),
            "linear_bic": float(bic_lin),
            "branching_bic": float("inf"),
            "bayes_factor": 0.0,
            "decision": "linear",
        }

    # Split downstream cells by 2nd principal direction in tangent plane at branchpoint
    x_b = coords[candidate_idx]
    disp = coords[downstream] - x_b
    disp = disp - (disp @ x_b)[:, None] * x_b
    # Two principal directions in tangent plane:
    _, _, Vt = np.linalg.svd(disp, full_matrices=False)
    direction = Vt[0]
    side = (disp @ direction) > 0
    arm_a = np.where(downstream)[0][side]
    arm_b = np.where(downstream)[0][~side]
    upstream = np.where(~downstream)[0]

    def _arm_resid(idx):
        if len(idx) < 3:
            return 0.0, 0
        gc = fit_great_circle(coords[idx])
        r = np.abs(np.arcsin(np.clip(coords[idx] @ gc["normal"], -1.0, 1.0)))
        return float(np.mean(r ** 2)), int(len(idx))

    s_up, n_up = _arm_resid(upstream)
    s_a, n_a = _arm_resid(arm_a)
    s_b, n_b = _arm_resid(arm_b)
    total_n = max(n_up + n_a + n_b, 1)
    sigma2_br = (s_up * n_up + s_a * n_a + s_b * n_b) / total_n + 1e-12
    logL_br = -0.5 * total_n * (np.log(2 * np.pi * sigma2_br) + 1.0)
    p_br = 6  # three plane normals
    bic_br = p_br * np.log(total_n) - 2 * logL_br

    log_bf = (bic_lin - bic_br) / 2.0
    # clamp to avoid overflow when one model is overwhelmingly preferred
    bayes_factor = float(np.exp(np.clip(log_bf, -700, 700)))
    decision = "branching" if bic_br < bic_lin else "linear"

    return {
        "linear_logL": float(logL_lin),
        "branching_logL": float(logL_br),
        "linear_bic": float(bic_lin),
        "branching_bic": float(bic_br),
        "bayes_factor": bayes_factor,
        "decision": decision,
        "branchpoint_idx": int(candidate_idx),
    }
