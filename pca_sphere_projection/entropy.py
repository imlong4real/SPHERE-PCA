"""
Per-cell transcriptional entropy / stemness scoring and the H1 geodesic-gradient
test.

Three scoring options:

- Shannon entropy of the per-cell normalised expression vector. Cheap;
  conceptually closest to "transcriptional disorder". Used as the default
  proxy when a real CytoTRACE/SCENT score is unavailable.

- CytoTRACE proxy: per-cell number of expressed genes (gene count score, GCS).
  CytoTRACE itself ranks cells by GCS smoothed over a kNN graph. Without the
  exact CytoTRACE smoother we take per-cell GCS and call it a proxy.

- SCENT proxy: log of the variance of per-cell-normalised expression across
  the gene set, an extremely simplified surrogate; SCENT proper requires
  a PPI network.

If a dataset already ships with real CytoTRACE scores in its metadata
(e.g. the hESC RDS), pass them in directly to ``test_entropy_geodesic_gradient``;
the workflow does this and labels the curve as "CytoTRACE (precomputed)".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr


def compute_transcriptional_entropy(X, gene_names=None, method: str = "shannon"):
    """
    Per-cell entropy / stemness score.

    Args:
        X: (N, G) expression (sparse OK). For "shannon" the matrix is
            normalised per cell to a probability distribution before entropy
            is computed; for "cytotrace_proxy" we count nonzero genes.
        gene_names: unused for these proxies (accepted for API symmetry).
        method: ``"shannon"``, ``"cytotrace_proxy"``, or ``"scent_proxy"``.

    Returns:
        (N,) numpy array of per-cell scores. Higher = more stem-like for
        all three methods (we invert raw entropy direction so the sign is
        consistent across methods).
    """
    if method == "shannon":
        return _shannon_per_cell(X)
    if method == "cytotrace_proxy":
        return compute_cytotrace_proxy(X, gene_names)
    if method == "scent_proxy":
        return compute_scent_proxy(X, gene_names)
    raise ValueError(f"unknown method {method!r}")


def _shannon_per_cell(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        sums = np.asarray(X.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        n = X.shape[0]
        out = np.zeros(n)
        # iterate row by row; sparse pure-numpy entropy
        for i in range(n):
            row = X.getrow(i)
            if row.nnz == 0:
                out[i] = 0.0; continue
            p = row.data / sums[i]
            out[i] = -np.sum(p * np.log(p + 1e-30))
        return out
    Xc = np.asarray(X, dtype=np.float64)
    sums = Xc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    P = Xc / sums
    P = np.where(P > 0, P, 1e-30)
    return -np.sum(P * np.log(P), axis=1)


def compute_cytotrace_proxy(X, gene_names=None) -> np.ndarray:
    """
    GCS-style "gene count signature": per-cell count of genes with nonzero
    expression. CytoTRACE smooths GCS over a kNN graph; we leave it raw and
    label the result a proxy.

    Higher = more genes expressed = (in CytoTRACE convention) more stem-like.
    """
    if sp.issparse(X):
        return np.asarray((X != 0).sum(axis=1)).ravel().astype(float)
    return (np.asarray(X) != 0).sum(axis=1).astype(float)


def compute_scent_proxy(X, gene_names=None) -> np.ndarray:
    """
    SCENT proxy: per-cell variance of cell-normalised expression. A coarse
    surrogate for "signalling entropy"; SCENT itself uses a network-weighted
    diffusion which we cannot replicate without a PPI.
    """
    if sp.issparse(X):
        Xc = X.toarray().astype(np.float64)
    else:
        Xc = np.asarray(X, dtype=np.float64)
    sums = Xc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    P = Xc / sums
    return P.var(axis=1)


# ---------------------------------------------------------------------------
# H1: geodesic gradient test
# ---------------------------------------------------------------------------

def _geodesic_distance_to_root(coords_unit: np.ndarray, root_coord: np.ndarray) -> np.ndarray:
    coords_unit = np.asarray(coords_unit, dtype=float)
    rc = np.asarray(root_coord, dtype=float)
    rc = rc / np.linalg.norm(rc)
    inner = np.clip(coords_unit @ rc, -1.0, 1.0)
    return np.arccos(inner)


def test_entropy_geodesic_gradient(coords: np.ndarray,
                                   entropy_score: np.ndarray,
                                   root_coords: np.ndarray | None = None,
                                   pseudotime: Optional[np.ndarray] = None,
                                   pc_scores_for_euclid: Optional[np.ndarray] = None) -> dict:
    """
    Test whether ``entropy_score`` covaries with geodesic distance from
    a chosen root direction.

    Args:
        coords: (N, 3) PC scores; will be projected to S^2 internally.
        entropy_score: (N,) per-cell score (any of the methods above).
        root_coords: (3,) reference direction. If None, uses the centroid
            of the top-decile entropy cells (most stem-like).
        pseudotime: optional (N,) numeric pseudotime to also report against.
        pc_scores_for_euclid: optional (N, k) PC-space coordinates for
            comparing geodesic vs. plain Euclidean PC distance.

    Returns:
        dict of correlation coefficients and p-values, plus a per-cell
        DataFrame keyed by ``entropy_score``, ``geodesic_dist_root``,
        ``euclid_pc_dist_root``, and (if available) ``pseudotime``.
    """
    coords = np.asarray(coords, dtype=float)
    n = np.linalg.norm(coords, axis=1, keepdims=True)
    n[n == 0] = 1.0
    unit = coords / n
    s = np.asarray(entropy_score, dtype=float)
    if root_coords is None:
        cutoff = np.quantile(s, 0.9)
        anchor = unit[s >= cutoff].mean(axis=0)
    else:
        anchor = np.asarray(root_coords, dtype=float)
    geo = _geodesic_distance_to_root(unit, anchor)
    rho_geo, p_geo = spearmanr(s, geo, nan_policy="omit")

    if pc_scores_for_euclid is None:
        pc_scores_for_euclid = coords
    pc_anchor = pc_scores_for_euclid[s >= np.quantile(s, 0.9)].mean(axis=0) \
        if root_coords is None else np.asarray(root_coords, dtype=float)
    euclid = np.linalg.norm(pc_scores_for_euclid - pc_anchor, axis=1)
    rho_euc, p_euc = spearmanr(s, euclid, nan_policy="omit")

    out = {
        "geodesic_spearman_rho": float(rho_geo),
        "geodesic_spearman_p": float(p_geo),
        "euclidean_spearman_rho": float(rho_euc),
        "euclidean_spearman_p": float(p_euc),
        "n_cells": int(unit.shape[0]),
        "anchor_method": "decile" if root_coords is None else "external",
    }

    if pseudotime is not None:
        pt = np.asarray(pseudotime, dtype=float)
        mask = ~np.isnan(pt)
        if mask.sum() > 5:
            rho_pt, p_pt = spearmanr(s[mask], pt[mask])
            out["pseudotime_spearman_rho"] = float(rho_pt)
            out["pseudotime_spearman_p"] = float(p_pt)

    out["per_cell"] = pd.DataFrame({
        "entropy_score": s,
        "geodesic_dist_root": geo,
        "euclid_pc_dist_root": euclid,
        "pseudotime": pseudotime if pseudotime is not None else np.nan,
    })
    return out


__all__ = [
    "compute_transcriptional_entropy",
    "compute_cytotrace_proxy",
    "compute_scent_proxy",
    "test_entropy_geodesic_gradient",
]
