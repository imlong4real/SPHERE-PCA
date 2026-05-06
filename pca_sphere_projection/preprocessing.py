"""
Preprocessing for raw cell x gene matrices.

Functions are sparse-friendly where it materially matters (normalisation,
HVG selection on celegan/uc_epi sized data), and fall back to dense
operations only when the downstream step needs a dense matrix (PCA).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    scores: np.ndarray  # (n_cells, k)
    components: np.ndarray  # (k, n_genes_used)
    mean: np.ndarray  # (n_genes_used,)
    explained_variance_ratio: np.ndarray
    gene_names: list


def normalize_log1p(X, target_sum: float = 1e4):
    """
    Library-size normalise to ``target_sum`` per cell, then log1p.

    Accepts sparse or dense. Returns dense float32 (most downstream code
    needs dense; we keep float32 to halve memory).
    """
    if sp.issparse(X):
        Xc = X.astype(np.float64).tocsr()
        sums = np.asarray(Xc.sum(axis=1)).ravel()
        sums[sums == 0] = 1.0
        scale = target_sum / sums
        # row-wise scaling
        Xn = sp.diags(scale) @ Xc
        Xn = Xn.toarray()
    else:
        Xc = np.asarray(X, dtype=np.float64)
        sums = Xc.sum(axis=1)
        sums[sums == 0] = 1.0
        Xn = (Xc.T * (target_sum / sums)).T
    np.log1p(Xn, out=Xn)
    return Xn.astype(np.float32)


def select_hvgs(X, gene_names, n_top_genes: int = 2000):
    """
    Select top-N highly variable genes by *log-mean / log-variance* style
    dispersion, returning the column indices and a subset matrix.

    For matrices with fewer than ``n_top_genes`` columns we return all of
    them (this is the case for UC epi with 1361 features).
    """
    gene_names = list(gene_names)
    n_g = len(gene_names)
    if n_g <= n_top_genes:
        return np.arange(n_g), gene_names, X

    if sp.issparse(X):
        # mean / variance from sparse
        mu = np.asarray(X.mean(axis=0)).ravel()
        sq = X.multiply(X)
        m2 = np.asarray(sq.mean(axis=0)).ravel()
        var = m2 - mu ** 2
    else:
        mu = X.mean(axis=0)
        var = X.var(axis=0)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    # dispersion: variance scaled by mean (Seurat-style proxy)
    eps = 1e-9
    disp = np.log1p(var) - np.log1p(mu)
    # rank by dispersion among genes with at least some expression
    keep_pool = mu > 1e-6
    disp[~keep_pool] = -np.inf
    idx = np.argsort(disp)[::-1][:n_top_genes]
    idx = np.sort(idx)
    sub_names = [gene_names[i] for i in idx]
    if sp.issparse(X):
        return idx, sub_names, X[:, idx]
    return idx, sub_names, X[:, idx]


# ---------------------------------------------------------------------------
# Mito / ribo / cell-cycle filters by gene-symbol regex
# ---------------------------------------------------------------------------

_MITO_RE = re.compile(r"^(mt-|MT-|MT_)", re.IGNORECASE)
_RIBO_RE = re.compile(r"^(rps|rpl|RPS|RPL)", re.IGNORECASE)
_CYCLE_GENES = {
    # tiny conservative set; users with proper signatures should plug them in.
    "MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CCNA2", "CCNE1", "CCNE2",
    "CDK1", "CDK2", "BIRC5", "AURKA", "AURKB", "MCM2", "MCM5", "MCM6",
    "Mki67", "Top2a", "Pcna", "Ccnb1", "Ccnb2", "Ccna2", "Ccne1", "Ccne2",
    "Cdk1", "Cdk2", "Birc5", "Aurka", "Aurkb", "Mcm2", "Mcm5", "Mcm6",
}


def filter_genes(X, gene_names, remove_mito: bool = True,
                 remove_ribo: bool = True, remove_cell_cycle: bool = False):
    """
    Drop columns matching mito / ribo / cell-cycle regexes.

    For datasets without symbol-style gene names (e.g. UC epi with
    `feat_<i>`) the regexes simply match nothing and the matrix is
    returned unchanged.
    """
    gene_names = list(gene_names)
    keep = np.ones(len(gene_names), dtype=bool)
    dropped = {"mito": [], "ribo": [], "cc": []}
    for i, g in enumerate(gene_names):
        if remove_mito and _MITO_RE.match(g):
            keep[i] = False; dropped["mito"].append(g)
        elif remove_ribo and _RIBO_RE.match(g):
            keep[i] = False; dropped["ribo"].append(g)
        elif remove_cell_cycle and g in _CYCLE_GENES:
            keep[i] = False; dropped["cc"].append(g)
    sub_names = [g for g, k in zip(gene_names, keep) if k]
    if sp.issparse(X):
        Xs = X[:, keep]
    else:
        Xs = X[:, keep]
    return Xs, sub_names, dropped


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_pca_embedding(X, gene_names, n_components: int = 10) -> PCAResult:
    """Fit a centred PCA and return scores plus the loadings."""
    if sp.issparse(X):
        Xd = np.asarray(X.todense(), dtype=np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)
    n_components = int(min(n_components, Xd.shape[1] - 1, Xd.shape[0] - 1))
    pca = PCA(n_components=n_components, svd_solver="auto")
    scores = pca.fit_transform(Xd).astype(np.float32)
    return PCAResult(
        scores=scores,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_,
        gene_names=list(gene_names),
    )


# ---------------------------------------------------------------------------
# HVG-vs-all comparison
# ---------------------------------------------------------------------------

def _compute_pc3_stripe_metrics(scores3: np.ndarray) -> dict:
    """Cheap geometry checks on PC1-3: stripe strength, great-circle R^2,
    anisotropy. Imports are local to keep startup fast."""
    from .sphere_stats import (
        _as_unit, fit_great_circle, spherical_anisotropy, stripe_strength_score
    )
    unit = _as_unit(scores3)
    out = {}
    try:
        out["stripe_strength"] = float(stripe_strength_score(unit)["score"])
    except Exception:
        out["stripe_strength"] = np.nan
    try:
        out["great_circle_R2"] = float(fit_great_circle(unit)["r_squared"])
    except Exception:
        out["great_circle_R2"] = np.nan
    try:
        out["anisotropy_linear"] = float(spherical_anisotropy(unit)["linear"])
    except Exception:
        out["anisotropy_linear"] = np.nan
    return out


def compare_hvg_vs_all_gene_pca(X, gene_names, *, n_top_genes: int = 2000,
                                n_components: int = 3,
                                pseudotime: np.ndarray | None = None,
                                random_seed: int = 0,
                                include_random: bool = True,
                                exclude_mito_ribo: bool = True) -> pd.DataFrame:
    """
    Run PCA under several gene-set choices and report PC1-3 metrics.

    Configurations:
        - "hvg": top-`n_top_genes` HVGs.
        - "all": every gene as-is.
        - "no_mito_ribo": all genes minus mito/ribo (only differs from "all"
          when gene names are symbol-style).
        - "random_match": random gene set of size `n_top_genes` (sanity null).
    """
    rows = []
    cfgs = [("hvg", "hvg"), ("all", "all")]
    if exclude_mito_ribo:
        cfgs.append(("no_mito_ribo", "no_mito_ribo"))
    if include_random and len(gene_names) > n_top_genes:
        cfgs.append(("random_match", "random_match"))

    for name, kind in cfgs:
        if kind == "hvg":
            idx, sub_names, Xs = select_hvgs(X, gene_names, n_top_genes)
        elif kind == "all":
            sub_names = list(gene_names); Xs = X
        elif kind == "no_mito_ribo":
            Xs, sub_names, _ = filter_genes(X, gene_names, True, True, False)
        elif kind == "random_match":
            rng = np.random.default_rng(random_seed)
            ids = np.sort(rng.choice(len(gene_names), n_top_genes, replace=False))
            sub_names = [gene_names[i] for i in ids]
            Xs = X[:, ids] if sp.issparse(X) else X[:, ids]
        else:
            continue

        pca = fit_pca_embedding(Xs, sub_names, n_components=max(3, n_components))
        scores3 = pca.scores[:, :3]
        row = {
            "config": name,
            "n_genes": pca.components.shape[1],
            "explained_var_PC1": float(pca.explained_variance_ratio[0]),
            "explained_var_PC2": float(pca.explained_variance_ratio[1]) if pca.explained_variance_ratio.size > 1 else np.nan,
            "explained_var_PC3": float(pca.explained_variance_ratio[2]) if pca.explained_variance_ratio.size > 2 else np.nan,
        }
        row.update(_compute_pc3_stripe_metrics(scores3))
        if pseudotime is not None:
            from .sphere_stats import _as_unit
            unit = _as_unit(scores3)
            theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
            mask = ~np.isnan(pseudotime)
            if mask.sum() > 5:
                row["theta_pseudotime_corr_spearman"] = float(
                    pd.Series(theta[mask]).corr(pd.Series(np.asarray(pseudotime)[mask]),
                                                method="spearman")
                )
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PCAResult",
    "normalize_log1p",
    "select_hvgs",
    "filter_genes",
    "fit_pca_embedding",
    "compare_hvg_vs_all_gene_pca",
]
