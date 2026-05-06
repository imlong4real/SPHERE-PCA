"""PC-count and feature-selection robustness for spherical PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .preprocessing import compare_hvg_vs_all_gene_pca, fit_pca_embedding
from .sphere_stats import _as_unit, fit_great_circle, spherical_anisotropy
from .stripe import multi_stripe_strength_score


def _corr(a, b):
    if b is None:
        return np.nan
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return np.nan
    return float(pd.Series(a[mask]).corr(pd.Series(b[mask]), method="spearman"))


def compute_scree_summary(X, max_pcs=50):
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    k = int(min(max_pcs, X.shape[0] - 1, X.shape[1] - 1))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    return pd.DataFrame({
        "pc": np.arange(1, k + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": np.cumsum(evr),
    })


def estimate_intrinsic_dimension(X, methods=("participation_ratio", "pca_elbow")):
    scree = compute_scree_summary(X, max_pcs=50)
    evr = scree["explained_variance_ratio"].to_numpy()
    out = {}
    if "participation_ratio" in methods:
        out["participation_ratio"] = float((evr.sum() ** 2) / np.sum(evr ** 2))
    if "pca_elbow" in methods:
        if evr.size < 3:
            out["pca_elbow"] = int(evr.size)
        else:
            second = np.diff(evr, n=2)
            out["pca_elbow"] = int(np.argmax(second) + 2)
    return out


def _metric_row(name, scores, pseudotime=None, cytotrace=None):
    unit = _as_unit(scores[:, :3])
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    gc = fit_great_circle(unit)
    an = spherical_anisotropy(unit)
    ms = multi_stripe_strength_score(unit)
    return {
        "config": name,
        "global_longitude_concentration": ms["global_longitude_concentration"],
        "multi_stripe_strength": ms["multi_stripe_strength"],
        "n_stripes": ms["n_stripes"],
        "great_circle_r2": float(gc["r_squared"]),
        "anisotropy_linear": float(an["linear"]),
        "anisotropy_planar": float(an["planar"]),
        "anisotropy_spherical": float(an["spherical"]),
        "theta_pseudotime_spearman": _corr(theta, pseudotime),
        "theta_cytotrace_spearman": _corr(theta, cytotrace),
    }


def compare_pc_count_spherical_geometry(X, gene_names=None, metadata=None,
                                        pc_counts=(3, 5, 10, 20),
                                        pseudotime=None, cytotrace=None):
    max_pc = int(max(pc_counts))
    pca = fit_pca_embedding(X, gene_names or [f"g{i}" for i in range(X.shape[1])],
                            n_components=max_pc)
    rows = []
    for k in pc_counts:
        if pca.scores.shape[1] >= min(k, 3):
            rows.append(_metric_row(f"PC1-{k}_truncated_to_3", pca.scores[:, :3],
                                    pseudotime, cytotrace))
    df = pd.DataFrame(rows)
    for i in range(min(20, pca.explained_variance_ratio.size)):
        df[f"explained_var_PC{i+1}"] = float(pca.explained_variance_ratio[i])
    return df


def compare_top3_vs_random3_pcs(X, n_random=100, max_pcs=20, random_state=0,
                                pseudotime=None, cytotrace=None):
    pca = fit_pca_embedding(X, [f"g{i}" for i in range(X.shape[1])], n_components=max_pcs)
    rows = [_metric_row("PC1-3", pca.scores[:, :3], pseudotime, cytotrace)]
    rng = np.random.default_rng(random_state)
    upper = pca.scores.shape[1]
    for i in range(n_random):
        ids = np.sort(rng.choice(upper, 3, replace=False))
        rows.append(_metric_row(f"random3_{i:03d}_{'-'.join(map(str, ids + 1))}",
                                pca.scores[:, ids], pseudotime, cytotrace))
    return pd.DataFrame(rows)


def compare_hvg_vs_all_gene_pc_geometry(X, gene_names, metadata=None,
                                        pseudotime=None, cytotrace=None):
    df = compare_hvg_vs_all_gene_pca(X, gene_names, pseudotime=pseudotime)
    return df.rename(columns={"stripe_strength": "global_longitude_concentration"})


__all__ = [
    "compute_scree_summary",
    "estimate_intrinsic_dimension",
    "compare_pc_count_spherical_geometry",
    "compare_top3_vs_random3_pcs",
    "compare_hvg_vs_all_gene_pc_geometry",
]
