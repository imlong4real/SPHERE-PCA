"""
Per-gene geometry on the spherical PCA embedding (H5).

We compute, for each gene, a tangent-space gradient of expression with
respect to position on S^2. The gradient is then decomposed in a local
(theta, phi) basis aligned with the chosen root pole. Genes whose gradient
is dominated by the phi (longitudinal) component move expression *across*
stripes; genes dominated by the theta (latitudinal) component move along
the trajectory.

Implementation note: an exact geodesic-gradient fit (as in
``sphere_stats.geodesic_gradient``) is O(N k) per gene, which becomes
expensive on 64k+ cells x thousands of genes. For the bulk H5 ranking we
use a much cheaper but still principled approach: aggregate per-cell
expression onto a coarse (theta, phi) grid and ask how strongly the gene
varies along phi vs. theta. The exact gradient remains available for
top-N genes after ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .sphere_stats import geodesic_gradient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coords_to_spherical(unit: np.ndarray):
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))   # 0 at north pole
    phi = np.arctan2(y, x)                     # in (-pi, pi]
    return theta, phi


def _bin_grid(theta, phi, n_theta: int = 18, n_phi: int = 36):
    """Return (i, j) grid indices for each cell."""
    theta_bins = np.linspace(0, np.pi, n_theta + 1)
    phi_bins = np.linspace(-np.pi, np.pi, n_phi + 1)
    i = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_theta - 1)
    j = np.clip(np.digitize(phi, phi_bins) - 1, 0, n_phi - 1)
    return i, j, n_theta, n_phi


# ---------------------------------------------------------------------------
# Bulk per-gene gradient ranking
# ---------------------------------------------------------------------------

def compute_gene_geodesic_gradients(X, gene_names, coords, k: int = 15,
                                     n_theta: int = 18, n_phi: int = 36):
    """
    Coarse-grid surrogate for per-gene gradients on S^2.

    For each gene we average expression within each (theta, phi) bin, then
    compute first differences along phi (within a theta row) and along theta
    (within a phi column). The summed |delta| in each direction gives a
    "phi variation" and "theta variation" score per gene.

    Args:
        X: (N, G) sparse or dense expression. Already normalised.
        gene_names: G gene symbols.
        coords: (N, 3) PC scores; will be unit-normalised.
        k: present for API compatibility with the docstring; unused here.
        n_theta, n_phi: grid resolution.

    Returns:
        DataFrame with columns ``gene``, ``theta_variation``,
        ``phi_variation``, ``mean_expr``, ``stripe_score`` (= phi/theta+eps).
    """
    coords = np.asarray(coords, dtype=float)
    nrm = np.linalg.norm(coords, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    unit = coords / nrm
    theta, phi = _coords_to_spherical(unit)
    i_idx, j_idx, n_t, n_p = _bin_grid(theta, phi, n_theta, n_phi)
    flat_idx = i_idx * n_p + j_idx
    n_bins = n_t * n_p

    counts = np.bincount(flat_idx, minlength=n_bins).astype(np.float64)

    if sp.issparse(X):
        # Build aggregator: for each gene column, sum into bins.
        # rows = bins, cols = genes; A = bin_indicator.T @ X
        bin_indicator = sp.csr_matrix(
            (np.ones(coords.shape[0]),
             (flat_idx, np.arange(coords.shape[0]))),
            shape=(n_bins, coords.shape[0]),
        )
        agg = bin_indicator @ X  # (n_bins, G)
        agg = agg.toarray() if sp.issparse(agg) else np.asarray(agg)
    else:
        Xa = np.asarray(X, dtype=np.float64)
        agg = np.zeros((n_bins, Xa.shape[1]), dtype=np.float64)
        np.add.at(agg, flat_idx, Xa)

    safe_counts = np.where(counts > 0, counts, 1.0)
    mean_grid = agg / safe_counts[:, None]
    # reshape to (n_t, n_p, G)
    mean_grid = mean_grid.reshape(n_t, n_p, -1)

    # phi-variation: sum |delta along phi axis| (with wrap)
    dphi = np.diff(mean_grid, axis=1, append=mean_grid[:, :1, :])
    phi_var = np.abs(dphi).sum(axis=(0, 1))

    # theta-variation: sum |delta along theta axis|
    dtheta = np.diff(mean_grid, axis=0)
    theta_var = np.abs(dtheta).sum(axis=(0, 1))

    mean_expr = mean_grid.mean(axis=(0, 1))
    eps = 1e-6
    stripe_score = phi_var / (theta_var + eps)
    return pd.DataFrame({
        "gene": list(gene_names),
        "theta_variation": theta_var.astype(float),
        "phi_variation": phi_var.astype(float),
        "mean_expr": mean_expr.astype(float),
        "stripe_score": stripe_score.astype(float),
    })


def decompose_gene_gradient_relative_to_stripes(coords, gene_gradient: np.ndarray) -> dict:
    """Project a per-cell gene-gradient field onto the local theta and phi
    tangent vectors, returning summary statistics.

    Args:
        coords: (N, 3) unit vectors.
        gene_gradient: (N, 3) tangent vectors from
            ``sphere_stats.geodesic_gradient``.

    Returns:
        dict with mean theta-component, mean phi-component, and mean tangent
        magnitude.
    """
    unit = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    cos_t = np.clip(z, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = np.arctan2(y, x)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    e_theta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    e_phi = np.column_stack([-sin_p, cos_p, np.zeros_like(sin_p)])
    th = np.einsum("ij,ij->i", gene_gradient, e_theta)
    ph = np.einsum("ij,ij->i", gene_gradient, e_phi)
    mag = np.linalg.norm(gene_gradient, axis=1)
    return {
        "mean_theta": float(np.mean(th)),
        "mean_phi": float(np.mean(ph)),
        "mean_abs_theta": float(np.mean(np.abs(th))),
        "mean_abs_phi": float(np.mean(np.abs(ph))),
        "mean_magnitude": float(np.mean(mag)),
    }


def _theta_phi_basis(coords):
    unit = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    cos_t = np.clip(z, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    phi = np.arctan2(y, x)
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    e_theta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    e_phi = np.column_stack([-sin_p, cos_p, np.zeros_like(sin_p)])
    return e_theta, e_phi


def compute_per_cell_gene_geodesic_gradient(X, gene_names, coords, genes=None, k=30,
                                            max_cells: int = 4000):
    """Exact local tangent gradients for selected genes on S2.

    This is the high-resolution H5 path. For large datasets the calculation is
    performed on a deterministic cell subsample to keep the local least-squares
    fits tractable and reproducible.
    """
    coords = np.asarray(coords, dtype=float)
    nrm = np.linalg.norm(coords, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    unit = coords / nrm
    n = unit.shape[0]
    if n > max_cells:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, max_cells, replace=False))
        unit_use = unit[idx]
        X_use = X[idx]
    else:
        idx = np.arange(n)
        unit_use = unit
        X_use = X

    gene_names = list(gene_names)
    if genes is None:
        gene_idx = list(range(len(gene_names)))
    else:
        wanted = set(genes)
        gene_idx = [i for i, g in enumerate(gene_names) if g in wanted]
    rows = []
    gradients = {}
    e_theta, e_phi = _theta_phi_basis(unit_use)
    for gi in gene_idx:
        expr = np.asarray(X_use[:, gi].todense()).ravel() if sp.issparse(X_use) else np.asarray(X_use[:, gi]).ravel()
        if np.nanstd(expr) <= 1e-12:
            continue
        grad = geodesic_gradient(unit_use, expr, k=min(k, max(2, unit_use.shape[0] - 1)))
        th = np.einsum("ij,ij->i", grad, e_theta)
        ph = np.einsum("ij,ij->i", grad, e_phi)
        expressed = expr > 0
        ubiq = float(expressed.mean())
        rows.append({
            "gene": gene_names[gi],
            "mean_abs_theta_component": float(np.mean(np.abs(th))),
            "mean_abs_phi_component": float(np.mean(np.abs(ph))),
            "localized_high_phi_component": float(np.quantile(np.abs(ph), 0.9)),
            "phi_theta_ratio": float(np.mean(np.abs(ph)) / (np.mean(np.abs(th)) + 1e-9)),
            "mean_expr": float(np.mean(expr)),
            "expr_specificity": float(1.0 - ubiq),
            "ubiquitous_fraction": ubiq,
        })
        gradients[gene_names[gi]] = grad
    return pd.DataFrame(rows), gradients, idx


def decompose_gene_gradient_theta_phi(coords, gradients):
    """Decompose a dict of per-cell gradient fields into theta/phi summaries."""
    e_theta, e_phi = _theta_phi_basis(np.asarray(coords, dtype=float))
    rows = []
    for gene, grad in gradients.items():
        th = np.einsum("ij,ij->i", grad, e_theta)
        ph = np.einsum("ij,ij->i", grad, e_phi)
        rows.append({
            "gene": gene,
            "mean_abs_theta_component": float(np.mean(np.abs(th))),
            "mean_abs_phi_component": float(np.mean(np.abs(ph))),
            "phi_theta_ratio": float(np.mean(np.abs(ph)) / (np.mean(np.abs(th)) + 1e-9)),
        })
    return pd.DataFrame(rows)


def filter_housekeeping_and_low_specificity_genes(df, gene_col="gene",
                                                  min_specificity: float = 0.05,
                                                  max_ubiquitous_fraction: float = 0.95):
    """Penalize canonical housekeeping and low-specificity genes."""
    out = df.copy()
    genes = out[gene_col].astype(str)
    hk = genes.str.match(r"^(RPS|RPL|MRPS|MRPL|MT-|mt-|ACTB$|GAPDH$|B2M$|MALAT1$|TUBB|EEF|HSP)", case=False)
    out["housekeeping_like"] = hk.astype(bool)
    if "expr_specificity" not in out.columns:
        out["expr_specificity"] = 1.0
    if "ubiquitous_fraction" not in out.columns:
        out["ubiquitous_fraction"] = 0.0
    penalty = np.where(out["housekeeping_like"], 0.25, 1.0)
    penalty *= np.where(out["expr_specificity"] < min_specificity, 0.35, 1.0)
    penalty *= np.where(out["ubiquitous_fraction"] > max_ubiquitous_fraction, 0.35, 1.0)
    out["specificity_penalty"] = penalty
    return out


def rank_stripe_boundary_genes_per_cell(gradient_df, top_n=50):
    df = filter_housekeeping_and_low_specificity_genes(gradient_df)
    df["stripe_boundary_score"] = (
        df["mean_abs_phi_component"]
        * df["localized_high_phi_component"]
        * df["phi_theta_ratio"]
        * df["specificity_penalty"]
    )
    return df.sort_values("stripe_boundary_score", ascending=False).head(top_n).reset_index(drop=True)


def rank_along_trajectory_genes_per_cell(gradient_df, top_n=50):
    df = filter_housekeeping_and_low_specificity_genes(gradient_df)
    df["along_trajectory_score"] = (
        df["mean_abs_theta_component"]
        / (df["mean_abs_phi_component"] + 1e-9)
        * df["specificity_penalty"]
    )
    return df.sort_values("along_trajectory_score", ascending=False).head(top_n).reset_index(drop=True)


def plot_gene_gradient_on_sphere(coords, expr, ax=None):
    import matplotlib.pyplot as plt
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    if ax is None:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=expr, s=4, cmap="viridis")
    return ax, sc


def plot_gene_gradient_equirectangular(coords, expr, ax=None):
    import matplotlib.pyplot as plt
    coords = coords / np.linalg.norm(coords, axis=1, keepdims=True)
    theta, phi = _coords_to_spherical(coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(np.degrees(phi), np.degrees(theta), c=expr, s=4, cmap="viridis")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("theta (deg)")
    return ax, sc


def rank_stripe_boundary_genes(grad_df: pd.DataFrame, top_n: int = 50,
                                 min_mean_expr: float = 0.05) -> pd.DataFrame:
    """Genes with high phi (across-stripe) variation relative to theta."""
    df = grad_df[grad_df["mean_expr"] >= min_mean_expr].copy()
    df = df.sort_values("stripe_score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def rank_along_trajectory_genes(grad_df: pd.DataFrame, top_n: int = 50,
                                  min_mean_expr: float = 0.05) -> pd.DataFrame:
    """Genes with high theta (along-stripe / along-trajectory) variation."""
    df = grad_df[grad_df["mean_expr"] >= min_mean_expr].copy()
    df["trajectory_score"] = df["theta_variation"] / (df["phi_variation"] + 1e-6)
    df = df.sort_values("trajectory_score", ascending=False).head(top_n).reset_index(drop=True)
    return df


def plot_gene_gradient_field(coords, X, gene_names, gene: str, ax=None,
                              n_theta: int = 18, n_phi: int = 36):
    """Equirectangular heatmap of gene expression averaged on the S^2 grid."""
    import matplotlib.pyplot as plt
    if gene not in gene_names:
        raise KeyError(gene)
    g_idx = list(gene_names).index(gene)
    if sp.issparse(X):
        col = np.asarray(X[:, g_idx].todense()).ravel()
    else:
        col = np.asarray(X[:, g_idx]).ravel()

    nrm = np.linalg.norm(coords, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    unit = coords / nrm
    theta, phi = _coords_to_spherical(unit)
    i, j, n_t, n_p = _bin_grid(theta, phi, n_theta, n_phi)
    grid = np.zeros((n_t, n_p))
    counts = np.zeros((n_t, n_p))
    np.add.at(grid, (i, j), col)
    np.add.at(counts, (i, j), 1)
    counts[counts == 0] = 1
    grid = grid / counts

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        grid,
        extent=(-180, 180, 180, 0),  # phi (lon), theta (lat from north pole)
        aspect="auto", cmap="viridis",
    )
    ax.set_xlabel("phi (deg, longitude)")
    ax.set_ylabel("theta (deg, 0=north)")
    ax.set_title(f"{gene} mean expression on S^2 grid")
    return ax, im


__all__ = [
    "compute_gene_geodesic_gradients",
    "compute_per_cell_gene_geodesic_gradient",
    "decompose_gene_gradient_theta_phi",
    "decompose_gene_gradient_relative_to_stripes",
    "rank_stripe_boundary_genes",
    "rank_stripe_boundary_genes_per_cell",
    "rank_along_trajectory_genes",
    "rank_along_trajectory_genes_per_cell",
    "filter_housekeeping_and_low_specificity_genes",
    "plot_gene_gradient_field",
    "plot_gene_gradient_on_sphere",
    "plot_gene_gradient_equirectangular",
]
