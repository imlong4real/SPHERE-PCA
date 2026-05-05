"""
Gene perturbation vector fields on the spherical PCA embedding.

CAVEAT (read first). This module performs a *fixed-PCA-loading sensitivity
analysis*, not a causal perturbation. We zero or double a single gene's
expression in the input matrix and re-project under the *same* (already-fitted)
PCA loadings, then measure how each cell's coordinates shift on the unit
sphere. Three things this is NOT:

1. It is not an in-silico knockout. We do not model gene regulatory feedback;
   silencing one gene in vivo would also change the expression of its
   targets, but that is invisible to a fixed-loading projection.
2. It is not a re-fit of the manifold. The loadings stay constant; what
   moves is each cell's location *relative to the existing axes*.
3. It is not symmetric. Zeroing and doubling have different magnitudes
   (zeroing's range is bounded by current expression; doubling is unbounded).

What it *does* measure: how much a gene's expression contributes to the
spherical embedding of each cell, decomposed into radial vs. tangent
(stripe-aligned vs. cross-stripe) components. Genes whose perturbation
moves cells across stripes are candidate switch-like regulators of the
geometry; genes whose perturbation moves cells along stripes contribute
to gradient-like programs. Both interpretations are hypotheses to be
tested with real perturbation data (CRISPR / RNAi / lineage tracing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sphere_stats import _as_unit


# ---------------------------------------------------------------------------
# Core projection / perturbation primitives
# ---------------------------------------------------------------------------

def project_expression_to_pca(X: np.ndarray,
                              pca_components: np.ndarray,
                              pca_mean: np.ndarray | None = None) -> np.ndarray:
    """
    Project an expression matrix to PC scores under fixed PCA loadings.

    Args:
        X: (N, G) expression matrix.
        pca_components: (k, G) loadings in sklearn convention
            (``PCA().components_``). Each row is one principal direction in
            gene space.
        pca_mean: (G,) per-gene mean used during PCA fit. If None, assumes
            data are already centred.

    Returns:
        (N, k) PC scores: ``(X - pca_mean) @ pca_components.T``.
    """
    X = np.asarray(X, dtype=float)
    pca_components = np.asarray(pca_components, dtype=float)
    if pca_components.ndim != 2 or pca_components.shape[1] != X.shape[1]:
        raise ValueError(
            f"pca_components shape {pca_components.shape} incompatible with X shape {X.shape}"
        )
    if pca_mean is None:
        pca_mean = np.zeros(X.shape[1])
    pca_mean = np.asarray(pca_mean, dtype=float)
    if pca_mean.shape != (X.shape[1],):
        raise ValueError(f"pca_mean shape {pca_mean.shape} does not match {X.shape[1]} genes")
    return (X - pca_mean) @ pca_components.T


def perturb_gene_expression(X,
                            gene_names,
                            gene: str,
                            mode: str = "zero"):
    """
    Return a copy of X with a single gene's expression replaced.

    Args:
        X: (N, G) expression as numpy array or DataFrame. The original is
            not mutated.
        gene_names: list/array of G gene names matching the columns of X.
            Ignored when X is a DataFrame (its columns are used).
        gene: name of the gene to perturb. Must be present.
        mode: one of:
            ``"zero"``: set the gene's column to 0.
            ``"double"``: multiply the gene's column by 2.

    Returns:
        Perturbed copy with the same type (ndarray or DataFrame) as X.

    Raises:
        ValueError: if gene is missing or mode is unknown.
    """
    if mode not in ("zero", "double"):
        raise ValueError(f"unknown mode {mode!r}; expected 'zero' or 'double'")

    if isinstance(X, pd.DataFrame):
        if gene not in X.columns:
            raise ValueError(f"gene {gene!r} not in DataFrame columns")
        out = X.copy()
        if mode == "zero":
            out[gene] = 0.0
        else:
            out[gene] = X[gene].values * 2.0
        return out

    X_arr = np.asarray(X, dtype=float)
    gene_names = list(gene_names)
    if gene not in gene_names:
        raise ValueError(f"gene {gene!r} not in gene_names")
    idx = gene_names.index(gene)
    out = X_arr.copy()
    if mode == "zero":
        out[:, idx] = 0.0
    else:
        out[:, idx] = X_arr[:, idx] * 2.0
    return out


def compute_gene_perturbation_vectors(X,
                                      gene_names,
                                      genes,
                                      pca_components: np.ndarray,
                                      pca_mean: np.ndarray | None = None,
                                      normalize_to_sphere: bool = False) -> dict:
    """
    For each requested gene, compute the original, zeroed, and doubled
    PC-space coordinates and the corresponding displacement vectors.

    Args:
        X: (N, G) expression (ndarray or DataFrame).
        gene_names: G gene names (ignored if X is a DataFrame).
        genes: iterable of gene names to perturb.
        pca_components: (k, G) sklearn loadings.
        pca_mean: (G,) gene means; default zero.
        normalize_to_sphere: if True and k == 3, L2-normalise each row of
            the PC scores onto S^2 *before* computing displacement vectors.
            This makes the analysis non-linear and lets it pick up the
            radial-projection curvature visible in the existing pipeline.
            Without it, the linear PCA projection guarantees
            ``(zeroed + doubled) / 2 == original`` exactly.

    Returns:
        dict ``{gene: {key: array}}`` where each per-gene dict has:
            - ``original``: (N, k) original PC scores (same for all genes).
            - ``zeroed``:   (N, k) PC scores with this gene set to 0.
            - ``doubled``:  (N, k) PC scores with this gene multiplied by 2.
            - ``delta_zero``:   ``zeroed - original``.
            - ``delta_double``: ``doubled - original``.
            - ``delta_double_minus_zero``: ``doubled - zeroed``, the
              symmetric range, twice the per-cell sensitivity in the
              linear case.
    """
    if isinstance(X, pd.DataFrame):
        gene_names = list(X.columns)
        X_arr = X.values
        df_input = True
    else:
        X_arr = np.asarray(X, dtype=float)
        gene_names = list(gene_names)
        df_input = False

    if X_arr.shape[1] != len(gene_names):
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match X columns {X_arr.shape[1]}"
        )

    Q = project_expression_to_pca(X_arr, pca_components, pca_mean)
    if normalize_to_sphere:
        if Q.shape[1] != 3:
            raise ValueError("normalize_to_sphere=True requires exactly 3 PCs")
        Q_used = _as_unit(Q)
    else:
        Q_used = Q

    results: dict = {}
    for g in genes:
        if g not in gene_names:
            raise ValueError(f"gene {g!r} not in gene_names")
        if df_input:
            X_z = perturb_gene_expression(X, gene_names, g, mode="zero").values
            X_d = perturb_gene_expression(X, gene_names, g, mode="double").values
        else:
            X_z = perturb_gene_expression(X_arr, gene_names, g, mode="zero")
            X_d = perturb_gene_expression(X_arr, gene_names, g, mode="double")
        Z = project_expression_to_pca(X_z, pca_components, pca_mean)
        D = project_expression_to_pca(X_d, pca_components, pca_mean)
        if normalize_to_sphere:
            Z = _as_unit(Z)
            D = _as_unit(D)
        results[g] = {
            "original": Q_used,
            "zeroed": Z,
            "doubled": D,
            "delta_zero": Z - Q_used,
            "delta_double": D - Q_used,
            "delta_double_minus_zero": D - Z,
        }
    return results


# ---------------------------------------------------------------------------
# Ranking and decomposition
# ---------------------------------------------------------------------------

def rank_genes_by_perturbation_magnitude(perturbation_results: dict,
                                         metric: str = "symmetric_range") -> pd.DataFrame:
    """
    Rank genes by aggregate displacement magnitude across cells.

    Args:
        perturbation_results: output of ``compute_gene_perturbation_vectors``.
        metric: which Frobenius norm to use:
            - ``"symmetric_range"``: ``||doubled - zeroed||_F``. Default.
              Independent of asymmetry, equal to twice the linear contribution
              of the gene to the embedding.
            - ``"doubled"``: ``||doubled - original||_F``.
            - ``"zeroed"``:  ``||zeroed - original||_F``.
            - ``"asymmetry"``: ``||(doubled + zeroed)/2 - original||_F``,
              the metric used in the exploratory notebook. Numerically zero
              for purely-linear projection; non-trivial only when
              ``normalize_to_sphere=True`` was used in the perturbation step.

    Returns:
        DataFrame with columns ``['gene', 'score']``, sorted descending.
    """
    rows = []
    for g, r in perturbation_results.items():
        if metric == "symmetric_range":
            v = r["doubled"] - r["zeroed"]
        elif metric == "doubled":
            v = r["delta_double"]
        elif metric == "zeroed":
            v = r["delta_zero"]
        elif metric == "asymmetry":
            v = (r["doubled"] + r["zeroed"]) / 2.0 - r["original"]
        else:
            raise ValueError(f"unknown metric {metric!r}")
        rows.append({"gene": g, "score": float(np.linalg.norm(v, ord="fro"))})
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


def decompose_perturbation_vectors(coords: np.ndarray,
                                   vectors: np.ndarray) -> dict:
    """
    Decompose displacement vectors into radial + tangent components, and
    further split the tangent piece into latitude (theta) and longitude
    (phi) components in the local spherical basis.

    Args:
        coords: (N, 3) cell positions in PC space. Need not be unit
            vectors; the radial direction is computed from each row.
        vectors: (N, 3) displacement vectors.

    Returns:
        dict with:
            - ``radial``: (N,) signed component along the outward radial.
            - ``tangent``: (N, 3) tangent vectors (perpendicular to radial).
            - ``theta_component``: (N,) signed component in the e_theta
              direction (positive = toward the south pole).
            - ``phi_component``: (N,) signed component in e_phi (eastward).
            - ``tangent_magnitude``: (N,) ``||tangent||``.

    Conventions: x = sin(theta) cos(phi); y = sin(theta) sin(phi); z = cos(theta).
    """
    coords = np.asarray(coords, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if coords.shape != vectors.shape:
        raise ValueError("coords and vectors must have matching shape")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("coords contains zero-norm rows")
    rhat = coords / norms

    radial = np.einsum("ij,ij->i", vectors, rhat)
    tangent = vectors - radial[:, None] * rhat

    x, y, z = rhat[:, 0], rhat[:, 1], rhat[:, 2]
    cos_theta = np.clip(z, -1.0, 1.0)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
    phi = np.arctan2(y, x)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    e_theta = np.column_stack(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta]
    )
    e_phi = np.column_stack(
        [-sin_phi, cos_phi, np.zeros_like(sin_phi)]
    )

    theta_comp = np.einsum("ij,ij->i", tangent, e_theta)
    phi_comp = np.einsum("ij,ij->i", tangent, e_phi)

    return {
        "radial": radial,
        "tangent": tangent,
        "theta_component": theta_comp,
        "phi_component": phi_comp,
        "tangent_magnitude": np.linalg.norm(tangent, axis=1),
    }


def identify_sensitive_cells(perturbation_results: dict,
                             gene: str,
                             top_n: int = 50,
                             metric: str = "symmetric_range") -> np.ndarray:
    """
    Per-gene: indices of the cells whose embedding moves the most.

    Args:
        perturbation_results: output of ``compute_gene_perturbation_vectors``.
        gene: which gene to inspect.
        top_n: how many top-ranked cell indices to return.
        metric: ``"symmetric_range"`` (default), ``"doubled"``, or ``"zeroed"``.

    Returns:
        (top_n,) array of cell indices, sorted by descending magnitude.
    """
    if gene not in perturbation_results:
        raise KeyError(f"gene {gene!r} not in results")
    r = perturbation_results[gene]
    if metric == "symmetric_range":
        v = r["doubled"] - r["zeroed"]
    elif metric == "doubled":
        v = r["delta_double"]
    elif metric == "zeroed":
        v = r["delta_zero"]
    else:
        raise ValueError(f"unknown metric {metric!r}")
    mag = np.linalg.norm(v, axis=1)
    n = min(top_n, mag.size)
    return np.argsort(mag)[::-1][:n]


# ---------------------------------------------------------------------------
# Visualisation and gene clustering
# ---------------------------------------------------------------------------

def plot_perturbation_vector_field(perturbation_results: dict,
                                   gene: str,
                                   frac: float = 0.015,
                                   figsize: tuple = (14, 6),
                                   seed: int = 0,
                                   ax=None,
                                   clip: float = 0.1):
    """
    Equirectangular plot of the per-cell perturbation vector field for one
    gene. Blue lines: zeroed - original (where zeroing pushes cells); red:
    doubled - original. Coordinates assumed to be on the unit sphere; if
    ``perturbation_results`` was computed without ``normalize_to_sphere=True``,
    inputs are projected radially before plotting.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if gene not in perturbation_results:
        raise KeyError(f"gene {gene!r} not in results")
    r = perturbation_results[gene]
    Q = _as_unit(r["original"])
    D = _as_unit(r["doubled"])
    Z = _as_unit(r["zeroed"])

    def _to_lonlat(c):
        x, y, z = c[:, 0], c[:, 1], c[:, 2]
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        return lon, lat

    lonQ, latQ = _to_lonlat(Q)
    lonD, latD = _to_lonlat(D)
    lonZ, latZ = _to_lonlat(Z)
    # Wrap longitude across the dateline so vector segments don't span 360.
    for lon_arr in (lonD, lonZ):
        lon_arr[(lon_arr - lonQ) > 180] -= 360
        lon_arr[(lon_arr - lonQ) < -180] += 360
    # Optional displacement clipping (per the original notebook).
    if clip is not None:
        for lon_arr, lat_arr in ((lonD, latD), (lonZ, latZ)):
            np.clip(lon_arr - lonQ, -clip * 180, clip * 180, out=lon_arr - lonQ)

    rng = np.random.default_rng(seed)
    mask = rng.random(Q.shape[0]) < frac

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot(lonQ, latQ, ".", color="lightgrey", markersize=0.5, alpha=0.4)
    for i in np.where(mask)[0]:
        ax.plot([lonQ[i], lonZ[i]], [latQ[i], latZ[i]],
                "-", color="#2222CC", lw=1, alpha=0.7)
        ax.plot([lonQ[i], lonD[i]], [latQ[i], latD[i]],
                "-", color="#CC2222", lw=1, alpha=0.7)
        ax.plot(lonQ[i], latQ[i], ".k", markersize=2, alpha=0.5)
    ax.set_xlim(-185, 185)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Perturbation vector field: {gene}")
    return ax


def cluster_genes_by_perturbation_signature(perturbation_results: dict,
                                            threshold: float = 0.01,
                                            mode: str = "zeroed_sensitive") -> dict:
    """
    Group genes by *which* cells they perturb, not by how strongly.
    For each gene, build a boolean bitmap over cells in the requested
    sensitivity category; return the bitmap and the gene-by-gene shared-cell
    overlap matrix (B B^T) for downstream clustering.

    Args:
        perturbation_results: output of ``compute_gene_perturbation_vectors``.
        threshold: norm cutoff defining "unperturbed" cells.
        mode: which bitmap to compute:
            - ``"doubled_sensitive"``: ``||delta_double|| > ||delta_zero||``
              with at least one above threshold.
            - ``"zeroed_sensitive"``: ``||delta_zero|| > ||delta_double||``
              with at least one above threshold.
            - ``"any_perturbed"``: at least one above threshold.

    Returns:
        dict with:
            - ``bitmap``: (n_genes, n_cells) boolean.
            - ``genes``: list of gene names (rows of bitmap).
            - ``shared_counts``: (n_genes, n_genes) DataFrame of co-perturbed
              cell counts.
    """
    if mode not in ("doubled_sensitive", "zeroed_sensitive", "any_perturbed"):
        raise ValueError(f"unknown mode {mode!r}")

    genes = list(perturbation_results.keys())
    if not genes:
        raise ValueError("perturbation_results is empty")
    n_cells = perturbation_results[genes[0]]["original"].shape[0]
    bitmap = np.zeros((len(genes), n_cells), dtype=bool)
    for i, g in enumerate(genes):
        r = perturbation_results[g]
        nz = np.linalg.norm(r["delta_zero"], axis=1)
        nd = np.linalg.norm(r["delta_double"], axis=1)
        small = (nz <= threshold) & (nd <= threshold)
        if mode == "doubled_sensitive":
            bitmap[i] = (nd > nz) & ~small
        elif mode == "zeroed_sensitive":
            bitmap[i] = (nz > nd) & ~small
        else:
            bitmap[i] = ~small

    shared = bitmap.astype(int) @ bitmap.astype(int).T
    return {
        "bitmap": bitmap,
        "genes": genes,
        "shared_counts": pd.DataFrame(shared, index=genes, columns=genes),
    }
