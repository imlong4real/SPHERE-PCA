"""
Cross-dataset comparisons of spherical embeddings.

Two PCA-on-sphere embeddings of homologous biology (replicates, conditions,
species) are compared after a constrained orthogonal Procrustes alignment.
The residuals after alignment are interpretable: they measure how much of
the geometric structure is *not* explained by a global rotation, which is
the relevant null for "is this conserved?"
"""

from __future__ import annotations

import numpy as np

from .sphere_stats import _as_unit


def _label_centroids(coords: np.ndarray, labels, common):
    coords = _as_unit(coords)
    labels = np.asarray(labels)
    cents = np.zeros((len(common), 3))
    for i, lab in enumerate(common):
        mask = labels == lab
        if not np.any(mask):
            raise ValueError(f"label {lab!r} not present")
        v = coords[mask].mean(axis=0)
        n = np.linalg.norm(v)
        if n < 1e-12:
            # Degenerate cluster (antipodal cells average to 0); use first cell
            cents[i] = coords[mask][0]
        else:
            cents[i] = v / n
    return cents


def procrustes_align_spheres(coords_a: np.ndarray,
                             coords_b: np.ndarray,
                             labels_a,
                             labels_b,
                             common_labels: list[str] | None = None) -> dict:
    """
    Find the rotation R that best aligns dataset B's per-label centroids
    to dataset A's per-label centroids on S^2.

    Args:
        coords_a, coords_b: (Na, 3), (Nb, 3) unit vectors.
        labels_a, labels_b: per-cell label arrays.
        common_labels: optional list of labels to use as anchors. If None,
            uses the intersection of the two label sets.

    Returns:
        dict with:
            R: 3x3 rotation matrix (det = +1) such that coords_b @ R.T
                aligns to A.
            anchor_residuals: per-label angular residual (radians) after
                rotation.
            rmse: scalar RMS residual.
            common_labels: the labels actually used.

    Algorithm: Kabsch on label centroids, with sign-flip to enforce det(R)=+1.
    """
    if common_labels is None:
        common_labels = sorted(set(labels_a).intersection(set(labels_b)))
    if len(common_labels) < 3:
        raise ValueError("Need at least 3 common labels for unique 3D rotation")

    A = _label_centroids(coords_a, labels_a, common_labels)
    B = _label_centroids(coords_b, labels_b, common_labels)

    H = B.T @ A
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    aligned = B @ R.T
    cosang = np.clip(np.einsum("ij,ij->i", aligned, A), -1.0, 1.0)
    residuals = np.arccos(cosang)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "R": R,
        "anchor_residuals": dict(zip(common_labels, residuals.tolist())),
        "rmse": rmse,
        "common_labels": common_labels,
    }


def conserved_stripe_test(coords_a: np.ndarray,
                          labels_a,
                          coords_b: np.ndarray,
                          labels_b,
                          stripes: list[tuple[float, float]],
                          n_perm: int = 500,
                          rng: np.random.Generator | None = None) -> dict:
    """
    Test whether stripe definitions (longitudinal ranges) capture the
    same cell-type composition in two pre-aligned spherical embeddings.

    Pre-aligned means: coords_b has already been rotated by the R from
    ``procrustes_align_spheres`` (otherwise this test is meaningless).

    Args:
        coords_a, coords_b: (Na, 3), (Nb, 3) unit vectors, already aligned.
        labels_a, labels_b: per-cell labels.
        stripes: list of (lon_min_deg, lon_max_deg) tuples.
        n_perm: permutations for null distribution of label assignments.
        rng: optional Generator.

    Returns:
        dict with:
            jaccard: list of Jaccard indices (over label sets) per stripe.
            p_value: list of permutation p-values per stripe.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    coords_a = _as_unit(coords_a)
    coords_b = _as_unit(coords_b)
    lon_a = np.degrees(np.arctan2(coords_a[:, 1], coords_a[:, 0]))
    lon_b = np.degrees(np.arctan2(coords_b[:, 1], coords_b[:, 0]))
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    jaccards = []
    pvals = []
    for lo, hi in stripes:
        in_a = (lon_a >= lo) & (lon_a <= hi)
        in_b = (lon_b >= lo) & (lon_b <= hi)
        set_a = set(labels_a[in_a].tolist())
        set_b = set(labels_b[in_b].tolist())
        if not set_a and not set_b:
            jaccards.append(np.nan)
            pvals.append(np.nan)
            continue
        union = set_a | set_b
        jacc = len(set_a & set_b) / len(union)
        # Null: shuffle labels within each dataset and recompute
        null = np.empty(n_perm)
        la_shuf = labels_a.copy()
        lb_shuf = labels_b.copy()
        for p in range(n_perm):
            rng.shuffle(la_shuf)
            rng.shuffle(lb_shuf)
            sa = set(la_shuf[in_a].tolist())
            sb = set(lb_shuf[in_b].tolist())
            u = sa | sb
            null[p] = len(sa & sb) / len(u) if u else 0.0
        pvals.append(float((np.sum(null >= jacc) + 1) / (n_perm + 1)))
        jaccards.append(jacc)

    return {"jaccard": jaccards, "p_value": pvals}


def spherical_replicate_residual(coords_a: np.ndarray,
                                 labels_a,
                                 coords_b: np.ndarray,
                                 labels_b) -> dict:
    """
    Per-cell-type angular RMSE between two replicates after best-rotation
    alignment. Cell types with high residual are candidates for batch-effect
    contamination (their position on the sphere is not reproducible across
    replicates).

    Returns:
        dict with:
            R: alignment rotation.
            per_label_rmse: dict {label -> RMSE in radians}.
            global_rmse: scalar RMSE across all common labels.
    """
    aln = procrustes_align_spheres(coords_a, coords_b, labels_a, labels_b)
    R = aln["R"]
    aligned_b = _as_unit(coords_b) @ R.T
    coords_a = _as_unit(coords_a)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    per_label = {}
    for lab in aln["common_labels"]:
        ca = coords_a[labels_a == lab].mean(axis=0)
        ca = ca / np.linalg.norm(ca)
        cb = aligned_b[labels_b == lab].mean(axis=0)
        cb = cb / np.linalg.norm(cb)
        per_label[lab] = float(np.arccos(np.clip(ca @ cb, -1.0, 1.0)))

    global_rmse = float(np.sqrt(np.mean(np.array(list(per_label.values())) ** 2)))
    return {"R": R, "per_label_rmse": per_label, "global_rmse": global_rmse}
