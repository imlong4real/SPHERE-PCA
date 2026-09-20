"""Lightweight, deterministic SPHERE-PCA transformations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import numpy as np

from .result import SpherePCAResult


def _as_pc_array(pcs: Any) -> np.ndarray:
    array = np.asarray(pcs, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            "pcs must be a two-dimensional array with exactly three columns "
            "(PC1, PC2, PC3)"
        )
    if array.shape[0] == 0:
        raise ValueError("pcs must contain at least one observation")
    if not np.all(np.isfinite(array)):
        raise ValueError("pcs contains NaN or infinite values")
    return array


def _as_root_mask(root_mask: Any, n_observations: int) -> np.ndarray:
    mask = np.asarray(root_mask)
    if mask.ndim != 1 or mask.shape[0] != n_observations:
        raise ValueError("root_mask must be one-dimensional and match pcs rows")
    if mask.dtype.kind != "b":
        raise TypeError("root_mask must contain boolean values")
    if not np.any(mask):
        raise ValueError("root_mask selects no observations")
    return mask.astype(bool, copy=False)


def _as_root_centroid(root_centroid: Any, zero_tol: float) -> np.ndarray:
    centroid = np.asarray(root_centroid, dtype=float)
    if centroid.shape != (3,):
        raise ValueError("root_centroid must be a one-dimensional vector of length 3")
    if not np.all(np.isfinite(centroid)):
        raise ValueError("root_centroid contains NaN or infinite values")
    if np.linalg.norm(centroid) <= zero_tol:
        raise ValueError("root_centroid must have non-zero magnitude")
    return centroid


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a deterministic proper rotation mapping source to target."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(source @ target, -1.0, 1.0))

    if sine < 1e-15:
        if cosine > 0.0:
            return np.eye(3)
        # Antiparallel vectors need a deterministic orthogonal rotation axis.
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis = axis / np.linalg.norm(axis)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + 2.0 * (skew @ skew)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / sine**2)


def transform(
    pcs: Any,
    *,
    root_mask: Any = None,
    root_centroid: Any = None,
    zero_tol: float = 1e-12,
) -> SpherePCAResult:
    """Project PC1-PC3 coordinates onto a root-aligned unit sphere.

    Parameters
    ----------
    pcs:
        An ``(n_observations, 3)`` array containing PC1-PC3 scores.
    root_mask:
        Optional boolean mask selecting a biologically justified root
        population. Its mean direction on the unit sphere is rotated to the
        north pole. If omitted, no rotation is applied.
    root_centroid:
        Optional three-dimensional reference vector in the same PC coordinate
        system as ``pcs``. Its direction is rotated to the north pole. Pass
        either ``root_mask`` or ``root_centroid``, not both. The supplied
        vector is retained unchanged in the result metadata; normalization is
        used only to construct the rotation.
    zero_tol:
        Vectors with norms at or below this threshold are rejected because
        their spherical direction is undefined.
    """

    if zero_tol < 0:
        raise ValueError("zero_tol must be non-negative")
    if root_mask is not None and root_centroid is not None:
        raise ValueError("Pass either root_mask or root_centroid, not both")

    pc_array = _as_pc_array(pcs)
    radii = np.linalg.norm(pc_array, axis=1)
    invalid = radii <= zero_tol
    if np.any(invalid):
        rows = np.flatnonzero(invalid)
        preview = ", ".join(str(int(i)) for i in rows[:5])
        suffix = "..." if rows.size > 5 else ""
        raise ValueError(
            "Spherical projection is undefined for zero or near-zero PC "
            f"vectors (rows: {preview}{suffix}; zero_tol={zero_tol:g})"
        )

    unit = pc_array / radii[:, None]
    rotation = np.eye(3)
    effective_centroid: Optional[np.ndarray] = None
    root_direction: Optional[np.ndarray] = None

    if root_mask is not None:
        mask = _as_root_mask(root_mask, pc_array.shape[0])
        centroid = unit[mask].mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm <= zero_tol:
            raise ValueError(
                "The selected root population has a zero or near-zero mean "
                "direction and cannot define north"
            )
        effective_centroid = centroid.copy()
        root_direction = centroid / centroid_norm
        rotation = _rotation_between(root_direction, np.array([0.0, 0.0, 1.0]))
    elif root_centroid is not None:
        centroid = _as_root_centroid(root_centroid, zero_tol)
        effective_centroid = centroid.copy()
        root_direction = centroid / np.linalg.norm(centroid)
        rotation = _rotation_between(root_direction, np.array([0.0, 0.0, 1.0]))

    aligned = unit @ rotation.T
    # Suppress accumulated round-off before inverse trigonometric functions.
    aligned = aligned / np.linalg.norm(aligned, axis=1, keepdims=True)
    theta = np.arccos(np.clip(aligned[:, 2], -1.0, 1.0))
    phi = np.arctan2(aligned[:, 1], aligned[:, 0])

    return SpherePCAResult(
        theta=theta,
        phi=phi,
        r=radii,
        pcs=pc_array.copy(),
        unit_vectors=unit,
        aligned_vectors=aligned,
        rotation_matrix=rotation,
        root_centroid=effective_centroid,
        root_direction=root_direction,
    )


def _is_anndata_like(data: Any) -> bool:
    return all(hasattr(data, attr) for attr in ("X", "obs", "obsm"))


def _resolve_root_mask(
    data: Any,
    *,
    root: Any,
    root_mask: Any,
    root_key: Optional[str],
    labels: Any,
    n_observations: int,
) -> Any:
    if root_mask is not None and root is not None:
        raise ValueError("Pass either root or root_mask, not both")
    if root_mask is not None:
        return _as_root_mask(root_mask, n_observations)
    if root is None:
        return None

    candidate = np.asarray(root)
    if candidate.ndim == 1 and candidate.shape[0] == n_observations and candidate.dtype.kind == "b":
        return _as_root_mask(candidate, n_observations)

    if _is_anndata_like(data):
        if root_key is None:
            matching_keys = [
                str(column)
                for column in data.obs.columns
                if np.any(np.asarray(data.obs[column]) == root)
            ]
            if len(matching_keys) != 1:
                detail = "none" if not matching_keys else ", ".join(matching_keys)
                raise ValueError(
                    "root_key is required unless root occurs in exactly one "
                    f"adata.obs column (matching columns: {detail})"
                )
            root_key = matching_keys[0]
        if root_key not in data.obs:
            raise ValueError(f"root_key {root_key!r} was not found in adata.obs")
        resolved = np.asarray(data.obs[root_key]) == root
    else:
        if labels is None:
            raise ValueError("labels is required when root is a label for a matrix")
        label_array = np.asarray(labels)
        if label_array.ndim != 1 or label_array.shape[0] != n_observations:
            raise ValueError("labels must be one-dimensional and match data rows")
        resolved = label_array == root

    if not np.any(resolved):
        raise ValueError(f"root label {root!r} selects no observations")
    return resolved


def _dense_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must be a two-dimensional cell-by-feature matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("data contains NaN or infinite values")
    if min(array.shape) < 3:
        raise ValueError("data must have at least three rows and three columns")
    return array


def fit(
    data: Any,
    *,
    root: Any = None,
    root_mask: Any = None,
    root_centroid: Any = None,
    root_key: Optional[str] = None,
    labels: Any = None,
    use_rep: Optional[str] = "X_pca",
    random_state: int = 0,
    zero_tol: float = 1e-12,
    write_back: bool = False,
    key_added: str = "sphere_pca",
) -> SpherePCAResult:
    """Fit PCA when needed, then run the spherical transformation.

    ``data`` may be a cell-by-feature matrix or an AnnData-like object. For
    AnnData, an existing ``adata.obsm[use_rep]`` is used when present;
    otherwise PCA is fit to ``adata.X``. Inputs should already contain the
    normalized expression representation appropriate for the analysis. This
    function does not perform count normalization, log transformation, highly
    variable gene selection, or feature scaling; reproduce those steps before
    calling ``fit`` when manuscript-equivalent preprocessing is required.
    ``root_centroid`` may provide a fixed three-dimensional reference in the
    resulting PC coordinate system; it is mutually exclusive with ``root``
    and ``root_mask``.
    """

    if root_centroid is not None and (root is not None or root_mask is not None):
        raise ValueError(
            "Pass root_centroid by itself; it is mutually exclusive with root and root_mask"
        )

    adata_like = _is_anndata_like(data)
    pca = None

    if adata_like and use_rep is not None and use_rep in data.obsm:
        representation = np.asarray(data.obsm[use_rep], dtype=float)
        if representation.ndim != 2 or representation.shape[1] < 3:
            raise ValueError(f"adata.obsm[{use_rep!r}] must contain at least 3 columns")
        pcs = representation[:, :3]
    else:
        matrix = data.X if adata_like else data
        dense = _dense_matrix(matrix)
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ImportError("fit() requires scikit-learn") from exc
        pca = PCA(n_components=3, svd_solver="auto", random_state=random_state)
        pcs = pca.fit_transform(dense)

    mask = _resolve_root_mask(
        data,
        root=root,
        root_mask=root_mask,
        root_key=root_key,
        labels=labels,
        n_observations=pcs.shape[0],
    )
    result = transform(
        pcs,
        root_mask=mask,
        root_centroid=root_centroid,
        zero_tol=zero_tol,
    )

    observation_names = None
    if adata_like and hasattr(data, "obs_names"):
        observation_names = np.asarray(data.obs_names).copy()

    if pca is not None:
        result = replace(
            result,
            pca_components=pca.components_.copy(),
            pca_mean=pca.mean_.copy(),
            explained_variance_ratio=pca.explained_variance_ratio_.copy(),
            observation_names=observation_names,
        )
    elif observation_names is not None:
        result = replace(result, observation_names=observation_names)

    if write_back:
        if not adata_like:
            raise TypeError("write_back=True is only valid for an AnnData-like input")
        result.write_to_adata(data, key_added=key_added)
    return result
