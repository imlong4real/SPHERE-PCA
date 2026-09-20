"""Result containers for the public SPHERE-PCA API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class SpherePCAResult:
    """Coordinates and fitted metadata produced by SPHERE-PCA.

    Angular coordinates are stored in radians. ``theta`` is the geodesic
    distance from the aligned north pole in ``[0, pi]``; ``phi`` is the
    azimuth in ``[-pi, pi]``; and ``r`` is the norm of the original PC1-PC3
    vector before projection. ``root_centroid`` is the exact supplied
    reference vector, or the mean of the root unit vectors when a mask was
    used. ``root_direction`` is its normalized direction. Together with
    ``rotation_matrix``, these fields record the effective alignment.
    """

    theta: np.ndarray
    phi: np.ndarray
    r: np.ndarray
    pcs: np.ndarray
    unit_vectors: np.ndarray
    aligned_vectors: np.ndarray
    rotation_matrix: np.ndarray
    root_centroid: Optional[np.ndarray] = None
    root_direction: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None
    observation_names: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.theta.shape[0])

    @property
    def coordinates(self) -> np.ndarray:
        """Return ``theta``, ``phi``, and ``r`` as an ``(n, 3)`` array."""

        return np.column_stack((self.theta, self.phi, self.r))

    def to_frame(self):
        """Return the coordinates as a pandas DataFrame.

        Pandas is imported only when this convenience method is called so it
        does not become a core dependency.
        """

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "SpherePCAResult.to_frame() requires pandas. "
                "Install it with `pip install pandas`."
            ) from exc

        index = self.observation_names
        return pd.DataFrame(
            {"theta": self.theta, "phi": self.phi, "r": self.r},
            index=index,
        )

    def write_to_adata(self, adata: Any, *, key_added: str = "sphere_pca") -> Any:
        """Write coordinates to an AnnData-like object and return it.

        Three columns are added to ``adata.obs`` and the aligned unit vectors
        are stored in ``adata.obsm[f"X_{key_added}"]``. The method uses the
        AnnData protocol without importing :mod:`anndata`.
        """

        if not hasattr(adata, "obs") or not hasattr(adata, "obsm"):
            raise TypeError("adata must provide .obs and .obsm attributes")
        if len(adata.obs) != len(self):
            raise ValueError(
                "AnnData observation count does not match the SPHERE-PCA result"
            )
        if not key_added or not isinstance(key_added, str):
            raise ValueError("key_added must be a non-empty string")

        adata.obs[f"{key_added}_theta"] = self.theta
        adata.obs[f"{key_added}_phi"] = self.phi
        adata.obs[f"{key_added}_r"] = self.r
        adata.obsm[f"X_{key_added}"] = self.aligned_vectors.copy()
        return adata
