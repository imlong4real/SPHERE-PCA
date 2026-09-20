"""Small plotting helpers for :class:`sphere_pca.SpherePCAResult`."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .result import SpherePCAResult


def _color_values(color: Any, default: np.ndarray):
    n_observations = default.shape[0]
    if color is None:
        return default, None

    values = np.asarray(color)
    if values.ndim != 1 or values.shape[0] != n_observations:
        raise ValueError("color must be one-dimensional and match the result length")
    if values.dtype.kind in "biufc":
        return values, None

    categories, codes = np.unique(values.astype(str), return_inverse=True)
    return codes, categories


def plot(
    result: SpherePCAResult,
    *,
    projection: str = "equirectangular",
    color: Any = None,
    ax: Optional[Any] = None,
    cmap: str = "viridis",
    s: float = 12.0,
    alpha: float = 0.8,
):
    """Plot an equirectangular map or 3D sphere and return the axes."""

    if not isinstance(result, SpherePCAResult):
        raise TypeError("result must be a SpherePCAResult")
    if projection not in {"equirectangular", "sphere"}:
        raise ValueError("projection must be 'equirectangular' or 'sphere'")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plot() requires matplotlib. Install it with "
            "`pip install 'sphere-pca[plot]'`."
        ) from exc

    color_values, categories = _color_values(color, result.theta)
    if ax is None:
        if projection == "sphere":
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        else:
            _, ax = plt.subplots()

    if projection == "sphere":
        xyz = result.aligned_vectors
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=color_values, cmap=cmap, s=s, alpha=alpha,
        )
        ax.set(xlabel="sphere x", ylabel="sphere y", zlabel="sphere z")
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))
    else:
        longitude = np.degrees(result.phi)
        latitude = 90.0 - np.degrees(result.theta)
        ax.scatter(
            longitude, latitude,
            c=color_values, cmap=cmap, s=s, alpha=alpha,
        )
        ax.set(
            xlabel=r"angular position $\phi$ (degrees)",
            ylabel=r"latitude $90^\circ-\theta$ (degrees)",
            xlim=(-180, 180),
            ylim=(-90, 90),
        )

    if categories is not None:
        colormap = plt.get_cmap(cmap)
        denominator = max(len(categories) - 1, 1)
        handles = [
            Line2D(
                [0], [0], marker="o", linestyle="", color=colormap(i / denominator),
                label=str(category), markersize=6,
            )
            for i, category in enumerate(categories)
        ]
        ax.legend(handles=handles, title="group", bbox_to_anchor=(1.02, 1), loc="upper left")

    return ax
