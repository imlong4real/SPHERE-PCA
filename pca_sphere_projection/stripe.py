"""Multi-stripe diagnostics for spherical PCA coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

from .sphere_stats import _as_unit, stripe_strength_score


def _spherical_angles(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = _as_unit(coords)
    theta = np.arccos(np.clip(coords[:, 2], -1.0, 1.0))
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    return theta, phi


def _angular_width(phi: np.ndarray) -> float:
    if phi.size < 2:
        return 0.0
    r = np.abs(np.mean(np.exp(1j * phi)))
    return float(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12)))))


def detect_spherical_stripes(coords, root=None, method="local_density_graph",
                             k: int = 15, max_stripes: int = 12,
                             min_stripe_size: int = 25,
                             random_state: int = 0) -> dict:
    """Detect stripe-like arcs on S2 using deterministic angular clustering.

    The implementation keeps the old longitude concentration separate from
    branch/stripe assignment. It clusters cells by circular longitude plus
    coarse polar position, then merges tiny clusters into ``-1``.
    """
    coords = _as_unit(coords)
    n = coords.shape[0]
    if n == 0:
        return {"assignments": np.array([], dtype=int), "summary": pd.DataFrame()}

    theta, phi = _spherical_angles(coords)
    n_clusters = int(min(max_stripes, max(1, np.sqrt(max(n, 1)) // 6 + 1)))
    if n_clusters == 1 or n < min_stripe_size * 2:
        labels = np.zeros(n, dtype=int)
    else:
        features = np.column_stack([
            np.cos(phi),
            np.sin(phi),
            np.cos(theta),
            0.5 * np.sin(theta),
        ])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(features)

    counts = pd.Series(labels).value_counts()
    small = set(counts[counts < min_stripe_size].index)
    if small:
        labels = np.array([-1 if x in small else int(x) for x in labels], dtype=int)
    uniq = [u for u in sorted(np.unique(labels)) if u >= 0]
    remap = {old: i for i, old in enumerate(uniq)}
    labels = np.array([remap.get(int(x), -1) for x in labels], dtype=int)
    return {"assignments": labels, "summary": local_stripe_strength(coords, labels)}


def local_stripe_strength(coords, stripe_assignments) -> pd.DataFrame:
    """Per-stripe length/width/strength summary."""
    coords = _as_unit(coords)
    labels = np.asarray(stripe_assignments)
    theta, phi = _spherical_angles(coords)
    rows = []
    for lab in [u for u in sorted(np.unique(labels)) if u >= 0]:
        mask = labels == lab
        if mask.sum() < 2:
            continue
        length = float(np.nanmax(theta[mask]) - np.nanmin(theta[mask]))
        width = _angular_width(phi[mask])
        strength = float(length / (length + width + 1e-9))
        rows.append({
            "stripe": int(lab),
            "per_stripe_size": int(mask.sum()),
            "per_stripe_length": length,
            "per_stripe_width": width,
            "per_stripe_strength": strength,
            "theta_min": float(np.nanmin(theta[mask])),
            "theta_max": float(np.nanmax(theta[mask])),
            "phi_center": float(np.angle(np.mean(np.exp(1j * phi[mask])))),
        })
    return pd.DataFrame(rows)


def multi_stripe_strength_score(coords, labels=None, root=None) -> dict:
    """Return old global concentration and new multi-stripe summaries."""
    coords = _as_unit(coords)
    if labels is None:
        detected = detect_spherical_stripes(coords, root=root)
        labels = detected["assignments"]
        per = detected["summary"]
    else:
        labels = np.asarray(labels)
        per = local_stripe_strength(coords, labels)
    global_score = stripe_strength_score(coords)
    if per.empty:
        mean_strength = median_strength = 0.0
    else:
        weights = per["per_stripe_size"].to_numpy(dtype=float)
        vals = per["per_stripe_strength"].to_numpy(dtype=float)
        mean_strength = float(np.average(vals, weights=weights))
        median_strength = float(np.median(vals))
    return {
        "n_stripes": int((per["per_stripe_size"] >= 1).sum()) if not per.empty else 0,
        "global_longitude_concentration": float(global_score["score"]),
        "global_longitude_entropy_score": float(global_score["score"]),
        "mean_local_stripe_strength": mean_strength,
        "median_local_stripe_strength": median_strength,
        "multi_stripe_strength": median_strength,
        "per_stripe": per,
        "stripe_assignments": np.asarray(labels, dtype=int),
    }


def plot_detected_stripes(coords, stripe_assignments, ax=None):
    """Plot detected stripes in longitude/theta coordinates."""
    import matplotlib.pyplot as plt

    coords = _as_unit(coords)
    theta, phi = _spherical_angles(coords)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(np.degrees(phi), np.degrees(theta), c=stripe_assignments,
                    s=4, cmap="tab20", linewidths=0)
    ax.set_xlabel("longitude phi (deg)")
    ax.set_ylabel("theta from root/pole (deg)")
    ax.set_title("Detected multi-stripe structure")
    return ax, sc


__all__ = [
    "detect_spherical_stripes",
    "multi_stripe_strength_score",
    "local_stripe_strength",
    "plot_detected_stripes",
]
