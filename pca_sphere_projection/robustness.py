"""
Robustness diagnostics for the spherical-PCA pipeline.

The hand-tuned root cluster and Euler rotation in the original pipeline are
both *choices* that look like data analysis but behave like figure-tuning.
This module quantifies how much downstream results depend on those choices,
so any biological claim can be reported alongside its sensitivity envelope.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from .core import align_to_north_pole, apply_euler_rotation
from .sphere_stats import _as_unit, fit_great_circle, stripe_strength_score


# ---------------------------------------------------------------------------
# Root-cluster sensitivity
# ---------------------------------------------------------------------------

def root_sensitivity_analysis(df: pd.DataFrame,
                              candidate_roots: Iterable[str],
                              root_column: str,
                              metric_fn: Callable[[np.ndarray], float] | None = None,
                              pcs_columns: list[str] | None = None,
                              euler_angles: list[float] | None = None) -> pd.DataFrame:
    """
    Sweep the choice of root cluster and report how a chosen scalar metric
    of the spherical embedding varies.

    Args:
        df: input DataFrame with PC1-PC3 columns and a categorical label
            column.
        candidate_roots: iterable of category values to test as roots.
        root_column: which column to look up the root cluster in.
        metric_fn: scalar function (coords) -> float. Defaults to
            ``stripe_strength_score(...)['score']``.
        pcs_columns: PC column names; default ``["PC1", "PC2", "PC3"]``.
        euler_angles: optional Euler angles in degrees applied after the
            north-pole alignment (held fixed across the sweep). If None,
            no Euler step is applied.

    Returns:
        DataFrame with one row per candidate, columns
        ``['root', 'metric', 'n_cells']``, sorted by descending metric.
    """
    if pcs_columns is None:
        pcs_columns = ["PC1", "PC2", "PC3"]
    if metric_fn is None:
        def metric_fn(c):
            return stripe_strength_score(c)["score"]

    rows = []
    for root in candidate_roots:
        if root not in df[root_column].unique():
            rows.append({"root": root, "metric": np.nan, "n_cells": 0})
            continue
        n = int((df[root_column] == root).sum())
        centroid = df.loc[df[root_column] == root, pcs_columns].mean().values
        aligned = align_to_north_pole(df, pcs_columns=pcs_columns,
                                      cluster_column=root_column,
                                      root_node=centroid)
        if euler_angles is not None:
            aligned = apply_euler_rotation(aligned, pcs_columns=pcs_columns,
                                           rotation_angles=euler_angles, degrees=True)
            cols = [f"rotated_{c}" for c in pcs_columns]
        else:
            cols = pcs_columns
        coords = _as_unit(aligned[cols].values)
        rows.append({"root": root, "metric": float(metric_fn(coords)), "n_cells": n})
    return pd.DataFrame(rows).sort_values("metric", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rotation-perturbation robustness
# ---------------------------------------------------------------------------

def rotation_robustness_analysis(df: pd.DataFrame,
                                 base_euler_angles: list[float],
                                 perturbation_degrees: float = 5.0,
                                 n_perturbations: int = 25,
                                 metric_fn: Callable[[np.ndarray], float] | None = None,
                                 pcs_columns: list[str] | None = None,
                                 seed: int = 0) -> dict:
    """
    Apply random small rotations on top of the base Euler choice and report
    the metric distribution. A robust stripe pattern should give a tight
    distribution around the metric at the base angles.

    Args:
        df: DataFrame with PC1-PC3 columns already centred on the north pole.
        base_euler_angles: [x, y, z] degrees.
        perturbation_degrees: max magnitude of each axis perturbation.
        n_perturbations: number of random perturbations to draw.
        metric_fn: scalar function (coords) -> float. Default
            stripe_strength_score.
        pcs_columns: default ``["PC1", "PC2", "PC3"]``.
        seed: RNG seed.

    Returns:
        dict with:
            'base_metric': scalar at base_euler_angles.
            'perturbed_metrics': (n_perturbations,) array.
            'mean': perturbed mean.
            'std': perturbed std.
            'cv': coefficient of variation (std / |mean|).
    """
    if pcs_columns is None:
        pcs_columns = ["PC1", "PC2", "PC3"]
    if metric_fn is None:
        def metric_fn(c):
            return stripe_strength_score(c)["score"]

    rng = np.random.default_rng(seed)

    def _rotate_with(angles):
        rotated = apply_euler_rotation(df, pcs_columns=pcs_columns,
                                       rotation_angles=list(angles),
                                       degrees=True)
        cols = [f"rotated_{c}" for c in pcs_columns]
        return _as_unit(rotated[cols].values)

    base_angles = np.asarray(base_euler_angles, dtype=float)
    base_metric = float(metric_fn(_rotate_with(base_angles)))

    perturbed = np.empty(n_perturbations)
    for i in range(n_perturbations):
        delta = rng.uniform(-perturbation_degrees, perturbation_degrees, size=3)
        perturbed[i] = float(metric_fn(_rotate_with(base_angles + delta)))

    mean = float(perturbed.mean())
    std = float(perturbed.std(ddof=1)) if n_perturbations > 1 else 0.0
    return {
        "base_metric": base_metric,
        "perturbed_metrics": perturbed,
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if abs(mean) > 1e-12 else float("inf"),
    }


# ---------------------------------------------------------------------------
# PC-coordinate quality summary
# ---------------------------------------------------------------------------

def pc_coordinate_quality_summary(df: pd.DataFrame,
                                  pcs_columns: list[str] | None = None) -> dict:
    """
    Quick health-check of the PC1-PC3 coordinates before any spherical
    transformation. Surfaces issues that frequently cause spherical-pipeline
    artefacts (e.g. one PC dominating, near-zero magnitudes, etc.).

    Args:
        df: input DataFrame.
        pcs_columns: default ``["PC1", "PC2", "PC3"]``.

    Returns:
        dict with:
            'n_cells', 'n_pcs', 'pc_variances', 'variance_ratio_top_to_bottom',
            'norm_min', 'norm_median', 'norm_max', 'frac_zero_norm',
            'great_circle_r_squared', 'spherical_anisotropy_summary'.
    """
    from .sphere_stats import spherical_anisotropy

    if pcs_columns is None:
        pcs_columns = ["PC1", "PC2", "PC3"]
    X = df[pcs_columns].values.astype(float)
    n, k = X.shape
    variances = X.var(axis=0)
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    if not np.any(nz):
        raise ValueError("all rows have zero PC norm")
    coords = _as_unit(X[nz])
    gc = fit_great_circle(coords)
    aniso = spherical_anisotropy(coords)
    return {
        "n_cells": int(n),
        "n_pcs": int(k),
        "pc_variances": variances.tolist(),
        "variance_ratio_top_to_bottom": float(variances.max() / max(variances.min(), 1e-12)),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "frac_zero_norm": float((~nz).mean()),
        "great_circle_r_squared": gc["r_squared"],
        "great_circle_normal": gc["normal"].tolist(),
        "spherical_anisotropy_summary": {k: float(v) if np.isscalar(v) else float(np.asarray(v).item())
                                          if hasattr(v, 'item') else v
                                          for k, v in aniso.items() if k in ("linear", "planar", "spherical")},
    }


# ---------------------------------------------------------------------------
# Manual-Euler vs. great-circle comparison
# ---------------------------------------------------------------------------

def compare_manual_vs_great_circle(df: pd.DataFrame,
                                   euler_angles: list[float],
                                   pcs_columns: list[str] | None = None,
                                   metric_fn: Callable[[np.ndarray], float] | None = None) -> dict:
    """
    Side-by-side comparison: stripe metric under the manually-tuned Euler
    rotation versus a fully automated great-circle alignment.

    Args:
        df: DataFrame already centred on the north pole.
        euler_angles: the manual angles to compare against.
        pcs_columns: default ``["PC1", "PC2", "PC3"]``.
        metric_fn: stripe_strength_score by default.

    Returns:
        dict with manual_metric, gc_metric, and the rotation matrix used
        for the great-circle alignment.
    """
    if pcs_columns is None:
        pcs_columns = ["PC1", "PC2", "PC3"]
    if metric_fn is None:
        def metric_fn(c):
            return stripe_strength_score(c)["score"]

    # Manual
    rotated = apply_euler_rotation(df, pcs_columns=pcs_columns,
                                   rotation_angles=euler_angles, degrees=True)
    cols = [f"rotated_{c}" for c in pcs_columns]
    manual_coords = _as_unit(rotated[cols].values)
    manual_metric = float(metric_fn(manual_coords))

    # Great-circle: rotate so the data's principal great circle is the equator.
    coords = _as_unit(df[pcs_columns].values)
    gc = fit_great_circle(coords)
    normal = gc["normal"]
    # Rotation that takes ``normal`` to (0, 0, 1) puts the great circle on the equator.
    # Wait — we want the great circle on the equator (i.e. data has z ~ 0), which
    # means we want the plane normal aligned with +z.
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c_ = float(normal @ target)
    if s < 1e-12:
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c_) / (s * s))
    gc_coords = _as_unit(coords @ R.T)
    gc_metric = float(metric_fn(gc_coords))

    return {
        "manual_metric": manual_metric,
        "gc_metric": gc_metric,
        "gc_rotation": R,
        "great_circle_r_squared": gc["r_squared"],
    }
