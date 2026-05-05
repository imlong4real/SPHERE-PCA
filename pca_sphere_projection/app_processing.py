"""Reusable processing pipeline for the interactive PCA sphere app."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sphere_stats import fit_great_circle, spherical_anisotropy, stripe_strength_score
from .topology import detect_branchpoints

PC_COLUMNS = ["PC1", "PC2", "PC3"]
EXAMPLE_FILES = {
    "C. elegans embryo": "celegan_pca.csv",
    "Ulcerative colitis epithelium": "uc_epi_pca.csv",
    "Planaria atlas": "planaria_pca.csv",
}


@dataclass
class PipelineResult:
    """Container for app-ready data, metrics, and optional topology results."""

    data: pd.DataFrame
    metrics: dict[str, Any]
    root_cluster: str | None
    color_columns: list[str]
    pseudotime_column: str | None
    branchpoint: dict[str, Any] | None


def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


def load_example_csv(filename: str) -> pd.DataFrame:
    path = examples_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Example CSV not found: {path}")
    return pd.read_csv(path)


def validate_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in PC_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV must contain PC1, PC2, and PC3 columns. "
            f"Missing: {', '.join(missing)}"
        )

    out = df.copy()
    out[PC_COLUMNS] = out[PC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad = out[PC_COLUMNS].isna().any(axis=1)
    if bad.all():
        raise ValueError("No rows have valid numeric PC1, PC2, PC3 values.")
    if bad.any():
        out = out.loc[~bad].copy()

    if "celltype" in out.columns:
        out["celltype"] = out["celltype"].fillna("Unknown celltype").astype(str)
    if "cluster" in out.columns:
        out["cluster"] = out["cluster"].fillna("Unknown cluster").astype(str)
    return out


def normalize_pc_coordinates(df: pd.DataFrame) -> np.ndarray:
    coords = df[PC_COLUMNS].to_numpy(dtype=float)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    keep = norms[:, 0] > 0
    if not keep.all():
        raise ValueError("PC coordinates include zero-length rows; cannot project all rows.")
    return coords / norms


def parse_cluster_pseudotime(values: pd.Series) -> pd.Series:
    """Parse numeric clusters or range labels such as '210-270' into midpoints."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        return numeric

    def parse_one(value: Any) -> float:
        text = str(value).strip()
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
        if len(nums) >= 2:
            return float(np.mean(nums[:2]))
        if len(nums) == 1:
            return nums[0]
        return np.nan

    parsed = values.map(parse_one)
    return parsed if parsed.notna().mean() >= 0.8 else pd.Series(np.nan, index=values.index)


def default_root_cluster(df: pd.DataFrame) -> str | None:
    if "cluster" not in df.columns:
        return None
    pt = parse_cluster_pseudotime(df["cluster"])
    if pt.notna().any():
        med = pt.groupby(df["cluster"]).median().sort_values()
        return str(med.index[0])
    counts = df["cluster"].value_counts()
    return str(counts.index[0]) if not counts.empty else None


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    v = np.cross(source, target)
    s = np.linalg.norm(v)
    c = float(source @ target)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source @ axis) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(source, axis)
        v = v / np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def align_root_to_north(coords: np.ndarray, df: pd.DataFrame, root_cluster: str | None) -> tuple[np.ndarray, np.ndarray]:
    if root_cluster is None or "cluster" not in df.columns:
        return coords, np.eye(3)
    mask = df["cluster"].astype(str) == str(root_cluster)
    if not mask.any():
        raise ValueError(f"Root cluster {root_cluster!r} was not found in the cluster column.")
    centroid = coords[mask].mean(axis=0)
    if np.linalg.norm(centroid) < 1e-12:
        raise ValueError(f"Root cluster {root_cluster!r} has a near-zero centroid.")
    R = _rotation_between(centroid, np.array([0.0, 0.0, 1.0]))
    return coords @ R.T, R


def add_spherical_coordinates(df: pd.DataFrame, coords: np.ndarray, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_x"] = coords[:, 0]
    out[f"{prefix}_y"] = coords[:, 1]
    out[f"{prefix}_z"] = coords[:, 2]
    out[f"{prefix}_longitude"] = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    out[f"{prefix}_latitude"] = np.degrees(np.arcsin(np.clip(coords[:, 2], -1.0, 1.0)))
    out[f"{prefix}_theta"] = np.degrees(np.arccos(np.clip(coords[:, 2], -1.0, 1.0)))
    out[f"{prefix}_phi"] = out[f"{prefix}_longitude"]
    return out


def _metric_table(stripe: dict, anisotropy: dict, gc: dict) -> dict[str, float]:
    return {
        "stripe_strength": float(stripe["score"]),
        "stripe_entropy_observed": float(stripe["entropy_observed"]),
        "stripe_entropy_uniform": float(stripe["entropy_uniform"]),
        "anisotropy_linear": float(anisotropy["linear"]),
        "anisotropy_planar": float(anisotropy["planar"]),
        "anisotropy_spherical": float(anisotropy["spherical"]),
        "great_circle_mean_residual_deg": float(np.degrees(gc["mean_residual_radians"])),
        "great_circle_r_squared": float(gc["r_squared"]),
    }


def run_projection_pipeline(
    df: pd.DataFrame,
    root_cluster: str | None = None,
    compute_topology: bool = True,
    max_topology_cells: int = 5000,
) -> PipelineResult:
    """Run the app workflow without requiring Streamlit."""
    df = validate_input_dataframe(df)
    if root_cluster is None:
        root_cluster = default_root_cluster(df)

    coords = normalize_pc_coordinates(df)
    root_coords, root_rotation = align_root_to_north(coords, df, root_cluster)
    gc = fit_great_circle(root_coords)
    display_rotation = _rotation_between(gc["normal"], np.array([0.0, 0.0, 1.0]))
    display_coords = root_coords @ display_rotation.T

    out = add_spherical_coordinates(df, coords, "unit")
    out = add_spherical_coordinates(out, root_coords, "root")
    out = add_spherical_coordinates(out, display_coords, "display")

    pseudotime_column = None
    branchpoint = None
    if "cluster" in out.columns:
        pt = parse_cluster_pseudotime(out["cluster"])
        if pt.notna().all():
            out["pseudotime"] = pt.astype(float)
            pseudotime_column = "pseudotime"

    stripe = stripe_strength_score(display_coords, n_bins=36)
    anisotropy = spherical_anisotropy(root_coords)
    metrics = _metric_table(stripe, anisotropy, gc)
    metrics["n_cells"] = int(out.shape[0])
    metrics["root_cluster"] = root_cluster if root_cluster is not None else "not used"
    metrics["has_celltype"] = "celltype" in out.columns
    metrics["has_pseudotime"] = pseudotime_column is not None
    metrics["great_circle_normal_x"] = float(gc["normal"][0])
    metrics["great_circle_normal_y"] = float(gc["normal"][1])
    metrics["great_circle_normal_z"] = float(gc["normal"][2])

    if pseudotime_column is not None:
        theta = out["root_theta"].to_numpy(dtype=float)
        pt = out[pseudotime_column].to_numpy(dtype=float)
        if np.std(theta) > 0 and np.std(pt) > 0:
            metrics["theta_pseudotime_corr"] = float(np.corrcoef(theta, pt)[0, 1])
        if compute_topology:
            if out.shape[0] > max_topology_cells:
                sample = out.sample(max_topology_cells, random_state=0).sort_index()
                idx = sample.index.to_numpy()
                topo_coords = root_coords[out.index.get_indexer(idx)]
                topo_pt = sample[pseudotime_column].to_numpy(dtype=float)
            else:
                sample = out
                idx = out.index.to_numpy()
                topo_coords = root_coords
                topo_pt = out[pseudotime_column].to_numpy(dtype=float)
            topo = detect_branchpoints(topo_coords, topo_pt, k=min(15, max(2, len(sample) - 1)))
            out["branchpoint_score"] = np.nan
            out.loc[idx, "branchpoint_score"] = topo["score"]
            branchpoint = {
                "score": topo["score"],
                "sample_index": idx,
                "max_score": float(np.nanmax(topo["score"])),
                "mean_score": float(np.nanmean(topo["score"])),
                "n_cells_scored": int(len(sample)),
            }
            metrics["branchpoint_max_score"] = branchpoint["max_score"]
            metrics["branchpoint_mean_score"] = branchpoint["mean_score"]

    color_columns = [col for col in ["celltype", "cluster", "pseudotime", "branchpoint_score"] if col in out.columns]
    return PipelineResult(
        data=out,
        metrics=metrics,
        root_cluster=root_cluster,
        color_columns=color_columns,
        pseudotime_column=pseudotime_column,
        branchpoint=branchpoint,
    )


def metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    return pd.DataFrame(rows)
