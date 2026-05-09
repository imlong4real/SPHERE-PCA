"""Shared helpers for the supplemental-figure modules."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SUPP_ROOT = ROOT / "outputs" / "figures" / "supplement"
RAW = ROOT / "raw_data"


_FONT = {
    "family": "DejaVu Sans",
    "title": 11.5,
    "label": 9.5,
    "tick": 8.0,
    "legend": 8.0,
}


def apply_publication_style():
    plt.rcParams.update({
        "font.family": _FONT["family"],
        "font.size": _FONT["tick"],
        "axes.titlesize": _FONT["title"],
        "axes.labelsize": _FONT["label"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": _FONT["tick"],
        "ytick.labelsize": _FONT["tick"],
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": _FONT["legend"],
        "figure.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def to_unit(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = np.linalg.norm(coords, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return coords / n


def angular_distance_matrix(unit_centroids: np.ndarray) -> np.ndarray:
    """Pairwise geodesic angular distance (radians) between unit vectors."""
    g = np.clip(unit_centroids @ unit_centroids.T, -1.0, 1.0)
    return np.arccos(g)


def euclidean_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _yaml_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def save_outputs(out_dir: Path, stem: str,
                 fig: Optional[plt.Figure],
                 config: Dict[str, Any],
                 data: Optional[Dict[str, pd.DataFrame]] = None,
                 *, registry: list | None = None) -> Path:
    """Write PNG + YAML + CSV(s) for one supplemental panel."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if fig is not None:
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        if registry is not None:
            registry.append(str(png))
    cfg = dict(config or {})
    cfg["panel_id"] = stem
    cfg["timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
    with open(out_dir / f"{stem}_config.yaml", "w") as fh:
        yaml.safe_dump(_yaml_safe(cfg), fh, sort_keys=False)
    for name, obj in (data or {}).items():
        csv_path = out_dir / f"{stem}_data_{name}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(csv_path, index=False)
        elif isinstance(obj, pd.Series):
            obj.to_frame().to_csv(csv_path, index=False)
        elif isinstance(obj, np.ndarray):
            pd.DataFrame(obj).to_csv(csv_path, index=False)
        else:
            try:
                pd.DataFrame(obj).to_csv(csv_path, index=False)
            except Exception:
                with open(csv_path.with_suffix(".json"), "w") as fh:
                    yaml.safe_dump(_yaml_safe(obj), fh)
        if registry is not None:
            registry.append(str(csv_path))
    return out_dir


# ---------------------------------------------------------------------------
# Sphere drawing
# ---------------------------------------------------------------------------

def draw_3d_sphere_backdrop(ax, *, color="#e8edf2", wire="#aab3bf",
                            alpha_surface=0.10, alpha_wire=0.20):
    u, v = np.mgrid[0:2 * np.pi:64j, 0:np.pi:32j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha_surface,
                    linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color=wire, alpha=alpha_wire,
                      linewidth=0.25)


def style_clean_3d(ax):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()


__all__ = [
    "ROOT", "SUPP_ROOT", "RAW",
    "apply_publication_style", "to_unit",
    "angular_distance_matrix", "euclidean_distance_matrix",
    "save_outputs", "draw_3d_sphere_backdrop", "style_clean_3d",
]
