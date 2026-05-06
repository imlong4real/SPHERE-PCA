"""Shared helpers for the manuscript-figure modules.

Reproducibility contract:

- Every panel function takes a ``config`` dict whose ``seed`` field
  (default 0) is the *only* RNG used in that panel; all subsampling and
  perturbation draws are derived from ``np.random.default_rng(seed)``.
- ``save_panel`` writes a sidecar YAML containing the resolved config
  (i.e. defaults + user overrides + a ``timestamp`` and ``panel_id``).
- All numeric data backing a panel is also written next to the PNG as
  one CSV per entry in the panel's ``data`` dict.

Output layout (single source of truth):

    outputs/figures/<FigN>/<FigNX>.png
    outputs/figures/<FigN>/<FigNX>_config.yaml
    outputs/figures/<FigN>/<FigNX>_data_<name>.csv  # per data item
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_COLORS = {
    "celegan": "#4c78a8",
    "uc_epi":  "#f58518",
    "klein":   "#54a24b",
    "hesc":    "#e45756",
    "planaria": "#b279a2",
    "human_germ_cell": "#9d755d",
    "pre_implant_human_embryo": "#ff9da6",
    "BrCa_atlas": "#bab0ac",
}


def apply_publication_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# coordinate / artifact helpers
# ---------------------------------------------------------------------------

def to_unit(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    n = np.linalg.norm(coords, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return coords / n


def xyz_to_lonlat(unit: np.ndarray):
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


def wrapped_longitude_delta(lon: np.ndarray, center: float) -> np.ndarray:
    """Shortest signed longitude difference in degrees."""
    lon = np.asarray(lon, dtype=float)
    return ((lon - float(center) + 180.0) % 360.0) - 180.0


def count_cells_within_bands_wrapped(data: pd.DataFrame | np.ndarray,
                                     centers: list[float],
                                     half_widths: list[float],
                                     lon_column: str = "longitude") -> pd.DataFrame:
    """Count cells in manual wrapped longitude bands.

    ``half_widths`` are interpreted as the requested ± band widths in degrees.
    Bands are manual illustrative annotations; overlapping bands are counted
    independently.
    """
    if isinstance(data, pd.DataFrame):
        lon = data[lon_column].to_numpy(dtype=float)
    else:
        lon = np.asarray(data, dtype=float)
    rows = []
    for i, (center, width) in enumerate(zip(centers, half_widths), start=1):
        delta = wrapped_longitude_delta(lon, center)
        mask = np.abs(delta) <= float(width)
        rows.append({
            "stripe": f"Stripe {i}",
            "center_longitude_deg": float(center),
            "band_half_width_deg": float(width),
            "longitude_min_wrapped_deg": float(((center - width + 180) % 360) - 180),
            "longitude_max_wrapped_deg": float(((center + width + 180) % 360) - 180),
            "n_cells_inside": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def count_between_bands_wrapped(lon: np.ndarray, centers: list[float],
                                half_widths: list[float]) -> pd.DataFrame:
    """Count cells outside manual bands in the gaps between adjacent bands."""
    lon = np.asarray(lon, dtype=float)
    intervals = []
    for i, (c, w) in enumerate(zip(centers, half_widths), start=1):
        intervals.append((float(c - w), float(c + w), f"Stripe {i}"))
    intervals = sorted(intervals, key=lambda x: x[0])
    rows = []
    lon360 = (lon + 360.0) % 360.0
    norm_intervals = [((a + 360) % 360, (b + 360) % 360, label) for a, b, label in intervals]
    norm_intervals = sorted(norm_intervals, key=lambda x: x[0])
    for idx, (_, end, left_label) in enumerate(norm_intervals):
        next_start, _, right_label = norm_intervals[(idx + 1) % len(norm_intervals)]
        if end <= next_start:
            mask = (lon360 > end) & (lon360 < next_start)
            start_out, end_out = end, next_start
        else:
            mask = (lon360 > end) | (lon360 < next_start)
            start_out, end_out = end, next_start
        rows.append({
            "gap": f"{left_label} to {right_label}",
            "from_longitude_deg": float(((start_out + 180) % 360) - 180),
            "to_longitude_deg": float(((end_out + 180) % 360) - 180),
            "n_cells_between": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def load_artifacts(val_out: Path, ds: str):
    p = val_out / ds / "_artifacts.npz"
    if not p.exists():
        return None
    return np.load(p)


def load_geometry(val_out: Path, ds: str) -> Optional[pd.DataFrame]:
    p = val_out / ds / "per_cell_geometry.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_pca_csv(name: str) -> Optional[pd.DataFrame]:
    cands = {"celegan": "celegan_pca.csv", "klein": "klein_pca.csv",
             "uc_epi": "uc_epi_pca.csv", "planaria": "planaria_pca.csv"}
    if name not in cands:
        return None
    p = ROOT / "examples" / cands[name]
    if not p.exists():
        return None
    # Some example CSVs have a leading "" index column (celegan, uc_epi);
    # klein does not. Detect by reading the header.
    head = pd.read_csv(p, nrows=0).columns
    if head[0] in ("", "Unnamed: 0"):
        return pd.read_csv(p, index_col=0)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# panel save + "data not present" plate
# ---------------------------------------------------------------------------

def merge_config(default: dict, override: Optional[dict]) -> dict:
    out = dict(default or {})
    if override:
        out.update(override)
    out.setdefault("seed", 0)
    return out


def _yaml_safe(obj):
    """Coerce numpy/pandas/Path values into YAML-safe Python primitives."""
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
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


def panel_dir(out_root: Path, fig_n: int) -> Path:
    p = out_root / f"Fig{fig_n}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_panel(result: Dict[str, Any], out_root: Path,
               *, registry: list | None = None) -> Path:
    """Write PNG, YAML, and one CSV per data entry. Returns the panel directory.

    `result` must include:
        - panel_id: e.g. "Fig1A" or "Fig3A_hesc"
        - figure_n: 1, 2, 3, or 4
        - figure: matplotlib Figure
        - config: resolved config dict (will be augmented with timestamp)
        - data: dict of pandas DataFrame / Series / np.ndarray

    The optional `registry` list is appended to with each saved PNG path
    (for the CLI's manifest).
    """
    panel_id = result["panel_id"]
    figure_n = result["figure_n"]
    fig = result["figure"]
    cfg = dict(result.get("config", {}))
    data = result.get("data", {}) or {}

    pdir = panel_dir(out_root, figure_n)
    png = pdir / f"{panel_id}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    for alias in result.get("output_aliases", []) or []:
        alias_dir = Path(alias.get("dir", pdir))
        if not alias_dir.is_absolute():
            alias_dir = out_root / alias_dir
        alias_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(alias_dir / f"{alias['stem']}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for extra in result.get("extra_figures", []) or []:
        extra_fig = extra["figure"]
        stem = extra["stem"]
        extra_fig.savefig(pdir / f"{stem}.png", dpi=300, bbox_inches="tight")
        for alias in extra.get("output_aliases", []) or []:
            alias_dir = Path(alias.get("dir", pdir))
            if not alias_dir.is_absolute():
                alias_dir = out_root / alias_dir
            alias_dir.mkdir(parents=True, exist_ok=True)
            extra_fig.savefig(alias_dir / f"{alias.get('stem', stem)}.png",
                              dpi=300, bbox_inches="tight")
        plt.close(extra_fig)

    cfg["panel_id"] = panel_id
    cfg["figure_n"] = figure_n
    cfg["timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
    yaml_path = pdir / f"{panel_id}_config.yaml"
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(_yaml_safe(cfg), fh, sort_keys=False)
    for alias in result.get("output_aliases", []) or []:
        alias_dir = Path(alias.get("dir", pdir))
        if not alias_dir.is_absolute():
            alias_dir = out_root / alias_dir
        alias_dir.mkdir(parents=True, exist_ok=True)
        with open(alias_dir / f"{alias['stem']}_config.yaml", "w") as fh:
            yaml.safe_dump(_yaml_safe(cfg), fh, sort_keys=False)
    for extra in result.get("extra_figures", []) or []:
        extra_cfg = dict(cfg)
        extra_cfg["panel_id"] = extra["stem"]
        with open(pdir / f"{extra['stem']}_config.yaml", "w") as fh:
            yaml.safe_dump(_yaml_safe(extra_cfg), fh, sort_keys=False)
        for alias in extra.get("output_aliases", []) or []:
            alias_dir = Path(alias.get("dir", pdir))
            if not alias_dir.is_absolute():
                alias_dir = out_root / alias_dir
            alias_dir.mkdir(parents=True, exist_ok=True)
            with open(alias_dir / f"{alias.get('stem', extra['stem'])}_config.yaml", "w") as fh:
                yaml.safe_dump(_yaml_safe(extra_cfg), fh, sort_keys=False)

    for name, obj in data.items():
        csv_path = pdir / f"{panel_id}_data_{name}.csv"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(csv_path, index=False)
        elif isinstance(obj, pd.Series):
            obj.to_frame().to_csv(csv_path, index=False)
        elif isinstance(obj, np.ndarray):
            df = pd.DataFrame(obj)
            df.to_csv(csv_path, index=False)
        else:
            try:
                pd.DataFrame(obj).to_csv(csv_path, index=False)
            except Exception:
                # Fallback: dump as a JSON line.
                with open(csv_path.with_suffix(".json"), "w") as fh:
                    yaml.safe_dump(_yaml_safe(obj), fh)
        for alias in result.get("output_aliases", []) or []:
            alias_dir = Path(alias.get("dir", pdir))
            if not alias_dir.is_absolute():
                alias_dir = out_root / alias_dir
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_csv = alias_dir / f"{alias['stem']}_data_{name}.csv"
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(alias_csv, index=False)
            elif isinstance(obj, pd.Series):
                obj.to_frame().to_csv(alias_csv, index=False)
            elif isinstance(obj, np.ndarray):
                pd.DataFrame(obj).to_csv(alias_csv, index=False)
            else:
                try:
                    pd.DataFrame(obj).to_csv(alias_csv, index=False)
                except Exception:
                    with open(alias_csv.with_suffix(".json"), "w") as fh:
                        yaml.safe_dump(_yaml_safe(obj), fh)
            if alias.get("single_data_name") == name:
                flat_csv = alias_dir / f"{alias['stem']}_data.csv"
                if isinstance(obj, pd.DataFrame):
                    obj.to_csv(flat_csv, index=False)
                elif isinstance(obj, pd.Series):
                    obj.to_frame().to_csv(flat_csv, index=False)
                elif isinstance(obj, np.ndarray):
                    pd.DataFrame(obj).to_csv(flat_csv, index=False)
                else:
                    try:
                        pd.DataFrame(obj).to_csv(flat_csv, index=False)
                    except Exception:
                        with open(flat_csv.with_suffix(".json"), "w") as fh:
                            yaml.safe_dump(_yaml_safe(obj), fh)

    if registry is not None:
        registry.append(str(png))
    return pdir


def data_not_present_result(panel_id: str, figure_n: int, title: str,
                             message: str) -> dict:
    """Build a panel result whose figure is an explicit 'data not present' plate."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.7, title, ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text(0.5, 0.4, textwrap.fill(message, 70),
            ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.1, "data not present in raw_data/",
            ha="center", va="center", fontsize=9,
            color="#666", style="italic")
    return {
        "panel_id": panel_id,
        "figure_n": figure_n,
        "figure": fig,
        "data": {},
        "config": {"status": "data_not_present", "message": message},
    }
