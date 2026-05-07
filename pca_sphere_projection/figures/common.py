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


# ---------------------------------------------------------------------------
# Multi-dataset PCA helpers used by Figure 2 panels
# ---------------------------------------------------------------------------

# Canonical dataset order across Fig 2 panels ("all datasets" in B/C/D).
FIG2_ALL_DATASETS = [
    "celegan",
    "klein",
    "hesc",
    "planaria",
    "human_germ_cell",
    "pre_implant_human_embryo",
    "uc_epi",
]

DATASET_DISPLAY = {
    "celegan": "C. elegans",
    "klein": "Klein mESC",
    "hesc": "hESC",
    "planaria": "Planaria",
    "human_germ_cell": "Human germ cell",
    "pre_implant_human_embryo": "Pre-impl. embryo",
    "uc_epi": "UC epithelium",
}


def _raw_data_dir() -> Path:
    return ROOT / "raw_data"


def _raw_data_subdir(ds: str) -> Optional[Path]:
    aliases = {"hesc": "hESC", "hESC": "hESC"}
    folder = _raw_data_dir() / aliases.get(ds, ds)
    return folder if folder.exists() else None


def _cache_dir() -> Path:
    p = ROOT / "outputs" / "figures" / "Fig2" / "_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_raw_expression(ds: str):
    """Load raw expression for a dataset using the existing IO helpers."""
    folder = _raw_data_subdir(ds)
    if folder is None:
        return None
    from .. import io as psp_io

    if ds == "klein":
        return psp_io.load_bz2_klein_dataset(str(folder))
    if ds == "celegan":
        return psp_io.load_celegan(str(folder))
    if ds == "uc_epi":
        return psp_io.load_uc_epi(str(folder))
    if ds in ("hesc", "hESC"):
        return psp_io.load_hesc(str(folder))
    if ds in ("planaria", "human_germ_cell", "pre_implant_human_embryo"):
        return psp_io.load_cytotrace_rds_dataset(str(folder))
    return None


def _scanpy_preprocess(expr, *, hvg: bool, n_top_genes: int = 2000,
                       n_components: int = 20, max_value: float = 10.0,
                       seed: int = 0):
    """Canonical Scanpy normalize -> log1p -> [HVG seurat] -> scale -> PCA.

    Returns dict with scores, explained_variance_ratio, hvg_genes (list or
    None for all-gene), n_genes_used, scanpy_version.
    """
    import scanpy as sc
    import anndata as ad

    sc.settings.verbosity = 0
    X = expr.X
    var = pd.DataFrame(index=list(expr.gene_names))
    obs = pd.DataFrame(index=list(expr.cell_ids))
    adata = ad.AnnData(X=X, obs=obs, var=var)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    hvg_genes = None
    if hvg:
        sc.pp.highly_variable_genes(
            adata, flavor="seurat",
            n_top_genes=min(n_top_genes, adata.n_vars - 1),
            inplace=True,
        )
        adata = adata[:, adata.var["highly_variable"]].copy()
        hvg_genes = list(adata.var_names)

    sc.pp.scale(adata, max_value=max_value)
    n_comp = int(min(n_components, adata.n_vars - 1, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=n_comp, random_state=seed, zero_center=True)
    return {
        "scores": np.asarray(adata.obsm["X_pca"], dtype=np.float32),
        "explained_variance_ratio": np.asarray(
            adata.uns["pca"]["variance_ratio"], dtype=np.float32),
        "hvg_genes": hvg_genes,
        "n_genes_used": int(adata.n_vars),
        "n_cells_used": int(adata.n_obs),
        "scanpy_version": sc.__version__,
    }


def compute_pca_scores(ds: str, *, n_components: int = 20, hvg: bool = True,
                       n_top_genes: int = 2000, seed: int = 0,
                       max_cells: Optional[int] = None,
                       use_cache: bool = True):
    """Run the canonical Scanpy preprocessing -> PCA pipeline (cached).

    Returns the dict from `_scanpy_preprocess` (plus `n_components_returned`).
    Returns None when the raw data is not available.
    """
    cache = _cache_dir() / (
        f"{ds}_{'hvg' + str(n_top_genes) if hvg else 'all'}"
        f"_pc{n_components}_seed{seed}"
        f"{f'_n{max_cells}' if max_cells else ''}.npz"
    )
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {
            "scores": z["scores"],
            "explained_variance_ratio": z["explained_variance_ratio"],
            "hvg_genes": list(z["hvg_genes"]) if "hvg_genes" in z.files
                         and z["hvg_genes"].dtype.kind in ("U", "O") else None,
            "n_genes_used": int(z["n_genes_used"]),
            "n_cells_used": int(z["n_cells_used"]),
            "scanpy_version": str(z["scanpy_version"]),
        }
    expr = load_raw_expression(ds)
    if expr is None:
        return None
    if max_cells is not None and expr.X.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(expr.X.shape[0], int(max_cells), replace=False))
        expr.X = expr.X[sel]
        expr.cell_ids = [expr.cell_ids[i] for i in sel]
        if expr.metadata is not None and len(expr.metadata) == max(sel) + 1 + 0:
            try:
                expr.metadata = expr.metadata.iloc[sel].copy()
            except Exception:
                pass
    out = _scanpy_preprocess(
        expr, hvg=hvg, n_top_genes=n_top_genes,
        n_components=n_components, seed=seed,
    )
    if use_cache:
        np.savez(
            cache,
            scores=out["scores"],
            explained_variance_ratio=out["explained_variance_ratio"],
            hvg_genes=np.asarray(out["hvg_genes"]) if out["hvg_genes"] is not None else np.asarray([]),
            n_genes_used=out["n_genes_used"],
            n_cells_used=out["n_cells_used"],
            scanpy_version=out["scanpy_version"],
        )
    return out


def load_unit_coords_pc13(ds: str, *, max_cells: Optional[int] = None,
                          seed: int = 0) -> Optional[np.ndarray]:
    """Best-effort PC1-3 unit-vector loader.

    Tries (in order): outputs/raw_expression_validation/<ds>/_artifacts.npz,
    examples/<ds>_pca.csv, then a fresh PCA run via compute_pca_scores.
    """
    val_out = ROOT / "outputs" / "raw_expression_validation"
    art = load_artifacts(val_out, ds)
    if art is not None and "coords3_unit" in art.files:
        coords = np.asarray(art["coords3_unit"], dtype=float)
    else:
        df = load_pca_csv(ds)
        if df is not None and {"PC1", "PC2", "PC3"}.issubset(df.columns):
            coords = to_unit(df[["PC1", "PC2", "PC3"]].to_numpy(dtype=float))
        else:
            r = compute_pca_scores(ds, n_components=3, hvg=True, seed=seed)
            if r is None:
                return None
            coords = to_unit(np.asarray(r["scores"][:, :3], dtype=float))
    if max_cells is not None and coords.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(coords.shape[0], int(max_cells), replace=False))
        coords = coords[idx]
    return coords


def load_regev_cell_cycle_genes() -> dict:
    """Load Regev/Kowalczyk cell-cycle gene list (S phase + G2/M).

    The standard layout of `regev_lab_cell_cycle_genes.txt` is 43 S-phase
    genes followed by 54 G2/M-phase genes (97 lines total).
    """
    p = ROOT / "raw_data" / "regev_lab_cell_cycle_genes.txt"
    if not p.exists():
        return {"all": [], "s_phase": [], "g2m_phase": [], "source": str(p)}
    genes = [line.strip() for line in open(p) if line.strip()]
    return {"all": genes, "s_phase": genes[:43], "g2m_phase": genes[43:],
            "source": str(p)}


def fit_great_circle_curve(unit: np.ndarray, *, n_points: int = 360):
    """Return (curve_xyz, gc_dict) for plotting."""
    from .. import sphere_stats as psp_sph
    gc = psp_sph.fit_great_circle(unit)
    n = gc["normal"]
    if abs(n[2]) < 0.9:
        u = np.cross(n, [0.0, 0.0, 1.0]); u /= np.linalg.norm(u)
    else:
        u = np.cross(n, [1.0, 0.0, 0.0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    curve = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    return curve, gc


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
