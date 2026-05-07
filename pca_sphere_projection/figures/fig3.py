"""Figure 3 — biological interpretation of spherical coordinates.

Scientific message:
    With biologically pre-specified roots, polar angle θ (geodesic distance
    from root) tracks loss of CytoTRACE stemness while radial norm captures
    distinct, non-trajectory biology — most clearly reflected in cell-cycle
    activity. θ is *not* a universal pseudotime; the pseudotime alignment
    holds only when the sphere is rooted at a biologically defined high-
    potency state.

Datasets (canonical CytoTRACE/stemness benchmarks):
    hESC, planaria, human_germ_cell, pre_implant_human_embryo

Preprocessing (canonical scanpy pipeline, see Fig 2 cache):
    sc.pp.normalize_total(target_sum=1e4) -> sc.pp.log1p
    -> sc.pp.highly_variable_genes(flavor="seurat", n_top_genes=2000)
    -> sc.pp.scale(max_value=10) -> sc.tl.pca(n_comps=20)
    -> PC1-3 -> L2 unit projection -> rotate root centroid to north pole

Roots are pre-specified, not chosen to maximize correlation:
    hESC -> "hESC" / planaria -> "x1" /
    human_germ_cell -> "19W" / pre_implant_human_embryo -> "Zygote"
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.stats import rankdata, spearmanr, t as student_t

from .. import sphere_stats as psp_sph
from . import common


DATASETS = ["hESC", "planaria", "human_germ_cell",
            "pre_implant_human_embryo"]

DATASET_KEY = {  # raw_data folder name -> canonical key (lowercase for cache)
    "hESC": "hesc",
    "planaria": "planaria",
    "human_germ_cell": "human_germ_cell",
    "pre_implant_human_embryo": "pre_implant_human_embryo",
}

DISPLAY = {
    "hESC": "hESC",
    "planaria": "Planaria",
    "human_germ_cell": "Human germ cell",
    "pre_implant_human_embryo": "Pre-impl. embryo",
}

PALETTE = {
    "hESC": "#e45756",
    "planaria": "#b279a2",
    "human_germ_cell": "#9d755d",
    "pre_implant_human_embryo": "#ff9da6",
}

# Pre-specified biological roots — chosen from prior knowledge of which
# population is the most stem-like / earliest, NOT by correlation maximization.
BIOLOGICAL_ROOTS = {
    "hESC": "hESC",
    "planaria": "x1",
    "human_germ_cell": "19W",
    "pre_implant_human_embryo": "Zygote",
}

PANEL_DEFAULT = {
    "datasets": list(DATASETS),
    "n_top_genes": 2000,
    "n_components": 20,
    "preprocessing": (
        "sc.pp.normalize_total(target_sum=1e4) -> sc.pp.log1p "
        "-> sc.pp.highly_variable_genes(flavor='seurat', n_top_genes=2000) "
        "-> sc.pp.scale(max_value=10) -> sc.tl.pca(n_comps=20)"),
    "biological_roots": dict(BIOLOGICAL_ROOTS),
    "root_choice_note": (
        "Roots are pre-specified from biological prior knowledge; "
        "Fig 3 panels do not select roots by maximizing correlation. "
        "Root sweeps are saved as supplementary diagnostics."),
    "seed": 0,
}


# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------

def _rotate_to_north_pole(unit: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Rotate every row of `unit` so that `anchor` (a unit vector) maps to +z."""
    a = anchor / max(np.linalg.norm(anchor), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z); s = float(np.linalg.norm(v)); c = float(a @ z)
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))
    return unit @ R.T


def _build_geometry(ds_folder: str, *, n_top_genes: int, n_components: int,
                    seed: int) -> dict:
    """Run the canonical Fig 2 / Fig 3 preprocessing for one dataset.

    Returns a dict with PC scores, unit vectors aligned to root, theta,
    radial, CytoTRACE stemness, phenotype labels, root mask, HVG list,
    PCA explained variance, and bookkeeping fields.
    """
    expr = common.load_raw_expression(DATASET_KEY[ds_folder])
    if expr is None:
        raise FileNotFoundError(f"raw_data/{ds_folder} missing")
    res = common.compute_pca_scores(
        DATASET_KEY[ds_folder], n_components=n_components, hvg=True,
        n_top_genes=n_top_genes, seed=seed)
    if res is None:
        raise RuntimeError(f"failed to compute PCA for {ds_folder}")
    scores = np.asarray(res["scores"], dtype=float)
    if scores.shape[1] < 3:
        raise RuntimeError(f"only {scores.shape[1]} PCs available for {ds_folder}")
    pc3 = scores[:, :3]
    radial = np.linalg.norm(pc3, axis=1)
    radial[radial == 0] = 1e-12
    unit_raw = pc3 / radial[:, None]

    pheno = expr.metadata.get("Phenotype")
    if pheno is None:
        raise RuntimeError(f"{ds_folder} has no Phenotype labels in metadata")
    pheno = pd.Series(pheno).astype(str).reset_index(drop=True)

    root_label = BIOLOGICAL_ROOTS[ds_folder]
    available = sorted(pheno.dropna().unique())
    if root_label not in available:
        raise ValueError(
            f"pre-specified root {root_label!r} not present in {ds_folder}; "
            f"available: {available}")
    root_mask = (pheno == root_label).to_numpy()
    if root_mask.sum() < 1:
        raise ValueError(f"{ds_folder}: no cells in root cluster {root_label!r}")

    # Centroid of root in unit space, re-normalised. This is the rotation
    # anchor — using the unit-space centroid (not the raw-PC mean) keeps the
    # rotation a pure rigid transform on S^2.
    root_centroid = unit_raw[root_mask].mean(axis=0)
    root_centroid /= max(np.linalg.norm(root_centroid), 1e-12)
    unit = _rotate_to_north_pole(unit_raw, root_centroid)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))

    cyto = expr.metadata.get("CytoTRACE")
    if cyto is None:
        raise RuntimeError(f"{ds_folder} metadata has no CytoTRACE column")
    cyto = pd.to_numeric(pd.Series(cyto), errors="coerce").to_numpy()

    return {
        "expr": expr,
        "dataset": ds_folder,
        "scores": scores,
        "pc3": pc3,
        "unit": unit,
        "unit_raw": unit_raw,
        "theta": theta,
        "radial": radial,
        "cytotrace": cyto,
        "phenotype": pheno.to_numpy(),
        "root_mask": root_mask,
        "root_label": root_label,
        "root_centroid_unit": root_centroid,
        "hvg_genes": res["hvg_genes"],
        "explained_variance_ratio": np.asarray(res["explained_variance_ratio"]),
        "n_cells": int(scores.shape[0]),
        "n_genes_used": int(res["n_genes_used"]),
        "scanpy_version": res["scanpy_version"],
    }


def _safe_geometry(ds: str, cfg: dict):
    try:
        return _build_geometry(
            ds, n_top_genes=int(cfg["n_top_genes"]),
            n_components=int(cfg["n_components"]),
            seed=int(cfg["seed"]))
    except Exception as exc:
        return {"error": str(exc), "dataset": ds}


def _xyz_to_lonlat(unit: np.ndarray):
    lon = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))
    return lon, lat


# ---------------------------------------------------------------------------
# Cell-cycle scoring (Regev list)
# ---------------------------------------------------------------------------

def _cell_cycle_score(expr, gene_list_all, gene_list_s, gene_list_g2m,
                      *, n_top_genes: int = 2000) -> dict:
    """Compute a canonical cell-cycle score with scanpy.tl.score_genes_cell_cycle.

    Mirrors Fig 2 preprocessing: normalize_total -> log1p -> scale.
    Returns dict with overlap counts and per-cell composite cell-cycle score
    (G2M − S, with mean of (S, G2M) reported as `cc_score` for plotting).
    """
    import scanpy as sc
    import anndata as ad
    sc.settings.verbosity = 0

    obs = pd.DataFrame(index=list(expr.cell_ids))
    var = pd.DataFrame(index=list(expr.gene_names))
    adata = ad.AnnData(X=expr.X, obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    var_names = set(adata.var_names)
    s_overlap = [g for g in gene_list_s if g in var_names]
    g2m_overlap = [g for g in gene_list_g2m if g in var_names]
    all_overlap = [g for g in gene_list_all if g in var_names]

    if len(s_overlap) < 5 and len(g2m_overlap) < 5:
        return {"score": None, "n_overlap_s": len(s_overlap),
                "n_overlap_g2m": len(g2m_overlap),
                "n_overlap_all": len(all_overlap),
                "fraction_overlap": float(len(all_overlap) / max(len(gene_list_all), 1))}

    if len(s_overlap) >= 5 and len(g2m_overlap) >= 5:
        try:
            sc.tl.score_genes_cell_cycle(
                adata, s_genes=s_overlap, g2m_genes=g2m_overlap,
                use_raw=False, random_state=0)
            s_score = np.asarray(adata.obs["S_score"], dtype=float)
            g2m_score = np.asarray(adata.obs["G2M_score"], dtype=float)
            score = (s_score + g2m_score) / 2.0
        except Exception:
            sc.tl.score_genes(adata, gene_list=all_overlap,
                              use_raw=False, random_state=0,
                              score_name="cc_score")
            s_score = g2m_score = None
            score = np.asarray(adata.obs["cc_score"], dtype=float)
    else:
        sc.tl.score_genes(adata, gene_list=all_overlap,
                          use_raw=False, random_state=0,
                          score_name="cc_score")
        s_score = g2m_score = None
        score = np.asarray(adata.obs["cc_score"], dtype=float)

    return {
        "score": score,
        "s_score": s_score, "g2m_score": g2m_score,
        "n_overlap_s": len(s_overlap),
        "n_overlap_g2m": len(g2m_overlap),
        "n_overlap_all": len(all_overlap),
        "fraction_overlap": float(len(all_overlap) / max(len(gene_list_all), 1)),
    }


# ---------------------------------------------------------------------------
# Panel A — equirectangular CytoTRACE projections, four datasets
# ---------------------------------------------------------------------------

def panel_A(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config)
    geos = [_safe_geometry(ds, cfg) for ds in cfg["datasets"]]
    valid = [g for g in geos if "error" not in g]
    if not valid:
        msg = "; ".join(f"{g['dataset']}: {g['error']}" for g in geos)
        return common.data_not_present_result(
            "Fig3A", 3, "Fig 3A — Stemness gradients on spherical PCA",
            f"No datasets loaded ({msg}).")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), facecolor="white")
    axes = axes.ravel()
    per_cell_rows = []
    summary_rows = []
    for ax, geo in zip(axes, geos):
        if "error" in geo:
            ax.text(0.5, 0.5, f"{geo['dataset']}\n{geo['error']}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8.5, color="#888")
            ax.set_axis_off()
            continue
        lon, lat = _xyz_to_lonlat(geo["unit"])
        c = np.asarray(geo["cytotrace"], dtype=float)
        valid_mask = np.isfinite(c)
        order = np.argsort(c[valid_mask])  # plot high-CytoTRACE on top
        sub = np.where(valid_mask)[0][order]
        # adapt point size to cell count for visual consistency
        n_cells = geo["n_cells"]
        ps = 22.0 if n_cells < 200 else (10.0 if n_cells < 1500 else 4.0)
        sc = ax.scatter(lon[sub], lat[sub], c=c[sub], s=ps,
                        cmap="magma", alpha=0.85, linewidths=0,
                        rasterized=True)
        # outline root cluster
        root_idx = np.where(geo["root_mask"])[0]
        if root_idx.size > 0:
            ax.scatter(lon[root_idx], lat[root_idx],
                       s=ps + 14, facecolors="none", edgecolors="#0d3b66",
                       linewidths=0.7, alpha=0.95, zorder=5)
        ax.set_xlim(-180, 180); ax.set_ylim(-92, 92)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-90, 0, 90])
        ax.set_title(f"{DISPLAY[geo['dataset']]}    n={n_cells}    "
                     f"root: {geo['root_label']}",
                     fontsize=9.5, pad=4)
        ax.tick_params(width=0.7, length=2.5, labelsize=7.5)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.set_xlabel("longitude (°)", fontsize=8.5, labelpad=1)
        ax.set_ylabel("latitude (°)", fontsize=8.5, labelpad=1)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("CytoTRACE", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        # annotate root with a single small label near the north pole, off-axis
        ax.annotate("root", xy=(170, 87), xytext=(170, 87),
                    fontsize=7, color="#0d3b66", ha="right", va="top")

        for cid, lo, la, t, r, cc, ph in zip(
                geo["expr"].cell_ids, lon, lat,
                geo["theta"], geo["radial"], c, geo["phenotype"]):
            per_cell_rows.append({
                "dataset": geo["dataset"], "cell_id": cid,
                "longitude_deg": float(lo), "latitude_deg": float(la),
                "theta_rad": float(t), "radial_norm": float(r),
                "cytotrace": float(cc) if np.isfinite(cc) else np.nan,
                "phenotype": ph, "is_root_cell": bool(ph == geo["root_label"]),
            })
        summary_rows.append({
            "dataset": geo["dataset"], "root_label": geo["root_label"],
            "n_cells": int(geo["n_cells"]),
            "n_root_cells": int(geo["root_mask"].sum()),
            "explained_var_PC1": float(geo["explained_variance_ratio"][0]),
            "explained_var_PC2": float(geo["explained_variance_ratio"][1])
                                 if geo["explained_variance_ratio"].size > 1 else np.nan,
            "explained_var_PC3": float(geo["explained_variance_ratio"][2])
                                 if geo["explained_variance_ratio"].size > 2 else np.nan,
            "n_genes_used": int(geo["n_genes_used"]),
            "scanpy_version": geo["scanpy_version"],
        })

    fig.suptitle("Stemness gradients on spherical PCA",
                 fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout(pad=0.5)

    return {
        "panel_id": "Fig3A", "figure_n": 3, "figure": fig, "config": cfg,
        "output_aliases": [{"dir": "fig3",
                            "stem": "Fig3A_cytotrace_sphere_four_datasets"}],
        "data": {
            "per_cell": pd.DataFrame(per_cell_rows),
            "summary": pd.DataFrame(summary_rows),
        },
    }


# ---------------------------------------------------------------------------
# Panel B — theta vs CytoTRACE
# ---------------------------------------------------------------------------

def panel_B(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config)
    geos = [_safe_geometry(ds, cfg) for ds in cfg["datasets"]]
    valid_geos = [g for g in geos if "error" not in g]
    if not valid_geos:
        return common.data_not_present_result(
            "Fig3B", 3, "Fig 3B — θ vs CytoTRACE",
            "No datasets loaded.")
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), facecolor="white")
    axes = axes.ravel()
    summary_rows = []
    per_cell_rows = []
    for ax, geo in zip(axes, geos):
        if "error" in geo:
            ax.text(0.5, 0.5, f"{geo['dataset']}\n{geo['error']}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8.5, color="#888")
            ax.set_axis_off()
            continue
        cyto = np.asarray(geo["cytotrace"], dtype=float)
        theta = np.asarray(geo["theta"], dtype=float)
        mask = np.isfinite(cyto) & np.isfinite(theta)
        if mask.sum() < 5:
            ax.text(0.5, 0.5, "insufficient data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        rho, p = spearmanr(theta[mask], cyto[mask])
        n_use = int(mask.sum())
        # density scatter / hexbin
        if n_use >= 600:
            hb = ax.hexbin(theta[mask], cyto[mask], gridsize=40,
                           cmap="magma", mincnt=1, linewidths=0)
            cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("# cells", fontsize=8)
            cb.ax.tick_params(labelsize=7)
        else:
            ax.scatter(theta[mask], cyto[mask], s=14, alpha=0.65,
                       color=PALETTE[geo["dataset"]],
                       edgecolors="#222", linewidths=0.25)
        ax.set_xlabel(r"$\theta$ from biological root (rad)",
                      fontsize=9, labelpad=2)
        ax.set_ylabel("CytoTRACE", fontsize=9, labelpad=2)
        ax.set_title(f"{DISPLAY[geo['dataset']]}    "
                     fr"$\rho$={rho:+.2f}, p={p:.1e}, n={n_use}",
                     fontsize=9.5, pad=4)
        ax.tick_params(width=0.7, length=2.5, labelsize=8)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

        summary_rows.append({
            "dataset": geo["dataset"], "root_label": geo["root_label"],
            "spearman_rho_theta_cytotrace": float(rho),
            "p_value": float(p),
            "n_used": n_use,
            "expected_sign": "negative (θ increases away from high-stemness root)",
            "observed_sign": "negative" if rho < 0 else "positive",
        })
        for cid, t, c in zip(geo["expr"].cell_ids, theta, cyto):
            per_cell_rows.append({
                "dataset": geo["dataset"], "cell_id": cid,
                "theta_rad": float(t),
                "cytotrace": float(c) if np.isfinite(c) else np.nan,
            })
    fig.suptitle(
        r"Polar angle $\theta$ tracks CytoTRACE when rooted at biological"
        " stem state",
        fontsize=11.5, fontweight="bold", y=1.00)
    fig.tight_layout(pad=0.5)
    return {
        "panel_id": "Fig3B", "figure_n": 3, "figure": fig, "config": cfg,
        "output_aliases": [{"dir": "fig3",
                            "stem": "Fig3B_theta_vs_cytotrace_four_datasets"}],
        "data": {
            "summary": pd.DataFrame(summary_rows),
            "per_cell": pd.DataFrame(per_cell_rows),
        },
    }


# ---------------------------------------------------------------------------
# Gene-coordinate association helpers (used by panels C and D)
# ---------------------------------------------------------------------------

PANEL_CD_DEFAULT = {
    "datasets": ["hESC", "human_germ_cell", "pre_implant_human_embryo"],
    "n_top_genes": 2000,
    "n_components": 20,
    "rho_threshold": 0.30,
    "specificity_threshold": 0.15,
    "fdr_threshold": 0.05,
    "preprocessing": (
        "sc.pp.normalize_total(target_sum=1e4) -> sc.pp.log1p "
        "-> sc.pp.highly_variable_genes(flavor='seurat', n_top_genes=2000) "
        "(scale + PCA used for Fig 3A/B; Fig 3C/D uses log-normalised, "
        "non-scaled HVG expression to keep gene-level Spearman invariant)"),
    "biological_roots": dict(BIOLOGICAL_ROOTS),
    "enrichment_libraries": [
        "GO_Biological_Process_2023",
        "MSigDB_Hallmark_2020",
        "Reactome_2022",
    ],
    "category_keywords": {
        "Cell cycle": [
            r"cell\s*cycle", r"\bmitot", r"\bmitos", r"DNA[ _]replication",
            r"\bE2F\b", r"G2[/-_ ]?M", r"chromosome\s*segregation",
            r"sister[- ]chromatid", r"kinetochore",
        ],
        "Ribosome / translation": [
            r"\bribosom", r"\btranslat", r"\brRNA\b",
            r"protein\s*synthesis",
        ],
        "Pluripotency / stemness": [
            r"\bstem[\s-]?cell", r"\bpluripoten", r"\bself[\s-]?renewal",
            r"embryonic\s*stem", r"germ[\s-]?cell",
        ],
        "Lineage differentiation": [
            r"\bdifferentiation\b", r"lineage\s*commitment",
            r"fate\s*determination", r"fate\s*specification",
            r"\bspecification\b", r"epithelial.*differentiation",
            r"neuron\s*differentiation", r"muscle.*differentiation",
        ],
        "Morphogenesis / development": [
            r"morphogenesis", r"\bdevelopment\b",
            r"pattern\s*specification", r"organogenesis",
            r"\btissue\b", r"\borgan\b", r"embryonic\s*development",
        ],
    },
    "seed": 0,
}

CATEGORY_ORDER = list(PANEL_CD_DEFAULT["category_keywords"].keys())

GENE_CLASS_ORDER = ["theta_high_radial_low", "radial_high_theta_low",
                    "both_high"]
GENE_CLASS_DISPLAY = {
    "theta_high_radial_low": "θ-specific",
    "radial_high_theta_low": "radial-specific",
    "both_high": "both",
}
GENE_CLASS_COLOR = {
    "theta_high_radial_low": "#1f6f8b",   # cool blue
    "radial_high_theta_low": "#d9b450",   # warm gold
    "both_high": "#7a3b69",               # plum
    "background": "#cccccc",
}


def _fig3_cache_dir() -> Path:
    p = common.ROOT / "outputs" / "figures" / "Fig3" / "_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    finite = np.isfinite(p)
    out = np.full(n, np.nan)
    if finite.sum() == 0:
        return out
    order = np.argsort(p[finite])
    ranked = p[finite][order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    out[finite] = adj[inv]
    return out


def _vectorized_spearman(X: np.ndarray, y: np.ndarray):
    """Per-column Spearman ρ and two-sided p-value of X (n_cells x n_genes)
    against the 1D vector y.

    Uses rank transformation + Pearson + Student-t approximation for the p-
    value (matches scipy.stats.spearmanr to within ties; avoids the n_genes-
    way python loop)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    Xr = np.apply_along_axis(rankdata, 0, X)
    yr = rankdata(y)
    Xrm = Xr - Xr.mean(axis=0, keepdims=True)
    yrm = yr - yr.mean()
    Xrs = Xrm.std(axis=0, ddof=0)
    Xrs = np.where(Xrs > 0, Xrs, np.nan)
    yrs = yrm.std(ddof=0) or 1.0
    rho = (Xrm * yrm[:, None]).mean(axis=0) / (Xrs * yrs)
    rho = np.clip(rho, -1.0, 1.0)
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = rho * np.sqrt(df / np.clip(1 - rho ** 2, 1e-12, None))
    pval = 2 * student_t.sf(np.abs(t_stat), df=df)
    return rho, pval


def _load_hvg_expression(ds_folder: str, *, n_top_genes: int, seed: int):
    """Recompute (or load) the HVG-filtered, log-normalised expression matrix.

    The cache key encodes preprocessing parameters so we never silently mix
    incompatible caches. The HVG list is verified against the Fig 2 PCA cache
    when present.
    """
    key = DATASET_KEY[ds_folder]
    cache = _fig3_cache_dir() / f"{key}_hvg{n_top_genes}_seed{seed}_lognorm.npz"
    pca_cache_genes = None
    pca_res = common.compute_pca_scores(
        key, n_components=PANEL_DEFAULT["n_components"], hvg=True,
        n_top_genes=n_top_genes, seed=seed)
    if pca_res is not None and pca_res.get("hvg_genes"):
        pca_cache_genes = list(pca_res["hvg_genes"])
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        cached_genes = list(z["genes"].astype(str))
        if pca_cache_genes is not None and cached_genes != pca_cache_genes:
            warnings.warn(
                f"Fig3 HVG cache for {ds_folder} is out-of-sync with the "
                "Fig 2 PCA cache; rebuilding to keep preprocessing aligned.")
        else:
            return {
                "X": z["X"].astype(np.float32),
                "genes": cached_genes,
                "cell_ids": list(z["cell_ids"].astype(str)),
                "preprocessing_hash": str(z["preprocessing_hash"]),
                "scanpy_version": str(z["scanpy_version"]),
            }
    # Recompute via scanpy with the SAME parameters as Fig 2.
    import scanpy as sc
    import anndata as ad
    sc.settings.verbosity = 0

    expr = common.load_raw_expression(key)
    if expr is None:
        raise FileNotFoundError(f"raw_data/{ds_folder} missing")
    adata = ad.AnnData(
        X=expr.X,
        obs=pd.DataFrame(index=list(expr.cell_ids)),
        var=pd.DataFrame(index=list(expr.gene_names)))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat",
        n_top_genes=min(n_top_genes, adata.n_vars - 1),
        inplace=True)
    adata = adata[:, adata.var["highly_variable"]].copy()
    X = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray")
                   else adata.X, dtype=np.float32)
    genes = list(adata.var_names)
    pre_hash = (
        f"normalize_total=1e4|log1p|hvg=seurat|n_top_genes={n_top_genes}|"
        f"scanpy={sc.__version__}")
    np.savez(cache,
             X=X, genes=np.asarray(genes),
             cell_ids=np.asarray(list(expr.cell_ids)),
             preprocessing_hash=np.asarray(pre_hash),
             scanpy_version=np.asarray(sc.__version__))
    return {"X": X, "genes": genes, "cell_ids": list(expr.cell_ids),
            "preprocessing_hash": pre_hash, "scanpy_version": sc.__version__}


def _compute_gene_associations(ds_folder: str, geo: dict,
                                cfg: dict) -> pd.DataFrame:
    """Per-gene Spearman ρ vs theta and radial, with BH-FDR + class labels."""
    expr_pkg = _load_hvg_expression(
        ds_folder, n_top_genes=int(cfg["n_top_genes"]),
        seed=int(cfg["seed"]))
    X = expr_pkg["X"]
    genes = expr_pkg["genes"]
    if X.shape[0] != geo["theta"].shape[0]:
        raise RuntimeError(
            f"{ds_folder}: expression matrix has {X.shape[0]} cells but "
            f"geometry has {geo['theta'].shape[0]} cells")

    rho_t, p_t = _vectorized_spearman(X, np.asarray(geo["theta"], dtype=float))
    rho_r, p_r = _vectorized_spearman(X, np.asarray(geo["radial"], dtype=float))
    fdr_t = _bh_fdr(p_t)
    fdr_r = _bh_fdr(p_r)
    theta_spec = np.abs(rho_t) - np.abs(rho_r)
    radial_spec = np.abs(rho_r) - np.abs(rho_t)

    rho_thr = float(cfg["rho_threshold"])
    spec_thr = float(cfg["specificity_threshold"])
    fdr_thr = float(cfg["fdr_threshold"])
    klass = np.full(len(genes), "unspecific", dtype=object)
    is_theta = (np.abs(rho_t) >= rho_thr) & (fdr_t < fdr_thr) & \
               (theta_spec >= spec_thr)
    is_radial = (np.abs(rho_r) >= rho_thr) & (fdr_r < fdr_thr) & \
                (radial_spec >= spec_thr)
    is_both = (np.abs(rho_t) >= rho_thr) & (np.abs(rho_r) >= rho_thr) & \
              (fdr_t < fdr_thr) & (fdr_r < fdr_thr) & ~is_theta & ~is_radial
    klass[is_theta] = "theta_high_radial_low"
    klass[is_radial] = "radial_high_theta_low"
    klass[is_both] = "both_high"
    df = pd.DataFrame({
        "gene": genes,
        "rho_theta": rho_t, "p_theta": p_t, "fdr_theta": fdr_t,
        "rho_radial": rho_r, "p_radial": p_r, "fdr_radial": fdr_r,
        "theta_specificity": theta_spec,
        "radial_specificity": radial_spec,
        "gene_class": klass,
    })
    return df


# ---------------------------------------------------------------------------
# Enrichment helpers (gseapy / Enrichr, with disk cache)
# ---------------------------------------------------------------------------

def _enrich_cache_dir() -> Path:
    p = _fig3_cache_dir() / "enrichment"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _category_for_term(term: str, keyword_map: dict) -> Optional[str]:
    term_low = term.lower()
    for cat, patterns in keyword_map.items():
        for pat in patterns:
            if re.search(pat, term_low, flags=re.IGNORECASE):
                return cat
    return None


def _run_enrichr(genes: list, background: list, libraries: list,
                 *, cache_key: str) -> pd.DataFrame:
    """Run Enrichr ORA via gseapy with a disk cache.

    The cache key is suffixed with a stable hash of the requested libraries
    so changing the library set automatically invalidates the cache rather
    than silently re-using mismatched results.
    """
    import hashlib
    if not genes:
        return pd.DataFrame()
    lib_hash = hashlib.sha1(
        "|".join(sorted(libraries)).encode("utf-8")).hexdigest()[:8]
    cache = _enrich_cache_dir() / f"{cache_key}_lib{lib_hash}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import gseapy as gp
    out_dir = _enrich_cache_dir() / cache_key
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        enr = gp.enrichr(
            gene_list=list(genes),
            gene_sets=list(libraries),
            background=list(background) if background else None,
            organism="human",
            outdir=str(out_dir),
            no_plot=True,
            verbose=False,
        )
        df = enr.results.copy()
    except Exception as exc:
        warnings.warn(f"Enrichr failed for {cache_key}: {exc}")
        df = pd.DataFrame()
    df.to_csv(cache, index=False)
    return df


def _summarise_enrichment(enr_df: pd.DataFrame, keyword_map: dict
                          ) -> pd.DataFrame:
    """Annotate a gseapy enrichr result with our category mapping and keep
    the most significant term per category. gseapy.enrichr returns slightly
    different schemas depending on whether a custom background is supplied;
    we handle missing optional columns gracefully."""
    cols_out = ["category", "term", "library", "adjusted_p",
                "minus_log10_fdr", "overlap", "odds_ratio",
                "combined_score", "overlap_genes"]
    if enr_df is None or enr_df.empty:
        return pd.DataFrame(columns=cols_out)
    df = enr_df.copy()
    if "Adjusted P-value" not in df.columns:
        return pd.DataFrame(columns=cols_out)
    rename = {
        "Term": "term", "Adjusted P-value": "adjusted_p",
        "P-value": "p_value", "Gene_set": "library",
        "Overlap": "overlap", "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score", "Genes": "overlap_genes",
    }
    df = df.rename(columns={k: v for k, v in rename.items()
                            if k in df.columns})
    for opt in ("overlap", "odds_ratio", "combined_score", "overlap_genes"):
        if opt not in df.columns:
            df[opt] = np.nan
    df["category"] = df["term"].apply(lambda t: _category_for_term(t, keyword_map))
    df = df.dropna(subset=["category"]).copy()
    df["minus_log10_fdr"] = -np.log10(np.clip(df["adjusted_p"].astype(float),
                                                1e-300, 1.0))
    df = df.sort_values("adjusted_p")
    top = df.drop_duplicates("category", keep="first").copy()
    return top[cols_out]


# ---------------------------------------------------------------------------
# Panel C — gene-coordinate specificity scatter
# ---------------------------------------------------------------------------

def panel_C(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config({**PANEL_DEFAULT, **PANEL_CD_DEFAULT}, config)
    panel_dir = common.ROOT / "outputs" / "figures" / "Fig3"
    panel_dir.mkdir(parents=True, exist_ok=True)

    per_dataset_tables = {}
    summary_rows = []
    top_specific_rows = []
    valid = []
    for ds in cfg["datasets"]:
        geo = _safe_geometry(ds, cfg)
        if "error" in geo:
            summary_rows.append({"dataset": ds, "error": geo["error"]})
            continue
        try:
            df_assoc = _compute_gene_associations(ds, geo, cfg)
        except Exception as exc:
            summary_rows.append({"dataset": ds, "error": str(exc)})
            continue
        per_dataset_tables[ds] = df_assoc
        df_assoc.to_csv(
            panel_dir / f"Fig3C_{ds}_gene_coordinate_associations.csv",
            index=False)
        # top-K specific genes per dataset
        top_theta = df_assoc[df_assoc["gene_class"] == "theta_high_radial_low"
                              ].sort_values("theta_specificity",
                                            ascending=False).head(15)
        top_radial = df_assoc[df_assoc["gene_class"] == "radial_high_theta_low"
                               ].sort_values("radial_specificity",
                                             ascending=False).head(15)
        for rank, (_, row) in enumerate(top_theta.iterrows(), start=1):
            top_specific_rows.append({
                "dataset": ds, "rank_within_class": rank,
                "axis": "theta", "gene": row["gene"],
                "rho_theta": float(row["rho_theta"]),
                "rho_radial": float(row["rho_radial"]),
                "theta_specificity": float(row["theta_specificity"]),
                "fdr_theta": float(row["fdr_theta"]),
            })
        for rank, (_, row) in enumerate(top_radial.iterrows(), start=1):
            top_specific_rows.append({
                "dataset": ds, "rank_within_class": rank,
                "axis": "radial", "gene": row["gene"],
                "rho_theta": float(row["rho_theta"]),
                "rho_radial": float(row["rho_radial"]),
                "radial_specificity": float(row["radial_specificity"]),
                "fdr_radial": float(row["fdr_radial"]),
            })
        n_classes = df_assoc["gene_class"].value_counts().to_dict()
        summary_rows.append({
            "dataset": ds,
            "n_genes_tested": int(len(df_assoc)),
            "n_theta_specific": int(n_classes.get("theta_high_radial_low", 0)),
            "n_radial_specific": int(n_classes.get("radial_high_theta_low", 0)),
            "n_both_high": int(n_classes.get("both_high", 0)),
            "rho_threshold": float(cfg["rho_threshold"]),
            "specificity_threshold": float(cfg["specificity_threshold"]),
            "fdr_threshold": float(cfg["fdr_threshold"]),
            "fallback_top_ranked": bool(
                n_classes.get("theta_high_radial_low", 0) == 0
                or n_classes.get("radial_high_theta_low", 0) == 0),
        })
        valid.append(ds)

    if not per_dataset_tables:
        return common.data_not_present_result(
            "Fig3C", 3, "Fig 3C — gene coordinate specificity",
            "No datasets produced gene-coordinate associations.")

    # ---- plot (Nature/PNAS-style; adjustText for label de-overlap)
    try:
        from adjustText import adjust_text
        _have_adjust = True
    except ImportError:
        _have_adjust = False
        warnings.warn("adjustText not available; falling back to static labels")

    import matplotlib.patheffects as pe

    n_panels = len(per_dataset_tables)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(3.7 * n_panels + 1.6, 4.8),
                              facecolor="white", sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    rho_thr = float(cfg["rho_threshold"])
    for ax, ds in zip(axes, per_dataset_tables):
        df = per_dataset_tables[ds]
        klass = df["gene_class"].to_numpy()
        rho_t = df["rho_theta"].to_numpy()
        rho_r = df["rho_radial"].to_numpy()

        # axis limits with breathing room
        lim = max(abs(rho_t).max(), abs(rho_r).max(), rho_thr + 0.05)
        lim = min(lim * 1.10, 1.0)

        # subtle quadrant + threshold guides (drawn first, very light)
        ax.plot([-lim, lim], [-lim, lim], color="#ececec", lw=0.35,
                ls=(0, (4, 4)), zorder=1)
        ax.plot([-lim, lim], [lim, -lim], color="#ececec", lw=0.35,
                ls=(0, (4, 4)), zorder=1)
        for v in (-rho_thr, rho_thr):
            ax.axvline(v, color="#eeeeee", lw=0.35, ls=(0, (2, 3)), zorder=1)
            ax.axhline(v, color="#eeeeee", lw=0.35, ls=(0, (2, 3)), zorder=1)
        ax.axhline(0, color="#bdbdbd", lw=0.45, zorder=2)
        ax.axvline(0, color="#bdbdbd", lw=0.45, zorder=2)

        # background HVGs — very pale, small
        bg = klass == "unspecific"
        ax.scatter(rho_t[bg], rho_r[bg], s=2.2, alpha=0.06,
                   color="#cfcfcf", linewidths=0,
                   rasterized=True, zorder=2)
        # highlighted classes — saturated, slightly smaller
        for cls in GENE_CLASS_ORDER:
            m = klass == cls
            ax.scatter(rho_t[m], rho_r[m], s=6.5, alpha=0.92,
                       color=GENE_CLASS_COLOR[cls], linewidths=0,
                       label=GENE_CLASS_DISPLAY[cls],
                       rasterized=True, zorder=3)

        # collect labels for top θ-specific / radial-specific genes
        top_theta = df[df["gene_class"] == "theta_high_radial_low"
                       ].sort_values("theta_specificity",
                                     ascending=False).head(3)
        top_radial = df[df["gene_class"] == "radial_high_theta_low"
                        ].sort_values("radial_specificity",
                                      ascending=False).head(3)
        labelled = pd.concat([top_theta, top_radial])
        text_objects = []
        # marker rings around labelled genes
        for _, r in labelled.iterrows():
            cls_color = GENE_CLASS_COLOR[r["gene_class"]]
            ax.scatter(r["rho_theta"], r["rho_radial"],
                       s=22, facecolors="none", edgecolors=cls_color,
                       linewidths=0.65, alpha=0.95, zorder=4)
            t = ax.text(
                r["rho_theta"], r["rho_radial"], r["gene"],
                fontsize=7.2, color="#1a1a1a",
                ha="center", va="center", zorder=6,
                fontweight="medium")
            t.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
            text_objects.append(t)

        if _have_adjust and text_objects:
            # Repel from highlighted-class points (the ones that would overlap
            # labels) and from each other. Keep labels inside the panel.
            highlight_mask = klass != "unspecific"
            adjust_text(
                text_objects, ax=ax,
                x=rho_t[highlight_mask].tolist(),
                y=rho_r[highlight_mask].tolist(),
                expand=(1.35, 1.55),
                expand_axes=True,
                ensure_inside_axes=True,
                force_text=(0.7, 0.9),
                force_static=(0.35, 0.45),
                force_pull=(0.005, 0.005),
                force_explode=(0.5, 0.6),
                only_move={"text": "xy", "static": "xy", "explode": "xy"},
                prevent_crossings=True,
                min_arrow_len=6,
                arrowprops=dict(arrowstyle="-", color="#9a9a9a",
                                lw=0.4, alpha=0.85,
                                shrinkA=3, shrinkB=2),
                iter_lim=500,
            )

        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        n_t = (klass == "theta_high_radial_low").sum()
        n_r = (klass == "radial_high_theta_low").sum()
        n_b = (klass == "both_high").sum()
        ax.set_title(
            f"{DISPLAY[ds]}\nθ-spec n={n_t}   radial-spec n={n_r}   both n={n_b}",
            fontsize=9.5, pad=8, color="#222")
        ax.set_xlabel(r"Spearman $\rho$ (gene, $\theta$)", fontsize=9,
                      labelpad=4, color="#333")
        ax.tick_params(width=0.45, length=2.2, labelsize=7.8, colors="#555")
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.45)
            ax.spines[side].set_color("#888")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(r"Spearman $\rho$ (gene, radial norm)", fontsize=9,
                       labelpad=4, color="#333")
    handles = [Patch(facecolor=GENE_CLASS_COLOR[c],
                     edgecolor="none",
                     label=GENE_CLASS_DISPLAY[c])
               for c in GENE_CLASS_ORDER]
    handles.append(Patch(facecolor="#cfcfcf", edgecolor="none",
                         label="other HVGs"))
    axes[-1].legend(handles=handles, fontsize=7.6, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.04, 1.0),
                    handletextpad=0.45, borderaxespad=0.0,
                    title="gene class", title_fontsize=8)
    fig.suptitle(
        "Genes separate into angular and radial coordinate programs",
        fontsize=12, fontweight="bold", y=1.02, color="#111")
    fig.tight_layout(pad=0.8)
    fig.subplots_adjust(wspace=0.22, top=0.86, bottom=0.13,
                         left=0.07, right=0.92)

    pd.DataFrame(top_specific_rows).to_csv(
        panel_dir / "Fig3C_top_coordinate_specific_genes.csv", index=False)

    return {
        "panel_id": "Fig3C", "figure_n": 3, "figure": fig, "config": cfg,
        "output_aliases": [{"dir": "fig3",
                            "stem": "Fig3C_gene_coordinate_specificity"}],
        "data": {
            "summary": pd.DataFrame(summary_rows),
            "top_specific_genes": pd.DataFrame(top_specific_rows),
        },
    }


# ---------------------------------------------------------------------------
# Panel D — gene-set enrichment heatmap (database-backed via Enrichr)
# ---------------------------------------------------------------------------

def panel_D(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config({**PANEL_DEFAULT, **PANEL_CD_DEFAULT}, config)
    panel_dir = common.ROOT / "outputs" / "figures" / "Fig3"
    panel_dir.mkdir(parents=True, exist_ok=True)

    libraries = list(cfg["enrichment_libraries"])
    keyword_map = dict(cfg["category_keywords"])

    enrichment_records = []
    heatmap_records = []
    enrichment_log = []

    for ds in cfg["datasets"]:
        geo = _safe_geometry(ds, cfg)
        if "error" in geo:
            enrichment_log.append({"dataset": ds, "status": "geometry_failed",
                                    "error": geo["error"]})
            continue
        try:
            df_assoc = _compute_gene_associations(ds, geo, cfg)
        except Exception as exc:
            enrichment_log.append({"dataset": ds, "status": "associations_failed",
                                    "error": str(exc)})
            continue
        background = list(df_assoc["gene"])
        for cls in GENE_CLASS_ORDER:
            cls_genes = df_assoc.loc[df_assoc["gene_class"] == cls, "gene"
                                      ].tolist()
            fallback_used = False
            if len(cls_genes) < 5:
                # threshold too strict -> fall back to top-ranked specificity
                fallback_used = True
                if cls == "theta_high_radial_low":
                    cls_genes = df_assoc.sort_values(
                        "theta_specificity", ascending=False
                    ).head(100)["gene"].tolist()
                elif cls == "radial_high_theta_low":
                    cls_genes = df_assoc.sort_values(
                        "radial_specificity", ascending=False
                    ).head(100)["gene"].tolist()
                else:  # both_high
                    composite = (df_assoc["rho_theta"].abs()
                                 + df_assoc["rho_radial"].abs())
                    cls_genes = df_assoc.assign(_c=composite).sort_values(
                        "_c", ascending=False).head(100)["gene"].tolist()
            cache_key = f"{ds}_{cls}_n{len(cls_genes)}"
            enr_raw = _run_enrichr(
                cls_genes, background, libraries, cache_key=cache_key)
            enrichment_log.append({
                "dataset": ds, "gene_class": cls,
                "n_genes_in_class": len(cls_genes),
                "fallback_used": fallback_used,
                "n_libraries": len(libraries),
                "n_terms_returned": int(0 if enr_raw is None else len(enr_raw)),
            })
            top_per_cat = _summarise_enrichment(enr_raw, keyword_map)
            for _, row in top_per_cat.iterrows():
                enrichment_records.append({
                    "dataset": ds, "gene_class": cls,
                    "category": row["category"],
                    "term": row["term"], "library": row["library"],
                    "adjusted_p": float(row["adjusted_p"]),
                    "minus_log10_fdr": float(row["minus_log10_fdr"]),
                    "overlap": row["overlap"],
                    "odds_ratio": row.get("odds_ratio"),
                    "combined_score": row.get("combined_score"),
                    "overlap_genes": row.get("overlap_genes"),
                    "n_genes_in_class": len(cls_genes),
                    "background_size": len(background),
                    "fallback_used": fallback_used,
                })
                heatmap_records.append({
                    "dataset": ds, "gene_class": cls,
                    "category": row["category"],
                    "minus_log10_fdr": float(row["minus_log10_fdr"]),
                    "term": row["term"], "library": row["library"],
                    "overlap": row["overlap"],
                    "fallback_used": fallback_used,
                })
            # Save raw enrichment per (ds, gene_class)
            full = enr_raw.copy() if enr_raw is not None and not enr_raw.empty \
                else pd.DataFrame()
            if not full.empty:
                full.insert(0, "gene_class", cls)
                full.insert(0, "dataset", ds)
            full.to_csv(
                panel_dir / f"Fig3D_{ds}_{cls}_enrichment.csv", index=False)

    enr_df = pd.DataFrame(enrichment_records)
    log_df = pd.DataFrame(enrichment_log)

    # ---- assemble heatmap matrix
    rows = []
    row_labels = []
    annotation_map = {}
    for ds in cfg["datasets"]:
        for cls in GENE_CLASS_ORDER:
            row_label = f"{DISPLAY.get(ds, ds)} · {GENE_CLASS_DISPLAY[cls]}"
            row_labels.append(row_label)
            row_vals = []
            for cat in CATEGORY_ORDER:
                hits = [r for r in heatmap_records
                        if r["dataset"] == ds and r["gene_class"] == cls
                        and r["category"] == cat]
                if not hits:
                    row_vals.append(np.nan)
                else:
                    best = max(hits, key=lambda r: r["minus_log10_fdr"])
                    row_vals.append(best["minus_log10_fdr"])
                    annotation_map[(row_label, cat)] = best["term"]
            rows.append(row_vals)
    matrix = np.asarray(rows, dtype=float)

    n_rows = matrix.shape[0]
    fig, ax = plt.subplots(
        figsize=(7.8, max(0.46 * n_rows + 2.3, 4.8)),
        facecolor="white")
    cap = 10.0
    cap_matrix = np.clip(matrix, 0.0, cap)

    # Softer perceptually-uniform sequential cmap (seaborn 'crest' if
    # available; otherwise fall back to a hand-built pale-to-deep teal).
    try:
        import seaborn as _sns  # noqa: F401
        cmap = plt.get_cmap("crest").copy()
    except Exception:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "soft_teal",
            ["#f7fbfb", "#dcebe9", "#a6cdd0", "#5b9aa8", "#2c5e6e", "#163a4a"])
    cmap.set_bad("#f5f5f5")
    masked = np.ma.masked_invalid(cap_matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=cap,
                   interpolation="nearest")

    # ultra-thin cell grid (white spacers)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.55, zorder=2)
    for j in range(len(CATEGORY_ORDER) + 1):
        ax.axvline(j - 0.5, color="white", lw=0.55, zorder=2)

    ax.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=22, ha="right",
                       fontsize=8.4, color="#333")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8.3, color="#333")
    for (i, j), v in np.ndenumerate(matrix):
        if np.isnan(v):
            ax.text(j, i, "n.s.", ha="center", va="center", fontsize=7.0,
                    color="#9a9a9a", zorder=3, style="italic")
            continue
        # Pick text color for legibility against the soft cmap
        rgba = cmap(v / cap)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        txt_color = "#111" if luminance > 0.55 else "white"
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                fontsize=7.6, zorder=3, color=txt_color, fontweight="medium")

    # subtle dataset separators (between adjacent datasets)
    n_classes = len(GENE_CLASS_ORDER)
    for i in range(n_classes, n_rows, n_classes):
        ax.axhline(i - 0.5, color="#9e9e9e", lw=0.55, zorder=4,
                   alpha=0.85)

    # axes styling — minimal
    for spine_name, spine in ax.spines.items():
        spine.set_color("#cfcfcf")
        spine.set_linewidth(0.45)
    ax.tick_params(axis="x", which="both", length=0, pad=3)
    ax.tick_params(axis="y", which="both", length=0, pad=3)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018, extend="max")
    cb.set_label("−log10(FDR)  (capped at 10)", fontsize=8.4, color="#333",
                 labelpad=6)
    cb.ax.tick_params(labelsize=7.8, width=0.4, length=2.0, color="#aaaaaa")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#cfcfcf")

    ax.set_title("Angular and radial coordinates enrich distinct gene programs",
                 fontsize=11.5, fontweight="bold", pad=24, loc="left",
                 color="#111")
    ax.text(0.0, 1.01,
            "Enrichment derived from GO BP, Hallmark, and Reactome only.",
            transform=ax.transAxes, fontsize=8, color="#777",
            ha="left", va="bottom", style="italic")
    fig.tight_layout(pad=0.7)

    # Save aggregate enrichment + library report
    enr_df.to_csv(panel_dir / "Fig3D_enrichment_table.csv", index=False)
    log_df.to_csv(panel_dir / "Fig3D_enrichment_log.csv", index=False)
    library_report = {
        "libraries": libraries,
        "source": "Enrichr (gseapy)",
        "background": "all HVGs tested per dataset",
        "method": "ORA via gseapy.enrichr; BH-FDR within each library",
        "category_keywords": keyword_map,
    }
    with open(panel_dir / "Fig3D_enrichment_sources.json", "w") as fh:
        json.dump(library_report, fh, indent=2)

    return {
        "panel_id": "Fig3D", "figure_n": 3, "figure": fig,
        "config": {**cfg, "library_report": library_report},
        "output_aliases": [{"dir": "fig3",
                            "stem": "Fig3D_coordinate_program_enrichment"}],
        "data": {
            "enrichment_top_per_category": enr_df,
            "enrichment_log": log_df,
        },
    }


# ---------------------------------------------------------------------------
# Supplemental — root sensitivity (saved under outputs/figures/supplement/...)
# ---------------------------------------------------------------------------

def panel_root_sensitivity(config: Optional[dict] = None,
                            out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config)
    rows = []
    for ds in cfg["datasets"]:
        geo = _safe_geometry(ds, cfg)
        if "error" in geo:
            rows.append({"dataset": ds, "anchor": None, "error": geo["error"]})
            continue
        unit_raw = geo["unit_raw"]
        cyto = np.asarray(geo["cytotrace"], dtype=float)
        pheno = geo["phenotype"]
        for cat in sorted(pd.Series(pheno).dropna().unique()):
            mask = pheno == cat
            if mask.sum() == 0:
                continue
            anchor = unit_raw[mask].mean(axis=0)
            anchor = anchor / max(np.linalg.norm(anchor), 1e-12)
            cosine = np.clip(unit_raw @ anchor, -1.0, 1.0)
            geo_dist = np.arccos(cosine)
            m = np.isfinite(cyto)
            if m.sum() < 5:
                continue
            rho, p = spearmanr(geo_dist[m], cyto[m])
            rows.append({
                "dataset": ds, "anchor": cat,
                "n_cells_in_anchor": int(mask.sum()),
                "rho_anchor_geodesic_cytotrace": float(rho),
                "p_value": float(p),
                "is_specified_root": cat == BIOLOGICAL_ROOTS[ds],
            })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cfg["datasets"]),
                              figsize=(3.2 * len(cfg["datasets"]), 3.6),
                              facecolor="white", sharey=False)
    if len(cfg["datasets"]) == 1:
        axes = [axes]
    for ax, ds in zip(axes, cfg["datasets"]):
        sub = df[df["dataset"] == ds].copy()
        if sub.empty or "rho_anchor_geodesic_cytotrace" not in sub.columns:
            ax.text(0.5, 0.5, ds, ha="center", va="center")
            ax.set_axis_off()
            continue
        sub = sub.dropna(subset=["rho_anchor_geodesic_cytotrace"])
        if sub.empty:
            ax.set_axis_off(); continue
        sub = sub.sort_values("rho_anchor_geodesic_cytotrace")
        colors = ["#e45756" if r else "#cccccc" for r in sub["is_specified_root"]]
        ax.barh(np.arange(len(sub)), sub["rho_anchor_geodesic_cytotrace"],
                color=colors, edgecolor="#222", linewidth=0.4)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["anchor"], fontsize=7.5)
        ax.axvline(0, color="#222", lw=0.5)
        ax.set_title(DISPLAY.get(ds, ds), fontsize=10)
        ax.set_xlabel(r"$\rho$(geodesic dist., CytoTRACE)", fontsize=8)
        ax.tick_params(width=0.6, length=2.5, labelsize=7.5)
        ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    fig.suptitle("Root-sensitivity diagnostic (specified root highlighted)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.4)
    return {
        "panel_id": "Fig3_supplement_root_sensitivity",
        "figure_n": 3, "figure": fig,
        "config": {**cfg, "note": "supplementary diagnostic; not a main panel"},
        "output_aliases": [{"dir": "supplement/root_sensitivity",
                            "stem": "Fig3_supplement_root_sensitivity"}],
        "data": {"root_sensitivity": df},
    }


# ---------------------------------------------------------------------------
# Supplemental — Regev cell-cycle overlap with coordinate-specific genes
# ---------------------------------------------------------------------------

def panel_regev_overlap(config: Optional[dict] = None,
                         out_root: Optional[Path] = None) -> dict:
    """Orthogonal validation that radial-/θ-/both-specific gene classes overlap
    canonical Regev/Tirosh cell-cycle genes (Fisher's exact, BH-FDR).

    This is supplementary; it complements (not replaces) the main Fig 3D
    enrichment heatmap. Background = all HVGs tested per dataset.
    """
    cfg = common.merge_config({**PANEL_DEFAULT, **PANEL_CD_DEFAULT}, config)
    out_dir = (common.ROOT / "outputs" / "figures" / "supplement"
               / "regev_cell_cycle_overlap")
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle = common.load_regev_cell_cycle_genes()
    regev_all = set(cycle["all"])
    regev_s = set(cycle["s_phase"])
    regev_g2m = set(cycle["g2m_phase"])

    from scipy.stats import fisher_exact

    rows = []
    for ds in cfg["datasets"]:
        geo = _safe_geometry(ds, cfg)
        if "error" in geo:
            rows.append({"dataset": ds, "error": geo["error"]})
            continue
        try:
            df_assoc = _compute_gene_associations(ds, geo, cfg)
        except Exception as exc:
            rows.append({"dataset": ds, "error": str(exc)})
            continue
        background = set(df_assoc["gene"])
        regev_in_bg = regev_all & background
        if len(regev_in_bg) < 5:
            rows.append({"dataset": ds, "n_regev_in_background": len(regev_in_bg),
                          "note": "<5 Regev genes in HVG background; skipped"})
            continue
        for cls in GENE_CLASS_ORDER:
            cls_genes = set(df_assoc.loc[df_assoc["gene_class"] == cls,
                                          "gene"])
            n_class = len(cls_genes)
            if n_class < 5:
                rows.append({
                    "dataset": ds, "gene_class": cls,
                    "n_genes_in_class": n_class,
                    "n_regev_in_background": len(regev_in_bg),
                    "n_overlap": len(cls_genes & regev_in_bg),
                    "n_overlap_s_phase": len(cls_genes & regev_s),
                    "n_overlap_g2m_phase": len(cls_genes & regev_g2m),
                    "odds_ratio": np.nan, "p_value": np.nan,
                    "fdr": np.nan,
                    "overlap_genes": ";".join(sorted(cls_genes & regev_in_bg)),
                    "note": "n_class<5; Fisher not run",
                })
                continue
            a = len(cls_genes & regev_in_bg)
            b = n_class - a
            c = len(regev_in_bg) - a
            d = len(background) - n_class - c
            try:
                odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            except Exception:
                odds, p = np.nan, np.nan
            rows.append({
                "dataset": ds, "gene_class": cls,
                "n_genes_in_class": n_class,
                "n_regev_in_background": len(regev_in_bg),
                "n_background": len(background),
                "n_overlap": a,
                "n_overlap_s_phase": len(cls_genes & regev_s),
                "n_overlap_g2m_phase": len(cls_genes & regev_g2m),
                "expected": float(n_class * len(regev_in_bg) / len(background)),
                "odds_ratio": float(odds),
                "p_value": float(p),
                "overlap_genes": ";".join(sorted(cls_genes & regev_in_bg)),
            })
    summary = pd.DataFrame(rows)

    # BH-FDR over rows where Fisher was run
    if "p_value" in summary.columns:
        mask = summary["p_value"].notna()
        if mask.any():
            summary.loc[mask, "fdr"] = _bh_fdr(
                summary.loc[mask, "p_value"].to_numpy())

    summary.to_csv(out_dir / "Fig3S_regev_overlap_summary.csv", index=False)

    # ---- compact horizontal-bar plot
    plot_df = summary.dropna(subset=["odds_ratio", "fdr"]).copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(7.4, 3), facecolor="white")
        ax.text(0.5, 0.5,
                "No (dataset, gene class) groups had enough genes for Fisher's "
                "test against the Regev/Tirosh cell-cycle gene list.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="#444")
        ax.set_axis_off()
    else:
        # Order: dataset (canonical) × gene class (canonical)
        plot_df["dataset_order"] = plot_df["dataset"].map(
            {d: i for i, d in enumerate(cfg["datasets"])})
        plot_df["class_order"] = plot_df["gene_class"].map(
            {c: i for i, c in enumerate(GENE_CLASS_ORDER)})
        plot_df = plot_df.sort_values(["dataset_order", "class_order"])
        labels = [
            f"{DISPLAY.get(r['dataset'], r['dataset'])} · "
            f"{GENE_CLASS_DISPLAY[r['gene_class']]}"
            for _, r in plot_df.iterrows()
        ]
        nl10 = -np.log10(np.clip(plot_df["fdr"].astype(float), 1e-300, 1.0))
        odds = plot_df["odds_ratio"].astype(float).to_numpy()
        ovl = plot_df["n_overlap"].astype(int).to_numpy()
        gene_classes = plot_df["gene_class"].tolist()
        fig, ax = plt.subplots(
            figsize=(7.6, 0.36 * len(plot_df) + 1.9),
            facecolor="white")
        positions = np.arange(len(plot_df))
        # Saturate radial-specific bars (key validation result for radial
        # axis); subtly fade θ-specific / both for visual hierarchy. The
        # actual colour is still the canonical class colour, so reading is
        # consistent with Fig 3C/D.
        alphas = [0.96 if c == "radial_high_theta_low" else 0.78
                  for c in gene_classes]
        colors = [GENE_CLASS_COLOR[c] for c in gene_classes]
        # bars with class-specific alpha
        for y, w, col, a in zip(positions, nl10, colors, alphas):
            ax.barh(y, w, color=col, edgecolor="#3a3a3a",
                    linewidth=0.35, alpha=a, height=0.74)
        ax.axvline(-np.log10(0.05), color="#bbbbbb", lw=0.55,
                    ls=(0, (3, 3)), label="FDR = 0.05", zorder=1)
        for y, val, OR, k in zip(positions, nl10, odds, ovl):
            if np.isfinite(OR) and OR > 0:
                txt = f"OR={OR:.1f}, k={k}"
            elif k == 0:
                txt = "k=0 (n.s.)"
            else:
                txt = f"k={k}"
            ax.text(val + 0.10, y, txt, va="center", ha="left",
                    fontsize=7.4, color="#444")
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.4, color="#333")
        ax.invert_yaxis()
        ax.set_xlabel("−log10(BH-adjusted Fisher p) for Regev cell-cycle overlap",
                      fontsize=9, color="#333", labelpad=5)
        ax.tick_params(axis="x", width=0.45, length=2.2, labelsize=8,
                       colors="#555")
        ax.tick_params(axis="y", which="both", length=0, pad=3)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(0.45)
            ax.spines[side].set_color("#888")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Dataset separator lines (subtle)
        n_classes = len(GENE_CLASS_ORDER)
        for i in range(n_classes, len(plot_df), n_classes):
            ax.axhline(i - 0.5, color="#dcdcdc", lw=0.5)
        ax.set_xlim(0, max(nl10.max() * 1.22, -np.log10(0.05) + 0.8))
        ax.legend(loc="lower right", frameon=False, fontsize=7.8)
        ax.set_title(
            "Coordinate-specific gene classes vs. canonical Regev cell-cycle markers",
            fontsize=11, fontweight="bold", pad=22, loc="left", color="#111")
        ax.text(0.0, 1.015,
                "Supplementary validation: Fisher's exact test (alt = greater); "
                "background = HVGs tested per dataset.",
                transform=ax.transAxes, fontsize=7.8, color="#777",
                ha="left", va="bottom", style="italic")
    fig.tight_layout(pad=0.6)

    return {
        "panel_id": "Fig3S_regev_overlap",
        "figure_n": 3, "figure": fig,
        "config": {**cfg, "regev_n_genes": len(regev_all),
                    "regev_source": cycle.get("source"),
                    "test": "Fisher's exact, alternative='greater'",
                    "background": "HVGs tested per dataset",
                    "fdr": "BH across all (dataset, gene_class) tests"},
        "output_aliases": [{"dir": "supplement/regev_cell_cycle_overlap",
                            "stem": "Fig3S_regev_overlap"}],
        "data": {"regev_overlap_summary": summary},
    }


PANELS = {
    "A": panel_A,
    "B": panel_B,
    "C": panel_C,
    "D": panel_D,
    "supplement_root_sensitivity": panel_root_sensitivity,
    "supplement_regev_overlap": panel_regev_overlap,
}
