"""Figure 4 — mechanistic gene interpretation of spherical PCA coordinates.

Scientific message
------------------
PC1-3 → S² coordinates (theta, phi, radial-norm) decompose into three
biologically meaningful axes. A fixed-loading sensitivity analysis
(z' = z + delta * loading_g) lets us classify each highly-variable gene
by *where on the sphere* its activity moves cells:

    P_g (signed Δθ)     — progression / counter-progression along the
                          root → tip developmental geodesic.
    B_g (|Δφ|)          — branch / angular position around the trajectory.
    R_g (signed Δlog r) — radial / biosynthetic activity (PC magnitude).

Datasets (main)
    hESC (Chu et al. 2016)             — root: hESC.
    Klein mESC (Klein et al. 2015)     — root: Day 0.
    Human germline (Li et al. 2017)    — root: 19W.

Panels
    A. Conceptual perturbation decomposition (hESC, three example genes).
    B. Mechanistic gene map (hESC scatter, P × |B|, color = R).
    C. Cross-species rankings (Klein, germline).
    D. Gene-set enrichment heatmap (Hallmark / GO BP / Reactome).

Supplement
    S_BrCa: BrCa atlas — three S² camera angles, no overinterpretation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .. import io as psp_io
from . import common


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_FONT = {
    "family": "DejaVu Sans",
    "title": 11.5,
    "label": 9.5,
    "tick": 8.0,
    "legend": 8.0,
}

_COLORS = {
    "progression": "#d1495b",   # warm red
    "counter":     "#2e86ab",   # cool blue
    "branch":      "#7c5295",   # purple
    "radial":      "#f0a035",   # gold
    "background":  "#dadbde",
    "axis":        "#3c3c3c",
    "trajectory":  "#fffbe6",
    "sphere":      "#1d2230",
    "wireframe":   "#454c5b",
    "rad_pos":     "#d97706",
    "rad_neg":     "#0f766e",
    "label_text":  "#1c1c1c",
}


def _apply_panel_style():
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Dataset definitions and root selectors
# ---------------------------------------------------------------------------

@dataclass
class _DatasetSpec:
    key: str
    display: str
    species: str
    root_label: str
    root_descr: str


_MAIN_DATASETS: Dict[str, _DatasetSpec] = {
    "hesc": _DatasetSpec("hesc", "Human ESC (Chu et al.)",
                          "human", "hESC", "Phenotype = hESC"),
    "klein": _DatasetSpec("klein", "Mouse ESC (Klein et al.)",
                           "mouse", "Day0", "day == Day0"),
    "human_germ_cell": _DatasetSpec("human_germ_cell",
                                     "Human germ cell (Li et al.)",
                                     "human", "19W", "Phenotype = 19W"),
}


def _root_mask_from_metadata(ds: str, expr) -> np.ndarray:
    """Return a boolean mask over expr.cell_ids selecting root cells."""
    md = expr.metadata
    n = expr.n_cells
    if ds == "hesc":
        if "Phenotype" in md.columns:
            return md["Phenotype"].astype(str).values == "hESC"
        return np.zeros(n, dtype=bool)
    if ds == "klein":
        if "day" in md.columns:
            return md["day"].astype(str).values == "Day0"
        return np.zeros(n, dtype=bool)
    if ds == "human_germ_cell":
        if "Phenotype" in md.columns:
            return md["Phenotype"].astype(str).values == "19W"
        return np.zeros(n, dtype=bool)
    return np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# Cached PCA-with-loadings computation
# ---------------------------------------------------------------------------

def _state_cache_path(ds: str) -> Path:
    cache = common.ROOT / "outputs" / "figures" / "Fig4" / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{ds}_perturbation_state_seed0.npz"


def _compute_dataset_state(ds: str, *, seed: int = 0,
                           force_recompute: bool = False) -> dict:
    """Run the canonical Scanpy pipeline for `ds` and return a state dict
    containing PCA scores, loadings, root-aligned spherical coordinates,
    and per-cell auxiliary fields.

    Pipeline (matches the spec):
        normalize_total(target_sum=1e4) -> log1p ->
        highly_variable_genes(flavor='seurat', n_top_genes=2000) ->
        scale(max_value=10) -> tl.pca(n_comps=20)

    Then PC1-3 are L2-normalised onto S^2 and rotated so that the root
    centroid maps to the north pole.
    """
    cache_path = _state_cache_path(ds)
    if cache_path.exists() and not force_recompute:
        z = np.load(cache_path, allow_pickle=True)
        return {
            "ds": ds,
            "scores": np.asarray(z["scores"], dtype=np.float32),
            "loadings_pc3": np.asarray(z["loadings_pc3"], dtype=np.float32),
            "explained_variance_ratio":
                np.asarray(z["explained_variance_ratio"], dtype=np.float32),
            "hvg_genes": [str(g) for g in z["hvg_genes"]],
            "root_mask": np.asarray(z["root_mask"], dtype=bool),
            "rotation": np.asarray(z["rotation"], dtype=np.float64),
            "unit_orig": np.asarray(z["unit_orig"], dtype=np.float32),
            "rot_orig": np.asarray(z["rot_orig"], dtype=np.float32),
            "theta_orig": np.asarray(z["theta_orig"], dtype=np.float32),
            "phi_orig": np.asarray(z["phi_orig"], dtype=np.float32),
            "r_orig": np.asarray(z["r_orig"], dtype=np.float32),
            "cell_ids": [str(c) for c in z["cell_ids"]],
            "phenotype": [str(c) for c in z["phenotype"]],
        }

    import scanpy as sc
    import anndata as ad

    sc.settings.verbosity = 0
    if ds == "hesc":
        expr = psp_io.load_hesc(str(common.ROOT / "raw_data" / "hESC"))
    elif ds == "klein":
        expr = psp_io.load_bz2_klein_dataset(
            str(common.ROOT / "raw_data" / "klein"))
    elif ds == "human_germ_cell":
        expr = psp_io.load_cytotrace_rds_dataset(
            str(common.ROOT / "raw_data" / "human_germ_cell"))
    else:
        raise ValueError(f"unknown dataset {ds!r}")

    adata = ad.AnnData(
        X=expr.X,
        obs=pd.DataFrame(index=list(expr.cell_ids)),
        var=pd.DataFrame(index=list(expr.gene_names)),
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat",
        n_top_genes=min(2000, adata.n_vars - 1),
        inplace=True,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=20, random_state=seed, zero_center=True)

    scores = np.asarray(adata.obsm["X_pca"], dtype=np.float32)  # (N, 20)
    pcs_full = np.asarray(adata.varm["PCs"], dtype=np.float32)   # (n_var, 20)
    loadings_pc3 = pcs_full[:, :3].T.copy()                      # (3, n_var)
    evr = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float32)
    hvg_genes = list(adata.var_names)

    root_mask = _root_mask_from_metadata(ds, expr)
    if root_mask.sum() == 0:
        raise RuntimeError(f"root cells not found for {ds!r}")

    z3 = scores[:, :3]
    norms = np.linalg.norm(z3, axis=1)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    unit = z3 / norms_safe[:, None]

    centroid = z3[root_mask].mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn < 1e-12:
        raise RuntimeError(f"root centroid has zero magnitude for {ds!r}")
    centroid_unit = centroid / cn
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(centroid_unit, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        R = np.eye(3, dtype=np.float64)
    else:
        axis = axis / axis_norm
        cos_a = float(np.clip(np.dot(centroid_unit, target), -1.0, 1.0))
        ang = np.arccos(cos_a)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    rot_orig = (R @ unit.T).T.astype(np.float32)
    theta_orig = np.arccos(np.clip(rot_orig[:, 2], -1.0, 1.0)).astype(np.float32)
    phi_orig = np.arctan2(rot_orig[:, 1], rot_orig[:, 0]).astype(np.float32)
    r_orig = norms.astype(np.float32)

    if "Phenotype" in expr.metadata.columns:
        phenotype = expr.metadata["Phenotype"].astype(str).tolist()
    elif "day" in expr.metadata.columns:
        phenotype = expr.metadata["day"].astype(str).tolist()
    else:
        phenotype = [""] * scores.shape[0]

    np.savez(
        cache_path,
        scores=scores,
        loadings_pc3=loadings_pc3,
        explained_variance_ratio=evr,
        hvg_genes=np.asarray(hvg_genes),
        root_mask=root_mask,
        rotation=R,
        unit_orig=unit.astype(np.float32),
        rot_orig=rot_orig,
        theta_orig=theta_orig,
        phi_orig=phi_orig,
        r_orig=r_orig,
        cell_ids=np.asarray(list(expr.cell_ids)),
        phenotype=np.asarray(phenotype),
    )
    return {
        "ds": ds,
        "scores": scores,
        "loadings_pc3": loadings_pc3,
        "explained_variance_ratio": evr,
        "hvg_genes": hvg_genes,
        "root_mask": root_mask,
        "rotation": R,
        "unit_orig": unit.astype(np.float32),
        "rot_orig": rot_orig,
        "theta_orig": theta_orig,
        "phi_orig": phi_orig,
        "r_orig": r_orig,
        "cell_ids": list(expr.cell_ids),
        "phenotype": phenotype,
    }


# ---------------------------------------------------------------------------
# Per-gene perturbation scoring
# ---------------------------------------------------------------------------

def _per_gene_perturbation(state: dict, *, delta: float = 1.0) -> pd.DataFrame:
    """Compute (P_g, B_g, R_g) for every HVG.

    Definitions (after rotating to root-aligned PC1-3 frame):
        P_g = median_n( theta_perturb - theta_orig )            [signed]
        B_g = median_n( |delta_phi_wrapped| )                    [unsigned, ≥0]
        R_g = median_n( log2(r_perturb / r_orig) )               [signed]

    All medians are computed in radians for theta/phi and log-ratio
    units for r. Returned scores are in degrees for theta and phi, and in
    log2-ratio for r, to make units comparable across datasets.
    """
    z3 = state["scores"][:, :3].astype(np.float64)         # (N, 3)
    L = state["loadings_pc3"].astype(np.float64)           # (3, G)
    R = state["rotation"].astype(np.float64)
    G = L.shape[1]
    N = z3.shape[0]
    genes = state["hvg_genes"]

    theta_o = state["theta_orig"].astype(np.float64)
    phi_o = state["phi_orig"].astype(np.float64)
    r_o = state["r_orig"].astype(np.float64)

    # Pre-rotate the loading vectors so we can compute the perturbed unit
    # vector directly in the root-aligned frame for each gene.
    rot_z = (R @ z3.T).T                                   # (N, 3)
    rot_L = R @ L                                          # (3, G)

    # Compute scores per gene.  Vectorise per-cell over a chunk of genes to
    # cap memory.
    chunk = 256
    P_med = np.empty(G, dtype=np.float64)
    B_med = np.empty(G, dtype=np.float64)
    R_med = np.empty(G, dtype=np.float64)
    P_25 = np.empty(G, dtype=np.float64)
    P_75 = np.empty(G, dtype=np.float64)
    R_signed_dir = np.empty(G, dtype=np.float64)  # sign of phi shift dominance

    for g0 in range(0, G, chunk):
        g1 = min(g0 + chunk, G)
        Lc = rot_L[:, g0:g1]                              # (3, gc)
        # Perturbed PC scores in rotated frame: (N, gc, 3)
        zc = rot_z[:, None, :] + delta * Lc.T[None, :, :]
        norms = np.linalg.norm(zc, axis=2)                # (N, gc)
        norms_safe = np.where(norms < 1e-12, 1.0, norms)
        units = zc / norms_safe[..., None]                # (N, gc, 3)
        theta_p = np.arccos(np.clip(units[..., 2], -1.0, 1.0))    # (N, gc)
        phi_p = np.arctan2(units[..., 1], units[..., 0])          # (N, gc)
        d_theta = theta_p - theta_o[:, None]
        d_phi = phi_p - phi_o[:, None]
        # wrap to [-pi, pi]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        d_logr = np.log2(norms / np.maximum(r_o[:, None], 1e-9))

        P_med[g0:g1] = np.median(d_theta, axis=0)
        P_25[g0:g1] = np.percentile(d_theta, 25, axis=0)
        P_75[g0:g1] = np.percentile(d_theta, 75, axis=0)
        B_med[g0:g1] = np.median(np.abs(d_phi), axis=0)
        R_med[g0:g1] = np.median(d_logr, axis=0)
        R_signed_dir[g0:g1] = np.sign(np.mean(d_phi, axis=0))

    # convert to degrees (theta, phi); R stays in log2-units
    deg = 180.0 / np.pi
    df = pd.DataFrame({
        "gene": genes,
        "progression_score": P_med * deg,        # signed deg
        "progression_iqr_low": P_25 * deg,
        "progression_iqr_high": P_75 * deg,
        "branch_position_score": B_med * deg,    # unsigned deg
        "branch_signed_direction": R_signed_dir,
        "radial_score": R_med,                   # log2 ratio
        "loading_pc1": state["loadings_pc3"][0, :],
        "loading_pc2": state["loadings_pc3"][1, :],
        "loading_pc3": state["loadings_pc3"][2, :],
        "loading_norm": np.linalg.norm(state["loadings_pc3"], axis=0),
    })
    df["abs_progression"] = df["progression_score"].abs()
    df = df.sort_values("abs_progression", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Helpers shared by panels
# ---------------------------------------------------------------------------

def _draw_3d_sphere_backdrop(ax, *, color="#e8edf2", wire="#9aa9b4",
                              alpha_surface=0.10, alpha_wire=0.18):
    u, v = np.mgrid[0:2 * np.pi:64j, 0:np.pi:32j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha_surface,
                    linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color=wire, alpha=alpha_wire,
                      linewidth=0.25)


def _style_clean_3d(ax):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()


# Backwards-compat alias used by the supplement panel.
_style_dark_3d = _style_clean_3d


def _great_circle_through(start: np.ndarray, end: np.ndarray,
                          n: int = 200) -> np.ndarray:
    s = start / np.linalg.norm(start)
    e = end / np.linalg.norm(end)
    cos_a = float(np.clip(np.dot(s, e), -1.0, 1.0))
    ang = np.arccos(cos_a)
    if ang < 1e-9:
        return np.tile(s, (n, 1))
    perp = e - cos_a * s
    perp /= np.linalg.norm(perp)
    t = np.linspace(0, ang, n)
    return np.outer(np.cos(t), s) + np.outer(np.sin(t), perp)


# ---------------------------------------------------------------------------
# Panel 4A — conceptual decomposition
# ---------------------------------------------------------------------------

PANEL_A_DEFAULT = {
    "dataset": "hesc",
    "delta": 1.0,
    # Visualization-only amplification of the perturbation arrow length on the
    # 3D sphere.  Scoring statistics still use `delta`; this simply scales the
    # rendered arrow length so a +1 SD perturbation is visible.
    "arrow_visual_amplification": 30.0,
    "n_arrows_per_gene": 26,
    "seed": 0,
}


def _select_concept_genes(score_df: pd.DataFrame) -> dict:
    """Pick three illustrative genes from a per-gene score table."""
    df = score_df.copy()
    # use 80th-percentile loading-norm to filter very weakly-loaded genes
    th = np.percentile(df["loading_norm"], 30)
    df = df[df["loading_norm"] >= th]
    progression = df.sort_values("progression_score", ascending=False).iloc[0]
    counter = df.sort_values("progression_score", ascending=True).iloc[0]
    branch_pool = df[df["progression_score"].abs() <
                     df["progression_score"].abs().median()]
    branch = branch_pool.sort_values("branch_position_score",
                                     ascending=False).iloc[0]
    return {
        "progression": progression,
        "counter": counter,
        "branch": branch,
    }


def panel_A(config: Optional[dict] = None,
            out_root: Optional[Path] = None) -> dict:
    _apply_panel_style()
    cfg = common.merge_config(PANEL_A_DEFAULT, config)
    ds = cfg["dataset"]
    state = _compute_dataset_state(ds)
    score_df = _per_gene_perturbation(state, delta=float(cfg["delta"]))
    chosen = _select_concept_genes(score_df)
    cfg["chosen_genes"] = {k: str(v["gene"]) for k, v in chosen.items()}

    fig = plt.figure(figsize=(11.6, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 0.95],
                          left=0.03, right=0.985, top=0.86, bottom=0.10,
                          wspace=0.32)

    # ---- left: 3D sphere with three perturbation tracks
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    _draw_3d_sphere_backdrop(ax3)
    rng = np.random.default_rng(cfg["seed"])
    rot = state["rot_orig"]
    n = rot.shape[0]
    cells = rng.choice(n, size=min(900, n), replace=False)
    theta_vals = state["theta_orig"][cells]
    cell_cmap = mpl.cm.get_cmap("viridis")
    tnorm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
    cell_colors = cell_cmap(tnorm(theta_vals))
    cell_colors[:, 3] = 0.55
    ax3.scatter(rot[cells, 0], rot[cells, 1], rot[cells, 2],
                s=4.5, c=cell_colors, linewidths=0, depthshade=False)

    z3 = state["scores"][:, :3].astype(np.float64)
    R = state["rotation"]
    L = state["loadings_pc3"].astype(np.float64)
    genes_in = list(state["hvg_genes"])

    pole = np.array([0.0, 0.0, 1.0])
    arrows_data = []
    legend_items = []
    track_palette = [
        ("progression", _COLORS["progression"], "→ progression"),
        ("counter", _COLORS["counter"], "← counter"),
        ("branch", _COLORS["branch"], "↔ branch"),
    ]
    arrow_rng = np.random.default_rng(cfg["seed"] + 1)
    n_arrows = int(cfg["n_arrows_per_gene"])
    amp = float(cfg["arrow_visual_amplification"])
    for kind, color, label in track_palette:
        gname = chosen[kind]["gene"]
        gi = genes_in.index(str(gname))
        sel = arrow_rng.choice(n, size=min(n_arrows, n), replace=False)
        z_sel = z3[sel]
        # Visualization-only amplification: scale the perturbation magnitude
        # by `amp` so that a +1 SD displacement is rendered legibly on the S^2
        # sphere.  Per-gene scores in the table still use the unamplified
        # delta value.
        z_pert_vis = z_sel + (cfg["delta"] * amp) * L[:, gi]
        u_orig = z_sel / np.linalg.norm(z_sel, axis=1, keepdims=True)
        u_pert = z_pert_vis / np.linalg.norm(z_pert_vis, axis=1, keepdims=True)
        ru_orig = (R @ u_orig.T).T
        ru_pert = (R @ u_pert.T).T
        for i in range(len(sel)):
            ax3.plot([ru_orig[i, 0], ru_pert[i, 0]],
                     [ru_orig[i, 1], ru_pert[i, 1]],
                     [ru_orig[i, 2], ru_pert[i, 2]],
                     color=color, lw=1.0, alpha=0.85)
            ax3.scatter([ru_pert[i, 0]], [ru_pert[i, 1]], [ru_pert[i, 2]],
                        s=11, color=color, alpha=0.95, linewidths=0,
                        depthshade=False)
        legend_items.append(Line2D([0], [0], color=color, lw=2.2,
                                   label=f"{label}: {gname}"))
        arrows_data.append({
            "kind": kind, "gene": str(gname),
            "P_deg": float(chosen[kind]["progression_score"]),
            "B_deg": float(chosen[kind]["branch_position_score"]),
            "R_log2": float(chosen[kind]["radial_score"]),
        })

    # mark north pole (root) and indicate developmental progression
    ax3.scatter([0], [0], [1], s=120, marker="*", color="#f5b942",
                edgecolor="#1d2230", linewidths=0.5, depthshade=False, zorder=10)
    ax3.text(0, 0, 1.20, "root", color="#1d2230",
             ha="center", fontsize=8.5, weight="bold")
    ax3.view_init(elev=14, azim=-60)
    _style_clean_3d(ax3)
    lg = ax3.legend(handles=legend_items, loc="lower left",
                    bbox_to_anchor=(-0.05, 0.05),
                    frameon=False, fontsize=_FONT["legend"] - 0.5,
                    handlelength=1.6)
    for t in lg.get_texts():
        t.set_color("#1d2230")

    # ---- middle: signed Δθ, Δφ, Δlog2 r decomposition (per-cell distributions)
    ax2 = fig.add_subplot(gs[0, 1])
    bp_data = []
    bp_colors = []
    bp_labels = []
    for kind, color, _ in track_palette:
        g = str(chosen[kind]["gene"])
        bp_data.append(_per_cell_components(state, g, cfg["delta"], "theta"))
        bp_colors.append(color)
        bp_labels.append(g)
    pos = np.array([0.5, 1.5, 2.5])
    bp = ax2.boxplot(bp_data, positions=pos, widths=0.55, patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color="white", lw=1.4),
                     whiskerprops=dict(color="#3c3c3c", lw=0.7),
                     capprops=dict(color="#3c3c3c", lw=0.7),
                     boxprops=dict(linewidth=0.0))
    for patch, c in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax2.axhline(0, color="#777", lw=0.6, linestyle="--")
    ax2.set_xticks(pos)
    ax2.set_xticklabels(bp_labels, rotation=20, ha="right",
                        fontsize=_FONT["tick"] - 0.2)
    ax2.set_ylabel(r"per-cell $\Delta\theta$ (deg)")
    ax2.set_title("Progression component", pad=4, fontsize=_FONT["title"])
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    # ---- right: |Δφ| and Δlog2 r side by side as compact bars (medians)
    ax4 = fig.add_subplot(gs[0, 2])
    ax4b = ax4.twinx()
    ax4.spines["top"].set_visible(False)
    ax4b.spines["top"].set_visible(False)
    bw = 0.32
    xs = np.arange(3)
    bvals = [float(chosen[k]["branch_position_score"]) for k, _, _ in track_palette]
    rvals = [float(chosen[k]["radial_score"]) for k, _, _ in track_palette]
    ax4.bar(xs - bw / 2, bvals, width=bw, color=_COLORS["branch"],
            alpha=0.85, edgecolor="white", linewidth=0.6, label=r"$|\Delta\varphi|$")
    ax4b.bar(xs + bw / 2, rvals, width=bw,
             color=[_COLORS["rad_pos"] if v >= 0 else _COLORS["rad_neg"]
                    for v in rvals],
             alpha=0.85, edgecolor="white", linewidth=0.6,
             label=r"$\Delta\log_{2} r$")
    ax4.set_xticks(xs)
    ax4.set_xticklabels(bp_labels, rotation=20, ha="right",
                        fontsize=_FONT["tick"] - 0.2)
    ax4.set_ylabel(r"$|\Delta\varphi|$ (deg)", color=_COLORS["branch"])
    ax4b.set_ylabel(r"$\Delta\log_{2} r$",
                    color=_COLORS["rad_pos"])
    ax4.tick_params(axis="y", colors=_COLORS["branch"])
    ax4b.tick_params(axis="y", colors=_COLORS["rad_pos"])
    ax4.set_title("Branch & radial components",
                  pad=4, fontsize=_FONT["title"])
    ax4b.axhline(0, color="#bbb", lw=0.5, linestyle=":")

    fig.text(0.012, 0.945,
             "A   Signed gene perturbations decompose movement on the sphere",
             fontsize=_FONT["title"] + 1.5, weight="bold", color="#1d2230")
    fig.text(0.012, 0.905,
             f"hESC; root = hESC; perturbation = +{cfg['delta']:.0f} SD; "
             f"3D arrows visually amplified ×{int(cfg['arrow_visual_amplification'])}",
             fontsize=_FONT["tick"], color="#5a5e6b")

    # data tables
    arrows_df = pd.DataFrame(arrows_data)
    return {
        "panel_id": "Fig4A",
        "figure_n": 4,
        "figure": fig,
        "config": cfg,
        "data": {
            "concept_genes": arrows_df,
            "per_gene_scores": score_df.head(40).copy(),
        },
    }


def _per_cell_components(state: dict, gene: str, delta: float,
                         which: str) -> np.ndarray:
    z3 = state["scores"][:, :3].astype(np.float64)
    R = state["rotation"].astype(np.float64)
    L = state["loadings_pc3"].astype(np.float64)
    gi = list(state["hvg_genes"]).index(gene)
    z_pert = z3 + delta * L[:, gi]
    u_orig = z3 / np.linalg.norm(z3, axis=1, keepdims=True)
    u_pert = z_pert / np.linalg.norm(z_pert, axis=1, keepdims=True)
    ru_o = (R @ u_orig.T).T
    ru_p = (R @ u_pert.T).T
    if which == "theta":
        th_o = np.arccos(np.clip(ru_o[:, 2], -1.0, 1.0))
        th_p = np.arccos(np.clip(ru_p[:, 2], -1.0, 1.0))
        return np.degrees(th_p - th_o)
    if which == "phi":
        p_o = np.arctan2(ru_o[:, 1], ru_o[:, 0])
        p_p = np.arctan2(ru_p[:, 1], ru_p[:, 0])
        d = (p_p - p_o + np.pi) % (2 * np.pi) - np.pi
        return np.degrees(np.abs(d))
    if which == "r":
        return np.log2(np.linalg.norm(z_pert, axis=1) /
                       np.linalg.norm(z3, axis=1))
    raise ValueError(which)


# ---------------------------------------------------------------------------
# Panel 4B — hESC mechanistic gene scatter
# ---------------------------------------------------------------------------

PANEL_B_DEFAULT = {
    "dataset": "hesc",
    "delta": 1.0,
    "n_label_progression_pos": 6,
    "n_label_progression_neg": 6,
    "n_label_branch": 6,
    "n_label_radial": 4,
    "min_loading_norm_quantile": 0.30,
    "seed": 0,
}


def panel_B(config: Optional[dict] = None,
            out_root: Optional[Path] = None) -> dict:
    _apply_panel_style()
    cfg = common.merge_config(PANEL_B_DEFAULT, config)
    state = _compute_dataset_state(cfg["dataset"])
    df = _per_gene_perturbation(state, delta=float(cfg["delta"]))

    # filter weakly-loaded genes for visualisation but keep all rows in CSV
    thr = float(np.quantile(df["loading_norm"],
                            float(cfg["min_loading_norm_quantile"])))
    plot_df = df[df["loading_norm"] >= thr].copy()

    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    # background scatter
    rmax = float(np.max(np.abs(plot_df["radial_score"])))
    norm = TwoSlopeNorm(vmin=-rmax, vcenter=0.0, vmax=rmax)
    sc = ax.scatter(
        plot_df["progression_score"], plot_df["branch_position_score"],
        c=plot_df["radial_score"], cmap="RdBu_r", norm=norm,
        s=10 + 35 * (plot_df["loading_norm"] / plot_df["loading_norm"].max()),
        alpha=0.45, edgecolor="white", linewidths=0.18,
    )

    # axis cosmetics
    pmax = float(np.max(np.abs(plot_df["progression_score"])))
    ax.axvline(0, color="#666", lw=0.6, linestyle="--", alpha=0.7)
    ax.axhline(0, color="#666", lw=0.6, linestyle="--", alpha=0.7)
    ax.set_xlim(-1.08 * pmax, 1.08 * pmax)
    bmax = float(np.quantile(plot_df["branch_position_score"], 0.99))
    ax.set_ylim(0, max(bmax * 1.10, 0.5))
    ax.set_xlabel(r"Progression score $P_g$  =  median $\Delta\theta$  (deg)")
    ax.set_ylabel(r"Branch-position score $B_g$  =  median $|\Delta\varphi|$  (deg)")
    ax.set_title(
        f"B   Mechanistic gene map — {_MAIN_DATASETS[cfg['dataset']].display}",
        loc="left", pad=8,
        fontsize=_FONT["title"] + 1.0, weight="bold", color="#1d2230",
    )
    ax.text(0.99, -0.10,
            "color: radial score $R_g$ (Δlog₂ r)   •   size: |loading| on PC1-3",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=_FONT["tick"], color="#555")

    # quadrant labels (minimal)
    bbox_kw = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
    ax.text(0.97, 0.97, "branch ↔ position",
            transform=ax.transAxes, ha="right", va="top",
            color="#555", fontsize=8.5, bbox=bbox_kw)
    ax.text(0.97, 0.04, "progression-promoting",
            transform=ax.transAxes, ha="right", va="bottom",
            color=_COLORS["progression"], fontsize=8.5, bbox=bbox_kw)
    ax.text(0.03, 0.04, "counter-progressing",
            transform=ax.transAxes, ha="left", va="bottom",
            color=_COLORS["counter"], fontsize=8.5, bbox=bbox_kw)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, shrink=0.65)
    cbar.set_label(r"Radial $R_g$ (Δlog₂ r)", fontsize=_FONT["label"])
    cbar.outline.set_linewidth(0.4)

    # ---- pick top genes for labelling (pure rank-based)
    pos = (plot_df.sort_values("progression_score", ascending=False)
                .head(int(cfg["n_label_progression_pos"]))["gene"].tolist())
    neg = (plot_df.sort_values("progression_score", ascending=True)
                .head(int(cfg["n_label_progression_neg"]))["gene"].tolist())
    midpool = plot_df[plot_df["progression_score"].abs()
                      < plot_df["progression_score"].abs().median()]
    branch = (midpool.sort_values("branch_position_score", ascending=False)
              .head(int(cfg["n_label_branch"]))["gene"].tolist())
    radial = (plot_df.assign(absR=plot_df["radial_score"].abs())
              .sort_values("absR", ascending=False)
              .head(int(cfg["n_label_radial"]))["gene"].tolist())

    label_genes = pos + neg + branch + radial
    label_genes = list(dict.fromkeys(label_genes))
    label_records = []
    text_handles = []
    for g in label_genes:
        row = plot_df[plot_df["gene"] == g].iloc[0]
        category = (
            "progression" if g in pos
            else "counter" if g in neg
            else "branch" if g in branch
            else "radial"
        )
        c = {
            "progression": _COLORS["progression"],
            "counter": _COLORS["counter"],
            "branch": _COLORS["branch"],
            "radial": _COLORS["axis"],
        }[category]
        ax.scatter(row["progression_score"], row["branch_position_score"],
                   s=22, edgecolor=c, facecolor="none", linewidth=0.9,
                   zorder=4)
        t = ax.text(row["progression_score"], row["branch_position_score"], g,
                    fontsize=_FONT["tick"] + 0.2, color=c,
                    weight="bold", zorder=5)
        text_handles.append(t)
        label_records.append({"gene": g, "category": category,
                              "progression_score": float(row["progression_score"]),
                              "branch_position_score": float(row["branch_position_score"]),
                              "radial_score": float(row["radial_score"])})

    try:
        from adjustText import adjust_text
        adjust_text(text_handles, ax=ax,
                    expand_points=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#777", lw=0.4,
                                    alpha=0.7))
    except Exception:
        pass

    fig.tight_layout()
    return {
        "panel_id": "Fig4B",
        "figure_n": 4,
        "figure": fig,
        "config": cfg,
        "data": {
            "per_gene_scores": df,
            "labelled_genes": pd.DataFrame(label_records),
        },
    }


# ---------------------------------------------------------------------------
# Panel 4C — cross-species rankings (Klein, germline)
# ---------------------------------------------------------------------------

PANEL_C_DEFAULT = {
    "datasets": ["klein", "human_germ_cell"],
    "delta": 1.0,
    "top_n": 12,
    "min_loading_norm_quantile": 0.30,
    "seed": 0,
}


def _ranked_lollipop(ax, df: pd.DataFrame, value_col: str, *,
                     color: str, title: str, xlabel: str,
                     ascending: bool = False, accent_neg: bool = False):
    df = df.copy().reset_index(drop=True)
    y = np.arange(len(df))
    vals = df[value_col].values
    colors = [color] * len(df)
    if accent_neg:
        colors = [_COLORS["counter"] if v < 0 else _COLORS["progression"]
                  for v in vals]
    for yi, vi, ci in zip(y, vals, colors):
        ax.plot([0, vi], [yi, yi], color=ci, lw=1.4, alpha=0.85)
    ax.scatter(vals, y, color=colors, s=22, edgecolor="white",
               linewidths=0.4, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"].values, fontsize=_FONT["tick"] - 0.4)
    ax.invert_yaxis()
    ax.axvline(0, color="#aaa", lw=0.6, linestyle=":")
    ax.set_xlabel(xlabel, fontsize=_FONT["label"] - 0.2)
    ax.set_title(title, fontsize=_FONT["label"] + 0.4, pad=4)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def panel_C(config: Optional[dict] = None,
            out_root: Optional[Path] = None) -> dict:
    _apply_panel_style()
    cfg = common.merge_config(PANEL_C_DEFAULT, config)
    datasets = list(cfg["datasets"])
    top_n = int(cfg["top_n"])

    fig, axes = plt.subplots(len(datasets), 3, figsize=(11.4, 6.6),
                             gridspec_kw={"wspace": 0.55, "hspace": 0.85})
    if len(datasets) == 1:
        axes = axes[None, :]

    out_data: Dict[str, pd.DataFrame] = {}
    for i, ds in enumerate(datasets):
        state = _compute_dataset_state(ds)
        df_all = _per_gene_perturbation(state, delta=float(cfg["delta"]))
        thr = float(np.quantile(df_all["loading_norm"],
                                float(cfg["min_loading_norm_quantile"])))
        df = df_all[df_all["loading_norm"] >= thr].copy()
        out_data[f"per_gene_{ds}"] = df_all

        # Top progression-promoting (positive Δθ)
        prog = (df.sort_values("progression_score", ascending=False)
                  .head(top_n)[["gene", "progression_score"]])
        # Top counter-progressing (most negative Δθ)
        ctr = (df.sort_values("progression_score", ascending=True)
                .head(top_n)[["gene", "progression_score"]])
        # Top branch-position (high |Δφ|, small |Δθ|)
        midpool = df[df["progression_score"].abs()
                     < df["progression_score"].abs().median()]
        branch = (midpool.sort_values("branch_position_score", ascending=False)
                  .head(top_n)[["gene", "branch_position_score"]])

        _ranked_lollipop(
            axes[i, 0], prog, "progression_score",
            color=_COLORS["progression"],
            title=f"{_MAIN_DATASETS[ds].display}\nprogression-promoting (+Δθ)",
            xlabel=r"$P_g$  (deg)")
        _ranked_lollipop(
            axes[i, 1], ctr, "progression_score",
            color=_COLORS["counter"],
            title="counter-progressing (−Δθ)",
            xlabel=r"$P_g$  (deg)", accent_neg=True)
        _ranked_lollipop(
            axes[i, 2], branch, "branch_position_score",
            color=_COLORS["branch"],
            title=r"branch-position (high $|\Delta\varphi|$)",
            xlabel=r"$B_g$  (deg)")

        out_data[f"top_progression_{ds}"] = prog.reset_index(drop=True)
        out_data[f"top_counter_{ds}"] = ctr.reset_index(drop=True)
        out_data[f"top_branch_{ds}"] = branch.reset_index(drop=True)

    fig.suptitle(
        "C   Cross-species mechanistic rankings",
        x=0.04, y=0.995, ha="left", fontsize=_FONT["title"] + 1.0,
        weight="bold", color="#1d2230",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return {
        "panel_id": "Fig4C",
        "figure_n": 4,
        "figure": fig,
        "config": cfg,
        "data": out_data,
    }


# ---------------------------------------------------------------------------
# Panel 4D — gene-set enrichment heatmap
# ---------------------------------------------------------------------------

PANEL_D_DEFAULT = {
    "dataset": "hesc",
    "delta": 1.0,
    "categories": ["progression", "counter", "branch", "radial"],
    "libraries": ["MSigDB_Hallmark_2020", "Reactome_2022",
                  "GO_Biological_Process_2023"],
    "library_short": {
        "MSigDB_Hallmark_2020": "Hallmark",
        "Reactome_2022": "Reactome",
        "GO_Biological_Process_2023": "GO BP",
    },
    "top_n_per_category": 200,
    "min_loading_norm_quantile": 0.30,
    "max_terms_per_category": 6,
    "max_neglog10_fdr_cap": 10.0,
    "seed": 0,
}


def _select_category_genes(df: pd.DataFrame, category: str,
                            top_n: int) -> List[str]:
    df = df.copy()
    if category == "progression":
        return df.sort_values("progression_score", ascending=False
                              ).head(top_n)["gene"].astype(str).tolist()
    if category == "counter":
        return df.sort_values("progression_score", ascending=True
                              ).head(top_n)["gene"].astype(str).tolist()
    if category == "branch":
        midpool = df[df["progression_score"].abs() <
                     df["progression_score"].abs().median()]
        return midpool.sort_values("branch_position_score", ascending=False
                                   ).head(top_n)["gene"].astype(str).tolist()
    if category == "radial":
        return (df.assign(absR=df["radial_score"].abs())
                  .sort_values("absR", ascending=False)
                  .head(top_n)["gene"].astype(str).tolist())
    raise ValueError(category)


_LIBRARY_CACHE: Dict[str, Dict[str, list]] = {}


def _fetch_gene_set_library(name: str, organism: str = "human") -> Dict[str, list]:
    """Cached download of an Enrichr gene-set library as {term: [genes]}."""
    key = f"{name}::{organism}"
    if key in _LIBRARY_CACHE:
        return _LIBRARY_CACHE[key]
    cache_dir = (common.ROOT / "outputs" / "figures" / "Fig4" / "_cache" /
                 "gene_set_libraries")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{name}_{organism}.json"
    if cache_path.exists():
        with open(cache_path, "r") as fh:
            gs = json.load(fh)
        _LIBRARY_CACHE[key] = gs
        return gs
    import gseapy as gp
    gs = gp.get_library(name, organism=organism)
    with open(cache_path, "w") as fh:
        json.dump(gs, fh)
    _LIBRARY_CACHE[key] = gs
    return gs


def _hypergeometric_enrichment(gene_list: List[str],
                                background: List[str],
                                libraries: List[str],
                                *, organism: str = "human",
                                min_term_size: int = 5,
                                max_term_size: int = 1500
                                ) -> pd.DataFrame:
    """Run hypergeometric over-representation tests against Enrichr libraries.

    Genes are case-folded to uppercase to match library symbols.  FDR is
    computed via Benjamini–Hochberg across all (library, term) combinations
    so the same multiple-testing universe is used per category.
    """
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    bg = {str(g).upper() for g in background}
    query = {str(g).upper() for g in gene_list} & bg
    M = len(bg)
    n_query = len(query)
    if n_query == 0 or M == 0:
        return pd.DataFrame(columns=[
            "library", "term", "term_size_in_bg", "overlap_size",
            "n_query", "n_background", "pvalue", "fdr", "fold_enrichment",
            "overlap_genes",
        ])

    rows: list = []
    for lib in libraries:
        try:
            gs = _fetch_gene_set_library(lib, organism=organism)
        except Exception as exc:
            print(f"[Fig4D] could not load {lib}: {exc}")
            continue
        for term, term_genes in gs.items():
            term_set = {str(g).upper() for g in term_genes} & bg
            K = len(term_set)
            if K < min_term_size or K > max_term_size:
                continue
            overlap = term_set & query
            k = len(overlap)
            if k == 0:
                continue
            # P(X >= k) where X ~ Hypergeom(M, K, n_query)
            pvalue = float(hypergeom.sf(k - 1, M, K, n_query))
            expected = K * n_query / M
            fold = (k / expected) if expected > 0 else np.nan
            rows.append({
                "library": lib,
                "term": term,
                "term_size_in_bg": K,
                "overlap_size": k,
                "n_query": n_query,
                "n_background": M,
                "pvalue": pvalue,
                "fold_enrichment": fold,
                "overlap_genes": ";".join(sorted(overlap)),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # FDR per library so each library has its own multiple-testing universe.
    df["fdr"] = np.nan
    for lib in df["library"].unique():
        m = df["library"] == lib
        pvals = df.loc[m, "pvalue"].values
        if len(pvals):
            _, fdrs, _, _ = multipletests(pvals, method="fdr_bh")
            df.loc[m, "fdr"] = fdrs
    return df


def panel_D(config: Optional[dict] = None,
            out_root: Optional[Path] = None) -> dict:
    _apply_panel_style()
    cfg = common.merge_config(PANEL_D_DEFAULT, config)
    state = _compute_dataset_state(cfg["dataset"])
    species = _MAIN_DATASETS[cfg["dataset"]].species
    organism = "human" if species == "human" else "mouse"

    df_scores = _per_gene_perturbation(state, delta=float(cfg["delta"]))
    thr = float(np.quantile(df_scores["loading_norm"],
                            float(cfg["min_loading_norm_quantile"])))
    df = df_scores[df_scores["loading_norm"] >= thr].copy()
    background = df["gene"].astype(str).tolist()  # tested HVGs

    cache_dir = (common.ROOT / "outputs" / "figures" / "Fig4" / "_cache" /
                 f"enrichr_{cfg['dataset']}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_csv = cache_dir / "all_categories.csv"
    full_df: Optional[pd.DataFrame] = None
    if cache_csv.exists():
        try:
            full_df = pd.read_csv(cache_csv)
        except Exception:
            full_df = None

    if full_df is None:
        all_rows = []
        for category in cfg["categories"]:
            genes_in = _select_category_genes(
                df, category, int(cfg["top_n_per_category"]))
            sub = _hypergeometric_enrichment(
                genes_in, background, cfg["libraries"],
                organism=organism)
            if len(sub) == 0:
                continue
            sub["category"] = category
            sub["n_query_genes"] = len(genes_in)
            all_rows.append(sub)
        full_df = pd.concat(all_rows, ignore_index=True) if all_rows \
            else pd.DataFrame()
        if len(full_df) > 0:
            full_df.to_csv(cache_csv, index=False)

    if full_df is None or len(full_df) == 0:
        # Network-free fallback: empty heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Enrichr unreachable.\nRun once with internet to populate "
                f"{cache_csv.name}.",
                ha="center", va="center", fontsize=11, color="#555")
        return {"panel_id": "Fig4D", "figure_n": 4, "figure": fig,
                "config": cfg, "data": {}}

    # Per (category, library) keep top N terms by FDR
    keep_rows = []
    for cat in cfg["categories"]:
        for lib in cfg["libraries"]:
            sub = full_df[(full_df["category"] == cat) &
                          (full_df["library"] == lib)]
            sub = sub.copy()
            sub["fdr"] = pd.to_numeric(sub["fdr"], errors="coerce")
            sub = sub.dropna(subset=["fdr"]).sort_values("fdr")
            keep_rows.append(sub.head(int(cfg["max_terms_per_category"])))
    keep = pd.concat(keep_rows, ignore_index=True) if keep_rows \
        else pd.DataFrame()

    # Build the term × category matrix (one heatmap per library, stacked rows)
    cap = float(cfg["max_neglog10_fdr_cap"])
    cats = list(cfg["categories"])

    cat_labels_full = {
        "progression": "progression\n(+Δθ)",
        "counter":     "counter\n(−Δθ)",
        "branch":      "branch\n(|Δφ|)",
        "radial":      "radial\n(|Δlog₂r|)",
    }
    cat_colors = {
        "progression": _COLORS["progression"],
        "counter":     _COLORS["counter"],
        "branch":      _COLORS["branch"],
        "radial":      _COLORS["rad_pos"],
    }

    # Assemble a global matrix [terms_total x len(cats)] with library section
    # boundaries.  Sort terms within each library by best (smallest) FDR.
    rows_data = []
    section_boundaries = []
    section_labels = []
    cursor = 0
    for lib in cfg["libraries"]:
        sub = keep[keep["library"] == lib]
        if len(sub) == 0:
            continue
        terms_lib = (sub.sort_values("fdr")
                     .drop_duplicates("term")["term"].tolist())
        for term in terms_lib:
            row = np.full(len(cats), np.nan)
            for ci, cat in enumerate(cats):
                r = sub[(sub["term"] == term) & (sub["category"] == cat)]
                if len(r):
                    fdr = float(r["fdr"].values[0])
                    if not np.isnan(fdr):
                        val = -np.log10(max(fdr, 1e-30))
                        row[ci] = min(val, cap)
            rows_data.append((cfg["library_short"].get(lib, lib), term, row))
        section_boundaries.append((cursor, cursor + len(terms_lib)))
        section_labels.append(cfg["library_short"].get(lib, lib))
        cursor += len(terms_lib)

    if not rows_data:
        # No enrichment passed thresholds — explicit empty figure
        fig, ax = plt.subplots(figsize=(8.6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "No terms passed FDR threshold.",
                ha="center", va="center", fontsize=11)
        return {"panel_id": "Fig4D", "figure_n": 4, "figure": fig,
                "config": cfg, "data": {}}

    n_rows = len(rows_data)
    # mat_v rows = terms, cols = categories. Transpose so the figure is
    # horizontal: x-axis = terms (grouped by library), y-axis = categories.
    mat_v = np.vstack([r[2] for r in rows_data])
    term_labels = [r[1] for r in rows_data]
    lib_labels = [r[0] for r in rows_data]
    mat = mat_v.T  # shape (len(cats), n_rows)

    cmap = mpl.cm.get_cmap("magma_r").copy()
    cmap.set_bad("#f4f4f4")
    norm = mpl.colors.Normalize(vmin=0, vmax=cap)

    # Horizontal compact heatmap: ~0.235 inch per term column
    fig_w = max(9.5, 1.6 + 0.235 * n_rows)
    fig_h = 5.4
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(2, 2,
                          width_ratios=[0.965, 0.022],
                          height_ratios=[0.07, 0.22],
                          left=0.06, right=0.995, top=0.92, bottom=0.50,
                          wspace=0.012, hspace=0.05)
    libax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=libax)
    cax = fig.add_subplot(gs[1, 1])

    ma = np.ma.masked_invalid(mat)
    im = ax.imshow(ma, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # y axis = categories
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels([cat_labels_full[c] for c in cats],
                       fontsize=_FONT["tick"] + 0.5, ha="right")
    for ci, cat in enumerate(cats):
        ax.get_yticklabels()[ci].set_color(cat_colors[cat])
    ax.tick_params(axis="y", length=0)

    # x axis = terms, rotated labels (truncate aggressively to keep figure
    # readable in print)
    def _trim_term(t: str, max_chars: int = 42) -> str:
        # Strip Reactome 'R-HSA-' / 'R-MMU-' suffixes for readability
        s = t.replace("Homo sapiens", "").strip()
        if " R-HSA-" in s:
            s = s.split(" R-HSA-")[0]
        if " R-MMU-" in s:
            s = s.split(" R-MMU-")[0]
        # Strip "(GO:...)" suffixes from GO BP labels
        if "(GO:" in s:
            s = s.split("(GO:")[0].rstrip()
        return s if len(s) <= max_chars else s[:max_chars - 1] + "…"

    ax.set_xticks(np.arange(n_rows))
    ax.set_xticklabels(
        [_trim_term(t) for t in term_labels],
        fontsize=_FONT["tick"], rotation=55, ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=0, pad=2)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # white vertical section bands between libraries on the heatmap
    for i, (start, end) in enumerate(section_boundaries):
        if start == end:
            continue
        if i > 0:
            ax.axvline(start - 0.5, color="white", lw=2.6)

    # library labels above the heatmap on libax
    libax.set_xlim(-0.5, n_rows - 0.5)
    libax.set_ylim(0, 1)
    libax.axis("off")
    band_colors = ["#cfd8dc", "#dfe6e9", "#cfd8dc"]
    for i, ((start, end), lab) in enumerate(
            zip(section_boundaries, section_labels)):
        if start == end:
            continue
        libax.add_patch(plt.Rectangle((start - 0.5, 0.05),
                                      end - start, 0.55,
                                      facecolor=band_colors[i % len(band_colors)],
                                      edgecolor="white", linewidth=2.0))
        libax.text((start + end - 1) / 2.0, 0.32, lab,
                   ha="center", va="center", fontsize=_FONT["label"],
                   color="#1d2230", weight="bold")

    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label(r"$-\log_{10}$ FDR  (capped)", fontsize=_FONT["tick"])
    cbar.ax.tick_params(labelsize=_FONT["tick"] - 0.5, length=2)
    cbar.outline.set_linewidth(0.4)

    fig.suptitle(
        f"D   Distinct programs map onto distinct sphere coordinates "
        f"({_MAIN_DATASETS[cfg['dataset']].display})",
        x=0.02, y=0.985, ha="left",
        fontsize=_FONT["title"] + 0.5, weight="bold", color="#1d2230",
    )
    return {
        "panel_id": "Fig4D",
        "figure_n": 4,
        "figure": fig,
        "config": cfg,
        "data": {
            "enrichment_full": full_df,
            "enrichment_top": keep,
        },
    }


# ---------------------------------------------------------------------------
# Supplementary panel — BrCa atlas, three sphere views
# ---------------------------------------------------------------------------

PANEL_S_BRCA_DEFAULT = {
    "dataset": "BrCa_atlas",
    "n_plot_cells": 12000,
    "color_priority": ["celltype_major", "celltype_minor", "subtype"],
    # three orthogonal camera angles that together cover the full sphere
    "view_angles": [(20, -60), (20, 120), (-60, -60)],
    "seed": 0,
}


def panel_S_brca(config: Optional[dict] = None,
                 out_root: Optional[Path] = None) -> dict:
    _apply_panel_style()
    cfg = common.merge_config(PANEL_S_BRCA_DEFAULT, config)
    folder = common.ROOT / "raw_data" / cfg["dataset"]
    try:
        df = psp_io.load_brca_3d_coords(str(folder))
    except Exception as e:
        return common.data_not_present_result(
            "SuppFig4_BrCa", 4, "Suppl. Fig 4 — BrCa atlas",
            f"BrCa loader failed: {e}")
    pts = df[["X", "Y", "Z"]].values.astype(float)
    norm = np.linalg.norm(pts, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = pts / norm
    color_col = next((c for c in cfg["color_priority"] if c in df.columns), None)

    rng = np.random.default_rng(cfg["seed"])
    n_plot = min(unit.shape[0], int(cfg["n_plot_cells"]))
    idx = np.sort(rng.choice(unit.shape[0], n_plot, replace=False))

    fig = plt.figure(figsize=(12.4, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.02, left=0.02, right=0.98,
                          top=0.92, bottom=0.18)

    if color_col is not None:
        cats = pd.Categorical(df[color_col].astype(str))
        codes = cats.codes
        color_levels = list(cats.categories)
        n_lev = max(1, len(color_levels))
        cmap = mpl.cm.get_cmap("tab10" if n_lev <= 10 else "tab20", n_lev)
        colors = cmap(codes / max(1, n_lev - 1))
        colors_idx = colors[idx]
    else:
        colors_idx = np.tile(np.array([[0.4, 0.4, 0.45, 0.6]]), (n_plot, 1))
        color_levels = []
        cmap = None

    for i, (elev, azim) in enumerate(cfg["view_angles"]):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        _draw_3d_sphere_backdrop(ax, color="#e8edf2", wire="#aab3bf",
                                 alpha_surface=0.10, alpha_wire=0.20)
        ax.scatter(unit[idx, 0], unit[idx, 1], unit[idx, 2],
                   c=colors_idx, s=2.4, alpha=0.78,
                   linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        _style_clean_3d(ax)
        ax.set_title(f"view {i + 1}  (elev={elev}°, azim={azim}°)",
                     fontsize=_FONT["tick"], color="#1d2230", pad=2)

    if color_col is not None and cmap is not None:
        n_lev = max(1, len(color_levels))
        handles = []
        for lvi, lv in enumerate(color_levels):
            handles.append(Line2D(
                [0], [0], marker="o", lw=0,
                color=cmap(lvi / max(1, n_lev - 1)),
                markersize=6, label=str(lv)))
        leg = fig.legend(handles=handles, loc="lower center",
                         bbox_to_anchor=(0.5, 0.025),
                         ncol=min(6, len(handles)),
                         frameon=False, fontsize=_FONT["tick"],
                         columnspacing=1.2, handletextpad=0.4)

    fig.text(0.02, 0.97,
             f"Suppl. Fig 4 — BrCa atlas on S²  "
             f"({n_plot:,} of {unit.shape[0]:,} cells; "
             f"color: {color_col or 'uniform'})",
             fontsize=_FONT["title"], weight="bold", color="#1d2230")
    fig.text(0.98, 0.97,
             "Cautionary supplement: spherical structure exists in adult "
             "atlases too — do not over-interpret developmentally.",
             fontsize=_FONT["tick"] - 0.4, color="#5a5e6b",
             ha="right", style="italic")

    out = pd.DataFrame({
        "x": unit[idx, 0], "y": unit[idx, 1], "z": unit[idx, 2],
        "name": df.index.values[idx],
    })
    if color_col is not None:
        out[color_col] = df[color_col].values[idx]
    return {
        "panel_id": "SuppFig4_BrCa",
        "figure_n": 4,
        "figure": fig,
        "config": cfg,
        "output_aliases": [
            {"dir": "supplement/Fig4", "stem": "SuppFig4_BrCa_3D_views"},
        ],
        "data": {"plotted_cells": out},
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PANELS = {
    "A": panel_A,
    "B": panel_B,
    "C": panel_C,
    "D": panel_D,
    "S_BrCa": panel_S_brca,
}
