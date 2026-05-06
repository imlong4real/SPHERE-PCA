"""Figure 4 — gene + control.

Panels:
    A: Pou5f1 (Oct4) perturbation vector field on Klein
    B: UC epi gene gradients (top 3 stripe-boundary genes)
    C: BrCa atlas sphere plot (3D + equirectangular)
    D: BrCa vs developmental atlas comparison
    E: H5 gene-ranking visualisation (uc_epi, klein, celegan)
    F: summary conceptual diagram
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .. import io as psp_io
from .. import preprocessing as psp_pp
from .. import perturbation as psp_pert
from .. import gene_geometry as psp_gg
from .. import sphere_stats as psp_sph
from . import common


# ---------------------------------------------------------------------------
# Panel A — Pou5f1 perturbation
# ---------------------------------------------------------------------------

PANEL_A_DEFAULT = {
    "dataset": "klein",
    "gene": "Pou5f1",
    "frac_to_plot": 0.04,
    "clip_displacement_deg": 0.1,
    "normalize_to_sphere": True,
    "preprocessing": "log1p+all-genes+sklearn-PCA(3)",
    "seed": 0,
}


def panel_A(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_A_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    art = common.load_artifacts(val_out, cfg["dataset"])
    if art is None or "pca_components_full" not in art.files:
        return common.data_not_present_result(
            "Fig4A", 4, "Fig 4A — Pou5f1 perturbation",
            f"{cfg['dataset']} full PCA loadings missing; rerun the workflow.")
    expr = psp_io.load_bz2_klein_dataset(str(common.ROOT / "raw_data" / "klein"))
    if cfg["gene"] not in expr.gene_names:
        return common.data_not_present_result(
            "Fig4A", 4, f"Fig 4A — {cfg['gene']}",
            f"{cfg['gene']} not in {cfg['dataset']} gene panel.")
    X_norm = psp_pp.normalize_log1p(expr.X)
    pert = psp_pert.compute_gene_perturbation_vectors(
        np.asarray(X_norm), expr.gene_names, [cfg["gene"]],
        art["pca_components_full"], art["pca_mean_full"],
        normalize_to_sphere=cfg["normalize_to_sphere"])
    fig, ax = plt.subplots(figsize=(9, 5))
    psp_pert.plot_perturbation_vector_field(
        pert, cfg["gene"], frac=cfg["frac_to_plot"], ax=ax,
        clip=cfg["clip_displacement_deg"], seed=cfg["seed"])
    ax.set_title(f"Fig 4A — {cfg['gene']} fixed-loading sensitivity, {cfg['dataset']}\n"
                 f"blue = zeroed, red = doubled (sphere-normalised)")
    fig.tight_layout()
    r = pert[cfg["gene"]]
    df = pd.DataFrame({
        "x_orig": r["original"][:, 0], "y_orig": r["original"][:, 1],
        "z_orig": r["original"][:, 2],
        "x_zeroed": r["zeroed"][:, 0], "y_zeroed": r["zeroed"][:, 1],
        "z_zeroed": r["zeroed"][:, 2],
        "x_doubled": r["doubled"][:, 0], "y_doubled": r["doubled"][:, 1],
        "z_doubled": r["doubled"][:, 2],
    })
    return {"panel_id": "Fig4A", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {"perturbation_vectors": df}}


# ---------------------------------------------------------------------------
# Panel B — UC epi gene gradients
# ---------------------------------------------------------------------------

PANEL_B_DEFAULT = {
    "dataset": "uc_epi",
    "n_top": 3,
    "n_theta": 18,
    "n_phi": 36,
    "preferred_genes": None,  # None = pick top N from H5 boundary list, prefer known
    "seed": 0,
}


def panel_B(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_B_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    p = val_out / cfg["dataset"] / "H5_stripe_boundary_genes.csv"
    if not p.exists():
        return common.data_not_present_result(
            "Fig4B", 4, "Fig 4B", f"{cfg['dataset']} H5 boundary CSV missing.")
    df_h5 = pd.read_csv(p)
    if cfg["preferred_genes"]:
        chosen = list(cfg["preferred_genes"])[: cfg["n_top"]]
    else:
        # Take top n_top by stripe_score; if any known regulators are in the top 30,
        # bubble them to the front so the panel highlights at least one.
        top_n = df_h5.head(cfg["n_top"])["gene"].tolist()
        if "is_known_regulator" in df_h5.columns:
            knowns = df_h5.head(30)[df_h5.head(30)["is_known_regulator"]]["gene"].tolist()
            for g in reversed(knowns):  # prepend each known
                if g in top_n:
                    top_n.remove(g)
                top_n.insert(0, g)
        chosen = top_n[: cfg["n_top"]]
    cfg["chosen_genes"] = chosen

    expr = psp_io.load_uc_epi(str(common.ROOT / "raw_data" / "uc_epi"))
    art = common.load_artifacts(val_out, cfg["dataset"])
    if art is None:
        return common.data_not_present_result(
            "Fig4B", 4, "Fig 4B", f"{cfg['dataset']} artifacts missing.")
    geom = common.load_geometry(val_out, cfg["dataset"])
    if geom is not None and len(geom) != expr.n_cells:
        cid_to_idx = {c: i for i, c in enumerate(expr.cell_ids)}
        keep = [cid_to_idx[c] for c in geom["cell_id"].astype(str) if c in cid_to_idx]
        expr = psp_io._reindex_expression(expr, keep)
    X_norm = psp_pp.normalize_log1p(expr.X)
    coords = art["coords3_unit"]

    fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 4))
    if len(chosen) == 1: axes = [axes]
    for ax, g in zip(axes, chosen):
        ax2, im = psp_gg.plot_gene_gradient_field(
            coords, X_norm, expr.gene_names, g, ax=ax,
            n_theta=cfg["n_theta"], n_phi=cfg["n_phi"])
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="mean expr")
    fig.suptitle("Fig 4B. UC epithelium top stripe-boundary genes (mean expression on S² grid)",
                 y=1.03)
    fig.tight_layout()
    return {"panel_id": "Fig4B", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {"top_h5_boundary": df_h5.head(20)}}


# ---------------------------------------------------------------------------
# Panel C — BrCa sphere plot
# ---------------------------------------------------------------------------

PANEL_C_DEFAULT = {
    "dataset": "BrCa_atlas",
    "color_by_priority": ["celltype_major", "celltype_minor", "subtype"],
    "n_plot_cells": 20000,
    "seed": 0,
}


def panel_C(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_C_DEFAULT, config)
    folder = common.ROOT / "raw_data" / cfg["dataset"]
    try:
        df = psp_io.load_brca_3d_coords(str(folder))
    except Exception as e:
        return common.data_not_present_result(
            "Fig4C", 4, "Fig 4C — BrCa sphere",
            f"BrCa loader failed: {e}")
    pts = df[["X", "Y", "Z"]].values.astype(float)
    unit = common.to_unit(pts)
    color_col = next((c for c in cfg["color_by_priority"] if c in df.columns), None)
    cfg["color_column_used"] = color_col
    rng = np.random.default_rng(cfg["seed"])
    n_plot = min(unit.shape[0], cfg["n_plot_cells"])
    idx = np.sort(rng.choice(unit.shape[0], n_plot, replace=False))
    fig = plt.figure(figsize=(11, 4.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    if color_col is not None:
        cats = pd.Categorical(df[color_col].astype(str)).codes
        ax3d.scatter(unit[idx, 0], unit[idx, 1], unit[idx, 2],
                     c=cats[idx], s=2, cmap="tab20", alpha=0.5)
    else:
        ax3d.scatter(unit[idx, 0], unit[idx, 1], unit[idx, 2], s=2, alpha=0.4)
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    ax3d.set_title(f"BrCa atlas on S² (n={unit.shape[0]} cells, plotted {n_plot})")
    ax2 = fig.add_subplot(1, 2, 2)
    lon, lat = common.xyz_to_lonlat(unit[idx])
    if color_col is not None:
        cats = pd.Categorical(df[color_col].astype(str)).codes
        ax2.scatter(lon, lat, c=cats[idx], s=2, cmap="tab20", alpha=0.5)
    else:
        ax2.scatter(lon, lat, s=2, alpha=0.4)
    ax2.set_xlim(-180, 180); ax2.set_ylim(-90, 90)
    ax2.set_xlabel("longitude (°)"); ax2.set_ylabel("latitude (°)")
    ax2.set_title(f"Equirectangular (color = {color_col or 'uniform'})")
    fig.suptitle("Fig 4C. BrCa whole-atlas spherical embedding", y=1.02)
    fig.tight_layout()
    out = pd.DataFrame({"x": unit[idx, 0], "y": unit[idx, 1], "z": unit[idx, 2],
                        "lon": lon, "lat": lat,
                        "name": df.index.values[idx]})
    if color_col is not None:
        out[color_col] = df[color_col].values[idx]
    return {"panel_id": "Fig4C", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {"plotted_cells": out}}


# ---------------------------------------------------------------------------
# Panel D — BrCa vs developmental geometry
# ---------------------------------------------------------------------------

PANEL_D_DEFAULT = {
    "developmental": ["celegan", "uc_epi", "klein", "hesc"],
    "tumor": "BrCa_atlas",
    "metrics": ["stripe_strength", "great_circle_R2", "anisotropy_linear"],
    "seed": 0,
}


def panel_D(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_D_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    rows = []
    for ds in cfg["developmental"]:
        art = common.load_artifacts(val_out, ds)
        if art is None: continue
        unit = art["coords3_unit"]
        rows.append({
            "dataset": ds, "kind": "developmental",
            "stripe_strength": float(psp_sph.stripe_strength_score(unit)["score"]),
            "great_circle_R2": float(psp_sph.fit_great_circle(unit)["r_squared"]),
            "anisotropy_linear": float(psp_sph.spherical_anisotropy(unit)["linear"]),
        })
    folder = common.ROOT / "raw_data" / cfg["tumor"]
    try:
        bdf = psp_io.load_brca_3d_coords(str(folder))
        unit = common.to_unit(bdf[["X", "Y", "Z"]].values.astype(float))
        rows.append({
            "dataset": cfg["tumor"], "kind": "tumor",
            "stripe_strength": float(psp_sph.stripe_strength_score(unit)["score"]),
            "great_circle_R2": float(psp_sph.fit_great_circle(unit)["r_squared"]),
            "anisotropy_linear": float(psp_sph.spherical_anisotropy(unit)["linear"]),
        })
    except Exception:
        pass
    if not rows:
        return common.data_not_present_result(
            "Fig4D", 4, "Fig 4D", "no comparable artifacts available.")
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(cfg["metrics"]), figsize=(4 * len(cfg["metrics"]), 4))
    for ax, m in zip(axes, cfg["metrics"]):
        cols = [common.DATASET_COLORS.get(d, "#444") for d in df["dataset"]]
        ax.bar(df["dataset"], df[m], color=cols, edgecolor="black", linewidth=0.4)
        ax.set_title(m); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Fig 4D. BrCa atlas geometry vs developmental atlases", y=1.02)
    fig.tight_layout()
    return {"panel_id": "Fig4D", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {"geometry_metrics": df}}


# ---------------------------------------------------------------------------
# Panel E — H5 gene ranking visualisation
# ---------------------------------------------------------------------------

PANEL_E_DEFAULT = {
    "datasets": ["uc_epi", "klein", "celegan"],
    "n_top": 15,
    "seed": 0,
}


def panel_E(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_E_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    fig, axes = plt.subplots(1, len(cfg["datasets"]),
                              figsize=(5 * len(cfg["datasets"]), 5))
    seen_data = {}
    for ax, ds in zip(axes, cfg["datasets"]):
        p = val_out / ds / "H5_stripe_boundary_genes.csv"
        if not p.exists():
            ax.text(0.5, 0.5, f"{ds}: no H5", transform=ax.transAxes, ha="center")
            continue
        df = pd.read_csv(p).head(cfg["n_top"]).iloc[::-1]
        cols = ["#e45756" if k else common.DATASET_COLORS.get(ds, "#888")
                for k in df.get("is_known_regulator", [False] * len(df))]
        ax.barh(df["gene"], df["stripe_score"], color=cols)
        ax.set_xlabel("stripe_score = phi_var / theta_var")
        ax.set_title(f"{ds}: top H5 stripe-boundary genes")
        seen_data[ds] = df.iloc[::-1]
    if not seen_data:
        return common.data_not_present_result(
            "Fig4E", 4, "Fig 4E", "no H5 CSVs available.")
    fig.suptitle("Fig 4E. H5 stripe-boundary gene ranking (red = curated regulator)",
                 y=1.02)
    fig.tight_layout()
    return {"panel_id": "Fig4E", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {f"top_h5_{ds}": v for ds, v in seen_data.items()}}


# ---------------------------------------------------------------------------
# Panel F — summary conceptual diagram
# ---------------------------------------------------------------------------

PANEL_F_DEFAULT = {"seed": 0}


def panel_F(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_F_DEFAULT, config)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.add_patch(FancyBboxPatch((0.3, 3), 2.5, 1.2, boxstyle="round,pad=0.1",
                                facecolor="#fde79c", edgecolor="black"))
    ax.text(1.55, 3.6, "θ — angular axis\nlineage / pseudotime",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.3, 1.3), 2.5, 1.2, boxstyle="round,pad=0.1",
                                facecolor="#a8d8ff", edgecolor="black"))
    ax.text(1.55, 1.9, "radial norm\ncell-cycle / metabolic",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.3, 0.0), 2.5, 1.2, boxstyle="round,pad=0.1",
                                facecolor="#c8e6c9", edgecolor="black"))
    ax.text(1.55, 0.6, "φ / stripe boundary\nswitch-like programs",
            ha="center", va="center", fontsize=10, fontweight="bold")
    ax.add_patch(FancyBboxPatch((4.5, 1.5), 2.5, 2.5, boxstyle="round,pad=0.1",
                                facecolor="#f3f3f3", edgecolor="black"))
    ax.text(5.75, 3.6, "S² embedding", ha="center", va="center", fontsize=12, fontweight="bold")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(5.75 + 0.9 * np.cos(th), 2.8 + 0.9 * np.sin(th), color="grey", lw=0.8)
    ax.plot(5.75 + 0.9 * np.cos(th), 2.8 + 0.9 * np.sin(th) * 0.3,
            color="grey", lw=0.8, ls="--")
    for y0, label in [(3.6, "θ"), (1.9, "radial"), (0.6, "φ")]:
        ax.add_patch(FancyArrowPatch((2.85, y0), (4.45, y0 + (2.8 - y0) * 0.5),
                                     arrowstyle="->", mutation_scale=12, color="black"))
        ax.text(3.7, y0 + (2.8 - y0) * 0.25, label, fontsize=10, ha="center", va="bottom")
    cards = [
        ("H1", "stemness ↘ along geodesic\n(hESC, planaria, germ cell, pre-implant)"),
        ("H4", "radial vs θ split:\ncell-cycle is radial, lineage is θ"),
        ("H7", "fixed-loading sensitivity\nrecapitulates known regulators\n(Klein: Oct4; UC: MUC2/OLFM4)"),
    ]
    for i, (h, txt) in enumerate(cards):
        y = 3.6 - i * 1.4
        ax.add_patch(FancyBboxPatch((7.5, y - 0.3), 2.3, 1.0, boxstyle="round,pad=0.1",
                                    facecolor="white", edgecolor="black"))
        ax.text(7.7, y + 0.55, h, fontsize=11, fontweight="bold")
        ax.text(7.7, y + 0.05, txt, fontsize=8.5, va="top")
    ax.set_title("Fig 4F. What the spherical decomposition tells us, and what it does not",
                 fontsize=12, pad=14)
    ax.text(5, 0.1,
            "Fixed-loading perturbation is a sensitivity analysis, not a CRISPR/RNAi prediction.",
            fontsize=8.5, ha="center", style="italic", color="#444")
    return {"panel_id": "Fig4F", "figure_n": 4, "figure": fig, "config": cfg,
            "data": {}}


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D,
          "E": panel_E, "F": panel_F}
