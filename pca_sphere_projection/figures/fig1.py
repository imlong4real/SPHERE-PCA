"""Figure 1 — conceptual framework.

Panels:
    A: pipeline schematic (no data)
    B: 3D sphere of celegan
    C: equirectangular celegan
    D: theta/radial conceptual diagram (synthetic, illustrative)
    E: C. elegans root/reference sensitivity
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

from .. import core as pc
from .. import sphere_stats as psp_sph
from .. import stripe as psp_stripe
from . import common


# ---------------------------------------------------------------------------
# Panel A — pipeline schematic
# ---------------------------------------------------------------------------

PANEL_A_DEFAULT = {
    "title": "Spherical PCA pipeline",
    "boxes": [
        "raw counts\n(cell × gene)", "log-norm\n+ HVGs",
        "PCA\n(PC1–PC3)", "L2 → S²\nalign root",
        "θ, φ,\nradial norm", 
    ],
    "seed": 0,
}


def panel_A(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_A_DEFAULT, config)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    n = len(cfg["boxes"])
    width = 1.4
    spacing = (10 - 0.4 - n * width) / (n - 1)
    boxes = []
    x = 0.2
    for txt in cfg["boxes"]:
        boxes.append((x, 1.5, width, 1.0, txt))
        x += width + spacing
    for (bx, by, bw, bh, t) in boxes:
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.05",
                                    edgecolor="black", facecolor="#f0f0f0"))
        ax.text(bx + bw / 2, by + bh / 2, t, ha="center", va="center", fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2]
        x1 = boxes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x0, boxes[i][1] + boxes[i][3] / 2),
                                     (x1, boxes[i + 1][1] + boxes[i + 1][3] / 2),
                                     arrowstyle="->", mutation_scale=12, lw=1.0))
    ax.text(5, 3.5, cfg["title"], fontsize=14, fontweight="bold", ha="center")
    return {"panel_id": "Fig1A", "figure_n": 1, "figure": fig, "config": cfg,
            "data": {"box_labels": pd.DataFrame({"step": cfg["boxes"]})}}


# ---------------------------------------------------------------------------
# Panel B — 3D sphere celegan
# ---------------------------------------------------------------------------

PANEL_B_DEFAULT = {
    "dataset": "celegan",
    "source_csv": "examples/celegan_pca.csv",
    "cluster_order": ['< 100', "100-130", "130-170", "170-210", "210-270",
                      "270-330", "330-390", "390-450", "450-510",
                      '510-580', '580-650', '> 650'],
    "root_cluster": "100-130",
    "root_centroid_uses_nonmissing_celltype": True,
    "euler_rotation_degrees": [-30, 0, -50],
    "point_size": 0.65,
    "alpha": 0.48,
    "cmap": "viridis",
    "show_plot_text": False,
    "show_3d_annotations": False,
    "manual_band_note": "not used in Fig1B",
    "preprocessing": "precomputed PC1-PC3 from examples/celegan_pca.csv",
    "sphere_alignment": "align_to_north_pole(root centroid from cluster 100-130) + apply_euler_rotation([-30, 0, -50])",
    "seed": 0,
}


def _load_and_transform_celegan(cfg: dict) -> tuple[pd.DataFrame, dict]:
    data = common.load_pca_csv("celegan")
    if data is None:
        raise FileNotFoundError("examples/celegan_pca.csv")
    data = data.copy()
    order = list(cfg["cluster_order"])
    cluster_to_time = {c: i for i, c in enumerate(order)}
    data["embryo_time_order"] = data["cluster"].map(cluster_to_time)
    unexpected = sorted(set(data["cluster"].dropna().astype(str)) - set(order))
    missing_cluster = int(data["cluster"].isna().sum())

    filtered_data = data.dropna(subset=["celltype"])
    root_mask = filtered_data["cluster"].astype(str) == str(cfg["root_cluster"])
    if not root_mask.any():
        raise ValueError(f"root cluster {cfg['root_cluster']!r} absent after celltype filtering")
    centroid = filtered_data.loc[root_mask, ["PC1", "PC2", "PC3"]].mean().values

    aligned = pc.align_to_north_pole(
        data,
        pcs_columns=["PC1", "PC2", "PC3"],
        cluster_column="cluster",
        root_node=centroid,
    )
    rotated = pc.apply_euler_rotation(
        aligned,
        pcs_columns=["PC1", "PC2", "PC3"],
        rotation_angles=cfg["euler_rotation_degrees"],
        degrees=True,
    )
    coords = rotated[["rotated_PC1", "rotated_PC2", "rotated_PC3"]].to_numpy(dtype=float)
    lon, lat = common.xyz_to_lonlat(coords)
    rotated["longitude"] = lon
    rotated["latitude"] = lat
    rotated["embryo_time_order"] = data["embryo_time_order"].to_numpy()
    metadata = {
        "n_cells_plotted": int(len(rotated)),
        "n_cells_missing_celltype": int(data["celltype"].isna().sum()),
        "n_cells_missing_cluster": missing_cluster,
        "unexpected_cluster_values": unexpected,
        "root_centroid": centroid.tolist(),
        "root_centroid_n_cells": int(root_mask.sum()),
    }
    return rotated, metadata


def _embryo_time_cmap(cfg: dict):
    order = list(cfg["cluster_order"])
    cmap = mpl.colormaps[cfg["cmap"]]
    norm = mpl.colors.Normalize(vmin=0, vmax=len(order) - 1)
    return cmap, norm


def _style_3d_axis(ax):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_zlim(-1.02, 1.02)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.set_axis_off()


def panel_B(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_B_DEFAULT, config)
    try:
        data, meta = _load_and_transform_celegan(cfg)
    except Exception as exc:
        return common.data_not_present_result(
            "Fig1B", 1, "Fig 1B — celegan 3D sphere",
            f"Could not load/transform examples/celegan_pca.csv: {exc}")

    coords = data[["rotated_PC1", "rotated_PC2", "rotated_PC3"]].to_numpy(dtype=float)
    color = data["embryo_time_order"].to_numpy(dtype=float)
    cmap, norm = _embryo_time_cmap(cfg)

    fig = plt.figure(figsize=(6.2, 6.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_surface(xs, ys, zs, color="#d9e3ea", alpha=0.055, linewidth=0, shade=False)
    ax.plot_wireframe(xs, ys, zs, color="#9aa9b4", alpha=0.11, linewidth=0.28)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=color, s=float(cfg["point_size"]), cmap=cmap, norm=norm,
                    alpha=float(cfg["alpha"]), linewidths=0, depthshade=False)
    _style_3d_axis(ax)
    ax.view_init(elev=21, azim=-52, roll=0)
    if cfg.get("show_3d_annotations", False):
        ax.text(0.02, 0.02, 1.16, "root / early embryo", ha="center", va="bottom",
                fontsize=8.5, color="#263238")
        ax.quiver(0, 0, 1.05, 0.36, -0.20, -0.22, color="#2f3a45",
                  linewidth=1.0, arrow_length_ratio=0.18)
        ax.text(0.43, -0.25, 0.78, "developmental\nprogression", ha="left",
                va="center", fontsize=8.5, color="#2f3a45")
    if cfg.get("show_plot_text", False):
        ax.set_title("C. elegans embryogenesis on spherical PCA", pad=0, fontsize=12)
    cb = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.005, fraction=0.045)
    ticks = np.arange(len(cfg["cluster_order"]))
    cb.set_ticks(ticks)
    cb.set_label("Embryo time", fontsize=9)
    cb.set_ticklabels(cfg["cluster_order"])
    cb.ax.tick_params(labelsize=6.5, length=2)
    fig.tight_layout(pad=0.2)

    df = data.reset_index(names="cell_id")[[
        "cell_id", "cluster", "celltype", "PC1", "PC2", "PC3",
        "rotated_PC1", "rotated_PC2", "rotated_PC3",
        "longitude", "latitude", "embryo_time_order",
    ]]
    cfg.update(meta)
    return {
        "panel_id": "Fig1B", "figure_n": 1, "figure": fig, "config": cfg,
        "output_aliases": [{"dir": "fig1", "stem": "Fig1B_celegans_3d_sphere"}],
        "data": {"processed_coordinates": df},
    }


# ---------------------------------------------------------------------------
# Panel C — equirectangular celegan
# ---------------------------------------------------------------------------

PANEL_C_DEFAULT = {
    **PANEL_B_DEFAULT,
    "manual_stripe_longitudes": [-60, 30, 100, 150],
    "manual_stripe_half_widths": [50, 30, 20, 20],
    "manual_band_note": "manual illustrative stripe bands from the original notebook; not automated multi-stripe detection",
    "point_size": 0.55,
    "alpha": 0.58,
    "show_manual_bands": False,
    "show_plot_text": False,
}


def _draw_wrapped_span(ax, center, width, **kwargs):
    lo = center - width
    hi = center + width
    spans = []
    if lo < -180:
        spans = [(-180, hi), (lo + 360, 180)]
    elif hi > 180:
        spans = [(lo, 180), (-180, hi - 360)]
    else:
        spans = [(lo, hi)]
    for a, b in spans:
        ax.axvspan(a, b, **kwargs)


def _make_panel_c_figure(data, cfg, *, diagnostic: bool = False):
    cmap, norm = _embryo_time_cmap(cfg)
    colors = data["embryo_time_order"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.7, 4.3), facecolor="white")
    ax.set_facecolor("#fbfcfd")
    centers = cfg["manual_stripe_longitudes"]
    widths = cfg["manual_stripe_half_widths"]
    band_colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    if cfg.get("show_manual_bands", False):
        for i, (center, width) in enumerate(zip(centers, widths), start=1):
            _draw_wrapped_span(ax, center, width, color=band_colors[i - 1],
                               alpha=0.12 if not diagnostic else 0.16, lw=0)
            if cfg.get("show_plot_text", False):
                ax.text(center, 73, f"Stripe {i}", ha="center", va="center",
                        fontsize=8.5, color="#263238", fontweight="bold")
    sc = ax.scatter(data["longitude"], data["latitude"], c=colors,
                    s=float(cfg["point_size"]), cmap=cmap, norm=norm,
                    alpha=float(cfg["alpha"]), linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-80, 80)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-80, 81, 40))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    if cfg.get("show_plot_text", False):
        ax.set_title("C. elegans spherical PCA: equirectangular projection", fontsize=12)
    ax.tick_params(width=0.7, length=3, labelsize=8.5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.018)
    cb.set_ticks(np.arange(len(cfg["cluster_order"])))
    cb.set_label("Embryo time", fontsize=9)
    cb.set_ticklabels(cfg["cluster_order"])
    cb.ax.tick_params(labelsize=6.5, length=2)

    stripe_counts = common.count_cells_within_bands_wrapped(
        data, centers, widths, lon_column="longitude")
    if diagnostic and cfg.get("show_plot_text", False):
        between = common.count_between_bands_wrapped(
            data["longitude"].to_numpy(dtype=float), centers, widths)
        for _, row in stripe_counts.iterrows():
            center = row["center_longitude_deg"]
            width = row["band_half_width_deg"]
            txt = f"{row['stripe']}\n{center:.0f} ± {width:.0f}°\nn={int(row['n_cells_inside']):,}"
            ax.text(center, 50, txt, ha="center", va="center", fontsize=7.2,
                    color="#111", bbox=dict(boxstyle="round,pad=0.22",
                                            facecolor="white", edgecolor="#c7ced6",
                                            alpha=0.78, linewidth=0.5))
        y_positions = [-64, -70, -64, -70]
        for y, (_, row) in zip(y_positions, between.iterrows()):
            x = row["from_longitude_deg"]
            x2 = row["to_longitude_deg"]
            if abs(x2 - x) > 180:
                xm = 175 if x > 0 else -175
            else:
                xm = (x + x2) / 2
            ax.text(xm, y, f"between\nn={int(row['n_cells_between']):,}",
                    ha="center", va="center", fontsize=6.7, color="#4b5563")
        ax.set_title("C. elegans spherical PCA: manual stripe-band diagnostics", fontsize=12)
    fig.tight_layout(pad=0.35)
    return fig


def panel_C(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_C_DEFAULT, config)
    try:
        data, meta = _load_and_transform_celegan(cfg)
    except Exception as exc:
        return common.data_not_present_result(
            "Fig1C", 1, "Fig 1C — celegan equirectangular",
            f"Could not load/transform examples/celegan_pca.csv: {exc}")

    stripe_counts = common.count_cells_within_bands_wrapped(
        data, cfg["manual_stripe_longitudes"], cfg["manual_stripe_half_widths"],
        lon_column="longitude")
    between_counts = common.count_between_bands_wrapped(
        data["longitude"].to_numpy(dtype=float),
        cfg["manual_stripe_longitudes"],
        cfg["manual_stripe_half_widths"])
    clean_fig = _make_panel_c_figure(data, cfg, diagnostic=False)
    diagnostic_fig = _make_panel_c_figure(data, cfg, diagnostic=True)

    processed = data.reset_index(names="cell_id")[[
        "cell_id", "cluster", "celltype", "PC1", "PC2", "PC3",
        "rotated_PC1", "rotated_PC2", "rotated_PC3",
        "longitude", "latitude", "embryo_time_order",
    ]]
    unexpected_df = pd.DataFrame({
        "unexpected_cluster": meta["unexpected_cluster_values"] or [],
    })
    cfg.update(meta)
    cfg["caption_note"] = "Manual illustrative stripe bands from the original notebook; automated multi-stripe metrics are separate."
    return {
        "panel_id": "Fig1C", "figure_n": 1, "figure": clean_fig, "config": cfg,
        "output_aliases": [{"dir": "fig1", "stem": "Fig1C_celegans_equirectangular_stripes_clean"}],
        "extra_figures": [{
            "stem": "Fig1C_celegans_equirectangular_stripes_diagnostic",
            "figure": diagnostic_fig,
            "output_aliases": [{"dir": "fig1", "stem": "Fig1C_celegans_equirectangular_stripes_diagnostic"}],
        }],
        "data": {
            "processed_coordinates": processed,
            "stripe_counts": stripe_counts,
            "between_band_counts": between_counts,
            "unexpected_clusters": unexpected_df,
        },
    }


# ---------------------------------------------------------------------------
# Panel D — theta / radial conceptual diagram
# ---------------------------------------------------------------------------

PANEL_D_DEFAULT = {
    "n_points": 800,
    "stripe_width_phi": 0.05,
    "radial_jitter_sd": 0.15,
    "great_circle_extent": 1.2,
    "seed": 0,
}


def panel_D(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_D_DEFAULT, config)
    rng = np.random.default_rng(cfg["seed"])
    n = int(cfg["n_points"])
    t = rng.uniform(-cfg["great_circle_extent"], cfg["great_circle_extent"], n)
    phi = rng.normal(0, cfg["stripe_width_phi"], n)
    radial = rng.normal(1.0, cfg["radial_jitter_sd"], n)
    x = radial * np.cos(t) * np.cos(phi)
    y = radial * np.cos(t) * np.sin(phi)
    z = radial * np.sin(t)
    pts = np.column_stack([x, y, z])
    unit = common.to_unit(pts)
    radial_norm = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.scatter(unit[:, 0], unit[:, 1], unit[:, 2], c=theta, s=4, cmap="cividis", alpha=0.9)
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax1.plot_wireframe(xs, ys, zs, color="lightgrey", alpha=0.15, linewidth=0.5)
    ax1.quiver(0, 0, 1.05, 0, 0, 0.45, color="black", lw=1.5)
    ax1.text(0, 0, 1.6, "north pole = root", fontsize=9, ha="center")
    ax1.set_title("S² with θ along trajectory")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(theta, radial_norm, c=theta, cmap="cividis", s=6, alpha=0.7)
    ax2.set_xlabel(r"$\theta$ (polar angle, rad)")
    ax2.set_ylabel("radial norm  ‖x‖₂")
    ax2.set_title("Two coordinates carry different biology")
    ax2.axhline(1.0, color="grey", lw=0.5, ls="--")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.text(0.5, 0.85,
             "θ — angular axis\ntracks the developmental program\n(pseudotime, lineage progression).",
             ha="center", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fde79c"))
    ax3.text(0.5, 0.45,
             "radial norm — scalar\ncaptures non-trajectory variance:\ncell-cycle, ribosome content, depth.",
             ha="center", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#a8d8ff"))
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.axis("off")

    fig.suptitle("Fig 1D. Why the spherical decomposition matters", y=1.02)
    fig.tight_layout()
    df = pd.DataFrame({"theta": theta, "radial_norm": radial_norm,
                       "x": unit[:, 0], "y": unit[:, 1], "z": unit[:, 2]})
    return {"panel_id": "Fig1D", "figure_n": 1, "figure": fig, "config": cfg,
            "data": {"synthetic_points": df}}


# ---------------------------------------------------------------------------
# Panel E — root/reference sensitivity (C. elegans PC CSV)
# ---------------------------------------------------------------------------

PANEL_E_DEFAULT = {
    "dataset": "celegan",
    "source_csv": "examples/celegan_pca.csv",
    "root_column": "cluster",
    "root_metric": "theta_pseudotime",
    "chosen_root_cluster": "100-130",
    "cluster_order": ['< 100', "100-130", "130-170", "170-210", "210-270",
                      "270-330", "330-390", "390-450", "450-510",
                      '510-580', '580-650', '> 650'],
    "cluster_time_map": {
        "< 100": 80,
        "100-130": 115,
        "130-170": 150,
        "170-210": 190,
        "210-270": 240,
        "270-330": 300,
        "330-390": 360,
        "390-450": 420,
        "450-510": 480,
        "510-580": 545,
        "580-650": 615,
        "> 650": 700,
    },
    "apply_fixed_euler_rotation": True,
    "rotation_angles": [-30, 0, -50],
    "early_plausible_roots": ['< 100', "100-130", "130-170"],
    "stripe_metric_max_cells": 3000,
    "caption_note": (
        "Root/reference sensitivity analysis using embryonic-time bins as candidate "
        "biological reference states. This panel does not automatically discover "
        "the true root and does not label the highest-correlation root as best. "
        "Rotation robustness is intentionally excluded from Fig 1E."
    ),
    "seed": 0,
}


def _root_sensitivity_celegan(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    df = common.load_pca_csv(cfg["dataset"])
    if df is None:
        raise FileNotFoundError(f"{cfg['dataset']}_pca.csv missing from examples/")
    required = {"PC1", "PC2", "PC3", cfg["root_column"]}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"missing required columns: {missing_cols}")

    cluster_order = list(cfg["cluster_order"])
    present = set(df[cfg["root_column"]].dropna().astype(str))
    missing_bins = [c for c in cluster_order if c not in present]
    unexpected = sorted(present - set(cluster_order))

    time_map = cfg["cluster_time_map"]
    pseudotime = df[cfg["root_column"]].astype(str).map(time_map).to_numpy(dtype=float)
    valid_pt = np.isfinite(pseudotime)
    if valid_pt.sum() < 5:
        raise ValueError("too few cells have parseable embryonic-time bins")

    rows = []
    for root in cluster_order:
        mask = df[cfg["root_column"]].astype(str) == root
        n_root = int(mask.sum())
        if n_root == 0:
            rows.append({
                "candidate_root": root,
                "root_time_midpoint": float(time_map[root]),
                "n_root_cells": 0,
                "theta_pseudotime_spearman": np.nan,
                "theta_pseudotime_p_value": np.nan,
                "multi_stripe_strength": np.nan,
                "global_longitude_concentration": np.nan,
                "is_chosen_root": root == cfg["chosen_root_cluster"],
                "is_early_plausible_root": root in cfg["early_plausible_roots"],
                "status": "missing root bin",
            })
            continue

        centroid = df.loc[mask, ["PC1", "PC2", "PC3"]].mean().values
        aligned = pc.align_to_north_pole(
            df,
            pcs_columns=["PC1", "PC2", "PC3"],
            cluster_column=cfg["root_column"],
            root_node=centroid,
        )
        if cfg.get("apply_fixed_euler_rotation", True):
            aligned = pc.apply_euler_rotation(
                aligned,
                pcs_columns=["PC1", "PC2", "PC3"],
                rotation_angles=cfg["rotation_angles"],
                degrees=True,
            )
            coord_cols = ["rotated_PC1", "rotated_PC2", "rotated_PC3"]
        else:
            coord_cols = ["PC1", "PC2", "PC3"]

        coords = common.to_unit(aligned[coord_cols].to_numpy(dtype=float))
        theta = np.arccos(np.clip(coords[:, 2], -1.0, 1.0))
        rho, p = spearmanr(theta[valid_pt], pseudotime[valid_pt])
        metric_coords = coords
        max_metric_cells = int(cfg.get("stripe_metric_max_cells", 0) or 0)
        if max_metric_cells > 0 and coords.shape[0] > max_metric_cells:
            rng = np.random.default_rng(int(cfg.get("seed", 0)))
            metric_idx = np.sort(rng.choice(coords.shape[0], max_metric_cells, replace=False))
            metric_coords = coords[metric_idx]
        try:
            multi_score = float(psp_stripe.multi_stripe_strength_score(metric_coords)["multi_stripe_strength"])
        except Exception:
            multi_score = np.nan
        try:
            global_score = float(psp_sph.stripe_strength_score(metric_coords)["score"])
        except Exception:
            global_score = np.nan
        rows.append({
            "candidate_root": root,
            "root_time_midpoint": float(time_map[root]),
            "n_root_cells": n_root,
            "theta_pseudotime_spearman": float(rho),
            "theta_pseudotime_p_value": float(p),
            "multi_stripe_strength": multi_score,
            "global_longitude_concentration": global_score,
            "stripe_metric_n_cells": int(metric_coords.shape[0]),
            "is_chosen_root": root == cfg["chosen_root_cluster"],
            "is_early_plausible_root": root in cfg["early_plausible_roots"],
            "status": "ok",
        })
    out = pd.DataFrame(rows)
    out["missing_cluster_bins"] = ";".join(missing_bins)
    out["unexpected_cluster_bins"] = ";".join(unexpected)
    return out, missing_bins


def panel_E(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_E_DEFAULT, config)
    try:
        sweep, missing_bins = _root_sensitivity_celegan(cfg)
    except Exception as exc:
        return common.data_not_present_result(
            "Fig1E", 1, "Fig 1E — C. elegans root/reference sensitivity",
            f"Could not compute root sensitivity: {exc}")

    x = np.arange(len(sweep))
    y = sweep["theta_pseudotime_spearman"].to_numpy(dtype=float)
    chosen = sweep["is_chosen_root"].to_numpy(dtype=bool)
    early = sweep["is_early_plausible_root"].to_numpy(dtype=bool)
    later = ~(chosen | early)

    fig = plt.figure(figsize=(8.9, 4.9), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.75], hspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_counts = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.axvspan(-0.45, 2.45, color="#dbeafe", alpha=0.65, lw=0, zorder=0)
    ax.axhline(0, color="#6b7280", lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.plot(x, y, color="#23395b", lw=1.8, zorder=2)
    ax.scatter(x[later], y[later], s=42, color="#9ca3af",
               edgecolor="white", linewidth=0.6, zorder=3, label="later reference")
    ax.scatter(x[early & ~chosen], y[early & ~chosen], s=58, color="#4f46e5",
               edgecolor="white", linewidth=0.7, zorder=4, label="early plausible reference")
    ax.scatter(x[chosen], y[chosen], s=165, marker="*", color="#f97316",
               edgecolor="#7c2d12", linewidth=0.8, zorder=5,
               label=f"main-figure reference ({cfg['chosen_root_cluster']})")
    ax.text(1.0, 0.97, "early biological reference states",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color="#1d4ed8")
    ax.text(0.02, 0.06, "root choice sets theta orientation",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.8, color="#374151")
    ax.set_ylabel(r"Spearman $\rho$ ($\theta$, embryonic time)")
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlim(-0.55, len(sweep) - 0.45)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=8.5, width=0.8, length=3)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=7.8, handletextpad=0.4)

    counts = sweep["n_root_cells"].to_numpy(dtype=float)
    count_colors = np.where(chosen, "#f97316", np.where(early, "#4f46e5", "#c7cbd1"))
    ax_counts.bar(x, counts, color=count_colors, width=0.58, edgecolor="none")
    ax_counts.set_ylabel("cells", fontsize=8)
    ax_counts.set_xlabel("candidate reference cluster")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(sweep["candidate_root"], rotation=35, ha="right", fontsize=7.5)
    ax_counts.tick_params(axis="y", labelsize=7.5, width=0.7, length=2)
    ax_counts.spines["left"].set_linewidth(0.7)
    ax_counts.spines["bottom"].set_linewidth(0.7)
    ax_counts.set_ylim(0, max(counts) * 1.18 if len(counts) else 1)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.23, hspace=0.08)

    cfg["missing_cluster_bins"] = missing_bins
    cfg["n_candidate_roots"] = int(len(sweep))
    cfg["root_cluster_chosen_for_main_figures"] = cfg["chosen_root_cluster"]
    cfg["rotation_robustness_excluded"] = True

    return {
        "panel_id": "Fig1E", "figure_n": 1, "figure": fig, "config": cfg,
        "output_aliases": [{
            "dir": "fig1",
            "stem": "Fig1E_root_sensitivity_celegans",
            "single_data_name": "root_sensitivity",
        }],
        "data": {
            "root_sensitivity": sweep,
        },
    }


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D, "E": panel_E}
