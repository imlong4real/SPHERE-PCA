"""Figure 2 — robust spherical geometry.

Scientific message:
    Spherical PCA geometry is a measurable, robust geometric structure.
    Its sharpness ranges from a clean global geodesic (Klein mESC) to multiple
    local arcs/ridges (C. elegans). Figure 2 documents this robustness without
    overclaiming a universal "great-circle" fit. HVG vs all-gene preprocessing
    is a robustness control, not the central message.

Panels:
    A: Global trajectory baseline (Klein) vs multi-arc geometry (C. elegans).
    B: Rotation robustness across all datasets.
    C: PC-axis specificity heatmap across all datasets.
    D: Random PC-triplet null distributions across all datasets.
    E: Canonical Scanpy seurat-flavor HVG vs all-gene PCA.
    F: C. elegans equirectangular spherical density / multi-arc structure.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .. import sphere_stats as psp_sph
from .. import robustness as psp_rob
from . import common
from . import fig1 as _fig1


# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------

_PALETTE = {
    "celegan": "#4c78a8",
    "klein": "#54a24b",
    "hesc": "#e45756",
    "planaria": "#b279a2",
    "human_germ_cell": "#9d755d",
    "pre_implant_human_embryo": "#ff9da6",
    "uc_epi": "#f58518",
}


def _ds_color(ds: str) -> str:
    return _PALETTE.get(ds, common.DATASET_COLORS.get(ds, "#444"))


def _ds_label(ds: str) -> str:
    return common.DATASET_DISPLAY.get(ds, ds)


def _structure_metric(unit: np.ndarray) -> float:
    return float(psp_sph.stripe_strength_score(unit)["score"])


def _great_circle_r2(unit: np.ndarray) -> float:
    return float(psp_sph.fit_great_circle(unit)["r_squared"])


def _anisotropy_linear(unit: np.ndarray) -> float:
    return float(psp_sph.spherical_anisotropy(unit)["linear"])


# ---------------------------------------------------------------------------
# Panel A — global trajectory baseline vs multi-arc geometry
# ---------------------------------------------------------------------------

PANEL_A_DEFAULT = {
    "datasets": ["klein", "celegan"],
    "n_plot_cells": 4500,
    "circle_resolution": 360,
    # C. elegans is plotted in the Fig. 1B orientation:
    "celegan_alignment": dict(_fig1.PANEL_B_DEFAULT),
    "seed": 0,
}


def _celegan_unit_fig1b():
    """Return (unit_coords, color_index) for celegan in Fig 1B's orientation."""
    cfg = dict(_fig1.PANEL_B_DEFAULT)
    data, _ = _fig1._load_and_transform_celegan(cfg)
    coords = data[["rotated_PC1", "rotated_PC2", "rotated_PC3"]].to_numpy(dtype=float)
    unit = common.to_unit(coords)
    color = data["embryo_time_order"].to_numpy(dtype=float)
    return unit, color, cfg["cluster_order"]


def _klein_unit():
    """Return PC1-3 unit vectors for Klein, prefer the cached artifacts."""
    return common.load_unit_coords_pc13("klein")


def _draw_sphere_backdrop(ax, *, color="#d9e3ea", wire="#9aa9b4"):
    u, v = np.mgrid[0:2 * np.pi:48j, 0:np.pi:24j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax.plot_surface(xs, ys, zs, color=color, alpha=0.06, linewidth=0,
                    shade=False)
    ax.plot_wireframe(xs, ys, zs, color=wire, alpha=0.10, linewidth=0.28)


def _style_3d(ax):
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.04, 1.04); ax.set_ylim(-1.04, 1.04); ax.set_zlim(-1.04, 1.04)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis._axinfo["axisline"]["color"] = (1, 1, 1, 0)
    ax.set_axis_off()


def panel_A(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_A_DEFAULT, config)
    rng = np.random.default_rng(cfg["seed"])
    n_plot = int(cfg["n_plot_cells"])

    # Klein
    klein_unit = _klein_unit()
    if klein_unit is None or klein_unit.shape[0] == 0:
        return common.data_not_present_result(
            "Fig2A", 2, "Fig 2A — global vs multi-arc",
            "Klein PC1-3 unit coords unavailable.")
    klein_curve, klein_gc = common.fit_great_circle_curve(
        klein_unit, n_points=int(cfg["circle_resolution"]))
    if klein_unit.shape[0] > n_plot:
        idx = rng.choice(klein_unit.shape[0], n_plot, replace=False)
        klein_plot = klein_unit[idx]
    else:
        klein_plot = klein_unit

    # C. elegans (Fig 1B orientation)
    try:
        cel_unit_full, cel_color_full, cluster_order = _celegan_unit_fig1b()
    except Exception as exc:
        return common.data_not_present_result(
            "Fig2A", 2, "Fig 2A — global vs multi-arc",
            f"Could not load celegan in Fig 1B orientation: {exc}")
    cel_curve, cel_gc = common.fit_great_circle_curve(
        cel_unit_full, n_points=int(cfg["circle_resolution"]))
    if cel_unit_full.shape[0] > n_plot:
        idx = rng.choice(cel_unit_full.shape[0], n_plot, replace=False)
        cel_unit = cel_unit_full[idx]
        cel_color = cel_color_full[idx]
    else:
        cel_unit = cel_unit_full
        cel_color = cel_color_full

    fig = plt.figure(figsize=(9.6, 4.6), facecolor="white")
    # Klein
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_sphere_backdrop(ax1)
    ax1.scatter(klein_plot[:, 0], klein_plot[:, 1], klein_plot[:, 2],
                c=_PALETTE["klein"], s=4.0, alpha=0.55, linewidths=0,
                depthshade=False)
    ax1.plot(klein_curve[:, 0], klein_curve[:, 1], klein_curve[:, 2],
             color="#222222", lw=1.8, alpha=0.9, zorder=4)
    _style_3d(ax1)
    ax1.view_init(elev=18, azim=-58)
    ax1.set_title("Klein mESC\nNear-linear global trajectory",
                  fontsize=11, pad=2)
    ax1.text2D(0.02, 0.04, f"GC R² = {klein_gc['r_squared']:.3f}",
               transform=ax1.transAxes, fontsize=10, color="#111",
               bbox=dict(boxstyle="round,pad=0.3", fc="white",
                         ec="#cccccc", lw=0.5, alpha=0.85))

    # Celegan
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _draw_sphere_backdrop(ax2)
    ax2.scatter(cel_unit[:, 0], cel_unit[:, 1], cel_unit[:, 2],
                c=cel_color, cmap="viridis", s=2.4, alpha=0.55,
                linewidths=0, depthshade=False)
    ax2.plot(cel_curve[:, 0], cel_curve[:, 1], cel_curve[:, 2],
             color="#222222", lw=1.2, alpha=0.55, ls=(0, (4, 2)), zorder=4)
    _style_3d(ax2)
    ax2.view_init(elev=21, azim=-52)
    ax2.set_title("C. elegans\nMulti-arc developmental geometry",
                  fontsize=11, pad=2)
    ax2.text2D(0.02, 0.04,
               f"GC R² = {cel_gc['r_squared']:.3f}\n(single-trajectory baseline)",
               transform=ax2.transAxes, fontsize=9.5, color="#111",
               bbox=dict(boxstyle="round,pad=0.3", fc="white",
                         ec="#cccccc", lw=0.5, alpha=0.85))

    fig.suptitle("Global geodesic baseline vs. multi-arc geometry",
                 fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout(pad=0.4)

    rows = [
        {"dataset": "klein", "n_cells_total": int(klein_unit.shape[0]),
         "n_cells_plotted": int(klein_plot.shape[0]),
         "great_circle_R2": float(klein_gc["r_squared"]),
         "mean_residual_radians": float(klein_gc["mean_residual_radians"]),
         "interpretation": "near-linear global trajectory"},
        {"dataset": "celegan", "n_cells_total": int(cel_unit_full.shape[0]),
         "n_cells_plotted": int(cel_unit.shape[0]),
         "great_circle_R2": float(cel_gc["r_squared"]),
         "mean_residual_radians": float(cel_gc["mean_residual_radians"]),
         "interpretation": "multi-arc; great circle is a baseline only"},
    ]
    return {
        "panel_id": "Fig2A", "figure_n": 2, "figure": fig, "config": cfg,
        "data": {"great_circle_fits": pd.DataFrame(rows)},
    }


# ---------------------------------------------------------------------------
# Panel B — rotation robustness across all datasets
# ---------------------------------------------------------------------------

PANEL_B_DEFAULT = {
    "datasets": list(common.FIG2_ALL_DATASETS),
    "perturbation_degrees": 20.0,
    "n_perturbations": 200,
    "metric": "global_longitude_concentration",
    "max_cells": 8000,
    "seed": 0,
}


def _build_pca_df_for_robustness(unit: np.ndarray) -> pd.DataFrame:
    """Wrap unit coords as a PC1/PC2/PC3 dataframe for psp_rob helpers."""
    return pd.DataFrame({"PC1": unit[:, 0], "PC2": unit[:, 1], "PC3": unit[:, 2]})


def panel_B(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_B_DEFAULT, config)
    perturb = float(cfg["perturbation_degrees"])
    n_pert = int(cfg["n_perturbations"])
    rows, all_records = [], []
    for ds in cfg["datasets"]:
        unit = common.load_unit_coords_pc13(ds, max_cells=cfg.get("max_cells"),
                                             seed=cfg["seed"])
        if unit is None or unit.shape[0] < 50:
            continue
        df = _build_pca_df_for_robustness(unit)
        rot_res = psp_rob.rotation_robustness_analysis(
            df=df, base_euler_angles=[0.0, 0.0, 0.0],
            perturbation_degrees=perturb, n_perturbations=n_pert,
            seed=cfg["seed"])
        perturbed = np.asarray(rot_res["perturbed_metrics"], dtype=float)
        base = float(rot_res["base_metric"])
        rows.append({
            "dataset": ds,
            "n_cells_used": int(unit.shape[0]),
            "base_metric": base,
            "perturbed_mean": float(perturbed.mean()),
            "perturbed_std": float(perturbed.std(ddof=1)),
            "perturbed_cv": float(perturbed.std(ddof=1) / max(abs(perturbed.mean()), 1e-12)),
            "n_perturbations": n_pert,
            "perturbation_degrees": perturb,
        })
        for v in perturbed:
            all_records.append({"dataset": ds, "kind": "perturbed", "metric": float(v)})
        all_records.append({"dataset": ds, "kind": "base", "metric": base})

    if not rows:
        return common.data_not_present_result(
            "Fig2B", 2, "Fig 2B — rotation robustness",
            "No PC1-3 coordinates available for any dataset.")

    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), facecolor="white")
    positions = np.arange(len(summary))
    perturbed_lists = [
        [r["metric"] for r in all_records
         if r["dataset"] == ds and r["kind"] == "perturbed"]
        for ds in summary["dataset"]
    ]
    parts = ax.violinplot(perturbed_lists, positions=positions,
                          widths=0.78, showmeans=False, showmedians=False,
                          showextrema=False)
    for body, ds in zip(parts["bodies"], summary["dataset"]):
        body.set_facecolor(_ds_color(ds))
        body.set_edgecolor("#333333")
        body.set_alpha(0.55)
        body.set_linewidth(0.6)
    # box-style 25/75
    for x, vals in zip(positions, perturbed_lists):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(x, q1, q3, color="#222", lw=1.6)
        ax.hlines(med, x - 0.18, x + 0.18, color="#fff", lw=1.4)
    # base / manual orientation marker
    for x, ds, base in zip(positions, summary["dataset"], summary["base_metric"]):
        ax.plot(x, base, marker="*", markersize=12, color=_ds_color(ds),
                markeredgecolor="#111", markeredgewidth=0.6, zorder=5)
    # CV labels
    ymax = max(max(v) for v in perturbed_lists)
    ymin = min(min(v) for v in perturbed_lists)
    pad = 0.04 * (ymax - ymin)
    for x, cv in zip(positions, summary["perturbed_cv"]):
        ax.text(x, ymax + pad, f"CV={cv:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#444")

    ax.set_xticks(positions)
    ax.set_xticklabels([_ds_label(d) for d in summary["dataset"]],
                       rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("Global longitude concentration\n(stable structure metric)",
                  fontsize=9.5)
    ax.set_title(f"Rotation perturbations preserve relative structure\n"
                 f"(±{perturb:.0f}° random Euler, n={n_pert} draws)",
                 fontsize=11, pad=6)
    ax.set_ylim(ymin - 4 * pad, ymax + 6 * pad)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=3, labelsize=8.5)
    # legend for star
    ax.plot([], [], marker="*", markersize=12, color="#888",
            markeredgecolor="#111", markeredgewidth=0.6,
            linestyle="None", label="base orientation")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handletextpad=0.3)
    fig.tight_layout(pad=0.35)

    long_records = pd.DataFrame(all_records)
    return {
        "panel_id": "Fig2B", "figure_n": 2, "figure": fig, "config": cfg,
        "data": {"rotation_metrics": long_records, "summary": summary},
    }


# ---------------------------------------------------------------------------
# Panel C — PC-axis specificity heatmap
# ---------------------------------------------------------------------------

PANEL_C_DEFAULT = {
    "datasets": list(common.FIG2_ALL_DATASETS),
    "pc_triplets": [(0, 1, 2), (0, 1, 3), (0, 1, 4),
                    (0, 2, 3), (1, 2, 3), (2, 3, 4)],
    "n_components": 6,
    "n_top_genes": 2000,
    "seed": 0,
}


def _triplet_label(triplet) -> str:
    return "PC" + ",".join(str(i + 1) for i in triplet)


def _planar_anisotropy(unit: np.ndarray) -> float:
    return float(psp_sph.spherical_anisotropy(unit)["planar"])


def _great_circle_concentration(unit: np.ndarray) -> float:
    """Longitude concentration computed AFTER aligning the dataset's best-fit
    great-circle plane to the equator. This depends on all three chosen PCs
    (great-circle plane uses all three) and rewards triplets whose cloud both
    lies near a single plane-through-origin and is organized along it."""
    gc = psp_sph.fit_great_circle(unit)
    n = gc["normal"]
    R = psp_sph._rotation_to_pole(n)
    rotated = unit @ R.T
    return float(psp_sph.stripe_strength_score(rotated)["score"])


def _compute_triplet_metrics(scores: np.ndarray, triplets) -> pd.DataFrame:
    """Per-triplet structure metrics. The headline metric is **planar
    anisotropy** = (λ₂ − λ₃) / λ₁ from the inertia tensor. It is high when
    the unit-projected cloud lies on a great-circle plane (λ₃ small, λ₂
    large) and depends on all three chosen PCs simultaneously, so it
    distinguishes triplets that share PC1+PC2 from triplets that drop them."""
    rows = []
    for t in triplets:
        unit = common.to_unit(scores[:, list(t)])
        rows.append({
            "triplet": _triplet_label(t),
            "structure": _planar_anisotropy(unit),
            "planar_anisotropy": _planar_anisotropy(unit),
            "great_circle_R2": _great_circle_r2(unit),
            "longitude_concentration_pole": _structure_metric(unit),
            "anisotropy_linear": _anisotropy_linear(unit),
            "great_circle_aligned_longitude": _great_circle_concentration(unit),
        })
    return pd.DataFrame(rows)


def panel_C(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_C_DEFAULT, config)
    triplets = [tuple(t) for t in cfg["pc_triplets"]]
    n_components = max(int(cfg["n_components"]), max(max(t) for t in triplets) + 1)

    rows, abs_rows = [], []
    valid_datasets = []
    for ds in cfg["datasets"]:
        r = common.compute_pca_scores(
            ds, n_components=n_components, hvg=True,
            n_top_genes=int(cfg["n_top_genes"]), seed=cfg["seed"])
        if r is None:
            continue
        scores = np.asarray(r["scores"], dtype=float)
        df_t = _compute_triplet_metrics(scores, triplets)
        ref = float(df_t.loc[df_t["triplet"] == _triplet_label(triplets[0]),
                             "structure"].iloc[0])
        ref = max(ref, 1e-9)
        for _, row in df_t.iterrows():
            abs_rows.append({
                "dataset": ds, "triplet": row["triplet"],
                "structure": row["structure"],
                "great_circle_R2": row["great_circle_R2"],
                "anisotropy_linear": row["anisotropy_linear"],
                "structure_relative_to_PC1_3": row["structure"] / ref,
            })
        rows.append({
            "dataset": ds,
            **{r["triplet"]: r["structure"] / ref for _, r in df_t.iterrows()},
        })
        valid_datasets.append(ds)

    if not rows:
        return common.data_not_present_result(
            "Fig2C", 2, "Fig 2C — PC-axis specificity",
            "No raw datasets available for PCA computation.")

    triplet_labels = [_triplet_label(t) for t in triplets]
    heat = pd.DataFrame(rows).set_index("dataset")[triplet_labels]
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(heat) + 2.0),
                           facecolor="white")
    vbound = 1.5
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vbound)
    im = ax.imshow(np.clip(heat.values, 0.0, vbound), aspect="auto",
                   cmap="RdBu_r", norm=norm)
    ax.set_xticks(np.arange(len(triplet_labels)))
    ax.set_xticklabels(triplet_labels, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(heat)))
    ax.set_yticklabels([_ds_label(d) for d in heat.index], fontsize=9)
    # mark reference column
    ref_col = triplet_labels.index(_triplet_label(triplets[0]))
    ax.add_patch(plt.Rectangle((ref_col - 0.5, -0.5), 1,
                               len(heat), fill=False,
                               edgecolor="#111", lw=1.6, zorder=4))
    for (i, j), v in np.ndenumerate(heat.values):
        # annotate with the actual (un-clipped) value
        text = f"{v:.2f}" if v < 9.99 else f"{v:.1f}"
        clipped = float(np.clip(v, 0.0, vbound))
        ax.text(j, i, text, ha="center", va="center",
                fontsize=7.8,
                color="#111" if 0.55 < clipped < 1.45 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02,
                      extend="max")
    cb.set_label("Relative planar anisotropy\n(× PC1–3)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Dominant PC axes carry spherical geometry",
                 fontsize=11.5, fontweight="bold", pad=14)
    ax.text(ref_col, -0.85, "reference", ha="center", va="bottom",
            fontsize=8.2, color="#111", fontweight="bold",
            transform=ax.transData)
    ax.tick_params(width=0.7, length=3)
    fig.tight_layout(pad=0.4)

    return {
        "panel_id": "Fig2C", "figure_n": 2, "figure": fig, "config": cfg,
        "data": {
            "relative_structure_heatmap": heat.reset_index(),
            "absolute_metrics": pd.DataFrame(abs_rows),
        },
    }


# ---------------------------------------------------------------------------
# Panel D — random PC triplet null distribution
# ---------------------------------------------------------------------------

PANEL_D_DEFAULT = {
    "datasets": list(common.FIG2_ALL_DATASETS),
    "n_components": 20,
    "n_top_genes": 2000,
    "n_random_triplets": 200,
    "seed": 0,
}


def panel_D(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_D_DEFAULT, config)
    n_components = int(cfg["n_components"])
    n_random = int(cfg["n_random_triplets"])
    rng = np.random.default_rng(cfg["seed"])
    rows, summary_rows = [], []
    valid = []
    for ds in cfg["datasets"]:
        r = common.compute_pca_scores(
            ds, n_components=n_components, hvg=True,
            n_top_genes=int(cfg["n_top_genes"]), seed=cfg["seed"])
        if r is None:
            continue
        scores = np.asarray(r["scores"], dtype=float)
        n_avail = scores.shape[1]
        if n_avail < 4:
            continue
        observed = _structure_metric(common.to_unit(scores[:, :3]))
        # generate random triplets from PC1..n_avail (1-indexed) excluding (1,2,3)
        all_triplets = [t for t in combinations(range(n_avail), 3)
                        if t != (0, 1, 2)]
        n_take = min(n_random, len(all_triplets))
        chosen_idx = rng.choice(len(all_triplets), size=n_take, replace=False)
        null_vals = []
        for k in chosen_idx:
            t = all_triplets[int(k)]
            null_vals.append(_structure_metric(common.to_unit(scores[:, list(t)])))
        null_vals = np.asarray(null_vals, dtype=float)
        for v in null_vals:
            rows.append({"dataset": ds, "metric": float(v), "kind": "random"})
        rows.append({"dataset": ds, "metric": float(observed), "kind": "observed_PC1_3"})
        z = (observed - null_vals.mean()) / max(null_vals.std(ddof=1), 1e-12)
        p_emp = float((np.sum(null_vals >= observed) + 1) / (n_take + 1))
        summary_rows.append({
            "dataset": ds, "observed_PC1_3": float(observed),
            "null_mean": float(null_vals.mean()),
            "null_std": float(null_vals.std(ddof=1)),
            "z_score": float(z), "p_value_empirical": p_emp,
            "n_random_triplets": int(n_take),
            "n_components_available": int(n_avail),
        })
        valid.append(ds)

    if not summary_rows:
        return common.data_not_present_result(
            "Fig2D", 2, "Fig 2D — random PC null",
            "No raw datasets available for PCA computation.")

    summary = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.0), facecolor="white")
    positions = np.arange(len(valid))
    null_lists = [long_df.loc[(long_df["dataset"] == ds) & (long_df["kind"] == "random"),
                              "metric"].to_numpy() for ds in valid]
    parts = ax.violinplot(null_lists, positions=positions, widths=0.78,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, ds in zip(parts["bodies"], valid):
        body.set_facecolor("#cfd8dc")
        body.set_edgecolor("#37474f")
        body.set_alpha(0.65)
        body.set_linewidth(0.6)
    for x, vals in zip(positions, null_lists):
        if vals.size:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(x, q1, q3, color="#37474f", lw=1.4)
    for x, ds, obs in zip(positions, valid,
                          [r["observed_PC1_3"] for r in summary_rows]):
        ax.plot(x, obs, marker="*", markersize=15, color=_ds_color(ds),
                markeredgecolor="#111", markeredgewidth=0.7, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([_ds_label(d) for d in valid],
                       rotation=22, ha="right", fontsize=8.5)
    ax.set_ylabel("Global longitude concentration", fontsize=9.5)
    ax.set_title(f"PC1–3 exceeds random PC-triplet structure\n"
                 f"({n_random} random triplets sampled from PC1–{n_components})",
                 fontsize=11, pad=6)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=3, labelsize=8.5)
    ax.plot([], [], marker="*", markersize=13, color="#888",
            markeredgecolor="#111", markeredgewidth=0.7,
            linestyle="None", label="observed PC1–3")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#cfd8dc", edgecolor="#37474f",
              label="random triplet null"),
        plt.Line2D([0], [0], marker="*", markersize=13, color="w",
                   markerfacecolor="#888", markeredgecolor="#111",
                   markeredgewidth=0.7, label="observed PC1–3"),
    ], frameon=False, fontsize=8, loc="upper left", handletextpad=0.3,
        bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(pad=0.4)

    return {
        "panel_id": "Fig2D", "figure_n": 2, "figure": fig, "config": cfg,
        "data": {"null_distribution": long_df, "summary": summary},
    }


# ---------------------------------------------------------------------------
# Panel E — canonical Scanpy seurat HVG vs all-gene
# ---------------------------------------------------------------------------

PANEL_E_DEFAULT = {
    "datasets": ["hesc", "pre_implant_human_embryo", "planaria",
                 "human_germ_cell"],
    "n_components": 3,
    "n_top_genes": 2000,
    "seed": 0,
}


def _theta_cytotrace_corr(unit: np.ndarray, expr) -> Optional[float]:
    """Absolute Spearman correlation of theta (polar angle from +z) with
    CytoTRACE. We use |corr| because the sphere has no canonical orientation
    and the sign of theta flips with arbitrary rotations of the great-circle
    plane."""
    if expr is None or expr.metadata is None or expr.metadata.empty:
        return None
    cyto = None
    for col in ("CytoTRACE", "output_CytoTRACE"):
        if col in expr.metadata.columns:
            cyto = expr.metadata[col].to_numpy()
            break
    if cyto is None:
        return None
    cyto = pd.to_numeric(pd.Series(cyto), errors="coerce").to_numpy()
    if unit.shape[0] != cyto.shape[0]:
        return None
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    mask = np.isfinite(cyto)
    if mask.sum() < 10:
        return None
    rho = pd.Series(theta[mask]).corr(pd.Series(cyto[mask]), method="spearman")
    return float(abs(rho))


def panel_E(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_E_DEFAULT, config)
    n_top_genes = int(cfg["n_top_genes"])
    abs_rows, hvg_lists = [], []
    metric_keys = ["structure", "great_circle_R2", "anisotropy_linear",
                   "theta_cytotrace_corr"]

    pre_cfg = []
    scanpy_version = None
    valid = []
    for ds in cfg["datasets"]:
        expr = common.load_raw_expression(ds)
        if expr is None:
            continue
        for hvg_flag, name in [(True, "hvg"), (False, "all")]:
            res = common.compute_pca_scores(
                ds, n_components=int(cfg["n_components"]), hvg=hvg_flag,
                n_top_genes=n_top_genes, seed=cfg["seed"])
            if res is None:
                continue
            scanpy_version = res["scanpy_version"]
            scores = np.asarray(res["scores"], dtype=float)[:, :3]
            unit = common.to_unit(scores)
            row = {
                "dataset": ds, "config": name,
                "n_genes_used": res["n_genes_used"],
                "n_cells_used": res["n_cells_used"],
                "structure": _structure_metric(unit),
                "great_circle_R2": _great_circle_r2(unit),
                "anisotropy_linear": _anisotropy_linear(unit),
                "theta_cytotrace_corr": _theta_cytotrace_corr(unit, expr),
            }
            abs_rows.append(row)
            if hvg_flag and res["hvg_genes"] is not None:
                for g in res["hvg_genes"]:
                    hvg_lists.append({"dataset": ds, "hvg_gene": g})
            pre_cfg.append({
                "dataset": ds, "config": name,
                "scanpy_version": scanpy_version,
                "normalize_total_target_sum": 1e4, "log1p": True,
                "highly_variable_genes_flavor": "seurat" if hvg_flag else None,
                "n_top_genes": n_top_genes if hvg_flag else None,
                "scale_max_value": 10.0,
                "n_components_pca": int(cfg["n_components"]),
                "seed": int(cfg["seed"]),
                "n_genes_used": res["n_genes_used"],
                "n_cells_used": res["n_cells_used"],
            })
        valid.append(ds)

    if not valid:
        return common.data_not_present_result(
            "Fig2E", 2, "Fig 2E — HVG vs all-gene",
            "Required raw RDS datasets not available.")

    abs_df = pd.DataFrame(abs_rows)
    # Build delta per dataset (HVG - all)
    pivot = abs_df.pivot_table(index="dataset", columns="config",
                               values=metric_keys)
    delta_records = []
    for ds in valid:
        row = {"dataset": ds}
        for m in metric_keys:
            try:
                hv = float(pivot.loc[ds, (m, "hvg")])
                al = float(pivot.loc[ds, (m, "all")])
            except KeyError:
                continue
            if np.isnan(hv) or np.isnan(al):
                continue
            row[f"delta_{m}"] = hv - al
            row[f"ratio_{m}"] = hv / al if abs(al) > 1e-9 else np.nan
        delta_records.append(row)
    delta_df = pd.DataFrame(delta_records).set_index("dataset")

    metric_pretty = {
        "delta_structure": "Δ structure",
        "delta_great_circle_R2": "Δ great-circle R²",
        "delta_anisotropy_linear": "Δ anisotropy",
        "delta_theta_cytotrace_corr": "Δ θ–CytoTRACE",
    }
    cols = [c for c in metric_pretty if c in delta_df.columns]
    delta_show = delta_df[cols].copy()
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(delta_show) + 1.9),
                           facecolor="white")
    if delta_show.empty:
        ax.text(0.5, 0.5, "no Δ metrics computed", ha="center", va="center",
                transform=ax.transAxes)
    else:
        bound = float(np.nanmax(np.abs(delta_show.values)))
        bound = max(bound, 0.05)
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        im = ax.imshow(delta_show.values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([metric_pretty[c] for c in cols],
                           rotation=18, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(delta_show)))
        ax.set_yticklabels([_ds_label(d) for d in delta_show.index],
                           fontsize=9)
        for (i, j), v in np.ndenumerate(delta_show.values):
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.8,
                    color="#111" if abs(v) < bound * 0.5 else "white")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("HVG − all-gene", fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    ax.set_title("Feature selection modulates but does not define geometry",
                 fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(width=0.7, length=3)
    fig.tight_layout(pad=0.4)

    # Sidecar files (also written into Fig2/ via output_aliases)
    pre_cfg_yaml_path = (out_root or common.ROOT / "outputs" / "figures") / "Fig2" / \
        "Fig2E_preprocessing_config.yaml"
    pre_cfg_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(pre_cfg_yaml_path, "w") as fh:
        yaml.safe_dump(common._yaml_safe(
            {"scanpy_version": scanpy_version,
             "n_top_genes": n_top_genes,
             "datasets": pre_cfg}), fh, sort_keys=False)
    pd.DataFrame(hvg_lists).to_csv(
        pre_cfg_yaml_path.parent / "Fig2E_HVG_gene_lists.csv", index=False)
    abs_df.to_csv(pre_cfg_yaml_path.parent / "Fig2E_metric_table.csv", index=False)

    return {
        "panel_id": "Fig2E", "figure_n": 2, "figure": fig,
        "config": {**cfg, "scanpy_version": scanpy_version},
        "data": {
            "absolute_metrics": abs_df,
            "delta_metrics": delta_df.reset_index(),
            "hvg_gene_lists": pd.DataFrame(hvg_lists),
        },
    }


# ---------------------------------------------------------------------------
# Panel F — C. elegans equirectangular spherical density / multi-arc
# ---------------------------------------------------------------------------

PANEL_F_DEFAULT = {
    "celegan_alignment": dict(_fig1.PANEL_B_DEFAULT),
    "kde_kappa": 90.0,
    "kde_grid": 5000,
    "kde_max_cells": 12000,
    "lon_bins": 180,
    "lat_bins": 90,
    "n_contour_levels": 7,
    "n_plot_cells": 8000,
    "seed": 0,
}


def _equirectangular_density(unit: np.ndarray, *, kappa: float, m_grid: int,
                             lon_bins: int, lat_bins: int):
    """Evaluate a vMF KDE on a regular lon/lat grid.

    Returns (lon_edges_deg, lat_edges_deg, density_2d).
    """
    lon = np.linspace(-np.pi, np.pi, lon_bins, endpoint=False) + np.pi / lon_bins
    lat = np.linspace(-np.pi / 2, np.pi / 2, lat_bins, endpoint=False) + np.pi / (2 * lat_bins)
    LON, LAT = np.meshgrid(lon, lat)
    eval_xyz = np.column_stack([
        np.cos(LAT.ravel()) * np.cos(LON.ravel()),
        np.cos(LAT.ravel()) * np.sin(LON.ravel()),
        np.sin(LAT.ravel()),
    ])
    density, _ = psp_sph.spherical_kde(unit, kappa=kappa,
                                       eval_points=eval_xyz, m_grid=m_grid)
    grid = density.reshape(lat_bins, lon_bins)
    return np.degrees(lon), np.degrees(lat), grid


def panel_F(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_F_DEFAULT, config)
    rng = np.random.default_rng(cfg["seed"])
    try:
        unit, color, cluster_order = _celegan_unit_fig1b()
    except Exception as exc:
        return common.data_not_present_result(
            "Fig2F", 2, "Fig 2F — C. elegans density",
            f"Could not load celegan in Fig 1B orientation: {exc}")
    lon_deg = np.degrees(np.arctan2(unit[:, 1], unit[:, 0]))
    lat_deg = np.degrees(np.arcsin(np.clip(unit[:, 2], -1.0, 1.0)))

    # KDE memory scales as (N * lon_bins * lat_bins). Subsample cells without
    # bias since vMF KDE is a uniform-weight average over cells.
    kde_max = int(cfg.get("kde_max_cells") or 0)
    if kde_max > 0 and unit.shape[0] > kde_max:
        kde_idx = rng.choice(unit.shape[0], kde_max, replace=False)
        unit_kde = unit[kde_idx]
    else:
        unit_kde = unit

    # KDE surface
    lon_centers, lat_centers, grid = _equirectangular_density(
        unit_kde, kappa=float(cfg["kde_kappa"]),
        m_grid=int(cfg["kde_grid"]),
        lon_bins=int(cfg["lon_bins"]),
        lat_bins=int(cfg["lat_bins"]))

    # Subsample for scatter
    n_plot = int(cfg["n_plot_cells"])
    if unit.shape[0] > n_plot:
        idx = rng.choice(unit.shape[0], n_plot, replace=False)
    else:
        idx = np.arange(unit.shape[0])

    fig, ax = plt.subplots(figsize=(8.7, 4.4), facecolor="white")
    extent = [lon_centers.min(), lon_centers.max(),
              lat_centers.min(), lat_centers.max()]
    im = ax.imshow(grid, extent=extent, origin="lower", aspect="auto",
                   cmap="magma", interpolation="bilinear")
    levels = np.linspace(grid.min(), grid.max(),
                         int(cfg["n_contour_levels"]) + 2)[1:-1]
    cs = ax.contour(lon_centers, lat_centers, grid, levels=levels,
                    colors="#ffffff", linewidths=0.55, alpha=0.75)
    ax.scatter(lon_deg[idx], lat_deg[idx], s=1.8, c="#cccccc",
               alpha=0.18, linewidths=0, rasterized=True)
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("longitude (°)", fontsize=10)
    ax.set_ylabel("latitude (°)", fontsize=10)
    ax.set_title("Local density ridges reveal multi-arc structure",
                 fontsize=11.5, fontweight="bold", pad=6)
    ax.tick_params(width=0.7, length=3, labelsize=8.5)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("vMF KDE density on S²", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.35)

    # Save data
    df_grid = pd.DataFrame(grid, index=np.round(lat_centers, 3),
                            columns=np.round(lon_centers, 3))
    df_grid.index.name = "latitude_deg"
    df_grid.columns.name = "longitude_deg"
    df_grid_long = df_grid.stack().reset_index()
    df_grid_long.columns = ["latitude_deg", "longitude_deg", "density"]
    df_points = pd.DataFrame({
        "longitude_deg": lon_deg, "latitude_deg": lat_deg,
        "embryo_time_order": color,
    })
    return {
        "panel_id": "Fig2F", "figure_n": 2, "figure": fig, "config": cfg,
        "data": {
            "equirectangular_density_grid": df_grid_long,
            "celegan_lonlat": df_points,
        },
    }


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D,
          "E": panel_E, "F": panel_F}
