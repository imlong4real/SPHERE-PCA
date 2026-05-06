"""Figure 2 — geometry validation.

Panels:
    A: great-circle fit overlay (celegan + klein)
    B: rotation robustness across datasets
    C: PC robustness (varying which 3 PCs we use, on klein)
    D: random-PC control comparison
    E: HVG vs all-gene PCA comparison
    F: multi-stripe detection (KDE + longitudinal histogram on klein)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import io as psp_io
from .. import preprocessing as psp_pp
from .. import sphere_stats as psp_sph
from .. import robustness as psp_rob
from . import common


# ---------------------------------------------------------------------------
# Panel A — great-circle fit
# ---------------------------------------------------------------------------

PANEL_A_DEFAULT = {
    "datasets": ["celegan", "klein"],
    "n_plot_cells": 5000,
    "circle_resolution": 200,
    "seed": 0,
}


def panel_A(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_A_DEFAULT, config)
    rng = np.random.default_rng(cfg["seed"])
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    fig, axes = plt.subplots(1, len(cfg["datasets"]), figsize=(5 * len(cfg["datasets"]), 4.5),
                             subplot_kw={"projection": "3d"})
    if len(cfg["datasets"]) == 1: axes = [axes]
    rows = []
    for ax, ds in zip(axes, cfg["datasets"]):
        art = common.load_artifacts(val_out, ds)
        if art is None:
            ax.text(0.5, 0.5, f"{ds}: artifact missing", transform=ax.transAxes,
                    ha="center")
            continue
        unit = art["coords3_unit"]
        gc = psp_sph.fit_great_circle(unit)
        n = gc["normal"]
        if abs(n[2]) < 0.9:
            u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u)
        else:
            u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        t = np.linspace(0, 2 * np.pi, cfg["circle_resolution"])
        circ = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
        idx = rng.choice(unit.shape[0], min(unit.shape[0], cfg["n_plot_cells"]),
                         replace=False)
        ax.scatter(unit[idx, 0], unit[idx, 1], unit[idx, 2],
                   c=common.DATASET_COLORS.get(ds, "#444"), s=2, alpha=0.4)
        ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="#e45756", lw=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{ds}\nR² = {gc['r_squared']:.3f}")
        rows.append({
            "dataset": ds,
            "great_circle_R2": float(gc["r_squared"]),
            "normal_x": float(n[0]), "normal_y": float(n[1]), "normal_z": float(n[2]),
            "mean_residual_radians": float(gc["mean_residual_radians"]),
        })
    fig.suptitle("Fig 2A. Great-circle fit on the spherical embedding", y=1.02)
    fig.tight_layout()
    return {"panel_id": "Fig2A", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"great_circle_fits": pd.DataFrame(rows)}}


# ---------------------------------------------------------------------------
# Panel B — rotation robustness across datasets (uses example PC CSVs)
# ---------------------------------------------------------------------------

PANEL_B_DEFAULT = {
    "datasets": ["celegan", "klein", "uc_epi"],
    "base_euler_angles": [0.0, 0.0, 0.0],
    "perturbation_degrees": 20.0,
    "n_perturbations": 150,
    "seed": 0,
}


def panel_B(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_B_DEFAULT, config)
    fig, ax = plt.subplots(figsize=(8, 4))
    rows, summary_rows = [], []
    for ds in cfg["datasets"]:
        df = common.load_pca_csv(ds)
        if df is None:
            continue
        rot_res = psp_rob.rotation_robustness_analysis(
            df=df, base_euler_angles=cfg["base_euler_angles"],
            perturbation_degrees=cfg["perturbation_degrees"],
            n_perturbations=cfg["n_perturbations"], seed=cfg["seed"])
        rows.append((ds, np.asarray(rot_res["perturbed_metrics"]),
                     float(rot_res["base_metric"])))
        for v in rot_res["perturbed_metrics"]:
            summary_rows.append({"dataset": ds, "kind": "perturbed", "metric": float(v)})
        summary_rows.append({"dataset": ds, "kind": "base", "metric": float(rot_res["base_metric"])})
    if not rows:
        return common.data_not_present_result(
            "Fig2B", 2, "Fig 2B", "No PCA CSVs found.")
    positions = np.arange(len(rows))
    ax.boxplot([r[1] for r in rows], positions=positions, widths=0.5,
               patch_artist=True, boxprops=dict(facecolor="#dddddd"))
    for i, (ds, _, base) in enumerate(rows):
        ax.plot(i, base, marker="*", markersize=14, linestyle="None",
                color=common.DATASET_COLORS.get(ds, "#444"),
                markeredgecolor="black", label=f"{ds} base")
    ax.set_xticks(positions); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel(f"stripe_strength_score under ±{cfg['perturbation_degrees']:.0f}° rotations")
    ax.set_title("Fig 2B. Rotation robustness")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return {"panel_id": "Fig2B", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"rotation_metrics": pd.DataFrame(summary_rows)}}


# ---------------------------------------------------------------------------
# Panel C — PC robustness (vary which 3 PCs we use)
# ---------------------------------------------------------------------------

PANEL_C_DEFAULT = {
    "dataset": "klein",
    "n_components_total": 5,
    "n_top_genes": 2000,
    "pc_triples": [[0, 1, 2], [0, 1, 3], [0, 1, 4], [1, 2, 3], [2, 3, 4]],
    "seed": 0,
}


def panel_C(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_C_DEFAULT, config)
    if cfg["dataset"] != "klein":
        # simple, self-contained: only klein implemented to keep runtime tight
        return common.data_not_present_result(
            "Fig2C", 2, "Fig 2C — PC robustness",
            "PC robustness is implemented for klein only.")
    expr = psp_io.load_bz2_klein_dataset(str(common.ROOT / "raw_data" / "klein"))
    X_norm = psp_pp.normalize_log1p(expr.X)
    idx, hvg_names, X_hvg = psp_pp.select_hvgs(X_norm, expr.gene_names, cfg["n_top_genes"])
    pca = psp_pp.fit_pca_embedding(X_hvg, hvg_names, n_components=cfg["n_components_total"])
    rows = []
    for triple in cfg["pc_triples"]:
        scores = pca.scores[:, list(triple)]
        unit = common.to_unit(scores)
        ss = psp_sph.stripe_strength_score(unit)["score"]
        gc = psp_sph.fit_great_circle(unit)["r_squared"]
        ani = psp_sph.spherical_anisotropy(unit)["linear"]
        rows.append({
            "PCs": "PC" + ",PC".join(str(i + 1) for i in triple),
            "stripe_strength": ss, "great_circle_R2": gc,
            "anisotropy_linear": ani,
        })
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df)); width = 0.27
    ax.bar(x - width, df["stripe_strength"], width, label="stripe_strength", color="#4c78a8")
    ax.bar(x, df["great_circle_R2"], width, label="great_circle_R²", color="#e45756")
    ax.bar(x + width, df["anisotropy_linear"], width, label="anisotropy_linear", color="#54a24b")
    ax.set_xticks(x); ax.set_xticklabels(df["PCs"], rotation=15)
    ax.set_ylabel("metric")
    ax.set_title("Fig 2C. PC-choice robustness (Klein)")
    ax.legend()
    fig.tight_layout()
    return {"panel_id": "Fig2C", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"pc_choice_metrics": df}}


# ---------------------------------------------------------------------------
# Panel D — random-PC control comparison
# ---------------------------------------------------------------------------

PANEL_D_DEFAULT = {
    "datasets": ["celegan", "uc_epi", "klein", "hesc"],
    "metric": "stripe_strength",
    "configs_in_order": ["hvg", "all", "no_mito_ribo", "random_match"],
    "seed": 0,
}


def panel_D(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_D_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    rows = []
    for ds in cfg["datasets"]:
        p = val_out / ds / "pca_comparison_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d["dataset"] = ds
        rows.append(d)
    if not rows:
        return common.data_not_present_result(
            "Fig2D", 2, "Fig 2D", "No pca_comparison_metrics found.")
    df_all = pd.concat(rows, ignore_index=True)
    pivot = df_all.pivot_table(index="dataset", columns="config", values=cfg["metric"])
    cfgs = [c for c in cfg["configs_in_order"] if c in pivot.columns]
    pivot = pivot[cfgs]
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
    ax.set_ylabel(f"PC1-3 {cfg['metric']}")
    ax.set_title("Fig 2D. Real PCA vs random-gene-set control")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="gene set", fontsize=8)
    fig.tight_layout()
    return {"panel_id": "Fig2D", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"metric_table": df_all}}


# ---------------------------------------------------------------------------
# Panel E — HVG vs all-gene PCA
# ---------------------------------------------------------------------------

PANEL_E_DEFAULT = {
    "datasets": ["celegan", "uc_epi", "klein", "hesc"],
    "metrics": ["explained_var_PC1", "stripe_strength"],
    "seed": 0,
}


def panel_E(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_E_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    rows = []
    for ds in cfg["datasets"]:
        p = val_out / ds / "pca_comparison_metrics.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d["dataset"] = ds
        rows.append(d)
    if not rows:
        return common.data_not_present_result(
            "Fig2E", 2, "Fig 2E", "No pca_comparison_metrics found.")
    df_all = pd.concat(rows, ignore_index=True)
    df_use = df_all[df_all["config"].isin(["hvg", "all"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, cfg["metrics"]):
        pivot = df_use.pivot_table(index="dataset", columns="config", values=m)
        pivot[["hvg", "all"]].plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.4)
        ax.set_ylabel(m)
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="gene set", fontsize=8)
    fig.suptitle("Fig 2E. HVG vs all-gene PCA on the spherical embedding", y=1.02)
    fig.tight_layout()
    return {"panel_id": "Fig2E", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"hvg_vs_all": df_use}}


# ---------------------------------------------------------------------------
# Panel F — multi-stripe detection (KDE + histogram)
# ---------------------------------------------------------------------------

PANEL_F_DEFAULT = {
    "dataset": "klein",
    "kde_kappa": 80.0,
    "kde_grid": 2000,
    "stripe_n_bins": 36,
    "seed": 0,
}


def panel_F(config: Optional[dict] = None, out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_F_DEFAULT, config)
    val_out = (out_root.parent if out_root else common.ROOT / "outputs") / "raw_expression_validation"
    art = common.load_artifacts(val_out, cfg["dataset"])
    if art is None:
        return common.data_not_present_result(
            "Fig2F", 2, "Fig 2F", f"{cfg['dataset']} artifacts missing.")
    unit = art["coords3_unit"]
    density, eval_pts = psp_sph.spherical_kde(
        unit, kappa=cfg["kde_kappa"], m_grid=cfg["kde_grid"])
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax1.scatter(eval_pts[:, 0], eval_pts[:, 1], eval_pts[:, 2],
                     c=density, s=8, cmap="magma")
    ax1.set_title(f"Spherical vMF-KDE ({cfg['dataset']})")
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
    fig.colorbar(sc, ax=ax1, fraction=0.04, pad=0.05, label="density")

    ax2 = fig.add_subplot(1, 2, 2)
    ss = psp_sph.stripe_strength_score(unit, n_bins=cfg["stripe_n_bins"])
    counts = ss["bin_counts"]
    centers = np.linspace(-180, 180, len(counts) + 1)[:-1] + (360 / len(counts) / 2)
    ax2.bar(centers, counts, width=360 / len(counts) * 0.9,
            color=common.DATASET_COLORS.get(cfg["dataset"], "#444"),
            edgecolor="black", linewidth=0.3)
    ax2.set_xlabel("longitude (°)")
    ax2.set_ylabel(f"# cells per {360 // cfg['stripe_n_bins']}° stripe")
    ax2.set_title(f"Stripe histogram (score={ss['score']:.3f})")
    fig.suptitle("Fig 2F. Multi-stripe structure on the embedding", y=1.02)
    fig.tight_layout()
    df_kde = pd.DataFrame({"x": eval_pts[:, 0], "y": eval_pts[:, 1],
                           "z": eval_pts[:, 2], "density": density})
    df_hist = pd.DataFrame({"longitude_center_deg": centers, "count": counts,
                            "stripe_strength_score": [ss["score"]] * len(counts)})
    return {"panel_id": "Fig2F", "figure_n": 2, "figure": fig, "config": cfg,
            "data": {"vmf_kde_grid": df_kde, "stripe_histogram": df_hist}}


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D,
          "E": panel_E, "F": panel_F}
