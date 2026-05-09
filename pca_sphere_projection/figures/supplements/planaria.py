"""Supplements 2 & 3 — Planaria distance-geometry & lineage-sector analyses.

Inputs (raw_data/planaria/):
    Planaria.csv  — PC0..PC49 + 'labels' column (cell type / lineage).

Supplement 2: pairwise *centroid-angular* distance between cell types,
computed in two geometries that share units (radians). For each cell
type we take its centroid direction (a) on S² in 3-D SPHERE-PCA space
(PC1-3, root-aligned and Euler-rotated as in the planaria notebook),
and (b) the unit vector pointing to its mean position in PC1-50 space.
Pairwise cosine similarity is converted to angular distance via
`arccos`. The two heatmaps share row/column ordering for direct visual
comparison.

Supplement 3: lineage angular-sector analysis. For six biologically
related lineage paths rooted at Neoblast 1 (Epidermal, Gut, Muscle,
Parenchymal/Glial, Neural-ChAT, Neural-GABA), we report Spearman
correlation of angular distance from the root with developmental
position, and a permutation test asking whether within-lineage cells
sit closer in S² than randomly drawn label-shuffled sets.

We frame the results conservatively — *lineage-associated angular
organisation* — and avoid causal language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr

from . import common


_PLAN_RAW = common.RAW / "planaria"
_OUT = common.SUPP_ROOT / "Planaria"


# Lineage paths defined by the original `examples/planaria.ipynb` author.
LINEAGES: Dict[str, List[str]] = {
    "Epidermal": [
        "neoblast 1",
        "epidermal neoblasts",
        "early epidermal progenitors",
        "late epidermal progenitors 1",
        "late epidermal progenitors 2",
        "epidermis",
    ],
    "Gut": [
        "neoblast 1",
        "gut progenitors",
        "phagocytes",
    ],
    "Muscle": [
        "neoblast 1",
        "muscle progenitors",
        "muscle body",
        "muscle pharynx",
    ],
    "Parenchymal/Glial": [
        "neoblast 1",
        "parenchymal progenitors",
        "ldlrr-1+ parenchymal cells",
        "pgrn+ parenchymal cells",
        "glia",
    ],
    "Neural (ChAT path)": [
        "neoblast 1",
        "neural progenitors",
        "ChAT neurons 1",
        "ChAT neurons 2",
    ],
    "Neural (GABA path)": [
        "neoblast 1",
        "neural progenitors",
        "cav-1+ neurons",
        "GABA neurons",
    ],
}


# ---------------------------------------------------------------------------
# Data loading and root alignment
# ---------------------------------------------------------------------------

def _load_planaria() -> pd.DataFrame:
    df = pd.read_csv(_PLAN_RAW / "Planaria.csv")
    pc_cols = [str(i) for i in range(50)]
    df = df.rename(columns={c: f"PC{int(c) + 1}" for c in pc_cols})
    df = df.rename(columns={"labels": "celltype"})
    df["celltype"] = df["celltype"].astype(str)
    return df


def _align_to_north_pole(unit: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Rotate so `centroid` (in raw 3D PC space) lands at the north pole."""
    c = np.asarray(centroid, dtype=float)
    c = c / np.linalg.norm(c)
    north = np.array([0.0, 0.0, 1.0])
    axis = np.cross(c, north)
    cos_a = float(np.clip(np.dot(c, north), -1.0, 1.0))
    angle = np.arccos(cos_a)
    if np.isclose(angle, 0.0):
        return unit
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return unit @ R.T


def _prep_geometry(df: pd.DataFrame) -> dict:
    """Root-align PC1-3 to Neoblast 1, then apply the canonical
    `R.from_euler('xyz', [80, 50, 50], degrees=True)` rotation used in
    the planaria notebook so that the lineage sectors land in their
    expected positions on the sphere."""
    pc3 = df[["PC1", "PC2", "PC3"]].to_numpy(dtype=float)
    pc50 = df[[f"PC{i}" for i in range(1, 51)]].to_numpy(dtype=float)
    unit_raw = common.to_unit(pc3)
    root_centroid = pc3[df["celltype"] == "neoblast 1"].mean(axis=0)
    unit_north = _align_to_north_pole(unit_raw, root_centroid)
    # Canonical Euler rotation from raw_data/planaria/planaria.ipynb.
    euler_rotation = R.from_euler("xyz", [80, 50, 50], degrees=True)
    unit_aligned = euler_rotation.apply(unit_north)
    theta = np.arccos(np.clip(unit_aligned[:, 2], -1.0, 1.0))
    phi = np.arctan2(unit_aligned[:, 1], unit_aligned[:, 0])
    return {
        "unit_raw": unit_raw,
        "unit_aligned": unit_aligned,
        "pc50": pc50,
        "theta": theta,
        "phi": phi,
        "root_centroid": root_centroid,
        "euler_rotation_deg": [80, 50, 50],
    }


# ---------------------------------------------------------------------------
# Supplement 2 — centroid-direction angular distance heatmaps
# ---------------------------------------------------------------------------

def _ordering_from_linkage(D: np.ndarray) -> np.ndarray:
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return np.asarray(leaves_list(Z), dtype=int)


def _heatmap(ax, M: np.ndarray, labels: list[str], *,
             cmap, vmin, vmax, title: str):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right",
                       rotation_mode="anchor", fontsize=7.0)
    ax.set_yticklabels(labels, fontsize=7.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10.0, weight="bold", color="#1d2230",
                 pad=8)
    ax.tick_params(length=0, pad=2)
    return im


def _normalised_centroid(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean of the unit-normalised cell vectors, then re-normalised.

    Returns (centroid_direction, norm_before_renormalisation). The
    pre-normalisation norm is a compactness proxy: 1.0 means perfectly
    aligned, 0.0 means opposing directions average out."""
    units = coords / np.linalg.norm(coords, axis=1, keepdims=True).clip(1e-12)
    mean_vec = units.mean(axis=0)
    nrm = float(np.linalg.norm(mean_vec))
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return mean_vec / nrm, nrm


def _centroid_directions(coords: np.ndarray, group_idx: Dict[str, np.ndarray]
                          ) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Stack of unit-norm centroid directions, one per group."""
    keys = list(group_idx.keys())
    cents = np.zeros((len(keys), coords.shape[1]), dtype=float)
    norms = np.zeros(len(keys), dtype=float)
    for i, k in enumerate(keys):
        c, nrm = _normalised_centroid(coords[group_idx[k]])
        cents[i] = c
        norms[i] = nrm
    return cents, keys, norms


def _supplement_2(df: pd.DataFrame, geom: dict,
                  *, registry: list | None,
                  min_cells_per_group: int = 30,
                  seed: int = 0) -> Dict[str, object]:
    """Centroid-direction angular distance heatmaps in two geometries.

    Both heatmaps are in the same units (radians / degrees). Geometry A
    uses 3-D SPHERE-PCA coordinates (root-aligned + Euler rotated);
    geometry B uses the full PC1-50 vector. For each cell type we
    compute the unit centroid direction, then the pairwise cosine
    similarity → arccos → angular distance. This avoids comparing
    angular and Euclidean units."""
    grouped: Dict[str, np.ndarray] = {}
    for ct, sub in df.groupby("celltype"):
        if len(sub) >= min_cells_per_group:
            grouped[ct] = np.asarray(sub.index, dtype=int)

    cents_3d, keys, norms_3d = _centroid_directions(
        geom["unit_aligned"], grouped)
    cents_50, _, norms_50 = _centroid_directions(geom["pc50"], grouped)

    M_ang_3d = common.angular_distance_matrix(cents_3d)
    M_ang_50 = common.angular_distance_matrix(cents_50)

    order = _ordering_from_linkage(M_ang_3d.copy())
    M_3d_o = M_ang_3d[np.ix_(order, order)]
    M_50_o = M_ang_50[np.ix_(order, order)]
    keys_o = [keys[i] for i in order]

    fig = plt.figure(figsize=(19.0, 10.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.34,
                          left=0.16, right=0.985, top=0.78, bottom=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cmap = mpl.cm.viridis
    im_a = _heatmap(ax_a, np.degrees(M_3d_o), keys_o, cmap=cmap,
                    vmin=0, vmax=180,
                    title=("A  Centroid-direction angular distance — "
                           "3-D SPHERE-PCA  (PC1-3, root-aligned)"))
    im_b = _heatmap(ax_b, np.degrees(M_50_o), keys_o, cmap=cmap,
                    vmin=0, vmax=180,
                    title=("B  Centroid-direction angular distance — "
                           "high-dim PC1-50 space"))

    cax_a = fig.add_axes([0.16, 0.12, 0.36, 0.020])
    cax_b = fig.add_axes([0.625, 0.12, 0.36, 0.020])
    cb_a = fig.colorbar(im_a, cax=cax_a, orientation="horizontal")
    cb_a.set_label("centroid-direction angular distance (degrees)",
                   fontsize=8.5, labelpad=5)
    cb_a.ax.tick_params(labelsize=7.5, length=2)
    cb_b = fig.colorbar(im_b, cax=cax_b, orientation="horizontal")
    cb_b.set_label("centroid-direction angular distance (degrees)",
                   fontsize=8.5, labelpad=5)
    cb_b.ax.tick_params(labelsize=7.5, length=2)

    fig.suptitle(
        "Suppl. Fig 2 — Planaria cell-type centroid-direction angular "
        "distances: 3-D SPHERE-PCA vs high-dim PC1-50",
        x=0.16, y=0.965, ha="left",
        fontsize=12, weight="bold", color="#1d2230",
    )
    fig.text(0.16, 0.915,
             "Both panels share units (radians / degrees) and the same "
             "row/column ordering (average-linkage on the 3-D matrix).\n"
             "Per cell type, vectors are unit-normalised before "
             "averaging to obtain a centroid direction; pairwise cosine "
             "similarity → arccos → angular distance.",
             fontsize=8.5, color="#3c3c3c", linespacing=1.35, va="top")

    ang3d_df = pd.DataFrame(M_ang_3d, index=keys, columns=keys)
    ang50_df = pd.DataFrame(M_ang_50, index=keys, columns=keys)
    ang3d_long = ang3d_df.stack().rename(
        "centroid_angular_distance_3d_radians").reset_index()
    ang3d_long.columns = ["row_celltype", "col_celltype",
                          "centroid_angular_distance_3d_radians"]
    ang3d_long["centroid_angular_distance_3d_degrees"] = np.degrees(
        ang3d_long["centroid_angular_distance_3d_radians"])
    ang50_long = ang50_df.stack().rename(
        "centroid_angular_distance_pc50_radians").reset_index()
    ang50_long.columns = ["row_celltype", "col_celltype",
                          "centroid_angular_distance_pc50_radians"]
    ang50_long["centroid_angular_distance_pc50_degrees"] = np.degrees(
        ang50_long["centroid_angular_distance_pc50_radians"])
    long = ang3d_long.merge(ang50_long,
                            on=["row_celltype", "col_celltype"])

    compactness = pd.DataFrame({
        "celltype": keys,
        "n_cells": [int(len(grouped[k])) for k in keys],
        "centroid_norm_3d": norms_3d,
        "centroid_norm_pc50": norms_50,
    })

    config_2 = {
        "module": "supplements.planaria",
        "panel": "SuppFig_Planaria_Angular_vs_PC50_Heatmaps",
        "raw_csv": str((_PLAN_RAW / "Planaria.csv").relative_to(common.ROOT)),
        "min_cells_per_group": min_cells_per_group,
        "n_groups": len(keys),
        "seed": seed,
        "ordering": "average-linkage on the 3-D centroid-angular matrix",
        "aggregator":
            "centroid direction = renormalised mean of unit-normalised "
            "cell vectors per cell type.",
        "distance":
            "arccos(clip(cos_sim(c_i, c_j), -1, 1)) — same units in both "
            "panels.",
        "interpretation_note":
            "Both heatmaps live on S² (angular distance, radians); the "
            "comparison is between low-D (3-D SPHERE-PCA) and high-D "
            "(PC1-50) angular geometry, not between angles and "
            "Euclidean distances. Centroid norm is reported as a "
            "per-cell-type compactness diagnostic (1 = perfectly "
            "aligned vectors).",
    }
    common.save_outputs(
        _OUT, "SuppFig_Planaria_Angular_vs_PC50_Heatmaps",
        fig=fig, config=config_2,
        data={
            "angular_matrix_sphere3d_degrees":
                pd.DataFrame(np.degrees(M_ang_3d),
                             index=keys, columns=keys
                             ).reset_index().rename(
                                 columns={"index": "celltype"}),
            "angular_matrix_pc50angular_degrees":
                pd.DataFrame(np.degrees(M_ang_50),
                             index=keys, columns=keys
                             ).reset_index().rename(
                                 columns={"index": "celltype"}),
            "long_form": long,
            "ordering": pd.DataFrame({"order": keys_o}),
            "centroid_compactness": compactness,
        },
        registry=registry,
    )

    # Canonical short-form CSVs requested in the spec.
    pd.DataFrame(np.degrees(M_ang_3d), index=keys, columns=keys).to_csv(
        _OUT / "planaria_distance_matrix_sphere3d.csv")
    pd.DataFrame(np.degrees(M_ang_50), index=keys, columns=keys).to_csv(
        _OUT / "planaria_distance_matrix_pc50angular.csv")
    if registry is not None:
        registry.append(str(_OUT / "planaria_distance_matrix_sphere3d.csv"))
        registry.append(str(_OUT / "planaria_distance_matrix_pc50angular.csv"))

    return {
        "labels": keys, "M_ang_3d": M_ang_3d, "M_ang_50": M_ang_50,
        "order": order, "compactness": compactness,
    }


# ---------------------------------------------------------------------------
# Supplement 3 — lineage-sector analysis
# ---------------------------------------------------------------------------

def _within_between_distance(unit: np.ndarray, group_idx: Dict[str, np.ndarray],
                             *, max_per_group: int = 250,
                             seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, idx in group_idx.items():
        if len(idx) > max_per_group:
            idx = rng.choice(idx, max_per_group, replace=False)
        u = unit[idx]
        g_in = u @ u.T
        g_in = np.clip(g_in, -1.0, 1.0)
        ang_in = np.arccos(g_in)
        iu = np.triu_indices_from(ang_in, k=1)
        ang_in_vals = ang_in[iu]

        # Between distances: vs random sample from "outside the lineage"
        all_idx = np.arange(unit.shape[0])
        outside = np.setdiff1d(all_idx, idx, assume_unique=False)
        if len(outside) > max_per_group:
            outside = rng.choice(outside, max_per_group, replace=False)
        u_out = unit[outside]
        g_bw = u @ u_out.T
        g_bw = np.clip(g_bw, -1.0, 1.0)
        ang_bw = np.arccos(g_bw).ravel()

        rows.extend([
            {"lineage": name, "group": "within",  "angular_distance": float(v)}
            for v in ang_in_vals
        ])
        rows.extend([
            {"lineage": name, "group": "between", "angular_distance": float(v)}
            for v in ang_bw
        ])
    return pd.DataFrame(rows)


def _lineage_compactness_permutation(unit: np.ndarray, df: pd.DataFrame,
                                      lineages: Dict[str, List[str]],
                                      *, n_permutations: int = 1000,
                                      max_per_group: int = 250,
                                      seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cell_idx_by_label = {ct: np.where(df["celltype"].values == ct)[0]
                         for ct in df["celltype"].unique()}

    def _median_within(idx_set: np.ndarray) -> float:
        if len(idx_set) > max_per_group:
            idx_set = rng.choice(idx_set, max_per_group, replace=False)
        u = unit[idx_set]
        g = np.clip(u @ u.T, -1.0, 1.0)
        ang = np.arccos(g)
        iu = np.triu_indices_from(ang, k=1)
        return float(np.median(ang[iu]))

    for name, types in lineages.items():
        idx = np.concatenate(
            [cell_idx_by_label.get(t, np.array([], dtype=int)) for t in types]
        )
        if len(idx) < 5:
            continue
        observed = _median_within(idx)
        n_size = len(idx)
        all_idx = np.arange(unit.shape[0])
        null_vals = np.empty(n_permutations, dtype=float)
        for k in range(n_permutations):
            rand_idx = rng.choice(all_idx, n_size, replace=False)
            null_vals[k] = _median_within(rand_idx)
        p_emp = float((null_vals <= observed).mean() + 1.0 / (n_permutations + 1))
        rows.append({
            "lineage": name,
            "n_cells": int(n_size),
            "observed_median_within_angular_dist": observed,
            "null_median_mean": float(np.mean(null_vals)),
            "null_median_std": float(np.std(null_vals)),
            "empirical_p_more_compact_than_null": min(1.0, p_emp),
            "n_permutations": n_permutations,
        })
    return pd.DataFrame(rows)


def _lineage_pseudotime_correlation(theta: np.ndarray, df: pd.DataFrame,
                                     lineages: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for name, types in lineages.items():
        positions = {t: i for i, t in enumerate(types)}
        mask = df["celltype"].isin(types).values
        if mask.sum() < 5:
            continue
        pos = df.loc[mask, "celltype"].map(positions).to_numpy(dtype=float)
        th = theta[mask]
        rho, p = spearmanr(pos, th)
        rows.append({
            "lineage": name,
            "n_cells": int(mask.sum()),
            "spearman_rho_pseudoorder_vs_theta": float(rho),
            "spearman_p": float(p),
            "lineage_path": " → ".join(types),
        })
    return pd.DataFrame(rows)


def _supplement_3_visualization(df: pd.DataFrame, geom: dict,
                                lineages: Dict[str, List[str]]) -> tuple[plt.Figure, pd.DataFrame]:
    """3D sphere + equirectangular sector view, lineages colour-coded."""
    fig = plt.figure(figsize=(12.0, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25],
                          left=0.04, right=0.985, top=0.85, bottom=0.18,
                          wspace=0.18)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    common.draw_3d_sphere_backdrop(ax3d, color="#e8edf2", wire="#aab3bf",
                                   alpha_surface=0.10, alpha_wire=0.20)

    ax_eq = fig.add_subplot(gs[0, 1])

    # Background grey: all cells.
    unit = geom["unit_aligned"]
    rng = np.random.default_rng(0)
    if unit.shape[0] > 8000:
        bg = rng.choice(unit.shape[0], 8000, replace=False)
    else:
        bg = np.arange(unit.shape[0])
    ax3d.scatter(unit[bg, 0], unit[bg, 1], unit[bg, 2],
                 c="#cccccc", s=1.0, alpha=0.25, linewidths=0,
                 depthshade=False)

    cmap = mpl.colormaps.get_cmap("tab10")
    palette_rows = []
    for li, (name, types) in enumerate(lineages.items()):
        col = cmap(li % 10)
        mask = df["celltype"].isin(types).values
        idx = np.where(mask)[0]
        if len(idx) > 4000:
            idx = rng.choice(idx, 4000, replace=False)
        ax3d.scatter(unit[idx, 0], unit[idx, 1], unit[idx, 2],
                     c=[col], s=2.4, alpha=0.85, linewidths=0,
                     depthshade=False, label=name)

        lon = np.degrees(np.arctan2(unit[idx, 1], unit[idx, 0]))
        lat = np.degrees(np.arcsin(np.clip(unit[idx, 2], -1.0, 1.0)))
        ax_eq.scatter(lon, lat, c=[col], s=2.0, alpha=0.65,
                      linewidths=0, label=name)
        palette_rows.append({"lineage": name, "color_hex":
                             mpl.colors.to_hex(col)})

    common.style_clean_3d(ax3d)
    ax3d.view_init(elev=20, azim=-60)
    ax3d.set_title("A  Sphere view (root-aligned)",
                   fontsize=10.5, weight="bold", color="#1d2230", pad=2)

    ax_eq.set_xlim(-180, 180); ax_eq.set_ylim(-90, 90)
    ax_eq.set_xlabel("Longitude (°)", fontsize=9)
    ax_eq.set_ylabel("Latitude (°)", fontsize=9)
    ax_eq.set_xticks(np.arange(-180, 181, 60))
    ax_eq.set_yticks(np.arange(-90, 91, 30))
    ax_eq.grid(True, color="#dadbde", linewidth=0.4, alpha=0.8)
    ax_eq.set_axisbelow(True)
    ax_eq.set_title("B  Equirectangular sectors",
                    fontsize=10.5, weight="bold", color="#1d2230", pad=2)
    ax_eq.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42),
                 ncol=3, fontsize=8, frameon=False, columnspacing=1.0,
                 handletextpad=0.3)

    fig.suptitle(
        "Suppl. Fig 3 — Planaria lineage-associated angular sectors  "
        "(Neoblast 1 root, S²)",
        x=0.04, y=0.97, ha="left",
        fontsize=12, weight="bold", color="#1d2230",
    )
    fig.text(0.04, 0.93,
             "Lineages occupy partially separable angular sectors.\n"
             "We frame this as lineage-associated angular organisation, "
             "not causal lineage determination.",
             fontsize=8.5, color="#3c3c3c", linespacing=1.35, va="top")
    return fig, pd.DataFrame(palette_rows)


def _supplement_3_statplot(within_between: pd.DataFrame,
                            corr: pd.DataFrame,
                            permutation: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14.2, 5.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.2, 1.2],
                          left=0.075, right=0.97, top=0.78, bottom=0.30,
                          wspace=0.55)

    ax_v = fig.add_subplot(gs[0, 0])
    lineages = list(within_between["lineage"].unique())
    cmap = mpl.colormaps.get_cmap("tab10")
    positions = []
    box_data = []
    box_colors = []
    labels = []
    for li, ln in enumerate(lineages):
        for j, grp in enumerate(["within", "between"]):
            vals = within_between.loc[
                (within_between["lineage"] == ln) &
                (within_between["group"] == grp),
                "angular_distance",
            ].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            positions.append(li * 2.4 + j * 0.9)
            box_data.append(vals)
            box_colors.append(cmap(li % 10) if grp == "within" else "#bbbbbb")
            labels.append(f"{ln}\n{grp}")
    bp = ax_v.boxplot(box_data, positions=positions, widths=0.7,
                      patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c); patch.set_edgecolor("#1d2230")
        patch.set_linewidth(0.6); patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("#1d2230"); med.set_linewidth(0.9)
    for whisker in bp["whiskers"] + bp["caps"]:
        whisker.set_color("#1d2230"); whisker.set_linewidth(0.5)
    ax_v.set_xticks([li * 2.4 + 0.45 for li in range(len(lineages))])
    ax_v.set_xticklabels(lineages, rotation=30, ha="right",
                         rotation_mode="anchor", fontsize=8.2)
    ax_v.set_ylabel("Angular distance (radians)", fontsize=9, labelpad=7)
    ax_v.set_title("A  Within-lineage vs between-lineage angular distance",
                   fontsize=10.5, weight="bold", color="#1d2230", pad=4)
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#888", alpha=0.85, label="within (lineage)"),
        Patch(facecolor="#bbb", alpha=0.85, label="between (lineage vs rest)"),
    ]
    ax_v.legend(handles=legend, fontsize=8, frameon=False, loc="lower left",
                bbox_to_anchor=(0.0, 1.03), ncol=2,
                columnspacing=1.2, handletextpad=0.4)

    ax_c = fig.add_subplot(gs[0, 1])
    if len(corr):
        ax_c.barh(range(len(corr)), corr["spearman_rho_pseudoorder_vs_theta"],
                  color=[cmap(i % 10) for i in range(len(corr))],
                  edgecolor="#1d2230", linewidth=0.6)
        ax_c.set_yticks(range(len(corr)))
        ax_c.set_yticklabels(corr["lineage"], fontsize=8.5)
        ax_c.axvline(0, color="#1d2230", linewidth=0.6)
        ax_c.set_xlabel("Spearman ρ (pseudo-order ↔ θ)",
                        fontsize=9, labelpad=8)
    ax_c.set_title("B  Lineage pseudo-order correlation",
                   fontsize=10.5, weight="bold", color="#1d2230", pad=4)
    ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)

    ax_p = fig.add_subplot(gs[0, 2])
    if len(permutation):
        # negative log10 p, capped.
        p = permutation["empirical_p_more_compact_than_null"].clip(
            lower=1.0 / (permutation["n_permutations"].iloc[0] + 1.0))
        nlog = -np.log10(p)
        ax_p.barh(range(len(permutation)), nlog,
                  color=[cmap(i % 10) for i in range(len(permutation))],
                  edgecolor="#1d2230", linewidth=0.6)
        ax_p.set_yticks(range(len(permutation)))
        ax_p.set_yticklabels(permutation["lineage"], fontsize=8.5)
        ax_p.axvline(-np.log10(0.05), color="#d1495b",
                     linestyle="--", linewidth=0.7, label="p = 0.05")
        ax_p.set_xlabel("−log₁₀ empirical p  "
                         "(within-lineage compactness)",
                         fontsize=8.7, labelpad=8)
        ax_p.legend(fontsize=8, frameon=False, loc="lower right",
                    bbox_to_anchor=(1.0, 1.02))
    ax_p.set_title("C  Lineage-compactness permutation test",
                   fontsize=10.5, weight="bold", color="#1d2230", pad=4)
    ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)

    fig.suptitle(
        "Suppl. Fig 3 (stats) — Lineage-associated angular structure  "
        "(planaria, root = Neoblast 1)",
        x=0.075, y=0.97, ha="left",
        fontsize=12, weight="bold", color="#1d2230",
    )
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(*, registry: list | None = None,
        n_permutations: int = 1000, seed: int = 0) -> Dict[str, Path]:
    common.apply_publication_style()
    out = _OUT
    out.mkdir(parents=True, exist_ok=True)

    df = _load_planaria()
    geom = _prep_geometry(df)

    # ---- Supplement 2 ----
    s2 = _supplement_2(df, geom, registry=registry, seed=seed)

    # ---- Supplement 3 ----
    fig_s3, palette = _supplement_3_visualization(df, geom, LINEAGES)
    common.save_outputs(
        out, "SuppFig_Planaria_Lineage_Sectors",
        fig=fig_s3,
        config={
            "module": "supplements.planaria",
            "root_label": "neoblast 1",
            "lineages": LINEAGES,
            "interpretation_note":
                "Lineage-associated angular organisation, not causal "
                "lineage determination.",
        },
        data={"palette": palette},
        registry=registry,
    )

    # Statistical summary
    group_idx = {name: np.concatenate(
        [np.where(df["celltype"].values == ct)[0] for ct in types]
    ) for name, types in LINEAGES.items()}
    within_between = _within_between_distance(
        geom["unit_aligned"], group_idx, seed=seed)
    corr = _lineage_pseudotime_correlation(geom["theta"], df, LINEAGES)
    permutation = _lineage_compactness_permutation(
        geom["unit_aligned"], df, LINEAGES,
        n_permutations=n_permutations, seed=seed,
    )

    fig_stats = _supplement_3_statplot(within_between, corr, permutation)
    config_3s = {
        "module": "supplements.planaria",
        "root_label": "neoblast 1",
        "lineages": LINEAGES,
        "n_permutations": n_permutations,
        "max_per_group_for_within_dist": 250,
        "primary_statistic":
            "Spearman correlation (monotone) of pseudo-order with theta.",
        "secondary_statistic":
            "Empirical p-value from a label-shuffling test on within-"
            "lineage median angular distance.",
        "seed": seed,
        "interpretation_note":
            "Significant compactness indicates lineages are non-randomly "
            "organised in S²; this is descriptive of geometry, not a "
            "lineage-determination claim.",
    }
    common.save_outputs(
        out, "SuppFig_Planaria_Lineage_Statistics",
        fig=fig_stats, config=config_3s,
        data={
            "within_between": within_between,
            "spearman_correlation": corr,
            "permutation_test": permutation,
        },
        registry=registry,
    )

    # Canonical short-form CSV
    summary = corr.merge(
        permutation, on="lineage", suffixes=("_corr", "_perm"), how="outer",
    )
    summary.to_csv(out / "planaria_lineage_statistics.csv", index=False)
    if registry is not None:
        registry.append(str(out / "planaria_lineage_statistics.csv"))
    return {"output_dir": out, "supp2": s2, "n_lineages": len(LINEAGES)}
