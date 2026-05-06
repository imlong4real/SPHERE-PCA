"""
Publication-quality figure generation for the raw-expression validation outputs.

Reads each dataset's `_artifacts.npz`, `per_cell_geometry.csv`, and the
hypothesis CSVs written by ``run_raw_expression_validation.py``, then writes
PNGs into ``<dataset>/figures/``.

Run with:

    python scripts/plot_raw_validation_figures.py \
        --out outputs/raw_expression_validation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pca_sphere_projection import perturbation as psp_pert  # noqa: E402
from pca_sphere_projection import io as psp_io  # noqa: E402
from pca_sphere_projection import preprocessing as psp_pp  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _xy_from_unit(coords3):
    x, y, z = coords3[:, 0], coords3[:, 1], coords3[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon, lat


def fig_h1(ds_name, ds_dir: Path, fig_dir: Path):
    art = np.load(ds_dir / "_artifacts.npz")
    coords3 = art["coords3_unit"]
    s = art["entropy_score"]
    x, y, z = coords3[:, 0], coords3[:, 1], coords3[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))

    geom = pd.read_csv(ds_dir / "per_cell_geometry.csv")
    h1 = pd.read_csv(ds_dir / "H1_entropy_gradient_metrics.csv").iloc[0]
    method = h1.get("score_method", "")

    fig = plt.figure(figsize=(13, 4.2))

    # 3D sphere
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax3d.scatter(x, y, z, c=s, s=2, cmap="magma", alpha=0.6)
    ax3d.set_title(f"{ds_name}: entropy on S^2")
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    plt.colorbar(sc, ax=ax3d, fraction=0.04, pad=0.02, label=method[:30])

    # Equirect
    ax2 = fig.add_subplot(1, 3, 2)
    sc2 = ax2.scatter(lon, lat, c=s, s=2, cmap="magma", alpha=0.6)
    ax2.set_xlim(-180, 180); ax2.set_ylim(-90, 90)
    ax2.set_xlabel("longitude (deg)"); ax2.set_ylabel("latitude (deg)")
    ax2.set_title("equirectangular")
    plt.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02)

    # Entropy vs geodesic
    ax3 = fig.add_subplot(1, 3, 3)
    g = geom["theta"].values  # geodesic from north pole; close to anchor for stem-like
    ax3.scatter(g, s, s=2, alpha=0.4)
    ax3.set_xlabel("theta (geodesic from north pole, rad)")
    ax3.set_ylabel("entropy / stemness score")
    rho = h1.get("geodesic_spearman_rho", float("nan"))
    ax3.set_title(f"rho_geo = {float(rho):.3f}")
    fig.suptitle(f"H1: entropy gradient — {ds_name} ({method})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "H1_entropy_gradient.png", bbox_inches="tight")
    plt.close(fig)


def fig_h4(ds_name, ds_dir: Path, fig_dir: Path):
    geom = pd.read_csv(ds_dir / "per_cell_geometry.csv")
    h4 = pd.read_csv(ds_dir / "H4_radial_angular_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(geom["radial_norm"], geom["theta"], s=2, alpha=0.4)
    ax.set_xlabel("radial norm (||PC1-3||)"); ax.set_ylabel("theta (rad)")
    ax.set_title("radial vs angular (per cell)")

    ax2 = axes[1]
    if not h4.empty:
        sub = h4.copy()
        sub["abs_rho"] = sub["spearman_rho"].abs()
        order = sub.sort_values("abs_rho", ascending=False)["variable"].drop_duplicates().tolist()
        x = np.arange(len(order))
        rad_vals = [sub[(sub["variable"] == v) & (sub["vs"] == "radial")]["spearman_rho"].mean() for v in order]
        the_vals = [sub[(sub["variable"] == v) & (sub["vs"] == "theta")]["spearman_rho"].mean() for v in order]
        w = 0.4
        ax2.bar(x - w/2, rad_vals, width=w, label="radial", color="#1f77b4")
        ax2.bar(x + w/2, the_vals, width=w, label="theta", color="#ff7f0e")
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xticks(x); ax2.set_xticklabels(order, rotation=30, ha="right")
        ax2.set_ylabel("Spearman rho"); ax2.legend()
        ax2.set_title("H4: which biology lives where?")
    fig.suptitle(f"H4 radial vs angular — {ds_name}", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "H4_radial_vs_angular.png", bbox_inches="tight")
    plt.close(fig)


def fig_h5(ds_name, ds_dir: Path, fig_dir: Path):
    boundary = pd.read_csv(ds_dir / "H5_stripe_boundary_genes.csv")
    along = pd.read_csv(ds_dir / "H5_along_trajectory_genes.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    if not boundary.empty:
        sub = boundary.head(20).iloc[::-1]
        ax = axes[0]
        col = ["#d62728" if k else "#1f77b4" for k in sub["is_known_regulator"]]
        ax.barh(sub["gene"], sub["stripe_score"], color=col)
        ax.set_xlabel("stripe_score = phi_var / theta_var")
        ax.set_title("Top stripe-boundary genes (red = known regulator)")
    if not along.empty:
        sub = along.head(20).iloc[::-1]
        ax = axes[1]
        col = ["#d62728" if k else "#2ca02c" for k in sub["is_known_regulator"]]
        ax.barh(sub["gene"], sub["trajectory_score"], color=col)
        ax.set_xlabel("trajectory_score = theta_var / phi_var")
        ax.set_title("Top along-trajectory genes")
    fig.suptitle(f"H5 gene gradient ranking — {ds_name}", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "H5_gene_gradient.png", bbox_inches="tight")
    plt.close(fig)


def fig_h7(ds_name, ds_dir: Path, fig_dir: Path, raw_data_root: Path):
    h7 = pd.read_csv(ds_dir / "H7_perturbation_gene_rankings.csv")
    if h7.empty:
        return
    # Pick top 5 for plotting; we need to recompute the perturbation for them
    # to get the displacement vectors. To avoid re-running the heavy loaders
    # we read the artefacts and the per-cell geometry, and reload expression.
    art = np.load(ds_dir / "_artifacts.npz")
    # H7 needs loadings on the *full* gene space (perturbation modifies
    # one gene column, which only exists in the full matrix).
    if "pca_components_full" in art.files:
        coords3_unit = art["coords3_unit_full"]
        pca_components = art["pca_components_full"]
        pca_mean = art["pca_mean_full"]
    else:
        # back-compat for runs from before the fix
        coords3_unit = art["coords3_unit"]
        pca_components = art["pca_components"]
        pca_mean = art["pca_mean"]

    expr = _reload_expression(ds_name, raw_data_root)
    if expr is None:
        return

    # Match expr cells to per_cell_geometry to ensure ordering is consistent
    geom = pd.read_csv(ds_dir / "per_cell_geometry.csv")
    if expr.n_cells != len(geom):
        # In the orchestrator we may have subsampled; replicate here by aligning
        # cell_ids when possible, otherwise skip H7 plotting safely.
        common = set(expr.cell_ids) & set(geom["cell_id"].astype(str))
        if not common:
            return
        cid_to_idx = {c: i for i, c in enumerate(expr.cell_ids)}
        order = [cid_to_idx[c] for c in geom["cell_id"].astype(str) if c in cid_to_idx]
        expr = psp_io._reindex_expression(expr, order)

    X_norm = psp_pp.normalize_log1p(expr.X)

    top_genes = h7["gene"].head(5).tolist()
    cands = [g for g in top_genes if g in expr.gene_names]
    if not cands:
        return
    # sphere-normalised perturbation
    import scipy.sparse as sp
    if sp.issparse(X_norm):
        X_dense = X_norm.toarray()
    else:
        X_dense = np.asarray(X_norm)
    pert = psp_pert.compute_gene_perturbation_vectors(
        X_dense, expr.gene_names, cands, pca_components, pca_mean,
        normalize_to_sphere=True,
    )

    n = len(cands)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1: axes = [axes]
    for ax, g in zip(axes, cands):
        psp_pert.plot_perturbation_vector_field(pert, g, frac=0.02, ax=ax, clip=0.1)
        ax.set_title(f"{ds_name}: {g}")
    fig.tight_layout()
    fig.savefig(fig_dir / "H7_perturbation_vector_field.png", bbox_inches="tight")
    plt.close(fig)


def _reload_expression(ds: str, raw_root: Path):
    try:
        if ds == "celegan":
            return psp_io.load_celegan(str(raw_root / "celegan"))
        if ds == "uc_epi":
            return psp_io.load_uc_epi(str(raw_root / "uc_epi"))
        if ds == "klein":
            return psp_io.load_bz2_klein_dataset(str(raw_root / "klein"))
        if ds == "hesc":
            return psp_io.load_hesc(str(raw_root / "hESC"))
    except Exception as e:
        print(f"_reload_expression {ds} failed: {e}")
        return None
    return None


def fig_summary_heatmap(out: Path):
    rows = []
    for ds_dir in sorted([p for p in out.iterdir() if p.is_dir()]):
        ds = ds_dir.name
        h1p = ds_dir / "H1_entropy_gradient_metrics.csv"
        h4p = ds_dir / "H4_radial_angular_metrics.csv"
        h5p = ds_dir / "H5_stripe_boundary_genes.csv"
        h7p = ds_dir / "H7_perturbation_gene_rankings.csv"
        if not h1p.exists(): continue
        h1 = pd.read_csv(h1p).iloc[0]
        rho_geo = float(h1.get("geodesic_spearman_rho", float("nan")))
        rho_euc = float(h1.get("euclidean_spearman_rho", float("nan")))
        h4 = pd.read_csv(h4p) if h4p.exists() else pd.DataFrame()
        radial_max = h4[h4["vs"] == "radial"]["spearman_rho"].abs().max() if not h4.empty else float("nan")
        theta_max = h4[h4["vs"] == "theta"]["spearman_rho"].abs().max() if not h4.empty else float("nan")
        h5 = pd.read_csv(h5p) if h5p.exists() else pd.DataFrame()
        n_known_h5 = int(h5["is_known_regulator"].sum()) if not h5.empty and "is_known_regulator" in h5 else 0
        h7 = pd.read_csv(h7p) if h7p.exists() else pd.DataFrame()
        n_known_h7 = int(h7["is_known_regulator"].sum()) if not h7.empty and "is_known_regulator" in h7 else 0
        rows.append({
            "dataset": ds,
            "H1 |rho_geo|": abs(rho_geo) if not np.isnan(rho_geo) else 0,
            "H1 |rho_euc|": abs(rho_euc) if not np.isnan(rho_euc) else 0,
            "H4 max |rho| radial": radial_max if not np.isnan(radial_max) else 0,
            "H4 max |rho| theta": theta_max if not np.isnan(theta_max) else 0,
            "H5 known-overlap (n)": n_known_h5,
            "H7 known-overlap (n)": n_known_h7,
        })
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("dataset")
    fig, ax = plt.subplots(figsize=(8, 0.8 + 0.45 * len(df)))
    im = ax.imshow(df.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center",
                    color="w" if df.values[i, j] < df.values.max() * 0.5 else "k", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.025)
    ax.set_title("Hypothesis-support summary heatmap")
    fig.tight_layout()
    fig.savefig(out / "summary_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/raw_expression_validation"))
    parser.add_argument("--raw-data", type=Path, default=Path("raw_data"))
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    out = args.out
    if not out.exists():
        raise SystemExit(f"output directory does not exist: {out}")

    ds_dirs = [p for p in out.iterdir() if p.is_dir()]
    if args.datasets:
        ds_dirs = [p for p in ds_dirs if p.name in args.datasets]

    for ds_dir in ds_dirs:
        ds = ds_dir.name
        if not (ds_dir / "_artifacts.npz").exists():
            continue
        fig_dir = ds_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        print(f"plotting {ds}")
        try:
            fig_h1(ds, ds_dir, fig_dir)
            fig_h4(ds, ds_dir, fig_dir)
            fig_h5(ds, ds_dir, fig_dir)
            fig_h7(ds, ds_dir, fig_dir, args.raw_data)
        except Exception as e:
            print(f"  {ds} failed: {e}")
    fig_summary_heatmap(out)
    print(f"figures written under {out}")


if __name__ == "__main__":
    main()
