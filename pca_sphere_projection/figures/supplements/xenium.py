"""Supplement 4 — Xenium / TRACER stress test (BrCa & NSCLC).

Inputs (raw_data/xenium/):
    BrCa/Pre_TRACER_breast_cancer.h5ad
    BrCa/Post_TRACER_breast_cancer.h5ad
    NSCLC/Pre_TRACER_lung_cancer.h5ad
    NSCLC/Post_TRACER_lung_cancer.h5ad

Positive-control reference (raw_data/BrCa_atlas/):
    matrix.mtx.gz / barcodes.tsv.gz / features.tsv.gz — whole-
    transcriptome BrCa scRNA-seq atlas. Used purely as a PC1-50
    explained-variance reference, processed with canonical
    HVG=2000 selection.

The Xenium panels carry only ~300 assayed genes so the canonical
HVG=2000 step does not apply. Per the spec, we use *all* assayed genes
as the feature set, then run normalize_total → log1p → scale → PCA(50).

Part A: cumulative + per-PC explained-variance curves (pre vs post,
both panels) plus the BrCa atlas reference curve.
Part B: pre vs post sphere views from three camera angles, no
biological root, no equirectangular projection — a stress-test
visualisation only.

Interpretation note (kept conservative throughout): we do **not** claim
developmental trajectories in these spatial panels. Spherical structure
emerging in a targeted spatial panel is a stress test of the geometry,
not a biological trajectory claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import gzip
import io as _io

import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from . import common


_X_RAW = common.RAW / "xenium"
_OUT = common.SUPP_ROOT / "Xenium"


PANELS = {
    "BrCa": {
        "Pre":  _X_RAW / "BrCa" / "Pre_TRACER_breast_cancer.h5ad",
        "Post": _X_RAW / "BrCa" / "Post_TRACER_breast_cancer.h5ad",
    },
    "NSCLC": {
        "Pre":  _X_RAW / "NSCLC" / "Pre_TRACER_lung_cancer.h5ad",
        "Post": _X_RAW / "NSCLC" / "Post_TRACER_lung_cancer.h5ad",
    },
}

_BRCA_ATLAS_DIR = common.RAW / "BrCa_atlas"


# ---------------------------------------------------------------------------
# h5ad reader (the bundled files break stock anndata.read_h5ad because of a
# null `uns/log1p/base` element; we read the raw counts directly).
# ---------------------------------------------------------------------------

def _read_counts(h5_path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    with h5py.File(str(h5_path), "r") as f:
        gx = f["layers/counts"]
        data = np.asarray(gx["data"])
        indices = np.asarray(gx["indices"])
        indptr = np.asarray(gx["indptr"])
        n_obs = indptr.shape[0] - 1
        var_idx = f["var/_index"][...]
        gene_names = np.asarray([
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in var_idx
        ])
        n_var = len(gene_names)
    counts = sp.csr_matrix((data, indices, indptr),
                           shape=(n_obs, n_var))
    return counts, gene_names


# ---------------------------------------------------------------------------
# Canonical preprocessing (all genes, normalize_total → log1p → scale → PCA)
# ---------------------------------------------------------------------------

_PCA_CACHE = _OUT / "_cache"


def _cache_path(panel: str, state: str, seed: int, n_components: int,
                max_cells: int | None) -> Path:
    suffix = "" if max_cells is None else f"_n{int(max_cells)}"
    return _PCA_CACHE / f"{panel}_{state}_pc{n_components}_seed{seed}{suffix}.npz"


def _preprocess_pca(panel: str, state: str, *,
                    n_components: int = 50, seed: int = 0,
                    use_cache: bool = True,
                    max_cells: int | None = None) -> Dict[str, object]:
    cache_p = _cache_path(panel, state, seed, n_components, max_cells)
    if use_cache and cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        return {
            "scores": np.asarray(z["scores"], dtype=np.float32),
            "explained_variance_ratio":
                np.asarray(z["explained_variance_ratio"], dtype=np.float32),
            "n_genes_used": int(z["n_genes_used"]),
            "n_cells_used": int(z["n_cells_used"]),
        }
    _PCA_CACHE.mkdir(parents=True, exist_ok=True)

    import scanpy as sc
    import anndata as ad

    sc.settings.verbosity = 0
    counts, gene_names = _read_counts(PANELS[panel][state])
    if max_cells is not None and counts.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(counts.shape[0], max_cells, replace=False))
        counts = counts[sel]

    var = pd.DataFrame(index=list(gene_names))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(counts.shape[0])])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Use ALL assayed genes (the panel is already targeted/compact).
    sc.pp.scale(adata, max_value=10)
    n_comp = int(min(n_components, adata.n_vars - 1, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=n_comp, random_state=seed, zero_center=True)
    out = {
        "scores": np.asarray(adata.obsm["X_pca"], dtype=np.float32),
        "explained_variance_ratio":
            np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float32),
        "n_genes_used": int(adata.n_vars),
        "n_cells_used": int(adata.n_obs),
    }
    if use_cache:
        np.savez(
            cache_p,
            scores=out["scores"],
            explained_variance_ratio=out["explained_variance_ratio"],
            n_genes_used=out["n_genes_used"],
            n_cells_used=out["n_cells_used"],
        )
    return out


# ---------------------------------------------------------------------------
# BrCa atlas (whole-transcriptome scRNA-seq) — positive control reference
# ---------------------------------------------------------------------------

def _read_brca_atlas_counts() -> tuple[sp.csr_matrix, np.ndarray]:
    """Read 10x-style barcodes/features/matrix for the BrCa atlas."""
    with gzip.open(_BRCA_ATLAS_DIR / "features.tsv.gz", "rb") as fh:
        lines = fh.read().decode("utf-8").splitlines()
        gene_names = np.array([ln.split("\t")[0] for ln in lines])
    with gzip.open(_BRCA_ATLAS_DIR / "matrix.mtx.gz", "rb") as fh:
        m = mmread(_io.BytesIO(fh.read()))
    # Cell Ranger MTX is genes-by-cells; we want cells-by-genes.
    counts = sp.csr_matrix(m).T.tocsr()
    if counts.shape[1] != len(gene_names):
        # If transpose is wrong (depends on file), correct.
        if counts.T.shape[1] == len(gene_names):
            counts = counts.T.tocsr()
    return counts, gene_names


def _brca_atlas_pca(*, n_components: int = 50, seed: int = 0,
                     max_cells: int = 30000,
                     n_top_genes: int = 2000,
                     use_cache: bool = True) -> Dict[str, object]:
    cache_p = _PCA_CACHE / (
        f"BrCa_atlas_pc{n_components}_seed{seed}"
        f"_n{max_cells}_hvg{n_top_genes}.npz"
    )
    if use_cache and cache_p.exists():
        z = np.load(cache_p, allow_pickle=True)
        return {
            "explained_variance_ratio":
                np.asarray(z["explained_variance_ratio"], dtype=np.float32),
            "n_genes_used": int(z["n_genes_used"]),
            "n_cells_used": int(z["n_cells_used"]),
        }
    _PCA_CACHE.mkdir(parents=True, exist_ok=True)

    import scanpy as sc
    import anndata as ad

    sc.settings.verbosity = 0

    counts, gene_names = _read_brca_atlas_counts()
    n_obs = counts.shape[0]
    rng = np.random.default_rng(seed)
    if max_cells is not None and n_obs > max_cells:
        sel = np.sort(rng.choice(n_obs, max_cells, replace=False))
        counts = counts[sel]

    var = pd.DataFrame(index=list(gene_names))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(counts.shape[0])])
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Whole-transcriptome → canonical HVG=2000.
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes,
    )
    adata = adata[:, adata.var["highly_variable"].values].copy()
    sc.pp.scale(adata, max_value=10)
    n_comp = int(min(n_components, adata.n_vars - 1, adata.n_obs - 1))
    sc.tl.pca(adata, n_comps=n_comp, random_state=seed, zero_center=True)

    out = {
        "explained_variance_ratio":
            np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float32),
        "n_genes_used": int(adata.n_vars),
        "n_cells_used": int(adata.n_obs),
    }
    if use_cache:
        np.savez(
            cache_p,
            explained_variance_ratio=out["explained_variance_ratio"],
            n_genes_used=out["n_genes_used"],
            n_cells_used=out["n_cells_used"],
        )
    return out


# ---------------------------------------------------------------------------
# Part A — explained variance curves
# ---------------------------------------------------------------------------

def _variance_figure(states: Dict[str, Dict[str, dict]],
                     brca_atlas: dict | None = None
                     ) -> Tuple[plt.Figure, pd.DataFrame]:
    """Three-panel layout: (1) BrCa Xenium pre/post, (2) NSCLC Xenium
    pre/post, (3) BrCa scRNA-seq atlas reference (separate panel)."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    n_panels = 3 if brca_atlas is not None else 2
    fig_w = 15.4 if n_panels == 3 else 12.8
    fig = plt.figure(figsize=(fig_w, 6.3), facecolor="white")
    gs = fig.add_gridspec(1, n_panels, wspace=0.36,
                          left=0.065, right=0.985, top=0.74, bottom=0.30)

    palette = {"Pre": "#4c78a8", "Post": "#d1495b"}
    atlas_color = "#2f6f3f"
    rows = []

    def _plot_inset_perpc(ax, series):
        axin = inset_axes(ax, width="40%", height="36%",
                          loc="upper right", borderpad=1.3)
        max_top = 0.0
        for ev, color in series:
            ks = np.arange(1, len(ev) + 1)
            axin.plot(ks, ev, color=color, linewidth=0.9)
            max_top = max(max_top, float(np.nanmax(ev[:5])))
        axin.set_xlim(0.5, 50.5)
        axin.set_ylim(0, max(0.02, max_top * 1.18))
        axin.set_xlabel("PC", fontsize=7.5, labelpad=0.5)
        axin.set_ylabel("per-PC ratio", fontsize=7.5, labelpad=1.0)
        axin.tick_params(labelsize=6.5, length=2)
        axin.grid(True, color="#dadbde", linewidth=0.3, alpha=0.7)
        axin.set_axisbelow(True)

    # --- panels 1 & 2: Xenium BrCa, NSCLC ---
    for ci, panel in enumerate(["BrCa", "NSCLC"]):
        ax = fig.add_subplot(gs[0, ci])
        per_pc_series = []
        for state in ["Pre", "Post"]:
            ev = np.asarray(states[panel][state]["explained_variance_ratio"])
            cum = np.cumsum(ev)
            ks = np.arange(1, len(ev) + 1)
            ax.plot(ks, cum, color=palette[state], linewidth=1.7,
                    label=f"{state}-TRACER (cumulative)")
            ax.plot(ks, ev, color=palette[state], linewidth=0.8,
                    linestyle="--", alpha=0.6,
                    label=f"{state}-TRACER (per-PC)")
            per_pc_series.append((ev, palette[state]))
            for k, ev_k, c in zip(ks, ev, cum):
                rows.append({
                    "panel": panel, "state": state,
                    "PC": int(k), "explained_variance_ratio": float(ev_k),
                    "cumulative": float(c),
                    "n_cells_used": states[panel][state]["n_cells_used"],
                    "n_genes_used": states[panel][state]["n_genes_used"],
                })
        _plot_inset_perpc(ax, per_pc_series)
        ax.set_xlabel("Principal component (1..50)", fontsize=9,
                      labelpad=8)
        ax.set_ylabel("Explained variance", fontsize=9, labelpad=8)
        ax.set_xlim(0.5, 50.5)
        ax.set_ylim(0.0, 1.02)
        title = (f"{panel} — Xenium  "
                 f"(targeted panel, ~"
                 f"{states[panel]['Pre']['n_genes_used']} genes)")
        ax.set_title(title, fontsize=10.5, weight="bold",
                     color="#1d2230", pad=4)
        ax.grid(True, color="#dadbde", linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7.4, frameon=False, ncol=2,
                  columnspacing=1.0, handletextpad=0.4)

    # --- panel 3: BrCa scRNA-seq atlas (separate) ---
    if brca_atlas is not None:
        atlas_ev = np.asarray(brca_atlas["explained_variance_ratio"])
        atlas_cum = np.cumsum(atlas_ev)
        atlas_ks = np.arange(1, len(atlas_ev) + 1)
        for k, ev_k, c in zip(atlas_ks, atlas_ev, atlas_cum):
            rows.append({
                "panel": "BrCa_atlas (scRNA-seq, reference)",
                "state": "reference",
                "PC": int(k),
                "explained_variance_ratio": float(ev_k),
                "cumulative": float(c),
                "n_cells_used": int(brca_atlas["n_cells_used"]),
                "n_genes_used": int(brca_atlas["n_genes_used"]),
            })

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(atlas_ks, atlas_cum, color=atlas_color, linewidth=1.7,
                label="BrCa atlas scRNA-seq (cumulative)")
        ax.plot(atlas_ks, atlas_ev, color=atlas_color, linewidth=0.8,
                linestyle="--", alpha=0.65,
                label="BrCa atlas scRNA-seq (per-PC)")
        _plot_inset_perpc(ax, [(atlas_ev, atlas_color)])
        ax.set_xlabel("Principal component (1..50)", fontsize=9,
                      labelpad=8)
        ax.set_ylabel("Explained variance", fontsize=9, labelpad=8)
        ax.set_xlim(0.5, 50.5)
        ax.set_ylim(0.0, 1.02)
        ax.set_title(
            f"BrCa atlas — scRNA-seq  "
            f"(positive control, HVG={brca_atlas['n_genes_used']})",
            fontsize=10.5, weight="bold", color="#1d2230", pad=4,
        )
        ax.grid(True, color="#dadbde", linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7.4, frameon=False, ncol=1,
                  handletextpad=0.4)

    fig.suptitle(
        "Suppl. Fig 4A — Xenium TRACER explained-variance curves  "
        "vs BrCa scRNA-seq atlas reference (separate panel)",
        x=0.065, y=0.965, ha="left",
        fontsize=12, weight="bold", color="#1d2230",
    )
    fig.text(0.065, 0.895,
             "Xenium targeted spatial panels (~300 assayed genes, all "
             "used) shown alongside the whole-transcriptome BrCa "
             "scRNA-seq atlas\n(HVG=2000) as a positive-control "
             "reference. Differences reflect panel design and assay "
             "modality, not biological\ntrajectories — interpretation "
             "is descriptive only.",
             fontsize=8.5, color="#3c3c3c", linespacing=1.35, va="top")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part B — sphere visualisation stress test
# ---------------------------------------------------------------------------

_VIEW_ANGLES = [(20, -60), (20, 120), (-60, -60)]


def _sphere_views(unit_pre: np.ndarray, unit_post: np.ndarray, *,
                   panel: str, n_plot: int = 8000,
                   seed: int = 0) -> Tuple[plt.Figure, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(11.5, 7.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, wspace=0.04, hspace=0.05,
                          left=0.02, right=0.98, top=0.84, bottom=0.05)

    plotted = []
    for ri, (label, unit, color) in enumerate([
        ("Pre-TRACER", unit_pre, "#4c78a8"),
        ("Post-TRACER", unit_post, "#d1495b"),
    ]):
        n = unit.shape[0]
        sel = rng.choice(n, min(n_plot, n), replace=False)
        sel = np.sort(sel)
        for ci, (elev, azim) in enumerate(_VIEW_ANGLES):
            ax = fig.add_subplot(gs[ri, ci], projection="3d")
            common.draw_3d_sphere_backdrop(ax, color="#e8edf2", wire="#aab3bf",
                                           alpha_surface=0.10, alpha_wire=0.18)
            ax.scatter(unit[sel, 0], unit[sel, 1], unit[sel, 2],
                       c=color, s=1.4, alpha=0.50, linewidths=0,
                       depthshade=False)
            ax.view_init(elev=elev, azim=azim)
            common.style_clean_3d(ax)
            if ri == 0:
                ax.set_title(f"view {ci + 1}  (elev={elev}°, azim={azim}°)",
                             fontsize=8.5, color="#1d2230", pad=2)
            if ci == 0:
                ax.text2D(0.0, 0.5, label, transform=ax.transAxes,
                          rotation=90, ha="right", va="center",
                          fontsize=10.5, weight="bold", color="#1d2230")
        for s in sel:
            plotted.append({"state": label, "x": unit[s, 0],
                            "y": unit[s, 1], "z": unit[s, 2]})

    fig.text(0.02, 0.965,
             f"Suppl. Fig 4B — {panel}: pre vs post-TRACER sphere views  "
             f"(unrooted; {n_plot} cells/state)",
             fontsize=12, weight="bold", color="#1d2230")
    fig.text(0.02, 0.92,
             "Stress test only. Spherical structure can be visualised "
             "in targeted panels after transcript reassignment, but\n"
             "interpretation is constrained by panel design and mixed "
             "tissue composition — we do not claim developmental\n"
             "trajectories here.",
             fontsize=8.0, color="#3c3c3c", linespacing=1.35, va="top")
    return fig, pd.DataFrame(plotted)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(*, registry: list | None = None, seed: int = 0,
        n_components: int = 50, n_plot_cells: int = 8000,
        max_cells: int | None = None,
        atlas_max_cells: int = 30000,
        atlas_n_top_genes: int = 2000) -> Dict[str, object]:
    common.apply_publication_style()
    _OUT.mkdir(parents=True, exist_ok=True)
    _PCA_CACHE.mkdir(parents=True, exist_ok=True)

    states: Dict[str, Dict[str, dict]] = {}
    for panel in PANELS:
        states[panel] = {}
        for state in ("Pre", "Post"):
            states[panel][state] = _preprocess_pca(
                panel, state, n_components=n_components,
                seed=seed, use_cache=True, max_cells=max_cells,
            )

    # BrCa atlas (whole-transcriptome positive control).
    try:
        brca_atlas = _brca_atlas_pca(
            n_components=n_components, seed=seed,
            max_cells=atlas_max_cells,
            n_top_genes=atlas_n_top_genes,
            use_cache=True,
        )
    except Exception as exc:  # pragma: no cover - environmental fallback
        print(f"  [BrCa atlas PCA failed: {exc}; "
              f"variance figure will omit reference curve]")
        brca_atlas = None

    # Part A — variance curves (with BrCa atlas reference)
    fig_a, var_df = _variance_figure(states, brca_atlas=brca_atlas)
    common.save_outputs(
        _OUT, "SuppFig_Xenium_PC50_ExplainedVariance",
        fig=fig_a,
        config={
            "module": "supplements.xenium",
            "feature_selection_xenium": "ALL assayed genes (no HVG)",
            "feature_selection_brca_atlas":
                f"HVG (flavor=seurat, n_top_genes={atlas_n_top_genes})",
            "preprocessing":
                "normalize_total(target_sum=1e4) → log1p → "
                "[HVG for atlas] → scale(max_value=10) → PCA(n_comps=50)",
            "panels": list(PANELS.keys()),
            "states": ["Pre", "Post"],
            "reference": ("BrCa_atlas scRNA-seq positive-control"
                          if brca_atlas is not None else "unavailable"),
            "atlas_max_cells": int(atlas_max_cells),
            "n_components": n_components,
            "seed": seed,
            "interpretation_note":
                "Variance concentration differs between Xenium targeted "
                "panels and the whole-transcriptome BrCa atlas; this is "
                "a panel-design effect, not a biological trajectory "
                "claim.",
            "n_cells": {p: {s: int(states[p][s]["n_cells_used"])
                            for s in states[p]} for p in states},
            "n_genes": {p: {s: int(states[p][s]["n_genes_used"])
                            for s in states[p]} for p in states},
            "n_cells_brca_atlas":
                int(brca_atlas["n_cells_used"]) if brca_atlas else None,
            "n_genes_brca_atlas":
                int(brca_atlas["n_genes_used"]) if brca_atlas else None,
        },
        data={"variance_table": var_df},
        registry=registry,
    )
    # Spec-canonical filename.
    var_df.to_csv(_OUT / "xenium_pc50_variance_tables.csv", index=False)
    if registry is not None:
        registry.append(str(_OUT / "xenium_pc50_variance_tables.csv"))
    # Drop the legacy filename if present so we don't leave stale files.
    legacy = _OUT / "xenium_pca_variance_tables.csv"
    if legacy.exists():
        legacy.unlink()

    # Part B — sphere views (pre vs post)
    sphere_summary_rows = []
    for panel in PANELS:
        unit_pre = common.to_unit(states[panel]["Pre"]["scores"][:, :3])
        unit_post = common.to_unit(states[panel]["Post"]["scores"][:, :3])
        fig_b, plotted = _sphere_views(
            unit_pre, unit_post, panel=panel,
            n_plot=n_plot_cells, seed=seed,
        )
        common.save_outputs(
            _OUT, f"SuppFig_Xenium_{panel}_PrePost_SphereViews",
            fig=fig_b,
            config={
                "module": "supplements.xenium",
                "panel": panel,
                "n_plot_cells": int(n_plot_cells),
                "view_angles": _VIEW_ANGLES,
                "seed": seed,
                "feature_selection": "ALL assayed genes (no HVG)",
                "rooting": "unrooted / canonical orientation",
                "interpretation_note":
                    "Stress-test visualisation only. We do not assert "
                    "developmental trajectories from spatial panels.",
            },
            data={"plotted_cells": plotted},
            registry=registry,
        )
        sphere_summary_rows.append({
            "panel": panel,
            "n_pre_cells": int(states[panel]["Pre"]["n_cells_used"]),
            "n_post_cells": int(states[panel]["Post"]["n_cells_used"]),
        })
    return {
        "output_dir": _OUT,
        "states": states,
        "sphere_summary": pd.DataFrame(sphere_summary_rows),
    }
