"""Supplement 1 — Benchmark embedding analysis (C. elegans).

Inputs (raw_data/benchmark/):
    Celegan_combined_reductions_with_50PCs_and_times.csv  — PC1..PC50,
        PC_Sphere1..3, t-SNE 2D/3D, UMAP 2D/3D, embryo_time.
    pca_celegan_ori.csv  — original (unrotated) PC1..PC3 with celltype/cluster.
    celegan_pca.csv       — sphere-aligned PC1..PC3.
    celegan_latent_normal_10epoch_2572sec.tsv — scPhere 3D latent.

The benchmark frames PCA, SPHERE-PCA, UMAP, and t-SNE side-by-side. We
*do not* claim SPHERE-PCA universally outperforms its peers; we report
quantitative metrics (neighborhood preservation, continuity, correlation
with developmental time) and a compact visual comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness

from . import common


_BENCH_RAW = common.RAW / "benchmark"
_OUT = common.SUPP_ROOT / "Benchmark"


_EMBEDDINGS = {
    # display name: list of column names (3-D embedding)
    "PCA (PC1-3)":      ["PC1", "PC2", "PC3"],
    "SPHERE-PCA":       ["PC_Sphere1", "PC_Sphere2", "PC_Sphere3"],
    "UMAP (3D)":        ["UMAP1_3D", "UMAP2_3D", "UMAP3_3D"],
    "t-SNE (3D)":       ["tSNE1_3D", "tSNE2_3D", "tSNE3_3D"],
    "scPhere (3D)":     ["scPhere1", "scPhere2", "scPhere3"],
}

_PALETTE = {
    "PCA (PC1-3)":  "#4c78a8",
    "SPHERE-PCA":   "#d1495b",
    "UMAP (3D)":    "#54a24b",
    "t-SNE (3D)":   "#7c5295",
    "scPhere (3D)": "#e08c2c",
}


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def _embryo_time_to_minutes(label) -> float:
    """Parse embryo_time labels like '210-270', '< 100', '> 600'."""
    if label is None:
        return np.nan
    s = str(label).strip()
    if s in ("", "nan", "NA", "None", "NaN"):
        return np.nan
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except ValueError:
            return np.nan
    if s.startswith("<"):
        try:
            return float(s.replace("<", "").strip()) - 1.0
        except ValueError:
            return np.nan
    if s.startswith(">"):
        try:
            return float(s.replace(">", "").strip()) + 1.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    """Reconstruct the benchmark dataframe used by `examples/benchmark.ipynb`.

    Adds scPhere 3-D latent coordinates from
    ``celegan_latent_normal_10epoch_2572sec.tsv`` (whitespace-separated;
    rows align positionally with the combined CSV)."""
    combined = pd.read_csv(_BENCH_RAW /
                           "Celegan_combined_reductions_with_50PCs_and_times.csv")
    pca_orig = pd.read_csv(_BENCH_RAW / "pca_celegan_ori.csv", index_col=0)
    pca_sphere = pd.read_csv(_BENCH_RAW / "celegan_pca.csv", index_col=0)

    # Restore the original PC1-3 (combined CSV holds rotated-to-sphere PC1-3).
    combined = combined.copy()
    combined[["PC1", "PC2", "PC3"]] = pca_orig[["PC1", "PC2", "PC3"]].to_numpy()

    # Ensure PC_Sphere columns are unit-norm.
    sph = combined[["PC_Sphere1", "PC_Sphere2", "PC_Sphere3"]].to_numpy(dtype=float)
    norms = np.linalg.norm(sph, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined[["PC_Sphere1", "PC_Sphere2", "PC_Sphere3"]] = sph / norms

    # Cluster/celltype from the index-aligned PCA original CSV.
    combined["cluster"] = pca_orig["cluster"].to_numpy()
    if "celltype" in pca_sphere.columns:
        combined["celltype"] = pca_sphere["celltype"].to_numpy()
    else:
        combined["celltype"] = pca_orig["celltype"].to_numpy()

    combined["time_minutes"] = combined["embryo_time"].map(_embryo_time_to_minutes)

    # scPhere latent (3-D, whitespace-delimited).
    sp_path = _BENCH_RAW / "celegan_latent_normal_10epoch_2572sec.tsv"
    sp = pd.read_csv(sp_path, sep=r"\s+", header=None,
                     names=["scPhere1", "scPhere2", "scPhere3"])
    if len(sp) != len(combined):
        raise ValueError(
            f"scPhere latent rows ({len(sp)}) do not match combined "
            f"benchmark rows ({len(combined)})."
        )
    # scPhere natively lives on the unit sphere; normalise to enforce it.
    sp_arr = sp.to_numpy(dtype=float)
    sp_norm = np.linalg.norm(sp_arr, axis=1, keepdims=True)
    sp_norm[sp_norm == 0] = 1.0
    sp_arr = sp_arr / sp_norm
    combined["scPhere1"] = sp_arr[:, 0]
    combined["scPhere2"] = sp_arr[:, 1]
    combined["scPhere3"] = sp_arr[:, 2]
    return combined


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _knn_overlap(X_high: np.ndarray, X_low: np.ndarray,
                 *, n_neighbors: int = 15) -> float:
    """Mean Jaccard overlap between k-NN neighbourhoods of high-D and low-D."""
    from sklearn.neighbors import NearestNeighbors
    nbrs_h = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_high)
    nbrs_l = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_low)
    _, idx_h = nbrs_h.kneighbors(X_high)
    _, idx_l = nbrs_l.kneighbors(X_low)
    n = X_high.shape[0]
    overlaps = np.empty(n, dtype=float)
    for i in range(n):
        a = set(idx_h[i, 1:].tolist())
        b = set(idx_l[i, 1:].tolist())
        overlaps[i] = len(a & b) / float(n_neighbors)
    return float(np.mean(overlaps))


def _embedding_metrics(df: pd.DataFrame, *,
                       n_neighbors: int = 15,
                       n_subsample: int = 5000,
                       seed: int = 0) -> pd.DataFrame:
    """Compute trustworthiness, continuity, kNN overlap, time correlation."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n > n_subsample:
        idx = np.sort(rng.choice(n, n_subsample, replace=False))
        df_sub = df.iloc[idx].reset_index(drop=True).copy()
    else:
        df_sub = df.reset_index(drop=True).copy()

    pc50_cols = [f"PC{i}" for i in range(1, 51)]
    X_high = df_sub[pc50_cols].to_numpy(dtype=float)

    rows = []
    for name, cols in _EMBEDDINGS.items():
        X_low = df_sub[cols].to_numpy(dtype=float)
        if np.any(~np.isfinite(X_low)) or np.any(~np.isfinite(X_high)):
            mask = np.isfinite(X_low).all(1) & np.isfinite(X_high).all(1)
            X_h_use = X_high[mask]; X_l_use = X_low[mask]
        else:
            X_h_use = X_high; X_l_use = X_low

        trust = float(trustworthiness(X_h_use, X_l_use, n_neighbors=n_neighbors))
        cont = float(trustworthiness(X_l_use, X_h_use, n_neighbors=n_neighbors))
        knn = _knn_overlap(X_h_use, X_l_use, n_neighbors=n_neighbors)

        # Spearman of distance-from-developmental-anchor with time.
        # Anchor: the cluster '100-130' in the original notebook.
        time = df_sub["time_minutes"].to_numpy(dtype=float)
        anchor_mask = df_sub["cluster"].astype(str).values == "100-130"
        if anchor_mask.sum() > 0:
            anchor = X_low[anchor_mask].mean(axis=0)
            dist = np.linalg.norm(X_low - anchor, axis=1)
            tmask = np.isfinite(time)
            if tmask.sum() > 5:
                rho, _ = spearmanr(dist[tmask], time[tmask])
                spearman_time = float(rho)
            else:
                spearman_time = np.nan
        else:
            spearman_time = np.nan

        rows.append({
            "embedding": name,
            "n_components": len(cols),
            "trustworthiness_T": trust,
            "continuity_C": cont,
            "knn_jaccard": knn,
            "spearman_distance_vs_time": spearman_time,
            "n_cells_evaluated": int(X_h_use.shape[0]),
            "n_neighbors": n_neighbors,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproducibility (seed sweep, kNN-overlap)
# ---------------------------------------------------------------------------

def _reproducibility_sweep(df: pd.DataFrame, *, seeds=(0, 1, 2),
                           n_subsample: int = 4000,
                           n_neighbors: int = 15) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        out = _embedding_metrics(
            df, n_neighbors=n_neighbors, n_subsample=n_subsample, seed=seed,
        )
        out["seed"] = seed
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _draw_embedding_panel(ax, df: pd.DataFrame, cols, time_col: str,
                          title: str):
    pts = df[cols].to_numpy(dtype=float)
    t = df[time_col].to_numpy(dtype=float)
    finite = np.isfinite(t) & np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]; t = t[finite]
    if len(cols) >= 3:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=t,
                        cmap="viridis", s=1.6, alpha=0.55, linewidths=0)
    else:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=t,
                        cmap="viridis", s=1.6, alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=10.5, weight="bold", color="#1d2230", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbb"); ax.spines["bottom"].set_color("#bbb")
    return sc


def _make_figure(df: pd.DataFrame, metrics: pd.DataFrame) -> plt.Figure:
    n_emb = len(_EMBEDDINGS)
    n_metric = 4
    # Place panels on a common 20-column grid so 5 scatter panels
    # (4 cols each) and 4 metric panels (5 cols each) align cleanly.
    fig = plt.figure(figsize=(16.2, 8.8), facecolor="white")
    gs = fig.add_gridspec(
        2, 20, height_ratios=[1.0, 0.95],
        hspace=0.86, wspace=1.45,
        left=0.075, right=0.94, top=0.78, bottom=0.19,
    )

    # Top row: 5 scatter panels, 4 columns each.
    sc_last = None
    for i, (name, cols) in enumerate(_EMBEDDINGS.items()):
        ax = fig.add_subplot(gs[0, 4 * i:4 * (i + 1)])
        sc_last = _draw_embedding_panel(ax, df, cols[:2],
                                        time_col="time_minutes",
                                        title=name)

    if sc_last is not None:
        cax = fig.add_axes([0.955, 0.505, 0.0085, 0.27])
        cb = fig.colorbar(sc_last, cax=cax)
        cb.set_label("Embryo time (min, midpoint)", fontsize=8.5,
                     labelpad=7)
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.outline.set_linewidth(0.4)

    # Bottom row: 4 metric panels, 5 columns each.
    metric_columns = [
        ("trustworthiness_T", "Trustworthiness  ↑",
         "k-NN preservation (high→low) [local]"),
        ("continuity_C", "Continuity  ↑",
         "k-NN preservation (low→high) [local]"),
        ("knn_jaccard", "k-NN Jaccard  ↑",
         "neighborhood overlap [local]"),
        ("spearman_distance_vs_time", "|Spearman| dist ↔ time  ↑",
         "developmental ordering [global]"),
    ]
    metrics_for_plot = metrics.set_index("embedding")
    names = list(_EMBEDDINGS.keys())
    for j, (col, ylabel, sub) in enumerate(metric_columns):
        ax = fig.add_subplot(gs[1, 5 * j:5 * (j + 1)])
        vals = []
        for n in names:
            v = (metrics_for_plot.loc[n, col]
                 if n in metrics_for_plot.index else np.nan)
            if col == "spearman_distance_vs_time" and pd.notna(v):
                v = abs(v)
            vals.append(v)
        colors = [_PALETTE[n] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors,
                      edgecolor="#1d2230", linewidth=0.5, width=0.72)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right",
                           rotation_mode="anchor", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9.0, labelpad=10)
        ax.text(0.0, 1.06, sub, transform=ax.transAxes,
                fontsize=7.5, color="#5a5e6b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for b, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.4, color="#1d2230")
        ax.set_ylim(0, max(0.05, max(np.nan_to_num(vals)) * 1.18))

    fig.suptitle(
        "Suppl. Fig 1 — Benchmark embedding comparison on C. elegans",
        x=0.075, y=0.97, ha="left",
        fontsize=12.5, weight="bold", color="#1d2230",
    )
    fig.text(0.075, 0.93,
             "PCA, SPHERE-PCA, UMAP (3D), t-SNE (3D), and scPhere (3D) "
             "compared against the high-dim PC1-50 reference. SPHERE-PCA\n"
             "provides interpretable angular/radial coordinates and is "
             "competitive with specialised spherical embeddings "
             "(scPhere) and\nneighborhood embeddings (UMAP, t-SNE). "
             "T, C, Jaccard favour local neighborhood preservation; "
             "|ρ(dist, time)| reflects\nglobal developmental ordering. "
             "No embedding dominates universally.",
             fontsize=8.5, color="#3c3c3c", linespacing=1.35,
             va="top")
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(*, registry: list | None = None, seeds=(0, 1, 2),
        n_subsample: int = 5000, n_neighbors: int = 15) -> Dict[str, Path]:
    common.apply_publication_style()
    out = _OUT
    out.mkdir(parents=True, exist_ok=True)

    df = _load_data()
    metrics = _embedding_metrics(
        df, n_neighbors=n_neighbors, n_subsample=n_subsample, seed=seeds[0],
    )
    repro = _reproducibility_sweep(
        df, seeds=tuple(seeds),
        n_subsample=min(n_subsample, 4000),
        n_neighbors=n_neighbors,
    )

    summary = repro.groupby("embedding").agg(
        T_mean=("trustworthiness_T", "mean"),
        T_std=("trustworthiness_T", "std"),
        C_mean=("continuity_C", "mean"),
        C_std=("continuity_C", "std"),
        knn_jaccard_mean=("knn_jaccard", "mean"),
        knn_jaccard_std=("knn_jaccard", "std"),
        spearman_time_mean=("spearman_distance_vs_time", "mean"),
        spearman_time_std=("spearman_distance_vs_time", "std"),
    ).reset_index()

    fig = _make_figure(df, metrics)

    config = {
        "module": "supplements.benchmark",
        "raw_data_dir": str(_BENCH_RAW.relative_to(common.ROOT)),
        "n_cells_total": int(len(df)),
        "n_subsample": n_subsample,
        "n_neighbors": n_neighbors,
        "seeds": list(seeds),
        "embeddings": list(_EMBEDDINGS.keys()),
        "metrics_definition": {
            "trustworthiness_T":
                "sklearn.manifold.trustworthiness(high, low) — "
                "fraction of k-NN structure preserved (high→low).  "
                "[favours local neighborhood preservation]",
            "continuity_C":
                "sklearn.manifold.trustworthiness(low, high) — "
                "k-NN structure preserved (low→high).  "
                "[favours local neighborhood preservation]",
            "knn_jaccard":
                "Mean Jaccard overlap of k-NN sets between high-D PC50 "
                "and the embedding.  [local neighborhood overlap]",
            "spearman_distance_vs_time":
                "Spearman correlation between distance from cluster "
                "'100-130' centroid in the embedding and the embryo "
                "time-window midpoint (minutes).  [favours global "
                "developmental ordering, not local preservation]",
        },
        "interpretation_note":
            "SPHERE-PCA provides interpretable angular/radial coordinates "
            "and is competitive with specialised spherical embeddings "
            "(scPhere) and neighborhood embeddings (UMAP, t-SNE). We do "
            "not claim universal superiority — UMAP/t-SNE typically lead "
            "on T/C/Jaccard (local preservation), while SPHERE-PCA and "
            "scPhere provide a structured global coordinate system "
            "favouring developmental-label monotonicity.",
    }

    common.save_outputs(
        out, "SuppFig_Benchmark_Embedding_Comparison",
        fig=fig, config=config,
        data={
            "summary": summary,
            "metrics": metrics,
            "metrics_seed_sweep": repro,
        },
        registry=registry,
    )
    # Also write the canonical short-form CSVs requested in the spec.
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    metrics.to_csv(out / "benchmark_metrics.csv", index=False)
    if registry is not None:
        registry.append(str(out / "benchmark_summary.csv"))
        registry.append(str(out / "benchmark_metrics.csv"))
    return {"output_dir": out}
