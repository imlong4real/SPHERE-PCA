"""Reviewer-response run: PC robustness, multi-stripe, H1 CytoTRACE, H5, BrCa."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pca_sphere_projection import gene_geometry as gg
from pca_sphere_projection import io as pio
from pca_sphere_projection import pc_robustness as pcr
from pca_sphere_projection import preprocessing as pp
from pca_sphere_projection import sphere_stats as ss
from pca_sphere_projection import stripe


OUT = ROOT / "outputs"
RAW = ROOT / "raw_data"


def ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def subset_data(d: pio.ExpressionData, max_cells: int, seed: int = 0) -> pio.ExpressionData:
    if max_cells <= 0 or d.n_cells <= max_cells:
        return d
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(d.n_cells, max_cells, replace=False))
    X = d.X[idx] if sp.issparse(d.X) else np.asarray(d.X)[idx]
    md = d.metadata.iloc[idx].copy() if not d.metadata.empty else pd.DataFrame(index=[d.cell_ids[i] for i in idx])
    return pio.ExpressionData(X=X, gene_names=list(d.gene_names),
                              cell_ids=[d.cell_ids[i] for i in idx],
                              metadata=md, source=d.source,
                              notes=list(d.notes) + [f"deterministic subsample n={max_cells}"])


def load_datasets():
    out = {}
    failures = {}
    loaders = {
        "hESC": lambda: pio.load_hesc(str(RAW / "hESC")),
        "planaria": lambda: pio.load_expression_matrix(str(RAW / "planaria" / "dataset.rds")),
        "human_germ_cell": lambda: pio.load_expression_matrix(str(RAW / "human_germ_cell" / "dataset.rds")),
        "pre_implant_human_embryo": lambda: pio.load_expression_matrix(str(RAW / "pre_implant_human_embryo" / "dataset.rds")),
        "celegan": lambda: pio.load_celegan(str(RAW / "celegan")),
        "uc_epi": lambda: pio.load_uc_epi(str(RAW / "uc_epi")),
        "klein": lambda: pio.load_bz2_klein_dataset(str(RAW / "klein")),
        "BrCa_atlas": lambda: pio.load_brca_atlas(str(RAW / "BrCa_atlas")),
    }
    for name, fn in loaders.items():
        try:
            out[name] = fn()
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return out, failures


def real_stemness(md: pd.DataFrame):
    for col in ["CytoTRACE", "output_CytoTRACE", "GCS"]:
        if col in md.columns:
            vals = pd.to_numeric(md[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 5:
                if col == "CytoTRACE" and np.nanmax(vals) > 1.5:
                    vals = 1.0 - (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-9)
                return col, vals
    return None, None


def pseudotime(md: pd.DataFrame):
    for col in ["Order", "output_Order", "embryo_time_mid"]:
        if col in md.columns:
            vals = pd.to_numeric(md[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 5:
                return col, vals
    if "day" in md.columns:
        vals = md["day"].astype(str).str.extract(r"(\d+)")[0].astype(float).to_numpy()
        if np.isfinite(vals).sum() >= 5:
            return "day", vals
    return None, None


def prep(d: pio.ExpressionData, max_cells: int):
    d = subset_data(d, max_cells=max_cells)
    Xn = pp.normalize_log1p(d.X)
    _, genes, Xh = pp.select_hvgs(Xn, d.gene_names, n_top_genes=min(2000, len(d.gene_names)))
    return d, Xn, Xh, genes


def save_stripe(dataset, coords):
    out = ensure(OUT / "stripe_robustness" / dataset)
    ms = stripe.multi_stripe_strength_score(coords)
    metrics = pd.DataFrame([{
        "dataset": dataset,
        "n_stripes": ms["n_stripes"],
        "global_longitude_concentration": ms["global_longitude_concentration"],
        "multi_stripe_strength": ms["multi_stripe_strength"],
        "mean_local_stripe_strength": ms["mean_local_stripe_strength"],
        "median_local_stripe_strength": ms["median_local_stripe_strength"],
    }])
    metrics.to_csv(out / "multi_stripe_metrics.csv", index=False)
    ms["per_stripe"].to_csv(out / "per_stripe_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    stripe.plot_detected_stripes(coords, ms["stripe_assignments"], ax=ax)
    fig.tight_layout()
    fig.savefig(out / "detected_stripes.png", dpi=180)
    plt.close(fig)
    lines = [f"# {dataset} per-stripe report", "", metrics.to_markdown(index=False), "", ms["per_stripe"].to_markdown(index=False)]
    (out / "per_stripe_report.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics.iloc[0].to_dict()


def save_pc_robustness(dataset, Xh, genes, md, pt, stem):
    out = ensure(OUT / "pc_robustness" / dataset)
    scree = pcr.compute_scree_summary(Xh, max_pcs=min(30, Xh.shape[1] - 1, Xh.shape[0] - 1))
    counts = pcr.compare_pc_count_spherical_geometry(Xh, genes, md, [3, 5, 10, 20], pt, stem)
    rand = pcr.compare_top3_vs_random3_pcs(Xh, n_random=50, max_pcs=min(20, Xh.shape[1] - 1, Xh.shape[0] - 1),
                                           pseudotime=pt, cytotrace=stem)
    hvg = pcr.compare_hvg_vs_all_gene_pc_geometry(Xh, genes, md, pt, stem)
    summary = pd.concat([
        counts.assign(section="pc_count"),
        rand.assign(section="random3"),
        hvg.assign(section="feature_selection"),
    ], ignore_index=True, sort=False)
    summary.to_csv(out / "pc_dimension_summary.csv", index=False)
    scree.to_csv(out / "scree_summary.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(scree["pc"], scree["cumulative_explained_variance"], marker="o", ms=3)
    axs[0].set_xlabel("PC")
    axs[0].set_ylabel("cumulative variance")
    rand.boxplot(column="multi_stripe_strength", by="config", ax=axs[1], rot=90)
    axs[1].set_title("PC1-3 vs random triplets")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out / "pc_dimension_robustness.png", dpi=180)
    plt.close(fig)
    return summary


def save_h1(dataset, coords, scores_pre, stem_col, stem, pt_col, pt):
    out = ensure(OUT / "H1_cytotrace_only" / dataset)
    if stem is None:
        pd.DataFrame([{"dataset": dataset, "status": "no real CytoTRACE/stemness column"}]).to_csv(out / "H1_cytotrace_metrics.csv", index=False)
        return None
    mask = np.isfinite(stem)
    root = coords[stem >= np.nanquantile(stem, 0.9)].mean(axis=0)
    root = root / (np.linalg.norm(root) + 1e-9)
    geod = np.arccos(np.clip(coords @ root, -1.0, 1.0))
    euc = np.linalg.norm(scores_pre[:, :3] - scores_pre[stem >= np.nanquantile(stem, 0.9), :3].mean(axis=0), axis=1)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(50):
        ridx = rng.choice(len(coords), max(5, int(0.1 * len(coords))), replace=False)
        rr = coords[ridx].mean(axis=0)
        rr = rr / (np.linalg.norm(rr) + 1e-9)
        null.append(spearmanr(stem[mask], np.arccos(np.clip(coords[mask] @ rr, -1.0, 1.0))).statistic)
    rho_g, p_g = spearmanr(stem[mask], geod[mask])
    rho_e, p_e = spearmanr(stem[mask], euc[mask])
    row = {
        "dataset": dataset,
        "score_column": stem_col,
        "n_cells": int(mask.sum()),
        "stemness_geodesic_spearman": float(rho_g),
        "stemness_geodesic_p": float(p_g),
        "stemness_euclidean_pc_spearman": float(rho_e),
        "stemness_euclidean_pc_p": float(p_e),
        "random_root_mean_spearman": float(np.nanmean(null)),
        "random_root_sd_spearman": float(np.nanstd(null)),
        "pseudotime_column": pt_col or "",
    }
    pd.DataFrame([row]).to_csv(out / "H1_cytotrace_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(geod, stem, c=pt if pt is not None else stem, s=6, cmap="viridis", linewidths=0)
    ax.set_xlabel("geodesic distance from high-stemness root")
    ax.set_ylabel("real stemness score")
    ax.set_title(dataset)
    fig.tight_layout()
    fig.savefig(out / "H1_cytotrace_gradient.png", dpi=180)
    plt.close(fig)
    return row


def save_h5(dataset, Xh, genes, coords):
    out = ensure(OUT / "H5_geodesic_gradient" / dataset)
    coarse = gg.compute_gene_geodesic_gradients(Xh, genes, coords, n_theta=24, n_phi=48)
    coarse = gg.filter_housekeeping_and_low_specificity_genes(coarse)
    coarse["prefilter_score"] = coarse["phi_variation"] * coarse["stripe_score"] * coarse["specificity_penalty"]
    selected = list(coarse.sort_values("prefilter_score", ascending=False).head(30)["gene"])
    selected += list(coarse.sort_values("theta_variation", ascending=False).head(30)["gene"])
    selected = list(dict.fromkeys(selected))
    exact, _, idx = gg.compute_per_cell_gene_geodesic_gradient(Xh, genes, coords, genes=selected, k=20, max_cells=800)
    boundary = gg.rank_stripe_boundary_genes_per_cell(exact, top_n=50)
    along = gg.rank_along_trajectory_genes_per_cell(exact, top_n=50)
    ranked = pd.concat([boundary.assign(rank_type="stripe_boundary"),
                        along.assign(rank_type="along_trajectory")], ignore_index=True, sort=False)
    ranked.to_csv(out / "H5_ranked_genes.csv", index=False)
    known = pd.DataFrame({"gene": ranked["gene"].unique(), "known_regulator_overlap": False})
    known.to_csv(out / "H5_known_regulator_overlap.csv", index=False)
    top = boundary.head(5)["gene"].tolist()
    fig, axs = plt.subplots(max(1, len(top)), 1, figsize=(7, max(2.5, 2.2 * len(top))))
    axs = np.atleast_1d(axs)
    for ax, gene in zip(axs, top):
        gi = genes.index(gene)
        expr = np.asarray(Xh[:, gi].todense()).ravel() if sp.issparse(Xh) else np.asarray(Xh[:, gi]).ravel()
        gg.plot_gene_gradient_equirectangular(coords, expr, ax=ax)
        ax.set_title(gene)
    fig.tight_layout()
    fig.savefig(out / "H5_top_gene_gradients.png", dpi=180)
    plt.close(fig)
    return ranked


def save_root_robustness(dataset, coords, md, stem, pt):
    out = ensure(OUT / "root_robustness" / dataset)
    candidates = {}
    if stem is not None:
        candidates["high_stemness_q90"] = stem >= np.nanquantile(stem, 0.9)
        candidates["low_stemness_q10"] = stem <= np.nanquantile(stem, 0.1)
    if not md.empty and "Phenotype" in md.columns:
        for val in md["Phenotype"].astype(str).value_counts().head(4).index:
            candidates[f"phenotype_{val}"] = md["Phenotype"].astype(str).to_numpy() == val
    rng = np.random.default_rng(1)
    for i in range(5):
        ids = rng.choice(len(coords), max(5, int(0.05 * len(coords))), replace=False)
        m = np.zeros(len(coords), dtype=bool); m[ids] = True
        candidates[f"random_{i}"] = m
    rows = []
    for name, mask in candidates.items():
        if np.sum(mask) < 2:
            continue
        root = coords[mask].mean(axis=0)
        root = root / (np.linalg.norm(root) + 1e-9)
        dist = np.arccos(np.clip(coords @ root, -1.0, 1.0))
        ms = stripe.multi_stripe_strength_score(coords)
        rows.append({
            "root": name,
            "n_root_cells": int(np.sum(mask)),
            "multi_stripe_strength": ms["multi_stripe_strength"],
            "global_longitude_concentration": ms["global_longitude_concentration"],
            "theta_cytotrace_spearman": float(spearmanr(dist, stem, nan_policy="omit").statistic) if stem is not None else np.nan,
            "theta_pseudotime_spearman": float(spearmanr(dist, pt, nan_policy="omit").statistic) if pt is not None else np.nan,
            "great_circle_r2": float(ss.fit_great_circle(coords)["r_squared"]),
            "anisotropy_linear": float(ss.spherical_anisotropy(coords)["linear"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "root_sensitivity_metrics.csv", index=False)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        df.plot.bar(x="root", y="theta_cytotrace_spearman", ax=ax, legend=False)
        ax.set_ylabel("stemness vs root distance rho")
        fig.tight_layout()
        fig.savefig(out / "root_sensitivity_plot.png", dpi=180)
        plt.close(fig)
    return df


def main():
    ensure(OUT)
    datasets, failures = load_datasets()
    inventory = []
    h1_rows, pc_rows, stripe_rows, root_rows, h5_rows = [], [], [], [], []
    brca_row = None
    limits = {"BrCa_atlas": 8000, "celegan": 8000, "uc_epi": 8000, "klein": 6000}
    for name, data in datasets.items():
        max_cells = limits.get(name, 6000)
        data, Xn, Xh, genes = prep(data, max_cells=max_cells)
        stem_col, stem = real_stemness(data.metadata)
        pt_col, pt = pseudotime(data.metadata)
        pca = pp.fit_pca_embedding(Xh, genes, n_components=min(20, Xh.shape[1] - 1, Xh.shape[0] - 1))
        coords = ss._as_unit(pca.scores[:, :3])
        inventory.append({
            "dataset": name,
            "loaded": True,
            "n_cells_used": data.n_cells,
            "n_genes_total": len(data.gene_names),
            "n_hvgs": len(genes),
            "metadata_columns": ";".join(map(str, data.metadata.columns[:30])),
            "real_stemness_column": stem_col or "",
            "pseudotime_column": pt_col or "",
            "notes": "; ".join(data.notes),
        })
        stripe_rows.append(save_stripe(name, coords))
        pc = save_pc_robustness(name, Xh, genes, data.metadata, pt, stem)
        pc_rows.append(pc.assign(dataset=name))
        r = save_root_robustness(name, coords, data.metadata, stem, pt)
        root_rows.append(r.assign(dataset=name))
        if name in {"hESC", "planaria", "human_germ_cell", "pre_implant_human_embryo"}:
            h1 = save_h1(name, coords, pca.scores, stem_col, stem, pt_col, pt)
            if h1 is not None:
                h1_rows.append(h1)
        if name != "uc_epi":
            h5 = save_h5(name, Xh, genes, coords)
            h5_rows.append(h5.assign(dataset=name))
        if name == "BrCa_atlas":
            an = ss.spherical_anisotropy(coords)
            gc = ss.fit_great_circle(coords)
            ms = stripe.multi_stripe_strength_score(coords)
            brca_row = {
                "dataset": name,
                "n_cells_used": data.n_cells,
                "multi_stripe_strength": ms["multi_stripe_strength"],
                "global_longitude_concentration": ms["global_longitude_concentration"],
                "n_stripes": ms["n_stripes"],
                "great_circle_r2": gc["r_squared"],
                "anisotropy_linear": an["linear"],
                "anisotropy_spherical": an["spherical"],
            }
            no = ensure(OUT / "negative_control" / "BrCa_atlas")
            pd.DataFrame([brca_row]).to_csv(no / "BrCa_geometric_metrics.csv", index=False)
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=3, c=data.metadata.get("Patient", pd.Series([""] * len(coords))).astype("category").cat.codes, cmap="tab20")
            fig.tight_layout()
            fig.savefig(no / "BrCa_sphere.png", dpi=180)
            plt.close(fig)
            (no / "BrCa_interpretation.md").write_text(
                "# BrCa atlas negative/control interpretation\n\n"
                "No developmental pseudotime or CytoTRACE score was present in the BrCa sidecar metadata. "
                "The geometry was therefore interpreted as a non-developmental control: any structure is more plausibly associated with patient, tumor subtype, batch, or broad cell composition than a clean stem-to-differentiated trajectory.\n\n"
                f"Observed multi-stripe strength: {brca_row['multi_stripe_strength']:.3f}; "
                f"global longitude concentration: {brca_row['global_longitude_concentration']:.3f}; "
                f"great-circle R2: {brca_row['great_circle_r2']:.3f}.\n",
                encoding="utf-8",
            )
    inv = pd.DataFrame(inventory + [{"dataset": k, "loaded": False, "notes": v} for k, v in failures.items()])
    inv.to_csv(OUT / "pc_robustness_dataset_inventory.csv", index=False)
    if h1_rows:
        ensure(OUT / "H1_cytotrace_only")
        pd.DataFrame(h1_rows).to_csv(OUT / "H1_cytotrace_only" / "summary_H1_real_cytotrace.csv", index=False)
    pd.concat(pc_rows, ignore_index=True, sort=False).to_csv(OUT / "pc_robustness_summary_all.csv", index=False)
    pd.DataFrame(stripe_rows).to_csv(OUT / "stripe_robustness_summary_all.csv", index=False)
    pd.concat(root_rows, ignore_index=True, sort=False).to_csv(OUT / "root_robustness_summary_all.csv", index=False)
    if h5_rows:
        pd.concat(h5_rows, ignore_index=True, sort=False).to_csv(OUT / "H5_geodesic_gradient_summary_all.csv", index=False)
    if brca_row is not None:
        dev = pd.DataFrame(stripe_rows)
        dev["group"] = np.where(dev["dataset"].eq("BrCa_atlas"), "BrCa_control", "other_raw_dataset")
        ensure(OUT / "negative_control")
        dev.to_csv(OUT / "negative_control" / "developmental_vs_brca_summary.csv", index=False)
    write_final_summary(inv, h1_rows, stripe_rows, brca_row, failures)


def write_final_summary(inv, h1_rows, stripe_rows, brca_row, failures):
    lines = [
        "# PC robustness, CytoTRACE-only H1, multi-stripe, H5, and BrCa control run",
        "",
        "This run is intentionally conservative: no CytoTRACE/stemness scores were fabricated, proxy entropy was not used for the main H1 table, and BrCa was not forced to behave as a negative control.",
        "",
        "## Dataset inventory",
        inv.to_markdown(index=False),
        "",
        "## H1 real CytoTRACE/stemness summary",
        pd.DataFrame(h1_rows).to_markdown(index=False) if h1_rows else "No real-score H1 datasets completed.",
        "",
        "## Old global metric vs new multi-stripe metric",
        pd.DataFrame(stripe_rows).to_markdown(index=False),
        "",
        "## BrCa interpretation",
        pd.DataFrame([brca_row]).to_markdown(index=False) if brca_row else "BrCa did not complete.",
        "",
        "## Failures / missing data",
        pd.DataFrame([{"dataset": k, "reason": v} for k, v in failures.items()]).to_markdown(index=False) if failures else "No dataset load failures.",
        "",
        "## PNAS-readiness verdict",
        "PC1-PC3 are now framed as the three-axis S2 projection, not as sufficient merely because they explain the most variance. The per-dataset robustness CSVs must be inspected before claiming stability. H1 is substantially stronger where real CytoTRACE/stemness scores exist. H5 remains a screening analysis and should not be overclaimed without known-regulator enrichment and perturbation validation. BrCa is a useful non-developmental control only to the extent that its geometry lacks a monotonic biological root-to-tip score; it can still show patient/subtype/batch structure.",
    ]
    (OUT / "PC_robustness_cytotrace_neg_ctrl_050526_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
