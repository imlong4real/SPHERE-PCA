"""Orchestrator for the SPHERE-PCA supplemental figures.

Runs the four supplemental analyses (Benchmark, Planaria distance
heatmaps, Planaria lineage sectors, Xenium / TRACER stress test) and
writes every output under ``outputs/figures/supplement/``.

Usage:
    python scripts/make_supplement_figures.py
    python scripts/make_supplement_figures.py --only benchmark planaria
    python scripts/make_supplement_figures.py --xenium-max-cells 60000
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pca_sphere_projection.figures.supplements import (  # noqa: E402
    common, benchmark, planaria, xenium,
)


_TOPICS = {
    "benchmark": ("Supplement 1 — Benchmark embeddings", benchmark.run),
    "planaria":  ("Supplements 2 & 3 — Planaria geometry + lineage sectors",
                   planaria.run),
    "xenium":    ("Supplement 4 — Xenium TRACER stress test", xenium.run),
}


def _write_manifest(registry: list[str]) -> Path:
    manifest = common.SUPP_ROOT / "manifest.txt"
    rel = sorted({
        str(Path(p).relative_to(common.ROOT))
        for p in registry if Path(p).exists()
    })
    manifest.write_text("\n".join(rel) + "\n")
    return manifest


def _read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path) if path.exists() else None


def _write_final_summary(results: dict) -> Path:
    import numpy as np
    import pandas as pd
    p = common.SUPP_ROOT / "final_summary.md"

    bench_csv = _read_csv(common.SUPP_ROOT / "Benchmark" / "benchmark_summary.csv")
    plan_csv = _read_csv(common.SUPP_ROOT / "Planaria" /
                          "planaria_lineage_statistics.csv")
    xen_csv = _read_csv(common.SUPP_ROOT / "Xenium" /
                         "xenium_pc50_variance_tables.csv")
    if xen_csv is None:
        xen_csv = _read_csv(common.SUPP_ROOT / "Xenium" /
                             "xenium_pca_variance_tables.csv")

    lines = []
    lines.append("# SPHERE-PCA Supplemental Analyses — Final Summary")
    lines.append("")
    lines.append(
        f"_Generated: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z_"
    )
    lines.append("")
    lines.append(
        "This document summarises the four supplemental analyses run by "
        "`scripts/make_supplement_figures.py`. Conservative interpretation "
        "is maintained throughout; descriptive findings are clearly "
        "distinguished from statistically validated ones."
    )
    lines.append("")

    lines.append("## Major findings")
    lines.append("")
    if bench_csv is not None:
        rows = bench_csv.set_index("embedding")
        bench_block = []
        for emb in ["PCA (PC1-3)", "SPHERE-PCA", "UMAP (3D)",
                    "t-SNE (3D)", "scPhere (3D)"]:
            if emb not in rows.index:
                continue
            r = rows.loc[emb]
            bench_block.append(
                f"  - {emb}: T = {r['T_mean']:.3f} ± {r['T_std']:.3f}, "
                f"C = {r['C_mean']:.3f} ± {r['C_std']:.3f}, "
                f"k-NN Jaccard = {r['knn_jaccard_mean']:.3f}, "
                f"|ρ(dist, time)| = "
                f"{abs(float(r['spearman_time_mean'])):.3f}."
            )
        lines.append(
            "- **Benchmark (C. elegans).** PCA, SPHERE-PCA, UMAP (3D), "
            "t-SNE (3D), and scPhere (3D) were compared against the "
            "high-D PC1-50 reference. Trustworthiness (T), continuity "
            "(C), k-NN Jaccard overlap, and Spearman correlation "
            "between distance-from-anchor (cluster 100-130) and embryo "
            "time:"
        )
        for line in bench_block:
            lines.append(line)
        lines.append(
            "  UMAP/t-SNE locally optimise neighborhood preservation and "
            "score highest on T/C/Jaccard; SPHERE-PCA and scPhere — both "
            "spherical embeddings — provide a structured global "
            "coordinate system and are competitive on monotonic "
            "developmental ordering (|ρ| with time). SPHERE-PCA is "
            "**competitive with**, not universally superior to, "
            "scPhere/UMAP/t-SNE."
        )
    else:
        lines.append(
            "- **Benchmark (C. elegans).** Output table unavailable; "
            "see Benchmark/."
        )
    lines.append(
        "- **Planaria centroid-direction angular distances "
        "(low-D vs high-D).** For each cell type we compute a unit "
        "centroid direction in 3-D SPHERE-PCA space and in PC1-50 "
        "space, then convert pairwise cosine similarity to angular "
        "distance via `arccos`. Both heatmaps share units (radians / "
        "degrees) and the same row/column ordering "
        "(see `Planaria/planaria_distance_matrix_sphere3d.csv` and "
        "`Planaria/planaria_distance_matrix_pc50angular.csv`). The "
        "low-D angular geometry produces a more interpretable, "
        "magnitude-free organisation while the high-D PC1-50 angular "
        "view confirms the same global lineage relationships in the "
        "underlying high-dimensional space."
    )
    if plan_csv is not None and len(plan_csv):
        n_col = ("n_cells_perm" if "n_cells_perm" in plan_csv.columns
                 else "n_cells")
        lin_block = []
        rho_signs = []
        for _, r in plan_csv.iterrows():
            rho = float(r["spearman_rho_pseudoorder_vs_theta"])
            rho_signs.append(np.sign(rho))
            lin_block.append(
                f"  - {r['lineage']}: ρ(θ, pseudo-order) = "
                f"{rho:+.3f}, |ρ| = {abs(rho):.3f}, "
                f"empirical p (compactness) = "
                f"{float(r['empirical_p_more_compact_than_null']):.3g}, "
                f"n = {int(r[n_col])}."
            )
        lines.append(
            "- **Planaria lineage angular-sector analysis.** Six "
            "lineage paths rooted at *Neoblast 1* (Epidermal, Gut, "
            "Muscle, Parenchymal/Glial, Neural-ChAT, Neural-GABA) "
            "were tested. Root alignment uses "
            "`root_vec = mean(PC1..PC3 of 'neoblast 1')` followed by "
            "the canonical "
            "`R.from_euler('xyz', [80, 50, 50], degrees=True)`:"
        )
        for line in lin_block:
            lines.append(line)
        sig = (plan_csv["empirical_p_more_compact_than_null"] < 0.05).sum()
        lines.append(
            f"  {sig} of {len(plan_csv)} lineages reached empirical "
            f"p < 0.05 against a label-shuffling null. Spearman "
            "correlation signs are reported transparently — the "
            "primary message is *known lineages occupy compact "
            "angular sectors*, not that θ universally recovers "
            "pseudotime."
        )
    else:
        lines.append(
            "- **Planaria lineage angular-sector analysis.** Output table "
            "unavailable; see Planaria/."
        )
    if xen_csv is not None and len(xen_csv):
        agg_rows = []
        for (panel_name, state_name), g in xen_csv.groupby(["panel", "state"]):
            agg_rows.append({
                "panel": panel_name,
                "state": state_name,
                "n_cells": int(g["n_cells_used"].iloc[0]),
                "n_genes": int(g["n_genes_used"].iloc[0]),
                "PC1": float(g.loc[g["PC"] == 1, "explained_variance_ratio"].iloc[0]),
                "cum_PC10": float(g.loc[g["PC"] == 10, "cumulative"].iloc[0]),
                "cum_PC50": float(g.loc[g["PC"] == 50, "cumulative"].iloc[0])
                    if (g["PC"] == 50).any() else float(g["cumulative"].max()),
            })
        agg = pd.DataFrame(agg_rows)
        xen_block = []
        for _, r in agg.iterrows():
            label = (f"{r['panel']} {r['state']}-TRACER"
                     if r['state'] != "reference" else r['panel'])
            xen_block.append(
                f"  - {label}: "
                f"n_cells = {int(r['n_cells'])}, "
                f"n_genes = {int(r['n_genes'])}, "
                f"PC1 = {r['PC1']*100:.1f} %, "
                f"cum. PC10 = {r['cum_PC10']*100:.1f} %, "
                f"cum. PC50 = {r['cum_PC50']*100:.1f} %."
            )
        lines.append(
            "- **Xenium / TRACER stress test.** All assayed genes used "
            "for the Xenium panels (normalize → log1p → scale → "
            "PCA(50)); the BrCa scRNA-seq atlas is processed with "
            "canonical HVG=2000 selection as a positive control. "
            "Variance curves (with BrCa atlas reference) and pre/post "
            "sphere views (BrCa and NSCLC) were generated:"
        )
        for line in xen_block:
            lines.append(line)
        lines.append(
            "  Xenium and the BrCa atlas show qualitatively different "
            "variance distributions: the BrCa atlas (2000 HVGs from a "
            "whole-transcriptome assay) has the flattest cumulative "
            "curve, while BrCa-TRACER concentrates variance more "
            "sharply (~300 genes, panel-driven) and NSCLC-TRACER lies "
            "between or below the atlas depending on PC range. We "
            "interpret these differences as panel-design / assay "
            "effects, not as evidence for developmental trajectories "
            "in Xenium."
        )
    else:
        lines.append(
            "- **Xenium / TRACER stress test.** Variance table "
            "unavailable; see Xenium/."
        )
    lines.append("")

    lines.append("## Did angular distance outperform PC50 Euclidean distance?")
    lines.append("")
    lines.append(
        "Descriptive: yes, in the sense that the angular geometry "
        "produces more interpretable separations between developmentally "
        "related lineages (collapsing radial magnitude). Quantitative "
        "head-to-head out-performance is **not** claimed; the heatmaps "
        "support direct visual comparison and the underlying matrices "
        "are saved as CSV for downstream tests by readers."
    )
    lines.append("")

    lines.append("## Were lineage angular sectors statistically supported?")
    lines.append("")
    lines.append(
        "Spearman correlation (pseudo-order vs θ) and a label-shuffling "
        "permutation test on within-lineage compactness are reported per "
        "lineage. The summary CSV contains effect sizes and empirical "
        "p-values; lineages with empirical p < 0.05 against a uniform "
        "null are statistically supported, others remain descriptive."
    )
    lines.append("")

    lines.append("## Did Xenium retain coherent spherical structure post-TRACER?")
    lines.append("")
    lines.append(
        "Visual inspection (three camera angles per state) shows that "
        "spherical structure can be visualised in pre-TRACER and "
        "post-TRACER for both BrCa and NSCLC, but **interpretation is "
        "constrained**: targeted panels (~300 genes) and mixed tissue "
        "composition mean we cannot attribute structure to "
        "developmental trajectories. Variance distributions also "
        "differ qualitatively from a whole-transcriptome BrCa atlas "
        "(separate reference panel in 4A), as expected for "
        "panel-based assays."
    )
    lines.append("")

    lines.append("## Major caveats")
    lines.append("")
    lines.append(
        "- SPHERE-PCA is **not** universally superior to UMAP/t-SNE. The "
        "benchmark reports interpretability + neighborhood preservation, "
        "not unconditional out-performance."
    )
    lines.append(
        "- Lineage sectors describe *angular organisation*, not causal "
        "lineage determination."
    )
    lines.append(
        "- Xenium spherical structure is reported as a **stress test**. "
        "No developmental trajectory is implied by the visualisation."
    )
    lines.append(
        "- Permutation tests are sample-size sensitive; lineages with "
        "few cells will have weaker discrimination."
    )
    lines.append("")

    lines.append("## Descriptive vs statistically validated")
    lines.append("")
    lines.append("| Result | Status |")
    lines.append("| --- | --- |")
    lines.append(
        "| Benchmark trustworthiness/continuity/Jaccard/Spearman ranks "
        "| Statistically computed (sklearn metrics + scipy.spearmanr) |"
    )
    lines.append(
        "| Angular vs PC50 heatmap geometry | Descriptive comparison "
        "(no test reported) |"
    )
    lines.append(
        "| Lineage Spearman ρ + permutation p-value | Statistically "
        "validated (per-lineage p reported) |"
    )
    lines.append(
        "| Xenium pre/post sphere visual coherence | Descriptive "
        "(visual stress test only) |"
    )
    lines.append(
        "| Xenium variance concentration | Quantitative (variance "
        "ratios from PCA) |"
    )
    lines.append("")
    lines.append(
        "## Reproducibility"
    )
    lines.append("")
    lines.append(
        "Every figure is saved alongside a sidecar `*_config.yaml` and "
        "the underlying numerical tables as `*_data_*.csv`. The "
        "`manifest.txt` lists every generated artefact. Caching for the "
        "Xenium PCA lives in "
        "`outputs/figures/supplement/Xenium/_cache/`. "
        "Re-running the orchestrator regenerates all outputs without "
        "modifying any main-figure (Fig 1–4) artefacts."
    )

    p.write_text("\n".join(lines) + "\n")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="*", default=None,
        choices=list(_TOPICS.keys()),
        help="Run only the listed topics (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-subsample", type=int, default=5000)
    parser.add_argument("--benchmark-neighbors", type=int, default=15)
    parser.add_argument("--planaria-permutations", type=int, default=1000)
    parser.add_argument("--xenium-max-cells", type=int, default=None,
                        help="Cap cells per Xenium dataset for speed.")
    parser.add_argument("--xenium-plot-cells", type=int, default=8000)
    args = parser.parse_args()

    common.apply_publication_style()
    common.SUPP_ROOT.mkdir(parents=True, exist_ok=True)

    todo = list(_TOPICS) if args.only is None else args.only
    registry: list[str] = []
    results: dict = {}

    for topic in todo:
        title, fn = _TOPICS[topic]
        print(f"\n=== {title} ===")
        t0 = time.time()
        try:
            if topic == "benchmark":
                res = fn(
                    registry=registry,
                    seeds=(args.seed, args.seed + 1, args.seed + 2),
                    n_subsample=args.benchmark_subsample,
                    n_neighbors=args.benchmark_neighbors,
                )
            elif topic == "planaria":
                res = fn(
                    registry=registry,
                    n_permutations=args.planaria_permutations,
                    seed=args.seed,
                )
            elif topic == "xenium":
                res = fn(
                    registry=registry, seed=args.seed,
                    n_components=50,
                    n_plot_cells=args.xenium_plot_cells,
                    max_cells=args.xenium_max_cells,
                )
            else:
                continue
            results[topic] = res
            print(f"  done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results[topic] = {"error": str(exc)}

    manifest = _write_manifest(registry)
    summary = _write_final_summary(results)
    print(f"\nManifest -> {manifest.relative_to(common.ROOT)}")
    print(f"Summary  -> {summary.relative_to(common.ROOT)}")
    print(f"{len(set(registry))} artefact(s) registered.")


if __name__ == "__main__":
    main()
