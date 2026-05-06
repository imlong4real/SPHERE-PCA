"""
End-to-end raw-expression validation workflow.

Run with:

    python scripts/run_raw_expression_validation.py \
        --raw-data raw_data \
        --out outputs/raw_expression_validation

Produces, per dataset, under ``<out>/<dataset>/``:

    - pca_comparison_metrics.csv
    - H1_entropy_gradient_metrics.csv
    - H4_radial_angular_metrics.csv
    - H5_stripe_boundary_genes.csv
    - H5_along_trajectory_genes.csv
    - H7_perturbation_gene_rankings.csv
    - per_cell_geometry.parquet (PC1-3 + theta + phi + radial_norm + entropy)
    - figures/ (filled later by scripts/plot_raw_validation_figures.py)

A combined ``<out>/final_summary.md`` is written at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pca_sphere_projection as psp  # noqa: E402
from pca_sphere_projection import (  # noqa: E402
    io as psp_io,
    preprocessing as psp_pp,
    entropy as psp_ent,
    gene_geometry as psp_gg,
    known_regulators as psp_known,
    perturbation as psp_pert,
    sphere_stats as psp_sph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_unit(coords):
    coords = np.asarray(coords, dtype=float)
    n = np.linalg.norm(coords, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return coords / n


def _safe_corr(a, b, method="spearman"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return float("nan"), float("nan")
    sa = pd.Series(a[mask])
    sb = pd.Series(b[mask])
    if method == "spearman":
        from scipy.stats import spearmanr
        rho, p = spearmanr(sa, sb)
        return float(rho), float(p)
    from scipy.stats import pearsonr
    r, p = pearsonr(sa, sb)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Per-hypothesis routines
# ---------------------------------------------------------------------------

def run_h1(coords3, X_norm, gene_names, metadata, ds_name: str,
           cytotrace_precomputed: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """H1: entropy / stemness vs. geodesic distance from root.

    Returns dict with:
        - score_method
        - geodesic / euclidean correlations
        - per_cell DataFrame
    """
    if cytotrace_precomputed is not None:
        # CytoTRACE rank: lower rank = more stem-like in CytoTRACE convention.
        # Convert to a 0..1 stemness score where 1 = most stem-like.
        ranks = np.asarray(cytotrace_precomputed, dtype=float)
        score_method = "CytoTRACE (precomputed)"
        if np.nanmax(ranks) > 1.5:
            n = ranks.size
            score = 1.0 - (ranks - np.nanmin(ranks)) / (np.nanmax(ranks) - np.nanmin(ranks) + 1e-9)
        else:
            score = ranks
    else:
        score_method = "Shannon entropy (proxy)"
        score = psp_ent.compute_transcriptional_entropy(X_norm, gene_names, "shannon")

    # Pseudotime if available
    pseudotime = None
    if "embryo_time_mid" in metadata.columns:
        pseudotime = pd.to_numeric(metadata["embryo_time_mid"], errors="coerce").values
    elif "day" in metadata.columns:
        pseudotime = metadata["day"].astype(str).map(
            lambda s: int(s.replace("Day", "").replace("d", "")) if any(c.isdigit() for c in s) else np.nan
        ).values

    res = psp_ent.test_entropy_geodesic_gradient(
        coords=coords3,
        entropy_score=score,
        root_coords=None,
        pseudotime=pseudotime,
        pc_scores_for_euclid=coords3,
    )
    res["score_method"] = score_method
    return res


def run_h4(pc_scores_pre: np.ndarray, coords3_unit: np.ndarray,
           metadata: pd.DataFrame, X_norm, gene_names, dataset: str,
           entropy_score: np.ndarray) -> Dict[str, Any]:
    """H4: radial norm vs. angular theta -- which carries which biology.

    pc_scores_pre: (N, 3) PC1-3 BEFORE L2 normalisation.
    coords3_unit: (N, 3) L2-normalised PC1-3.
    """
    radial = np.linalg.norm(pc_scores_pre, axis=1)
    theta = np.arccos(np.clip(coords3_unit[:, 2], -1.0, 1.0))

    rows = []

    # Cell-cycle score (mean expression of cycle genes if present)
    cc_score = _gene_set_score(X_norm, gene_names, psp_pp._CYCLE_GENES)
    if cc_score is not None:
        rho, p = _safe_corr(radial, cc_score)
        rows.append({"variable": "cell_cycle_score", "vs": "radial",
                     "spearman_rho": rho, "p_value": p})
        rho, p = _safe_corr(theta, cc_score)
        rows.append({"variable": "cell_cycle_score", "vs": "theta",
                     "spearman_rho": rho, "p_value": p})

    # Mito and ribo scores
    mito_score = _gene_set_score_re(X_norm, gene_names, psp_pp._MITO_RE)
    ribo_score = _gene_set_score_re(X_norm, gene_names, psp_pp._RIBO_RE)
    for name, sc in [("mito_score", mito_score), ("ribo_score", ribo_score)]:
        if sc is not None:
            rho, p = _safe_corr(radial, sc)
            rows.append({"variable": name, "vs": "radial", "spearman_rho": rho, "p_value": p})
            rho, p = _safe_corr(theta, sc)
            rows.append({"variable": name, "vs": "theta", "spearman_rho": rho, "p_value": p})

    # Entropy
    rho, p = _safe_corr(radial, entropy_score)
    rows.append({"variable": "entropy_score", "vs": "radial", "spearman_rho": rho, "p_value": p})
    rho, p = _safe_corr(theta, entropy_score)
    rows.append({"variable": "entropy_score", "vs": "theta", "spearman_rho": rho, "p_value": p})

    # Pseudotime / day if present
    if "embryo_time_mid" in metadata.columns:
        pt = pd.to_numeric(metadata["embryo_time_mid"], errors="coerce").values
    elif "day" in metadata.columns:
        pt = metadata["day"].astype(str).map(
            lambda s: int(s.replace("Day", "").replace("d", "")) if any(c.isdigit() for c in s) else np.nan
        ).values
    else:
        pt = None
    if pt is not None:
        rho, p = _safe_corr(radial, pt)
        rows.append({"variable": "pseudotime/day", "vs": "radial", "spearman_rho": rho, "p_value": p})
        rho, p = _safe_corr(theta, pt)
        rows.append({"variable": "pseudotime/day", "vs": "theta", "spearman_rho": rho, "p_value": p})

    # Radial vs theta
    rho, p = _safe_corr(radial, theta)
    rows.append({"variable": "theta", "vs": "radial", "spearman_rho": rho, "p_value": p})

    return {"per_variable": pd.DataFrame(rows), "radial": radial, "theta": theta}


def _gene_set_score(X, gene_names, gene_set):
    gn_lower = {g.lower(): i for i, g in enumerate(gene_names)}
    idxs = [gn_lower[g.lower()] for g in gene_set if g.lower() in gn_lower]
    if not idxs:
        return None
    import scipy.sparse as sp
    if sp.issparse(X):
        sub = X[:, idxs].toarray()
    else:
        sub = X[:, idxs]
    return sub.mean(axis=1)


def _gene_set_score_re(X, gene_names, regex):
    idxs = [i for i, g in enumerate(gene_names) if regex.match(g)]
    if not idxs:
        return None
    import scipy.sparse as sp
    if sp.issparse(X):
        sub = X[:, idxs].toarray()
    else:
        sub = X[:, idxs]
    return sub.mean(axis=1)


def run_h5(X_norm, gene_names, coords3_unit) -> Dict[str, Any]:
    """H5: identify stripe-boundary vs along-trajectory genes."""
    grad_df = psp_gg.compute_gene_geodesic_gradients(
        X_norm, gene_names, coords3_unit, n_theta=18, n_phi=36)
    boundary = psp_gg.rank_stripe_boundary_genes(grad_df, top_n=50)
    along = psp_gg.rank_along_trajectory_genes(grad_df, top_n=50)
    return {"all": grad_df, "stripe_boundary": boundary, "along_trajectory": along}


def run_h7(X_norm, gene_names, coords3_unit, candidate_genes,
           pca_loadings: np.ndarray, pca_mean: np.ndarray) -> Dict[str, Any]:
    """H7: fixed-loading perturbation sensitivity vector fields."""
    # Drop genes not present in the matrix
    gene_set = set(gene_names)
    cands = [g for g in candidate_genes if g in gene_set]
    if not cands:
        return {"ranking": pd.DataFrame(), "results": {}, "decompositions": {}}

    import scipy.sparse as sp
    if sp.issparse(X_norm):
        X_dense = X_norm.toarray()
    else:
        X_dense = np.asarray(X_norm)

    pert = psp_pert.compute_gene_perturbation_vectors(
        X_dense, gene_names, cands, pca_loadings, pca_mean,
        normalize_to_sphere=True,
    )
    rank_df = psp_pert.rank_genes_by_perturbation_magnitude(pert, "symmetric_range")

    decomp_rows = []
    for g, r in pert.items():
        v = r["doubled"] - r["zeroed"]
        d = psp_pert.decompose_perturbation_vectors(r["original"], v)
        decomp_rows.append({
            "gene": g,
            "global_magnitude": float(np.linalg.norm(v, ord="fro")),
            "mean_radial": float(np.mean(d["radial"])),
            "mean_abs_theta": float(np.mean(np.abs(d["theta_component"]))),
            "mean_abs_phi": float(np.mean(np.abs(d["phi_component"]))),
            "mean_tangent_mag": float(np.mean(d["tangent_magnitude"])),
        })
    decomp_df = pd.DataFrame(decomp_rows).sort_values("global_magnitude", ascending=False).reset_index(drop=True)
    return {"ranking": rank_df, "decompositions": decomp_df, "results": pert}


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def process_dataset(name: str, expr: psp_io.ExpressionData, out_dir: Path,
                    pc_csv_path: Optional[Path] = None,
                    n_top_genes: int = 2000,
                    n_perturbation_genes: int = 30,
                    log: list = None) -> Dict[str, Any]:
    if log is None: log = []
    t0 = time.time()
    ds_dir = _ensure_dir(out_dir / name)
    fig_dir = _ensure_dir(ds_dir / "figures")
    log.append(f"[{name}] {expr.n_cells} cells x {expr.n_genes} genes")

    # 1. Normalise
    X_norm = psp_pp.normalize_log1p(expr.X)
    log.append(f"[{name}] normalize_log1p done")

    # 2. PCA comparison HVG vs all
    pseudotime = None
    if "embryo_time_mid" in expr.metadata.columns:
        pseudotime = pd.to_numeric(expr.metadata["embryo_time_mid"], errors="coerce").values
    elif "day" in expr.metadata.columns:
        pseudotime = expr.metadata["day"].astype(str).map(
            lambda s: int(s.replace("Day", "").replace("d", "")) if any(c.isdigit() for c in s) else np.nan
        ).values

    # The "all genes" PCA on celegan (86k x 2766) and uc_epi (64k x 1361) is
    # heavy; we compare configurations on a 20% random subsample of cells
    # for the comparison metrics, then fit the canonical HVG PCA on the full
    # dataset for downstream H1-H7. This trade-off is the only place we
    # subsample; everything else uses all cells.
    sub_n = min(expr.n_cells, 8000)
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(expr.n_cells, sub_n, replace=False)
    pca_compare = psp_pp.compare_hvg_vs_all_gene_pca(
        X_norm[sub_idx], expr.gene_names,
        n_top_genes=min(n_top_genes, expr.n_genes - 1),
        n_components=3,
        pseudotime=(pseudotime[sub_idx] if pseudotime is not None else None),
        include_random=True,
        exclude_mito_ribo=True,
    )
    pca_compare["subsample_n"] = sub_n
    pca_compare.to_csv(ds_dir / "pca_comparison_metrics.csv", index=False)
    log.append(f"[{name}] pca_comparison_metrics.csv written")

    # 3. Canonical HVG PCA on full data (3 components, used for H1-H7)
    n_top = min(n_top_genes, expr.n_genes - 1)
    hvg_idx, hvg_names, X_hvg = psp_pp.select_hvgs(X_norm, expr.gene_names, n_top)
    pca = psp_pp.fit_pca_embedding(X_hvg, hvg_names, n_components=3)
    pc_scores_pre = pca.scores  # (N, 3) BEFORE L2 normalisation
    coords3_unit = _to_unit(pc_scores_pre)
    log.append(f"[{name}] PCA on {len(hvg_names)} HVGs; explained var "
               f"PC1-3 = {pca.explained_variance_ratio[:3].round(3).tolist()}")

    # 4. PC CSV match (advisory)
    pc_match = {"matched": False}
    if pc_csv_path and pc_csv_path.exists():
        pc_df = pd.read_csv(pc_csv_path, index_col=0)
        pc_match = psp_io.match_expression_to_pc_csv(expr, pc_df)
        log.append(f"[{name}] PC CSV match: {pc_match.get('matched')} / "
                   f"{pc_match.get('method')} / n={pc_match.get('n_matched')}")

    # 5. H1
    cyto = None
    if "CytoTRACE" in expr.metadata.columns:
        cyto = expr.metadata["CytoTRACE"].astype(float).values
    h1 = run_h1(coords3_unit, X_norm, expr.gene_names, expr.metadata, name, cyto)
    h1_metrics = {k: v for k, v in h1.items() if k != "per_cell"}
    pd.DataFrame([h1_metrics]).to_csv(ds_dir / "H1_entropy_gradient_metrics.csv", index=False)
    log.append(f"[{name}] H1 done; method={h1_metrics['score_method']}; "
               f"rho_geodesic={h1_metrics['geodesic_spearman_rho']:.3f}")
    entropy_score_for_save = h1["per_cell"]["entropy_score"].values

    # 6. H4
    h4 = run_h4(pc_scores_pre, coords3_unit, expr.metadata, X_norm,
                expr.gene_names, name, entropy_score_for_save)
    h4["per_variable"].to_csv(ds_dir / "H4_radial_angular_metrics.csv", index=False)
    log.append(f"[{name}] H4 done")

    # 7. H5
    h5 = run_h5(X_norm, expr.gene_names, coords3_unit)
    # Add known-regulator overlap column
    boundary = h5["stripe_boundary"].copy()
    boundary["is_known_regulator"] = boundary["gene"].isin(psp_known.get_regulator_set(name))
    boundary.to_csv(ds_dir / "H5_stripe_boundary_genes.csv", index=False)
    along = h5["along_trajectory"].copy()
    along["is_known_regulator"] = along["gene"].isin(psp_known.get_regulator_set(name))
    along.to_csv(ds_dir / "H5_along_trajectory_genes.csv", index=False)
    log.append(f"[{name}] H5 done; top boundary genes overlap known: "
               f"{int(boundary['is_known_regulator'].sum())}")

    # 8. H7: candidate genes = (top H5 stripe-boundary U known regulators)
    candidate_genes = []
    candidate_genes.extend(boundary.head(20)["gene"].tolist())
    known = psp_known.get_regulator_set(name)
    candidate_genes.extend([g for g in known if g in expr.gene_names][:20])
    candidate_genes = list(dict.fromkeys(candidate_genes))[:n_perturbation_genes]
    # H7 requires 3 PCs and the loadings on the SAME gene-space we project;
    # here X_norm is the full gene space; pca was fit on HVGs only, so we
    # need to reconstruct loadings on the full gene set. Re-fit a 3-PC PCA
    # on the *full* X_norm so loadings span all gene_names. To keep this
    # tractable for celegan/uc_epi we always fit 3 components on full set.
    pca_full = psp_pp.fit_pca_embedding(X_norm, expr.gene_names, n_components=3)
    pc_scores_full_pre = pca_full.scores
    coords3_unit_full = _to_unit(pc_scores_full_pre)
    h7 = run_h7(X_norm, expr.gene_names, coords3_unit_full, candidate_genes,
                pca_full.components, pca_full.mean)
    if not h7["decompositions"].empty:
        df7 = h7["decompositions"].copy()
        df7["is_known_regulator"] = df7["gene"].isin(psp_known.get_regulator_set(name))
        df7.to_csv(ds_dir / "H7_perturbation_gene_rankings.csv", index=False)
        h7["decompositions"] = df7
    else:
        pd.DataFrame().to_csv(ds_dir / "H7_perturbation_gene_rankings.csv", index=False)
    log.append(f"[{name}] H7 done; n_candidates={len(candidate_genes)}")

    # 9. Per-cell geometry table
    radial_norm = np.linalg.norm(pc_scores_pre, axis=1)
    theta = np.arccos(np.clip(coords3_unit[:, 2], -1.0, 1.0))
    phi = np.arctan2(coords3_unit[:, 1], coords3_unit[:, 0])
    geom = pd.DataFrame({
        "cell_id": expr.cell_ids,
        "PC1": pc_scores_pre[:, 0],
        "PC2": pc_scores_pre[:, 1],
        "PC3": pc_scores_pre[:, 2],
        "x": coords3_unit[:, 0],
        "y": coords3_unit[:, 1],
        "z": coords3_unit[:, 2],
        "theta": theta,
        "phi": phi,
        "radial_norm": radial_norm,
        "entropy_score": entropy_score_for_save,
    })
    if not expr.metadata.empty:
        for c in expr.metadata.columns:
            geom[c] = expr.metadata[c].values
    geom.to_csv(ds_dir / "per_cell_geometry.csv", index=False)
    log.append(f"[{name}] per_cell_geometry.csv written")

    # 10. Save in-memory artefacts for the plotting script.
    # We keep both HVG-PCA and full-gene-PCA loadings: HVG is the canonical
    # H1-H5 embedding; full-gene is the one H7 perturbation uses (loadings
    # must span all gene_names because perturbation modifies one column).
    coords3_unit_full_save = _to_unit(pca_full.scores)
    np.savez_compressed(ds_dir / "_artifacts.npz",
                        pc_scores_pre=pc_scores_pre,
                        coords3_unit=coords3_unit,
                        pca_components=pca.components,
                        pca_mean=pca.mean,
                        explained_var=pca.explained_variance_ratio,
                        pca_components_full=pca_full.components,
                        pca_mean_full=pca_full.mean,
                        coords3_unit_full=coords3_unit_full_save,
                        radial_norm=radial_norm,
                        theta=theta,
                        phi=phi,
                        entropy_score=entropy_score_for_save,
                        )

    # 11. Caption manifest
    caps = [
        f"Dataset: {name}",
        f"Cells: {expr.n_cells}; genes: {expr.n_genes}; HVGs used: {len(hvg_names)}",
        f"Notes: {'; '.join(expr.notes) if expr.notes else 'none'}",
        f"H1 method: {h1_metrics['score_method']}",
        f"H1 geodesic Spearman rho = {h1_metrics['geodesic_spearman_rho']:.3f} (p={h1_metrics['geodesic_spearman_p']:.2e})",
        f"H1 Euclidean Spearman rho = {h1_metrics['euclidean_spearman_rho']:.3f}",
        f"H7 candidate genes: {', '.join(candidate_genes[:10])}{'...' if len(candidate_genes)>10 else ''}",
    ]
    (ds_dir / "captions.md").write_text("\n".join(caps) + "\n")

    log.append(f"[{name}] all hypotheses processed in {time.time()-t0:.1f}s")
    return {
        "h1": h1_metrics,
        "h4": h4["per_variable"],
        "h5_boundary": boundary,
        "h5_along": along,
        "h7": h7["decompositions"],
        "candidate_genes": candidate_genes,
        "n_cells": expr.n_cells,
        "n_genes": expr.n_genes,
        "expr_notes": expr.notes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", type=Path, default=Path("raw_data"))
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/raw_expression_validation"))
    parser.add_argument("--datasets", nargs="*",
                        default=["celegan", "uc_epi", "klein", "hesc"],
                        help="Subset of {celegan, uc_epi, klein, hesc}")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--n-perturbation-genes", type=int, default=30)
    parser.add_argument("--max-cells", type=int, default=20000,
                        help="Cap on cells per dataset (random subsample if exceeded). "
                             "Use 0 for no cap.")
    args = parser.parse_args()

    out = _ensure_dir(args.out)
    log = []
    summaries: Dict[str, Any] = {}

    for ds in args.datasets:
        print(f"\n=== {ds} ===")
        try:
            if ds == "celegan":
                expr = psp_io.load_celegan(str(args.raw_data / "celegan"))
                pc_csv = ROOT / "examples" / "celegan_pca.csv"
            elif ds == "uc_epi":
                expr = psp_io.load_uc_epi(str(args.raw_data / "uc_epi"))
                pc_csv = ROOT / "examples" / "uc_epi_pca.csv"
            elif ds == "klein":
                expr = psp_io.load_bz2_klein_dataset(str(args.raw_data / "klein"))
                pc_csv = ROOT / "examples" / "klein_pca.csv"
            elif ds == "hesc":
                expr = psp_io.load_hesc(str(args.raw_data / "hESC"))
                pc_csv = None
            else:
                print(f"  skipping unknown dataset {ds!r}")
                continue
            # cell cap
            if args.max_cells and expr.n_cells > args.max_cells:
                rng = np.random.default_rng(0)
                idx = rng.choice(expr.n_cells, args.max_cells, replace=False)
                idx.sort()
                expr = psp_io._reindex_expression(expr, idx)
                expr.notes.append(f"random subsample of {args.max_cells} cells (cap)")
                print(f"  subsampled to {args.max_cells} cells")

            res = process_dataset(
                ds, expr, out,
                pc_csv_path=pc_csv,
                n_top_genes=args.n_top_genes,
                n_perturbation_genes=args.n_perturbation_genes,
                log=log,
            )
            summaries[ds] = res
        except Exception as exc:  # don't let one dataset block the others
            tb = traceback.format_exc()
            log.append(f"[{ds}] FAILED: {exc}\n{tb}")
            print(f"[{ds}] FAILED: {exc}")
            (out / f"{ds}_FAILED.log").write_text(tb)

    (out / "run_log.txt").write_text("\n".join(log) + "\n")
    _write_final_summary(out, summaries)
    print("\nLog:")
    print("\n".join(log[-20:]))
    print(f"\nWrote {out}/final_summary.md")


def _write_final_summary(out: Path, summaries: Dict[str, Any]):
    rows = []
    for ds, res in summaries.items():
        h1 = res["h1"]
        n_known_h5 = int(res["h5_boundary"]["is_known_regulator"].sum()) if not res["h5_boundary"].empty else 0
        n_known_h7 = int(res["h7"]["is_known_regulator"].sum()) if not res["h7"].empty else 0
        cand_top = ", ".join((res["h7"]["gene"].head(5).tolist() if not res["h7"].empty else []))
        rows.append({
            "Dataset": ds,
            "H1 entropy method": h1.get("score_method"),
            "H1 geodesic rho": f"{h1.get('geodesic_spearman_rho', float('nan')):.3f}",
            "H1 euclid rho": f"{h1.get('euclidean_spearman_rho', float('nan')):.3f}",
            "H4 ran": "yes",
            "H5 stripe-boundary genes (top hits)": ", ".join(
                res["h5_boundary"].head(5)["gene"].tolist()) if not res["h5_boundary"].empty else "n/a",
            "H5 known overlap (top 50)": n_known_h5,
            "H7 top genes": cand_top,
            "H7 known overlap": n_known_h7,
            "Notes": "; ".join(res.get("expr_notes", []))[:160],
        })
    df = pd.DataFrame(rows)
    body = ["# Raw-expression validation — final summary", "",
            f"Generated by `scripts/run_raw_expression_validation.py`.",
            "",
            "## Per-dataset summary table", "",
            df.to_markdown(index=False) if not df.empty else "_no datasets processed_",
            "", "## Caveats",
            "- Gene perturbation vector fields are *fixed-PCA-loading sensitivity* analyses, not real CRISPR/RNAi predictions.",
            "- 'Shannon entropy (proxy)' rows in H1 are not CytoTRACE/SCENT proper.",
            "- UC epithelium ships without gene symbols; H5/H7 rankings there are by anonymous feature index and cannot be cross-referenced to known regulators.",
            "- Datasets larger than `--max-cells` are randomly subsampled before PCA.",
            "", "## What still blocks PNAS-level claims",
            "- Causal validation of candidate stripe-boundary regulators with real perturbation data.",
            "- Replicate datasets per system (one per system here).",
            "- A dataset-independent calibration of the stripe vs. theta scoring grid.",
           ]
    # Auto-generated; the curated `final_summary.md` is hand-edited and
    # should not be overwritten by reruns. Keep machine-readable output here.
    (out / "auto_summary.md").write_text("\n".join(body) + "\n")
    fs = out / "final_summary.md"
    if not fs.exists():
        # First run: copy auto into final_summary so the file always exists.
        fs.write_text("\n".join(body) + "\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
