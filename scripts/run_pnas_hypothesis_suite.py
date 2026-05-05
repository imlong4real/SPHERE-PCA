"""
End-to-end hypothesis suite runner for the example CSVs.

Reads ``examples/example_configs.yaml`` and, for each dataset, executes the
runnable subset of H1-H7 from PNAS_EXTENSION_PROPOSAL.md. Outputs go under
``outputs/hypothesis_suite/<dataset>/``:

  metrics.json                -- machine-readable summary of every metric
  metrics.csv                 -- flat one-row CSV of the same content
  sphere_3d.html              -- interactive 3D sphere (Plotly)
  equirectangular_2d.html     -- interactive 2D map (Plotly)
  theta_vs_pseudotime.html    -- per-lineage scatter (if pseudotime available)
  stripe_density.html         -- longitude histogram (Plotly)
  branchpoints_3d.html        -- branchpoint score on the sphere (if pseudotime)
  root_sensitivity.html       -- bar chart over candidate roots (Plotly)
  rotation_robustness.html    -- distribution under small Euler perturbations
  report.md                   -- per-dataset narrative interpretation

A top-level ``hypothesis_status_table.md`` is written under
``outputs/hypothesis_suite/`` summarising H1-H7 across all datasets.

Usage:
    python scripts/run_pnas_hypothesis_suite.py \
        --config examples/example_configs.yaml \
        --outdir outputs/hypothesis_suite

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_pnas_hypothesis_suite.py` from the repo root even
# when the package is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from scipy.stats import spearmanr

import pca_sphere_projection as ps
from pca_sphere_projection.sphere_stats import _as_unit


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------

def _parse_embryo_time_bin(label: str) -> float:
    """Map a C. elegans-style time bin like '210-270' or '< 100' to a float."""
    if not isinstance(label, str):
        return float("nan")
    s = label.strip()
    if "-" in s:
        try:
            lo, hi = s.split("-")
            return (int(lo) + int(hi)) / 2.0
        except ValueError:
            return float("nan")
    if s.startswith("<"):
        try:
            return float(s.replace("<", "").strip()) - 1.0
        except ValueError:
            return float("nan")
    if s.startswith(">"):
        try:
            return float(s.replace(">", "").strip()) + 1.0
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _build_pseudotime(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, str | None]:
    """Return (pseudotime array of length len(df), description string)."""
    pt_col = cfg.get("pseudotime_column")
    parser = cfg.get("pseudotime_parser", "ordinal")
    lineages = cfg.get("lineage_orderings")

    if pt_col is not None and pt_col in df.columns and parser == "embryo_time_bin":
        pt = df[pt_col].apply(_parse_embryo_time_bin).values
        return pt, f"embryo time bin midpoint of {pt_col}"

    if lineages:
        # Build a pooled ordinal pseudotime using whichever lineage a cell
        # belongs to. Cells outside any lineage get NaN.
        global_order: dict[str, int] = {}
        offset = 0
        for _name, order in lineages.items():
            for cell in order:
                if cell not in global_order:
                    global_order[cell] = offset + order.index(cell)
            offset += len(order) + 5
        label_col = cfg["label_column"]
        pt = df[label_col].map(global_order).values.astype(float)
        return pt, "pooled ordinal rank from lineage_orderings"

    if pt_col is not None and pt_col in df.columns and parser == "ordinal":
        codes = df[pt_col].astype("category").cat.codes.values.astype(float)
        codes[codes < 0] = float("nan")
        return codes, f"category-rank of {pt_col}"

    return np.full(len(df), np.nan), None


# ---------------------------------------------------------------------------
# Plotting helpers (Plotly, dark template)
# ---------------------------------------------------------------------------

DARK = "plotly_dark"

# Cap the cell count used for O(N^2) operations (branchpoint detection,
# linear-vs-branching geodesic matrix). Visualisation and stripe statistics
# still use all cells.
CELL_SAMPLE_CAP = 8000


def _color_traces_by_label(coords_xy: list[np.ndarray], labels: pd.Series,
                           kind: str = "2d", marker_size: int = 3,
                           opacity: float = 0.7) -> list[go.Scatter | go.Scatter3d]:
    """Return one trace per unique label. Avoids plotly.express dependency."""
    cats = pd.Categorical(labels.astype(str))
    traces = []
    for cat in cats.categories:
        mask = cats == cat
        if kind == "3d":
            traces.append(go.Scatter3d(
                x=coords_xy[0][mask], y=coords_xy[1][mask], z=coords_xy[2][mask],
                mode="markers",
                marker=dict(size=marker_size, opacity=opacity),
                name=str(cat),
            ))
        else:
            traces.append(go.Scattergl(
                x=coords_xy[0][mask], y=coords_xy[1][mask],
                mode="markers",
                marker=dict(size=marker_size, opacity=opacity),
                name=str(cat),
            ))
    return traces


def _scatter3d_sphere(coords: np.ndarray, labels: pd.Series, title: str) -> go.Figure:
    fig = go.Figure(data=_color_traces_by_label(
        [coords[:, 0], coords[:, 1], coords[:, 2]], labels, kind="3d", marker_size=2,
    ))
    fig.update_layout(template=DARK, title=title,
                      scene=dict(aspectmode="cube"))
    return fig


def _scatter2d_equirect(coords: np.ndarray, labels: pd.Series, title: str) -> go.Figure:
    lon = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(coords[:, 2], -1.0, 1.0)))
    fig = go.Figure(data=_color_traces_by_label([lon, lat], labels, kind="2d"))
    fig.update_layout(template=DARK, title=title,
                      xaxis=dict(title="Longitude (deg)", range=[-180, 180]),
                      yaxis=dict(title="Latitude (deg)", range=[-90, 90]))
    return fig


def _theta_vs_pseudotime_plot(theta: np.ndarray, pt: np.ndarray, labels: pd.Series,
                              title: str) -> go.Figure:
    mask = ~np.isnan(pt)
    if not np.any(mask):
        fig = go.Figure()
        fig.update_layout(template=DARK, title=f"{title} (no pseudotime data)")
        return fig
    theta_deg = np.degrees(theta)
    sub_labels = labels[mask]
    fig = go.Figure(data=_color_traces_by_label(
        [pt[mask], theta_deg[mask]], sub_labels, kind="2d", marker_size=4, opacity=0.6,
    ))
    fig.update_layout(template=DARK, title=title,
                      xaxis_title="Pseudotime",
                      yaxis_title="theta (deg from north pole)")
    return fig


def _stripe_density_plot(coords: np.ndarray, n_bins: int, title: str) -> go.Figure:
    lon = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    fig = go.Figure(data=go.Histogram(x=lon, nbinsx=n_bins))
    fig.update_layout(template=DARK, title=title,
                      xaxis_title="Longitude (deg)",
                      yaxis_title="cell count")
    return fig


def _branchpoint_3d(coords: np.ndarray, score: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers",
        marker=dict(size=2, color=score, colorscale="Inferno",
                    colorbar=dict(title="branch score"), opacity=0.8),
    ))
    fig.update_layout(title=title, template=DARK,
                      scene=dict(aspectmode="cube"))
    return fig


def _root_sensitivity_plot(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(data=go.Bar(x=df["root"].astype(str), y=df["metric"]))
    fig.update_layout(template=DARK, title=title,
                      xaxis_title="root cluster",
                      yaxis_title="stripe-strength")
    return fig


def _rotation_robustness_plot(rr: dict, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rr["perturbed_metrics"], nbinsx=20,
                                name="perturbed"))
    fig.add_vline(x=rr["base_metric"], line_dash="dash", line_color="red",
                  annotation_text="base", annotation_position="top")
    fig.update_layout(title=title, template=DARK,
                      xaxis_title="stripe-strength score",
                      yaxis_title="count")
    return fig


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------

def run_dataset(name: str, cfg: dict, outdir: Path, repo_root: Path) -> dict:
    out = outdir / name
    out.mkdir(parents=True, exist_ok=True)

    csv_path = repo_root / cfg["csv"]
    df = pd.read_csv(csv_path)
    label_col = cfg["label_column"]
    root_col = cfg["root_column"]
    root_value = cfg["root_value"]
    euler = cfg.get("euler_angles", [0, 0, 0])

    metrics: dict = {"dataset": name, "csv": cfg["csv"]}

    # ---- PC coordinate quality ----
    quality = ps.pc_coordinate_quality_summary(df)
    metrics["pc_quality"] = quality

    # ---- Align + Euler ----
    centroid = df.loc[df[root_col] == root_value, ["PC1", "PC2", "PC3"]].mean().values
    aligned = ps.align_to_north_pole(df, pcs_columns=["PC1", "PC2", "PC3"],
                                     cluster_column=root_col, root_node=centroid)
    rotated = ps.apply_euler_rotation(aligned, pcs_columns=["PC1", "PC2", "PC3"],
                                      rotation_angles=euler, degrees=True)
    coords = _as_unit(rotated[["rotated_PC1", "rotated_PC2", "rotated_PC3"]].values)

    # ---- H4: radial-vs-angular split. Available from CSV alone. ----
    raw_norms = np.linalg.norm(df[["PC1", "PC2", "PC3"]].values, axis=1)
    metrics["radial_norm"] = {
        "mean": float(raw_norms.mean()), "std": float(raw_norms.std()),
        "min": float(raw_norms.min()), "max": float(raw_norms.max()),
    }
    # If radial norm is constant (preprocessed CSV already on the sphere) the
    # radial-vs-angular hypothesis cannot be tested.
    radial_testable = float(raw_norms.std() / max(raw_norms.mean(), 1e-12)) > 0.05
    metrics["h4_radial_testable"] = radial_testable

    # ---- Spherical anisotropy + stripe strength + great-circle fit ----
    aniso = ps.spherical_anisotropy(coords)
    metrics["spherical_anisotropy"] = aniso
    stripe = ps.stripe_strength_score(coords, n_bins=36)
    metrics["stripe_strength_score"] = {k: v for k, v in stripe.items() if k != "bin_counts"}
    metrics["stripe_bin_counts"] = stripe["bin_counts"].tolist()

    gc = ps.fit_great_circle(coords)
    metrics["great_circle"] = {
        "r_squared": gc["r_squared"],
        "mean_residual_radians": gc["mean_residual_radians"],
        "normal": gc["normal"].tolist(),
    }

    # Manual-Euler vs great-circle metric.
    cmp_gc = ps.compare_manual_vs_great_circle(aligned, euler_angles=euler)
    metrics["manual_vs_great_circle"] = {
        "manual_metric": cmp_gc["manual_metric"],
        "gc_metric": cmp_gc["gc_metric"],
        "great_circle_r_squared": cmp_gc["great_circle_r_squared"],
    }

    # ---- Pseudotime + theta correlation (per-lineage) ----
    pt, pt_desc = _build_pseudotime(rotated, cfg)
    theta = np.arccos(np.clip(coords[:, 2], -1.0, 1.0))
    metrics["pseudotime_source"] = pt_desc
    if pt_desc is not None and np.any(~np.isnan(pt)):
        mask = ~np.isnan(pt)
        rho, p = spearmanr(pt[mask], theta[mask])
        metrics["theta_pseudotime_spearman"] = {"rho": float(rho), "p": float(p),
                                                 "n": int(mask.sum())}

        # Per-lineage if available
        per_lineage = {}
        if "lineage_orderings" in cfg:
            for ln_name, order in cfg["lineage_orderings"].items():
                idx = rotated[label_col].isin(order)
                if idx.sum() < 5:
                    per_lineage[ln_name] = {"n": int(idx.sum()), "rho": None, "p": None}
                    continue
                rank_map = {c: i for i, c in enumerate(order)}
                ln_pt = rotated.loc[idx, label_col].map(rank_map).values
                ln_theta = theta[idx.values]
                rho_l, p_l = spearmanr(ln_pt, ln_theta)
                per_lineage[ln_name] = {"n": int(idx.sum()), "rho": float(rho_l),
                                        "p": float(p_l)}
        metrics["per_lineage_spearman"] = per_lineage
    else:
        metrics["theta_pseudotime_spearman"] = None

    # ---- H2: branchpoints + linear-vs-branching score ----
    # Branchpoint detection builds an N x N geodesic matrix; cap N at
    # CELL_SAMPLE_CAP to keep the script tractable on >50k-cell datasets.
    h2_run = pt_desc is not None and np.any(~np.isnan(pt))
    if h2_run:
        mask = ~np.isnan(pt)
        idx_full = np.where(mask)[0]
        if idx_full.size > CELL_SAMPLE_CAP:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx_full, size=CELL_SAMPLE_CAP, replace=False)
            sampled = True
        else:
            idx = idx_full
            sampled = False
        bp = ps.detect_branchpoints(coords[idx], pt[idx], k=15)
        lvb = ps.linear_vs_branching_score(coords[idx], pt[idx], k=15)
        metrics["branching"] = {
            "decision": lvb["decision"],
            "linear_bic": lvb["linear_bic"],
            "branching_bic": lvb["branching_bic"],
            "bayes_factor": lvb["bayes_factor"],
            "branchpoint_score_max": float(bp["score"].max()),
            "branchpoint_score_mean": float(bp["score"].mean()),
            "n_cells_with_pseudotime": int(mask.sum()),
            "n_cells_used": int(len(idx)),
            "subsampled": bool(sampled),
        }
        # store branchpoint coords for plotting
        bp_coords = coords[idx]
        bp_score = bp["score"]
    else:
        metrics["branching"] = None
        bp_coords = None
        bp_score = None

    # ---- Root sensitivity ----
    candidate_roots = cfg.get("candidate_roots", [root_value])
    rs = ps.root_sensitivity_analysis(df, candidate_roots=candidate_roots,
                                      root_column=root_col,
                                      euler_angles=euler)
    metrics["root_sensitivity"] = rs.to_dict(orient="records")

    # ---- Rotation robustness ----
    rr = ps.rotation_robustness_analysis(aligned, base_euler_angles=euler,
                                         perturbation_degrees=5.0,
                                         n_perturbations=25)
    metrics["rotation_robustness"] = {
        "base_metric": rr["base_metric"],
        "mean": rr["mean"], "std": rr["std"], "cv": rr["cv"],
    }

    # ---- Spherical KDE summary (subsample input for memory) ----
    if coords.shape[0] > CELL_SAMPLE_CAP:
        rng = np.random.default_rng(1)
        kde_idx = rng.choice(coords.shape[0], size=CELL_SAMPLE_CAP, replace=False)
        kde_coords = coords[kde_idx]
    else:
        kde_coords = coords
    density, _ = ps.spherical_kde(kde_coords, kappa=20.0, m_grid=2000)
    metrics["spherical_kde"] = {
        "min": float(density.min()), "max": float(density.max()),
        "p99_over_p01": float(np.quantile(density, 0.99) /
                              max(np.quantile(density, 0.01), 1e-12)),
        "n_cells_used": int(kde_coords.shape[0]),
    }

    # ---- Save metrics ----
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    pd.json_normalize(metrics, max_level=2).to_csv(out / "metrics.csv", index=False)

    # ---- Plots ----
    labels = rotated[label_col].astype(str)
    _scatter3d_sphere(coords, labels, f"{name} sphere (3D)").write_html(out / "sphere_3d.html")
    _scatter2d_equirect(coords, labels, f"{name} equirectangular").write_html(out / "equirectangular_2d.html")
    _stripe_density_plot(coords, 36, f"{name} stripe density").write_html(out / "stripe_density.html")

    if pt_desc is not None and np.any(~np.isnan(pt)):
        _theta_vs_pseudotime_plot(theta, pt, labels,
                                  f"{name} theta vs pseudotime ({pt_desc})"
                                  ).write_html(out / "theta_vs_pseudotime.html")

    if metrics["branching"] is not None and bp_coords is not None:
        _branchpoint_3d(bp_coords, bp_score,
                        f"{name} branchpoint score"
                        ).write_html(out / "branchpoints_3d.html")

    _root_sensitivity_plot(rs, f"{name} root sensitivity"
                           ).write_html(out / "root_sensitivity.html")
    _rotation_robustness_plot(rr, f"{name} rotation robustness (5 deg)"
                              ).write_html(out / "rotation_robustness.html")

    # ---- Markdown report ----
    _write_dataset_report(out / "report.md", name, cfg, metrics)

    return metrics


def _write_dataset_report(path: Path, name: str, cfg: dict, metrics: dict) -> None:
    aniso = metrics["spherical_anisotropy"]
    stripe = metrics["stripe_strength_score"]
    gc = metrics["great_circle"]
    rs = metrics["root_sensitivity"]
    rr = metrics["rotation_robustness"]
    cmp_ = metrics["manual_vs_great_circle"]
    branching = metrics["branching"]
    sp = metrics.get("theta_pseudotime_spearman")

    lines = [
        f"# {name} -- hypothesis suite report",
        "",
        f"Source: `{cfg['csv']}`. Label column: `{cfg['label_column']}`. "
        f"Root: `{cfg['root_value']}` in `{cfg['root_column']}`. "
        f"Euler angles (deg): {cfg.get('euler_angles')}.",
        "",
        "## Geometry",
        f"- Spherical anisotropy: linear={aniso['linear']:.3f}, "
        f"planar={aniso['planar']:.3f}, spherical={aniso['spherical']:.3f}.",
        f"- Stripe-strength score: {stripe['score']:.3f} "
        f"(observed entropy {stripe['entropy_observed']:.3f}; "
        f"uniform {stripe['entropy_uniform']:.3f}).",
        f"- Great-circle fit: R^2={gc['r_squared']:.3f}, "
        f"mean residual {gc['mean_residual_radians']:.3f} rad.",
        "",
        "## Manual Euler vs. automated great-circle alignment",
        f"- Manual stripe-strength: {cmp_['manual_metric']:.3f}.",
        f"- Great-circle stripe-strength: {cmp_['gc_metric']:.3f}.",
        f"- Great-circle fit R^2: {cmp_['great_circle_r_squared']:.3f}.",
        ("- Verdict: " + (
            "automated alignment matches or exceeds the manual choice."
            if cmp_["gc_metric"] >= cmp_["manual_metric"] - 0.02
            else "manual tuning still beats automated alignment, suggesting the "
                 "stripe pattern depends on the manual rotation."
        )),
        "",
        "## Root-cluster sensitivity",
    ]
    for row in rs:
        lines.append(
            f"- root=`{row['root']}` (n={row['n_cells']}): "
            f"stripe-strength={row['metric']}"
        )
    lines += ["", "## Rotation robustness (+- 5 deg, 25 draws)",
              f"- Base score: {rr['base_metric']:.3f}",
              f"- Perturbed mean +- std: {rr['mean']:.3f} +- {rr['std']:.3f} "
              f"(CV {rr['cv']:.2f}).",
              ""]

    if sp is not None:
        lines += [
            "## Pseudotime ~ theta",
            f"- Source: {metrics['pseudotime_source']}",
            f"- Spearman rho={sp['rho']:.3f} (p={sp['p']:.2e}, n={sp['n']}).",
        ]
        per_lin = metrics.get("per_lineage_spearman", {})
        for ln, v in per_lin.items():
            if v["rho"] is None:
                lines.append(f"  - {ln}: n={v['n']} (too few cells)")
            else:
                lines.append(f"  - {ln}: rho={v['rho']:.3f} (p={v['p']:.2e}, n={v['n']})")
    else:
        lines += ["## Pseudotime ~ theta",
                  "- *Not applicable*: no pseudotime column or lineage ordering provided."]

    if branching is not None:
        lines += ["", "## Linear vs. branching",
                  f"- Decision: **{branching['decision']}**",
                  f"- BIC: linear={branching['linear_bic']:.1f}; "
                  f"branching={branching['branching_bic']:.1f} "
                  f"(Bayes factor {branching['bayes_factor']:.2e}).",
                  f"- Branchpoint score: max={branching['branchpoint_score_max']:.3f}, "
                  f"mean={branching['branchpoint_score_mean']:.3f}."]
    else:
        lines += ["", "## Linear vs. branching",
                  "- *Not applicable*: requires pseudotime."]

    if not metrics.get("h4_radial_testable", False):
        lines += ["", "## Caveats",
                  "- The input CSV appears to have already been L2-normalised "
                  "(radial norm has < 5% CV). H4 (radial-vs-angular split) is "
                  "**not testable** without raw pre-normalisation PC scores."]
    else:
        lines += ["", "## H4 radial component (preliminary)",
                  f"- Pre-normalisation PC norm range: "
                  f"[{metrics['radial_norm']['min']:.3f}, "
                  f"{metrics['radial_norm']['max']:.3f}]; "
                  f"std/mean = {metrics['radial_norm']['std'] / max(metrics['radial_norm']['mean'], 1e-12):.3f}.",
                  "  (Association with cell-cycle / metabolic scores requires those scores in the input.)"]

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Hypothesis status table
# ---------------------------------------------------------------------------

def _write_status_table(out_path: Path, all_metrics: dict[str, dict],
                        cfgs: dict) -> None:
    """Compose a single H1-H7 status table across all datasets."""
    rows = []
    for ds, m in all_metrics.items():
        cfg = cfgs[ds]
        # H1: differentiation entropy decay -- requires CytoTRACE/SCENT scalar.
        rows.append(["H1", "differentiation-entropy decay", ds,
                     "scalar CytoTRACE/SCENT score per cell", "no",
                     "*Not testable* from PC-only CSV. Interface accepts a "
                     "user-provided scalar via geodesic_gradient.",
                     "outputs/hypothesis_suite/" + ds + "/",
                     "cell-state entropy needs an external scalar field"])

        # H2: branchpoint detection -- runnable if pseudotime
        if m["branching"] is not None:
            rows.append(["H2", "branching vs linear", ds,
                         "ordinal pseudotime", "yes",
                         f"linear_vs_branching_score and detect_branchpoints; "
                         f"decision={m['branching']['decision']}, BF="
                         f"{m['branching']['bayes_factor']:.2e}",
                         f"outputs/hypothesis_suite/{ds}/branchpoints_3d.html",
                         "Pseudotime is ordinal-rank, not a continuous DPT/scVelo estimate; "
                         "single-branchpoint model only."])
        else:
            rows.append(["H2", "branching vs linear", ds, "ordinal pseudotime",
                         "no", "*Not testable*: no pseudotime/lineage ordering.",
                         "-", "Provide pseudotime_column or lineage_orderings."])

        # H3: cross-species conservation -- needs a paired dataset.
        rows.append(["H3", "conserved stripes", ds,
                     "second species/dataset with shared labels", "no",
                     "*Not testable from a single CSV.* Module "
                     "comparison.procrustes_align_spheres is exposed for paired runs.",
                     "-", "needs a second matched dataset"])

        # H4: radial component
        if m.get("h4_radial_testable"):
            rows.append(["H4", "radial component informative", ds,
                         "pre-normalisation PC1-3 + cell-cycle/metabolic scalar",
                         "partial",
                         f"PC norm CV = "
                         f"{m['radial_norm']['std']/max(m['radial_norm']['mean'],1e-12):.3f}; "
                         "association test requires external scalar.",
                         "metrics.json/radial_norm",
                         "Need cell-cycle/metabolic gene-set scores."])
        else:
            rows.append(["H4", "radial component informative", ds,
                         "pre-normalisation PC1-3", "no",
                         "*Not testable*: input CSV already L2-normalised "
                         "(radial norm has < 5% CV).", "-",
                         "Provide raw PC scores upstream of normalisation."])

        # H5: stripe-boundary genes
        rows.append(["H5", "stripe-boundary switch genes", ds,
                     "gene expression matrix on the same cells", "no",
                     "*Not testable from PC CSV alone.* Use geodesic_gradient "
                     "with an expression scalar per gene.",
                     "-", "needs gene x cell expression matrix"])

        # H6: replicate residual
        rows.append(["H6", "replicate-residual QC", ds,
                     "two replicate datasets of the same condition", "no",
                     "*Not testable from a single CSV.* Module "
                     "comparison.spherical_replicate_residual is exposed.",
                     "-", "needs paired replicate"])

        # H7: perturbation vector field
        rows.append(["H7", "gene perturbation vector field", ds,
                     "expression matrix + sklearn PCA loadings + selected genes",
                     "no",
                     "*Not testable from PC CSV alone.* Module "
                     "perturbation.compute_gene_perturbation_vectors is exposed.",
                     "-", "needs raw expression + PCA loadings"])

    header = ["Hypothesis", "Description", "Dataset", "Required input",
              "Available in CSV?", "Implemented test", "Output",
              "Remaining caveat"]
    lines = ["# Hypothesis status table",
             "",
             "Generated by `scripts/run_pnas_hypothesis_suite.py`.",
             "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="examples/example_configs.yaml")
    parser.add_argument("--outdir", default="outputs/hypothesis_suite")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset names to run; default: all.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfgs = yaml.safe_load((repo_root / args.config).read_text())
    if args.datasets is not None:
        cfgs = {k: v for k, v in cfgs.items() if k in args.datasets}

    out = repo_root / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    for name, cfg in cfgs.items():
        print(f"==> {name}")
        try:
            all_metrics[name] = run_dataset(name, cfg, out, repo_root)
            print(f"   ok -> {out / name}")
        except Exception as e:  # pragma: no cover
            print(f"   FAILED: {type(e).__name__}: {e}")
            all_metrics[name] = {"error": str(e)}

    _write_status_table(out / "hypothesis_status_table.md", all_metrics, cfgs)
    print(f"\nWrote status table -> {out / 'hypothesis_status_table.md'}")


if __name__ == "__main__":
    main()
