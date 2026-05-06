"""Standalone HTML report generation for app pipeline results."""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from .app_plots import (
    branchpoint_scatter,
    equirectangular_scatter,
    longitude_density,
    metric_table_figure,
    spherical_density_heatmap,
    sphere_scatter_3d,
    theta_vs_pseudotime,
)
from .app_processing import PipelineResult, metrics_dataframe


BIOLOGICAL_EXPLANATION = """
<h2>Biological Interpretation</h2>
<p>This app treats spherical PCA as a geometric diagnostic. PC1-PC3 vectors are
L2-normalized onto the unit sphere, so angular direction is retained while the
radial magnitude of the original three-PC vector is discarded.</p>
<p>Root alignment rotates a selected starting cluster to the north pole. This
makes theta interpretable as angular distance from the chosen root, but it also
means root choice is part of the model. Manual Euler tuning is avoided; instead
the app fits a data-driven great circle and uses that frame for display.</p>
<p>The legacy single-number stripe metric is now reported as global longitude
concentration. Multi-stripe structure is reported separately with detected
stripe counts and per-stripe strength, so several arcs are not collapsed into
one weak global score. Anisotropy separates arc-like, ring-like, and sphere-filling geometry. Great-circle
residual measures how tightly the embedding lies near a single spherical
trajectory. If pseudotime-like cluster labels are available, theta-pseudotime
correlation is shown as an association, not proof of trajectory.</p>
<p><strong>Caveat:</strong> PCA horseshoes can arise as projection artifacts.
Stripe patterns should be followed by null models, root sensitivity analyses,
and independent biological validation before manuscript-level claims.</p>
"""


def caveats_html() -> str:
    return """
<h2>Caveats and Recommended Next Analyses</h2>
<ul>
  <li>Run a gene-wise permutation or matched simulation null before interpreting stripes biologically.</li>
  <li>Run PC-count, HVG-vs-all-gene, and root-sensitivity robustness before claiming PC1-PC3 are sufficient.</li>
  <li>Repeat root alignment with plausible alternative roots to quantify sensitivity.</li>
  <li>Compare theta against independent pseudotime, staged sampling, or RNA velocity when available.</li>
  <li>Inspect radial magnitude separately; normalization discards possible cell-cycle or metabolic signal.</li>
  <li>Use branchpoint scores as screening diagnostics, then validate with lineage-aware methods.</li>
</ul>
"""


def build_report_html(result: PipelineResult, dataset_name: str) -> str:
    color = "celltype" if "celltype" in result.data.columns else result.color_columns[0] if result.color_columns else None
    figures = [
        metric_table_figure(metrics_dataframe(result.metrics)),
        sphere_scatter_3d(result.data, color=color or "celltype"),
        equirectangular_scatter(result.data, color=color or "celltype"),
        longitude_density(result.data),
        spherical_density_heatmap(result.data),
    ]
    if result.pseudotime_column is not None:
        figures.append(theta_vs_pseudotime(result.data, color=color or "cluster"))
    if "branchpoint_score" in result.data.columns:
        figures.append(branchpoint_scatter(result.data))

    body = [
        "<html><head><meta charset='utf-8'><title>PCA Sphere Projection Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#17202a}"
        "h1,h2{color:#102a43}.metric-note{background:#f4f7fb;padding:12px;border-left:4px solid #486581}</style>",
        "</head><body>",
        f"<h1>PCA Sphere Projection Report: {dataset_name}</h1>",
        "<p class='metric-note'>Generated locally from the reusable non-UI pipeline used by the Streamlit app.</p>",
        BIOLOGICAL_EXPLANATION,
    ]
    for i, fig in enumerate(figures):
        body.append(pio.to_html(fig, include_plotlyjs="cdn" if i == 0 else False, full_html=False))
    body.append(caveats_html())
    body.append("</body></html>")
    return "\n".join(body)


def save_html_report(result: PipelineResult, output_path: str | Path, dataset_name: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report_html(result, dataset_name), encoding="utf-8")
    return path
