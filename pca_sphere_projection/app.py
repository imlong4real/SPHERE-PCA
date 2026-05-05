"""
SPHERE-PCA: Spherical PCA Hypothesis Explorer.

A local Streamlit dashboard for inspecting the spherical PCA embedding of a
single-cell PCA dataset and running the runnable subset of hypotheses
H1-H7 from PNAS_EXTENSION_PROPOSAL.md.

Launch from a terminal:

    sphere-trace                # if installed (pip install -e .)
    python -m pca_sphere_projection.app   # without console script
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_streamlit():
    try:
        import streamlit as st  # noqa: F401
        return
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "streamlit is required to run sphere-trace. "
            "Install with `pip install streamlit`."
        ) from e


def main() -> None:
    """Console-script entry point: ``sphere-trace``.

    Streamlit must own the python process to bind a port and serve, so we
    re-execute this module under ``streamlit run``.
    """
    import os
    import subprocess

    _ensure_streamlit()
    here = Path(__file__).resolve()
    # If we are already inside a Streamlit run, fall through to the UI.
    if os.environ.get("STREAMLIT_SERVER_RUN_ON_SAVE") or os.environ.get("STREAMLIT_RUNTIME"):
        _render_app()
        return
    args = ["streamlit", "run", str(here), "--theme.base=dark", *sys.argv[1:]]
    subprocess.run(args, check=False)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DARK = "plotly_dark"


def _render_app() -> None:
    import streamlit as st
    import yaml
    from scipy.stats import spearmanr

    import pca_sphere_projection as ps
    from pca_sphere_projection.sphere_stats import _as_unit

    st.set_page_config(
        page_title="SPHERE-PCA: Spherical PCA Hypothesis Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- Custom CSS for a PNAS-style minimal look on the dark theme ----
    st.markdown(
        """
        <style>
            html, body, [class*="css"] { font-family: 'Helvetica Neue', Arial, sans-serif; }
            .pnas-title { font-size: 1.8rem; font-weight: 600; color: #E8EAF0;
                          letter-spacing: 0.05em; margin-bottom: 0.2rem; }
            .pnas-sub   { font-size: 0.95rem; color: #A0A6B0; margin-bottom: 1.2rem; }
            .pnas-warn  { background: #2E1A1A; color: #FFCDB2; padding: 0.6rem 0.9rem;
                          border-left: 3px solid #E76F51; border-radius: 2px; }
            .pnas-ok    { background: #15281C; color: #B8E0CF; padding: 0.6rem 0.9rem;
                          border-left: 3px solid #2A9D8F; border-radius: 2px; }
            div.stMetric { border-left: 2px solid #5C6373; padding-left: 0.6rem; }
        </style>
        <div class="pnas-title">SPHERE-PCA</div>
        <div class="pnas-sub">Spherical PCA Hypothesis Explorer for Regulatory
        Trajectories and Embeddings &mdash; local diagnostic dashboard for
        the runnable subset of H1&ndash;H7.</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Sidebar: data source + parameters ----
    cfg_path = REPO_ROOT / "examples" / "example_configs.yaml"
    examples = {}
    if cfg_path.exists():
        try:
            examples = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            examples = {}

    with st.sidebar:
        st.header("Data")
        source = st.radio("Source", ["Built-in example", "Upload CSV"],
                          horizontal=False)
        if source == "Built-in example":
            if not examples:
                st.error("No examples/example_configs.yaml found.")
                return
            ex_name = st.selectbox("Example", list(examples.keys()))
            cfg = examples[ex_name]
            csv_path = REPO_ROOT / cfg["csv"]
            if not csv_path.exists():
                st.error(f"CSV missing: {csv_path}")
                return
            df = pd.read_csv(csv_path)
            preset = cfg
        else:
            uploaded = st.file_uploader("CSV with PC1, PC2, PC3 + label", type=["csv"])
            if uploaded is None:
                st.info("Upload a CSV to begin.")
                st.stop()
            df = pd.read_csv(uploaded)
            preset = {}

        for col in ("PC1", "PC2", "PC3"):
            if col not in df.columns:
                st.error(f"Required column missing: {col}")
                st.stop()

        st.header("Annotations")
        label_col = st.selectbox("Label column", list(df.columns),
                                  index=list(df.columns).index(preset.get("label_column"))
                                  if preset.get("label_column") in df.columns else 0)
        root_col = st.selectbox("Root column", list(df.columns),
                                 index=list(df.columns).index(preset.get("root_column"))
                                 if preset.get("root_column") in df.columns else 0)
        root_choices = sorted(df[root_col].dropna().astype(str).unique().tolist())
        root_default = str(preset.get("root_value")) if preset.get("root_value") else root_choices[0]
        root_value = st.selectbox("Root value", root_choices,
                                   index=root_choices.index(root_default)
                                   if root_default in root_choices else 0)
        pt_col = st.selectbox("Pseudotime column (optional)",
                               ["(none)"] + list(df.columns),
                               index=(["(none)"] + list(df.columns)).index(preset.get("pseudotime_column"))
                               if preset.get("pseudotime_column") in df.columns else 0)
        pt_parser = st.selectbox("Pseudotime parser",
                                  ["ordinal", "embryo_time_bin"],
                                  index=["ordinal", "embryo_time_bin"].index(preset.get("pseudotime_parser", "ordinal")))

        st.header("Alignment")
        eul = preset.get("euler_angles", [0, 0, 0])
        c1, c2, c3 = st.columns(3)
        ex = c1.number_input("Euler X", value=float(eul[0]), step=5.0)
        ey = c2.number_input("Euler Y", value=float(eul[1]), step=5.0)
        ez = c3.number_input("Euler Z", value=float(eul[2]), step=5.0)
        align_mode = st.radio("Rotation mode",
                               ["Manual Euler", "Automated great-circle"],
                               horizontal=False)

        st.header("Geometry params")
        n_bins = st.slider("Stripe bins (longitude)", 8, 72, 36)
        k_top = st.slider("k (kNN/topology)", 5, 50, 15)

        st.header("Robustness")
        n_perturb = st.slider("Rotation perturbations", 5, 50, 25)
        perturb_deg = st.slider("Perturbation magnitude (deg)", 1.0, 20.0, 5.0)

        run = st.button("Run analysis", type="primary")

    if not run:
        st.info("Pick a dataset, then click **Run analysis**.")
        return

    # ---- Compute ----
    centroid = df.loc[df[root_col] == root_value, ["PC1", "PC2", "PC3"]].mean().values
    aligned = ps.align_to_north_pole(df, pcs_columns=["PC1", "PC2", "PC3"],
                                     cluster_column=root_col, root_node=centroid)

    if align_mode == "Manual Euler":
        rotated = ps.apply_euler_rotation(aligned,
                                          pcs_columns=["PC1", "PC2", "PC3"],
                                          rotation_angles=[ex, ey, ez],
                                          degrees=True)
        coords = _as_unit(rotated[["rotated_PC1", "rotated_PC2", "rotated_PC3"]].values)
    else:
        cmp_ = ps.compare_manual_vs_great_circle(aligned, euler_angles=[ex, ey, ez])
        coords = _as_unit(_as_unit(aligned[["PC1", "PC2", "PC3"]].values) @ cmp_["gc_rotation"].T)
        rotated = aligned.copy()
        for i, col in enumerate(["rotated_PC1", "rotated_PC2", "rotated_PC3"]):
            rotated[col] = coords[:, i]

    # ---- Quality + geometry summary ----
    quality = ps.pc_coordinate_quality_summary(df)
    aniso = ps.spherical_anisotropy(coords)
    stripe = ps.stripe_strength_score(coords, n_bins=n_bins)
    gc = ps.fit_great_circle(coords)

    st.subheader("Geometric summary")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Cells", f"{len(df):,}")
    metric_cols[1].metric("Stripe-strength", f"{stripe['score']:.3f}")
    metric_cols[2].metric("Anisotropy (linear)", f"{aniso['linear']:.3f}")
    metric_cols[3].metric("Anisotropy (planar)", f"{aniso['planar']:.3f}")
    metric_cols[4].metric("Great-circle R²", f"{gc['r_squared']:.3f}")

    if quality["frac_zero_norm"] > 0:
        st.warning(f"{quality['frac_zero_norm']:.1%} of cells have zero PC norm "
                   f"and were dropped by the projection.")

    if quality["norm_max"] / max(quality["norm_min"], 1e-12) < 1.05:
        st.markdown(
            '<div class="pnas-warn">Input PCs appear to be already L2-normalised '
            '(norms within 5%). Hypothesis H4 (radial-vs-angular split) '
            'is <b>not testable</b> from this CSV.</div>',
            unsafe_allow_html=True,
        )

    # ---- Plots: 3D, 2D ----
    labels = rotated[label_col].astype(str)
    cats = pd.Categorical(labels)

    col_3d, col_2d = st.columns(2)
    with col_3d:
        st.markdown("**Sphere (3D)**")
        fig3 = go.Figure()
        for cat in cats.categories:
            mask = (cats == cat)
            fig3.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers", marker=dict(size=2, opacity=0.7),
                name=str(cat),
            ))
        fig3.update_layout(template=DARK, height=480, margin=dict(l=0, r=0, t=10, b=0),
                           scene=dict(aspectmode="cube"))
        st.plotly_chart(fig3, use_container_width=True)
    with col_2d:
        st.markdown("**Equirectangular (2D)**")
        lon = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(coords[:, 2], -1.0, 1.0)))
        fig2 = go.Figure()
        for cat in cats.categories:
            mask = (cats == cat)
            fig2.add_trace(go.Scattergl(
                x=lon[mask], y=lat[mask],
                mode="markers", marker=dict(size=3, opacity=0.7),
                name=str(cat),
            ))
        fig2.update_layout(template=DARK, height=480, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(title="Longitude (deg)", range=[-180, 180]),
                           yaxis=dict(title="Latitude (deg)", range=[-90, 90]))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Stripe density ----
    st.subheader("Stripe density (longitude)")
    fig_h = go.Figure(data=go.Histogram(x=lon, nbinsx=n_bins))
    fig_h.update_layout(template=DARK, height=300,
                       xaxis_title="Longitude (deg)", yaxis_title="cells")
    st.plotly_chart(fig_h, use_container_width=True)

    # ---- Pseudotime ~ theta ----
    pt = None
    pt_desc = None
    if pt_col != "(none)":
        if pt_parser == "embryo_time_bin":
            def _emb(s):
                if isinstance(s, str) and "-" in s:
                    try:
                        a, b = s.split("-")
                        return (int(a) + int(b)) / 2.0
                    except ValueError:
                        return float("nan")
                return float("nan")
            pt = df[pt_col].apply(_emb).values
            pt_desc = f"embryo-time-bin midpoint of {pt_col}"
        else:
            codes = df[pt_col].astype("category").cat.codes.values.astype(float)
            codes[codes < 0] = float("nan")
            pt = codes
            pt_desc = f"category-rank of {pt_col}"

    if pt is not None and np.any(~np.isnan(pt)):
        theta = np.arccos(np.clip(coords[:, 2], -1.0, 1.0))
        mask = ~np.isnan(pt)
        rho, p = spearmanr(pt[mask], theta[mask])
        st.subheader(f"theta vs pseudotime ({pt_desc})")
        st.markdown(f"Spearman ρ = **{rho:.3f}** (p = {p:.2e}, n = {int(mask.sum())})")
        fig_t = go.Figure()
        sub_labels = labels[mask]
        sub_cats = pd.Categorical(sub_labels)
        for cat in sub_cats.categories:
            m = (sub_cats == cat)
            fig_t.add_trace(go.Scattergl(
                x=pt[mask][m], y=np.degrees(theta[mask])[m],
                mode="markers", marker=dict(size=4, opacity=0.6),
                name=str(cat),
            ))
        fig_t.update_layout(template=DARK, height=380,
                            xaxis_title="Pseudotime",
                            yaxis_title="theta (deg)")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.markdown(
            '<div class="pnas-warn">No pseudotime column selected. H2 (branching '
            'detection) and theta-vs-pseudotime correlation are <b>not '
            'computed</b>.</div>',
            unsafe_allow_html=True,
        )

    # ---- Branchpoint detection (H2) ----
    if pt is not None and np.any(~np.isnan(pt)):
        st.subheader("Branchpoint detection (H2)")
        idx_full = np.where(~np.isnan(pt))[0]
        cap = 8000
        if idx_full.size > cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(idx_full, size=cap, replace=False)
            st.caption(f"Subsampled to {cap:,} cells for the O(N²) "
                       f"geodesic matrix.")
        else:
            idx = idx_full
        bp = ps.detect_branchpoints(coords[idx], pt[idx], k=k_top)
        lvb = ps.linear_vs_branching_score(coords[idx], pt[idx], k=k_top)
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", lvb["decision"])
        c2.metric("Linear BIC", f"{lvb['linear_bic']:.1f}")
        c3.metric("Branching BIC", f"{lvb['branching_bic']:.1f}")
        fig_bp = go.Figure(data=go.Scatter3d(
            x=coords[idx, 0], y=coords[idx, 1], z=coords[idx, 2],
            mode="markers",
            marker=dict(size=2, color=bp["score"], colorscale="Inferno",
                        colorbar=dict(title="branch score"), opacity=0.85),
        ))
        fig_bp.update_layout(template=DARK, height=420,
                             scene=dict(aspectmode="cube"),
                             margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bp, use_container_width=True)
    else:
        bp = None
        lvb = None

    # ---- Root sensitivity ----
    st.subheader("Root-cluster sensitivity")
    candidates = preset.get("candidate_roots") or [root_value]
    candidates = list(dict.fromkeys(candidates + [root_value]))
    rs = ps.root_sensitivity_analysis(df, candidate_roots=candidates,
                                       root_column=root_col,
                                       euler_angles=[ex, ey, ez])
    fig_rs = go.Figure(data=go.Bar(x=rs["root"].astype(str),
                                    y=rs["metric"]))
    fig_rs.update_layout(template=DARK, height=300,
                         xaxis_title="root cluster",
                         yaxis_title="stripe-strength")
    st.plotly_chart(fig_rs, use_container_width=True)
    st.caption("Stripe-strength under each root choice. Large variation "
               "across plausible roots means the geometric story is "
               "anchor-dependent.")

    # ---- Rotation robustness ----
    st.subheader("Rotation robustness")
    rr = ps.rotation_robustness_analysis(aligned,
                                          base_euler_angles=[ex, ey, ez],
                                          perturbation_degrees=perturb_deg,
                                          n_perturbations=n_perturb)
    fig_rr = go.Figure()
    fig_rr.add_trace(go.Histogram(x=rr["perturbed_metrics"], nbinsx=20))
    fig_rr.add_vline(x=rr["base_metric"], line_dash="dash", line_color="red",
                     annotation_text="base", annotation_position="top")
    fig_rr.update_layout(template=DARK, height=300,
                         xaxis_title="stripe-strength under +- perturbation",
                         yaxis_title="count")
    st.plotly_chart(fig_rr, use_container_width=True)
    st.caption(f"Mean +- std: {rr['mean']:.3f} +- {rr['std']:.3f} "
               f"(CV {rr['cv']:.2f}).")

    # ---- H5 / H7 panels (require external data) ----
    st.subheader("H5 / H7 (gene-level analyses, optional)")
    with st.expander("Upload expression matrix to enable H5/H7"):
        st.markdown(
            '<div class="pnas-warn">H5 (stripe-boundary genes) and H7 (gene '
            'perturbation vector fields) <b>cannot be tested from PC1-PC3 '
            'alone</b>. They require: (i) a gene x cell expression matrix '
            'aligned to this CSV; (ii) for H7, the PCA loadings (sklearn '
            'components_) and gene means used to derive PC1-PC3.</div>',
            unsafe_allow_html=True,
        )
        expr_file = st.file_uploader("Expression matrix (cells x genes CSV)",
                                      type=["csv"], key="expr")
        comp_file = st.file_uploader("PCA components (k x genes CSV, optional, H7)",
                                      type=["csv"], key="comp")
        gene_input = st.text_input("Genes for H7 (comma-separated)",
                                    placeholder="e.g. sox-2, elt-2")
        h57_button = st.button("Run H5/H7")
        if h57_button:
            if expr_file is None:
                st.error("Upload an expression matrix.")
            else:
                expr = pd.read_csv(expr_file, index_col=0)
                if expr.shape[0] != len(df):
                    st.error(f"Expression matrix has {expr.shape[0]} rows but "
                             f"the PC CSV has {len(df)}; rows must align.")
                else:
                    # H5: theta-direction gradient, top genes by tangent magnitude
                    theta_field = np.arccos(np.clip(coords[:, 2], -1.0, 1.0))
                    g_grad = ps.geodesic_gradient(coords, theta_field, k=k_top)
                    # Crude H5 surrogate: per-gene Spearman of expression with
                    # *longitude* (cross-stripe direction) vs. *latitude* (along-
                    # stripe). Stripe-boundary genes have strong longitude effect.
                    lat = np.arcsin(np.clip(coords[:, 2], -1.0, 1.0))
                    lon_rad = np.arctan2(coords[:, 1], coords[:, 0])
                    rows = []
                    for g in expr.columns:
                        rho_lon = spearmanr(expr[g].values, lon_rad).statistic
                        rho_lat = spearmanr(expr[g].values, lat).statistic
                        rows.append({"gene": g,
                                     "rho_longitude": float(rho_lon),
                                     "rho_latitude": float(rho_lat),
                                     "abs_long_minus_lat": float(abs(rho_lon) - abs(rho_lat))})
                    h5 = pd.DataFrame(rows).sort_values("abs_long_minus_lat",
                                                        ascending=False)
                    st.markdown("**H5 (proxy):** longitude-vs-latitude correlation "
                                "per gene. Positive **abs_long_minus_lat** ⇒ "
                                "candidate stripe-boundary gene.")
                    st.dataframe(h5.head(50), use_container_width=True)

                    # H7
                    if comp_file is not None and gene_input.strip():
                        comp = pd.read_csv(comp_file, index_col=0)
                        if comp.shape[1] != expr.shape[1]:
                            st.error("PCA components columns must equal expression genes.")
                        else:
                            genes = [g.strip() for g in gene_input.split(",")
                                      if g.strip() in expr.columns]
                            if not genes:
                                st.error("No requested gene present in expression matrix.")
                            else:
                                pca_components = comp.values
                                pca_mean = expr.values.mean(axis=0)
                                results = ps.compute_gene_perturbation_vectors(
                                    expr.values, list(expr.columns), genes,
                                    pca_components, pca_mean,
                                    normalize_to_sphere=True,
                                )
                                rank = ps.rank_genes_by_perturbation_magnitude(
                                    results, metric="symmetric_range")
                                st.markdown("**H7:** Frobenius-norm ranking under "
                                            "doubled - zeroed perturbation. **Sensitivity, "
                                            "not causality.**")
                                st.dataframe(rank, use_container_width=True)

    # ---- Export ----
    st.subheader("Export")
    metrics = {
        "stripe_strength": stripe["score"],
        "anisotropy": aniso,
        "great_circle_r_squared": gc["r_squared"],
        "rotation_robustness": {k: v for k, v in rr.items()
                                 if k != "perturbed_metrics"},
        "root_sensitivity": rs.to_dict(orient="records"),
        "branching": (None if lvb is None else {
            "decision": lvb["decision"],
            "linear_bic": lvb["linear_bic"],
            "branching_bic": lvb["branching_bic"],
        }),
    }
    j = json.dumps(metrics, indent=2, default=str)
    st.download_button("metrics.json", data=j, file_name="sphere_pca_metrics.json")
    coords_df = rotated.copy()
    coords_df["sphere_x"] = coords[:, 0]
    coords_df["sphere_y"] = coords[:, 1]
    coords_df["sphere_z"] = coords[:, 2]
    coords_df["longitude_deg"] = lon
    coords_df["latitude_deg"] = lat
    csv_buf = io.StringIO()
    coords_df.to_csv(csv_buf, index=False)
    st.download_button("processed_coords.csv", data=csv_buf.getvalue(),
                        file_name="sphere_pca_processed_coords.csv")

    st.markdown(
        "<small>SPHERE-PCA is a diagnostic dashboard. It does not infer "
        "trajectories. Stripe-strength, branching, and pseudotime correlations "
        "depend on the chosen root and rotation; review the robustness "
        "panels before drawing conclusions.</small>",
        unsafe_allow_html=True,
    )


# Streamlit triggers when imported as the script.
if __name__ == "__main__" or __name__ == "pca_sphere_projection.app":
    try:
        import streamlit as _st  # noqa: F401
        if _st.runtime.exists():
            _render_app()
    except Exception:
        pass
