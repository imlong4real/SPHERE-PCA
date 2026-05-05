"""Plotly visualizations for the PCA sphere Streamlit app and HTML reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def _plot_df(df: pd.DataFrame, max_points: int = 20000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=0).copy()


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _colors_for(data: pd.DataFrame, color: str | None):
    if color is None or color not in data.columns:
        return {"marker_color": "#2f80ed", "showscale": False, "colorbar": None}
    series = data[color]
    if _is_numeric(series):
        return {"marker_color": series, "showscale": True, "colorbar": {"title": color}}
    categories = series.astype(str).fillna("Unknown")
    levels = sorted(categories.unique())
    palette = qualitative.Plotly
    mapping = {level: palette[i % len(palette)] for i, level in enumerate(levels)}
    return {
        "marker_color": categories.map(mapping),
        "showscale": False,
        "colorbar": None,
        "legend_mapping": mapping,
    }


def _add_discrete_legend(fig: go.Figure, mapping: dict[str, str]) -> None:
    for label, color in mapping.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 8, "color": color},
                name=label,
                showlegend=True,
            )
        )


def sphere_scatter_3d(df: pd.DataFrame, color: str = "celltype", max_points: int = 20000) -> go.Figure:
    data = _plot_df(df, max_points)
    colors = _colors_for(data, color)
    hover_cols = [c for c in ["celltype", "cluster", "root_theta", "display_longitude"] if c in data.columns]
    hover = data[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=data["root_x"],
            y=data["root_y"],
            z=data["root_z"],
            mode="markers",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover_cols else None,
            marker={
                "size": 2,
                "color": colors["marker_color"],
                "opacity": 0.78,
                "showscale": colors["showscale"],
                "colorbar": colors["colorbar"],
            },
            showlegend=False,
        )
    )
    if "legend_mapping" in colors:
        for label, marker_color in colors["legend_mapping"].items():
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker={"size": 5, "color": marker_color},
                    name=label,
                    showlegend=True,
                )
            )
    fig.update_layout(
        title="3D Spherical PCA Embedding",
        scene_aspectmode="cube",
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )
    return fig


def equirectangular_scatter(df: pd.DataFrame, color: str = "celltype", max_points: int = 30000) -> go.Figure:
    data = _plot_df(df, max_points)
    colors = _colors_for(data, color)
    hover_cols = [c for c in ["celltype", "cluster", "root_theta"] if c in data.columns]
    hover = data[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=data["display_longitude"],
            y=data["display_latitude"],
            mode="markers",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover_cols else None,
            marker={
                "size": 4,
                "color": colors["marker_color"],
                "opacity": 0.72,
                "showscale": colors["showscale"],
                "colorbar": colors["colorbar"],
            },
            showlegend=False,
        )
    )
    if "legend_mapping" in colors:
        _add_discrete_legend(fig, colors["legend_mapping"])
    fig.update_layout(title="Great-Circle-Aligned Equirectangular Projection")
    fig.update_xaxes(title="Longitude (degrees)", range=[-180, 180])
    fig.update_yaxes(title="Latitude (degrees)", range=[-90, 90])
    fig.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0})
    return fig


def theta_vs_pseudotime(df: pd.DataFrame, color: str = "celltype", max_points: int = 30000) -> go.Figure:
    data = _plot_df(df.dropna(subset=["pseudotime"]), max_points)
    colors = _colors_for(data, color)
    hover_cols = [c for c in ["celltype", "cluster"] if c in data.columns]
    hover = data[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=data["pseudotime"],
            y=data["root_theta"],
            mode="markers",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover_cols else None,
            marker={
                "size": 4,
                "color": colors["marker_color"],
                "opacity": 0.7,
                "showscale": colors["showscale"],
                "colorbar": colors["colorbar"],
            },
            showlegend=False,
        )
    )
    if "legend_mapping" in colors:
        _add_discrete_legend(fig, colors["legend_mapping"])
    fig.update_layout(title="Root Angular Distance vs Pseudotime / Cluster Order")
    fig.update_xaxes(title="Pseudotime / cluster midpoint")
    fig.update_yaxes(title="Theta from root-aligned north pole (degrees)")
    return fig


def longitude_density(df: pd.DataFrame) -> go.Figure:
    counts, edges = np.histogram(df["display_longitude"], bins=72, range=(-180, 180))
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure(go.Bar(x=centers, y=counts, width=np.diff(edges), marker={"color": "#2f80ed"}))
    fig.update_layout(title="Longitude / Stripe Density", bargap=0.02)
    fig.update_xaxes(title="Longitude (degrees)")
    fig.update_yaxes(title="Cells")
    return fig


def spherical_density_heatmap(df: pd.DataFrame) -> go.Figure:
    counts, xedges, yedges = np.histogram2d(
        df["display_longitude"],
        df["display_latitude"],
        bins=[72, 36],
        range=[[-180, 180], [-90, 90]],
    )
    fig = go.Figure(
        go.Heatmap(
            x=0.5 * (xedges[:-1] + xedges[1:]),
            y=0.5 * (yedges[:-1] + yedges[1:]),
            z=counts.T,
            colorscale="Viridis",
            colorbar={"title": "Cells"},
        )
    )
    fig.update_layout(title="Spherical Density Heatmap (Equirectangular Bins)")
    fig.update_xaxes(range=[-180, 180])
    fig.update_yaxes(range=[-90, 90])
    return fig


def branchpoint_scatter(df: pd.DataFrame, max_points: int = 30000) -> go.Figure:
    data = _plot_df(df.dropna(subset=["branchpoint_score"]), max_points)
    hover_cols = [c for c in ["celltype", "cluster", "pseudotime"] if c in data.columns]
    hover = data[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None
    fig = go.Figure(
        go.Scattergl(
            x=data["display_longitude"],
            y=data["display_latitude"],
            mode="markers",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover_cols else None,
            marker={
                "size": 5,
                "color": data["branchpoint_score"],
                "colorscale": "Magma",
                "showscale": True,
                "colorbar": {"title": "score"},
            },
        )
    )
    fig.update_layout(title="Branchpoint / Angular Divergence Score")
    fig.update_xaxes(range=[-180, 180])
    fig.update_yaxes(range=[-90, 90])
    return fig


def metric_table_figure(metrics_df: pd.DataFrame) -> go.Figure:
    values = metrics_df.copy()
    values["value"] = values["value"].map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else str(x))
    fig = go.Figure(
        data=[
            go.Table(
                header={"values": ["Metric", "Value"], "fill_color": "#243447", "font": {"color": "white"}},
                cells={"values": [values["metric"], values["value"]], "align": "left"},
            )
        ]
    )
    fig.update_layout(title="Quantitative Summary", margin={"l": 0, "r": 0, "t": 45, "b": 0})
    return fig
