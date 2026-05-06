"""Figure 3 — biological validation with REAL CytoTRACE.

For each dataset we expose four sub-panels A/B/C/D, named e.g.
``A_hesc``, ``B_planaria``. Each sub-panel is independently selectable
via the CLI's ``--panel`` flag.

Datasets: hesc, planaria, human_germ_cell, pre_implant_human_embryo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from .. import io as psp_io
from .. import preprocessing as psp_pp
from .. entropy import compute_transcriptional_entropy
from . import common


CYTO_DATASETS = {
    "hesc":            common.ROOT / "raw_data" / "hESC",
    "planaria":        common.ROOT / "raw_data" / "planaria",
    "human_germ_cell": common.ROOT / "raw_data" / "human_germ_cell",
    "pre_implant_human_embryo": common.ROOT / "raw_data" / "pre_implant_human_embryo",
}

CYCLE_GENES_HUMAN = {
    "MKI67", "TOP2A", "PCNA", "CCNB1", "CCNB2", "CCNA2", "CCNE1", "CCNE2",
    "CDK1", "CDK2", "BIRC5", "AURKA", "AURKB", "MCM2", "MCM5", "MCM6",
}

PANEL_DEFAULT = {
    "n_top_genes": 2000,
    "n_pcs": 3,
    "preprocessing": "log1p+HVG2000+sklearn-PCA(3)+L2-projection",
    "stemness_definition": "1 - (rank - rank_min) / (rank_max - rank_min)",
    "root_choice": "decile (top 10% stemness)",
    "seed": 0,
}


def _fit_sphere_for_cyto(folder: Path):
    expr = psp_io.load_cytotrace_rds_dataset(str(folder))
    X_norm = psp_pp.normalize_log1p(expr.X)
    n_top = min(PANEL_DEFAULT["n_top_genes"], expr.n_genes - 1)
    idx, hvg_names, X_hvg = psp_pp.select_hvgs(X_norm, expr.gene_names, n_top)
    pca = psp_pp.fit_pca_embedding(X_hvg, hvg_names, n_components=PANEL_DEFAULT["n_pcs"])
    pc_pre = pca.scores
    unit = common.to_unit(pc_pre)
    radial = np.linalg.norm(pc_pre, axis=1)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    cyto_rank = expr.metadata["CytoTRACE"].astype(float).values
    rmin, rmax = float(np.nanmin(cyto_rank)), float(np.nanmax(cyto_rank))
    stem = 1.0 - (cyto_rank - rmin) / (rmax - rmin + 1e-9)
    pheno = expr.metadata["Phenotype"].astype(str).values \
        if "Phenotype" in expr.metadata.columns else None
    return {
        "expr": expr, "unit": unit, "radial": radial, "theta": theta,
        "stem": stem, "phenotype": pheno, "X_norm": X_norm,
        "n_hvg": len(hvg_names),
        "expl_var": pca.explained_variance_ratio.tolist(),
    }


def _missing_card(panel_id: str, ds: str, sub: str):
    return common.data_not_present_result(
        panel_id, 3, f"Fig 3{sub} — {ds}",
        f"raw_data/{ds}/dataset.rds not present.")


def _panel_A(ds: str, folder: Path, config: Optional[dict]) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config); cfg["dataset"] = ds
    panel_id = f"Fig3A_{ds}"
    if not folder.exists() or not (folder / "dataset.rds").exists():
        return _missing_card(panel_id, ds, "A")
    d = _fit_sphere_for_cyto(folder)
    rho, p = spearmanr(d["theta"], d["stem"], nan_policy="omit")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(d["theta"], d["stem"], s=6, alpha=0.5,
               color=common.DATASET_COLORS.get(ds, "#444"))
    ax.set_xlabel(r"$\theta$ from north pole (rad)")
    ax.set_ylabel("CytoTRACE stemness (1 = most stem-like)")
    ax.set_title(f"Fig 3A — {ds}\nSpearman ρ(θ, stem) = {rho:.3f}, p = {p:.1e}")
    fig.tight_layout()
    df = pd.DataFrame({
        "cell_id": d["expr"].cell_ids, "theta": d["theta"], "stem": d["stem"],
        "radial": d["radial"], "phenotype": d["phenotype"]
        if d["phenotype"] is not None else "",
    })
    cfg["spearman_rho"] = float(rho); cfg["spearman_p"] = float(p)
    cfg["explained_var"] = d["expl_var"]; cfg["n_hvgs"] = d["n_hvg"]
    return {"panel_id": panel_id, "figure_n": 3, "figure": fig, "config": cfg,
            "data": {"per_cell": df}}


def _panel_B(ds: str, folder: Path, config: Optional[dict]) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config); cfg["dataset"] = ds
    panel_id = f"Fig3B_{ds}"
    if not folder.exists() or not (folder / "dataset.rds").exists():
        return _missing_card(panel_id, ds, "B")
    d = _fit_sphere_for_cyto(folder)
    fig = plt.figure(figsize=(11, 4.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax3d.scatter(d["unit"][:, 0], d["unit"][:, 1], d["unit"][:, 2],
                      c=d["stem"], s=4, cmap="magma", alpha=0.85)
    u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    xs, ys, zs = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
    ax3d.plot_wireframe(xs, ys, zs, color="lightgrey", alpha=0.15, linewidth=0.5)
    ax3d.set_title("S² coloured by CytoTRACE stemness")
    ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
    cb = fig.colorbar(sc, ax=ax3d, fraction=0.04, pad=0.05); cb.set_label("stemness", fontsize=9)
    ax2 = fig.add_subplot(1, 2, 2)
    lon, lat = common.xyz_to_lonlat(d["unit"])
    ax2.scatter(lon, lat, c=d["stem"], s=4, cmap="magma", alpha=0.8)
    ax2.set_xlim(-180, 180); ax2.set_ylim(-90, 90)
    ax2.set_xlabel("longitude (°)"); ax2.set_ylabel("latitude (°)")
    ax2.set_title("Equirectangular")
    fig.suptitle(f"Fig 3B — {ds}", y=1.02)
    fig.tight_layout()
    df = pd.DataFrame({"x": d["unit"][:, 0], "y": d["unit"][:, 1], "z": d["unit"][:, 2],
                       "lon": lon, "lat": lat, "stem": d["stem"],
                       "cell_id": d["expr"].cell_ids})
    return {"panel_id": panel_id, "figure_n": 3, "figure": fig, "config": cfg,
            "data": {"per_cell": df}}


def _panel_C(ds: str, folder: Path, config: Optional[dict]) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config); cfg["dataset"] = ds
    cfg["cell_cycle_genes"] = sorted(CYCLE_GENES_HUMAN)
    panel_id = f"Fig3C_{ds}"
    if not folder.exists() or not (folder / "dataset.rds").exists():
        return _missing_card(panel_id, ds, "C")
    d = _fit_sphere_for_cyto(folder)
    gn = {g.lower(): i for i, g in enumerate(d["expr"].gene_names)}
    cc_idx = [gn[g.lower()] for g in CYCLE_GENES_HUMAN if g.lower() in gn]
    cc_score = None
    if cc_idx:
        sub = (d["X_norm"][:, cc_idx].toarray() if hasattr(d["X_norm"], "toarray")
               else d["X_norm"][:, cc_idx])
        cc_score = np.asarray(sub).mean(axis=1)
    entr = compute_transcriptional_entropy(d["X_norm"], d["expr"].gene_names, "shannon")
    rho_e, p_e = spearmanr(d["radial"], entr, nan_policy="omit")
    fig, axes = plt.subplots(1, 2 if cc_score is not None else 1,
                              figsize=(11 if cc_score is not None else 5.5, 4))
    if cc_score is None: axes = [axes]
    ax = axes[0]
    ax.scatter(d["radial"], entr, s=6, alpha=0.5, color=common.DATASET_COLORS.get(ds, "#444"))
    ax.set_xlabel("radial norm  ‖PC1-3‖"); ax.set_ylabel("Shannon entropy (proxy)")
    ax.set_title(f"radial vs entropy   ρ = {rho_e:.3f}")
    cfg["radial_vs_entropy_rho"] = float(rho_e); cfg["radial_vs_entropy_p"] = float(p_e)
    if cc_score is not None:
        ax = axes[1]
        rho_cc, p_cc = spearmanr(d["radial"], cc_score, nan_policy="omit")
        ax.scatter(d["radial"], cc_score, s=6, alpha=0.5, color=common.DATASET_COLORS.get(ds, "#444"))
        ax.set_xlabel("radial norm  ‖PC1-3‖")
        ax.set_ylabel("cell-cycle score (mean of curated markers)")
        ax.set_title(f"radial vs cell-cycle   ρ = {rho_cc:.3f}")
        cfg["radial_vs_cell_cycle_rho"] = float(rho_cc)
        cfg["radial_vs_cell_cycle_p"] = float(p_cc)
        cfg["n_cell_cycle_genes_used"] = len(cc_idx)
    fig.suptitle(f"Fig 3C — {ds}", y=1.03); fig.tight_layout()
    out = {"radial": d["radial"], "entropy": entr}
    if cc_score is not None: out["cell_cycle_score"] = cc_score
    df = pd.DataFrame({"cell_id": d["expr"].cell_ids, **{k: v for k, v in out.items()}})
    return {"panel_id": panel_id, "figure_n": 3, "figure": fig, "config": cfg,
            "data": {"per_cell": df}}


def _panel_D(ds: str, folder: Path, config: Optional[dict]) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config); cfg["dataset"] = ds
    panel_id = f"Fig3D_{ds}"
    if not folder.exists() or not (folder / "dataset.rds").exists():
        return _missing_card(panel_id, ds, "D")
    d = _fit_sphere_for_cyto(folder)
    fig, ax = plt.subplots(figsize=(7, 4))
    if d["phenotype"] is None or len(np.unique(d["phenotype"])) < 2:
        ax.text(0.5, 0.5, "no Phenotype labels — cannot sweep root",
                ha="center", va="center", transform=ax.transAxes)
        rows = []
    else:
        rows = []
        for cat in np.unique(d["phenotype"]):
            mask = d["phenotype"] == cat
            anchor = d["unit"][mask].mean(axis=0)
            anchor /= np.linalg.norm(anchor)
            inner = np.clip(d["unit"] @ anchor, -1.0, 1.0)
            geo = np.arccos(inner)
            rho, p = spearmanr(d["stem"], geo, nan_policy="omit")
            rows.append({"phenotype": cat, "rho_stem_vs_geo": float(rho),
                         "p_value": float(p), "n": int(mask.sum())})
        rdf = pd.DataFrame(rows).sort_values("rho_stem_vs_geo",
                                             key=lambda s: -s.abs())
        ext = rdf["rho_stem_vs_geo"].abs().max()
        cols = ["#e45756" if abs(r) >= ext - 1e-9 else "#cccccc"
                for r in rdf["rho_stem_vs_geo"]]
        ax.bar(rdf["phenotype"], rdf["rho_stem_vs_geo"], color=cols,
               edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("ρ(stem, geodesic distance to anchor)")
        ax.set_xlabel("anchor phenotype")
        ax.set_title("Lower magnitude = anchor not stem-like")
        ax.tick_params(axis="x", rotation=30)
        rows = rdf.to_dict("records")
    fig.suptitle(f"Fig 3D — {ds}: root sensitivity", y=1.03); fig.tight_layout()
    return {"panel_id": panel_id, "figure_n": 3, "figure": fig, "config": cfg,
            "data": {"root_sweep": pd.DataFrame(rows)}}


def _panel_summary(config: Optional[dict] = None,
                    out_root: Optional[Path] = None) -> dict:
    cfg = common.merge_config(PANEL_DEFAULT, config)
    summaries = []
    for ds, folder in CYTO_DATASETS.items():
        if not (folder / "dataset.rds").exists():
            continue
        try:
            d = _fit_sphere_for_cyto(folder)
            rho, p = spearmanr(d["theta"], d["stem"], nan_policy="omit")
            summaries.append({"dataset": ds, "rho_theta_vs_stem": float(rho),
                              "p_value": float(p),
                              "n_cells": int(d["unit"].shape[0])})
        except Exception:
            continue
    if not summaries:
        return common.data_not_present_result(
            "Fig3_summary", 3, "Fig 3 summary", "no CytoTRACE datasets present.")
    df = pd.DataFrame(summaries)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cols = [common.DATASET_COLORS.get(d, "#444") for d in df["dataset"]]
    ax.bar(df["dataset"], df["rho_theta_vs_stem"], color=cols,
           edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel(r"Spearman ρ(θ, stemness)")
    ax.set_title("Fig 3. CytoTRACE vs spherical θ across datasets")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return {"panel_id": "Fig3_summary", "figure_n": 3, "figure": fig,
            "config": cfg, "data": {"summary": df}}


# ---------------------------------------------------------------------------
# Build PANELS dict: A_hesc, B_hesc, C_hesc, D_hesc, A_planaria, ..., summary
# ---------------------------------------------------------------------------

def _make(letter: str, ds: str):
    folder = CYTO_DATASETS[ds]
    fn = {"A": _panel_A, "B": _panel_B, "C": _panel_C, "D": _panel_D}[letter]
    def runner(config=None, out_root=None):
        return fn(ds, folder, config)
    runner.__name__ = f"panel_{letter}_{ds}"
    return runner


PANELS = {}
for _letter in ["A", "B", "C", "D"]:
    for _ds in CYTO_DATASETS.keys():
        PANELS[f"{_letter}_{_ds}"] = _make(_letter, _ds)
PANELS["summary"] = _panel_summary
