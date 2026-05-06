"""
Raw-expression I/O for the validation workflow.

Returns a small ``ExpressionData`` dataclass everywhere so downstream code
does not have to track which loader produced which fields. The loaders are
intentionally permissive: when a metadata file is missing they return None
rather than raising, so the caller can decide how to proceed.
"""

from __future__ import annotations

import bz2
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


@dataclass
class ExpressionData:
    """Container for a cell x gene expression matrix and its metadata."""

    X: np.ndarray | sp.spmatrix
    gene_names: list
    cell_ids: list
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = ""
    notes: list = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]

    def as_dense(self) -> np.ndarray:
        if sp.issparse(self.X):
            return np.asarray(self.X.todense())
        return np.asarray(self.X)


# ---------------------------------------------------------------------------
# Generic dispatch
# ---------------------------------------------------------------------------

def load_expression_matrix(path: str) -> ExpressionData:
    """
    Best-effort loader keyed off file extension.

    Supported: .csv, .tsv, .txt, .mtx, .h5ad, .rds, .rda, .bz2.
    For .h5ad/.rds/.rda we delegate to optional dependencies and raise a
    clear message if they are missing.
    """
    path = str(path)
    lower = path.lower()
    if lower.endswith(".mtx"):
        # .mtx alone has no gene/cell labels; return numeric matrix only.
        m = scipy.io.mmread(path)
        # Heuristic orientation: assume rows = genes if rows < cols, else as-is.
        if m.shape[0] < m.shape[1]:
            X = m.T.tocsr()
        else:
            X = m.tocsr()
        return ExpressionData(
            X=X,
            gene_names=[f"feat_{i}" for i in range(X.shape[1])],
            cell_ids=[f"cell_{i}" for i in range(X.shape[0])],
            source=path,
            notes=["loaded from .mtx without sidecar gene/cell labels"],
        )
    if lower.endswith(".h5ad"):
        try:
            import anndata as ad  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading .h5ad requires `pip install anndata`."
            ) from exc
        a = ad.read_h5ad(path)
        return ExpressionData(
            X=a.X,
            gene_names=list(a.var_names),
            cell_ids=list(a.obs_names),
            metadata=a.obs.copy(),
            source=path,
        )
    if lower.endswith(".rds") or lower.endswith(".rda"):
        return load_rds_expression(path)
    if lower.endswith(".bz2"):
        # Treat as a single bz2-compressed dense CSV (gene rows, cell cols).
        return _load_dense_csv_bz2(path)
    if lower.endswith((".csv", ".tsv", ".txt")):
        sep = "," if lower.endswith(".csv") else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        # Heuristic: if many more columns than rows, assume rows = genes.
        if df.shape[0] > df.shape[1]:
            X = df.values.astype(float)
            gene_names = list(df.columns)
            cell_ids = list(df.index)
        else:
            X = df.values.T.astype(float)
            gene_names = list(df.index)
            cell_ids = list(df.columns)
        return ExpressionData(
            X=X, gene_names=gene_names, cell_ids=cell_ids, source=path
        )
    raise ValueError(f"unsupported extension for {path}")


# ---------------------------------------------------------------------------
# RDS
# ---------------------------------------------------------------------------

def load_rds_expression(path: str) -> ExpressionData:
    """
    Load an .rds saved by R, returning the expression matrix and any
    fields that look like CytoTRACE / phenotype output.

    Uses ``rdata`` (pure Python, install with `pip install rdata`). If
    ``rdata`` is unavailable, prints clear instructions and raises.

    Recognised structure (CytoTRACE-style result):
        - exprMatrix: genes x cells DataArray
        - Phenotype: per-cell labels
        - CytoTRACE: per-cell stemness rank
        - GCS, Counts, Order: per-cell numeric vectors
    """
    if not hasattr(np, "unicode_"):
        # rdata<=1.0 still references the NumPy 1.x alias.
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    try:
        import rdata  # type: ignore
    except ImportError as exc:
        msg = (
            "Reading .rds requires the `rdata` package.\n"
            "Install with one of:\n"
            "  pip install rdata\n"
            "  # or for pyreadr (also works but builds C++ on Apple Silicon):\n"
            "  pip install pyreadr\n"
        )
        raise ImportError(msg) from exc

    parsed = rdata.read_rds(path)
    notes = []
    if isinstance(parsed, dict):
        keys = {str(k): v for k, v in parsed.items()}
        if "exprMatrix" in keys:
            expr = keys["exprMatrix"]
            try:
                vals = np.asarray(expr.values)
                row_labels = list(map(str, expr.coords[expr.dims[0]].values))
                col_labels = list(map(str, expr.coords[expr.dims[1]].values))
            except Exception:  # plain ndarray fallback
                vals = np.asarray(expr)
                row_labels = [f"feat_{i}" for i in range(vals.shape[0])]
                col_labels = [f"cell_{i}" for i in range(vals.shape[1])]
            # exprMatrix is typically genes x cells
            X = vals.T.astype(float)
            gene_names = row_labels
            cell_ids = col_labels
            md = {}
            n = X.shape[0]
            for k in ("Phenotype", "CytoTRACE", "GCS", "Counts", "Order"):
                if k in keys and getattr(keys[k], "shape", (0,))[0] == n:
                    md[k] = np.asarray(keys[k])
            md_df = pd.DataFrame(md, index=cell_ids) if md else pd.DataFrame(index=cell_ids)
            if "output" in keys and isinstance(keys["output"], pd.DataFrame) and len(keys["output"]) == n:
                out = keys["output"].copy()
                out.index = cell_ids
                # avoid name collision
                out.columns = [f"output_{c}" for c in out.columns]
                md_df = pd.concat([md_df, out], axis=1)
            return ExpressionData(
                X=X,
                gene_names=gene_names,
                cell_ids=cell_ids,
                metadata=md_df,
                source=path,
                notes=["RDS dictionary; exprMatrix decoded as genes x cells"],
            )
        # If it's a dict but no exprMatrix, fail clearly.
        raise ValueError(
            f"RDS dict from {path} has keys {list(keys)} but no 'exprMatrix'"
        )
    # Plain matrix or DataFrame
    if hasattr(parsed, "values") and hasattr(parsed, "shape"):
        X = np.asarray(parsed.values).astype(float)
        if X.shape[0] > X.shape[1]:
            X = X.T
        return ExpressionData(
            X=X,
            gene_names=[f"feat_{i}" for i in range(X.shape[1])],
            cell_ids=[f"cell_{i}" for i in range(X.shape[0])],
            source=path,
            notes=["RDS plain matrix; orientation guessed by shape"],
        )
    raise ValueError(f"unrecognised RDS object type for {path}: {type(parsed)}")


# ---------------------------------------------------------------------------
# Klein per-day .bz2 reassembly
# ---------------------------------------------------------------------------

def _load_dense_csv_bz2(path: str) -> ExpressionData:
    """
    Klein-style file: bz2 CSV, no header row, first column = gene name,
    remaining columns = per-cell counts.
    """
    rows = []
    gene_names = []
    n_cols = None
    with bz2.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            gene_names.append(parts[0])
            counts = [float(x) for x in parts[1:]]
            if n_cols is None:
                n_cols = len(counts)
            rows.append(counts)
    X = np.asarray(rows, dtype=float).T  # cells x genes
    cell_ids = [f"c{i:04d}" for i in range(X.shape[0])]
    return ExpressionData(
        X=X,
        gene_names=gene_names,
        cell_ids=cell_ids,
        source=path,
    )


def load_bz2_klein_dataset(folder: str) -> ExpressionData:
    """
    Reconstruct the Klein mESC matrix by stacking the four per-day CSVs
    (d0, d2, d4, d7) along the cell axis. The gene axis is checked to be
    identical across days.
    """
    file_specs = [
        ("Day0", "GSM1599494_ES_d0_main.csv.bz2"),
        ("Day2", "GSM1599497_ES_d2_LIFminus.csv.bz2"),
        ("Day4", "GSM1599498_ES_d4_LIFminus.csv.bz2"),
        ("Day7", "GSM1599499_ES_d7_LIFminus.csv.bz2"),
    ]
    parts = []
    for day, fn in file_specs:
        path = os.path.join(folder, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = _load_dense_csv_bz2(path)
        parts.append((day, d))

    # consistency
    g0 = parts[0][1].gene_names
    for day, d in parts[1:]:
        if d.gene_names != g0:
            raise ValueError(f"gene name list differs between Day0 and {day}")

    Xs = [d.X for _, d in parts]
    X = np.concatenate(Xs, axis=0)
    cell_ids = []
    day_col = []
    for day, d in parts:
        cell_ids.extend([f"{day.lower()}_{cid}" for cid in d.cell_ids])
        day_col.extend([day] * d.n_cells)
    md = pd.DataFrame({"day": day_col}, index=cell_ids)
    return ExpressionData(
        X=X,
        gene_names=list(g0),
        cell_ids=cell_ids,
        metadata=md,
        source=folder,
        notes=["reassembled from four per-day Klein CSVs (d0, d2, d4, d7)"],
    )


# ---------------------------------------------------------------------------
# Dataset-specific helpers for the directories actually present in raw_data
# ---------------------------------------------------------------------------

def load_celegan(folder: str) -> ExpressionData:
    """Load Packer et al. C. elegans embryogenesis from `raw_data/celegan/`."""
    mtx = scipy.io.mmread(os.path.join(folder, "celegan.mtx"))
    # header reports 2766 86024 -> rows = genes, cols = cells. Transpose.
    if mtx.shape[0] < mtx.shape[1]:
        X = mtx.T.tocsr()
    else:
        X = mtx.tocsr()
    genes = pd.read_csv(os.path.join(folder, "celegan_gene.tsv"), sep="\t",
                       header=None)[0].astype(str).tolist()
    cells = pd.read_csv(os.path.join(folder, "celegan_cell.tsv"), sep="\t",
                        header=None)[0].astype(str).tolist()
    md = pd.DataFrame(index=cells)
    for col, fn in [
        ("celltype", "celegan_celltype.tsv"),
        ("embryo_time", "celegan_embryo_time.tsv"),
        ("batch", "celegan_batch.tsv"),
    ]:
        p = os.path.join(folder, fn)
        if os.path.exists(p):
            md[col] = pd.read_csv(p, sep="\t", header=None)[0].astype(str).values
    md["embryo_time_mid"] = _embryo_time_to_mid(md.get("embryo_time"))
    return ExpressionData(
        X=X, gene_names=genes, cell_ids=cells, metadata=md,
        source=folder,
        notes=[
            "C. elegans .mtx is genes x cells; transposed to cells x genes",
            "celltype contains 'NA' strings (~half of cells unannotated)",
        ],
    )


def _embryo_time_to_mid(s: pd.Series | None) -> pd.Series | None:
    if s is None:
        return None
    out = []
    for v in s:
        v = str(v).strip()
        if v in ("", "NA", "nan"):
            out.append(np.nan); continue
        if v.startswith("<"):
            try: out.append(float(v.lstrip("<").strip()) / 2)
            except Exception: out.append(np.nan)
            continue
        if v.startswith(">"):
            try: out.append(float(v.lstrip(">").strip()))
            except Exception: out.append(np.nan)
            continue
        if "-" in v:
            try:
                a, b = v.split("-")
                out.append((float(a) + float(b)) / 2.0)
            except Exception: out.append(np.nan)
            continue
        try: out.append(float(v))
        except Exception: out.append(np.nan)
    return pd.Series(out, index=s.index)


def load_uc_epi(folder: str) -> ExpressionData:
    """Load Smillie UC epithelium from `raw_data/uc_epi/`.

    If `uc_epi_gene.tsv` is present, its lines are used as gene symbols
    (positional alignment with the MTX feature axis). If it is missing or
    its line count does not match the MTX's feature axis, we fall back to
    `feat_<i>` placeholders and print a clear diagnostic.
    """
    mtx = scipy.io.mmread(os.path.join(folder, "uc_epi.mtx"))
    # 1361 features x 64457 cells -> transpose
    if mtx.shape[0] < mtx.shape[1]:
        X = mtx.T.tocsr()
    else:
        X = mtx.tocsr()
    n_genes = X.shape[1]

    notes = []
    placeholder_gene_ids = [f"feat_{i}" for i in range(n_genes)]
    gene_path = os.path.join(folder, "uc_epi_gene.tsv")
    if os.path.exists(gene_path):
        symbols = pd.read_csv(gene_path, sep="\t", header=None)[0].astype(str).tolist()
        if len(symbols) == n_genes:
            gene_names = symbols
            notes.append(
                "gene symbols loaded from uc_epi_gene.tsv (positional alignment, n=1361)"
            )
        else:
            print(
                f"[load_uc_epi] WARNING: uc_epi_gene.tsv has {len(symbols)} entries "
                f"but the MTX feature axis is {n_genes}. Refusing to mismap; "
                f"falling back to feat_<i> placeholders."
            )
            gene_names = placeholder_gene_ids
            notes.append(
                f"uc_epi_gene.tsv length mismatch ({len(symbols)} vs {n_genes}); "
                "kept feat_<i> placeholders"
            )
    else:
        gene_names = placeholder_gene_ids
        notes.append(
            "no uc_epi_gene.tsv present; using feat_<i> placeholders"
        )

    cells = pd.read_csv(os.path.join(folder, "uc_epi_cell.tsv"), sep="\t",
                        header=None)[0].astype(str).tolist()
    md = pd.DataFrame(index=cells)
    for col, fn in [
        ("celltype", "uc_epi_celltype.tsv"),
        ("health", "uc_epi_batch_health.tsv"),
        ("location", "uc_epi_batch_location.tsv"),
        ("patient", "uc_epi_batch_patient.tsv"),
    ]:
        p = os.path.join(folder, fn)
        if os.path.exists(p):
            md[col] = pd.read_csv(p, sep="\t", header=None)[0].astype(str).values

    e = ExpressionData(
        X=X,
        gene_names=gene_names,
        cell_ids=cells,
        metadata=md,
        source=folder,
        notes=notes,
    )
    # Preserve the placeholder feature-IDs (one-to-one with the MTX feature
    # axis) in case downstream code wants to cross-reference earlier outputs
    # that were written with placeholders.
    e.feature_ids = placeholder_gene_ids
    return e


def load_hesc(folder: str) -> ExpressionData:
    return load_rds_expression(os.path.join(folder, "dataset.rds"))


def load_cytotrace_rds_dataset(folder: str) -> ExpressionData:
    """Generic loader for CytoTRACE-format RDS datasets that follow the same
    schema as the hESC file (planaria, human_germ_cell, pre_implant_human_embryo).
    """
    return load_rds_expression(os.path.join(folder, "dataset.rds"))


def load_brca_3d_coords(folder: str) -> pd.DataFrame:
    """Load the BrCa atlas precomputed 3D coordinates and merge cell metadata.

    Returns a DataFrame indexed by NAME with columns X, Y, Z plus any matching
    rows from `Whole_miniatlas_meta.csv`. We deliberately do *not* load the 2GB
    `matrix.mtx.gz` here; the 3D coordinates are sufficient for the manuscript
    figures.
    """
    coords_path = os.path.join(folder, "3D_BrCa_whole_cluster.tsv")
    meta_path = os.path.join(folder, "Whole_miniatlas_meta.csv")
    coords = pd.read_csv(coords_path, sep="\t", index_col=0)
    # second row is "TYPE numeric numeric numeric" — drop and coerce
    if str(coords.iloc[0, 0]).lower().startswith("numeric"):
        coords = coords.iloc[1:].copy()
    coords[["X", "Y", "Z"]] = coords[["X", "Y", "Z"]].astype(float)
    meta = pd.read_csv(meta_path, low_memory=False)
    if "NAME" in meta.columns:
        meta = meta.set_index("NAME")
        # second row is "TYPE group numeric ..." legend — drop if present
        if "TYPE" in meta.index:
            meta = meta.drop(index="TYPE")
    df = coords.join(meta, how="left")
    return df


def load_brca_atlas(folder: str) -> ExpressionData:
    """Load the BrCa miniatlas 10x-style matrix and sidecar metadata."""
    mtx = scipy.io.mmread(os.path.join(folder, "matrix.mtx.gz"))
    if mtx.shape[0] < mtx.shape[1]:
        X = mtx.T.tocsr()
    else:
        X = mtx.tocsr()
    features = pd.read_csv(os.path.join(folder, "features.tsv.gz"), sep="\t", header=None)
    genes = features.iloc[:, 1].astype(str).tolist()
    barcodes = pd.read_csv(os.path.join(folder, "barcodes.tsv.gz"), sep="\t", header=None)[0].astype(str).tolist()
    md = pd.DataFrame(index=barcodes)
    meta_path = os.path.join(folder, "Whole_miniatlas_meta.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if "NAME" in meta.columns:
            meta = meta[~meta["NAME"].isin(["TYPE", "NAME"])].copy()
            meta.index = meta["NAME"].astype(str)
            md = md.join(meta.drop(columns=["NAME"], errors="ignore"), how="left")
    cluster_path = os.path.join(folder, "3D_BrCa_whole_cluster.tsv")
    if os.path.exists(cluster_path):
        cl = pd.read_csv(cluster_path, sep="\t", header=None, names=["NAME", "X", "Y", "Z"])
        cl = cl[~cl["NAME"].isin(["TYPE", "NAME"])].copy()
        cl.index = cl["NAME"].astype(str)
        for c in ["X", "Y", "Z"]:
            md[f"published_3d_{c}"] = pd.to_numeric(cl[c], errors="coerce").reindex(md.index)
    return ExpressionData(
        X=X,
        gene_names=genes,
        cell_ids=barcodes,
        metadata=md,
        source=folder,
        notes=["BrCa atlas matrix loaded from 10x-style MTX sidecars"],
    )


# ---------------------------------------------------------------------------
# Match raw expression cells to a pre-computed PC CSV
# ---------------------------------------------------------------------------

def match_expression_to_pc_csv(expression: ExpressionData,
                               pc_df: pd.DataFrame,
                               metadata: pd.DataFrame | None = None) -> dict:
    """
    Try to align cells in an `ExpressionData` with rows in a PC1-PC3 CSV.

    Strategy:
      1. If both have matching index labels (set intersection > 50%), use those.
      2. Else if row counts match exactly, fall back to positional matching
         and label the result as "positional".
      3. Else return ``{"matched": False, ...}`` so the caller can choose to
         re-run PCA from raw expression.
    """
    out = {"matched": False, "method": None, "pc_df": None, "expression": None,
           "n_matched": 0}
    expr_ids = set(map(str, expression.cell_ids))
    pc_ids = set(map(str, pc_df.index))
    inter = expr_ids & pc_ids
    if len(inter) >= 0.5 * min(len(expr_ids), len(pc_ids)) and len(inter) > 0:
        keep = sorted(inter)
        idx_pc = pc_df.loc[keep]
        # subset expression to keep, preserving raw ordering of those keep IDs
        keep_set = set(keep)
        mask = [cid in keep_set for cid in expression.cell_ids]
        expr_sub = _subset_expression(expression, mask)
        # reorder to match keep
        order = [list(expression.cell_ids).index(c) for c in keep]
        expr_sub = _reindex_expression(expression, order)
        out.update({
            "matched": True, "method": "id-intersection", "pc_df": idx_pc,
            "expression": expr_sub, "n_matched": len(keep),
        })
        return out
    if expression.n_cells == len(pc_df):
        out.update({
            "matched": True, "method": "positional", "pc_df": pc_df.copy(),
            "expression": expression, "n_matched": expression.n_cells,
        })
        return out
    return out


def _subset_expression(e: ExpressionData, mask) -> ExpressionData:
    mask = np.asarray(mask, dtype=bool)
    return ExpressionData(
        X=e.X[mask],
        gene_names=list(e.gene_names),
        cell_ids=[c for c, m in zip(e.cell_ids, mask) if m],
        metadata=e.metadata.loc[mask].reset_index(drop=False) if not e.metadata.empty else pd.DataFrame(),
        source=e.source,
        notes=list(e.notes) + ["subset"],
    )


def _reindex_expression(e: ExpressionData, order) -> ExpressionData:
    order = list(order)
    return ExpressionData(
        X=e.X[order],
        gene_names=list(e.gene_names),
        cell_ids=[e.cell_ids[i] for i in order],
        metadata=e.metadata.iloc[order] if not e.metadata.empty else pd.DataFrame(),
        source=e.source,
        notes=list(e.notes) + ["reordered"],
    )


__all__ = [
    "ExpressionData",
    "load_expression_matrix",
    "load_rds_expression",
    "load_bz2_klein_dataset",
    "load_celegan",
    "load_uc_epi",
    "load_hesc",
    "load_brca_atlas",
    "match_expression_to_pc_csv",
]
