# Tutorials

These examples use tiny synthetic datasets, run in seconds, and require no
downloads.

## 1. Existing PCA coordinates

Install plotting support and run:

```bash
python -m pip install -e ".[plot]"
python examples/existing_pca.py
```

The script constructs a small PC1-PC3 matrix, selects a root population,
computes `theta`, `phi`, and `r`, and saves an equirectangular view.

## 2. AnnData

Install the single-cell and plotting extras and run:

```bash
python -m pip install -e ".[singlecell,plot]"
python examples/anndata_workflow.py
```

The script starts from a synthetic cell-by-gene `AnnData` object, fits PCA,
uses a known early-state population as the biological root, writes the three
coordinates to `adata.obs`, and saves a visualization. For a large or sparse
dataset, compute PCA with your normal single-cell workflow and place it in
`adata.obsm["X_pca"]`; `sphere_pca.fit` will use the first three columns.

Root choice is part of the scientific model. Use a population justified by
sampling time, markers, lineage knowledge, or another independent source—not
the spherical display alone.
