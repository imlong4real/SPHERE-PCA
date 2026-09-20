"""Tutorial 2: fit SPHERE-PCA from a small AnnData expression matrix."""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sphere_pca


rng = np.random.default_rng(12)
n_cells, n_genes = 60, 12
stage = np.repeat(["early", "middle", "late"], n_cells // 3)
signal = np.linspace(-2.0, 2.0, n_cells)
expression = rng.normal(0, 0.4, size=(n_cells, n_genes))
expression[:, 0:4] += signal[:, None]
expression[:, 4:8] += np.sin(signal[:, None])

adata = ad.AnnData(
    X=expression,
    obs=pd.DataFrame({"stage": stage}, index=[f"cell_{i}" for i in range(n_cells)]),
)

# Here use_rep=None asks SPHERE-PCA to fit PCA. In a typical Scanpy workflow,
# leave use_rep="X_pca" (the default) to reuse adata.obsm["X_pca"].
result = sphere_pca.fit(
    adata,
    root="early",
    root_key="stage",
    use_rep=None,
    write_back=True,
)

print(adata.obs[["sphere_pca_theta", "sphere_pca_phi", "sphere_pca_r"]].head())
sphere_pca.plot(result, color=adata.obs["stage"], projection="equirectangular")
plt.tight_layout()
output = Path(__file__).with_name("anndata_workflow.png")
plt.savefig(output, dpi=160)
print(f"Saved {output}")
