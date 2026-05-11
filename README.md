# SPHERE-PCA

**Interpretable spherical geometry of single-cell state transitions from dominant principal components.**

SPHERE-PCA projects the first three principal components (PC1–PC3) of normalized single-cell gene expression onto the unit sphere (S²), aligns a biologically defined root population to the north pole via Euler rotation, and analyzes root-aligned geodesic distance (θ), angular position (φ), and pre-projection radial magnitude (r) as interpretable coordinates for cell state and trajectory structure. 

---

## Overview

### What SPHERE-PCA measures

| Coordinate | Interpretation | Typical biological relevance |
|---|---|---|
| **θ** (root-aligned geodesic distance) | Progress along the principal great circle from the north pole | developmental stage, differentiation, cell-cycle phase *when validated against external labels or scores* |
| **φ** (angular position) | Position around the sphere perpendicular to θ | branching, branched cell-state architecture, regional gene-expression clustering |
| **r** (pre-projection radial norm) | Magnitude before L2 normalization | transcriptional activity, cell-cycle phase, biosynthetic state, depth-like or compositional variation |

---

## Algorithm outline

### Input
- **Normalized expression** (log-normalized, optional HVG filtering) or **precomputed PC1–PC3 scores**
- **Metadata:** cell type, pseudotime (optional), cell-cycle / metabolic / entropy scores (optional)
- **Configuration:** root cluster label, Euler rotation angles (or auto-align via great-circle fit)

### Steps

1. **Normalization & PCA** (if raw counts supplied):
   - Log-normalize expression, select highly variable genes
   - Center and scale
   - Compute PCA (scikit-learn)

2. **Sphere projection**:
   - Extract PC1–PC3 scores: **z_i = (PC1_i, PC2_i, PC3_i)**
   - Compute L2 norm: **r_i = ||z_i||**
   - Project to unit sphere: **x̂_i = z_i / r_i**

3. **Root alignment**:
   - Compute root centroid: **c_root = mean(x̂_i for i ∈ root cells)**
   - Determine Euler rotation **R** such that **R · c_root** aligns to north pole (0, 0, 1)
   - Rotate all cells: **ŷ_i = R · x̂_i**

4. **Coordinate extraction**:
   - **θ_i** = arccos(ŷ_i[2]) — geodesic latitude
   - **φ_i** = atan2(ŷ_i[1], ŷ_i[0]) — azimuth
   - **r_i** retained from step 2

5. **Geometry metrics** (deterministic):
   - Stripe strength: concentration of cells around principal great circles
   - Great-circle fit R²: variance explained by the strongest great-circle component
   - Anisotropy: eigenvalue asymmetry of the point cloud on S²
   - Branching score: local density variation consistent with branched topology

6. **Gene displacement scoring** (fixed-PCA-loading model):
   - For each gene **g**, add **δ · loading_g** to PC1–PC3
   - Reproject perturbed cell coordinates to the sphere
   - Compute per-cell **ΔθĤ_g**, **Δφ_g**, **Δr_g**
   - Summarize: **P_g** = median(Δθ), **B_g** = median(|Δφ|), **R_g** = median(Δr)**

### Output
- Per-cell: θ, φ, r, metadata
- Per-gene: loading, displacement scores (P, B, R)
---

## Installation

```bash
git clone https://github.com/imlong4real/pca_sphere_projection.git
cd pca_sphere_projection
pip install -e .

# Optional: interactive dashboard
pip install -e .[app]

# Optional: testing
pip install -e .[test]
```

**Requirements:** Python ≥3.9; NumPy ≥2.0.2, pandas ≥2.3.3, SciPy ≥1.13.1, Matplotlib ≥3.9.4, Statsmodels ≥0.14.6, Scanpy ≥1.10.3, gseapy ≥1.2.1 (pinned for reproducibility). See [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml).

---

## Quick start

### 1. Explore interactively

```bash
sphere-trace
# or:
python -m pca_sphere_projection.app
```

Opens a Streamlit dashboard in your browser. Upload a CSV with PC1, PC2, PC3, and a cell-type label column, then:
- Visualize 3D sphere and equirectangular 2D maps
- Adjust root cluster and rotation
- Compute geometry metrics
- Export metrics and processed coordinates

### 2. Reproduce manuscript figures

```bash
python scripts/make_manuscript_figures.py --figure 1 --all-panels
```

Generates Figures 1–4 under `outputs/figures/`. Each figure panel writes:
- PNG (publication-quality)
- YAML configuration used
- CSV data underlying the panel

**Note:** Requires `pca_sphere_projection/figures/` modules (fig1, fig2, fig3, fig4) and their configuration files. See [`pca_sphere_projection/figures/__init__.py`](pca_sphere_projection/figures/__init__.py) for panel registry.

---

## Example: Python API

```python
import numpy as np
import pandas as pd
import pca_sphere_projection as ps

# Load your expression data
expr = pd.read_csv("counts.csv", index_col=0)  # cells × genes

# Or load precomputed PCs
pcs = pd.read_csv("pc123_scores.csv", index_col=0)  # cells × 3
metadata = pd.read_csv("metadata.csv", index_col=0)

# Project to sphere and align root
root_mask = metadata["celltype"] == "stem"  # or your root definition
theta, phi, r = ps.align_to_north_pole(
    pcs.values, 
    root_mask=root_mask
)

# Compute geometry metrics
stripe_score = ps.stripe_strength_score(theta, phi)
anisotropy = ps.spherical_anisotropy(pcs.values)
fit_r2, psi_opt = ps.fit_great_circle(theta, phi)

print(f"Stripe strength: {stripe_score:.3f}")
print(f"Anisotropy (eigenvalue ratio): {anisotropy:.3f}")
print(f"Great-circle fit R²: {fit_r2:.3f}")

# Visualize
fig = ps.visualize_globe(
    theta, phi, 
    color=metadata["celltype"],
    title="My data on S²"
)
```

---

## Repository structure

```
pca_sphere_projection/
├── core.py                   # Spherical projection, rotation, coordinate extraction
├── sphere_stats.py           # Geometric metrics (stripe, anisotropy, great-circle)
├── topology.py               # Branching detection, graph-based analysis
├── comparison.py             # Cross-dataset Procrustes, conservation tests
├── perturbation.py           # Gene-displacement scoring 
├── robustness.py             # Root sensitivity, rotation robustness
├── entropy.py                # Entropy / stemness scoring 
├── gene_geometry.py          # Per-gene gradients on S², stripe-boundary ranking 
├── stripe.py                 # Stripe definition and binning utilities
├── io.py                     # Loaders for .mtx, .rds, .bz2, standard CSVs
├── preprocessing.py          # Normalization, HVG selection, filtering
├── known_regulators.py       # Curated regulator sets (celegan, hESC, klein, uc_epi)
├── pc_robustness.py          # PC-count, HVG-selection, and negative-control robustness
├── app.py                    # Streamlit dashboard
├── figures/                  # Manuscript figure generation
│   ├── fig1.py, fig2.py, fig3.py, fig4.py
│   ├── common.py             # Plotting utilities
│   └── supplements/          # Supplementary figures
├── __init__.py               # Public API exports (31+ functions)
└── ...

scripts/
└── make_manuscript_figures.py            # CLI for figure generation

tests/
├── test_sphere_stats.py, test_topology.py, test_comparison.py, ...
└── [33 test cases]           # Unit tests for all major modules

```

---

## Key concepts & interpretability

### What SPHERE-PCA assumes

- **PCA linearity**: The first three PCs are sufficient and meaningful. If your biology requires non-linear embedding, SPHERE-PCA will project it linearly.
- **L2-norm preservation**: Radial magnitude in original PC space becomes an interpretable coordinate. This is valid only if PC1–PC3 capture coordinated, continuous variation.
- **Fixed-loading model**: Gene displacement scores assume the PCA loadings remain constant under small perturbations (not experimentally validated).
- **Deterministic root**: The root must be biologically justified (e.g., a known stem population). Automatic root detection is not implemented.

### What SPHERE-PCA does NOT do

- **Does not infer causal trajectories**: θ and φ are geometric projections, not causal orderings. Validation requires independent biological scores (pseudotime, stemness, etc.).
- **Does not predict gene perturbation effects**: Fixed-loading scores are computational sensitivities, not CRISPR knockout predictions or in-silico perturbation forecasts.
- **Does not replace dimensionality reduction**: UMAP, t-SNE, and diffusion maps solve different optimization problems. SPHERE-PCA is orthogonal to these.
- **Does not handle time-series or dynamic transitions**: The method assumes static cell snapshots. Velocity-based methods (scVelo, RNA velocity) are complementary.

---

### Reproducibility

All figure panels are generated deterministically from configuration files (`YAML`). Each panel run saves:
- PNG figure
- Configuration (YAML)
- Underlying data (CSV)

To reproduce a single figure:

```bash
python scripts/make_manuscript_figures.py --figure 3 --panel A_hesc --seed 0
```

To reproduce all figures:

```bash
python scripts/make_manuscript_figures.py --all-panels --seed 0
```

**Package versions** are pinned in [`requirements.txt`](requirements.txt) and [`pyproject.toml`](pyproject.toml) (NumPy, pandas, SciPy, Matplotlib, Statsmodels, Scanpy, gseapy). Enrichr library snapshot date: **May 11, 2026**. Use a compatible Python environment (3.9+) for exact reproducibility.

---

## Citation

If you use SPHERE-PCA in your research, please cite:

> Yuan, L., *et al.* Interpretable spherical geometry of single-cell state transitions from dominant principal components. *bioRxiv*, 2026.

Or use the [`CITATION.cff`](CITATION.cff) file for BibTeX / GitHub citation formatting.

---

## Testing

```bash
python -m pytest tests/ -v
```

**Current coverage:** 33 test cases across `sphere_stats`, `topology`, `comparison`, `perturbation`, and `robustness`. Core API (`core.py`) is hand-validated via the example workflows.

---

## License

MIT — see [`LICENSE`](LICENSE) for details.

---

## Authors

**Long Yuan** ([lyuan13[at]jhmi.edu](mailto:lyuan13@jhmi.edu))

*Johns Hopkins University, Department of Immunology and Computer Science*

---

## Issues & Contributing

Please report bugs, feature requests, or questions via GitHub Issues. Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md) if present, or open an issue for guidance.



