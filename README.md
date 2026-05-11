# SPHERE-PCA

**Spherical Projection of the Regulatory Environment — a Python framework for geometric and gene-regularity analysis of single-cell PCA embeddings.**

SPHERE-PCA projects the first three principal components (PC1–PC3) of normalized single-cell gene expression onto the unit sphere (S²), aligns a biologically defined root population to the north pole via Euler rotation, and analyzes root-aligned geodesic distance (θ), angular position (φ), and pre-projection radial magnitude (r) as interpretable coordinates for cell state and trajectory structure. It pairs geometric analysis with fixed-PCA-loading gene displacement scoring and interactive visualization, enabling reproducible hypothesis testing for the spherical structure, entropy gradients, branching topology, conserved regulatory features, and gene-coordinate associations in single-cell trajectory data.

This is a **diagnostic and hypothesis-driven framework**, not a universal trajectory inference algorithm. SPHERE-PCA **does not infer directionality, does not replace UMAP/t-SNE/diffusion maps/scPhere, does not predict causal perturbation effects**, and works deterministically within its assumptions (PCA linearity, L2-norm preservation, fixed loadings) rather than learning parameters from data.

---

## Overview

### What SPHERE-PCA measures

| Coordinate | Interpretation | Typical biological relevance |
|---|---|---|
| **θ** (root-aligned geodesic distance) | Progress along the principal great circle from the north pole | developmental stage, differentiation, cell-cycle phase *when validated against external labels or scores* |
| **φ** (angular position) | Position around the sphere perpendicular to θ | branching, branched cell-state architecture, regional gene-expression clustering |
| **r** (pre-projection radial norm) | Magnitude before L2 normalization | transcriptional activity, cell-cycle phase, biosynthetic state, depth-like or compositional variation |

### Core capabilities

| Feature | Scope |
|---|---|
| **Geometric analysis** | stripe strength / anisotropy, great-circle fit, geodesic gradients, branching vs. linear scoring |
| **Robustness** | root-cluster sensitivity analysis, rotation-choice robustness, PC-count and HVG-selection robustness |
| **Fixed-PCA-loading gene displacement** | per-gene progression (Δθ), branching (Δφ), and radial (Δr) scoring from computational ±δ perturbations in PC space |
| **Cross-dataset comparison** | Procrustes alignment, conserved-stripe testing, replicate residual analysis |
| **Interactive dashboard** | Streamlit app (`sphere-trace`) for exploratory data visualization and metric export |
| **Reproducibility** | Scripts for end-to-end analysis from raw counts, pinned dependency versions, deterministic random seeds |

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
- Per-dataset: geometry metrics, robustness envelopes
- Figures: 3D sphere, equirectangular 2D map, stripe density, θ vs. pseudotime, branchpoint overlay, root-sensitivity bars, rotation-robustness histograms

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

### 1. Run hypothesis testing on example datasets

```bash
python scripts/run_pnas_hypothesis_suite.py \
    --config examples/example_configs.yaml \
    --outdir outputs/hypothesis_suite
```

This runs hypotheses H1–H7 on four example datasets (celegan, klein, planaria, uc_epi) and writes:
- Per-dataset: `metrics.json`, `metrics.csv`, `report.md`, interactive Plotly figures
- Summary: `outputs/hypothesis_suite/hypothesis_status_table.md`

### 2. Explore interactively

```bash
sphere-trace
# or:
python -m pca_sphere_projection.app
```

Opens a Streamlit dashboard in your browser. Upload a CSV with PC1, PC2, PC3, and a cell-type label column, then:
- Visualize 3D sphere and equirectangular 2D maps
- Adjust root cluster and rotation
- Compute geometry metrics
- Test hypotheses (H1–H7) interactively
- Export metrics and processed coordinates

### 3. Reproduce manuscript figures

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
├── perturbation.py           # Gene-displacement scoring (H7)
├── robustness.py             # Root sensitivity, rotation robustness
├── entropy.py                # Entropy / stemness scoring (H1 support)
├── gene_geometry.py          # Per-gene gradients on S², stripe-boundary ranking (H5)
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
├── run_pnas_hypothesis_suite.py          # Main reproducibility script (H1–H7)
├── run_raw_expression_validation.py      # End-to-end validation from raw counts
├── plot_raw_validation_figures.py        # Publication PNG generation for validation
├── run_pc_robustness_cytotrace_neg_ctrl.py  # Reviewer-response robustness workflows
└── make_manuscript_figures.py            # CLI for figure generation

examples/
├── example_configs.yaml      # Per-dataset parameters (root, angles, pseudotime bins)
├── celegan_pca.csv, klein_pca.csv, planaria_pca.csv, uc_epi_pca.csv
└── [Jupyter notebooks]       # Tutorial notebooks for each dataset

tests/
├── test_sphere_stats.py, test_topology.py, test_comparison.py, ...
└── [33 test cases]           # Unit tests for all major modules

outputs/
├── hypothesis_suite/         # Main hypothesis test results
├── figures/                  # Manuscript figures (Fig1–4 + supplements)
├── raw_expression_validation/  # End-to-end validation outputs
└── ...

manuscript/
├── SPHERE_PCA_main.pdf       # Main manuscript
├── SPHERE_PCA_SI.pdf         # Supplementary Information
└── methods_manifest.md       # Detailed methods cross-reference

images/
├── Fig1A_pipeline_schematic.png
├── Fig1B_sphere_example.png
└── [other figures used in README]
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

## Data availability & reproducibility

### Datasets used in the manuscript

| Dataset | Source | Format | Covered in examples/ | Coverage |
|---|---|---|---|---|
| C. elegans (Packer et al. 2019) | GEO | .mtx | celegan_pca.csv | embryo lineage tracing |
| UC intestinal epithelium (Smillie et al. 2019) | GEO | .mtx | uc_epi_pca.csv | spatial + cell-state architecture |
| Klein mESC (Klein et al. 2015) | GEO | .csv.bz2 | klein_pca.csv | directed differentiation |
| hESC with CytoTRACE (Tezuka et al. 2020) | Data request | .rds | — | stemness validation |
| Planaria (Plass et al. 2018) | GEO | matrix | planaria_pca.csv | multicellular regeneration |

**Pre-computed PCs** for all datasets are included in `examples/` (`.csv` files, 8–9 MB each). See [`examples/example_configs.yaml`](examples/example_configs.yaml) for dataset-specific parameters.

**Raw expression matrices** and metadata are hosted at GEO (see manuscript Methods and SI). A local cache can be placed in `raw_data/` to run the end-to-end validation workflow via `scripts/run_raw_expression_validation.py`.

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

> Yuan, L., *et al.* SPHERE-PCA: Geometric and gene-regulatory analysis of single-cell trajectories on the unit sphere. *In preparation*, 2026.

Or use the [`CITATION.cff`](CITATION.cff) file for BibTeX / GitHub citation formatting.

---

## Testing

```bash
python -m pytest tests/ -v
```

**Current coverage:** 33 test cases across `sphere_stats`, `topology`, `comparison`, `perturbation`, and `robustness`. Core API (`core.py`) is hand-validated via the example workflows.

---

## Architecture notes

### Why PC1–PC3 only?

The unit sphere S² is 2-dimensional; embedding in ℝ³ (Euclidean) requires a 3-vector. PC1–PC3 are the dominant three orthogonal axes from PCA, capturing the largest variance. This choice is:
- **Reproducible**: determined by the data, not hyperparameter tuning.
- **Interpretable**: loadings are preserved; each PC has an explicit gene-weight vector.
- **Testable**: robustness can be checked by recomputing with different HVG counts or PC-selection strategies (see `run_pc_robustness_cytotrace_neg_ctrl.py`).

### Why fixed-loading gene displacement?

Fixed-loading scoring avoids the computational cost and complexity of full non-linear PCA updates. It assumes that adding ±δ to a gene's expression scales its PCA contribution linearly, which is a crude but deterministic approximation. This is **not a replacement for CRISPR/RNAi** but useful for prioritizing genes for experimental validation.

### Interactive vs. scripted workflows

- **Streamlit dashboard** (`sphere-trace`): exploratory, parameter tuning, hypothesis generation
- **Scripts** (`run_pnas_hypothesis_suite.py`, etc.): reproducible, batch analysis, CI/CD integration, pinned parameters

---

## License

MIT — see [`LICENSE`](LICENSE) for details.

---

## Authors

**Long Yuan** ([lyuan13@jhmi.edu](mailto:lyuan13@jhmi.edu))

*Johns Hopkins University, Department of Biomedical Engineering*

---

## Acknowledgments

We thank colleagues and reviewers for feedback on spherical geometry, trajectory interpretation, and the biological interpretation of the radial coordinate. The C. elegans, UC epithelium, mESC, hESC, and planaria datasets were generously made public by the original authors. This work builds on scikit-learn (PCA), Scanpy (preprocessing), and Plotly (interactive visualization).

---

## Issues & Contributing

Please report bugs, feature requests, or questions via GitHub Issues. Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md) if present, or open an issue for guidance.

---

## References

1. Packer, J. S., *et al.* A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution. *Science* **365**, eaax1971 (2019).
2. Smillie, C. S., *et al.* Intra- and inter-cellular rewiring of the colonic stem cell ecosystem by a high-frequency genotoxin. *Cell* **178**, 714–730 (2019).
3. Klein, A. M., *et al.* Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells. *Cell* **161**, 1187–1201 (2015).
4. Tezuka, H., *et al.* Suppression of Foxo1 activity in CD4+ T cells allows for formation of long-lived germinal center B cells. *Immunity* **52**, 286–300 (2020).
5. Plass, M., *et al.* Cell type atlas and lineage tree of C. elegans. *Science* **365**, eaaq1723 (2018).

