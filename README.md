# pca_sphere_projection — SPHERE-PCA

**Spherical PCA Hypothesis Explorer for Regulatory Trajectories and
Embeddings.** A Python package and local web dashboard for projecting
single-cell PCA embeddings onto a unit sphere, quantifying their
geometric structure, and running the runnable subset of the H1–H7
hypotheses from
[`PNAS_EXTENSION_PROPOSAL.md`](PNAS_EXTENSION_PROPOSAL.md).

It is a **diagnostic** framework. It does **not** infer trajectories, and
its perturbation module is a fixed-PCA-loading sensitivity analysis, not a
causal in-silico knockout.

---

## What the package does

| Capability | Module | Entry-points |
|---|---|---|
| Project PC1–PC3 onto S² and align a chosen root cluster to the north pole | `core` | `align_to_north_pole`, `apply_euler_rotation`, `equirectangular_projection`, `visualize_globe` |
| Quantify spherical structure with proper geometry | `sphere_stats` | `stripe_strength_score`, `spherical_anisotropy`, `fit_great_circle`, `geodesic_gradient`, `spherical_kde`, `horseshoe_null_test` |
| Detect branching vs. linear trajectories | `topology` | `detect_branchpoints`, `linear_vs_branching_score`, `build_spherical_knn_graph` |
| Compare datasets / replicates / species on S² | `comparison` | `procrustes_align_spheres`, `conserved_stripe_test`, `spherical_replicate_residual` |
| Quantify root- and rotation-choice sensitivity | `robustness` | `root_sensitivity_analysis`, `rotation_robustness_analysis`, `compare_manual_vs_great_circle`, `pc_coordinate_quality_summary` |
| Fixed-PCA-loading gene perturbation vector field | `perturbation` | `compute_gene_perturbation_vectors`, `decompose_perturbation_vectors`, `rank_genes_by_perturbation_magnitude`, `cluster_genes_by_perturbation_signature` |
| Local interactive dashboard | `app` | `sphere-trace` (CLI) / `python -m pca_sphere_projection.app` |
| **Raw cell × gene I/O for .mtx, .rds, .bz2** | `io` | `load_celegan`, `load_uc_epi`, `load_hesc`, `load_bz2_klein_dataset`, `load_expression_matrix`, `match_expression_to_pc_csv` |
| **Preprocessing: log-norm, HVG selection, mito/ribo/cell-cycle filtering, comparative PCA** | `preprocessing` | `normalize_log1p`, `select_hvgs`, `filter_genes`, `fit_pca_embedding`, `compare_hvg_vs_all_gene_pca` |
| **Per-cell entropy / stemness scoring (Shannon, CytoTRACE proxy, SCENT proxy) and the H1 geodesic-gradient test** | `entropy` | `compute_transcriptional_entropy`, `compute_cytotrace_proxy`, `compute_scent_proxy`, `test_entropy_geodesic_gradient` |
| **Per-gene gradient on S² and stripe-boundary vs. along-trajectory ranking (H5)** | `gene_geometry` | `compute_gene_geodesic_gradients`, `rank_stripe_boundary_genes`, `rank_along_trajectory_genes`, `decompose_gene_gradient_relative_to_stripes`, `plot_gene_gradient_field` |
| **Curated, conservative known-regulator sets for celegan / hESC / klein / uc_epi** | `known_regulators` | `REGULATOR_SETS`, `get_regulator_set`, `annotate_overlap` |

---

## What the package needs as input

The minimum input is a CSV with three columns of pre-computed PCA scores
plus a categorical label, like the four examples in [examples/](examples/):

| Column | Required? | Notes |
|---|---|---|
| `PC1`, `PC2`, `PC3` | yes | top three PC scores per cell. |
| label column (e.g. `celltype`, `cluster`) | yes | used for picking the root and for visual colouring. |
| pseudotime column (e.g. `cluster`) | optional | enables θ-vs-pseudotime tests, branchpoint detection, linear-vs-branching score. Can be ordinal or `lo-hi` time bins (parser configurable). |
| gene expression matrix (`cells × genes`) | optional | unlocks H5 (stripe-boundary genes) and H7 (perturbation vector fields). |
| sklearn PCA loadings (`components × genes`) | optional | required for H7. |
| second matched dataset | optional | unlocks H3 (cross-dataset conservation) and H6 (replicate residual). |

Per-dataset parameters (root cluster, Euler angles, lineage orderings,
stripe ranges, candidate-roots for the sweep) live in
[examples/example_configs.yaml](examples/example_configs.yaml). **Do not
hard-code these inside functions.**

---

## Install

```bash
git clone <repo>
cd pca_sphere_projection
pip install -e .
pip install -e .[app]      # adds streamlit for the dashboard
pip install -e .[test]     # adds pytest
```

---

## Run the example analyses

```bash
# Run the full hypothesis suite on every CSV in examples/example_configs.yaml.
python scripts/run_pnas_hypothesis_suite.py \
    --config examples/example_configs.yaml \
    --outdir outputs/hypothesis_suite

# Or restrict to a subset:
python scripts/run_pnas_hypothesis_suite.py --datasets klein planaria
```

For each dataset the script writes
`outputs/hypothesis_suite/<dataset>/`:

- `metrics.json`, `metrics.csv` — every metric, machine-readable.
- `report.md` — short scientific narrative with caveats.
- `sphere_3d.html`, `equirectangular_2d.html` — interactive Plotly views.
- `theta_vs_pseudotime.html`, `branchpoints_3d.html` — only when pseudotime
  is configured.
- `stripe_density.html`, `root_sensitivity.html`,
  `rotation_robustness.html` — geometric robustness checks.

A consolidated [hypothesis status table](outputs/hypothesis_suite/hypothesis_status_table.md)
is also produced, marking each H1–H7 across each dataset as runnable or
not from the available CSV.

---

## Raw-expression validation workflow

The PC1–PC3 CSV pipeline above starts from pre-computed PCs. The
**raw-expression validation workflow** instead starts from the original
cell × gene matrices and re-derives PCA, geometry, gene gradients, and
fixed-loading perturbations end-to-end. It is what the H1, H4, H5, H7
hypotheses in the proposal actually need.

### Datasets covered

Place these under `raw_data/`:

| Dataset | Format | What's there | What's missing |
|---|---|---|---|
| C. elegans (Packer) | `celegan.mtx` + sidecar tsvs | counts, gene symbols, cell barcodes, embryo-time bins, celltype, batch | celltype is `NA` for ~half of cells |
| UC epithelium (Smillie) | `uc_epi.mtx` + sidecar tsvs | counts, cell barcodes, celltype, health/location/patient batch | **no gene-name file** — features are anonymous indices |
| Klein mESC | four `GSM1599*.csv.bz2` | counts, mouse gene symbols, day labels (d0/d2/d4/d7) | original cell barcodes were not preserved in the column headers |
| hESC (CytoTRACE example) | `dataset.rds` (read with `rdata`) | log-norm exprMatrix, gene symbols, cell IDs, phenotype, **real CytoTRACE rank scores** | nothing material |
| planaria | `Planaria.csv` | already a 50-PC matrix | raw counts not present — planaria is excluded from this workflow |

A full inventory is at
[outputs/raw_expression_validation/data_inventory.md](outputs/raw_expression_validation/data_inventory.md).

### Run

```bash
# Full workflow: H1, H4, H5, H7 on celegan, uc_epi, klein, hESC.
python scripts/run_raw_expression_validation.py \
    --raw-data raw_data \
    --out outputs/raw_expression_validation

# Restrict datasets / cell counts:
python scripts/run_raw_expression_validation.py \
    --datasets klein hesc \
    --max-cells 0          # disable the random subsample cap

# After the orchestrator, generate publication PNGs:
python scripts/plot_raw_validation_figures.py \
    --out outputs/raw_expression_validation \
    --raw-data raw_data
```

For each dataset the orchestrator writes
`outputs/raw_expression_validation/<dataset>/`:

- `pca_comparison_metrics.csv` — PC1–3 variance, stripe strength,
  great-circle R², linear anisotropy, θ-vs-pseudotime ρ for HVG / all-gene
  / no-mito-ribo / random-matched-set PCA.
- `H1_entropy_gradient_metrics.csv` — Spearman ρ of entropy/stemness vs.
  geodesic distance from the stem anchor (and vs. plain Euclidean PC
  distance for comparison). hESC uses the **precomputed CytoTRACE
  rank**; the others use a labelled Shannon-entropy proxy.
- `H4_radial_angular_metrics.csv` — Spearman ρ of radial norm and θ
  against cell-cycle / mito / ribo / entropy / pseudotime scores.
- `H5_stripe_boundary_genes.csv`, `H5_along_trajectory_genes.csv` —
  per-gene grid-based variation on S²; ranked phi-variation vs.
  theta-variation, with a `is_known_regulator` overlap column.
- `H7_perturbation_gene_rankings.csv` — fixed-loading sensitivity
  magnitude per gene, decomposed into radial / θ / φ tangent components
  on S². **Not a CRISPR/RNAi prediction.**
- `per_cell_geometry.csv`, `_artifacts.npz` — per-cell PC1–3, (x, y, z),
  θ, φ, radial norm, entropy score, plus all metadata; consumed by the
  plotting script.

A combined `outputs/raw_expression_validation/final_summary.md` with a
per-dataset hypothesis-support table is written at the end of every run.

### What the workflow can and cannot conclude

- **Can:** quantify the strength of an entropy gradient along geodesic
  distance; show whether radial norm and θ correlate with different
  biology; produce per-gene phi/θ-variation rankings; rank genes by
  fixed-loading sensitivity magnitude on S².
- **Cannot:** prove H1 from a Shannon-entropy proxy alone; declare a
  stripe-boundary gene a causal regulator without CRISPR/RNAi; treat
  H7's perturbation magnitudes as in-vivo knockout effects.

---

## Launch the local dashboard

```bash
sphere-trace
# equivalent:
python -m pca_sphere_projection.app
```

The app opens in your browser. Sidebar controls:

- **Source.** Pick a built-in example (loads its config) or upload a CSV.
- **Annotations.** Choose label, root, pseudotime, and pseudotime parser.
- **Alignment.** Manual Euler angles *or* automated great-circle alignment.
- **Geometry params.** Stripe-bin count and kNN k.
- **Robustness.** Number and magnitude of rotation perturbations.

The dashboard renders 3D sphere, equirectangular 2D map, stripe density,
θ-vs-pseudotime, branchpoint score, root-sensitivity bars, rotation
robustness histogram, and exposes optional H5 / H7 panels behind file
uploaders. **Warning banners appear when a hypothesis cannot be tested
from the current input** (e.g. CSV already L2-normalised → H4 not
testable; no pseudotime → H2 disabled).

Outputs can be exported as JSON metrics or processed-coordinate CSV.

---

## Reviewer-response robustness workflow

The prompt-driven reviewer response in
`PC_robustness_cytotrace_neg_ctrl_050526` is implemented by:

```bash
python scripts/run_pc_robustness_cytotrace_neg_ctrl.py
```

It writes outputs under `outputs/pc_robustness/`,
`outputs/stripe_robustness/`, `outputs/H1_cytotrace_only/`,
`outputs/H5_geodesic_gradient/`, `outputs/root_robustness/`, and
`outputs/negative_control/`.

Key framing updates:

- PC1-PC3 are used because S² needs three Euclidean axes and these are the
  dominant orthogonal PCA axes, but interpretation now depends on PC-count,
  random-PC, and HVG-vs-all-gene robustness.
- The legacy single stripe metric is now named
  `global_longitude_concentration`. The preferred reported metric is
  `multi_stripe_strength` with per-stripe tables.
- Main H1 evidence is restricted to datasets with real CytoTRACE/stemness
  columns. Proxy entropy is exploratory.
- H5 now includes a per-cell geodesic-gradient refinement with housekeeping
  and low-specificity penalties; it remains a screening analysis.
- BrCa atlas is treated as a non-developmental control. Any structure in it
  should be interpreted as patient, subtype, batch, cell-composition, or
  tumor-program structure unless an independent developmental score exists.

---

## What each hypothesis requires

See the full table in
[outputs/hypothesis_suite/hypothesis_status_table.md](outputs/hypothesis_suite/hypothesis_status_table.md)
or the proposal. Quick summary:

| H | Idea | Runnable from PC-only CSV? | What's missing |
|---|---|---|---|
| H1 | Differentiation entropy decays along geodesics | no | external CytoTRACE / SCENT / SLICER scalar |
| H2 | Branching vs. linear trajectory test | yes if pseudotime present | continuous DPT/scVelo would be stronger |
| H3 | Conserved stripes across species | no | second matched dataset |
| H4 | Radial component encodes cell-cycle / metabolic state | no when CSV pre-normalised | pre-normalisation PC scores + cycle/metabolic scores |
| H5 | Stripe-boundary genes are switch-like regulators | no | gene × cell expression matrix |
| H6 | Replicate-residual QC | no | matched replicate dataset |
| H7 | Gene perturbation vector fields | no | expression matrix + sklearn PCA loadings + selected genes |

---

## What can and cannot be concluded from PC1–PC3-only CSVs

**Can be concluded.** Whether the embedding has stripe-like geometry
(`spherical_anisotropy`, `stripe_strength_score`); whether one principal
great circle captures most of the variance (`fit_great_circle`); whether
ordinal cell-type rank correlates with polar angle (Spearman ρ on θ);
whether a branching model beats a linear one given that ordinal
pseudotime (`linear_vs_branching_score`); how sensitive the geometry is
to the root cluster (`root_sensitivity_analysis`) and the manual Euler
choice (`rotation_robustness_analysis`, `compare_manual_vs_great_circle`).

**Cannot be concluded.** Causal trajectory direction; gene-level
mechanism; cross-species conservation; whether the radial coordinate
encodes biology; whether perturbing a gene in vivo would move cells the
way the sensitivity analysis suggests; whether the geometry would
survive re-PCA with different HVG selection. Every notebook example
ships with a hand-tuned root and Euler rotation; any biological claim
must be reported alongside its `root_sensitivity_analysis` and
`rotation_robustness_analysis` envelopes.

---

## Test suite

```bash
python -m pytest tests/ -q
```

Currently covers `sphere_stats`, `topology`, `comparison`, `perturbation`,
and `robustness`. The original `core.py` API is unchanged.

---

## License

MIT — see [LICENSE](LICENSE).
