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
