# PNAS-Level Extension Proposal: PCA Sphere Projection

A critical analysis and methodological / biological extension plan for
[`pca_sphere_projection`](pca_sphere_projection/), prepared for high-impact
journal submission.

---

## 2026-05-05 reviewer-response update

The package now includes a runnable robustness/control workflow:
`scripts/run_pc_robustness_cytotrace_neg_ctrl.py`. It addresses the major
reviewer concerns without changing the central caveat: spherical PCA is a
diagnostic coordinate system, not a trajectory inference method.

### PC dimension justification

PC1-PC3 are no longer justified only by variance explained. They are used
because projection to S² requires three Euclidean axes and PC1-PC3 are the
dominant orthogonal PCA axes. Manuscript-level claims must be conditional on
the output of `outputs/pc_robustness/<dataset>/pc_dimension_summary.csv`,
including PC-count, random-PC-triplet, and HVG-vs-all-gene comparisons.

### Multi-stripe structure metric

The old global stripe score has been renamed
`global_longitude_concentration`. The new `multi_stripe_strength` reports
detected stripe counts and per-stripe length/width/strength, so multiple
arcs are not collapsed into one weak global entropy score. This is useful
but not itself biological evidence: BrCa can also show fragmented
multi-stripe geometry.

### H1 restricted to real CytoTRACE/stemness

The main H1 table now uses only real CytoTRACE/stemness columns from hESC,
planaria, human germ cell, and pre-implantation human embryo. Shannon
entropy remains supplemental/exploratory and should not support the main
claim.

### H5 per-cell geodesic-gradient analysis

H5 has been upgraded from coarse grid scoring to a per-cell local
geodesic-gradient refinement after HVG prefiltering. Rankings include
theta and phi tangent components, localized phi gradients, expression
specificity, and housekeeping penalties. These are candidate regulators
only; H5 is unsupported if top hits remain housekeeping-like or lack
known-regulator/perturbation validation.

### Root and PC robustness

Root sensitivity is now reported across high-stemness roots, low-stemness
roots, phenotype roots, and random roots. PC robustness is reported across
PC-count summaries, top-three versus random PC triplets, and feature-set
choices.

### Negative control: BrCa atlas

The BrCa atlas is included as a non-developmental control. The correct
interpretation is not "must be negative"; rather, any observed spherical
structure should be checked against patient, subtype, batch, tumor program,
and major cell-composition annotations before developmental language is
used.

### Current readiness verdict

The workflow materially improves PC justification, H1 score provenance,
multi-stripe quantification, root robustness, and negative-control coverage.
Remaining blockers for a PNAS submission are independent biological
validation of H5, stronger null models for multi-stripe detection, and
clear evidence that PC/random-root robustness supports each dataset-specific
claim.

---

## Phase 1 — Codebase Understanding

### 1.1 Repository layout
- [pca_sphere_projection/core.py](pca_sphere_projection/core.py): all current logic
  (~590 lines, 7 public functions).
- [examples/](examples/): four Jupyter notebooks (C. elegans embryo, planaria
  whole-body atlas, Klein iPSC-to-day7, UC epithelium) plus their pre-computed
  PC1–PC3 CSVs.
- [setup.py](setup.py), [requirements.txt](requirements.txt), [README.md](README.md).

### 1.2 Data flow (current pipeline, plain language)

```
HVG-filtered counts  ──►  PCA (k=3, externally)  ──►  CSV with PC1, PC2, PC3, celltype
                                                              │
                                                              ▼
                                            align_to_north_pole(root_node centroid)
                                                              │
                                                              ▼
                                          apply_euler_rotation(angles_x, angles_y, angles_z)
                                                              │
                                                              ▼
                                       (longitude, latitude) = (atan2(y,x), arcsin(z/r))
                                                              │
                              ┌───────────────────────────────┼────────────────────────────────┐
                              ▼                               ▼                                ▼
                  visualize_globe (3D)         equirectangular_projection (2D)     visualize_lineage_correlation
                                                                                              │
                                                                                              ▼
                                                                       statistical_test_on_lineages (KS / t-test on θ)
                                                                                              │
                                                                                              ▼
                                                              find_and_visualize_correlated_genes (per-stripe Spearman)
```

### 1.3 What each function actually does

| Function | What it does | Mathematical step |
|---|---|---|
| [`align_to_north_pole`](pca_sphere_projection/core.py#L13-L82) | L2-normalises every PC vector onto S², then computes a Rodrigues rotation that takes the **mean of the user-specified root cluster** to (0,0,1). | Radial projection ‖x‖₂⁻¹·x; Rodrigues rotation; **note**: the centroid is computed on the un-normalised PCs but only its direction is used, so the rotation is correct directionally but ignores intra-cluster magnitude variance. |
| [`apply_euler_rotation`](pca_sphere_projection/core.py#L84-L114) | Applies an `xyz`-Euler rotation (user-tuned by hand) and re-normalises. Cosmetic. | `scipy.spatial.transform.Rotation.from_euler` |
| [`equirectangular_projection`](pca_sphere_projection/core.py#L116-L194) | `λ = atan2(y,x), φ = arcsin(z/r)`. Plain Plate-Carrée. | Distorts area away from the equator; preserves no metric exactly. |
| [`visualize_globe`](pca_sphere_projection/core.py#L196-L295) | Static 3D scatter (matplotlib) or interactive (plotly). | None. |
| [`calculate_spherical_angles`](pca_sphere_projection/core.py#L297-L308) | (θ, φ) — polar angle from +z and azimuth in xy. | Pure trig. |
| [`visualize_lineage_correlation`](pca_sphere_projection/core.py#L310-L411) | **Important:** maps each cell to its **rank in a user-supplied cell-type list** (not a real pseudotime), then computes Pearson r between this rank and θ. | The "pseudotime" here is a discrete ordinal; Pearson r on ordinal data inflates significance and depends entirely on the user's ordering. |
| [`statistical_test_on_lineages`](pca_sphere_projection/core.py#L413-L481) | Shapiro normality → KS or Welch t-test on θ between two lineages. | Tests *whether* two distributions of polar angle differ — not whether the geometry is biologically meaningful. |
| [`find_and_visualize_correlated_genes`](pca_sphere_projection/core.py#L483-L589) | Bins cells by longitude into "stripes" (user-defined ranges), Spearman-correlates every gene against the cell-type rank within a stripe, BH-adjusts. | Same ordinal-rank pseudotime caveat. |

### 1.4 What the package does not do
- It does **not** perform HVG selection, normalisation, or PCA. The user supplies
  pre-computed PC1–PC3 in a CSV.
- It does **not** quantify whether the spherical structure is statistically
  more "stripey" than random (no null model).
- It does **not** preserve geodesic distance — equirectangular projection
  distorts area, and θ-based statistics implicitly weight points near the poles.
- It does **not** detect branching: every analysis assumes a linear lineage.

---

## Phase 2 — Critical Evaluation of the Current Scientific Claim

The current claim has four chained components. Each carries risk.

### 2.1 "PCA on HVGs produces a horseshoe."

**This is the weakest link.**

- The horseshoe / arch artefact is a known consequence of running linear PCA on
  data with strong unimodal gradients (Diaconis et al. 2008, Goodall 1991, and
  Morton et al. 2017 in microbiome). It appears whether or not the underlying
  manifold is actually 1D — and in fact appears most reliably when the data is
  a noisy linear ramp. So a horseshoe in PC1–PC2 is **not evidence of a
  developmental trajectory**; it is evidence of a dominant gradient plus the
  saturating non-linearity of L²-orthogonal projection.
- Three datasets that *do not* have a single dominant time axis (e.g. an adult
  homeostatic atlas, a cell-cycle dominated culture, a treatment-vs-control
  study) will **also** often show horseshoes, but they would be driven by cell
  cycle, batch, ribosomal content, etc.
- Risk of overgeneralisation: claiming the horseshoe is universal would invert
  causality. The horseshoe is a *projection artefact whose interpretation
  depends on the dominant variance source*, which must be identified
  independently (by gene-loadings inspection, by held-out batch labels, etc.).

### 2.2 "Projecting PC1–PC3 onto S² yields stripes."

- Radial L²-normalisation (`x ↦ x/‖x‖`) collapses radial information. If the
  point cloud is roughly conical (a single cone emanating from the origin),
  radial projection produces a tight cap. If it is roughly planar through the
  origin, radial projection produces a great circle. **A 1D horseshoe in 3D
  generically projects to an arc on the sphere, not stripes.** Stripes arise
  when the *length* of the horseshoe exceeds π in normalised units, so the
  arc wraps. Whether this happens depends on PC scaling (which is
  variance-dependent and dataset-specific).
- Therefore the appearance of stripes is partly a function of (i) how strongly
  PC1–PC3 dominate the embedding and (ii) the relative variance ratio across
  PCs. A different normalisation (e.g. ZCA-whitening) would change the
  pattern.
- Risk: "stripes" may be re-projections of the same biological gradient
  visualised under a non-isometric coordinate change — visually striking but
  not adding new information beyond what PC1 alone carries.

### 2.3 "Equirectangular projection + Euler rotation aligns stripes pole-to-pole."

- The Euler angles in [`apply_euler_rotation`](pca_sphere_projection/core.py#L84) are
  set **manually per dataset** (e.g. `[-30, 0, -50]` for *C. elegans*,
  `[-90, -50, 50]` for the README example). This is a hand-tuned rotation
  chosen to make a particular figure look clean. It is not a property of the
  data; it is a property of the figure. **Any quantitative claim that uses θ
  as a "developmental axis" is therefore conditional on a manual choice and
  cannot be reproduced without it.**
- Equirectangular projection introduces severe area distortion at high
  latitudes; statistics on (lon, lat) (e.g. linear regressions, KDE) are
  biased unless area-corrected.

### 2.4 "θ correlates with pseudotime."

This is **almost certainly true under the construction** because:

1. The root cluster is forced to the north pole.
2. The horseshoe runs from the root outward.
3. θ measures angular distance from the north pole.

So the construction *defines* a high correlation when there is any single
gradient from root → tip. The correlation reported is therefore largely a
self-fulfilling consequence of (a) choosing the root cluster and (b) choosing
the Euler angles.

What the paper **needs** to claim instead, to be rigorous:

> The angular separation along the principal great-circle axis of the
> spherical embedding recovers a continuous monotone ordering of cells that
> matches independently-inferred pseudotime (DPT, Palantir, RNA velocity,
> staged scRNA-seq), with effect sizes that exceed those expected under
> matched null models.

### 2.5 Concrete risks to call out in the manuscript

| Risk | What it would look like | How to defuse |
|---|---|---|
| Horseshoe artefact masquerading as biology | Same stripe pattern emerges on simulated linear-gradient data with no branching | Permutation null + simulated controls (Phase 4) |
| Root-cluster bias | Choosing a different "root" produces a different θ-pseudotime correlation | Sensitivity analysis across alternative roots |
| PC count dependence | Adding PC4–PC10 changes the topology | Explicitly justify k=3 with a scree/ID-estimation argument |
| Manual Euler tuning | Stripes only line up after dataset-specific knob-twiddling | Replace with automated great-circle fitting (Phase 4) |
| Linear-trajectory assumption | Branching lineages get squashed into one stripe | Branching score (Phase 4) |
| Cell-type rank ≠ pseudotime | Pearson r looks high because of ordinal compression | Use continuous external pseudotime (DPT, scVelo) |

---

## Phase 3 — Biological Interpretations (Novel, Testable)

Each interpretation goes beyond "θ ≈ pseudotime" and treats the spherical
embedding as a *coordinate system* with explicit geometric meaning, not a
visualisation trick.

### H1. "Differentiation entropy decays along geodesics, not Euclidean PC distance."

**Hypothesis.** If the spherical embedding faithfully captures fate
commitment, then transcriptional entropy (e.g. SCENT or CytoTRACE score)
should decrease monotonically along **geodesic distance from the root**, with
a steeper slope than along Euclidean PC distance, because radial L²
normalisation removes magnitude-only variation that is uncorrelated with
fate.

**Why the spherical transformation reveals something new.** The sphere
discards |x|, which is dominated by total mRNA / library size in
normalised-but-not-batch-corrected data. Geodesic distance therefore measures
*direction of transcriptional state change*, isolating fate-relevant variance
from amplitude-relevant variance.

**Validation data.** (i) Any dataset with paired CytoTRACE / StemID
predictions (planaria, *C. elegans*, mouse gastrulation Pijuan-Sala 2019).
(ii) Lineage-traced datasets (LARRY, Weinreb et al. 2020) where ground-truth
fate probability is known per clone.

**Pitfalls.** Library-size confounding; entropy estimators are noisy at low
counts. Mitigate by downsampling to common depth and using
clone-resolved data.

---

### H2. "Bifurcation events leave detectable angular-divergence signatures on the sphere."

**Hypothesis.** Where a lineage bifurcates, neighbouring cells on the sphere
should have *high local geodesic-gradient anisotropy* — i.e. their local
pseudotime gradient vectors fan out rather than align. Branching points are
therefore identifiable as cells whose tangent-space gradient field has high
angular variance among k-nearest neighbours.

**Why this is new.** Most branch detectors (Slingshot, Monocle3, PAGA)
operate on graph topology. Here the test is geometric and local: at a
true branchpoint, the field of "downstream" arrows on the sphere should
diverge, while along a pure linear stretch it stays parallel. This gives
single-cell-resolution branch labels rather than cluster-level edges.

**Validation.** Synthetic branching (dyngen, PROSSTT). Then planaria
(neoblast → neural / muscle / gut split is well annotated). Then mouse
hematopoiesis (LARRY MPP fate split).

**Pitfalls.** High-k-neighbour smoothing washes out true splits;
low-k inflates noise. Solution: scan k and report stability; require
bootstrap reproducibility.

---

### H3. "Conserved stripes across species reveal evolutionarily fixed transcriptional axes."

**Hypothesis.** When two species' developmental atlases are independently
embedded on S² and then aligned by orthologous transcription-factor loadings,
the residual angular discrepancy of homologous cell types is small for
deep-conserved programs (notochord, mesendoderm) and large for taxon-specific
programs (planarian neoblast specialisations, *C. elegans* P-lineage).

**Why this is new.** Existing cross-species comparisons rely on
embedding integration (Seurat CCA, scVI), which assumes shared latent
factors. Spherical alignment compares the *intrinsic geometry* of each
species independently embedded, providing a falsifiable null: if the geometry
is conserved, a single 3D rotation should align them.

**Validation.** *C. elegans* embryogenesis (Packer et al. 2019) vs.
ascidian embryogenesis (Cao et al. 2019). Or planaria vs. acoel
flatworm cell atlases.

**Pitfalls.** PC space is not directly comparable across datasets;
orthology mapping is lossy; allometric differences in cell-type abundance
distort PCA. Mitigate by computing PCs on the *intersection of one-to-one
orthologous TF loadings* rather than on independent HVGs.

---

### H4. "The radial component (discarded by L²-projection) is itself biologically informative."

**Hypothesis.** ‖PC₁:₃‖ before normalisation correlates with cell-cycle phase
or metabolic state, not lineage. The package currently throws this away.
Decomposing each cell's embedding into (radial, angular) coordinates allows
explicit separation of "where the cell is going" (angular = lineage) from
"how active it is" (radial = cell cycle / metabolism).

**Why this is new.** Cell-cycle regression is a standard preprocessing
step that *removes* this signal. We instead claim it should be *retained
as a separate coordinate*, because the radial ↔ angular split is a natural
factorisation of biological state.

**Validation.** Cell-cycle scored datasets (Tirosh et al. 2016 melanoma;
mouse intestinal organoids with Fucci reporters). Correlate ‖PC‖ with
S/G2M scores and with metabolic gene-set scores.

**Pitfalls.** Library-size will dominate ‖PC‖ in poorly normalised data.
Requires log-CPM or scTransform inputs.

---

### H5. "Stripe boundaries on the sphere correspond to GRN switch-point genes, not gradient genes."

**Hypothesis.** Genes whose expression is sharpest *across* a stripe
boundary (high spatial derivative on S²) are bistable transcription
factors driving fate decisions, while genes that vary smoothly along a
stripe (parallel to it) are gradient/cytoskeletal/metabolic.

**Why this is new.** The current
[`find_and_visualize_correlated_genes`](pca_sphere_projection/core.py#L483)
ranks genes by correlation with cluster rank inside one stripe. This finds
"genes that change with pseudotime" — already a solved problem. The new
test would rank genes by **angular gradient orthogonal to the local stripe
direction**, which selects for switch-like rather than gradient-like genes.

**Validation.** Compare retrieved gene lists against ChIP-seq–derived
master-regulator TF sets (e.g. Sox2, GATA1/PU.1, Notch effectors)
in mouse hematopoiesis.

**Pitfalls.** Numerically estimating directional derivatives on S² is
noisy. Use vMF-kernel local regression and bootstrap.

---

### H6. "Spherical embedding stability is a quality-control signature for atlas integration."

**Hypothesis.** Two technical replicates of the same biological sample
should produce *the same* spherical geometry up to a rotation. Batch
effects, in contrast, produce non-rigid distortions (specific stripes
shrink/expand). The Procrustes residual after best-rotation alignment is
therefore a per-cell-type batch-effect score.

**Why this is new.** Batch-correction QC is mostly visual (UMAP overlays)
or summary (kBET). Geometric alignment provides a **stripe-resolved,
cell-type-resolved** measure of integration failure.

**Validation.** Datasets with known batch structure (HCA Lung benchmark,
HuBMAP integration challenge).

**Pitfalls.** Real biological replicates also differ; need to control with
matched-individual designs.

---

### H7. "Gene perturbation vector fields reveal directional regulators of spherical geometry."

**Hypothesis.** For a fixed PCA embedding, the displacement induced by
zeroing or doubling a single gene's expression decomposes per-cell into a
*radial* component (changing how strongly the cell is committed) and a
*tangent* component, which itself splits into a θ-direction (along
pseudotime) and a φ-direction (across stripes). Genes whose displacements
are predominantly:

- **θ-aligned and pole-ward** are candidate drivers of pseudotime
  progression.
- **φ-aligned and stripe-crossing** are candidate switch-like fate-decision
  regulators (the geometric signature of bistable TFs).
- **radial-only** are candidate metabolic / cell-cycle / housekeeping
  modulators that contribute to amplitude but not direction.

Cells whose embedding is most sensitive to a gene (largest displacement
norm) localise the gene's *spatial* effect on the spherical landscape;
genes that share their sensitive-cell sets are candidates for the same
regulatory program.

**Why the spherical transformation reveals something new.** Classical
gene–pseudotime correlation tells you *whether* a gene varies along the
trajectory; this analysis tells you *which direction in the embedding it
pushes cells when its expression changes*, with explicit stripe-vs-along-
stripe decomposition. The same correlation can correspond to very different
geometric effects (radial-only vs. φ-aligned), and only the latter is
consistent with the gene being a fate regulator rather than a passenger
gradient.

**Validation data.**
- Genes annotated as master TFs (Sox2, GATA1/PU.1, Notch effectors, lin-26
  family in *C. elegans*) should have high φ-component scores at the
  appropriate stripe boundary.
- Cell-cycle genes (Top2a, Mki67) should be radial-dominant.
- Direct: comparing the predicted sensitive-cell set against held-out
  CRISPRi-FlowFISH or RNAi survival screens.

**Pitfalls — read these carefully before interpreting results.**

1. **This is not causal.** It is a sensitivity analysis under a *fixed* PCA
   loading matrix. Knocking out a gene in vivo also changes the loadings
   themselves (because downstream genes change too). Re-fitting PCA after
   the perturbation is the proper causal model and is *not* what this
   module does.
2. **Linear projection has zero asymmetry.** Without spherical
   normalisation, ``(zeroed + doubled) / 2 == original`` exactly, so the
   "asymmetry" metric is identically zero. The analysis becomes informative
   only when ``normalize_to_sphere=True`` (the radial projection's
   non-linearity creates the asymmetry).
3. **Zero/double is biologically artificial.** Doubling has no upper bound
   on the simplex of feasible expression states, and most genes never
   actually go to zero (their floor is the technical detection limit).
   Treat magnitude rankings as relative, not absolute.
4. **Loading scale matters.** A gene with a large loading wins not because
   it is biologically central but because the PCA chose to align an axis
   with it. Always normalise rankings by ``||pca_components[:, gene]||``
   when comparing across genes for biological interpretation.
5. **No regulatory feedback.** This is a "what if we directly edited PC
   scores" analysis. CRISPR / RNAi / lineage-tracing validation is
   mandatory before any causal claim.

---

## Phase 4 — Methodological / Tool Extensions

New modules to add to the package:

| Module | Public functions | Purpose |
|---|---|---|
| `sphere_stats` | `stripe_strength_score`, `spherical_kde`, `geodesic_gradient`, `fit_great_circle`, `spherical_anisotropy`, `horseshoe_null_test` | Quantify structure on S² with proper geometry. |
| `topology` | `build_spherical_knn_graph`, `detect_branchpoints`, `linear_vs_branching_score` | Distinguish linear from branching trajectories. |
| `comparison` | `procrustes_align_spheres`, `conserved_stripe_test`, `spherical_replicate_residual` | Compare datasets / replicates / species. |
| `perturbation` | `project_expression_to_pca`, `perturb_gene_expression`, `compute_gene_perturbation_vectors`, `rank_genes_by_perturbation_magnitude`, `decompose_perturbation_vectors`, `identify_sensitive_cells`, `plot_perturbation_vector_field`, `cluster_genes_by_perturbation_signature` | Fixed-PCA-loading sensitivity analysis (gene perturbation vector fields). |

Each is specified below. Inputs follow the existing convention: a DataFrame
with rotated PC columns. Outputs are DataFrames or dicts (no plotting
inside the stats — separate `_plot` helpers, like the existing code).

---

### 4.1 `sphere_stats.stripe_strength_score`

**Inputs**
- `coords`: (N,3) array of unit vectors (post-`apply_euler_rotation`).
- `n_bins`: int, longitudinal bins (default 36 → 10° each).
- `axis`: which axis to bin around (default z, i.e. longitudes).

**Output**
- `dict` with `score` (0–1, where 1 = perfect equipartition into bins),
  `entropy_observed`, `entropy_uniform`, `bin_counts`.

**Algorithm.**
Compute the Shannon entropy of the longitudinal distribution; compare
against entropy of a uniform-on-S² null (which is constant in longitude
but **not** in latitude — the projection introduces a cos φ Jacobian, so
the comparison must area-correct). Score = 1 − H_observed / H_uniform.
A genuine "stripe" pattern produces low H (peaked in a few bins).

---

### 4.2 `sphere_stats.spherical_kde`

**Inputs**
- `coords`: (N,3) unit vectors.
- `kappa`: vMF concentration (default 50).
- `eval_points`: optional (M,3) grid; default Fibonacci sphere with M≈4000.

**Output**
- (M,) density values; (M,3) eval_points.

**Algorithm.**
Sum-of-vMF kernels: f(x) = (1/N) Σᵢ Cₖ(κ) exp(κ xᵀxᵢ). Use this for any
density-based downstream analysis instead of (lon,lat) histograms, which
distort area.

---

### 4.3 `sphere_stats.geodesic_gradient`

**Inputs**
- `coords`: (N,3).
- `scalar_field`: (N,) — pseudotime, gene expression, or any per-cell scalar.
- `k`: neighbours (default 15).

**Output**
- (N,3) tangent-space gradient vectors (perpendicular to the local
  position vector).

**Algorithm.**
For each cell xᵢ:
1. Find k nearest neighbours by geodesic distance (= arccos xᵢᵀxⱼ).
2. Project neighbour displacements into the tangent plane at xᵢ:
   `vⱼ = (xⱼ − xᵢ) − ((xⱼ − xᵢ)ᵀ xᵢ) xᵢ`.
3. Solve a weighted least-squares regression `Δscalar ≈ vⱼᵀ g` for
   gradient vector `g` in the tangent plane.

This is the spherical analogue of finite differences and underlies H1, H2,
H5.

---

### 4.4 `sphere_stats.fit_great_circle`

**Inputs**
- `coords`: (N,3).
- `weights`: optional (N,).

**Output**
- `dict` with `normal` (3-vector to the best-fit plane through origin),
  `mean_residual_radians`, `r_squared`.

**Algorithm.**
The best-fit great circle is the plane through the origin that minimises
weighted squared sine of angular deviation. Solve as the eigenvector of
the smallest eigenvalue of the weighted Σ xᵢxᵢᵀ. Use as a *replacement*
for manual Euler tuning: the Euler rotation that aligns the data's
principal great circle to the equator is fully determined by `normal`.

---

### 4.5 `sphere_stats.spherical_anisotropy`

**Inputs**
- `coords`: (N,3).

**Output**
- `dict` with eigenvalues `λ₁ ≥ λ₂ ≥ λ₃` of (1/N) Σ xᵢxᵢᵀ, and three
  derived shape descriptors:
  - `linear = (λ₁ − λ₂) / λ₁` (great-circle arc-like)
  - `planar = (λ₂ − λ₃) / λ₁` (great-circle ring-like)
  - `spherical = λ₃ / λ₁` (uniform on S²)

**Algorithm.**
Standard inertia-tensor decomposition. Provides the first principled
test of "is this dataset stripe-shaped at all?" before invoking
biological interpretations.

---

### 4.6 `sphere_stats.horseshoe_null_test`

**Inputs**
- `expression_hvg`: (N, G) HVG-filtered, log-normalised matrix.
- `n_perm`: int, default 200.
- `score_fn`: function (coords) → float (e.g. stripe_strength_score).

**Output**
- `dict` with `observed`, `null_distribution`, `p_value`,
  `effect_size_z`.

**Algorithm.**
Permute each gene independently across cells (preserves marginal
expression distribution; destroys cell-cell covariance). Re-PCA, project
to S², score. The p-value tests whether the observed spherical structure
is more than a horseshoe-of-randomness.

---

### 4.7 `topology.build_spherical_knn_graph`

**Inputs**
- `coords`: (N,3).
- `k`: int.

**Output**
- `scipy.sparse` (N,N) adjacency, edge weights = geodesic distance.

---

### 4.8 `topology.detect_branchpoints`

**Inputs**
- `coords`, `pseudotime` (N,), `k=15`.

**Output**
- (N,) angular-divergence score per cell; higher = more branchpoint-like.

**Algorithm.**
At each cell, compute geodesic gradient of pseudotime among its k
neighbours (4.3). Then compute the angular variance (1 − ‖ḡ‖ where
ḡ is the mean unit gradient vector among the cell's neighbours). High
angular variance ⇒ neighbours' downstream-direction vectors fan out ⇒
branchpoint. Calibrate against per-cell bootstrap to assign FDR.

---

### 4.9 `topology.linear_vs_branching_score`

**Inputs**
- `coords`, `pseudotime`.

**Output**
- `dict` with `linear_logL`, `branching_logL`, `bayes_factor`,
  `decision`.

**Algorithm.**
Fit two generative models:
- Linear: cells live near a single great circle (4.4) parameterised by
  pseudotime.
- Branching: a tree of geodesic segments (start with a single
  branchpoint detected via 4.8; branch directions = principal
  tangent-vectors of the two sub-populations).

Compare via BIC-corrected log-likelihood. Use to triage when in 2.5's
matrix the "linear-trajectory assumption" is unsafe.

---

### 4.10 `comparison.procrustes_align_spheres`

**Inputs**
- `coords_A`, `coords_B`: (Nₐ,3), (N_b,3).
- `labels_A`, `labels_B`: cell-type labels (lists of strings; common
  labels used as anchors).

**Output**
- 3×3 rotation `R`, residual per common label, total RMSE.

**Algorithm.**
Compute centroid (mean unit vector → renormalised) per common cell type
in each dataset; solve constrained orthogonal Procrustes (Kabsch,
det(R)=+1) on the matched centroids; report per-label residual angles.

---

### 4.11 `comparison.conserved_stripe_test`

**Inputs**
- Two pre-aligned spherical embeddings; stripe definitions (lists of
  longitudinal ranges).

**Output**
- per-stripe Jaccard of cell-type composition, permutation p-value.

---

### 4.12 `comparison.spherical_replicate_residual`

**Inputs**
- Two replicates of the same condition.

**Output**
- per-cell-type angular RMSE between rotated replicates; flags batch-effect-
  contaminated cell types.

---

### 4.13 `perturbation` module — fixed-PCA-loading sensitivity analysis

The `perturbation` module operationalises H7. **It is not a causal in-silico
knockout**: PCA loadings are held fixed, so the analysis reports how a
gene's *current* expression contributes to each cell's coordinates under
the existing axes, not what would happen if the gene were biologically
perturbed.

| Function | Inputs | Output |
|---|---|---|
| `project_expression_to_pca(X, components, mean=None)` | `(N, G)` matrix; `(k, G)` sklearn loadings; `(G,)` mean. | `(N, k)` PC scores. |
| `perturb_gene_expression(X, gene_names, gene, mode)` | matrix or DataFrame; gene names; gene; mode `"zero"` or `"double"`. | Perturbed matrix copy. |
| `compute_gene_perturbation_vectors(X, gene_names, genes, components, mean=None, normalize_to_sphere=False)` | matrix; gene names; iterable of genes; loadings; flag. | `dict[gene] -> {original, zeroed, doubled, delta_zero, delta_double, delta_double_minus_zero}`. |
| `rank_genes_by_perturbation_magnitude(results, metric)` | results dict; one of `symmetric_range / doubled / zeroed / asymmetry`. | DataFrame `[gene, score]` sorted descending. |
| `decompose_perturbation_vectors(coords, vectors)` | `(N,3)` positions; `(N,3)` displacements. | dict with `radial`, `tangent`, `theta_component`, `phi_component`, `tangent_magnitude`. |
| `identify_sensitive_cells(results, gene, top_n)` | results dict; gene; integer. | `(top_n,)` cell indices, sorted by sensitivity. |
| `plot_perturbation_vector_field(results, gene, frac, ...)` | results dict; gene; subsample fraction. | matplotlib Axes (equirectangular vector field). |
| `cluster_genes_by_perturbation_signature(results, threshold, mode)` | results dict; threshold; mode. | dict with `bitmap`, `genes`, `shared_counts` (gene-by-gene cell-overlap matrix). |

Hard constraints documented at the module top of [pca_sphere_projection/perturbation.py](pca_sphere_projection/perturbation.py):
- The asymmetry metric is identically zero under purely-linear PCA
  projection. It only carries information when `normalize_to_sphere=True`.
- Loading scale must be normalised before cross-gene ranking.
- Plots and rankings are sensitivity diagnostics; they are not knockouts.

---

## Phase 5 — Implementation

Reference implementations of the highest-priority new functions are now in
the package. They follow the existing style (DataFrame in / DataFrame or
dict out, NumPy + SciPy only):

- [pca_sphere_projection/sphere_stats.py](pca_sphere_projection/sphere_stats.py) — 4.1 to 4.6.
- [pca_sphere_projection/topology.py](pca_sphere_projection/topology.py) — 4.7 to 4.9.
- [pca_sphere_projection/comparison.py](pca_sphere_projection/comparison.py) — 4.10 to 4.12.
- [pca_sphere_projection/perturbation.py](pca_sphere_projection/perturbation.py) — 4.13.

Public API exposed via [`pca_sphere_projection.__init__`](pca_sphere_projection/__init__.py).

### Worked example

```python
import numpy as np, pandas as pd
import pca_sphere_projection as ps

df = pd.read_csv("examples/celegan_pca.csv")
root = df[df["cluster"] == "100-130"][["PC1","PC2","PC3"]].mean().values
df = ps.align_to_north_pole(df, root_node=root)

# NEW: replace manual Euler tuning with a great-circle fit.
coords = df[["PC1","PC2","PC3"]].values
gc = ps.fit_great_circle(coords)
print("great-circle plane normal:", gc["normal"], "R²:", gc["r_squared"])

# NEW: is the structure stripey at all?
print(ps.spherical_anisotropy(coords))
print(ps.stripe_strength_score(coords, n_bins=36))

# NEW: is this more structured than a HVG-shuffle null?
# (requires the HVG matrix; example skipped if not present)
# print(ps.horseshoe_null_test(X_hvg, n_perm=200))

# NEW: is it linear or branching?
pt = df["cluster"].map(ps.utils.cluster_to_time)  # user supplies
print(ps.linear_vs_branching_score(coords, pt.values))

# NEW (H7): gene perturbation vector field. Requires the HVG matrix and
# fitted PCA loadings (e.g. from sklearn.decomposition.PCA).
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3).fit(X_hvg)
# results = ps.compute_gene_perturbation_vectors(
#     X_hvg, gene_names, ["sox-2", "elt-2"], pca.components_, pca.mean_,
#     normalize_to_sphere=True,
# )
# rank = ps.rank_genes_by_perturbation_magnitude(results, metric="symmetric_range")
# decomp = ps.decompose_perturbation_vectors(
#     results["sox-2"]["original"], results["sox-2"]["doubled"] - results["sox-2"]["zeroed"],
# )
# print("phi-component magnitude (stripe-crossing):", np.abs(decomp["phi_component"]).mean())
```

---

## Phase 6 — Packaging

Existing layout already supports `pip install -e .`. Recommended additions:

```
pca_sphere_projection/
├── pyproject.toml          # NEW — modern build metadata
├── setup.py                # kept for editable installs
├── README.md
├── LICENSE
├── PNAS_EXTENSION_PROPOSAL.md
├── pca_sphere_projection/
│   ├── __init__.py         # exports old + new public API
│   ├── core.py             # unchanged
│   ├── sphere_stats.py     # NEW
│   ├── topology.py         # NEW
│   ├── comparison.py       # NEW
│   └── perturbation.py     # NEW
├── examples/
│   ├── celegan.ipynb, planaria.ipynb, ...
│   └── vector-plot.ipynb   # exploratory notebook refactored into perturbation.py
└── tests/
    ├── test_sphere_stats.py
    ├── test_topology.py
    ├── test_comparison.py
    └── test_perturbation.py   # NEW (16 tests, all passing)
```

`pyproject.toml` is added so the package builds under PEP-517 (`pip install
.` and `python -m build` both work without invoking `setup.py` directly).
The existing `setup.py` is kept for backward compatibility with editable
installs from the README's instructions.

Install:

```bash
pip install -e .            # development
pip install .               # release
python -m build && twine upload dist/*   # publish
```

---

## Phase 7 — Version Control Plan

Goal: keep the additions reviewable and isolated from `main` until the
manuscript is accepted.

```bash
# 1. Create the feature branch from current main
git checkout main
git pull
git checkout -b feature/pnas-extension

# 2. Logical commit groups (suggest one PR, multiple commits):
#    a) Proposal document only — DONE
git add PNAS_EXTENSION_PROPOSAL.md
git commit -m "docs: add PNAS extension proposal"

#    b) New stats module — DONE
git add pca_sphere_projection/sphere_stats.py
git commit -m "feat(sphere_stats): geodesic gradient, great-circle fit, anisotropy, null test"

#    c) Topology module — DONE
git add pca_sphere_projection/topology.py
git commit -m "feat(topology): k-NN graph, branchpoint detector, linear-vs-branching score"

#    d) Comparison module — DONE
git add pca_sphere_projection/comparison.py
git commit -m "feat(comparison): Procrustes alignment, conserved-stripe test, replicate residual"

#    e) Package wiring — DONE
git add pca_sphere_projection/__init__.py pyproject.toml
git commit -m "build: expose new API; add pyproject.toml"

#    f) Tests — DONE
git add tests/
git commit -m "test: unit tests for new modules"

# 3. Push and open PR (do not merge to main yet)
git push -u origin feature/pnas-extension

# 4. Tag a pre-release for the submission snapshot, on the branch
git tag -a v0.2.0-rc1 -m "PNAS submission snapshot"
git push origin v0.2.0-rc1
```

Branch hygiene rules for the submission window:

1. **Never rebase** `feature/pnas-extension` onto `main` after the
   submission snapshot is tagged — referees should be able to check out
   exactly the tagged commit.
2. Keep the existing `core.py` API stable; new functionality is purely
   additive. This ensures the existing README examples continue to work
   for reviewers who clone `main`.
3. Reviewer-suggested changes go on a child branch
   `feature/pnas-extension-r1` to preserve provenance:
   ```bash
   git checkout feature/pnas-extension
   git checkout -b feature/pnas-extension-r1
   ```
4. After acceptance, squash-merge the chain into `main` and tag `v0.2.0`.

---

## Constraints check

| Constraint | How this proposal honours it |
|---|---|
| Avoid vague biology | Each H1–H7 names a specific dataset, score, and falsification criterion. |
| Prioritise testability | Every hypothesis lists "Validation data" + "Pitfalls". |
| Don't assume horseshoe is universal | §2.1 explicitly treats it as a possibly-artefactual projection effect, and §4.6 provides a permutation null test. |
| Ground in what the transformation preserves/distorts | §2.2–§2.3 enumerate exactly what is lost (radial magnitude) and what is added (Jacobian distortion of equirectangular). H4 turns the discarded radial coordinate into a hypothesis; H7 turns the discarded perturbation direction into another. |

---

## Reassessment after implementation

This section reviews the package state *after* the additions in this branch
(`sphere_stats`, `topology`, `comparison`, `perturbation`) against the
weaknesses enumerated in §2.5. "Resolved" requires both production code
and a passing unit test (or a worked example). I am being deliberately
strict: a function that exists but has no validation test is at most
"Partially resolved".

| Previous weakness | Status | Evidence in code | Remaining caveat |
|---|---|---|---|
| Horseshoe artefact masquerading as biology (§2.1) | **Partially resolved** | `sphere_stats.horseshoe_null_test` permutes HVGs and re-PCAs; tested in [tests/test_sphere_stats.py](tests/test_sphere_stats.py). | Null permutes genes independently, which is a strong null. Real biology may have correlated noise; need a more conservative permutation that preserves gene-gene covariance modules. |
| Stripe pattern is just a re-projection of PC1 (§2.2) | **Partially resolved** | `sphere_stats.spherical_anisotropy` provides a quantitative `linear / planar / spherical` triplet to disqualify datasets where stripes do not exist. | Does not directly test "is this just PC1 in disguise?"; would need an information-theoretic comparison of `MI(label; lon)` vs. `MI(label; PC1)`. |
| Manual Euler tuning (§2.3) | **Resolved** | `sphere_stats.fit_great_circle` returns the plane normal; the Euler rotation that aligns this normal to (0,0,1) is determined automatically. Tested. | An end-to-end helper `align_to_great_circle(df) → df` would be a friendlier API; not yet provided. |
| Root-cluster bias (§2.5) | **Not resolved** | No code change. | A `sensitivity_to_root(df, candidate_roots)` sweep would directly address it; a single function call away. |
| PC count dependence (§2.5) | **Not resolved** | No code change. | An intrinsic-dimension estimator (TWO-NN, MLE, PCA elbow) plus a sweep over k would address it. Recommended for the next branch. |
| Lack of null models (§2.5) | **Resolved** | `horseshoe_null_test` provides one. Branchpoint detection has its own per-cell bootstrap (`detect_branchpoints(n_bootstrap=...)`). Conserved-stripe test uses label permutation. | Multiple-testing correction across stripes/genes is the user's responsibility; consider exposing a single FDR-aware wrapper. |
| Lack of branching/topology detection (§2.5) | **Partially resolved** | `topology.detect_branchpoints` and `topology.linear_vs_branching_score` exist and are tested. | Only one branchpoint per dataset is supported in `linear_vs_branching_score`; multi-branch trees are out of scope. The BIC penalty is a coarse model selector. |
| Cell-type rank ≠ pseudotime (§2.5) | **Not resolved in code** | The existing `visualize_lineage_correlation` still uses the user-supplied ordinal ranking; new modules accept arbitrary scalar fields (DPT, Palantir, scVelo) but no helper is provided to import them. | Add a thin `pseudotime_from_paga(adata)` / `pseudotime_from_dpt(adata)` adapter that returns a continuous (N,) scalar. |
| Lack of gene-level biological interpretation | **Partially resolved** | `find_and_visualize_correlated_genes` (existing) plus the new `perturbation` module provide two complementary views: per-stripe correlation and per-gene perturbation sensitivity. | Neither directly identifies *master regulators*; H5's "angular gradient orthogonal to local stripe direction" is specified but not yet implemented. |
| Lack of perturbation-style functional interpretation | **Resolved** | `perturbation` module implements zeroing, doubling, vector-field decomposition (radial / θ / φ), gene ranking, sensitive-cell identification, and gene clustering by shared sensitive cells. 16 unit tests in [tests/test_perturbation.py](tests/test_perturbation.py), all passing. | This is sensitivity analysis, **not causal**. The proposal text and module docstring make this explicit. CRISPR / RNAi validation is mandatory before any causal claim. |
| Equirectangular distortion biases θ-statistics (§2.3) | **Partially resolved** | `spherical_kde` (vMF, area-correct) replaces lon/lat histograms; `geodesic_gradient` and great-circle fits are intrinsic. | The legacy `equirectangular_projection` plotting still presents lon/lat coordinates; users may still over-interpret high-latitude clustering. A warning banner on the plot would help. |

---

## Updated PNAS-fit assessment

A blunt editorial review of the package as it stands.

**Is this still merely a visualisation method?** No, it is no longer just
that. Before this branch, every module either drew a figure or computed a
statistic on a hand-tuned coordinate. After the branch, the package
exposes (i) a null model for the entire spherical-structure claim
(`horseshoe_null_test`), (ii) automated alignment that removes the
hand-tuning (`fit_great_circle`), (iii) intrinsic geometric statistics
(`spherical_anisotropy`, `geodesic_gradient`), (iv) a branching-vs-linear
test, (v) cross-dataset comparison via Procrustes, and (vi) a
sensitivity-style functional interpretation per gene. That is a
diagnostic framework, not a viewer.

**Is it now a geometric diagnostic framework?** Yes — for *linear,
single-trajectory* developmental data. The framework still implicitly
assumes one dominant axis and linear PCA. Datasets with multi-modal
variance (heterogeneous tumour, immune atlas, compound treatment screens)
will not benefit from the spherical projection, and the framework should
say so explicitly via `spherical_anisotropy` before any biological
interpretation.

**Does the perturbation module strengthen the biological contribution?**
Yes, but only if the manuscript respects the no-causality constraint.
Used correctly, it converts a passive correlation-based gene list into a
ranked, geometrically-decomposable set of candidate regulators with
spatial annotations (which cells, which stripe direction). Used
incorrectly — i.e. treating "highest perturbation Frobenius norm" as a
list of master regulators — it will be punished in review. The H7 caveats
section is the firewall.

**Highest-risk weaknesses remaining before submission.**

1. **Root-cluster sensitivity**. The whole spherical alignment is anchored
   on one user-chosen root cluster; we have not shown the geometry is
   robust to that choice. *Fix:* sweep across plausible roots and report
   the distribution of stripe-strength scores.
2. **PC-count justification**. The package never tells the user *why*
   3 PCs. *Fix:* TWO-NN intrinsic dimension + scree elbow + a sweep that
   shows stripe-strength saturates around k = 3.
3. **Continuous pseudotime**. Several claims (H1, H2, H7) require a real
   continuous pseudotime. The code currently accepts one but provides no
   adapter. *Fix:* a 30-line `pseudotime_from_dpt(adata)` helper.
4. **No external benchmark**. Every example is qualitatively persuasive.
   *Fix:* run the pipeline blind on (a) a known linear trajectory
   (Klein iPSC), (b) a known branching one (planaria neoblast), and (c) a
   known *non-trajectory* dataset (e.g. a homeostatic adult atlas), and
   show that the framework's outputs (stripe strength, branching score,
   anisotropy) cleanly separate them. Without this, reviewers cannot
   tell when the method *should not* be applied.
5. **Cell-cycle confound**. `H4` predicts that ‖PC‖ encodes cycle phase;
   the existing pipeline silently discards it. Until the cycle confound
   is shown to be controlled, every "stripe = pseudotime" claim is
   confounded with cycle.
6. **Statistical claims rest on θ-correlations** that were largely
   guaranteed by construction (§2.4). Replace `visualize_lineage_correlation`'s
   Pearson r against ordinal ranks with a permutation test against a
   random root choice and against an isotropic spherical null.

**Mandatory analyses before claiming PNAS-level readiness.**

- Run `horseshoe_null_test` on every dataset in the manuscript and report
  p-values in the main text, not the supplement.
- Replace every figure that uses a hand-tuned Euler rotation with one
  produced by `fit_great_circle`, and report the residual `r_squared`.
- Run `linear_vs_branching_score` on every dataset; show the Bayes
  factor in the main figure.
- Run `spherical_anisotropy` on a control dataset (e.g. a homeostatic
  adult atlas with no developmental gradient) and show that the
  stripe-strength score is at chance, demonstrating the framework's
  specificity.
- For H7, validate at least three predicted regulators per dataset
  against published perturbation data (Wormbase RNAi, planaria knockdown
  atlas, CRISPRi-FlowFISH).
- Add the root-cluster sensitivity sweep, the PC-count sweep, and the
  cell-cycle ‖PC‖ correlation to the supplement.

**Verdict.** The package is now a credible methods paper *if* the
mandatory analyses above are completed. The geometric framework, the null
model, the branching-vs-linear test, and the perturbation sensitivity
analysis together exceed the original "visualisation tool" threshold and
are within reach of a strong methods-and-applications PNAS submission.
Without those analyses, it remains a sophisticated visualisation tool
with one falsifiable null — a stronger journal than methods-only outlets,
but not yet PNAS.

---

## Hypothesis execution plan using example CSV files

The end-to-end runner is
[scripts/run_pnas_hypothesis_suite.py](scripts/run_pnas_hypothesis_suite.py).
It reads [examples/example_configs.yaml](examples/example_configs.yaml)
and writes per-dataset outputs under
[outputs/hypothesis_suite/](outputs/hypothesis_suite/). Below is the
runnability matrix and the actual numbers obtained on the four
example CSVs (*C. elegans* embryo, planaria atlas, UC epithelium,
Klein iPSC) using the dataset-specific parameters extracted from the
notebooks.

| H | Runnable now? | Function / script | Output | Result on examples |
|---|---|---|---|---|
| H1 differentiation entropy | **No** from PC-only CSVs | `geodesic_gradient(coords, scalar_field)` accepts user scalar; no scalar in CSVs | placeholder | needs CytoTRACE / SCENT scores |
| H2 branching vs linear | **Yes** when pseudotime is configured | `linear_vs_branching_score`, `detect_branchpoints` via the suite script | `branchpoints_3d.html`, `metrics.json/branching` | celegan, planaria, uc_epi, klein all → "branching" with linear–branching BIC gaps of 3170 / 4180 / 4520 / 3781 (huge; confirms non-linearity but the test prefers branching even on a noisy great circle, so use as a *one-sided sanity check*, not a positive identification) |
| H3 cross-species conservation | **No** | `comparison.procrustes_align_spheres`; only one species per CSV | — | requires a paired second species/dataset |
| H4 radial component | **No** for these CSVs | `pc_coordinate_quality_summary`; the input CSVs are already L2-normalised (norm CV < 5%) | — | requires raw pre-normalisation PC scores upstream |
| H5 stripe-boundary genes | **No** from PC-only CSVs | `geodesic_gradient` with per-gene scalar; app surfaces a longitude-vs-latitude proxy when an expression CSV is uploaded | — | needs gene × cell expression matrix |
| H6 replicate residual | **No** | `comparison.spherical_replicate_residual` | — | needs a paired replicate |
| H7 perturbation vector field | **No** from PC-only CSVs | `compute_gene_perturbation_vectors` (refactored from `vector-plot.ipynb`) | — | needs expression matrix + sklearn PCA loadings |

**Geometric numbers actually produced** (from
[outputs/hypothesis_suite/](outputs/hypothesis_suite/)):

| Dataset | n cells | stripe-strength | aniso (lin/plan/sph) | GC R² | manual vs GC | Spearman θ ~ pseudotime | rot. robustness CV |
|---|---|---|---|---|---|---|---|
| celegan | 86,024 | 0.137 | 0.51 / 0.09 / 0.40 | 0.787 | 0.137 vs 0.099 (manual wins) | ρ=0.81 (n=86k) | 0.11 |
| planaria | 21,612 | 0.134 | 0.48 / 0.23 / 0.29 | 0.840 | 0.134 vs 0.129 (≈ tie) | ρ=0.70 (n=12.9k pooled; per-lineage 0.39–0.68) | 0.04 |
| uc_epi | 64,457 | 0.054 | 0.39 / 0.20 / 0.41 | 0.796 | 0.054 vs 0.047 (≈ tie) | ρ=−0.32 (negative; ordering goes from south pole) | 0.09 |
| klein | 2,717 | 0.143 | 0.49 / 0.42 / 0.10 | 0.941 | 0.143 vs 0.177 (**GC wins**) | ρ=0.71 | 0.11 |

What this **does and does not** tell us:

1. **Stripe-strength scores are uniformly low** (0.05–0.14 out of a
   theoretical maximum of 1). The visual stripes seen in the example
   notebooks are real but mild; cells are not concentrated into a few
   longitudinal bins, they merely have non-uniform longitude marginals.
2. **Klein iPSC has the highest planar anisotropy and the highest GC
   R²** (0.94), and is the only dataset where automated great-circle
   alignment beats the hand-tuned Euler. This is consistent with
   Klein being a near-linear day0→day7 trajectory.
3. **`uc_epi` produces a negative θ-vs-pseudotime correlation** under the
   pooled lineage ordering. The notebook's manually-chosen Euler rotation
   placed the Stem cells at a polar angle larger than mature lineages —
   which means the stripe direction in that figure is *inverted*
   relative to the convention used elsewhere. This was invisible until
   the hypothesis suite computed it numerically.
4. **`linear_vs_branching_score` returns "branching" on every dataset**,
   including klein which is functionally linear by construction. The
   branching-model BIC always wins because the BIC penalty for two extra
   plane normals is small relative to the residual reduction. The score
   is therefore a *poor positive test* and a *fine null test* — useful
   to flag genuine linearity, not to prove branching.
5. **Root sensitivity is small** (top-vs-bottom stripe-strength range ≤
   0.04 across plausible roots) on planaria, celegan, uc_epi, but
   `Day2` outscores the canonical `Day0` root on klein (0.18 vs 0.14).
   The "Day0 is the root" choice is therefore a notebook convention,
   not a data property.
6. **Rotation robustness is good**: CV across ±5° perturbations is
   0.04–0.11, so the stripe metric is not a fragile artefact of the
   Euler tuning.

---

## Updated PNAS-readiness assessment after example execution

I will not soften this verdict.

**Demonstrated, end-to-end:**
- The package now runs as a reproducible pipeline on four real datasets,
  with all parameters externalised to YAML.
- Every dataset gets metrics.json + interactive HTMLs + a per-dataset
  report.md that includes caveats automatically.
- A consolidated hypothesis status table is generated and cleanly marks
  each H1–H7 as runnable / not-runnable with reason.
- The Streamlit dashboard (`sphere-trace`) opens, accepts upload or
  example data, computes everything live, and refuses to draw conclusions
  it cannot support (warning banners on H4 / H5 / H7).
- All 33 unit tests pass.

**Demonstrated, but only as software:** the existence of the metrics.
The numerical results above show that "stripes" are mildly present
(stripe-strength 0.05–0.14, not 0.5+); that great-circle alignment
matches manual on three of four datasets and exceeds it on klein; that
ordinal-rank pseudotime ↔ θ correlations are 0.7–0.8 on the trajectory
datasets, but **uc_epi flips sign**, exposing a previously-uncorrected
orientation issue in the manual Euler choice. These are *diagnostic
findings about the existing pipeline*, not biological discoveries.

**Not demonstrated, and not demonstrable from the current CSVs:**
- H1 differentiation-entropy decay (needs CytoTRACE/SCENT).
- H3 cross-species conservation (needs a second species).
- H4 radial component (CSVs are pre-normalised; no radial information
  remains).
- H5 stripe-boundary regulators (needs gene × cell expression).
- H6 replicate residual (needs replicates).
- H7 perturbation vector field (needs expression + PCA loadings).

**What this means for PNAS submission.**

The current artefact is a credible **methods + diagnostics** paper. It
proves the existing visualisation pipeline against geometric and
robustness null models, exposes a real orientation bug in one of the
existing example notebooks (uc_epi), and provides a complete
reproducibility harness. That alone is publishable in a methods journal.

For PNAS specifically, the manuscript will need at least *one* dataset
where:

1. PC1–PC3 are accompanied by the gene × cell expression matrix and the
   sklearn PCA loadings (for H5 and H7);
2. an external pseudotime (DPT, Palantir, scVelo) replaces ordinal cell-
   type rank (so the θ-correlation is not partly self-fulfilling);
3. a CytoTRACE / SCENT scalar is computed on the same cells (for H1);
4. matched replicates exist (for H6);
5. for at least three of the predicted H5 / H7 genes, a published
   perturbation phenotype is available (for biological validation).

None of these requires changes to the package — the interfaces are
already in place. They require additional data. Until that data arrives,
the proper claim is "geometric diagnostic framework with a faithful
end-to-end demonstration on four published atlases", not "spherical
embedding reveals novel regulators of differentiation".

**Bottom line.** Closer to PNAS-ready than before; not there yet. The
package is the right shape; the data is the bottleneck. The honest
manuscript framing is the geometric framework + the orientation bug we
caught + a partial validation on one dataset for which the missing data
exists. Anything broader will be punished in review, and the proposal
text now embeds enough caveats that the framing should hold.

---

## Raw-expression validation workflow

**Status: implemented and run.** A new end-to-end pipeline starts from raw
cell × gene matrices (Matrix Market, .bz2 dense CSV per day, .rds via
`rdata`), re-derives PCA, fits the spherical embedding, and runs H1, H4,
H5, H7 — replacing the previous "PC1–PC3 CSV only" entry-point for
hypotheses that need gene-level data. See
[outputs/raw_expression_validation/final_summary.md](outputs/raw_expression_validation/final_summary.md)
for the full results writeup.

### Datasets tested

| Dataset | What was available | Hypotheses runnable |
|---|---|---|
| C. elegans (Packer) | Matrix Market counts, gene symbols, embryo-time bins, celltype, batch | H1 (proxy), H4, H5, H7 |
| UC epithelium (Smillie) | counts, cell barcodes, celltype, health/location/patient, gene symbols (1361-gene panel; **`uc_epi_gene.tsv` added 2026-05-05**) | H1 (proxy), H4 (now with cell-cycle / mito / ribo scores), H5, H7 (with real symbols) |
| Klein mESC | four per-day .bz2 CSVs, mouse gene symbols, day labels | H1 (proxy), H4, H5, H7 |
| hESC (CytoTRACE example RDS) | log-norm exprMatrix, **precomputed CytoTRACE rank**, phenotype | H1 (proper CytoTRACE), H4, H5, H7 |

### Hypothesis support, observed

| Hypothesis | Verdict on raw-expression data |
|---|---|
| **H1** entropy gradient | Supported in all four datasets in direction (negative ρ between entropy/stemness and geodesic distance from a stem anchor). Strongest, and methodologically cleanest, on hESC with **real CytoTRACE rank**: Spearman ρ = −0.637 (p ≪ 1e-100). UC epi's ρ = −0.886 is suspect (proxy + no symbols → likely capturing depth, not stemness). |
| **H4** radial vs angular | Supported — but the split is dataset-specific. hESC: cell-cycle vs radial ρ = −0.74 — strong. Klein: pseudotime/day vs radial ρ = +0.51. C. elegans: pseudotime/embryo-time vs θ ρ = +0.815 (the angular axis is the real developmental clock). |
| **H5** stripe-boundary genes | **Not supported by the coarse 18 × 36 grid score.** Top hits are housekeeping in Klein and uncharacterised genes in C. elegans. Recommended fix: per-cell geodesic-gradient field on the top-N H5 candidates instead of grid means. |
| **H7** fixed-loading perturbation | Strongest result on Klein: **Pou5f1 (Oct4)** ranks #1 with magnitude 0.91, > 4× the next gene; Sox2, Mixl1, Klf2, Dnmt3b round out the top 6. 10 of 30 candidate genes overlap the curated naive-pluripotency list. hESC's top 5 (HAPLN1, FGF12, GABRB3, JARID2, CDC20) include JARID2 (epigenetic, plausible) and CDC20 (cell-cycle); none are in the strict OCT4/SOX2/NANOG list. **UC epithelium (re-run 2026-05-05 with symbols)**: top 5 = CA2, **MUC2**, HMGN2, **PLA2G2A**, **OLFM4**; KRT20 and LYZ rank 6 and 9 — 6 of the top 11 are curated intestinal-epithelium regulators. Two independent recapitulations (Klein, UC epi). |

### Candidate genes identified

**Klein mESC (top 10 by H7 magnitude, fixed-loading sensitivity):**
Pou5f1, Sox2, Mixl1, Klf2, Nol11, Dnmt3b, Ttc9c, Mynn, Rn7s2, Rn7s1
— 4/5 of the top 5 are well-known pluripotency or primitive-streak
markers (Pou5f1/Oct4, Sox2, Mixl1, Klf2). This is the cleanest
hypothesis-supporting result of the workflow.

**hESC (top 5 by H7 magnitude):**
HAPLN1, FGF12, GABRB3, JARID2, CDC20 — JARID2 and CDC20 are biologically
plausible pluripotency-context hits but are not in the strict curated
list, so they are reported as the analysis's *predictions* rather than
literature recapitulations.

**C. elegans (top 5):** oac-51, C49F8.3, Y32F6A.5, osm-11, F22F4.9 — all
reasonably-expressed but none are canonical embryogenesis regulators
in the conservative WormBase list. These are nominations, not
recapitulations.

**UC epithelium (top 11 by H7 magnitude, after symbol annotation 2026-05-05):**
CA2, MUC2, HMGN2, PLA2G2A, OLFM4, KRT20, SFN, AOC1, LYZ, TGOLN2, SPDEF —
six of these (MUC2, PLA2G2A, OLFM4, KRT20, LYZ, SPDEF) are in the
curated intestinal-epithelium regulator set, spanning goblet, Paneth,
stem, enterocyte, and secretory-lineage TF programs. This is the
second independent dataset on which H7 recapitulates known regulators
(Klein being the first), so the gene-prioritisation tool's behaviour
is consistent across systems and species.

### Figures produced

For each dataset under
`outputs/raw_expression_validation/<dataset>/figures/`:

- `H1_entropy_gradient.png` — entropy on the 3D sphere, equirectangular
  projection, and entropy-vs-θ scatter.
- `H4_radial_vs_angular.png` — radial vs θ scatter and a Spearman-ρ bar
  chart of every variable against both axes.
- `H5_gene_gradient.png` — top-20 stripe-boundary and along-trajectory
  genes; known regulators are highlighted in red.
- `H7_perturbation_vector_field.png` — equirectangular vector field for
  the top 5 H7 candidate genes per dataset (blue = zeroed, red =
  doubled).
- A combined `outputs/raw_expression_validation/summary_heatmap.png`
  shows the |ρ| of every H1 / H4 metric across datasets at a glance.

### Caveats (carried over from the run)

- H7 vector fields are fixed-PCA-loading sensitivities, not in-silico
  knockouts. They tell you which genes most align with the spherical
  axes given the current loadings; they do not predict perturbation
  phenotypes.
- UC epi has no gene symbols, so its H5/H7 rankings are not biologically
  interpretable.
- Three of the four H1 results use a **Shannon-entropy proxy**, not
  CytoTRACE or SCENT. The hESC RDS dataset is the only one that ships
  with a real CytoTRACE rank, and that is the dataset where H1's
  geometric claim is properly testable.
- C. elegans and UC epi were run on a 20k random subsample because of
  the (cells × HVGs) PCA cost. Re-running with `--max-cells 0` would
  use all cells; we provide the option but did not commit it as the
  default.
- The H5 grid score is too coarse to localise switch-like regulators;
  the raw-expression infrastructure now in place makes the next iteration
  (per-cell geodesic gradient on top-N candidates) a one-script change.

### What's still missing for PNAS-level claims

The workflow now answers what was previously blocked by "no gene-level
input". What still blocks the paper is:

1. CRISPR/RNAi follow-up on at least one nominated H7 gene that is
   *not* already a canonical regulator (e.g. JARID2 in hESC).
2. Real CytoTRACE/SCENT runs on celegan and UC epi (would also resolve
   the depth-vs-biology ambiguity in UC epi's H1 result).
3. Gene-symbol annotation of the UC epi matrix.
4. Matched replicates per system (for H6).
5. A continuous DPT/Palantir pseudotime per dataset to replace the
   ordinal day/embryo-time bins.

The package now provides every interface those analyses need; the
remaining blockers are data, not code.
