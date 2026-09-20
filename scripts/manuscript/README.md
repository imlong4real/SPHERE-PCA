# Manuscript reproduction

This directory contains the entry points for reproducing the bioRxiv figures.
The implementation remains in the legacy `pca_sphere_projection.figures`
namespace so the published numerical workflows and imports continue to work.

## Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-manuscript.txt
python -m pip install -e . --no-deps
```

`requirements-manuscript.txt` records the dependency bounds used by the
public manuscript workflow. Exact enrichment results also depend on the
Enrichr library snapshot noted in the preprint (May 11, 2026).

## Inputs

Large datasets are not shipped in the Python package. Follow the preprint's
data-availability statement, then reproduce the repository-relative input
layout under `raw_data/`. The figure code expects dataset subdirectories such
as `benchmark/`, `celegan/`, `hESC/`, `human_germ_cell/`, `klein/`,
`planaria/`, `pre_implant_human_embryo/`, `uc_epi/`, `xenium/`, and
`BrCa_atlas/`. Some Figure 1 panels can use precomputed `examples/*_pca.csv`
files when present.

## Commands

```bash
# Main figures
python scripts/manuscript/make_manuscript_figures.py

# One figure or panel
python scripts/manuscript/make_manuscript_figures.py --figure 3
python scripts/manuscript/make_manuscript_figures.py --figure 3 --panel A_hesc

# Supplemental analyses
python scripts/manuscript/make_supplement_figures.py

# Small end-to-end check when examples/celegan_pca.csv is available
python scripts/manuscript/smoke_test_celegan.py
```

Generated figures, resolved YAML configurations, and underlying CSV tables
are written below `outputs/figures/`. Both `raw_data/` and `outputs/` are
ignored so local manuscript data and generated artifacts cannot accidentally
enter the package or a source release.
