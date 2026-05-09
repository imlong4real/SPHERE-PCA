"""Supplemental figures for the SPHERE-PCA manuscript.

These modules formalise exploratory notebook analyses
(`examples/benchmark.ipynb`, `examples/planaria.ipynb`, Xenium notebook)
into reproducible publication-quality outputs.

The orchestrator is `scripts/make_supplement_figures.py`; each module
exposes a `run(out_root)` function that produces PNG + sidecar config +
CSV tables under `outputs/figures/supplement/<topic>/`.
"""

from . import common  # noqa: F401
from . import benchmark  # noqa: F401
from . import planaria   # noqa: F401
from . import xenium     # noqa: F401
