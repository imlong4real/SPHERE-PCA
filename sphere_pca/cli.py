"""Lightweight console entry points for optional SPHERE-PCA tools."""

from __future__ import annotations

from importlib.util import find_spec
import sys


_APP_MODULES = {
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "plotly": "plotly",
    "yaml": "pyyaml",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "statsmodels": "statsmodels",
    "streamlit": "streamlit",
}


def _missing_app_dependencies() -> list[str]:
    return [
        distribution
        for module, distribution in _APP_MODULES.items()
        if find_spec(module) is None
    ]


def main() -> None:
    """Launch the optional dashboard or explain how to install it."""

    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        print(
            "usage: sphere-trace [STREAMLIT_OPTIONS]\n\n"
            "Launch the optional SPHERE-PCA dashboard. Install dashboard "
            "dependencies with `python -m pip install 'sphere-pca[app]'`."
        )
        return

    missing = _missing_app_dependencies()
    if missing:
        names = ", ".join(sorted(missing))
        raise SystemExit(
            "sphere-trace requires the optional dashboard dependencies "
            f"(missing: {names}). Install them with "
            "`python -m pip install 'sphere-pca[app]'`."
        )

    from pca_sphere_projection.app import main as app_main

    app_main()
