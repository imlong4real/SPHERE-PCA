"""Public SPHERE-PCA interface."""

from importlib.metadata import PackageNotFoundError, version

from .core import fit, transform
from .plotting import plot
from .result import SpherePCAResult

try:
    __version__ = version("sphere-pca")
except PackageNotFoundError:  # source checkout without an installed distribution
    __version__ = "0+unknown"

__all__ = ["SpherePCAResult", "fit", "plot", "transform"]
