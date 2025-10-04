"""JAX-based iLQR (iterative Linear Quadratic Regulator) solver."""

from .solvers import LQR, iLQR, iLQRAdaptive, iLQRAdaptiveAugmented

__all__ = ["LQR", "iLQR", "iLQRAdaptive", "iLQRAdaptiveAugmented"]
__version__ = "0.1.0"
