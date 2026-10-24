"""
General utility functions for parameter initialization and reproduciblity.
"""

from __future__ import annotations

import numpy as np

_THETA_MIN: float = -np.pi / 8
_THETA_MAX: float = np.pi / 8


def get_initial_theta(n_params: int, seed: int) -> np.ndarray:
    """
    Generate a deterministic initial parameter vector for a given seed.

    Parameters
    ----------
    n_params : int
        Dimension of the parameter vector.
    seed : int
        RNG seed.

    Returns
    -------
    theta0 : np.ndarray
        Initial parameters sampled uniformly from [-π/8, π/8].
    """
    np.random.seed(seed)
    return np.random.uniform(_THETA_MIN, _THETA_MAX, n_params)
