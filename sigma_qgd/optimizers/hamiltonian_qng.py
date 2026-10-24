"""
Hamiltonian-aware Quantum Natural Gradient Descent (H-QNG).

Reference:
    Shi & Wang (2025). Efficient Hamiltonian-aware Quantum Natural
    Gradient Descent for Variational Quantum Eigensolvers. arXiv:2511.14511.
"""

from __future__ import annotations

import time
import numpy as np

try:
    from .base import BaseOptimizer
    from ..core.config import Result
    from ..style import DEFAULT_LR
except ImportError:
    raise ImportError("Failed to import BaseOptimizer or core modules.")


class HamiltonianQNG(BaseOptimizer):
    """
    Diagonal H-QNG: Hamiltonian-induced pullback metric preconditioning.

    Metric:
        T_ii = (∂_i ⟨H⟩)^2 / (2 * ‖a‖ + λ)
    Update:
        θ_{t+1} = θ_t - lr * g / (T_diag + λ)

    Reference: Shi & Wang (2025), Eq. 8 (diagonal approximation).
    """

    supports_lr = True
    default_lr = 0.001

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        lr: float = 0.001,
        lambda_reg: float = 0.001,
        **kwargs,
    ) -> tuple:
        theta = theta_init.copy()
        p = len(theta)

        try:
            coeffs = np.array([float(c) for c in cost_fn.H.coeffs])
            a_norm = float(np.sqrt(np.sum(coeffs ** 2)))
        except Exception:
            a_norm = 1.0

        scale = 2.0 * a_norm + lambda_reg

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="Hess-Mag")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)

            T_diag = (g ** 2) / scale
            d = g / (T_diag + lambda_reg)

            theta -= lr * d
            gnorm = float(np.linalg.norm(g))
            eh.append(cost)
            gh.append(gnorm)
            ch.append(2 * p + 1)

            if cost < best_e:
                best_e = cost
                best_theta = theta.copy()

            rep.update(t, cost, theta, gnorm=gnorm, circuits=cost_fn.n_calls)
            if gnorm < 1e-6:
                break

        rep.finish(final_energy=best_e, final_theta=best_theta, circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "Hess-Mag",
            best_e,
            eh,
            gh,
            cost_fn.n_calls,
            len(eh),
            circuits_hist=ch,
            wall_time=time.time() - t0,
            best_theta=best_theta.copy(),
        )
        return result, [], best_theta
