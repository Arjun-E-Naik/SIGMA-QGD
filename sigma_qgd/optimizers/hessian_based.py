"""
Diagonal Hessian-based optimizer.

Reuses parameter-shift evaluations to obtain the diagonal Hessian elements.
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


class DiagHessian(BaseOptimizer):
    """
    Diagonal Newton step with Levenberg-Marquardt positivity safeguard.

    F_i = h_ii + λ   if h_ii > 0
        = λ          otherwise

    References: Levenberg (1944); Marquardt (1963); Wierichs et al. (2022).
    """

    supports_lr = True
    default_lr = DEFAULT_LR

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        lr: float = DEFAULT_LR,
        lambda_reg: float = 0.001,
        **kwargs,
    ) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="Diag-Hessian")

        for t in range(1, max_steps + 1):
            cost, g, hess_diag = cost_fn.cost_gradient_and_diag_hessian(theta)

            F_diag = np.where(hess_diag > 0.0, hess_diag + lambda_reg, lambda_reg)
            d = g / F_diag
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
            "Diag-Hessian",
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
