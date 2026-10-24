"""
Derivative-free optimizers: COBYLA, SPSA, and CMA-ES.
"""

from __future__ import annotations

import time
from typing import Optional
import numpy as np
from scipy.optimize import minimize

try:
    from .base import BaseOptimizer
    from ..core.config import Result
except ImportError:
    raise ImportError("Failed to import base or core modules.")

try:
    import cma as _cma_pkg
    _CMA_AVAILABLE = True
except ImportError:
    _CMA_AVAILABLE = False


class COBYLA(BaseOptimizer):
    """COBYLA derivative-free optimizer."""

    supports_lr = False

    def run(self, cost_fn, theta_init, max_steps: int = 200, **kwargs) -> tuple:
        best_e = [np.inf]
        best_x = [theta_init.copy()]
        eh = []
        gh = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="COBYLA")
        call_count = [0]

        def obj(x):
            cost = cost_fn(x)
            call_count[0] += 1
            eh.append(cost)
            gh.append(0.0)
            if cost < best_e[0]:
                best_e[0] = cost
                best_x[0] = np.array(x, copy=True)
            rep.update(call_count[0], cost, x, circuits=cost_fn.n_calls)
            return cost

        minimize(
            obj,
            theta_init.copy(),
            method="COBYLA",
            options={"maxiter": max_steps, "rhobeg": 0.1},
        )
        rep.finish(final_energy=best_e[0], final_theta=best_x[0], circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "COBYLA",
            best_e[0],
            eh,
            gh,
            cost_fn.n_calls,
            len(eh),
            circuits_hist=[1] * len(eh),
            wall_time=time.time() - t0,
            best_theta=best_x[0].copy(),
        )
        return result, [], best_x[0]


class SPSA(BaseOptimizer):
    """Simultaneous Perturbation Stochastic Approximation (SPSA)."""

    supports_lr = False

    def run(self, cost_fn, theta_init, max_steps: int = 200, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        a = 0.1
        c = 0.1
        A = max_steps / 10.0
        alpha = 0.602
        gamma = 0.101

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="SPSA")

        for k in range(1, max_steps + 1):
            ak = a / (A + k) ** alpha
            ck = c / k ** gamma
            delta = np.random.choice([-1.0, 1.0], size=p)
            fp = cost_fn(theta + ck * delta)
            fm = cost_fn(theta - ck * delta)
            g_sp = (fp - fm) / (2.0 * ck * delta)
            theta -= ak * g_sp
            cost = cost_fn(theta)
            gnorm = float(np.linalg.norm(g_sp))
            eh.append(cost)
            gh.append(gnorm)
            ch.append(3)
            if cost < best_e:
                best_e = cost
                best_theta = theta.copy()
            rep.update(k, cost, theta, gnorm=gnorm, circuits=cost_fn.n_calls)

        rep.finish(final_energy=best_e, final_theta=best_theta, circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "SPSA",
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


class CMAES(BaseOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    Uses budget-matched evaluation termination.

    Reference: Hansen & Ostermeier (2001).
    """

    supports_lr = False

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        total_circuit_budget: Optional[int] = None,
        sigma0: float = 0.3,
        seed: int = 0,
        popsize: Optional[int] = None,
        **kwargs,
    ) -> tuple:
        if not _CMA_AVAILABLE:
            raise ImportError("cma package is required to run CMA-ES.")

        p = len(theta_init)
        if total_circuit_budget is None:
            total_circuit_budget = (2 * p + 1) * max_steps

        t0 = time.time()
        opts = {"seed": seed, "verbose": -9}
        if popsize is not None:
            opts["popsize"] = popsize
        es = _cma_pkg.CMAEvolutionStrategy(theta_init.tolist(), sigma0, opts)

        best_e = np.inf
        best_theta = theta_init.copy()
        eh = []
        gh = []
        ch = []
        start_calls = cost_fn.n_calls
        rep = self._create_progress_reporter(
            cost_fn, total_circuit_budget, kwargs, default_desc="CMA-ES"
        )

        while (cost_fn.n_calls - start_calls) < total_circuit_budget and not es.stop():
            solutions = es.ask()
            fitnesses = []
            budget_hit = False

            for x in solutions:
                if (cost_fn.n_calls - start_calls) >= total_circuit_budget:
                    budget_hit = True
                    break
                f = cost_fn(np.array(x))
                fitnesses.append(f)
                ch.append(1)
                eh.append(f)
                gh.append(0.0)
                if f < best_e:
                    best_e = f
                    best_theta = np.array(x, copy=True)
                rep.update(cost_fn.n_calls - start_calls, best_e, best_theta, circuits=cost_fn.n_calls)

            if budget_hit:
                break

            es.tell(solutions, fitnesses)

        rep.finish(final_energy=best_e, final_theta=best_theta, circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "CMA-ES",
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
