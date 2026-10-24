"""
Adam-family gradient-based optimizers: Adam, AdaBelief, RMSProp, Yogi, and SRAdam.
"""

from __future__ import annotations

import time
import numpy as np

try:
    from .base import BaseOptimizer
    from ..core.config import Result
    from ..style import DEFAULT_LR
except ImportError:
    raise RuntimeError("Failed to import BaseOptimizer or core modules.")


class Adam(BaseOptimizer):
    """
    Adam optimizer.
    Reference: Kingma & Ba (ICLR 2015).
    """

    supports_lr = True
    default_lr = DEFAULT_LR

    def run(self, cost_fn, theta_init, max_steps: int = 200, lr: float = DEFAULT_LR, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        m = np.zeros(p)
        v = np.zeros(p)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="Adam")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g ** 2)
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            theta -= lr * mh / (np.sqrt(vh) + eps)

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
            "Adam",
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


class AdaBelief(BaseOptimizer):
    """
    AdaBelief optimizer.
    Reference: Zhuang et al. (NeurIPS 2020).
    """

    supports_lr = True
    default_lr = DEFAULT_LR

    def run(self, cost_fn, theta_init, max_steps: int = 200, lr: float = DEFAULT_LR, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        m = np.zeros(p)
        s = np.zeros(p)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="AdaBelief")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            m = b1 * m + (1.0 - b1) * g
            s = b2 * s + (1.0 - b2) * ((g - m) ** 2) + eps
            mh = m / (1.0 - b1 ** t)
            sh = s / (1.0 - b2 ** t)
            theta -= lr * mh / (np.sqrt(sh) + eps)

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
            "AdaBelief",
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


class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer.
    Reference: Tieleman & Hinton (2012).
    """

    supports_lr = True
    default_lr = 0.01

    def run(self, cost_fn, theta_init, max_steps: int = 200, lr: float = 0.01, decay: float = 0.9, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        eps = 1e-8
        v = np.zeros(p)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="RMSProp")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            v = decay * v + (1.0 - decay) * (g ** 2)
            theta -= lr * g / (np.sqrt(v) + eps)

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
            "RMSProp",
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


class Yogi(BaseOptimizer):
    """
    Yogi optimizer.
    Reference: Zaheer et al. (NeurIPS 2018).
    """

    supports_lr = True
    default_lr = DEFAULT_LR

    def run(self, cost_fn, theta_init, max_steps: int = 200, lr: float = DEFAULT_LR, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-7
        m = np.zeros(p)
        v = np.full(p, 1e-6)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="Yogi")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            m = b1 * m + (1.0 - b1) * g
            g2 = g ** 2
            v = v - (1.0 - b2) * np.sign(v - g2) * g2
            mh = m / (1.0 - b1 ** t)
            vh = v / (1.0 - b2 ** t)
            theta -= lr * mh / (np.sqrt(np.abs(vh)) + eps)

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
            "Yogi",
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


class SRAdam(BaseOptimizer):
    """
    AdaBelief-SR optimizer: Adam momentum with AdaBelief variance and linear regularization.
    References: Sorella (1998); Zhuang et al. (2020).
    """

    supports_lr = True
    default_lr = DEFAULT_LR

    def run(self, cost_fn, theta_init, max_steps: int = 200, lr: float = DEFAULT_LR, lambda_reg: float = 0.001, **kwargs) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        m = np.zeros(p)
        s = np.zeros(p)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="SRAdam")

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            m = b1 * m + (1.0 - b1) * g
            s = b2 * s + (1.0 - b2) * ((g - m) ** 2)
            mhat = m / (1.0 - b1 ** t)
            shat = s / (1.0 - b2 ** t)
            theta -= lr * mhat / (shat + lambda_reg + eps)

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
            "SRAdam",
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
