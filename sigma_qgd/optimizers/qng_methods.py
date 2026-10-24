"""
Quantum Natural Gradient (QNG) methods: Diag-QNG and QN-SPSA.
"""

from __future__ import annotations

import csv
import os
import time
import numpy as np

try:
    from .base import BaseOptimizer
    from ..core.config import Result
except ImportError:
    raise ImportError("Failed to import BaseOptimizer or core modules.")


class DiagQNG(BaseOptimizer):
    """
    Diagonal Quantum Natural Gradient (Stokes et al., Quantum 4, 269, 2020).

    Computes diagonal Fubini-Study metric elements:
        F_ii = 0.25 * (1 - |<ψ(θ)|ψ(θ + π e_i)>|^2)
    """

    supports_lr = False

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        lr: float = 0.05,
        lambda_reg: float = 0.001,
        log_circuit_breakdown: bool = False,
        results_dir: str = ".",
        tag: str = "",
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

        n_total = 3 * p + 2

        if log_circuit_breakdown:
            suffix = f"_{tag}" if tag else ""
            bpath = os.path.join(results_dir, f"diag_qng_circuit_breakdown{suffix}.csv")
            write_header = not os.path.exists(bpath)
            with open(bpath, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "p",
                        "cost_circuits",
                        "grad_circuits",
                        "metric_tensor_circuits",
                        "total_circuits_per_step",
                    ])
                w.writerow([p, 1, 2 * p, p + 1, n_total])

        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="Diag-QNG")

        for t in range(1, max_steps + 1):
            cost = cost_fn(theta)
            g = cost_fn.gradient(theta)
            gnorm = float(np.linalg.norm(g))
            sv_base = cost_fn.statevector(theta)

            F_diag = np.ones(p)
            for i in range(p):
                theta_shift = theta.copy()
                theta_shift[i] += np.pi
                overlap = cost_fn.overlap_call(sv_base, theta_shift)
                F_diag[i] = 0.25 * max(1.0 - overlap, 1e-8)

            d = g / (F_diag + lambda_reg)
            theta -= lr * d

            eh.append(cost)
            gh.append(gnorm)
            ch.append(n_total)

            if cost < best_e:
                best_e = cost
                best_theta = theta.copy()

            rep.update(t, cost, theta, gnorm=gnorm, circuits=cost_fn.n_calls)
            if gnorm < 1e-6:
                break

        rep.finish(final_energy=best_e, final_theta=best_theta, circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "Diag-QNG",
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


class QNSPSA(BaseOptimizer):
    """
    Quantum Natural SPSA (QN-SPSA).
    References: Gacon et al. (Quantum 5, 567, 2021); Spall (1998, 2000).
    """

    supports_lr = False
    N_CIRCUITS = 8

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        lr: float = 0.05,
        lambda_reg: float = 0.001,
        eps_metric: float = 0.01,
        mode: str = "full",
        blocking: bool = True,
        **kwargs,
    ) -> tuple:
        theta = theta_init.copy()
        p = len(theta)
        a = lr
        c = 0.1
        A = max_steps / 10.0
        alpha_sp = 0.602
        gamma_sp = 0.101

        if mode == "full":
            F_bar = np.eye(p, dtype=float)
        else:
            F_bar = np.ones(p, dtype=float)

        best_e = np.inf
        best_theta = theta.copy()
        eh = []
        gh = []
        ch = []
        t0 = time.time()
        rep = self._create_progress_reporter(cost_fn, max_steps, kwargs, default_desc="QN-SPSA")

        for k in range(1, max_steps + 1):
            ak = a / (A + k) ** alpha_sp
            ck = c / (k ** gamma_sp)

            cost = cost_fn(theta)

            # SPSA gradient estimation
            delta1 = np.random.choice([-1.0, 1.0], size=p)
            fp = cost_fn(theta + ck * delta1)
            fm = cost_fn(theta - ck * delta1)
            g_sp = (fp - fm) / (2.0 * ck * delta1)

            # Fidelity-based metric estimation
            sv_base = cost_fn.statevector(theta)
            m1 = np.random.choice([-1.0, 1.0], size=p)
            m2 = np.random.choice([-1.0, 1.0], size=p)
            e = eps_metric

            def fid(th):
                return cost_fn.overlap_call(sv_base, th)

            dF = (
                fid(theta + e * m1 + e * m2)
                - fid(theta + e * m1)
                - fid(theta - e * m1 + e * m2)
                + fid(theta - e * m1)
            )

            if mode == "full":
                F_point = -dF / (8.0 * (e ** 2)) * 0.5 * (np.outer(m1, m2) + np.outer(m2, m1))
                F_bar = (k / (k + 1.0)) * F_bar + (1.0 / (k + 1.0)) * F_point

                vals, vecs = np.linalg.eigh(F_bar)
                F_psd = vecs @ np.diag(np.maximum(np.abs(vals), 1e-8)) @ vecs.T
                F_reg = (F_psd + lambda_reg * np.eye(p)) / (1.0 + lambda_reg)

                try:
                    d = np.linalg.solve(F_reg, g_sp)
                except np.linalg.LinAlgError:
                    d = g_sp / (np.diag(F_psd) + lambda_reg)
            else:
                F_est = -dF / (8.0 * (e ** 2)) * m1 * m2
                F_bar = (k / (k + 1.0)) * F_bar + (1.0 / (k + 1.0)) * F_est
                F_diag = np.maximum(np.abs(F_bar), 1e-4)
                d = g_sp / (F_diag + lambda_reg)

            # Gradient clipping to prevent instability
            d_norm = np.linalg.norm(d)
            clip_bound = np.sqrt(p) * 10.0
            if d_norm > clip_bound:
                d = d * (clip_bound / d_norm)

            theta_cand = theta - ak * d

            # Step-blocking rule
            if blocking:
                cand_cost = cost_fn(theta_cand)
                tol = 0.1 / np.sqrt(k)
                if cand_cost <= cost + tol:
                    theta = theta_cand
                    cost = cand_cost
            else:
                theta = theta_cand

            gnorm = float(np.linalg.norm(g_sp))
            eh.append(cost)
            gh.append(gnorm)
            ch.append(self.N_CIRCUITS)

            if cost < best_e:
                best_e = cost
                best_theta = theta.copy()

            rep.update(k, cost, theta, gnorm=gnorm, circuits=cost_fn.n_calls)

        rep.finish(final_energy=best_e, final_theta=best_theta, circuits=cost_fn.n_calls, steps=len(eh))
        result = Result(
            "QN-SPSA",
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
