"""
SIGMA-QGD optimizer: variance-aware diagonal preconditioning for variational quantum circuits.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional

import numpy as np

from .base import BaseOptimizer
from ..core.config import derive_config, Result, StepRecord
from ..core.logging import ParameterLogger


class VarianceEngine:
    """
    Online gradient variance estimator using Welford's algorithm,
    with James-Stein shrinkage and exponential moving average smoothing.
    """

    def __init__(
        self,
        p: int,
        W: int,
        eps: float = 1e-10,
        variance_window: Optional[int] = None,
    ):
        self.p = p
        self.eps = eps
        self.alpha = 2.0 / (W + 1)
        self.alpha_var = (2.0 / (variance_window + 1)) if variance_window else None
        self.ewm_mean = np.zeros(p)
        self.t = 0
        self.M = np.zeros(p)
        self.S = np.zeros(p)
        self.welford_raw = np.zeros(p)
        self.js_shrunk = np.full(p, 1e-4)
        self.ema = np.zeros(p)
        self.ema_mean = np.zeros(p)
        self.mean = np.zeros(p)

    def update(self, g: np.ndarray) -> None:
        self.t += 1
        d1 = g - self.M
        self.M += d1 / self.t
        d2 = g - self.M
        self.S += d1 * d2
        self.mean = self.M.copy()

        if self.t < 2:
            return

        if self.alpha_var is not None:
            d1w = g - self.ewm_mean
            self.ewm_mean = self.ewm_mean + self.alpha_var * d1w
            d2w = g - self.ewm_mean
            if self.t == 2:
                self.welford_raw = np.maximum(d1w * d2w, self.eps)
            else:
                self.welford_raw = np.maximum(
                    (1.0 - self.alpha_var) * self.welford_raw + self.alpha_var * d1w * d2w,
                    self.eps,
                )
        else:
            self.welford_raw = self.S / (self.t - 1)

        sigma_bar = float(np.mean(self.welford_raw)) + self.eps
        deviations = self.welford_raw - sigma_bar
        Q = float(np.sum(deviations ** 2)) + self.eps
        shrink = max(0.0, 1.0 - (self.p - 2) * (sigma_bar ** 2) / (Q * self.t))
        self.js_shrunk = np.maximum(sigma_bar + shrink * deviations, 1e-8)

        self.ema_mean = (1.0 - self.alpha) * self.ema_mean + self.alpha * g
        diff_ema = g - self.ema_mean
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * (diff_ema ** 2)


class DriftCorrectedVarianceEngine(VarianceEngine):
    """Variance engine with parameter drift compensation."""

    def __init__(
        self,
        p: int,
        W: int,
        eps: float = 1e-10,
        drift_alpha: float = 0.1,
        drift_floor: float = 1e-10,
        variance_window: Optional[int] = None,
    ):
        super().__init__(p, W, eps, variance_window=variance_window)
        self.drift_alpha = drift_alpha
        self.drift_floor = drift_floor
        self.delta_ema = np.zeros(p)
        self.prev_theta = None
        self.welford_corrected = np.zeros(p)

    def update_with_theta(self, g: np.ndarray, theta: np.ndarray) -> None:
        self.update(g)
        if self.prev_theta is not None:
            delta = theta - self.prev_theta
            self.delta_ema = (1.0 - self.drift_alpha) * self.delta_ema + self.drift_alpha * (delta ** 2)
            drift_var = self.delta_ema * self.welford_raw
            self.welford_corrected = np.maximum(self.welford_raw - drift_var, self.drift_floor)
        else:
            self.welford_corrected = self.welford_raw.copy()
        self.prev_theta = theta.copy()

    def update(self, g: np.ndarray) -> None:
        super().update(g)
        self.welford_corrected = self.welford_raw.copy()


class CurvatureProxy:
    """
    Diagonal curvature proxy preconditioner:
        F_diag = c * sigma_hat^2 + lambda_reg
    """

    def __init__(
        self,
        eta_0: float,
        lambda_reg: float,
        eps: float = 1e-8,
        multiplier: float = 1.0,
    ):
        self.eta_0 = eta_0
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.multiplier = multiplier

    def precondition(self, welford_raw: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        F_diag = self.multiplier * welford_raw + self.lambda_reg
        return gradient / (F_diag + self.eps)

    def gamma(self, welford_raw: np.ndarray) -> float:
        F_mean = float(self.multiplier * np.mean(welford_raw) + self.lambda_reg)
        gamma_floor = 2.0 * self.eta_0
        return float(max(self.eta_0 / (F_mean + self.eps), gamma_floor))


class LandscapeClassifier:
    """Classifies trajectory steps into active, stagnant, or converged phases."""

    def __init__(self, cfg: dict, eta_0: float = 0.05):
        self.stag_win = cfg["stag_win"]
        self.no_imp_lim = cfg["no_imp_lim"]
        self.p = cfg["p"]
        self.gnorm_abs_tol = float(np.sqrt(self.p)) * 0.20
        self.energy_buf = deque(maxlen=self.stag_win)
        self.best_energy = np.inf
        self.no_imp = 0
        self.phase_history: List[str] = []
        self.n_stagnant = 0
        self.n_active = 0

    def classify(self, energy: float, gnorm: float) -> str:
        self.energy_buf.append(energy)
        if energy < self.best_energy:
            self.best_energy = energy
            self.no_imp = 0
        else:
            self.no_imp += 1

        if gnorm < 1e-6:
            return self._rec("converged")
        if gnorm >= self.gnorm_abs_tol:
            return self._rec("active")

        stag_tol = max(1e-4, 1e-3 * abs(float(min(self.energy_buf)) if self.energy_buf else energy))
        stagnant = (
            len(self.energy_buf) == self.energy_buf.maxlen
            and abs(float(self.energy_buf[-1]) - float(self.energy_buf[0])) < stag_tol
        )
        return self._rec("stagnant" if stagnant else "active")

    def _rec(self, phase: str) -> str:
        self.phase_history.append(phase)
        if phase == "stagnant":
            self.n_stagnant += 1
        elif phase == "active":
            self.n_active += 1
        return phase

    def reset_no_imp(self) -> None:
        self.no_imp = 0
        self.best_energy = np.inf


class SigmaQGD(BaseOptimizer):
    """
    SIGMA-QGD optimizer with variance-aware diagonal preconditioning.
    """

    supports_lr = True
    default_lr = 0.1

    def __init__(
        self,
        use_precond: bool = True,
        use_momentum: bool = True,
        use_clip: bool = True,
        use_drift_correction: bool = False,
        precond_variance: str = "raw",
        precond_multiplier: float = 1.0,
        use_restarts: bool = False,
        ema_window: Optional[int] = None,
        variance_window: Optional[int] = None,
        logger: Optional[ParameterLogger] = None,
    ):
        if precond_variance not in ("raw", "shrunk"):
            raise ValueError(f"precond_variance must be 'raw' or 'shrunk', got {precond_variance!r}")
        self.use_precond = use_precond
        self.use_momentum = use_momentum
        self.use_clip = use_clip
        self.use_drift_correction = use_drift_correction
        self.precond_variance = precond_variance
        self.precond_multiplier = precond_multiplier
        self.use_restarts = use_restarts
        self.ema_window = ema_window
        self.variance_window = variance_window
        self.logger = logger

    def run(
        self,
        cost_fn,
        theta_init,
        max_steps: int = 200,
        ham_type: str = "tfim",
        n_qubits: int = 10,
        reps: int = 3,
        eta_0: float = 0.1,
        lr: float = None,
        lambda_reg: float = 0.005,
        tau: float = 1.0,
        seed: int = 0,
        **kwargs,
    ) -> tuple:
        if lr is not None:
            eta_0 = lr

        lambda_reg = kwargs.get("lambda_reg", lambda_reg)
        precond_mult = kwargs.get("multiplier", kwargs.get("precond_multiplier", self.precond_multiplier))

        t0 = time.time()
        cfg = derive_config(
            n_qubits, reps, max_steps, ham_type,
            eta_0, lambda_reg, tau, ema_window=self.ema_window,
        )
        p = cfg["p"]

        if self.use_drift_correction:
            var_eng = DriftCorrectedVarianceEngine(p, cfg["W"], variance_window=self.variance_window)
        else:
            var_eng = VarianceEngine(p, cfg["W"], variance_window=self.variance_window)

        curvature = CurvatureProxy(eta_0, lambda_reg, multiplier=precond_mult)
        lscape = LandscapeClassifier(cfg, eta_0)

        theta = theta_init.copy()
        m = np.zeros(p)
        beta = cfg["beta"]
        best_e = np.inf
        best_theta = theta.copy()
        n_restarts = 0
        max_restarts = 3

        energy_hist: List[float] = []
        gnorm_hist: List[float] = []
        circ_hist: List[int] = []
        run_records: List[StepRecord] = []

        rep = kwargs.get("progress_reporter") or self._create_progress_reporter(
            cost_fn, max_steps, kwargs, default_desc="SIGMA-QGD v9.0"
        )

        for t in range(1, max_steps + 1):
            cost, g = cost_fn.cost_and_gradient(theta)
            n_step_circs = 2 * p + 1

            gnorm = float(np.linalg.norm(g))
            cn = cfg["clip_norm"]
            gc = g * cn / (gnorm + 1e-30) if (self.use_clip and gnorm > cn) else g.copy()

            if self.use_drift_correction and isinstance(var_eng, DriftCorrectedVarianceEngine):
                var_eng.update_with_theta(gc, theta)
                welford_for_precond = var_eng.welford_corrected
            else:
                var_eng.update(gc)
                welford_for_precond = var_eng.welford_raw

            precond_var_vec = (
                var_eng.js_shrunk if self.precond_variance == "shrunk"
                else welford_for_precond
            )

            if self.use_momentum:
                m = beta * m + (1.0 - beta) * gc
                mhat = m / (1.0 - beta ** t)
            else:
                mhat = gc

            if self.use_precond and t >= cfg["t_qfim"]:
                d = curvature.precondition(precond_var_vec, mhat)
            else:
                d = mhat

            theta = theta - eta_0 * d
            phase = lscape.classify(cost, gnorm)

            if cost < best_e:
                best_e = cost
                best_theta = theta.copy()

            restarted = False
            if (self.use_restarts and lscape.no_imp >= cfg["no_imp_lim"] and n_restarts < max_restarts):
                theta = best_theta + 0.05 * np.random.randn(p)
                m = np.zeros(p)
                var_eng = (
                    DriftCorrectedVarianceEngine(p, cfg["W"], variance_window=self.variance_window)
                    if self.use_drift_correction
                    else VarianceEngine(p, cfg["W"], variance_window=self.variance_window)
                )
                lscape.reset_no_imp()
                lscape.energy_buf.clear()
                n_restarts += 1
                restarted = True

            energy_hist.append(cost)
            gnorm_hist.append(gnorm)
            circ_hist.append(n_step_circs)

            rep.update(
                t,
                cost,
                theta,
                gnorm=gnorm,
                circuits=cost_fn.n_calls,
                extra=f"ph={phase}",
            )

            if self.logger is not None:
                gamma_val = curvature.gamma(precond_var_vec)
                run_records.append(StepRecord(
                    seed=seed, step=t, ham_type=ham_type,
                    n_qubits=n_qubits, reps=reps,
                    eta_0=eta_0, lambda_reg=lambda_reg, tau=tau,
                    energy=float(cost), gnorm=float(gnorm), phase=phase,
                    welford_mean=float(np.mean(var_eng.welford_raw)),
                    js_mean=float(np.mean(var_eng.js_shrunk)),
                    ema_mean=float(np.mean(var_eng.ema)),
                    momentum_norm=float(np.linalg.norm(mhat)),
                    curvature_gamma=float(gamma_val),
                    n_circuits_step=n_step_circs,
                    restarted=restarted,
                ))

        circ_info = f"rst={n_restarts}" if n_restarts > 0 else ""
        rep.finish(
            final_energy=best_e,
            final_theta=best_theta,
            circuits=cost_fn.n_calls,
            steps=len(energy_hist),
            extra=circ_info,
        )

        result = Result(
            method="SIGMA-QGD v9.0",
            energy_opt=best_e,
            energy_hist=energy_hist,
            gnorm_hist=gnorm_hist,
            n_circuits=cost_fn.n_calls,
            n_steps=len(energy_hist),
            restarts=n_restarts,
            phase_hist=lscape.phase_history.copy(),
            circuits_hist=circ_hist,
            wall_time=time.time() - t0,
            best_theta=best_theta.copy(),
        )
        return result, run_records, best_theta
