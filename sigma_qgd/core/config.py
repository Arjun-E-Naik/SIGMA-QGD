"""
Configuration derivation and experiment data structures.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepRecord:
    """Per-step diagnostic record written to the experiment log."""

    seed: int
    step: int
    ham_type: str
    n_qubits: int
    reps: int
    eta_0: float
    lambda_reg: float
    tau: float
    energy: float
    gnorm: float
    phase: str
    welford_mean: float
    js_mean: float
    ema_mean: float
    momentum_norm: float
    curvature_gamma: float
    n_circuits_step: int
    restarted: bool
    final_gap: float = 0.0
    n_steps_total: int = 0


@dataclass
class Result:
    """Container for optimizer execution metrics and trajectory records."""

    method: str
    energy_opt: float
    energy_hist: List[float]
    gnorm_hist: List[float]
    n_circuits: int
    n_steps: int
    restarts: int = 0
    phase_hist: List[str] = field(default_factory=list)
    escape_log: List[dict] = field(default_factory=list)
    circuits_hist: List[int] = field(default_factory=list)
    wall_time: float = 0.0
    best_theta: Optional[object] = None


def derive_config(
    n_qubits: int,
    reps: int,
    max_steps: int,
    ham_type: str,
    eta_0: float = 0.1,
    lambda_reg: float = 0.005,
    tau: float = 1.0,
    ema_window: Optional[int] = None,
) -> dict:
    """
    Derive internal optimizer hyperparameters for a given system configuration.
    """
    p = n_qubits * reps
    xxz = ham_type.lower() in ("xxz", "heisenberg", "xyz")

    W = int(ema_window) if ema_window is not None else min(30, max(15, max_steps // 10))
    W_med = max(10, 3 * W)
    t_qfim = min(3 * p, max(10, max_steps // 5))
    beta = float(np.clip(1.0 - 1.0 / np.sqrt(max(p, 4)), 0.5, 0.98))

    clip_norm = float(np.sqrt(p))
    warmup = max(int(0.75 * p), max_steps // 5, 15)
    cooldown = max(10, max_steps // 12)
    stag_win = max(10, max_steps // 12)
    no_imp_lim = max(60, max_steps // 4)

    return dict(
        p=p,
        n_qubits=n_qubits,
        reps=reps,
        max_steps=max_steps,
        ham_type=ham_type,
        xxz=xxz,
        eta_0=eta_0,
        lambda_reg=lambda_reg,
        tau=tau,
        W=W,
        W_med=W_med,
        t_qfim=t_qfim,
        beta=beta,
        eps=1e-8,
        clip_norm=clip_norm,
        warmup=warmup,
        cooldown=cooldown,
        stag_win=stag_win,
        no_imp_lim=no_imp_lim,
    )
