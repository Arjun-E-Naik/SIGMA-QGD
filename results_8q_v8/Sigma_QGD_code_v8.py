

"""
SIGMA-QGD v8.0 — Geometry-Inspired Adaptive Optimizer for VQE


Usage:
  python Sigma_QGD_code.py                        # Default 4q benchmark
  python Sigma_QGD_code.py --qubits 6 --seeds 5   # 6-qubit benchmark
  python Sigma_QGD_code.py --ablation              # Run ablation study
  python Sigma_QGD_code.py --scaling               # Scaling study (4-8q)
  python Sigma_QGD_code.py --shot-noise            # Shot-noise experiment
"""

import argparse
import json
import csv
import os
import sys
import time
import warnings


if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ═══════════════════════════════════════════════════════════════════
#  Visual 
# ════════════════════════════lette═══════════════════════════════════════

COLOURS = {
    "SIGMA-QGD v8.0": "#00C9A7",
    "Adam (best)":     "#E24B4A",
    "COBYLA":          "#708090",
    "SPSA":            "#8B4513",
    "Diag-QNG":        "#6A0DAD",
    "QN-SPSA":         "#FF6F00",
}
LINES   = {
    "SIGMA-QGD v8.0": "-",  "Adam (best)": "-.", "COBYLA": "--",
    "SPSA": ":",             "Diag-QNG": "-",     "QN-SPSA": "--",
}
LW      = {
    "SIGMA-QGD v8.0": 3.0,  "Adam (best)": 1.8,  "COBYLA": 1.5,
    "SPSA": 1.5,             "Diag-QNG": 2.0,     "QN-SPSA": 1.8,
}
MARKERS = {
    "SIGMA-QGD v8.0": "D",  "Adam (best)": "X",  "COBYLA": "H",
    "SPSA": "p",             "Diag-QNG": "^",     "QN-SPSA": "v",
}


# ═══════════════════════════════════════════════════════════════════
#  Data logging
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
   
    seed:             int
    step:             int
    ham_type:         str
    n_qubits:         int
    reps:             int
    eta_0:            float
    lambda_reg:       float
    tau:              float
    energy:           float
    gnorm:            float
    phase:            str
    cusum_alarm_frac: float
    welford_mean:     float
    js_mean:          float
    ema_mean:         float
    momentum_norm:    float
    curvature_gamma:  float   
    n_circuits_step:  int
    escaped:          bool
    escape_mode:      str
    restarted:        bool
    final_gap:        float = 0.0
    n_steps_total:    int   = 0


class ParameterLogger:
    

    def __init__(self, out_dir: str = "sigma_data"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.records: List[StepRecord] = []

    def finalise_run(self, run_records: List[StepRecord],
                     final_gap: float, n_steps: int) -> None:
        for r in run_records:
            r.final_gap     = final_gap
            r.n_steps_total = n_steps
        self.records.extend(run_records)

    def save(self) -> None:
        if not self.records:
            return
        jpath = os.path.join(self.out_dir, "step_records.json")
        with open(jpath, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        cpath = os.path.join(self.out_dir, "step_records.csv")
        keys  = list(asdict(self.records[0]).keys())
        with open(cpath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        print(f"  [Logger] {len(self.records)} records → {self.out_dir}/")

    def run_summary_csv(self, summaries: List[dict]) -> None:
        if not summaries:
            return
        spath = os.path.join(self.out_dir, "run_summary.csv")
        keys  = list(summaries[0].keys())
        with open(spath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)
        print(f"  [Logger] Run summaries → {spath}")


# ═══════════════════════════════════════════════════════════════════
#  Auto-configuration from problem dimensions
# ═══════════════════════════════════════════════════════════════════

def derive_config(n_qubits: int, reps: int, max_steps: int,
                  ham_type: str, eta_0: float = 0.05,
                  lambda_reg: float = 0.01, tau: float = 1.0) -> dict:
    """Derive all internal hyperparameters from (n_qubits, reps, T)."""

    p   = n_qubits * reps
    xxz = ham_type.lower() in ('xxz', 'heisenberg', 'xyz')

    W      = min(30, max(15, max_steps // 10))
    W_med  = max(10, 3 * W)
    t_qfim = min(3 * p, max(10, max_steps // 5))

  
    beta = float(np.clip(1.0 - 1.0 / np.sqrt(max(p, 4)), 0.5, 0.98))

    clip_norm  = float(np.sqrt(p))
    warmup     = max(int(0.75 * p), max_steps // 5, 15)
    cooldown   = max(10, max_steps // 12)
    stag_win   = max(10, max_steps // 12)
    no_imp_lim = max(60, max_steps // 4)

    # VSNG cache lifetime for reliable dimensions
    k_cache = 3

    return dict(
        p=p, n_qubits=n_qubits, reps=reps, max_steps=max_steps,
        ham_type=ham_type, xxz=xxz,
        eta_0=eta_0, lambda_reg=lambda_reg, tau=tau,
        W=W, W_med=W_med, t_qfim=t_qfim, beta=beta, eps=1e-8,
        clip_norm=clip_norm, warmup=warmup, cooldown=cooldown,
        stag_win=stag_win, no_imp_lim=no_imp_lim,
        k_cache=k_cache,
    )


# ═══════════════════════════════════════════════════════════════════
#  Component 1: Welford Variance Tracker (Curvature Proxy)
# ═══════════════════════════════════════════════════════════════════


class VarianceEngine:
    """Online Welford + EMA + Efron-Morris shrinkage variance tracker."""

    def __init__(self, p: int, W: int, eps: float = 1e-10):
        self.p     = p
        self.eps   = eps
        self.alpha = 2.0 / (W + 1)

        self.t           = 0
        self.M           = np.zeros(p)
        self.S           = np.zeros(p)
        self.welford_raw = np.zeros(p)
        self.js_shrunk   = np.full(p, 1e-4)
        self.ema         = np.zeros(p)
        self.ema_mean    = np.zeros(p)
        self.mean        = np.zeros(p)

    def update(self, g: np.ndarray) -> None:
        self.t += 1

       
        d1      = g - self.M
        self.M += d1 / self.t
        d2      = g - self.M
        self.S += d1 * d2
        self.mean = self.M.copy()

        if self.t < 2:
            return

        sigma_bessel     = self.S / (self.t - 1)
        self.welford_raw = sigma_bessel.copy()

        # Efron-Morris shrinkage toward grand mean
        sigma_bar = float(np.mean(sigma_bessel)) + self.eps
        deviations = sigma_bessel - sigma_bar
        Q = float(np.sum(deviations ** 2)) + self.eps
        shrink_coef = max(0.0, 1.0 - (self.p - 2) * (sigma_bar ** 2) / (Q * self.t))

        self.js_shrunk = np.maximum(
            sigma_bar + shrink_coef * deviations,
            1e-8
        )

        # EMA variance 
        self.ema_mean = (1.0 - self.alpha) * self.ema_mean + self.alpha * g
        diff_ema = g - self.ema_mean
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * diff_ema ** 2


# ═══════════════════════════════════════════════════════════════════
#  Component 2: Normalized CUSUM Change-Point Detector
# ═══════════════════════════════════════════════════════════════════

class NormalizedCUSUM:
   

    def __init__(self, p: int, W_med: int, tau: float,
                 h_factor: float = 5.0, eps: float = 1e-9):
        self.p           = p
        self.k           = tau
        self.h           = tau * h_factor
        self.eps         = eps
        self.S           = np.zeros(p)
        self.buf         = deque(maxlen=W_med)
        self.alarm_state = np.zeros(p, dtype=bool)

    def update(self, ema_var: np.ndarray) -> None:
        self.buf.append(ema_var.copy())
        if len(self.buf) < 3:
            return
        arr = np.array(self.buf)
        med = np.median(arr, axis=0)
        z_t = ema_var / (med + self.eps)
        self.S = np.maximum(0.0, self.S + z_t - self.k)
        self.alarm_state = self.S > self.h

    def alarm(self) -> np.ndarray:
        return self.alarm_state.copy()

    def reset(self, mask: np.ndarray) -> None:
        self.S[mask] = 0.0

    @property
    def alarm_frac(self) -> float:
        return float(np.mean(self.alarm_state))


# ═══════════════════════════════════════════════════════════════════
#  Component 3: Curvature-Proxy Preconditioner
# ═══════════════════════════════════════════════════════════════════

class CurvatureProxy:
    """Curvature-proxy preconditioner using Welford variance."""

    def __init__(self, eta_0: float, lambda_reg: float, eps: float = 1e-8):
        self.eta_0      = eta_0
        self.lambda_reg = lambda_reg
        self.eps        = eps

    def precondition(self, welford_raw: np.ndarray,
                     gradient: np.ndarray) -> np.ndarray:

        F_diag = 4.0 * welford_raw + self.lambda_reg
        return gradient / (F_diag + self.eps)

    def gamma(self, welford_raw: np.ndarray) -> float:
        """Scalar escape amplitude: eta_0 / F_mean, floored at 2*eta_0."""
        F_mean      = float(4.0 * np.mean(welford_raw) + self.lambda_reg)
        gamma_floor = 2.0 * self.eta_0
        return float(max(self.eta_0 / (F_mean + self.eps), gamma_floor))

    def per_dim_sigma(self, welford_raw: np.ndarray,
                      gamma: float) -> np.ndarray:
        """Per-parameter escape amplitude: gamma / sqrt(F_ii)."""
        F_diag = 4.0 * welford_raw + self.lambda_reg
        return gamma / (np.sqrt(F_diag) + self.eps)


# ═══════════════════════════════════════════════════════════════════
#  Component 4: Variance-Signal Natural Gradient (VSNG) v2
# ═══════════════════════════════════════════════════════════════════


class VarianceSignalNaturalGradient:


    def __init__(self, p: int, k_cache: int = 3,
                 frac: float = 0.6, eps: float = 1e-10):
        self.p       = p
        self.k_cache = k_cache
        self.frac    = frac
        self.eps     = eps

        self.g_cache             = np.zeros(p)
        self.steps_since_update  = np.full(p, k_cache, dtype=int)
        self.prev_ema_var        = np.zeros(p)
        self.n_circuits_last     = 0

    def compute(self,
                theta:     np.ndarray,
                cost_fn,
                alarm:     np.ndarray,
                cusum_S:   np.ndarray,
                cusum_h:   float,
                ema_var:   np.ndarray,
                E_current: float,
                ) -> np.ndarray:
       

        g         = np.zeros(self.p)
        n_circ    = 0

        # Partition dimensions
        full_alarm  = alarm & (cusum_S >= cusum_h * self.frac)
        transition  = alarm & ~full_alarm
        reliable    = ~alarm

        
        reliable_idx = np.where(reliable)[0]
        for i in reliable_idx:
            self.steps_since_update[i] += 1
            if self.steps_since_update[i] >= self.k_cache:
                p_plus  = theta.copy(); p_plus[i]  += np.pi / 2
                p_minus = theta.copy(); p_minus[i] -= np.pi / 2
                self.g_cache[i] = 0.5 * (cost_fn(p_plus) - cost_fn(p_minus))
                self.steps_since_update[i] = 0
                n_circ += 2
            g[i] = self.g_cache[i]

      
        trans_idx = np.where(transition)[0]
        for i in trans_idx:
            delta_i = float(np.sqrt(max(ema_var[i], 1e-6)))
            p_pert  = theta.copy()
            p_pert[i] += delta_i
            g_1pt   = (cost_fn(p_pert) - E_current) / (delta_i + self.eps)
            g[i]    = g_1pt
            n_circ += 1

       
        full_alarm_idx = np.where(full_alarm)[0]
        if len(full_alarm_idx) > 0:
          
            delta = np.zeros(self.p)
            delta[full_alarm_idx] = np.random.choice([-1.0, 1.0],
                                                      size=len(full_alarm_idx))
            ck = max(0.01, float(np.mean(np.sqrt(
                np.maximum(ema_var[full_alarm_idx], 1e-8)))))

            fp = cost_fn(theta + ck * delta)
            fm = cost_fn(theta - ck * delta)
            n_circ += 2

          
            for i in full_alarm_idx:
                if abs(delta[i]) > 0.5:
                    g[i] = (fp - fm) / (2.0 * ck * delta[i])

        self.prev_ema_var    = ema_var.copy()
        self.n_circuits_last = n_circ
        return g


# ═══════════════════════════════════════════════════════════════════
#  Component 5: Landscape Classifier
# ═══════════════════════════════════════════════════════════════════

class LandscapeClassifier:
    """Classifies optimization landscape phase from gradient & CUSUM."""

    def __init__(self, cfg: dict, eta_0: float = 0.05):
        self.stag_win       = cfg['stag_win']
        self.no_imp_lim     = cfg['no_imp_lim']
        self.p              = cfg['p']
        self.q_alpha        = 0.25
        self.gnorm_abs_tol  = float(np.sqrt(self.p)) * 0.20
        self.stag_tol       = 1e-4

        self.energy_buf     = deque(maxlen=self.stag_win)
        self.progress_win   = deque(maxlen=10)
        self.best_energy    = np.inf
        self.no_imp         = 0
        self.phase_history: List[str] = []

    def classify(self, energy: float, gnorm: float,
                 cusum_alarm: np.ndarray) -> str:
        self.energy_buf.append(energy)
        self.progress_win.append(energy)

        if energy < self.best_energy:
            self.best_energy = energy
            self.no_imp      = 0
        else:
            self.no_imp += 1

        if gnorm < 1e-6:
            return self._rec('converged')

        if gnorm >= self.gnorm_abs_tol:
            return self._rec('active')

        stagnant = (
            len(self.energy_buf) == self.energy_buf.maxlen
            and abs(float(self.energy_buf[-1]) - float(self.energy_buf[0]))
            < self.stag_tol
        )

        cusum_frac  = float(np.mean(cusum_alarm))
        cusum_fired = cusum_frac > self.q_alpha

        if stagnant and cusum_fired:
            phase = 'plateau'
        elif stagnant and not cusum_fired:
            phase = 'local_min'
        else:
            phase = 'active'

        return self._rec(phase)

    def _rec(self, phase: str) -> str:
        self.phase_history.append(phase)
        return phase

    def reset_no_imp(self) -> None:
        self.no_imp      = 0
        self.best_energy = np.inf


# ═══════════════════════════════════════════════════════════════════
#  Component 6: Single-Dimension Metropolis Escape
# ═══════════════════════════════════════════════════════════════════

class SingleDimEscape:
    """Single-dimension Metropolis perturbation for escaping plateaus."""

    def __init__(self, cfg: dict, eta_0: float = 0.05,
                 lambda_reg: float = 0.01,
                 T_scale: float = 8.0,
                 sigma_scale: float = 0.08):
        self.warmup     = cfg['warmup']
        self.cooldown   = cfg['cooldown']
        self.t_half     = max(cfg['max_steps'] * 3 // 4, 50)
        self.last_esc   = -cfg['cooldown']
        self.eta_0      = eta_0
        self.lambda_reg = lambda_reg
        self.eps        = cfg.get('eps', 1e-8)
        self.T_scale    = T_scale
        self.sigma_scale= sigma_scale
        self.min_stuck  = cfg['cooldown']
        self.escapes_a  = 0
        self.escape_log: List[dict] = []

    def _temperature(self, welford_raw: np.ndarray, step: int) -> float:
        F_mean  = float(4.0 * np.mean(welford_raw) + self.lambda_reg)
        T_base  = self.T_scale * self.eta_0 / (F_mean + self.eps)
        decay   = 1.0 / (1.0 + step / (self.t_half + self.eps))
        return float(T_base * decay)

    def escape(self,
               theta:       np.ndarray,
               phase:       str,
               no_imp:      int,
               cusum_alarm: np.ndarray,
               welford_raw: np.ndarray,
               cost_fn,
               cost:        float,
               sigma:       np.ndarray,
               step:        int,
               ) -> Tuple[np.ndarray, float, bool]:

        # Guards
        if step <= self.warmup:
            return theta, cost, False
        if (step - self.last_esc) < self.cooldown:
            return theta, cost, False
        if phase not in ('plateau', 'local_min'):
            return theta, cost, False
        if no_imp < self.min_stuck:
            return theta, cost, False
        if not np.any(cusum_alarm):
            return theta, cost, False

        T = self._temperature(welford_raw, step)

       
        alarmed_idx = np.where(cusum_alarm)[0]
        i           = int(np.random.choice(alarmed_idx))

        # Propose small step
        direction = float(np.sign(np.random.randn()))
        amp       = self.sigma_scale * float(sigma[i])
        cand      = theta.copy()
        cand[i]   = theta[i] + direction * amp

        new_cost = cost_fn(cand)
        dc       = new_cost - cost
        accepted = dc < 0 or (
            np.random.rand() < np.exp(-dc / (T + 1e-30))
        )

        self.escape_log.append(dict(
            step=step, dim=i, dc=round(float(dc), 6),
            amp=round(amp, 6), accepted=accepted
        ))

        if accepted:
            self.last_esc  = step
            self.escapes_a += 1
            return cand, new_cost, True

        return theta, cost, False

    def summary(self) -> dict:
        n_tried  = len(self.escape_log)
        n_acc    = sum(1 for e in self.escape_log if e['accepted'])
        rate     = (n_acc / n_tried) if n_tried > 0 else 0.0
        return dict(escapes_A=self.escapes_a, n_tried=n_tried,
                    accept_rate=round(rate, 4))


# ═══════════════════════════════════════════════════════════════════
#  Hamiltonians
# ═══════════════════════════════════════════════════════════════════

def build_tfim(n: int, J: float = 1.0, h: float = 0.5) -> SparsePauliOp:
    """Transverse-field Ising: H = −J Σ ZZ − h Σ X  (periodic BC)."""
    terms = []
    for i in range(n):
        j = (i + 1) % n
        p = ['I'] * n; p[i] = 'Z'; p[j] = 'Z'
        terms.append((''.join(reversed(p)), -J))
    for i in range(n):
        p = ['I'] * n; p[i] = 'X'
        terms.append((''.join(reversed(p)), -h))
    return SparsePauliOp.from_list(terms)


def build_xxz(n: int, Jxy: float = 1.0,
              Jz: float = 0.5, h: float = 0.1) -> SparsePauliOp:
    """XXZ Heisenberg: H = Jxy Σ (XX+YY) + Jz Σ ZZ + h Σ Z  (periodic BC)."""
    terms = []
    for i in range(n):
        j = (i + 1) % n
        for op in ('X', 'Y', 'Z'):
            p = ['I'] * n; p[i] = op; p[j] = op
            coeff = Jxy if op in ('X', 'Y') else Jz
            terms.append((''.join(reversed(p)), coeff))
    for i in range(n):
        p = ['I'] * n; p[i] = 'Z'
        terms.append((''.join(reversed(p)), h))
    return SparsePauliOp.from_list(terms)


def build_xyz(n: int, Jx: float = 1.0, Jy: float = 0.8,
              Jz: float = 0.5, h: float = 0.1) -> SparsePauliOp:
    """Heisenberg XYZ: H = Jx Σ XX + Jy Σ YY + Jz Σ ZZ + h Σ Z (periodic)."""
    terms = []
    J_map = {'X': Jx, 'Y': Jy, 'Z': Jz}
    for i in range(n):
        j = (i + 1) % n
        for op in ('X', 'Y', 'Z'):
            p = ['I'] * n; p[i] = op; p[j] = op
            terms.append((''.join(reversed(p)), J_map[op]))
    for i in range(n):
        p = ['I'] * n; p[i] = 'Z'
        terms.append((''.join(reversed(p)), h))
    return SparsePauliOp.from_list(terms)


def exact_gs(H: SparsePauliOp) -> float:
    """Exact ground-state energy via dense diagonalisation."""
    return float(np.min(np.linalg.eigvalsh(H.to_matrix())))


# ═══════════════════════════════════════════════════════════════════
#  Ansätze
# ═══════════════════════════════════════════════════════════════════

def build_ansatz(n: int, reps: int) -> QuantumCircuit:
    """Hardware-efficient ansatz: Ry layers + CX entanglement."""
    from qiskit.circuit import ParameterVector
    theta = ParameterVector('θ', n * reps)
    qc    = QuantumCircuit(n)
    idx   = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(theta[idx], q)
            idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def build_qaoa_ansatz(n: int, reps: int) -> QuantumCircuit:
    """QAOA-style ansatz: mixer (Rx) + problem (ZZ) layers."""
    from qiskit.circuit import ParameterVector
    n_params = 2 * reps  # gamma + beta per layer
    theta = ParameterVector('θ', n_params)
    qc    = QuantumCircuit(n)
    # Initial superposition
    for q in range(n):
        qc.h(q)
    idx = 0
    for r in range(reps):
     
        gamma = theta[idx]; idx += 1
        for q in range(n - 1):
            qc.cx(q, q + 1)
            qc.rz(gamma, q + 1)
            qc.cx(q, q + 1)

        beta = theta[idx]; idx += 1
        for q in range(n):
            qc.rx(beta, q)
    return qc


# ═══════════════════════════════════════════════════════════════════
#  Cost evaluator (statevector)
# ═══════════════════════════════════════════════════════════════════

class CostEvaluator:
    """Wraps Qiskit StatevectorEstimator; counts circuit evaluations."""

    def __init__(self, circuit: QuantumCircuit, H: SparsePauliOp):
        self.circuit   = circuit
        self.H         = H
        self.n_calls   = 0
        self.estimator = StatevectorEstimator()

    def __call__(self, theta: np.ndarray) -> float:
        self.n_calls += 1
        bound = self.circuit.assign_parameters(
            dict(zip(self.circuit.parameters, theta)))
        return float(
            self.estimator.run([(bound, self.H)]).result()[0].data.evs)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Full parameter-shift gradient."""
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            p_plus  = theta.copy(); p_plus[i]  += np.pi / 2
            p_minus = theta.copy(); p_minus[i] -= np.pi / 2
            g[i] = 0.5 * (self(p_plus) - self(p_minus))
        return g


class ShotNoiseCostEvaluator:


    def __init__(self, circuit: QuantumCircuit, H: SparsePauliOp,
                 n_shots: int = 1024):
        self.base  = CostEvaluator(circuit, H)
        self.circuit = circuit
        self.H       = H
        self.n_shots = n_shots
        self.n_calls = 0
        self._var_H  = None 

    def __call__(self, theta: np.ndarray) -> float:
        self.n_calls += 1
        exact_E = self.base(theta)
        self.base.n_calls -= 1 

     
        if self._var_H is None:
            H2 = self.H @ self.H
            bound = self.circuit.assign_parameters(
                dict(zip(self.circuit.parameters, theta)))
            ev2 = float(
                self.base.estimator.run([(bound, H2)]).result()[0].data.evs)
            self._var_H = max(ev2 - exact_E**2, 0.01)

        # Add shot noise
        shot_std = np.sqrt(self._var_H / self.n_shots)
        return exact_E + np.random.normal(0, shot_std)

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Full parameter-shift gradient with shot noise."""
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            p_plus  = theta.copy(); p_plus[i]  += np.pi / 2
            p_minus = theta.copy(); p_minus[i] -= np.pi / 2
            g[i] = 0.5 * (self(p_plus) - self(p_minus))
        return g

    @property
    def shot_var_estimate(self) -> float:
        """Estimated per-component shot variance for subtraction."""
        if self._var_H is None:
            return 1e-4
        return self._var_H / (4.0 * self.n_shots)


# ═══════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Result:
    method:          str
    energy_opt:      float
    energy_hist:     List[float]
    gnorm_hist:      List[float]
    n_circuits:      int
    n_steps:         int
    escapes_a:       int = 0
    restarts:        int = 0
    phase_hist:      List[str]  = field(default_factory=list)
    escape_log:      List[dict] = field(default_factory=list)
    circuits_hist:   List[int]  = field(default_factory=list)
    wall_time:       float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  SIGMA-QGD v8.0 — Main optimizer
# ═══════════════════════════════════════════════════════════════════

def run_sigma_qgd_v8(cost_fn,
                     theta_init:   np.ndarray,
                     n_qubits:     int,
                     reps:         int,
                     max_steps:    int  = 200,
                     ham_type:     str  = 'tfim',
                     eta_0:        float = 0.05,
                     lambda_reg:   float = 0.01,
                     tau:          float = 1.0,
                     use_escape:   bool = True,
                     use_precond:  bool = True,
                     use_momentum: bool = True,
                     use_cusum:    bool = True,
                     use_vsng:     bool = True,
                     use_anneal:   bool = True,
                     use_clip:     bool = True,
                     logger:       ParameterLogger = None,
                     seed:         int  = 0,
                     ) -> Tuple['Result', List[StepRecord]]:


    t0  = time.time()
    cfg = derive_config(n_qubits, reps, max_steps, ham_type,
                        eta_0, lambda_reg, tau)
    p   = cfg['p']

    # Instantiate components
    var_eng   = VarianceEngine(p, cfg['W'])
    cusum     = NormalizedCUSUM(p, cfg['W_med'], tau, h_factor=5.0)
    curvature = CurvatureProxy(eta_0, lambda_reg)
    lscape    = LandscapeClassifier(cfg, eta_0)
    escape    = SingleDimEscape(cfg, eta_0, lambda_reg) if use_escape else None
    vsng      = VarianceSignalNaturalGradient(p, k_cache=cfg['k_cache'])

   
    anneal_rate  = 3.0 / max_steps
    eta_floor    = 1e-4

  
    theta    = theta_init.copy()
    m        = np.zeros(p)
    beta     = cfg['beta']
    eps      = cfg['eps']

    best_e     = np.inf
    best_theta = theta.copy()
    n_restarts = 0
    max_restarts = 3

    energy_hist:   List[float] = []
    gnorm_hist:    List[float] = []
    circ_hist:     List[int]   = []
    run_records:   List[StepRecord] = []

    for t in range(1, max_steps + 1):

      
        cost   = cost_fn(theta)
        n_step_circs = 1

        
        alarm   = cusum.alarm() if use_cusum else np.zeros(p, dtype=bool)

        if t <= cfg['warmup'] or not use_vsng:
            g      = cost_fn.gradient(theta)
            n_step_circs += 2 * p
        else:
            g = vsng.compute(
                theta     = theta,
                cost_fn   = cost_fn,
                alarm     = alarm,
                cusum_S   = cusum.S,
                cusum_h   = cusum.h,
                ema_var   = var_eng.ema,
                E_current = cost,
            )
            n_step_circs += vsng.n_circuits_last

        gnorm = float(np.linalg.norm(g))

        
        cn = cfg['clip_norm']
        if use_clip and gnorm > cn:
            gc = g * cn / (gnorm + 1e-30)
        else:
            gc = g.copy()

      
        var_eng.update(gc)

        
        if use_cusum:
            cusum.update(var_eng.ema)
            alarm = cusum.alarm()

       
        if use_anneal and t > cfg['warmup']:
            anneal = 1.0 / np.sqrt(1.0 + anneal_rate * t)
            eta = max(eta_0 * anneal, eta_floor)
        else:
            eta = eta_0

        if use_momentum:
            m    = beta * m + (1.0 - beta) * gc
            mhat = m / (1.0 - beta ** t)
        else:
            mhat = gc

   
        if use_precond and t >= cfg['t_qfim']:
            d = curvature.precondition(var_eng.welford_raw, mhat)
        else:
            d = mhat

      
        theta = theta - eta * d

      
        phase = lscape.classify(cost, gnorm, alarm)

      
        if cost < best_e:
            best_e     = cost
            best_theta = theta.copy()

      
        escaped     = False
        escape_mode = ''
        if use_escape and escape is not None:
            gamma_val = curvature.gamma(var_eng.welford_raw)
            sigma_vec = curvature.per_dim_sigma(var_eng.welford_raw, gamma_val)

            new_theta, new_cost, escaped = escape.escape(
                theta       = theta,
                phase       = phase,
                no_imp      = lscape.no_imp,
                cusum_alarm = alarm,
                welford_raw = var_eng.welford_raw,
                cost_fn     = cost_fn,
                cost        = cost,
                sigma       = sigma_vec,
                step        = t,
            )
            if escaped:
                n_step_circs += 1

            if escaped:
                theta  = new_theta
                cost   = new_cost
                escape_mode = 'A'
                lscape.reset_no_imp()
                if use_cusum:
                    cusum.reset(alarm)

                if cost < best_e:
                    best_e     = cost
                    best_theta = theta.copy()

       
        restarted = False
        if (lscape.no_imp >= cfg['no_imp_lim']
                and n_restarts < max_restarts):
            theta   = best_theta + 0.05 * np.random.randn(p)
            m       = np.zeros(p)
            var_eng = VarianceEngine(p, cfg['W'])
            cusum   = NormalizedCUSUM(p, cfg['W_med'], tau, h_factor=5.0)
            vsng    = VarianceSignalNaturalGradient(p, k_cache=cfg['k_cache'])
            lscape.reset_no_imp()
            lscape.energy_buf.clear()
            lscape.progress_win.clear()
            n_restarts += 1
            restarted   = True

        
        energy_hist.append(cost)
        gnorm_hist.append(gnorm)
        circ_hist.append(n_step_circs)

        if logger is not None:
            gamma_val = curvature.gamma(var_eng.welford_raw)
            rec = StepRecord(
                seed=seed, step=t, ham_type=ham_type,
                n_qubits=n_qubits, reps=reps,
                eta_0=eta_0, lambda_reg=lambda_reg, tau=tau,
                energy=float(cost),
                gnorm=float(gnorm),
                phase=phase,
                cusum_alarm_frac=float(np.mean(alarm)),
                welford_mean=float(np.mean(var_eng.welford_raw)),
                js_mean=float(np.mean(var_eng.js_shrunk)),
                ema_mean=float(np.mean(var_eng.ema)),
                momentum_norm=float(np.linalg.norm(mhat)),
                curvature_gamma=float(gamma_val),
                n_circuits_step=n_step_circs,
                escaped=escaped,
                escape_mode=escape_mode,
                restarted=restarted,
            )
            run_records.append(rec)

       
        if t > cfg['W'] and len(energy_hist) > 2*cfg['W']:
            recent_best = min(energy_hist[-cfg['W']:])
            prev_best   = min(energy_hist[-2*cfg['W']:-cfg['W']])
            if abs(recent_best - prev_best) < 1e-5:
                break

    # Finalise
    ea   = escape.escapes_a if escape else 0
    elog = escape.escape_log if escape else []

    return Result(
        method        = "SIGMA-QGD v8.0",
        energy_opt    = best_e,
        energy_hist   = energy_hist,
        gnorm_hist    = gnorm_hist,
        n_circuits    = cost_fn.n_calls,
        n_steps       = len(energy_hist),
        escapes_a     = ea,
        restarts      = n_restarts,
        phase_hist    = lscape.phase_history.copy(),
        escape_log    = elog,
        circuits_hist = circ_hist,
        wall_time     = time.time() - t0,
    ), run_records


# ═══════════════════════════════════════════════════════════════════
#  Baseline optimizers
# ═══════════════════════════════════════════════════════════════════

def run_adam(cost_fn, theta_init: np.ndarray,
             max_steps: int = 200, lr: float = 0.01,
             label: str = "Adam") -> Tuple[Result, list]:
    """Adam optimizer with configurable learning rate."""
    theta = theta_init.copy(); p = len(theta)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    m = np.zeros(p); v = np.zeros(p)
    best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()
    for t in range(1, max_steps + 1):
        cost = cost_fn(theta); g = cost_fn.gradient(theta)
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        theta -= lr * mh / (np.sqrt(vh) + eps)
        gnorm = float(np.linalg.norm(g)); eh.append(cost); gh.append(gnorm)
        ch.append(2*p + 1)
        if cost < best_e: best_e = cost
        if gnorm < 1e-6: break
    return Result(label, best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


def run_adam_sweep(cost_fn_factory, theta_init: np.ndarray,
                   max_steps: int = 200) -> Tuple[Result, list, dict]:

    lrs = [0.001, 0.005, 0.01, 0.02, 0.05]
    best_result = None
    best_gap = np.inf
    sweep_results = {}

    for lr in lrs:
        cf = cost_fn_factory()
        r, _ = run_adam(cf, theta_init.copy(), max_steps, lr=lr,
                        label=f"Adam(lr={lr})")
        sweep_results[lr] = r.energy_opt
        if r.energy_opt < best_gap:
            best_gap = r.energy_opt
            best_result = r
            best_result.method = "Adam (best)"

    return best_result, [], sweep_results


def run_cobyla(cost_fn, theta_init: np.ndarray,
               max_steps: int = 200) -> Tuple[Result, list]:
    best_e = [np.inf]; eh = []; gh = []
    t0 = time.time()
    def obj(x):
        cost = cost_fn(x); eh.append(cost); gh.append(0.0)
        if cost < best_e[0]: best_e[0] = cost
        return cost
    minimize(obj, theta_init.copy(), method='COBYLA',
             options={'maxiter': max_steps, 'rhobeg': 0.1})
    ch = [1] * len(eh)
    return Result("COBYLA", best_e[0], eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


def run_spsa(cost_fn, theta_init: np.ndarray,
             max_steps: int = 200) -> Tuple[Result, list]:
    theta = theta_init.copy(); p = len(theta)
    a = 0.1; c = 0.1; A = max_steps / 10.0; alpha = 0.602; gamma = 0.101
    best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()
    for k in range(1, max_steps + 1):
        ak = a / (A + k)**alpha; ck = c / k**gamma
        delta = np.random.choice([-1.0, 1.0], size=p)
        fp = cost_fn(theta + ck*delta); fm = cost_fn(theta - ck*delta)
        g_sp = (fp - fm) / (2.0 * ck * delta)
        theta -= ak * g_sp; cost = cost_fn(theta)
        gnorm = float(np.linalg.norm(g_sp)); eh.append(cost); gh.append(gnorm)
        ch.append(3)
        if cost < best_e: best_e = cost
    return Result("SPSA", best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


# ═══════════════════════════════════════════════════════════════════
#  NEW Baseline: Diagonal QNG (exact diagonal QFIM)
# ═══════════════════════════════════════════════════════════════════

def run_diag_qng(cost_fn, theta_init: np.ndarray,
                 max_steps: int = 200, lr: float = 0.05,
                 lambda_reg: float = 0.01) -> Tuple[Result, list]:
    """
    Diagonal Quantum Natural Gradient.

    Computes exact diagonal QFIM via parameter-shift overlaps:
      F_ii = 1 - |⟨ψ(θ)|ψ(θ + π·e_i)⟩|²

    Cost: 3p + 1 circuits/step (p for QFIM diag + 2p for gradient + 1 eval)
    This is the natural baseline for geometry-aware methods.
    """
    theta = theta_init.copy(); p = len(theta)
    best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()

    estimator = StatevectorEstimator()

    for t in range(1, max_steps + 1):
        # Energy + gradient
        cost = cost_fn(theta)
        g = cost_fn.gradient(theta)
        gnorm = float(np.linalg.norm(g))

        # Compute diagonal QFIM via overlap
        # F_ii = 1 - |⟨ψ(θ)|ψ(θ + π·e_i)⟩|²
        # For Ry gates: ψ(θ+π·e_i) = applying Ry(θ_i+π) instead of Ry(θ_i)
        # |⟨ψ|ψ'⟩|² can be computed from statevector directly in simulation
        F_diag = np.ones(p)
        from qiskit.quantum_info import Statevector
        bound_base = cost_fn.circuit.assign_parameters(
            dict(zip(cost_fn.circuit.parameters, theta)))
        sv_base = Statevector.from_instruction(bound_base)

        for i in range(p):
            theta_shift = theta.copy()
            theta_shift[i] += np.pi
            bound_shift = cost_fn.circuit.assign_parameters(
                dict(zip(cost_fn.circuit.parameters, theta_shift)))
            sv_shift = Statevector.from_instruction(bound_shift)
            overlap = abs(sv_base.inner(sv_shift)) ** 2
            F_diag[i] = max(1.0 - overlap, 1e-8)

        # Natural gradient step
        d = g / (F_diag + lambda_reg)
        theta -= lr * d

        eh.append(cost); gh.append(gnorm)
        ch.append(3 * p + 1)  # p overlaps + 2p gradient + 1 energy
        if cost < best_e: best_e = cost
        if gnorm < 1e-6: break

    return Result("Diag-QNG", best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


# ═══════════════════════════════════════════════════════════════════
#  NEW Baseline: QN-SPSA (stochastic natural gradient)
# ═══════════════════════════════════════════════════════════════════

def run_qn_spsa(cost_fn, theta_init: np.ndarray,
                max_steps: int = 200, lr: float = 0.05,
                lambda_reg: float = 0.01) -> Tuple[Result, list]:

    theta = theta_init.copy(); p = len(theta)
    a = lr; c = 0.1; A = max_steps / 10.0
    alpha_sp = 0.602; gamma_sp = 0.101
    beta_metric = 0.01  # metric learning rate

    F_diag = np.ones(p) * 0.5  # initial metric estimate
    best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()

    for k in range(1, max_steps + 1):
        ak = a / (A + k)**alpha_sp
        ck = c / k**gamma_sp

        # SPSA gradient
        delta1 = np.random.choice([-1.0, 1.0], size=p)
        fp1 = cost_fn(theta + ck*delta1)
        fm1 = cost_fn(theta - ck*delta1)
        g_sp = (fp1 - fm1) / (2.0 * ck * delta1)

        # Metric update via second SPSA perturbation
        delta2 = np.random.choice([-1.0, 1.0], size=p)
        fp2 = cost_fn(theta + ck*delta2)
        fm2 = cost_fn(theta - ck*delta2)
        g_sp2 = (fp2 - fm2) / (2.0 * ck * delta2)

        # Diagonal metric: running average of squared gradient differences
        metric_sample = np.abs(g_sp * g_sp2)
        F_diag = (1 - beta_metric) * F_diag + beta_metric * metric_sample
        F_diag = np.maximum(F_diag, 1e-8)

        # Natural gradient step
        d = g_sp / (F_diag + lambda_reg)
        theta -= ak * d

        cost = cost_fn(theta)
        gnorm = float(np.linalg.norm(g_sp))
        eh.append(cost); gh.append(gnorm)
        ch.append(5)  # 2+2 perturbations + 1 eval
        if cost < best_e: best_e = cost

    return Result("QN-SPSA", best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


# ═══════════════════════════════════════════════════════════════════
#  Benchmark runner
# ═══════════════════════════════════════════════════════════════════

def _get_runner(name: str, n_qubits, reps, max_steps,
                ham_type, flags, logger, seed):
    if name == "SIGMA-QGD v8.0":
        def _run(cf, th):
            return run_sigma_qgd_v8(
                cf, th, n_qubits=n_qubits, reps=reps,
                max_steps=max_steps, ham_type=ham_type,
                use_escape=flags.get('escape', True),
                use_precond=flags.get('precond', True),
                use_momentum=flags.get('momentum', True),
                use_cusum=flags.get('cusum', True),
                use_vsng=flags.get('vsng', True),
                use_anneal=flags.get('anneal', True),
                use_clip=flags.get('clip', True),
                logger=logger, seed=seed)
        return _run
    runners = {
        "COBYLA": run_cobyla,
        "SPSA": run_spsa,
        "Diag-QNG": lambda cf, th, **kw: run_diag_qng(cf, th, **kw),
        "QN-SPSA": lambda cf, th, **kw: run_qn_spsa(cf, th, **kw),
    }
    if name in runners:
        def _run(cf, th):
            return runners[name](cf, th, max_steps=max_steps)
        return _run
    return None


def run_benchmark(H: SparsePauliOp, ansatz: QuantumCircuit,
                  ham_name: str, ham_type: str,
                  n_qubits: int, reps: int,
                  n_seeds: int, max_steps: int,
                  flags: dict,
                  logger: ParameterLogger,
                  methods: List[str] = None) -> Tuple[Dict, Dict, float]:
    """Run full benchmark with all methods across seeds."""

    exact_E  = exact_gs(H)
    if methods is None:
        methods = ["SIGMA-QGD v8.0", "Adam (best)", "COBYLA", "SPSA",
                    "Diag-QNG", "QN-SPSA"]
    all_res  = {m: [] for m in methods}
    cfg      = derive_config(n_qubits, reps, max_steps, ham_type)

    print(f"\n{'='*78}")
    print(f"  {ham_name}  |  n={n_qubits}  p={cfg['p']}  |  Exact: {exact_E:.4f} H")
    print(f"  W={cfg['W']}  W_med={cfg['W_med']}  t_qfim={cfg['t_qfim']}"
          f"  β={cfg['beta']:.3f}  clip={cfg['clip_norm']:.2f}"
          f"  warmup={cfg['warmup']}  k_cache={cfg['k_cache']}")
    print(f"{'='*78}")

    run_summaries = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, ansatz.num_parameters)
        print(f"\n  Seed {seed}  ‖θ₀‖={np.linalg.norm(theta0):.3f}")

        for name in methods:
            print(f"  {name:<22} ...", end=" ", flush=True)

            if name == "Adam (best)":
                # Adam sweep
                def cf_factory():
                    return CostEvaluator(ansatz, H)
                r, recs, sweep = run_adam_sweep(
                    cf_factory, theta0.copy(), max_steps)
            else:
                cf = CostEvaluator(ansatz, H)
                runner = _get_runner(name, n_qubits, reps, max_steps,
                                     ham_type, flags, logger, seed)
                if runner is None:
                    print("SKIP (unknown method)")
                    continue
                r, recs = runner(cf, theta0.copy())

            final_gap = abs(r.energy_opt - exact_E)
            if recs and logger:
                logger.finalise_run(recs, final_gap, r.n_steps)

            all_res[name].append(r)

            circ_info = ""
            if "SIGMA" in name and r.circuits_hist:
                mean_c = np.mean(r.circuits_hist)
                circ_info = f"  c/step={mean_c:.1f}  esc={r.escapes_a}  rst={r.restarts}"

            print(f"E={r.energy_opt:.4f}  gap={final_gap:.5f}"
                  f"  circ={r.n_circuits:,}  steps={r.n_steps}"
                  f"  t={r.wall_time:.1f}s{circ_info}")

            run_summaries.append(dict(
                ham=ham_name, method=name, seed=seed,
                energy_opt=round(r.energy_opt, 6),
                exact_E=round(exact_E, 6),
                gap=round(final_gap, 6),
                n_circuits=r.n_circuits,
                n_steps=r.n_steps,
                wall_time=round(r.wall_time, 3),
            ))

    if logger:
        logger.run_summary_csv(run_summaries)

    # Summary statistics
    summary = {}
    for name in methods:
        if not all_res[name]:
            continue
        energies = [r.energy_opt    for r in all_res[name]]
        gaps     = [abs(e-exact_E) for e in energies]
        summary[name] = dict(
            energy_mean   = float(np.mean(energies)),
            energy_std    = float(np.std(energies)),
            energy_all    = energies,
            gap_mean      = float(np.mean(gaps)),
            gap_std       = float(np.std(gaps)),
            steps_mean    = float(np.mean([r.n_steps    for r in all_res[name]])),
            circuits_mean = float(np.mean([r.n_circuits for r in all_res[name]])),
            circuits_all  = [r.n_circuits for r in all_res[name]],
            time_mean     = float(np.mean([r.wall_time  for r in all_res[name]])),
        )

    _print_table(summary, exact_E, ham_name, n_seeds)
    _save_benchmark_table(summary, exact_E, ham_name, n_seeds)
    return all_res, summary, exact_E


def _print_table(summary: dict, exact_E: float,
                 ham_name: str, n_seeds: int) -> None:
    v8_gap = summary.get("SIGMA-QGD v8.0", {}).get('gap_mean', float('nan'))
    print(f"\n{'='*78}")
    print(f"  RESULTS — {ham_name}  |  Exact: {exact_E:.6f} H  |  {n_seeds} seeds")
    print(f"  {'Method':<22} {'E mean±std':>20}  {'Gap':>9}  "
          f"{'Steps':>7}  {'Circuits':>10}")
    print(f"  {'─'*22} {'─'*20}  {'─'*9}  {'─'*7}  {'─'*10}")
    for name, s in sorted(summary.items(), key=lambda x: x[1]['gap_mean']):
        tag = " ★" if name == "SIGMA-QGD v8.0" else ""
        imp = ""
        if name != "SIGMA-QGD v8.0" and s['gap_mean'] > 0 and not np.isnan(v8_gap):
            ratio = s['gap_mean'] / max(v8_gap, 1e-12)
            if ratio > 1.1:
                imp = f"  ({ratio:.1f}× worse)"
        print(f"  {name:<22} {s['energy_mean']:>10.4f}±{s['energy_std']:>6.4f}"
              f"  {s['gap_mean']:>9.5f}  {s['steps_mean']:>7.1f}"
              f"  {s['circuits_mean']:>10.0f}{tag}{imp}")


def _save_benchmark_table(summary: dict, exact_E: float,
                          ham_name: str, n_seeds: int) -> None:
    os.makedirs("sigma_data", exist_ok=True)
    safe = (ham_name.replace(' ', '_').replace('(', '').replace(')', '')
            .replace(',', '').replace('=', ''))
    path = f"sigma_data/benchmark_{safe}.csv"
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'energy_mean', 'energy_std', 'gap_mean',
                    'gap_std', 'steps_mean', 'circuits_mean', 'time_mean',
                    'n_seeds', 'exact_E'])
        for name, s in summary.items():
            w.writerow([name, round(s['energy_mean'], 6), round(s['energy_std'], 6),
                        round(s['gap_mean'], 6), round(s['gap_std'], 6),
                        round(s['steps_mean'], 1), round(s['circuits_mean'], 1),
                        round(s['time_mean'], 3), n_seeds, round(exact_E, 6)])
    print(f"  [Table] Saved → {path}")


# ═══════════════════════════════════════════════════════════════════
#  Ablation study
# ═══════════════════════════════════════════════════════════════════

def run_ablation(H: SparsePauliOp, ansatz: QuantumCircuit,
                 ham_name: str, ham_type: str,
                 n_qubits: int, reps: int,
                 max_steps: int, n_seeds_abl: int = 5,
                 logger: ParameterLogger = None) -> Dict:
    """
    Full ablation study removing each component individually.
    Returns results dict and prints quantitative comparison.
    """
    exact_E = exact_gs(H)
    configs = {
        "Full v8.0":        dict(escape=True,  precond=True,  momentum=True,
                                 cusum=True,   vsng=True,     anneal=True,
                                 clip=True),
        "No escape":        dict(escape=False, precond=True,  momentum=True,
                                 cusum=True,   vsng=True,     anneal=True,
                                 clip=True),
        "No preconditioner":dict(escape=True,  precond=False, momentum=True,
                                 cusum=True,   vsng=True,     anneal=True,
                                 clip=True),
        "No momentum":      dict(escape=True,  precond=True,  momentum=False,
                                 cusum=True,   vsng=True,     anneal=True,
                                 clip=True),
        "No CUSUM":         dict(escape=True,  precond=True,  momentum=True,
                                 cusum=False,  vsng=True,     anneal=True,
                                 clip=True),
        "No VSNG":          dict(escape=True,  precond=True,  momentum=True,
                                 cusum=True,   vsng=False,    anneal=True,
                                 clip=True),
        "No annealing":     dict(escape=True,  precond=True,  momentum=True,
                                 cusum=True,   vsng=True,     anneal=False,
                                 clip=True),
        "No clipping":      dict(escape=True,  precond=True,  momentum=True,
                                 cusum=True,   vsng=True,     anneal=True,
                                 clip=False),
    }
    results = {k: [] for k in configs}
    print(f"\n{'='*78}\n  ABLATION — {ham_name}  |  {n_seeds_abl} seeds\n{'='*78}")

    for seed in range(n_seeds_abl):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, ansatz.num_parameters)
        for cname, flags in configs.items():
            cf = CostEvaluator(ansatz, H)
            r, recs = run_sigma_qgd_v8(
                cf, theta0.copy(), n_qubits=n_qubits, reps=reps,
                max_steps=max_steps, ham_type=ham_type,
                use_escape=flags['escape'], use_precond=flags['precond'],
                use_momentum=flags['momentum'], use_cusum=flags['cusum'],
                use_vsng=flags['vsng'], use_anneal=flags['anneal'],
                use_clip=flags['clip'],
                logger=logger, seed=seed)
            results[cname].append(r)

    full_gap = float(np.mean([abs(r.energy_opt-exact_E)
                               for r in results["Full v8.0"]]))

    print(f"\n  {'Config':<22} {'Gap mean':>10} {'Gap std':>10} "
          f"{'Circuits':>10} {'Ratio':>8}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for cname, rs in results.items():
        gaps   = [abs(r.energy_opt-exact_E) for r in rs]
        circs  = [r.n_circuits for r in rs]
        ratio  = float(np.mean(gaps)) / max(full_gap, 1e-12)
        tag    = " (base)" if cname == "Full v8.0" else f" {ratio:.1f}×"
        print(f"  {cname:<22} {np.mean(gaps):>10.5f} "
              f"{np.std(gaps):>10.5f} {np.mean(circs):>10.0f}{tag}")

    # Save ablation CSV
    os.makedirs("sigma_data", exist_ok=True)
    abl_path = "sigma_data/ablation_results.csv"
    with open(abl_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['config', 'gap_mean', 'gap_std', 'circuits_mean', 'ratio'])
        for cname, rs in results.items():
            gaps = [abs(r.energy_opt-exact_E) for r in rs]
            circs = [r.n_circuits for r in rs]
            ratio = float(np.mean(gaps)) / max(full_gap, 1e-12)
            w.writerow([cname, round(np.mean(gaps), 6), round(np.std(gaps), 6),
                        round(np.mean(circs), 1), round(ratio, 2)])
    print(f"  [Ablation] Saved → {abl_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  Shot-noise experiment
# ═══════════════════════════════════════════════════════════════════

def run_shot_noise_experiment(H: SparsePauliOp, ansatz: QuantumCircuit,
                              ham_name: str, ham_type: str,
                              n_qubits: int, reps: int,
                              max_steps: int, n_seeds: int = 3) -> dict:

    exact_E = exact_gs(H)
    shot_counts = [100, 500, 1000, 5000, 10000]
    results = {ns: [] for ns in shot_counts}
    # Also run noiseless as reference
    results['noiseless'] = []

    print(f"\n{'='*78}")
    print(f"  SHOT-NOISE EXPERIMENT — {ham_name}")
    print(f"  n_shots: {shot_counts}")
    print(f"{'='*78}")

    for seed in range(n_seeds):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, ansatz.num_parameters)

        # Noiseless reference
        cf = CostEvaluator(ansatz, H)
        r, _ = run_sigma_qgd_v8(
            cf, theta0.copy(), n_qubits, reps, max_steps, ham_type, seed=seed)
        results['noiseless'].append(r)

        # Shot-noise runs
        for ns in shot_counts:
            cf_noisy = ShotNoiseCostEvaluator(ansatz, H, n_shots=ns)
            r, _ = run_sigma_qgd_v8(
                cf_noisy, theta0.copy(), n_qubits, reps, max_steps,
                ham_type, seed=seed)
            results[ns].append(r)
            gap = abs(r.energy_opt - exact_E)
            print(f"  seed={seed}  shots={ns:>6}  gap={gap:.5f}")

    # Print summary
    print(f"\n  {'Shots':<12} {'Gap mean':>10} {'Gap std':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10}")
    all_shot_results = {}
    for key in ['noiseless'] + shot_counts:
        gaps = [abs(r.energy_opt - exact_E) for r in results[key]]
        label = str(key) if key != 'noiseless' else 'Noiseless'
        print(f"  {label:<12} {np.mean(gaps):>10.5f} {np.std(gaps):>10.5f}")
        all_shot_results[key] = {'gap_mean': np.mean(gaps),
                                  'gap_std': np.std(gaps)}

    return all_shot_results


# ═══════════════════════════════════════════════════════════════════
#  Scaling study
# ═══════════════════════════════════════════════════════════════════

def run_scaling_study(max_steps: int = 150, n_seeds: int = 3,
                      reps: int = 2) -> dict:
    """Run SIGMA-QGD vs Adam across multiple qubit counts."""
    qubit_counts = [4, 6, 8]
    scaling_results = {}

    print(f"\n{'='*78}")
    print(f"  SCALING STUDY — qubits: {qubit_counts}")
    print(f"{'='*78}")

    for nq in qubit_counts:
        H = build_tfim(nq)
        exact_E = exact_gs(H)
        ansatz = build_ansatz(nq, reps)

        sigma_gaps = []
        adam_gaps = []

        for seed in range(n_seeds):
            np.random.seed(seed)
            theta0 = np.random.uniform(-np.pi/8, np.pi/8,
                                        ansatz.num_parameters)

            # SIGMA-QGD
            cf = CostEvaluator(ansatz, H)
            r, _ = run_sigma_qgd_v8(
                cf, theta0.copy(), nq, reps, max_steps, 'tfim', seed=seed)
            sigma_gaps.append(abs(r.energy_opt - exact_E))

            # Best Adam
            def cf_factory():
                return CostEvaluator(ansatz, H)
            r_adam, _, _ = run_adam_sweep(cf_factory, theta0.copy(), max_steps)
            adam_gaps.append(abs(r_adam.energy_opt - exact_E))

        scaling_results[nq] = {
            'sigma_gap_mean': np.mean(sigma_gaps),
            'sigma_gap_std': np.std(sigma_gaps),
            'adam_gap_mean': np.mean(adam_gaps),
            'adam_gap_std': np.std(adam_gaps),
            'p': nq * reps,
        }

        print(f"  n={nq}  p={nq*reps}  "
              f"SIGMA gap={np.mean(sigma_gaps):.5f}±{np.std(sigma_gaps):.5f}  "
              f"Adam gap={np.mean(adam_gaps):.5f}±{np.std(adam_gaps):.5f}")

    return scaling_results


# ═══════════════════════════════════════════════════════════════════
#  Plotting — 11 publication-quality figures
# ═══════════════════════════════════════════════════════════════════

def plot_all(res_t, sum_t, E_t, res_x, sum_x, E_x,
             n_seeds: int, n_qubits: int,
             ablation_results: Dict = None,
             adam_sweep: Dict = None,
             shot_results: Dict = None,
             scaling_results: Dict = None) -> None:
    """Generate all publication-quality figures."""
    os.makedirs("sigma_data", exist_ok=True)
    methods = [m for m in COLOURS.keys() if m in (sum_t or {})]

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 8,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    # ── Helper: convergence axes ──
    def _conv_axes(ax, all_res, exact_E, title, log_scale):
        for name in COLOURS:
            if name not in all_res or not all_res[name]:
                continue
            hists = [r.energy_hist for r in all_res[name]]
            mn    = min(len(h) for h in hists)
            if mn == 0:
                continue
            arr   = np.array([h[:mn] for h in hists])
            mu = arr.mean(0); sd = arr.std(0); xs = np.arange(mn)
            c  = COLOURS[name]; lw = LW.get(name, 1.5); ls = LINES.get(name, '-')
            zo = 12 if "SIGMA" in name else 5
            if log_scale:
                ax.semilogy(xs, np.maximum(np.abs(mu - exact_E), 1e-10),
                            color=c, lw=lw, ls=ls, label=name, zorder=zo)
            else:
                ax.plot(xs, mu, color=c, lw=lw, ls=ls,
                        label=name, zorder=zo)
                ax.fill_between(xs, mu-sd, mu+sd, alpha=0.1, color=c)
        if not log_scale:
            ax.axhline(exact_E, color='k', ls=':', lw=1.2,
                       label=f'Exact ({exact_E:.3f})')
            ax.set_ylabel("Energy (H)")
        else:
            ax.set_ylabel("|E−E_exact| (log)")
        ax.set_xlabel("Step"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Fig 1: TFIM convergence ──
    if res_t and sum_t:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v8.0 — TFIM ({n_qubits}q, {n_seeds} seeds)",
                     fontsize=13, fontweight='bold')
        _conv_axes(axes[0], res_t, E_t, "Energy convergence — TFIM", False)
        _conv_axes(axes[1], res_t, E_t, "Energy gap — TFIM (log scale)", True)
        plt.tight_layout()
        plt.savefig("sigma_data/fig1_tfim_convergence.png", dpi=150,
                    bbox_inches="tight")
        plt.close(); print("  Saved: fig1_tfim_convergence.png")

    # ── Fig 2: XXZ convergence ──
    if res_x and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v8.0 — XXZ ({n_qubits}q, {n_seeds} seeds)",
                     fontsize=13, fontweight='bold')
        _conv_axes(axes[0], res_x, E_x, "Energy convergence — XXZ", False)
        _conv_axes(axes[1], res_x, E_x, "Energy gap — XXZ (log scale)", True)
        plt.tight_layout()
        plt.savefig("sigma_data/fig2_xxz_convergence.png", dpi=150,
                    bbox_inches="tight")
        plt.close(); print("  Saved: fig2_xxz_convergence.png")

    # ── Fig 3: Box plots ──
    if sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (smry, exact, title) in zip(
                axes, [(sum_t, E_t, "TFIM"), (sum_x, E_x, "XXZ")]):
            labels = [m for m in COLOURS if m in smry]
            groups = [smry[m]['energy_all'] for m in labels]
            colors = [COLOURS[m] for m in labels]
            bp = ax.boxplot(groups, labels=[l.replace(' ', '\n') for l in labels],
                            patch_artist=True,
                            medianprops=dict(color='k', lw=2))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)
            ax.axhline(exact, color='k', ls=':', lw=1.5,
                       label=f'Exact ({exact:.3f})')
            ax.set_ylabel("Final energy (H)")
            ax.set_title(f"{title} — {n_seeds} seeds")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=8)
        fig.suptitle("Final energy distributions", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("sigma_data/fig3_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig3_boxplots.png")

    # ── Fig 4: Circuit efficiency scatter ──
    if sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (smry, exact, title) in zip(
                axes, [(sum_t, E_t, "TFIM"), (sum_x, E_x, "XXZ")]):
            for name in COLOURS:
                if name not in smry:
                    continue
                s  = smry[name]; c = COLOURS[name]
                ms = 200 if "SIGMA" in name else 80
                for e, ci in zip(s['energy_all'], s['circuits_all']):
                    ax.scatter(ci, abs(e - exact), color=c, alpha=0.2,
                               s=25, marker=MARKERS.get(name, 'o'))
                ax.scatter(s['circuits_mean'], s['gap_mean'], color=c, s=ms,
                           marker=MARKERS.get(name, 'o'), zorder=10, label=name,
                           edgecolors='k', linewidths=1.5)
            ax.set_xlabel("Circuit evaluations")
            ax.set_ylabel("|E−E_exact| (log)")
            ax.set_title(f"Circuit efficiency — {title}")
            ax.set_yscale('log'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("Circuit efficiency: bottom-left = best",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("sigma_data/fig4_efficiency.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig4_efficiency.png")

    # ── Fig 5: Stability + gap bar ──
    if res_t and res_x and sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        ax5l, ax5r = axes
        for name in COLOURS:
            for all_res, ls_ in [(res_t, '-'), (res_x, '--')]:
                if name not in all_res or not all_res[name]:
                    continue
                hists = [r.energy_hist for r in all_res[name]]
                mn    = min(len(h) for h in hists)
                if mn == 0:
                    continue
                arr   = np.array([h[:mn] for h in hists])
                lw_   = 2.5 if "SIGMA" in name else 1.0
                ax5l.plot(arr.std(0), color=COLOURS[name], lw=lw_, ls=ls_,
                          alpha=0.8, label=name if ls_ == '-' else None)
        ax5l.set_yscale('log'); ax5l.legend(fontsize=7)
        ax5l.set_xlabel("Step"); ax5l.set_ylabel("Std dev (↓=stable)")
        ax5l.set_title("Stability  (solid=TFIM  dashed=XXZ)")
        ax5l.grid(True, alpha=0.3)

        labels_bar = [m for m in COLOURS if m in sum_t and m in sum_x]
        if labels_bar:
            x = np.arange(len(labels_bar)); w = 0.35
            g_t = [sum_t[m]['gap_mean'] for m in labels_bar]
            g_x = [sum_x[m]['gap_mean'] for m in labels_bar]
            clrs = [COLOURS[m] for m in labels_bar]
            ax5r.bar(x-w/2, g_t, w, color=clrs, alpha=0.85,
                     edgecolor='k', lw=0.5, label='TFIM')
            ax5r.bar(x+w/2, g_x, w, color=clrs, alpha=0.45,
                     edgecolor='k', lw=0.5, hatch='//', label='XXZ')
            ax5r.set_xticks(x)
            ax5r.set_xticklabels([l.replace(' ', '\n') for l in labels_bar],
                                rotation=20, ha='right', fontsize=8)
            ax5r.set_yscale('log')
            ax5r.set_ylabel("|E−E_exact| mean (↓=better)")
            ax5r.set_title("Final gap — TFIM vs XXZ")
            ax5r.legend(fontsize=9)
            ax5r.grid(True, alpha=0.3, axis='y')
        fig.suptitle("Stability and final energy gap summary",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("sigma_data/fig5_stability.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig5_stability.png")

    # ── Fig 6: Phase distribution ──
    _plot_phase_history(res_t, res_x, n_seeds)

    # ── Fig 7: VSNG circuit cost ──
    _plot_circuit_cost(res_t, res_x, n_seeds)

    # ── Fig 8: Ablation heatmap ──
    if ablation_results:
        _plot_ablation_heatmap(ablation_results, E_t or E_x)

    # ── Fig 9: Adam sensitivity ──
    if adam_sweep:
        _plot_adam_sweep(adam_sweep)

    # ── Fig 10: Shot-noise robustness ──
    if shot_results:
        _plot_shot_noise(shot_results)

    # ── Fig 11: Scaling study ──
    if scaling_results:
        _plot_scaling(scaling_results)


def _plot_phase_history(res_t, res_x, n_seeds: int) -> None:
    PHASE_COLS = {
        'active':    '#2ECC71',
        'plateau':   '#E74C3C',
        'local_min': '#F39C12',
        'converged': '#3498DB',
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    for ax, all_res, title in [
            (axes[0], res_t, 'TFIM'), (axes[1], res_x, 'XXZ')]:
        if all_res is None:
            continue
        key = "SIGMA-QGD v8.0"
        if key not in all_res or not all_res[key]:
            continue
        phase_counts = {p: [] for p in PHASE_COLS}
        phase_hists = [r.phase_hist for r in all_res[key] if r.phase_hist]
        if not phase_hists:
            continue
        min_steps = min(len(ph) for ph in phase_hists)
        for si in range(min_steps):
            cnts = {p: 0 for p in PHASE_COLS}
            for r in all_res[key]:
                if si < len(r.phase_hist):
                    ph = r.phase_hist[si]
                    if ph in cnts:
                        cnts[ph] += 1
            tot = max(sum(cnts.values()), 1)
            for p in PHASE_COLS:
                phase_counts[p].append(cnts[p] / tot)
        xs     = np.arange(min_steps)
        bottom = np.zeros(min_steps)
        for ph, col in PHASE_COLS.items():
            vals = np.array(phase_counts[ph])
            ax.fill_between(xs, bottom, bottom + vals,
                            alpha=0.75, color=col, label=ph)
            bottom += vals
        ax.set_ylim(0, 1); ax.set_xlabel("Step")
        ax.set_ylabel("Fraction of seeds")
        ax.set_title(f"Phase distribution — {title}")
        ax.legend(fontsize=7, loc='upper right')
    fig.suptitle("SIGMA-QGD v8.0: Landscape phase distribution across seeds",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sigma_data/fig6_phase_history.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig6_phase_history.png")


def _plot_circuit_cost(res_t, res_x, n_seeds: int) -> None:
    """Fig 7: VSNG adaptive circuit cost per step."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, all_res, title in [
            (axes[0], res_t, 'TFIM'), (axes[1], res_x, 'XXZ')]:
        if all_res is None:
            continue
        key = "SIGMA-QGD v8.0"
        if key not in all_res or not all_res[key]:
            continue
        hists = [r.circuits_hist for r in all_res[key]
                 if r.circuits_hist]
        if not hists:
            continue
        mn  = min(len(h) for h in hists)
        arr = np.array([h[:mn] for h in hists])
        mu  = arr.mean(0); sd = arr.std(0)
        xs  = np.arange(mn)
        ax.plot(xs, mu, color=COLOURS[key], lw=2.5, label="VSNG adaptive")
        ax.fill_between(xs, mu-sd, mu+sd, alpha=0.15, color=COLOURS[key])
        # Reference lines
        p_val = all_res[key][0].n_circuits // max(all_res[key][0].n_steps, 1)
        ax.axhline(2 * len(all_res[key][0].gnorm_hist) + 1 if False else 50,
                   color='gray', ls='--', lw=1.2,
                   label="Full param-shift")
        ax.axhline(3, color='brown', ls=':', lw=1.2, label="SPSA (3)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Circuits used this step")
        ax.set_title(f"VSNG adaptive circuit budget — {title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("VSNG: adaptive circuit cost adapts to landscape phase",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sigma_data/fig7_circuit_cost.png",
                dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig7_circuit_cost.png")


def _plot_ablation_heatmap(ablation_results: Dict, exact_E: float) -> None:
    """Fig 8: Ablation heatmap showing gap for each configuration."""
    configs = list(ablation_results.keys())
    n_configs = len(configs)

    gaps_mean = []
    gaps_std = []
    for cname in configs:
        rs = ablation_results[cname]
        g = [abs(r.energy_opt - exact_E) for r in rs]
        gaps_mean.append(np.mean(g))
        gaps_std.append(np.std(g))

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#00C9A7' if 'Full' in c else '#E24B4A' for c in configs]
    bars = ax.barh(range(n_configs), gaps_mean, xerr=gaps_std,
                   color=colors, alpha=0.8, edgecolor='k', lw=0.5,
                   capsize=4)

    # Add ratio labels
    base_gap = gaps_mean[0]
    for i, (g, c) in enumerate(zip(gaps_mean, configs)):
        ratio = g / max(base_gap, 1e-12)
        label = "1.0× (base)" if 'Full' in c else f"{ratio:.1f}×"
        ax.text(g + gaps_std[i] + 0.001, i, label,
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(n_configs))
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_xlabel("Energy gap |E − E_exact| (H)", fontsize=11)
    ax.set_title("Ablation Study: Component Contribution to Performance",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("sigma_data/fig8_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig8_ablation.png")


def _plot_adam_sweep(adam_sweep: Dict) -> None:
    """Fig 9: Adam sensitivity — gap vs learning rate."""
    if not adam_sweep:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    lrs = sorted(adam_sweep.keys())
    gaps = [adam_sweep[lr] for lr in lrs]

    ax.plot(lrs, gaps, 'o-', color='#E24B4A', lw=2, markersize=10,
            markeredgecolor='k', markeredgewidth=1.5, label='Adam')

    best_idx = np.argmin(gaps)
    ax.plot(lrs[best_idx], gaps[best_idx], '*', color='gold',
            markersize=20, markeredgecolor='k', markeredgewidth=1.5,
            zorder=10, label=f'Best (lr={lrs[best_idx]})')

    ax.set_xlabel("Learning rate α", fontsize=11)
    ax.set_ylabel("Best energy found", fontsize=11)
    ax.set_xscale('log')
    ax.set_title("Adam Hyperparameter Sensitivity",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sigma_data/fig9_adam_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig9_adam_sweep.png")


def _plot_shot_noise(shot_results: Dict) -> None:
    """Fig 10: Shot-noise robustness — gap vs n_shots."""
    if not shot_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Separate noiseless from shot counts
    shot_keys = [k for k in shot_results.keys() if k != 'noiseless']
    shot_keys = sorted(shot_keys)

    gaps = [shot_results[k]['gap_mean'] for k in shot_keys]
    stds = [shot_results[k]['gap_std'] for k in shot_keys]

    ax.errorbar(shot_keys, gaps, yerr=stds, fmt='D-', color='#00C9A7',
                lw=2, markersize=8, capsize=5, markeredgecolor='k',
                markeredgewidth=1.5, label='SIGMA-QGD v8.0')

    # Noiseless reference line
    if 'noiseless' in shot_results:
        nl_gap = shot_results['noiseless']['gap_mean']
        ax.axhline(nl_gap, color='k', ls=':', lw=1.5,
                   label=f'Noiseless ({nl_gap:.4f})')

    ax.set_xlabel("Number of shots", fontsize=11)
    ax.set_ylabel("Energy gap |E − E_exact|", fontsize=11)
    ax.set_xscale('log')
    ax.set_title("Shot-Noise Robustness",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sigma_data/fig10_shot_noise.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig10_shot_noise.png")


def _plot_scaling(scaling_results: Dict) -> None:
    """Fig 11: Scaling study — gap vs n_qubits."""
    if not scaling_results:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    qubits = sorted(scaling_results.keys())
    sigma_gaps = [scaling_results[nq]['sigma_gap_mean'] for nq in qubits]
    sigma_stds = [scaling_results[nq]['sigma_gap_std'] for nq in qubits]
    adam_gaps = [scaling_results[nq]['adam_gap_mean'] for nq in qubits]
    adam_stds = [scaling_results[nq]['adam_gap_std'] for nq in qubits]

    x = np.arange(len(qubits))
    w = 0.3

    ax.bar(x - w/2, sigma_gaps, w, yerr=sigma_stds,
           color='#00C9A7', alpha=0.85, edgecolor='k', lw=0.5,
           capsize=5, label='SIGMA-QGD v8.0')
    ax.bar(x + w/2, adam_gaps, w, yerr=adam_stds,
           color='#E24B4A', alpha=0.85, edgecolor='k', lw=0.5,
           capsize=5, label='Adam (best)')

    # Add ratio labels
    for i, nq in enumerate(qubits):
        ratio = adam_gaps[i] / max(sigma_gaps[i], 1e-12)
        ax.text(i, max(sigma_gaps[i], adam_gaps[i]) +
                max(sigma_stds[i], adam_stds[i]) + 0.02,
                f'{ratio:.1f}×', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{nq}q (p={scaling_results[nq]["p"]})'
                        for nq in qubits], fontsize=10)
    ax.set_xlabel("System size", fontsize=11)
    ax.set_ylabel("Energy gap |E − E_exact| (H)", fontsize=11)
    ax.set_title("Scaling: SIGMA-QGD vs Adam (TFIM, best-tuned)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("sigma_data/fig11_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig11_scaling.png")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SIGMA-QGD v8.0 — Geometry-Inspired Adaptive VQE Optimizer")
    ap.add_argument('--qubits',      type=int,   default=4,
                    help="Number of qubits (default 4)")
    ap.add_argument('--reps',        type=int,   default=2,
                    help="Ansatz repetitions (default 2)")
    ap.add_argument('--seeds',       type=int,   default=5,
                    help="Number of random seeds (default 5)")
    ap.add_argument('--steps',       type=int,   default=200,
                    help="Maximum optimisation steps (default 200)")
    ap.add_argument('--hamiltonian', type=str,   default='both',
                    choices=['tfim', 'xxz', 'both'],
                    help="Hamiltonian to benchmark (default both)")
    ap.add_argument('--eta',         type=float, default=0.05,
                    help="Base learning rate η₀ (default 0.05)")
    ap.add_argument('--lam',         type=float, default=0.01,
                    help="Curvature regularisation λ (default 0.01)")
    ap.add_argument('--tau',         type=float, default=1.0,
                    help="CUSUM allowance τ (default 1.0)")
    ap.add_argument('--no-escape',   action='store_true',
                    help="Disable escape mechanism")
    ap.add_argument('--no-precond',  action='store_true',
                    help="Disable curvature preconditioning")
    ap.add_argument('--no-momentum', action='store_true',
                    help="Disable bias-corrected momentum")
    ap.add_argument('--no-cusum',    action='store_true',
                    help="Disable CUSUM landscape monitor")
    ap.add_argument('--no-vsng',     action='store_true',
                    help="Disable VSNG (use full parameter-shift)")
    ap.add_argument('--ablation',    action='store_true',
                    help="Run ablation study")
    ap.add_argument('--shot-noise',  action='store_true',
                    help="Run shot-noise robustness experiment")
    ap.add_argument('--scaling',     action='store_true',
                    help="Run scaling study across qubit counts")
    ap.add_argument('--quick',       action='store_true',
                    help="Quick run: fewer seeds/steps for testing")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick:
        args.seeds = min(args.seeds, 3)
        args.steps = min(args.steps, 100)

    flags = dict(
        escape   = not args.no_escape,
        precond  = not args.no_precond,
        momentum = not args.no_momentum,
        cusum    = not args.no_cusum,
        vsng     = not args.no_vsng,
        anneal   = True,
        clip     = True,
    )

    print("\n" + "=" * 78)
    print("  SIGMA-QGD v8.0 — Geometry-Inspired Adaptive Optimizer")

    print(f"  Config: {args.qubits}q  reps={args.reps}  seeds={args.seeds}"
          f"  steps={args.steps}")
    print(f"  η₀={args.eta}  λ={args.lam}  τ={args.tau}")
    print(f"  Flags: {flags}")
    print("=" * 78)

    os.makedirs("sigma_data", exist_ok=True)
    logger = ParameterLogger("sigma_data")
    ansatz = build_ansatz(args.qubits, args.reps)

    res_t = sum_t = E_t = None
    res_x = sum_x = E_x = None
    ablation_results = None
    adam_sweep_data = None
    shot_results = None
    scaling_results = None

    if args.hamiltonian in ('tfim', 'both'):
        H_t = build_tfim(args.qubits)
        res_t, sum_t, E_t = run_benchmark(
            H_t, ansatz, "TFIM (J=1, h=0.5)", "tfim",
            args.qubits, args.reps, args.seeds, args.steps, flags, logger)

    if args.hamiltonian in ('xxz', 'both'):
        H_x = build_xxz(args.qubits)
        res_x, sum_x, E_x = run_benchmark(
            H_x, ansatz, "XXZ (Jxy=1, Jz=0.5, h=0.1)", "xxz",
            args.qubits, args.reps, args.seeds, args.steps, flags, logger)

    if args.ablation:
        H_abl = build_tfim(args.qubits)
        abl_ham_type = 'tfim'
        E_abl = exact_gs(H_abl)
        ablation_results = run_ablation(
            H_abl, ansatz, "TFIM", abl_ham_type,
            args.qubits, args.reps, args.steps,
            n_seeds_abl=min(args.seeds, 5), logger=logger)

    if args.shot_noise:
        H_sn = build_tfim(args.qubits)
        shot_results = run_shot_noise_experiment(
            H_sn, ansatz, "TFIM", "tfim",
            args.qubits, args.reps,
            max_steps=min(args.steps, 100),
            n_seeds=min(args.seeds, 3))

    if args.scaling:
        scaling_results = run_scaling_study(
            max_steps=min(args.steps, 150),
            n_seeds=min(args.seeds, 3),
            reps=args.reps)

    logger.save()

    # Fill in missing results for plotting
    if res_t is None:
        res_t = res_x; sum_t = sum_x; E_t = E_x
    if res_x is None:
        res_x = res_t; sum_x = sum_t; E_x = E_t

    # Collect Adam sweep data from first seed if available
    if res_t and "Adam (best)" in res_t:
        # Create simple sweep visualization data
        adam_sweep_data = None  # Populated via benchmark

    print("\n  Generating publication-quality figures ...")
    plot_all(res_t, sum_t, E_t, res_x, sum_x, E_x, args.seeds, args.qubits,
             ablation_results=ablation_results,
             adam_sweep=adam_sweep_data,
             shot_results=shot_results,
             scaling_results=scaling_results)

    print("\n   OUTPUTS ─────")
    print("  sigma_data/fig1_tfim_convergence.png  — TFIM energy + gap")
    print("  sigma_data/fig2_xxz_convergence.png   — XXZ energy + gap")
    print("  sigma_data/fig3_boxplots.png           — final energy distributions")
    print("  sigma_data/fig4_efficiency.png         — circuit efficiency")
    print("  sigma_data/fig5_stability.png          — stability + gap bar")
    print("  sigma_data/fig6_phase_history.png      — phase distribution")
    print("  sigma_data/fig7_circuit_cost.png       — VSNG adaptive circuit budget")
    if ablation_results:
        print("  sigma_data/fig8_ablation.png           — ablation heatmap")
    if shot_results:
        print("  sigma_data/fig10_shot_noise.png        — shot-noise robustness")
    if scaling_results:
        print("  sigma_data/fig11_scaling.png           — scaling study")
    print("  sigma_data/step_records.json/.csv      — full step-level data")
    print("  sigma_data/run_summary.csv             — per-run summary")
    print("  sigma_data/benchmark_*.csv             — benchmark tables")
    if ablation_results:
        print("  sigma_data/ablation_results.csv       — ablation data")
    print("  ───────")
    print("  Ablation:    python Sigma_QGD_code.py --ablation")
    print("  Shot noise:  python Sigma_QGD_code.py --shot-noise")
    print("  Scaling:     python Sigma_QGD_code.py --scaling")
    print("  Quick test:  python Sigma_QGD_code.py --quick")
    print("  Full suite:  python Sigma_QGD_code.py --ablation --shot-noise --scaling")
