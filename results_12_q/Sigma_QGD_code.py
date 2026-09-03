import argparse
"""
SIGMA-QGD v9.0 -- PennyLane/GPU port (lightning.gpu / cuQuantum)

Install:
  pip install pennylane
  pip install pennylane-lightning-gpu   # needs NVIDIA GPU + CUDA + cuQuantum
  # falls back to: pip install pennylane-lightning  (CPU, still fast)
"""

import argparse
import json
import csv
import os
import sys
import time
import math
import warnings

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import pennylane as qml
from scipy.optimize import minimize
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================================================================
#  PennyLane device setup (lightning.gpu / cuQuantum)
# ===================================================================

DISABLE_VSNG: bool = True
BENCHMARK_METHODS: List[str] = ["SIGMA-QGD v9.0", "Adam (best)"]
ABLATION_MAX_SEEDS: int = 5

_GPU_STATUS = "CPU (default.qubit)"


def _make_device(n_qubits: int):

    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        qc = qml.tape.QuantumTape()
        with qc:
            qml.Hadamard(wires=0)
            qml.expval(qml.PauliZ(0))
        try:
            dev.execute(qc)
        except Exception:
            qml.execute([qc], dev)
        return dev, "GPU (lightning.gpu, cuQuantum)"
    except Exception as e_gpu:
        print(f"  [GPU] lightning.gpu unavailable ({e_gpu.__class__.__name__}: {e_gpu}). "
              f"Falling back to lightning.qubit (CPU).")

    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        return dev, "CPU (lightning.qubit)"
    except Exception as e_cpu:
        print(f"  [GPU] lightning.qubit also unavailable ({e_cpu}). "
              f"Falling back to default.qubit.")

    return qml.device("default.qubit", wires=n_qubits), "CPU (default.qubit)"


# ===================================================================
# Visual identity (unchanged)
# ===================================================================

COLOURS = {
    "SIGMA-QGD v9.0": "#00C9A7",
    "Adam (best)":     "#E24B4A",
    "COBYLA":          "#708090",
    "SPSA":            "#8B4513",
    "Diag-QNG":        "#6A0DAD",
    "QN-SPSA":         "#FF6F00",
}
LINES = {
    "SIGMA-QGD v9.0": "-",  "Adam (best)": "-.", "COBYLA": "--",
    "SPSA": ":",             "Diag-QNG": "-",     "QN-SPSA": "--",
}
LW = {
    "SIGMA-QGD v9.0": 3.0, "Adam (best)": 1.8, "COBYLA": 1.5,
    "SPSA": 1.5,            "Diag-QNG": 2.0,    "QN-SPSA": 1.8,
}
MARKERS = {
    "SIGMA-QGD v9.0": "D", "Adam (best)": "X", "COBYLA": "H",
    "SPSA": "p",            "Diag-QNG": "^",    "QN-SPSA": "v",
}


# ===================================================================
# Data logging (unchanged)
# ===================================================================

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

    def finalise_run(self, run_records, final_gap: float, n_steps: int):
        for r in run_records:
            r.final_gap     = final_gap
            r.n_steps_total = n_steps
        self.records.extend(run_records)

    def save(self):
        if not self.records:
            return
        jpath = os.path.join(self.out_dir, "step_records.json")
        with open(jpath, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
        cpath = os.path.join(self.out_dir, "step_records.csv")
        keys  = list(asdict(self.records[0]).keys())
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        print(f"  [Logger] {len(self.records)} records -> {self.out_dir}/")

    def run_summary_csv(self, summaries: List[dict]):
        if not summaries:
            return
        spath = os.path.join(self.out_dir, "run_summary.csv")
        keys  = list(summaries[0].keys())
        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)
        print(f"  [Logger] Run summaries -> {spath}")


# ===================================================================
# Auto-configuration 
# ===================================================================

def derive_config(n_qubits: int, reps: int, max_steps: int,
                  ham_type: str, eta_0: float = 0.05,
                  lambda_reg: float = 0.01, tau: float = 1.0) -> dict:
    p   = n_qubits * reps
    xxz = ham_type.lower() in ("xxz", "heisenberg", "xyz")
    W      = min(30, max(15, max_steps // 10))
    W_med  = max(10, 3 * W)
    t_qfim = min(3 * p, max(10, max_steps // 5))
    beta = float(np.clip(1.0 - 1.0 / np.sqrt(max(p, 4)), 0.5, 0.98))
    clip_norm  = float(np.sqrt(p))
    warmup     = max(int(0.75 * p), max_steps // 5, 15)
    cooldown   = max(10, max_steps // 12)
    stag_win   = max(10, max_steps // 12)
    no_imp_lim = max(60, max_steps // 4)
    k_cache    = 3
    return dict(
        p=p, n_qubits=n_qubits, reps=reps, max_steps=max_steps,
        ham_type=ham_type, xxz=xxz,
        eta_0=eta_0, lambda_reg=lambda_reg, tau=tau,
        W=W, W_med=W_med, t_qfim=t_qfim, beta=beta, eps=1e-8,
        clip_norm=clip_norm, warmup=warmup, cooldown=cooldown,
        stag_win=stag_win, no_imp_lim=no_imp_lim,
        k_cache=k_cache,
    )


# ===================================================================
# Component 1: VarianceEngine 
# ===================================================================

class VarianceEngine:
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

    def update(self, g: np.ndarray):
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
        sigma_bar = float(np.mean(sigma_bessel)) + self.eps
        deviations = sigma_bessel - sigma_bar
        Q = float(np.sum(deviations ** 2)) + self.eps
        shrink_coef = max(0.0, 1.0 - (self.p - 2) * (sigma_bar ** 2) / (Q * self.t))
        self.js_shrunk = np.maximum(sigma_bar + shrink_coef * deviations, 1e-8)
        self.ema_mean = (1.0 - self.alpha) * self.ema_mean + self.alpha * g
        diff_ema = g - self.ema_mean
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * diff_ema ** 2


class DriftCorrectedVarianceEngine(VarianceEngine):
    def __init__(self, p: int, W: int, eps: float = 1e-10,
                 drift_alpha: float = 0.1, drift_floor: float = 1e-10):
        super().__init__(p, W, eps)
        self.drift_alpha  = drift_alpha
        self.drift_floor  = drift_floor
        self.delta_ema    = np.zeros(p)
        self.prev_theta   = None
        self.welford_corrected = np.zeros(p)

    def update_with_theta(self, g: np.ndarray, theta: np.ndarray):
        self.update(g)
        if self.prev_theta is not None:
            delta = theta - self.prev_theta
            self.delta_ema = ((1.0 - self.drift_alpha) * self.delta_ema
                              + self.drift_alpha * delta ** 2)
            drift_var = self.delta_ema * self.welford_raw
            self.welford_corrected = np.maximum(
                self.welford_raw - drift_var, self.drift_floor)
        else:
            self.welford_corrected = self.welford_raw.copy()
        self.prev_theta = theta.copy()

    def update(self, g: np.ndarray):
        super().update(g)
        self.welford_corrected = self.welford_raw.copy()


# ===================================================================
# Component 2: NormalizedCUSUM 
# ===================================================================

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
        self.n_alarms    = 0

    def update(self, ema_var: np.ndarray):
        self.buf.append(ema_var.copy())
        if len(self.buf) < 3:
            return
        arr = np.array(self.buf)
        med = np.median(arr, axis=0)
        z_t = ema_var / (med + self.eps)
        self.S = np.maximum(0.0, self.S + z_t - self.k)
        new_alarms = self.S > self.h
        self.n_alarms += int(np.any(new_alarms & ~self.alarm_state))
        self.alarm_state = new_alarms

    def alarm(self) -> np.ndarray:
        return self.alarm_state.copy()

    def reset(self, mask: np.ndarray):
        self.S[mask] = 0.0

    @property
    def alarm_frac(self) -> float:
        return float(np.mean(self.alarm_state))


# ===================================================================
# Component 3: CurvatureProxy Preconditioner 
# ===================================================================

class CurvatureProxy:
    def __init__(self, eta_0: float, lambda_reg: float, eps: float = 1e-8):
        self.eta_0      = eta_0
        self.lambda_reg = lambda_reg
        self.eps        = eps

    def precondition(self, welford_raw: np.ndarray,
                     gradient: np.ndarray) -> np.ndarray:
        F_diag = 4.0 * welford_raw + self.lambda_reg
        return gradient / (F_diag + self.eps)

    def gamma(self, welford_raw: np.ndarray) -> float:
        F_mean      = float(4.0 * np.mean(welford_raw) + self.lambda_reg)
        gamma_floor = 2.0 * self.eta_0
        return float(max(self.eta_0 / (F_mean + self.eps), gamma_floor))

    def per_dim_sigma(self, welford_raw: np.ndarray, gamma: float) -> np.ndarray:
        F_diag = 4.0 * welford_raw + self.lambda_reg
        return gamma / (np.sqrt(F_diag) + self.eps)


# ===================================================================
# Component 4: VSNG v2 ( DISABLE_VSNG=True)
# ===================================================================

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

    def compute(self, theta, cost_fn, alarm, cusum_S, cusum_h,
                ema_var, E_current) -> np.ndarray:
        g         = np.zeros(self.p)
        n_circ    = 0
        full_alarm  = alarm & (cusum_S >= cusum_h * self.frac)
        transition  = alarm & ~full_alarm
        reliable    = ~alarm
        for i in np.where(reliable)[0]:
            self.steps_since_update[i] += 1
            if self.steps_since_update[i] >= self.k_cache:
                p_plus  = theta.copy(); p_plus[i]  += np.pi / 2
                p_minus = theta.copy(); p_minus[i] -= np.pi / 2
                self.g_cache[i] = 0.5 * (cost_fn(p_plus) - cost_fn(p_minus))
                self.steps_since_update[i] = 0
                n_circ += 2
            g[i] = self.g_cache[i]
        for i in np.where(transition)[0]:
            delta_i = float(np.sqrt(max(ema_var[i], 1e-6)))
            p_pert  = theta.copy(); p_pert[i] += delta_i
            g[i]    = (cost_fn(p_pert) - E_current) / (delta_i + self.eps)
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


# ===================================================================
# Component 5: LandscapeClassifier
# ===================================================================

class LandscapeClassifier:
    def __init__(self, cfg: dict, eta_0: float = 0.05):
        self.stag_win      = cfg["stag_win"]
        self.no_imp_lim    = cfg["no_imp_lim"]
        self.p             = cfg["p"]
        self.q_alpha       = 0.25
        self.gnorm_abs_tol = float(np.sqrt(self.p)) * 0.20
        self.energy_buf    = deque(maxlen=self.stag_win)
        self.progress_win  = deque(maxlen=10)
        self.best_energy   = np.inf
        self.no_imp        = 0
        self.phase_history: List[str] = []
        self.n_plateau   = 0
        self.n_local_min = 0
        self.n_active    = 0

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
            return self._rec("converged")
        if gnorm >= self.gnorm_abs_tol:
            self.n_active += 1
            return self._rec("active")
        recent_best = float(min(self.energy_buf)) if self.energy_buf else energy
        stag_tol = max(1e-4, 1e-3 * abs(recent_best))
        stagnant = (
            len(self.energy_buf) == self.energy_buf.maxlen
            and abs(float(self.energy_buf[-1]) - float(self.energy_buf[0]))
            < stag_tol
        )
        cusum_frac  = float(np.mean(cusum_alarm))
        cusum_fired = cusum_frac > self.q_alpha
        if stagnant and cusum_fired:
            phase = "plateau"
        elif stagnant and not cusum_fired:
            phase = "local_min"
        else:
            phase = "active"
        return self._rec(phase)

    def _rec(self, phase: str) -> str:
        self.phase_history.append(phase)
        if phase == "plateau":
            self.n_plateau += 1
        elif phase == "local_min":
            self.n_local_min += 1
        elif phase == "active":
            self.n_active += 1
        return phase

    def reset_no_imp(self):
        self.no_imp      = 0
        self.best_energy = np.inf


# ===================================================================
# Component 6: SingleDimEscape
# ===================================================================

class SingleDimEscape:
    def __init__(self, cfg: dict, eta_0: float = 0.05,
                 lambda_reg: float = 0.01, T_scale: float = 8.0,
                 sigma_scale: float = 0.08):
        self.warmup     = cfg["warmup"]
        self.cooldown   = cfg["cooldown"]
        self.t_half     = max(cfg["max_steps"] * 3 // 4, 50)
        self.last_esc   = -cfg["cooldown"]
        self.eta_0      = eta_0
        self.lambda_reg = lambda_reg
        self.eps        = cfg.get("eps", 1e-8)
        self.T_scale    = T_scale
        self.sigma_scale = sigma_scale
        self.min_stuck  = cfg["cooldown"]
        self.escapes_a  = 0
        self.escape_log: List[dict] = []

    def _temperature(self, welford_raw: np.ndarray, step: int) -> float:
        F_mean = float(4.0 * np.mean(welford_raw) + self.lambda_reg)
        T_base = self.T_scale * self.eta_0 / (F_mean + self.eps)
        decay  = 1.0 / (1.0 + step / (self.t_half + self.eps))
        return float(T_base * decay)

    def escape(self, theta, phase, no_imp, cusum_alarm, welford_raw,
               cost_fn, cost, sigma, step):
        if step <= self.warmup:
            return theta, cost, False
        if (step - self.last_esc) < self.cooldown:
            return theta, cost, False
        if phase not in ("plateau", "local_min"):
            return theta, cost, False
        if no_imp < self.min_stuck:
            return theta, cost, False
        if not np.any(cusum_alarm):
            return theta, cost, False
        T = self._temperature(welford_raw, step)
        alarmed_idx = np.where(cusum_alarm)[0]
        i           = int(np.random.choice(alarmed_idx))
        direction   = float(np.sign(np.random.randn()))
        amp         = self.sigma_scale * float(sigma[i])
        cand        = theta.copy()
        cand[i]     = theta[i] + direction * amp
        new_cost    = cost_fn(cand)
        dc          = new_cost - cost
        accepted    = dc < 0 or np.random.rand() < np.exp(-dc / (T + 1e-30))
        self.escape_log.append(dict(step=step, dim=i,
                                    dc=round(float(dc), 6),
                                    amp=round(amp, 6), accepted=accepted))
        if accepted:
            self.last_esc  = step
            self.escapes_a += 1
            return cand, new_cost, True
        return theta, cost, False

    def summary(self) -> dict:
        n_tried = len(self.escape_log)
        n_acc   = sum(1 for e in self.escape_log if e["accepted"])
        rate    = (n_acc / n_tried) if n_tried > 0 else 0.0
        return dict(escapes_A=self.escapes_a, n_tried=n_tried,
                    accept_rate=round(rate, 4))


# ===================================================================
# Hamiltonians ( SparsePauliOp -> qml.Hamiltonian)
# ===================================================================

def build_tfim(n: int, J: float = 1.0, h: float = 0.5) -> qml.Hamiltonian:
    coeffs, ops = [], []
    for i in range(n):
        j = (i + 1) % n
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    for i in range(n):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, ops)


def build_xxz(n: int, Jxy: float = 1.0,
              Jz: float = 0.5, h: float = 0.1) -> qml.Hamiltonian:
    coeffs, ops = [], []
    for i in range(n):
        j = (i + 1) % n
        coeffs += [Jxy, Jxy, Jz]
        ops += [qml.PauliX(i) @ qml.PauliX(j),
                qml.PauliY(i) @ qml.PauliY(j),
                qml.PauliZ(i) @ qml.PauliZ(j)]
    for i in range(n):
        coeffs.append(h)
        ops.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, ops)


def exact_gs(H: qml.Hamiltonian) -> float:
    mat = qml.matrix(H)
    return float(np.min(np.linalg.eigvalsh(mat)))


# ===================================================================
# Ansatz
# ===================================================================

def ansatz_fn(theta, n: int, reps: int):
    idx = 0
    for _ in range(reps):
        for q in range(n):
            qml.RY(theta[idx], wires=q)
            idx += 1
        for q in range(n - 1):
            qml.CNOT(wires=[q, q + 1])


def build_ansatz(n: int, reps: int):
    """Kept for API parity with the Qiskit version -- just returns (n, reps)."""
    return (n, reps)


# ===================================================================
# CostEvaluator ( lightning.gpu device + batched param-shift)
# ===================================================================

class CostEvaluator:
    """
    Wraps a PennyLane device (lightning.gpu / lightning.qubit / default.qubit);
    counts circuit evaluations.

     cost_and_gradient(theta) batches the energy eval + all 2p
    parameter-shift circuits into ONE device call (list of QuantumTapes
    handed to device.batch_execute / qml.execute), mirroring the Aer
    Estimator batching trick -- this is what lets the GPU amortize kernel
    launch overhead across many circuits instead of doing p+1 sequential
    small statevector evals.
    """

    def __init__(self, n_qubits: int, reps: int, H: qml.Hamiltonian, device=None):
        self.n_qubits = n_qubits
        self.reps     = reps
        self.H        = H
        self.n_calls  = 0
        self.device, self.status = (device, None) if device is not None \
            else _make_device(n_qubits)

    def _tape(self, theta: np.ndarray) -> qml.tape.QuantumTape:
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(theta, self.n_qubits, self.reps)
            qml.expval(self.H)
        return tape

    def _batch_execute(self, tapes):
        # PennyLane API differs slightly across versions; try both entry points.
        try:
            return self.device.batch_execute(tapes)
        except AttributeError:
            return qml.execute(tapes, self.device, diff_method=None)

    def __call__(self, theta: np.ndarray) -> float:
        self.n_calls += 1
        result = self._batch_execute([self._tape(theta)])
        return float(np.asarray(result[0]))

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        """Batched parameter-shift gradient: one device call for all 2p circuits."""
        p = len(theta)
        tapes = []
        for i in range(p):
            tp = theta.copy(); tp[i] += np.pi / 2
            tm = theta.copy(); tm[i] -= np.pi / 2
            tapes.append(self._tape(tp))
            tapes.append(self._tape(tm))
        results = self._batch_execute(tapes)
        self.n_calls += len(tapes)
        g = np.zeros(p)
        for i in range(p):
            ev_plus  = float(np.asarray(results[2 * i]))
            ev_minus = float(np.asarray(results[2 * i + 1]))
            g[i] = 0.5 * (ev_plus - ev_minus)
        return g

    def cost_and_gradient(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
         energy + full gradient in ONE batched call (2p+1 circuits).
        """
        p = len(theta)
        tapes = [self._tape(theta)]
        for i in range(p):
            tp = theta.copy(); tp[i] += np.pi / 2
            tm = theta.copy(); tm[i] -= np.pi / 2
            tapes.append(self._tape(tp))
            tapes.append(self._tape(tm))
        results = self._batch_execute(tapes)
        self.n_calls += len(tapes)
        cost = float(np.asarray(results[0]))
        g = np.zeros(p)
        for i in range(p):
            ev_plus  = float(np.asarray(results[1 + 2 * i]))
            ev_minus = float(np.asarray(results[2 + 2 * i]))
            g[i] = 0.5 * (ev_plus - ev_minus)
        return cost, g

    def overlap_call(self, sv_base: np.ndarray,
                     theta_shift: np.ndarray) -> float:
        """Diag-QNG overlap. Not on the batched GPU path (single small
        statevector op), same tradeoff as the qiskit version's Statevector use."""
        self.n_calls += 1
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(theta_shift, self.n_qubits, self.reps)
            qml.state()
        sv_shift = np.asarray(self._batch_execute([tape])[0])
        return float(abs(np.vdot(sv_base, sv_shift)) ** 2)

    def statevector(self, theta: np.ndarray) -> np.ndarray:
        """Helper: get the raw statevector (used by Diag-QNG for sv_base)."""
        self.n_calls += 1
        with qml.tape.QuantumTape() as tape:
            ansatz_fn(theta, self.n_qubits, self.reps)
            qml.state()
        return np.asarray(self._batch_execute([tape])[0])


class ShotNoiseCostEvaluator:
    """
     instead of manually injecting Gaussian shot noise on top of an
    exact expectation (as the Qiskit version did via a variance estimate),
    PennyLane devices natively support `shots=` for statistically sampled
    expectation values. We create a shots-enabled device on the same
    backend (GPU if available) and let the device itself produce the
    noisy estimate -- more faithful than reconstructing noise manually.
    """

    def __init__(self, n_qubits: int, reps: int, H: qml.Hamiltonian,
                 n_shots: int = 1024):
        self.n_qubits = n_qubits
        self.reps     = reps
        self.H        = H
        self.n_shots  = n_shots
        self.n_calls  = 0
        try:
            self.device = qml.device("lightning.gpu", wires=n_qubits, shots=n_shots)
        except Exception:
            try:
                self.device = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)
            except Exception:
                self.device = qml.device("default.qubit", wires=n_qubits, shots=n_shots)
        self.base = CostEvaluator(n_qubits, reps, H, device=self.device)

    def __call__(self, theta: np.ndarray) -> float:
        self.n_calls += 1
        val = self.base(theta)
        self.base.n_calls -= 1
        return val

    def gradient(self, theta: np.ndarray) -> np.ndarray:
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            p_plus  = theta.copy(); p_plus[i]  += np.pi / 2
            p_minus = theta.copy(); p_minus[i] -= np.pi / 2
            g[i] = 0.5 * (self(p_plus) - self(p_minus))
        return g

    def cost_and_gradient(self, theta: np.ndarray) -> Tuple[float, np.ndarray]:
        cost = self(theta)
        g = self.gradient(theta)
        return cost, g


# ===================================================================
# Result container 
# ===================================================================

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


# ===================================================================
# Annealing Schedules 
# ===================================================================

class PlateauTriggeredSchedule:
    def __init__(self, eta_0: float, anneal_threshold: float = 1e-4,
                 anneal_duration: int = 50, ema_alpha: float = 0.1):
        self.eta_0            = eta_0
        self.anneal_threshold = anneal_threshold
        self.anneal_duration  = anneal_duration
        self.ema_alpha        = ema_alpha
        self.ema_improvement  = float("inf")
        self.triggered_at     = None
        self.prev_energy      = None

    def step(self, t: int, energy: float, warmup: int) -> float:
        if t <= warmup or self.prev_energy is None:
            self.prev_energy = energy
            return self.eta_0
        improvement = abs(energy - self.prev_energy)
        self.prev_energy = energy
        if self.ema_improvement == float("inf"):
            self.ema_improvement = improvement
        else:
            self.ema_improvement = ((1.0 - self.ema_alpha) * self.ema_improvement
                                    + self.ema_alpha * improvement)
        if self.triggered_at is None:
            if self.ema_improvement < self.anneal_threshold:
                self.triggered_at = t
        if self.triggered_at is None:
            return self.eta_0
        steps_since = t - self.triggered_at
        frac = min(1.0, steps_since / max(self.anneal_duration, 1))
        eta = self.eta_0 * 0.5 * (1.0 + math.cos(math.pi * frac))
        return max(eta, 1e-4)


# ===================================================================
# SIGMA-QGD v9.0 Main Optimizer 
# ===================================================================

def run_sigma_qgd_v9(cost_fn, theta_init: np.ndarray,
                     n_qubits: int, reps: int,
                     max_steps: int = 200, ham_type: str = "tfim",
                     eta_0: float = 0.05, lambda_reg: float = 0.01,
                     tau: float = 1.0,
                     use_escape: bool = False,
                     use_precond: bool = True,
                     use_momentum: bool = True,
                     use_cusum: bool = False,
                     use_vsng: bool = True,
                     use_anneal: bool = False,
                     use_clip: bool = True,
                     use_drift_correction: bool = False,
                     logger: ParameterLogger = None,
                     seed: int = 0):
    if DISABLE_VSNG:
        use_vsng = False

    t0  = time.time()
    cfg = derive_config(n_qubits, reps, max_steps, ham_type,
                        eta_0, lambda_reg, tau)
    p   = cfg["p"]

    if use_drift_correction:
        var_eng = DriftCorrectedVarianceEngine(p, cfg["W"])
    else:
        var_eng = VarianceEngine(p, cfg["W"])

    cusum     = NormalizedCUSUM(p, cfg["W_med"], tau, h_factor=5.0)
    curvature = CurvatureProxy(eta_0, lambda_reg)
    lscape    = LandscapeClassifier(cfg, eta_0)
    escape    = SingleDimEscape(cfg, eta_0, lambda_reg) if use_escape else None
    vsng      = VarianceSignalNaturalGradient(p, k_cache=cfg["k_cache"])
    anneal_sched = PlateauTriggeredSchedule(eta_0) if use_anneal else None

    theta    = theta_init.copy()
    m        = np.zeros(p)
    beta     = cfg["beta"]
    eps      = cfg["eps"]

    best_e     = np.inf
    best_theta = theta.copy()
    n_restarts = 0
    max_restarts = 3

    energy_hist: List[float] = []
    gnorm_hist:  List[float] = []
    circ_hist:   List[int]   = []
    run_records: List[StepRecord] = []

    for t in range(1, max_steps + 1):
        n_step_circs = 0
        alarm        = cusum.alarm() if use_cusum else np.zeros(p, dtype=bool)

        if t <= cfg["warmup"] or not use_vsng:
            cost, g       = cost_fn.cost_and_gradient(theta)  # >>> GPU: batched
            n_step_circs += 2 * p + 1
        else:
            cost          = cost_fn(theta)
            n_step_circs += 1
            g = vsng.compute(
                theta=theta, cost_fn=cost_fn, alarm=alarm,
                cusum_S=cusum.S, cusum_h=cusum.h,
                ema_var=var_eng.ema, E_current=cost)
            n_step_circs += vsng.n_circuits_last

        gnorm = float(np.linalg.norm(g))
        cn    = cfg["clip_norm"]
        gc    = g * cn / (gnorm + 1e-30) if (use_clip and gnorm > cn) else g.copy()

        if use_drift_correction and isinstance(var_eng, DriftCorrectedVarianceEngine):
            var_eng.update_with_theta(gc, theta)
            welford_for_precond = var_eng.welford_corrected
        else:
            var_eng.update(gc)
            welford_for_precond = var_eng.welford_raw

        if use_cusum:
            cusum.update(var_eng.ema)
            alarm = cusum.alarm()

        eta = (anneal_sched.step(t, cost, cfg["warmup"])
               if (use_anneal and anneal_sched is not None) else eta_0)

        if use_momentum:
            m    = beta * m + (1.0 - beta) * gc
            mhat = m / (1.0 - beta ** t)
        else:
            mhat = gc

        if use_precond and t >= cfg["t_qfim"]:
            d = curvature.precondition(welford_for_precond, mhat)
        else:
            d = mhat

        theta = theta - eta * d
        phase = lscape.classify(cost, gnorm, alarm)

        if cost < best_e:
            best_e     = cost
            best_theta = theta.copy()

        escaped     = False
        escape_mode = ""
        if use_escape and escape is not None:
            gamma_val = curvature.gamma(welford_for_precond)
            sigma_vec = curvature.per_dim_sigma(welford_for_precond, gamma_val)
            new_theta, new_cost, escaped = escape.escape(
                theta=theta, phase=phase, no_imp=lscape.no_imp,
                cusum_alarm=alarm, welford_raw=welford_for_precond,
                cost_fn=cost_fn, cost=cost, sigma=sigma_vec, step=t)
            if escaped:
                n_step_circs += 1
                theta  = new_theta; cost = new_cost
                escape_mode = "A"
                lscape.reset_no_imp()
                if use_cusum: cusum.reset(alarm)
                if cost < best_e: best_e = cost; best_theta = theta.copy()

        restarted = False
        if lscape.no_imp >= cfg["no_imp_lim"] and n_restarts < max_restarts:
            theta  = best_theta + 0.05 * np.random.randn(p)
            m      = np.zeros(p)
            var_eng = (DriftCorrectedVarianceEngine(p, cfg["W"])
                       if use_drift_correction else VarianceEngine(p, cfg["W"]))
            cusum   = NormalizedCUSUM(p, cfg["W_med"], tau, h_factor=5.0)
            vsng    = VarianceSignalNaturalGradient(p, k_cache=cfg["k_cache"])
            lscape.reset_no_imp()
            lscape.energy_buf.clear()
            lscape.progress_win.clear()
            n_restarts += 1; restarted = True

        energy_hist.append(cost)
        gnorm_hist.append(gnorm)
        circ_hist.append(n_step_circs)

        if logger is not None:
            gamma_val = curvature.gamma(welford_for_precond)
            run_records.append(StepRecord(
                seed=seed, step=t, ham_type=ham_type,
                n_qubits=n_qubits, reps=reps,
                eta_0=eta_0, lambda_reg=lambda_reg, tau=tau,
                energy=float(cost), gnorm=float(gnorm), phase=phase,
                cusum_alarm_frac=float(np.mean(alarm)),
                welford_mean=float(np.mean(var_eng.welford_raw)),
                js_mean=float(np.mean(var_eng.js_shrunk)),
                ema_mean=float(np.mean(var_eng.ema)),
                momentum_norm=float(np.linalg.norm(mhat)),
                curvature_gamma=float(gamma_val),
                n_circuits_step=n_step_circs,
                escaped=escaped, escape_mode=escape_mode, restarted=restarted))

        if t > cfg["W"] and len(energy_hist) > 2*cfg["W"]:
            recent_best = min(energy_hist[-cfg["W"]:])
            prev_best   = min(energy_hist[-2*cfg["W"]:-cfg["W"]])
            if abs(recent_best - prev_best) < 1e-5:
                break

    ea   = escape.escapes_a if escape else 0
    elog = escape.escape_log if escape else []
    res  = Result(method="SIGMA-QGD v9.0", energy_opt=best_e,
                  energy_hist=energy_hist, gnorm_hist=gnorm_hist,
                  n_circuits=cost_fn.n_calls, n_steps=len(energy_hist),
                  escapes_a=ea, restarts=n_restarts,
                  phase_hist=lscape.phase_history.copy(), escape_log=elog,
                  circuits_hist=circ_hist, wall_time=time.time() - t0)
    return res, run_records, lscape, (cusum if use_cusum else None), \
           (escape if use_escape else None)


def run_sigma_qgd_v8(cost_fn, theta_init, n_qubits, reps,
                     max_steps=200, ham_type="tfim", eta_0=0.05,
                     lambda_reg=0.01, tau=1.0,
                     use_escape=False, use_precond=True, use_momentum=True,
                     use_cusum=False, use_vsng=True, use_anneal=False,
                     use_clip=True, use_drift_correction=False,
                     logger=None, seed=0):
    tup = run_sigma_qgd_v9(cost_fn, theta_init, n_qubits, reps,
                           max_steps, ham_type, eta_0, lambda_reg, tau,
                           use_escape, use_precond, use_momentum, use_cusum,
                           use_vsng, use_anneal, use_clip, use_drift_correction,
                           logger, seed)
    return tup[0], tup[1]


# ===================================================================
# Baselines 
# ===================================================================

def run_adam(cost_fn, theta_init: np.ndarray, max_steps: int = 200,
             lr: float = 0.01, label: str = "Adam"):
    theta = theta_init.copy(); p = len(theta)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    m = np.zeros(p); v = np.zeros(p)
    best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()
    for t in range(1, max_steps + 1):
        cost, g = cost_fn.cost_and_gradient(theta)   # >>> GPU: batched call
        m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g**2
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        theta -= lr * mh / (np.sqrt(vh) + eps)
        gnorm = float(np.linalg.norm(g)); eh.append(cost); gh.append(gnorm)
        ch.append(2*p + 1)
        if cost < best_e: best_e = cost
        if gnorm < 1e-6: break
    return Result(label, best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


def run_adam_sweep(cost_fn_factory, theta_init: np.ndarray, max_steps: int = 200):
    lrs = [0.01, 0.02, 0.05]
    best_result = None; best_gap = np.inf; sweep_results = {}
    for lr in lrs:
        cf = cost_fn_factory()
        r, _ = run_adam(cf, theta_init.copy(), max_steps, lr=lr, label=f"Adam(lr={lr})")
        sweep_results[lr] = r.energy_opt
        if r.energy_opt < best_gap:
            best_gap = r.energy_opt; best_result = r
            best_result.method = "Adam (best)"
    return best_result, [], sweep_results


def run_cobyla(cost_fn, theta_init: np.ndarray, max_steps: int = 200):
    best_e = [np.inf]; eh = []; gh = []; t0 = time.time()
    def obj(x):
        cost = cost_fn(x); eh.append(cost); gh.append(0.0)
        if cost < best_e[0]: best_e[0] = cost
        return cost
    minimize(obj, theta_init.copy(), method="COBYLA",
             options={"maxiter": max_steps, "rhobeg": 0.1})
    return Result("COBYLA", best_e[0], eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=[1]*len(eh), wall_time=time.time()-t0), []


def run_spsa(cost_fn, theta_init: np.ndarray, max_steps: int = 200):
    theta = theta_init.copy(); p = len(theta)
    a = 0.1; c = 0.1; A = max_steps / 10.0; alpha = 0.602; gamma = 0.101
    best_e = np.inf; eh = []; gh = []; ch = []; t0 = time.time()
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


def run_diag_qng(cost_fn, theta_init: np.ndarray, max_steps: int = 200,
                 lr: float = 0.05, lambda_reg: float = 0.01):
    theta = theta_init.copy(); p = len(theta)
    best_e = np.inf; eh = []; gh = []; ch = []; t0 = time.time()

    for t in range(1, max_steps + 1):
        cost  = cost_fn(theta)
        g     = cost_fn.gradient(theta)
        gnorm = float(np.linalg.norm(g))

        sv_base = cost_fn.statevector(theta)   # >>> GPU: PennyLane statevector

        F_diag = np.ones(p)
        for i in range(p):
            theta_shift    = theta.copy(); theta_shift[i] += np.pi
            overlap        = cost_fn.overlap_call(sv_base, theta_shift)
            F_diag[i]      = max(1.0 - overlap, 1e-8)

        d = g / (F_diag + lambda_reg)
        theta -= lr * d

        eh.append(cost); gh.append(gnorm)
        ch.append(3 * p + 2)  
        if cost < best_e: best_e = cost
        if gnorm < 1e-6: break

    result = Result("Diag-QNG", best_e, eh, gh, cost_fn.n_calls, len(eh),
                    circuits_hist=ch, wall_time=time.time()-t0)
    return result, []


def run_qn_spsa(cost_fn, theta_init: np.ndarray, max_steps: int = 200,
                lr: float = 0.05, lambda_reg: float = 0.01):
    theta = theta_init.copy(); p = len(theta)
    a = lr; c = 0.1; A = max_steps / 10.0; alpha_sp = 0.602; gamma_sp = 0.101
    beta_metric = 0.01
    F_diag = np.ones(p) * 0.5; best_e = np.inf; eh = []; gh = []; ch = []
    t0 = time.time()
    for k in range(1, max_steps + 1):
        ak = a / (A + k)**alpha_sp; ck = c / k**gamma_sp
        delta1 = np.random.choice([-1.0, 1.0], size=p)
        fp1 = cost_fn(theta + ck*delta1); fm1 = cost_fn(theta - ck*delta1)
        g_sp = (fp1 - fm1) / (2.0 * ck * delta1)
        delta2 = np.random.choice([-1.0, 1.0], size=p)
        fp2 = cost_fn(theta + ck*delta2); fm2 = cost_fn(theta - ck*delta2)
        g_sp2 = (fp2 - fm2) / (2.0 * ck * delta2)
        F_diag = (1 - beta_metric)*F_diag + beta_metric*np.abs(g_sp*g_sp2)
        F_diag = np.maximum(F_diag, 1e-8)
        d = g_sp / (F_diag + lambda_reg)
        theta -= ak * d
        cost = cost_fn(theta)
        gnorm = float(np.linalg.norm(g_sp)); eh.append(cost); gh.append(gnorm)
        ch.append(5)
        if cost < best_e: best_e = cost
    return Result("QN-SPSA", best_e, eh, gh, cost_fn.n_calls, len(eh),
                  circuits_hist=ch, wall_time=time.time()-t0), []


# ===================================================================
# Significance testing (unchanged)
# ===================================================================

def compute_significance_tests(all_res: dict, exact_E: float,
                                ham_name: str, out_dir: str = "sigma_data"):
    sigma_key = "SIGMA-QGD v9.0"
    if sigma_key not in all_res or not all_res[sigma_key]:
        print(f"  [Significance] No {sigma_key} results for {ham_name}, skipping.")
        return

    sigma_gaps = [abs(r.energy_opt - exact_E) for r in all_res[sigma_key]]
    n_sigma    = len(sigma_gaps)

    rows = []
    print(f"\n  Wilcoxon signed-rank tests vs {sigma_key} [{ham_name}]:")
    print(f"  {'Baseline':<22} {'stat':>8} {'p-val':>10} {'mean_diff':>12} {'n':>4} {'direction'}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*12} {'─'*4} {'─'*20}")

    for name, res_list in all_res.items():
        if name == sigma_key or not res_list:
            continue
        base_gaps = [abs(r.energy_opt - exact_E) for r in res_list]
        n_matched = min(n_sigma, len(base_gaps))
        if n_matched < 3:
            print(f"  {name:<22} SKIP (n<3)")
            continue
        s_gaps = np.array(sigma_gaps[:n_matched])
        b_gaps = np.array(base_gaps[:n_matched])
        diffs  = s_gaps - b_gaps
        try:
            stat, pval = scipy_stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            stat, pval = 0.0, 1.0
        mean_diff = float(np.mean(diffs))
        direction = "SIGMA better" if mean_diff < 0 else "Baseline better"
        print(f"  {name:<22} {stat:>8.2f} {pval:>10.4f} {mean_diff:>12.6f} "
              f"{n_matched:>4}  {direction}" + (" *" if pval < 0.05 else ""))
        rows.append(dict(ham=ham_name, baseline=name,
                         statistic=round(float(stat), 4),
                         p_value=round(float(pval), 6),
                         mean_diff=round(mean_diff, 6),
                         n=n_matched, direction=direction))

    os.makedirs(out_dir, exist_ok=True)
    sig_path = os.path.join(out_dir, "significance_tests.csv")
    write_header = not os.path.exists(sig_path)
    with open(sig_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ham", "baseline", "statistic",
                                           "p_value", "mean_diff", "n", "direction"])
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  [Significance] Saved -> {sig_path}")


# ===================================================================
# Component activation warning system 
# ===================================================================

def emit_component_warnings(lscape_list: list, cusum_list: list,
                            escape_list: list, n_total_steps: int):
    total_escapes      = sum(getattr(e, "escapes_a", 0) for e in escape_list if e)
    total_plateaus     = sum(getattr(l, "n_plateau", 0) for l in lscape_list if l)
    total_local_min    = sum(getattr(l, "n_local_min", 0) for l in lscape_list if l)
    total_cusum_alarms = sum(getattr(c, "n_alarms", 0) for c in cusum_list if c)

    warnings_emitted = []
    if escape_list and any(e is not None for e in escape_list):
        if total_escapes == 0:
            warnings_emitted.append(
                "[WARNING] Escape mechanism was NEVER exercised in this run "
                f"(0 escapes across {len(escape_list)} seed(s)). "
                "Ablation result for 'No escape' is NOT meaningful -- "
                "removing escape had no measurable effect because it never fired.")
    if cusum_list and any(c is not None for c in cusum_list):
        if total_cusum_alarms == 0:
            warnings_emitted.append(
                "[WARNING] CUSUM was NEVER alarmed in this run "
                f"(0 alarm events across {len(cusum_list)} seed(s)). "
                "Ablation result for 'No CUSUM' is NOT meaningful.")
    if lscape_list:
        if total_plateaus == 0 and total_local_min == 0:
            warnings_emitted.append(
                "[WARNING] LandscapeClassifier classified ALL steps as 'active' "
                f"(n_steps={n_total_steps}). Plateau/local_min never reached. "
                "Components gated on these phases (escape, CUSUM reset) "
                "were never exercised. Check stag_tol calibration.")

    if warnings_emitted:
        print(f"\n  {'='*70}")
        for w in warnings_emitted:
            print(f"  {w}")
        print(f"  {'='*70}\n")
    else:
        if total_escapes > 0 or total_cusum_alarms > 0:
            print(f"\n  [OK] Components exercised: "
                  f"escapes={total_escapes}, "
                  f"CUSUM alarms={total_cusum_alarms}, "
                  f"plateaus={total_plateaus}, "
                  f"local_mins={total_local_min}")


# ===================================================================
# Benchmark runner ( CostEvaluator now takes n_qubits/reps/H)
# ===================================================================

def _get_runner(name, n_qubits, reps, max_steps, ham_type, flags, logger, seed):
    if name == "SIGMA-QGD v9.0":
        def _run(cf, th):
            return run_sigma_qgd_v9(
                cf, th, n_qubits=n_qubits, reps=reps,
                max_steps=max_steps, ham_type=ham_type,
                use_escape=flags.get("escape", False),
                use_precond=flags.get("precond", True),
                use_momentum=flags.get("momentum", True),
                use_cusum=flags.get("cusum", False),
                use_vsng=flags.get("vsng", True),
                use_anneal=flags.get("anneal", False),
                use_clip=flags.get("clip", True),
                use_drift_correction=flags.get("drift_correction", False),
                logger=logger, seed=seed)
        return _run
    runners = {
        "COBYLA": lambda cf, th, **kw: run_cobyla(cf, th, **kw),
        "SPSA":   lambda cf, th, **kw: run_spsa(cf, th, **kw),
        "Diag-QNG": lambda cf, th, **kw: run_diag_qng(cf, th, **kw),
        "QN-SPSA":  lambda cf, th, **kw: run_qn_spsa(cf, th, **kw),
    }
    if name in runners:
        def _run(cf, th):
            return runners[name](cf, th, max_steps=max_steps)
        return _run
    return None


def run_benchmark(H: qml.Hamiltonian, ansatz_spec, ham_name: str, ham_type: str,
                  n_qubits: int, reps: int,
                  n_seeds: int, max_steps: int,
                  flags: dict, logger: ParameterLogger,
                  methods: List[str] = None):
    exact_E  = exact_gs(H)
    if methods is None:
        methods = list(BENCHMARK_METHODS)
    all_res  = {m: [] for m in methods}
    cfg      = derive_config(n_qubits, reps, max_steps, ham_type)

    print(f"\n{'='*78}")
    print(f"  {ham_name}  |  n={n_qubits}  p={cfg['p']}  |  Exact: {exact_E:.4f} H")
    print(f"  W={cfg['W']}  W_med={cfg['W_med']}  t_qfim={cfg['t_qfim']}"
          f"  beta={cfg['beta']:.3f}  clip={cfg['clip_norm']:.2f}"
          f"  warmup={cfg['warmup']}  k_cache={cfg['k_cache']}")
    print(f"  [Backend] {_GPU_STATUS}  |  methods={methods}  |  VSNG disabled={DISABLE_VSNG}")
    print(f"{'='*78}")

    run_summaries = []
    all_lscapes = []; all_cusums = []; all_escapes = []
    total_steps = 0
    n_params = n_qubits * reps

    for seed in range(n_seeds):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, n_params)
        print(f"\n  Seed {seed}  |theta0|={np.linalg.norm(theta0):.3f}")

        for name in methods:
            print(f"  {name:<22} ...", end=" ", flush=True)

            if name == "Adam (best)":
                def cf_factory(nq=n_qubits, r=reps, H_=H):
                    return CostEvaluator(nq, r, H_)
                r, recs, sweep = run_adam_sweep(cf_factory, theta0.copy(), max_steps)
            else:
                cf     = CostEvaluator(n_qubits, reps, H)
                runner = _get_runner(name, n_qubits, reps, max_steps,
                                     ham_type, flags, logger, seed)
                if runner is None:
                    print("SKIP (unknown method)"); continue
                result_tuple = runner(cf, theta0.copy())
                r, recs = result_tuple[0], result_tuple[1]
                if name == "SIGMA-QGD v9.0" and len(result_tuple) > 2:
                    all_lscapes.append(result_tuple[2])
                    all_cusums.append(result_tuple[3])
                    all_escapes.append(result_tuple[4])
                    total_steps += r.n_steps

            final_gap = abs(r.energy_opt - exact_E)
            if recs and logger:
                logger.finalise_run(recs, final_gap, r.n_steps)
            all_res[name].append(r)

            circ_info = ""
            if "SIGMA" in name and r.circuits_hist:
                circ_info = (f"  c/step={np.mean(r.circuits_hist):.1f}"
                             f"  esc={r.escapes_a}  rst={r.restarts}")
            print(f"E={r.energy_opt:.4f}  gap={final_gap:.5f}"
                  f"  circ={r.n_circuits:,}  steps={r.n_steps}"
                  f"  t={r.wall_time:.1f}s{circ_info}")
            run_summaries.append(dict(
                ham=ham_name, method=name, seed=seed,
                energy_opt=round(r.energy_opt, 6),
                exact_E=round(exact_E, 6), gap=round(final_gap, 6),
                n_circuits=r.n_circuits, n_steps=r.n_steps,
                wall_time=round(r.wall_time, 3)))

    if logger:
        logger.run_summary_csv(run_summaries)

    summary = {}
    for name in methods:
        if not all_res[name]: continue
        energies = [r.energy_opt for r in all_res[name]]
        gaps     = [abs(e - exact_E) for e in energies]
        summary[name] = dict(
            energy_mean=float(np.mean(energies)), energy_std=float(np.std(energies)),
            energy_all=energies, gap_mean=float(np.mean(gaps)),
            gap_std=float(np.std(gaps)),
            steps_mean=float(np.mean([r.n_steps for r in all_res[name]])),
            circuits_mean=float(np.mean([r.n_circuits for r in all_res[name]])),
            circuits_all=[r.n_circuits for r in all_res[name]],
            time_mean=float(np.mean([r.wall_time for r in all_res[name]])))

    compute_significance_tests(all_res, exact_E, ham_name)
    _print_table(summary, exact_E, ham_name, n_seeds)
    _save_benchmark_table(summary, exact_E, ham_name, n_seeds)
    emit_component_warnings(all_lscapes, all_cusums, all_escapes, total_steps)

    return all_res, summary, exact_E


def _print_table(summary: dict, exact_E: float, ham_name: str, n_seeds: int):
    v9_gap = summary.get("SIGMA-QGD v9.0", {}).get("gap_mean", float("nan"))
    print(f"\n{'='*78}")
    print(f"  RESULTS -- {ham_name}  |  Exact: {exact_E:.6f} H  |  {n_seeds} seeds")
    print(f"  (See significance_tests.csv for Wilcoxon p-values per comparison)")
    print(f"  {'Method':<22} {'E mean+/-std':>20}  {'Gap':>9}  {'Steps':>7}  {'Circuits':>10}")
    print(f"  {'─'*22} {'─'*20}  {'─'*9}  {'─'*7}  {'─'*10}")
    for name, s in sorted(summary.items(), key=lambda x: x[1]["gap_mean"]):
        tag = " *" if name == "SIGMA-QGD v9.0" else ""
        imp = ""
        if name != "SIGMA-QGD v9.0" and s["gap_mean"] > 0 and not np.isnan(v9_gap):
            ratio = s["gap_mean"] / max(v9_gap, 1e-12)
            if ratio > 1.1:
                imp = f"  ({ratio:.1f}x gap; see sig. tests)"
        print(f"  {name:<22} {s['energy_mean']:>10.4f}+/-{s['energy_std']:>6.4f}"
              f"  {s['gap_mean']:>9.5f}  {s['steps_mean']:>7.1f}"
              f"  {s['circuits_mean']:>10.0f}{tag}{imp}")


def _save_benchmark_table(summary: dict, exact_E: float, ham_name: str, n_seeds: int):
    os.makedirs("sigma_data", exist_ok=True)
    safe = (ham_name.replace(" ", "_").replace("(", "").replace(")", "")
            .replace(",", "").replace("=", ""))
    path = f"sigma_data/benchmark_{safe}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "energy_mean", "energy_std", "gap_mean",
                    "gap_std", "steps_mean", "circuits_mean", "time_mean",
                    "n_seeds", "exact_E"])
        for name, s in summary.items():
            w.writerow([name, round(s["energy_mean"], 6), round(s["energy_std"], 6),
                        round(s["gap_mean"], 6), round(s["gap_std"], 6),
                        round(s["steps_mean"], 1), round(s["circuits_mean"], 1),
                        round(s["time_mean"], 3), n_seeds, round(exact_E, 6)])
    print(f"  [Table] Saved -> {path}")


# ===================================================================
# Ablation study (CostEvaluator(n_qubits, reps, H))
# ===================================================================

def run_ablation(H: qml.Hamiltonian, ansatz_spec, ham_name: str, ham_type: str,
                 n_qubits: int, reps: int,
                 max_steps: int, n_seeds_abl: int = 5,
                 logger: ParameterLogger = None) -> Dict:
    n_seeds_abl = min(n_seeds_abl, ABLATION_MAX_SEEDS)
    print(f"  [Ablation] ABLATION_MAX_SEEDS={ABLATION_MAX_SEEDS}. "
          f"Running with n_seeds={n_seeds_abl}. "
          f"[Backend]={_GPU_STATUS}  DISABLE_VSNG={DISABLE_VSNG} -- "
          f"'No VSNG' rows will equal their VSNG-on counterparts.")

    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    configs = {
        "Full v9.0":           dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
        "No escape":           dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
        "No preconditioner":   dict(escape=False, precond=False, momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
        "No momentum":         dict(escape=False, precond=True,  momentum=False,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
        "No CUSUM":            dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
        "No VSNG":             dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=False,    anneal=False,
                                    clip=True,    drift_correction=False),
        "With annealing":      dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=True,
                                    clip=True,    drift_correction=False),
        "No clipping":         dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=False,   drift_correction=False),
        "With drift correct.": dict(escape=False, precond=True,  momentum=True,
                                    cusum=False,  vsng=True,     anneal=False,
                                    clip=True,    drift_correction=True),
        "With escape+CUSUM":   dict(escape=True,  precond=True,  momentum=True,
                                    cusum=True,   vsng=True,     anneal=False,
                                    clip=True,    drift_correction=False),
    }
    results     = {k: [] for k in configs}
    all_lscapes = {k: [] for k in configs}
    all_cusums  = {k: [] for k in configs}
    all_escapes = {k: [] for k in configs}

    print(f"\n{'='*78}\n  ABLATION -- {ham_name}  |  {n_seeds_abl} seeds\n{'='*78}")

    for seed in range(n_seeds_abl):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, n_params)
        for cname, flags in configs.items():
            cf = CostEvaluator(n_qubits, reps, H)
            tup = run_sigma_qgd_v9(
                cf, theta0.copy(), n_qubits=n_qubits, reps=reps,
                max_steps=max_steps, ham_type=ham_type,
                use_escape=flags["escape"], use_precond=flags["precond"],
                use_momentum=flags["momentum"], use_cusum=flags["cusum"],
                use_vsng=flags["vsng"], use_anneal=flags["anneal"],
                use_clip=flags["clip"],
                use_drift_correction=flags["drift_correction"],
                logger=logger, seed=seed)
            results[cname].append(tup[0])
            if len(tup) > 2:
                all_lscapes[cname].append(tup[2])
                all_cusums[cname].append(tup[3])
                all_escapes[cname].append(tup[4])

    full_gaps = [abs(r.energy_opt - exact_E) for r in results["Full v9.0"]]
    full_gap  = float(np.mean(full_gaps))

    print(f"\n  {'Config':<25} {'Gap mean':>10} {'Gap std':>10} "
          f"{'Circuits':>10} {'Ratio':>8} {'p-val':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    sig_rows = []
    for cname, rs in results.items():
        gaps  = [abs(r.energy_opt - exact_E) for r in rs]
        circs = [r.n_circuits for r in rs]
        ratio = float(np.mean(gaps)) / max(full_gap, 1e-12)
        tag   = " (base)" if cname == "Full v9.0" else f" {ratio:.1f}x"
        pval_str = "  (base)"
        if cname != "Full v9.0" and len(full_gaps) >= 3 and len(gaps) >= 3:
            n_m = min(len(full_gaps), len(gaps))
            diffs = np.array(gaps[:n_m]) - np.array(full_gaps[:n_m])
            try:
                stat, pval = scipy_stats.wilcoxon(diffs, alternative="two-sided")
                pval_str = f"  {pval:.4f}"
                sig_rows.append(dict(ham=ham_name, config=cname,
                                     statistic=round(float(stat), 4),
                                     p_value=round(float(pval), 6),
                                     ratio=round(ratio, 4), n=n_m))
            except ValueError:
                pval_str = "  N/A"
        print(f"  {cname:<25} {np.mean(gaps):>10.5f} {np.std(gaps):>10.5f} "
              f"{np.mean(circs):>10.0f}{tag}{pval_str}")
        emit_component_warnings(
            all_lscapes.get(cname, []), all_cusums.get(cname, []),
            all_escapes.get(cname, []), sum(r.n_steps for r in rs))

    os.makedirs("sigma_data", exist_ok=True)
    abl_path = "sigma_data/ablation_results.csv"
    with open(abl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "gap_mean", "gap_std", "circuits_mean", "ratio", "n_seeds"])
        for cname, rs in results.items():
            gaps  = [abs(r.energy_opt - exact_E) for r in rs]
            circs = [r.n_circuits for r in rs]
            ratio = float(np.mean(gaps)) / max(full_gap, 1e-12)
            w.writerow([cname, round(np.mean(gaps), 6), round(np.std(gaps), 6),
                        round(np.mean(circs), 1), round(ratio, 2), len(rs)])
    print(f"  [Ablation] Saved -> {abl_path}")

    if sig_rows:
        abl_sig_path = "sigma_data/ablation_significance.csv"
        with open(abl_sig_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ham", "config", "statistic",
                                               "p_value", "ratio", "n"])
            w.writeheader(); w.writerows(sig_rows)
        print(f"  [Ablation] Significance tests -> {abl_sig_path}")

    return results


# ===================================================================
# Shot-noise experiment ( ShotNoiseCostEvaluator(n_qubits, reps, H))
# ===================================================================

def run_shot_noise_experiment(H, ansatz_spec, ham_name, ham_type,
                              n_qubits, reps, max_steps, n_seeds=3) -> dict:
    exact_E    = exact_gs(H)
    shot_counts = [100, 500, 1000, 5000, 10000]
    results    = {ns: [] for ns in shot_counts}
    results["noiseless"] = []
    n_params = n_qubits * reps

    print(f"\n{'='*78}\n  SHOT-NOISE -- {ham_name}\n  n_shots: {shot_counts}\n{'='*78}")

    for seed in range(n_seeds):
        np.random.seed(seed)
        theta0 = np.random.uniform(-np.pi/8, np.pi/8, n_params)
        cf = CostEvaluator(n_qubits, reps, H)
        r, _ = run_sigma_qgd_v8(cf, theta0.copy(), n_qubits, reps, max_steps,
                                 ham_type, seed=seed)
        results["noiseless"].append(r)
        for ns in shot_counts:
            cf_noisy = ShotNoiseCostEvaluator(n_qubits, reps, H, n_shots=ns)
            r, _ = run_sigma_qgd_v8(cf_noisy, theta0.copy(), n_qubits, reps,
                                     max_steps, ham_type, seed=seed)
            results[ns].append(r)
            print(f"  seed={seed}  shots={ns:>6}  gap={abs(r.energy_opt-exact_E):.5f}")

    print(f"\n  {'Shots':<12} {'Gap mean':>10} {'Gap std':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*10}")
    all_shot_results = {}
    for key in ["noiseless"] + shot_counts:
        gaps  = [abs(r.energy_opt - exact_E) for r in results[key]]
        label = str(key) if key != "noiseless" else "Noiseless"
        print(f"  {label:<12} {np.mean(gaps):>10.5f} {np.std(gaps):>10.5f}")
        all_shot_results[key] = {"gap_mean": np.mean(gaps),
                                  "gap_std": np.std(gaps),
                                  "n_seeds": len(gaps)}

    os.makedirs("sigma_data", exist_ok=True)
    sn_path = "sigma_data/shot_noise_results.csv"
    with open(sn_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_shots", "gap_mean", "gap_std", "n_seeds"])
        nl = all_shot_results["noiseless"]
        w.writerow(["inf", round(nl["gap_mean"], 6),
                    round(nl["gap_std"], 6), nl["n_seeds"]])
        for ns in shot_counts:
            d = all_shot_results[ns]
            w.writerow([ns, round(d["gap_mean"], 6),
                        round(d["gap_std"], 6), d["n_seeds"]])
    print(f"  [Shot-noise] Saved -> {sn_path}")
    return all_shot_results


# ===================================================================
# Scaling study ( CostEvaluator(n_qubits, reps, H))
# ===================================================================

def run_scaling_study(max_steps: int = 150, n_seeds: int = 3,
                      reps: int = 2, qubit_counts: List[int] = None) -> dict:
    if qubit_counts is None:
        qubit_counts = [4, 6, 8]
    scaling_results = {}

    print(f"\n{'='*78}\n  SCALING STUDY -- qubits: {qubit_counts}\n{'='*78}")

    for nq in qubit_counts:
        H = build_tfim(nq); exact_E = exact_gs(H)
        n_params = nq * reps
        sigma_gaps = []; adam_gaps = []
        for seed in range(n_seeds):
            np.random.seed(seed)
            theta0 = np.random.uniform(-np.pi/8, np.pi/8, n_params)
            cf = CostEvaluator(nq, reps, H)
            r, _ = run_sigma_qgd_v8(cf, theta0.copy(), nq, reps, max_steps, "tfim", seed=seed)
            sigma_gaps.append(abs(r.energy_opt - exact_E))
            def cf_factory(nq_=nq, r_=reps, h=H):
                return CostEvaluator(nq_, r_, h)
            r_adam, _, _ = run_adam_sweep(cf_factory, theta0.copy(), max_steps)
            adam_gaps.append(abs(r_adam.energy_opt - exact_E))
        scaling_results[nq] = {
            "sigma_gap_mean": np.mean(sigma_gaps), "sigma_gap_std": np.std(sigma_gaps),
            "adam_gap_mean": np.mean(adam_gaps), "adam_gap_std": np.std(adam_gaps),
            "p": nq * reps, "n_seeds": n_seeds}
        note = f"  [n={n_seeds}: ratio descriptive only]" if n_seeds < 5 else ""
        print(f"  n={nq}  p={nq*reps}  "
              f"SIGMA={np.mean(sigma_gaps):.5f}+/-{np.std(sigma_gaps):.5f}  "
              f"Adam={np.mean(adam_gaps):.5f}+/-{np.std(adam_gaps):.5f}{note}")

    os.makedirs("sigma_data", exist_ok=True)
    sc_path = "sigma_data/scaling_results.csv"
    with open(sc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_qubits", "p", "n_seeds",
                                          "sigma_gap_mean", "sigma_gap_std",
                                          "adam_gap_mean", "adam_gap_std"])
        w.writeheader()
        for nq, d in scaling_results.items():
            w.writerow(dict(n_qubits=nq, **d))
    print(f"  [Scaling] Saved -> {sc_path}")
    return scaling_results


# ===================================================================
# Plotting 
# ===================================================================

def plot_all(res_t, sum_t, E_t, res_x, sum_x, E_x,
             n_seeds: int, n_qubits: int,
             ablation_results: Dict = None, adam_sweep: Dict = None,
             shot_results: Dict = None, scaling_results: Dict = None):
    os.makedirs("sigma_data", exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 12, "legend.fontsize": 8,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})

    def _conv_axes(ax, all_res, exact_E, title, log_scale):
        for name in COLOURS:
            if name not in all_res or not all_res[name]: continue
            hists = [r.energy_hist for r in all_res[name]]
            mn    = min(len(h) for h in hists)
            if mn == 0: continue
            arr   = np.array([h[:mn] for h in hists])
            mu = arr.mean(0); sd = arr.std(0); xs = np.arange(mn)
            c  = COLOURS[name]; lw = LW.get(name, 1.5); ls = LINES.get(name, "-")
            zo = 12 if "SIGMA" in name else 5
            if log_scale:
                ax.semilogy(xs, np.maximum(np.abs(mu - exact_E), 1e-10),
                            color=c, lw=lw, ls=ls, label=name, zorder=zo)
            else:
                ax.plot(xs, mu, color=c, lw=lw, ls=ls, label=name, zorder=zo)
                ax.fill_between(xs, mu-sd, mu+sd, alpha=0.1, color=c)
        if not log_scale:
            ax.axhline(exact_E, color="k", ls=":", lw=1.2,
                       label=f"Exact ({exact_E:.3f})")
            ax.set_ylabel("Energy (H)")
        else:
            ax.set_ylabel("|E-E_exact| (log)")
        ax.set_xlabel("Step"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if res_t and sum_t:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v9.0 -- TFIM ({n_qubits}q, {n_seeds} seeds)",
                     fontsize=13, fontweight="bold")
        _conv_axes(axes[0], res_t, E_t, "Energy convergence -- TFIM", False)
        _conv_axes(axes[1], res_t, E_t, "Energy gap -- TFIM (log scale)", True)
        plt.tight_layout()
        plt.savefig("sigma_data/fig1_tfim_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig1_tfim_convergence.png")

    if res_x and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v9.0 -- XXZ ({n_qubits}q, {n_seeds} seeds)",
                     fontsize=13, fontweight="bold")
        _conv_axes(axes[0], res_x, E_x, "Energy convergence -- XXZ", False)
        _conv_axes(axes[1], res_x, E_x, "Energy gap -- XXZ (log scale)", True)
        plt.tight_layout()
        plt.savefig("sigma_data/fig2_xxz_convergence.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig2_xxz_convergence.png")
    else:
        print("  [WARNING] No XXZ results -- fig2_xxz_convergence.png skipped")

    if sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (smry, exact, title) in zip(
                axes, [(sum_t, E_t, "TFIM"), (sum_x, E_x, "XXZ")]):
            labels = [m for m in COLOURS if m in smry]
            groups = [smry[m]["energy_all"] for m in labels]
            colors = [COLOURS[m] for m in labels]
            tick_labels = [l.replace(" ", "\n") for l in labels]
            bp = ax.boxplot(groups, patch_artist=True,
                            medianprops=dict(color="k", lw=2))
            ax.set_xticks(range(1, len(tick_labels) + 1))
            ax.set_xticklabels(tick_labels)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)
            ax.axhline(exact, color="k", ls=":", lw=1.5, label=f"Exact ({exact:.3f})")
            ax.set_ylabel("Final energy (H)"); ax.set_title(f"{title} -- {n_seeds} seeds")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        fig.suptitle("Final energy distributions", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig3_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig3_boxplots.png")
    elif sum_t:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        labels = [m for m in COLOURS if m in sum_t]
        groups = [sum_t[m]["energy_all"] for m in labels]
        colors = [COLOURS[m] for m in labels]
        tick_labels = [l.replace(" ", "\n") for l in labels]
        bp = ax.boxplot(groups, patch_artist=True,
                        medianprops=dict(color="k", lw=2))
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.axhline(E_t, color="k", ls=":", lw=1.5, label=f"Exact ({E_t:.3f})")
        ax.set_ylabel("Final energy (H)"); ax.set_title(f"TFIM -- {n_seeds} seeds")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        fig.suptitle("Final energy distributions (TFIM only)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig3_boxplots.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig3_boxplots.png")

    if sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, (smry, exact, title) in zip(axes, [(sum_t, E_t, "TFIM"), (sum_x, E_x, "XXZ")]):
            for name in COLOURS:
                if name not in smry: continue
                s = smry[name]; c = COLOURS[name]
                ms = 200 if "SIGMA" in name else 80
                for e, ci in zip(s["energy_all"], s["circuits_all"]):
                    ax.scatter(ci, abs(e-exact), color=c, alpha=0.2, s=25,
                               marker=MARKERS.get(name, "o"))
                ax.scatter(s["circuits_mean"], s["gap_mean"], color=c, s=ms,
                           marker=MARKERS.get(name, "o"), zorder=10, label=name,
                           edgecolors="k", linewidths=1.5)
            ax.set_xlabel("Circuit evaluations"); ax.set_ylabel("|E-E_exact| (log)")
            ax.set_title(f"Circuit efficiency -- {title}")
            ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        fig.suptitle("Circuit efficiency: bottom-left = best", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig4_efficiency.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig4_efficiency.png")
    else:
        print("  [WARNING] fig4_efficiency.png requires both TFIM and XXZ -- skipped")

    if res_t and res_x and sum_t and sum_x:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        ax5l, ax5r = axes
        for name in COLOURS:
            for all_res, ls_ in [(res_t, "-"), (res_x, "--")]:
                if name not in all_res or not all_res[name]: continue
                hists = [r.energy_hist for r in all_res[name]]
                mn    = min(len(h) for h in hists)
                if mn == 0: continue
                arr = np.array([h[:mn] for h in hists])
                lw_ = 2.5 if "SIGMA" in name else 1.0
                ax5l.plot(arr.std(0), color=COLOURS[name], lw=lw_, ls=ls_,
                          alpha=0.8, label=name if ls_=="-" else None)
        ax5l.set_yscale("log"); ax5l.legend(fontsize=7)
        ax5l.set_xlabel("Step"); ax5l.set_ylabel("Std dev")
        ax5l.set_title("Stability (solid=TFIM dashed=XXZ)"); ax5l.grid(True, alpha=0.3)
        labels_bar = [m for m in COLOURS if m in sum_t and m in sum_x]
        if labels_bar:
            x = np.arange(len(labels_bar)); w = 0.35
            g_t = [sum_t[m]["gap_mean"] for m in labels_bar]
            g_x = [sum_x[m]["gap_mean"] for m in labels_bar]
            clrs = [COLOURS[m] for m in labels_bar]
            ax5r.bar(x-w/2, g_t, w, color=clrs, alpha=0.85, edgecolor="k", lw=0.5, label="TFIM")
            ax5r.bar(x+w/2, g_x, w, color=clrs, alpha=0.45, edgecolor="k", lw=0.5, hatch="//", label="XXZ")
            ax5r.set_xticks(x)
            ax5r.set_xticklabels([l.replace(" ", "\n") for l in labels_bar],
                                rotation=20, ha="right", fontsize=8)
            ax5r.set_yscale("log"); ax5r.set_ylabel("|E-E_exact| mean")
            ax5r.set_title("Final gap -- TFIM vs XXZ"); ax5r.legend(fontsize=9)
            ax5r.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Stability and gap summary", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig5_stability.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig5_stability.png")
    else:
        print("  [WARNING] fig5_stability.png requires both TFIM and XXZ -- skipped")

    _plot_phase_history(res_t, res_x, n_seeds)
    _plot_circuit_cost(res_t, res_x, n_seeds)
    if ablation_results:
        _plot_ablation_heatmap(ablation_results, E_t if E_t is not None else E_x)
    if adam_sweep:
        _plot_adam_sweep(adam_sweep)
    if shot_results:
        _plot_shot_noise(shot_results)
    if scaling_results:
        _plot_scaling(scaling_results)


def _plot_phase_history(res_t, res_x, n_seeds: int):
    PHASE_COLS = {"active": "#2ECC71", "plateau": "#E74C3C",
                  "local_min": "#F39C12", "converged": "#3498DB"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plotted   = False
    for ax, all_res, title in [(axes[0], res_t, "TFIM"), (axes[1], res_x, "XXZ")]:
        if all_res is None: ax.set_visible(False); continue
        key = "SIGMA-QGD v9.0"
        if key not in all_res or not all_res[key]: ax.set_visible(False); continue
        phase_hists = [r.phase_hist for r in all_res[key] if r.phase_hist]
        if not phase_hists: ax.set_visible(False); continue
        min_steps = min(len(ph) for ph in phase_hists)
        phase_counts = {p: [] for p in PHASE_COLS}
        for si in range(min_steps):
            cnts = {p: 0 for p in PHASE_COLS}
            for r in all_res[key]:
                if si < len(r.phase_hist):
                    ph = r.phase_hist[si]
                    if ph in cnts: cnts[ph] += 1
            tot = max(sum(cnts.values()), 1)
            for p in PHASE_COLS:
                phase_counts[p].append(cnts[p] / tot)
        xs = np.arange(min_steps); bottom = np.zeros(min_steps)
        for ph, col in PHASE_COLS.items():
            vals = np.array(phase_counts[ph])
            ax.fill_between(xs, bottom, bottom + vals, alpha=0.75, color=col, label=ph)
            bottom += vals
        ax.set_ylim(0, 1); ax.set_xlabel("Step"); ax.set_ylabel("Fraction of seeds")
        ax.set_title(f"Phase distribution -- {title}"); ax.legend(fontsize=7, loc="upper right")
        plotted = True
    if plotted:
        fig.suptitle("SIGMA-QGD v9.0: Landscape phase distribution",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig6_phase_history.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig6_phase_history.png")
    else:
        plt.close(); print("  [WARNING] fig6: no phase data")


def _plot_circuit_cost(res_t, res_x, n_seeds: int):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plotted   = False
    for ax, all_res, title in [(axes[0], res_t, "TFIM"), (axes[1], res_x, "XXZ")]:
        if all_res is None: ax.set_visible(False); continue
        key = "SIGMA-QGD v9.0"
        if key not in all_res or not all_res[key]: ax.set_visible(False); continue
        hists = [r.circuits_hist for r in all_res[key] if r.circuits_hist]
        if not hists: ax.set_visible(False); continue
        mn  = min(len(h) for h in hists)
        arr = np.array([h[:mn] for h in hists])
        mu  = arr.mean(0); sd = arr.std(0); xs = np.arange(mn)
        ax.plot(xs, mu, color=COLOURS[key], lw=2.5, label="Full gradient (VSNG off, batched)")
        ax.fill_between(xs, mu-sd, mu+sd, alpha=0.15, color=COLOURS[key])
        ax.axhline(3, color="brown", ls=":", lw=1.2, label="SPSA (3)")
        ax.set_xlabel("Step"); ax.set_ylabel("Circuits / step")
        ax.set_title(f"Circuit budget -- {title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); plotted = True
    if plotted:
        fig.suptitle("Circuit cost per step (VSNG disabled -> constant 2p+1, batched call)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sigma_data/fig7_circuit_cost.png", dpi=150, bbox_inches="tight")
        plt.close(); print("  Saved: fig7_circuit_cost.png")
    else:
        plt.close()


def _plot_ablation_heatmap(ablation_results: Dict, exact_E: float):
    configs = list(ablation_results.keys())
    gaps_mean = [np.mean([abs(r.energy_opt - exact_E) for r in ablation_results[c]])
                 for c in configs]
    gaps_std  = [np.std([abs(r.energy_opt - exact_E) for r in ablation_results[c]])
                 for c in configs]
    fig, ax = plt.subplots(figsize=(10, max(6, len(configs) * 0.5 + 2)))
    colors  = ["#00C9A7" if "Full" in c else "#E24B4A" for c in configs]
    ax.barh(range(len(configs)), gaps_mean, xerr=gaps_std,
            color=colors, alpha=0.8, edgecolor="k", lw=0.5, capsize=4)
    base_gap = gaps_mean[0]
    for i, (g, gs, c) in enumerate(zip(gaps_mean, gaps_std, configs)):
        ratio = g / max(base_gap, 1e-12)
        label = "1.0x (base)" if "Full" in c else f"{ratio:.1f}x"
        ax.text(g + gs + 0.0001, i, label, va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel("Energy gap |E - E_exact| (H)", fontsize=11)
    ax.set_title("Ablation Study\n(see ablation_significance.csv for p-values)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("sigma_data/fig8_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig8_ablation.png")


def _plot_adam_sweep(adam_sweep: Dict):
    if not adam_sweep: return
    fig, ax = plt.subplots(figsize=(8, 5))
    lrs = sorted(adam_sweep.keys()); gaps = [adam_sweep[lr] for lr in lrs]
    ax.plot(lrs, gaps, "o-", color="#E24B4A", lw=2, markersize=10,
            markeredgecolor="k", markeredgewidth=1.5, label="Adam")
    best_idx = np.argmin(gaps)
    ax.plot(lrs[best_idx], gaps[best_idx], "*", color="gold",
            markersize=20, markeredgecolor="k", markeredgewidth=1.5,
            zorder=10, label=f"Best (lr={lrs[best_idx]})")
    ax.set_xlabel("Learning rate alpha", fontsize=11)
    ax.set_ylabel("Best energy found", fontsize=11); ax.set_xscale("log")
    ax.set_title("Adam Hyperparameter Sensitivity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sigma_data/fig9_adam_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig9_adam_sweep.png")


def _plot_shot_noise(shot_results: Dict):
    if not shot_results: return
    fig, ax = plt.subplots(figsize=(8, 5))
    shot_keys = sorted(k for k in shot_results.keys() if k != "noiseless")
    gaps = [shot_results[k]["gap_mean"] for k in shot_keys]
    stds = [shot_results[k]["gap_std"] for k in shot_keys]
    ax.errorbar(shot_keys, gaps, yerr=stds, fmt="D-", color="#00C9A7",
                lw=2, markersize=8, capsize=5, markeredgecolor="k",
                markeredgewidth=1.5, label="SIGMA-QGD v9.0")
    if "noiseless" in shot_results:
        nl_gap = shot_results["noiseless"]["gap_mean"]
        ax.axhline(nl_gap, color="k", ls=":", lw=1.5,
                   label=f"Noiseless ({nl_gap:.4f})")
    ax.set_xlabel("Number of shots", fontsize=11)
    ax.set_ylabel("Energy gap |E - E_exact|", fontsize=11); ax.set_xscale("log")
    ax.set_title("Shot-Noise Robustness\n(measured under tested conditions)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sigma_data/fig10_shot_noise.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig10_shot_noise.png")


def _plot_scaling(scaling_results: Dict):
    if not scaling_results: return
    fig, ax = plt.subplots(figsize=(9, 5))
    qubits     = sorted(scaling_results.keys())
    sigma_gaps = [scaling_results[nq]["sigma_gap_mean"] for nq in qubits]
    sigma_stds = [scaling_results[nq]["sigma_gap_std"]  for nq in qubits]
    adam_gaps  = [scaling_results[nq]["adam_gap_mean"]  for nq in qubits]
    adam_stds  = [scaling_results[nq]["adam_gap_std"]   for nq in qubits]
    x = np.arange(len(qubits)); w = 0.3
    ax.bar(x-w/2, sigma_gaps, w, yerr=sigma_stds, color="#00C9A7",
           alpha=0.85, edgecolor="k", lw=0.5, capsize=5, label="SIGMA-QGD v9.0")
    ax.bar(x+w/2, adam_gaps, w, yerr=adam_stds, color="#E24B4A",
           alpha=0.85, edgecolor="k", lw=0.5, capsize=5, label="Adam (best)")
    for i, nq in enumerate(qubits):
        ratio = adam_gaps[i] / max(sigma_gaps[i], 1e-12)
        ax.text(i, max(sigma_gaps[i], adam_gaps[i]) +
                max(sigma_stds[i], adam_stds[i]) + 0.005,
                f"{ratio:.1f}x*", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nq}q (p={scaling_results[nq]['p']})"
                        for nq in qubits], fontsize=9)
    ax.set_xlabel("System size", fontsize=11)
    ax.set_ylabel("Energy gap |E - E_exact| (H)", fontsize=11)
    ax.set_title("Scaling: SIGMA-QGD v9.0 vs Adam (TFIM)\n"
                 "(*ratio descriptive only; see significance_tests.csv for p-values)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("sigma_data/fig11_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(); print("  Saved: fig11_scaling.png")


# ===================================================================
# Inline unit tests ( adjusted CostEvaluator signature)
# ===================================================================

def test_escape_mechanism_fires():
    n, r = 2, 1
    cfg = derive_config(n, r, max_steps=300, ham_type="tfim")
    p   = cfg["p"]
    lscape = LandscapeClassifier(cfg)
    lscape.gnorm_abs_tol = 1e10
    cusum = NormalizedCUSUM(p, cfg["W_med"], tau=1.0, h_factor=0.1)
    cusum.alarm_state = np.ones(p, dtype=bool); cusum.n_alarms = 1
    energy = -1.5; gnorm = 0.001
    for _ in range(cfg["stag_win"] + 5):
        phase = lscape.classify(energy, gnorm, cusum.alarm())
    assert phase in ("plateau", "local_min"), \
        f"Expected plateau/local_min, got '{phase}'"
    var_eng = VarianceEngine(p, cfg["W"])
    for _ in range(10):
        var_eng.update(np.ones(p) * 0.1)
    escape = SingleDimEscape(cfg, eta_0=0.05)
    escape.last_esc = -(cfg["cooldown"] + 100)
    call_count = [0]
    def fake_cost(theta):
        call_count[0] += 1; return energy - 0.01
    curvature = CurvatureProxy(0.05, 0.01)
    gamma_val = curvature.gamma(var_eng.welford_raw)
    sigma_vec = np.maximum(curvature.per_dim_sigma(var_eng.welford_raw, gamma_val), 0.01)
    theta = np.zeros(p)
    _, _, escaped = escape.escape(
        theta=theta, phase=phase, no_imp=cfg["cooldown"] + 100,
        cusum_alarm=cusum.alarm(), welford_raw=var_eng.welford_raw + 0.01,
        cost_fn=fake_cost, cost=energy, sigma=sigma_vec,
        step=cfg["warmup"] + 100)
    assert escaped, "Escape did NOT fire under guaranteed plateau conditions."
    assert call_count[0] >= 1, "Escape did not call cost_fn."
    print("[test_escape_mechanism_fires] PASSED")
    return True


def test_cost_and_gradient_matches_separate():
    """>>> GPU: verify batched cost_and_gradient matches separate __call__+gradient()."""
    n, r = 2, 1
    H = build_tfim(n)
    theta = np.random.uniform(-0.5, 0.5, n * r)
    cf1 = CostEvaluator(n, r, H)
    c1 = cf1(theta); g1 = cf1.gradient(theta)
    cf2 = CostEvaluator(n, r, H)
    c2, g2 = cf2.cost_and_gradient(theta)
    assert abs(c1 - c2) < 1e-6, f"Cost mismatch: {c1} vs {c2}"
    assert np.allclose(g1, g2, atol=1e-6), f"Gradient mismatch: {g1} vs {g2}"
    print("[test_cost_and_gradient_matches_separate] PASSED")
    return True


def test_no_xxz_fallback():
    import os
    fig_path = "sigma_data/fig2_xxz_convergence.png"
    if os.path.exists(fig_path):
        os.remove(fig_path)
    plot_all(None, None, None, None, None, None, n_seeds=1, n_qubits=4)
    assert not os.path.exists(fig_path), \
        f"BUG FIX 2 failed: {fig_path} was generated with res_x=None."
    print("[test_no_xxz_fallback] PASSED: fig2 not generated with res_x=None")
    return True


# ===================================================================
# CLI 
# ===================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="SIGMA-QGD v9.0 -- PennyLane/GPU (lightning.gpu) SIGMA vs Adam")
    ap.add_argument("--qubits",          type=int,   default=12)
    ap.add_argument("--reps",            type=int,   default=2)
    ap.add_argument("--seeds",           type=int,   default=5)
    ap.add_argument("--steps",           type=int,   default=200)
    ap.add_argument("--hamiltonian",     type=str,   default="tfim",
                    choices=["tfim", "xxz", "both"])
    ap.add_argument("--eta",             type=float, default=0.05)
    ap.add_argument("--lam",             type=float, default=0.01)
    ap.add_argument("--tau",             type=float, default=1.0)
    ap.add_argument("--enable-escape",   action="store_true")
    ap.add_argument("--enable-cusum",    action="store_true")
    ap.add_argument("--enable-anneal",   action="store_true")
    ap.add_argument("--no-precond",      action="store_true")
    ap.add_argument("--no-momentum",     action="store_true")
    ap.add_argument("--no-vsng",         action="store_true",
                    help="No-op -- VSNG is force-disabled globally in this variant")
    ap.add_argument("--drift-correction",action="store_true")
    ap.add_argument("--ablation",        action="store_true")
    ap.add_argument("--shot-noise",      action="store_true")
    ap.add_argument("--scaling",         action="store_true")
    ap.add_argument("--scaling-qubits",  type=str, default="4,8,12")
    ap.add_argument("--quick",           action="store_true")
    ap.add_argument("--run-tests",       action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.run_tests:
        print("\n  Running inline unit tests...")
        t1 = test_escape_mechanism_fires()
        t2 = test_cost_and_gradient_matches_separate()
        t3 = test_no_xxz_fallback()
        if t1 and t2 and t3:
            print("\n  All 3 tests PASSED.")
        sys.exit(0)

    if args.quick:
        args.seeds = min(args.seeds, 3)
        args.steps = min(args.steps, 100)
        if args.hamiltonian == "both":
            args.hamiltonian = "tfim"
            print("  [Quick] --hamiltonian forced to 'tfim'")

    try:
        scaling_qubits = [int(x.strip()) for x in args.scaling_qubits.split(",")]
    except ValueError:
        print(f"  [ERROR] --scaling-qubits must be comma-separated ints, got: '{args.scaling_qubits}'")
        sys.exit(1)

    flags = dict(
        escape          = args.enable_escape,
        precond         = not args.no_precond,
        momentum        = not args.no_momentum,
        cusum           = args.enable_cusum,
        vsng            = not args.no_vsng,
        anneal          = args.enable_anneal,
        clip            = True,
        drift_correction= args.drift_correction,
    )

    
    _probe_dev, _GPU_STATUS = _make_device(max(args.qubits, 1))
    del _probe_dev

    print("\n" + "="*78)
    print("  SIGMA-QGD v9.0 -- PENNYLANE / GPU (lightning.gpu, cuQuantum) VARIANT")
    print(f"  Backend                 = {_GPU_STATUS}")
    print(f"  DISABLE_VSNG            = {DISABLE_VSNG}  (forced off everywhere)")
    print(f"  BENCHMARK_METHODS       = {BENCHMARK_METHODS}")
    print(f"  Config: {args.qubits}q reps={args.reps} seeds={args.seeds} steps={args.steps}")
    print(f"  Flags: {flags}")
    print("="*78)

    os.makedirs("sigma_data", exist_ok=True)
    logger = ParameterLogger("sigma_data")
    ansatz_spec = build_ansatz(args.qubits, args.reps)

    res_t = sum_t = E_t = None
    res_x = sum_x = E_x = None
    ablation_results = None
    shot_results     = None
    scaling_results  = None

    if args.hamiltonian in ("tfim", "both"):
        H_t = build_tfim(args.qubits)
        res_t, sum_t, E_t = run_benchmark(
            H_t, ansatz_spec, "TFIM (J=1, h=0.5)", "tfim",
            args.qubits, args.reps, args.seeds, args.steps, flags, logger)

    if args.hamiltonian in ("xxz", "both"):
        H_x = build_xxz(args.qubits)
        res_x, sum_x, E_x = run_benchmark(
            H_x, ansatz_spec, "XXZ (Jxy=1, Jz=0.5, h=0.1)", "xxz",
            args.qubits, args.reps, args.seeds, args.steps, flags, logger)

    if args.hamiltonian == "both":
        if res_t is None or res_x is None:
            missing = "TFIM" if res_t is None else "XXZ"
            raise RuntimeError(
                f"--hamiltonian both was requested but {missing} results are missing. "
                f"Re-run with --hamiltonian {'tfim' if res_t is None else 'xxz'} "
                f"to isolate the failure. NO data substitution will be performed.")

    if args.ablation:
        H_abl = build_tfim(args.qubits)
        ablation_results = run_ablation(
            H_abl, ansatz_spec, "TFIM", "tfim",
            args.qubits, args.reps, args.steps,
            n_seeds_abl=args.seeds, logger=logger)

    if args.shot_noise:
        H_sn = build_tfim(args.qubits)
        shot_results = run_shot_noise_experiment(
            H_sn, ansatz_spec, "TFIM", "tfim",
            args.qubits, args.reps,
            max_steps=min(args.steps, 100),
            n_seeds=min(args.seeds, 3))

    if args.scaling:
        scaling_results = run_scaling_study(
            max_steps=min(args.steps, 150),
            n_seeds=min(args.seeds, 3),
            reps=args.reps,
            qubit_counts=scaling_qubits)

    logger.save()

    print("\n  Generating figures ...")
    plot_all(res_t, sum_t, E_t, res_x, sum_x, E_x, args.seeds, args.qubits,
             ablation_results=ablation_results, adam_sweep=None,
             shot_results=shot_results, scaling_results=scaling_results)

    print("\n   OUTPUTS")
    print("  sigma_data/significance_tests.csv      -- Wilcoxon tests (SIGMA vs Adam)")
    if ablation_results:
        print("  sigma_data/ablation_results.csv        -- with n_seeds col ")
        print("  sigma_data/ablation_significance.csv   -- ablation Wilcoxon tests")
    if shot_results:
        print("  sigma_data/shot_noise_results.csv      -- shot-noise data ")
    if scaling_results:
        print("  sigma_data/scaling_results.csv         -- scaling data")
    print("  sigma_data/step_records.json/.csv      -- step-level data")
    print("  sigma_data/run_summary.csv             -- per-run summary")
    print("  sigma_data/benchmark_*.csv             -- benchmark tables")
    print("  --")
    print("  Run tests:    python Sigma_QGD_pennylane.py --run-tests")
    print("  12q run:      python Sigma_QGD_pennylane.py")
    print("  Full suite:   python Sigma_QGD_pennylane.py --ablation --shot-noise --scaling")
