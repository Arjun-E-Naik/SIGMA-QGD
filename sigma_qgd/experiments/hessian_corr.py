"""
Empirical correlation analysis between gradient variance and diagonal Hessian curvature.
"""

from __future__ import annotations
import csv
import os
from typing import List

import numpy as np

try:
    from ..core.config import derive_config
    from ..core.evaluator import CostEvaluator
    from ..core.progress import ProgressReporter
    from ..core.utils import get_initial_theta
    from ..optimizers.sigma_qgd import VarianceEngine, CurvatureProxy
    from ..problems.hamiltonians import exact_gs
    from ..analysis.stats import correlation_summary
    from ..style import ABLATION_MAX_SEEDS
except ImportError:
    raise ImportError("Could not import required modules. Please install sigma-qgd first.")


def run_hessian_correlation_experiment(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 200,
    n_seeds: int = 15,
    eta_0: float = 0.1,
    multiplier: float = 1.0,
    lambda_reg: float = 0.001,
    device=None,
) -> dict:
    """
    Empirical test: Var_t[g_i] ~ H_ii^2  (Proposition 1 diagnostic).

    Measures Pearson (log-space) and Spearman correlation between
    welford_raw (proxy) and hess_diag^2 (exact squared curvature).
    Circuit cost: 2p+1 per step (same as gradient call, zero extra).
    """
    n_seeds = min(n_seeds, ABLATION_MAX_SEEDS)
    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    if device is None:
        device, _ = make_device(n_qubits)

    per_seed_rows: List[dict] = []
    pearson_vals:  List[float] = []
    spearman_vals: List[float] = []

    

    for seed in range(n_seeds):
        theta = get_initial_theta(n_params, seed)

        cfg = derive_config(n_qubits, reps, max_steps, ham_type, eta_0=eta_0, lambda_reg=lambda_reg)
        p   = cfg["p"]
        var_eng   = VarianceEngine(p, cfg["W"])
        curvature = CurvatureProxy(eta_0, lambda_reg, multiplier=multiplier)
        m         = np.zeros(p)
        beta      = cfg["beta"]
        cn        = cfg["clip_norm"]
        cf        = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)

        welford_vals: List[np.ndarray] = []
        hess_sq_vals: List[np.ndarray] = []
        n_steps_run = 0

        rep = ProgressReporter(max_steps, desc="Hessian-Corr", exact_E=exact_E,
                               seed=seed, n_seeds=n_seeds)

        for t in range(1, max_steps + 1):
            cost, g, hess_diag = cf.cost_gradient_and_diag_hessian(theta)
            gnorm = float(np.linalg.norm(g))
            gc    = g * cn / (gnorm + 1e-30) if gnorm > cn else g.copy()
            var_eng.update(gc)
            if t >= 2:
                welford_vals.append(var_eng.welford_raw.copy())
                hess_sq_vals.append(hess_diag ** 2)
            m    = beta * m + (1.0 - beta) * gc
            mhat = m / (1.0 - beta ** t)
            if t >= cfg["t_qfim"]:
                d = curvature.precondition(var_eng.welford_raw, mhat)
            else:
                d = mhat
            theta = theta - eta_0 * d
            n_steps_run = t
            rep.update(t, cost, theta, gnorm=gnorm, circuits=cf.n_calls)
            if gnorm < 1e-6:
                break

        rep.finish(final_energy=cost, final_theta=theta, circuits=cf.n_calls, steps=n_steps_run)

        if len(welford_vals) < 2:
            continue

        w_arr = np.concatenate(welford_vals)
        h_arr = np.concatenate(hess_sq_vals)
        corr  = correlation_summary(w_arr, h_arr)
        pearson_vals.append(corr["pearson_log"])
        spearman_vals.append(corr["spearman"])
        print(f"  seed={seed}  steps={n_steps_run}  "
              f"pearson_log={corr['pearson_log']:.4f}  spearman={corr['spearman']:.4f}")
        per_seed_rows.append(dict(seed=seed, n_qubits=n_qubits, reps=reps,
                                   n_steps=n_steps_run, n_points=len(w_arr),
                                   eta_0=eta_0, multiplier=multiplier, lambda_reg=lambda_reg,
                                   **corr))

    p_arr = np.array([v for v in pearson_vals  if v == v])
    s_arr = np.array([v for v in spearman_vals if v == v])
    p_mean = float(np.mean(p_arr)) if len(p_arr) > 0 else float("nan")
    s_mean = float(np.mean(s_arr)) if len(s_arr) > 0 else float("nan")

    print(f"\n  Summary: Pearson(log) mean={p_mean:.4f}  Spearman mean={s_mean:.4f}")

    per_path = os.path.join(results_dir, f"hessian_correlation_results{sfx}.csv")
    with open(per_path, "w", newline="") as f:
        if per_seed_rows:
            w = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            w.writeheader(); w.writerows(per_seed_rows)
    print(f"  [Hessian Corr] Saved → {per_path}")

    return {"pearson_mean": p_mean, "spearman_mean": s_mean, "per_seed": per_seed_rows}
