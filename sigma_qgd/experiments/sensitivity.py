"""
Hyperparameter sensitivity sweep over preconditioning multiplier and regularization parameters.
"""

from __future__ import annotations
import csv
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    from ..core.device import make_device
    from ..core.evaluator import CostEvaluator
    from ..core.utils import get_initial_theta
    from ..optimizers.sigma_qgd import SigmaQGD
    from ..problems.hamiltonians import exact_gs
    from ..style import ABLATION_MAX_SEEDS
except ImportError:
    print("Could not import required modules. Please install sigma-qgd first.")


def run_sensitivity_sweep(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 200,
    n_seeds: int = 5,
    multiplier_values: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 6.0),
    lambda_reg_values: Tuple[float, ...] = (0.0005, 0.001, 0.005, 0.01, 0.05),
    base_eta: float = 0.05,
    base_lambda: float = 0.001,
    base_multiplier: float = 1.0,
    device=None,
) -> List[dict]:
    """One-factor-at-a-time sweep over SIGMA-QGD hyperparameters (multiplier, lambda_reg)."""
    n_seeds = min(n_seeds, ABLATION_MAX_SEEDS)
    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    if device is None:
        device, _ = make_device(n_qubits)

    rows: List[dict] = []

    

    def _run_point(sweep, value, multiplier, lambda_reg):
        gaps = []
        for seed in range(n_seeds):
            theta0 = get_initial_theta(n_params, seed)
            cf  = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)
            opt = SigmaQGD(precond_multiplier=multiplier)
            r, _, _ = opt.run(cf, theta0.copy(), max_steps=max_steps,
                               ham_type=ham_type, n_qubits=n_qubits, reps=reps,
                               eta_0=base_eta, lambda_reg=lambda_reg, seed=seed,
                               n_seeds=n_seeds, exact_E=exact_E,
                               method_name=f"Sens: {sweep}={value}")
            gaps.append(abs(r.energy_opt - exact_E))
        row = dict(sweep=sweep, value=value,
                   gap_mean=float(np.mean(gaps)), gap_std=float(np.std(gaps)),
                   n_seeds=n_seeds, n_qubits=n_qubits)
        rows.append(row)
        print(f"    {sweep:<12} = {str(value):<6}  gap={row['gap_mean']:.5f} "
              f"± {row['gap_std']:.5f}")
        return row

    print("\n  -- multiplier sweep --")
    for mv in multiplier_values:
        _run_point("multiplier", mv, mv, base_lambda)

    print("\n  -- lambda_reg sweep --")
    for lv in lambda_reg_values:
        _run_point("lambda_reg", lv, base_multiplier, lv)

    path = os.path.join(results_dir, f"sensitivity_sweep{sfx}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "value", "gap_mean",
                                           "gap_std", "n_seeds", "n_qubits"])
        w.writeheader(); w.writerows(rows)
    print(f"  [Sensitivity] Saved → {path}")
    return rows
