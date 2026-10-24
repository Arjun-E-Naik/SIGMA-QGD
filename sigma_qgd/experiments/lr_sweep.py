"""
Learning rate grid sweep for gradient-based optimizers.
"""

from __future__ import annotations
import csv
import os
from typing import List, Optional

import numpy as np

from ..core.evaluator import CostEvaluator
from ..core.utils import get_initial_theta
from ..optimizers.registry import OPTIMIZER_REGISTRY
from ..problems.hamiltonians import exact_gs
from ..style import DEFAULT_SEEDS

DEFAULT_LR_GRID = [0.001, 0.01, 0.05, 0.1]


def run_lr_sweep(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 200,
    n_seeds: int = DEFAULT_SEEDS,
    lr_grid: Optional[List[float]] = None,
    methods: Optional[List[str]] = None,
    sigma_kwargs: Optional[dict] = None,
    device=None,
) -> dict:
    """
    For each gradient-based optimizer: sweep over lr_grid, record mean gap.

    Writes:
      lr_sweep_detail_{tag}.csv   — one row per (method, lr, seed)
      lr_sweep_summary_{tag}.csv  — best LR and best gap per method

    Parameters
    ----------
    methods : list[str] or None
        If None, all LR-capable optimizers from the registry are used.
    sigma_kwargs : dict or None
        Optional dictionary of hyperparameter overrides for SIGMA-QGD (e.g. multiplier, lambda_reg).
    device : Device or None
        PennyLane device to use.
    """
    if lr_grid is None:
        lr_grid = DEFAULT_LR_GRID

    # Filter to only LR-capable optimizers
    if methods is None:
        methods = [name for name, opt in OPTIMIZER_REGISTRY.items() if opt.supports_lr]
    else:
        methods = [m for m in methods
                   if m in OPTIMIZER_REGISTRY and OPTIMIZER_REGISTRY[m].supports_lr]

    exact_E  = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    detail_rows  = []
    summary_rows = []

    

    best_by_method = {}

    for method_name in methods:
        opt = OPTIMIZER_REGISTRY[method_name]
        print(f"\n  [{method_name}]")
        lr_results = {}

        for lr in lr_grid:
            gaps = []
            for seed in range(n_seeds):
                theta0 = get_initial_theta(n_params, seed)
                cf = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)

                run_kw = dict(
                    max_steps=max_steps,
                    ham_type=ham_type,
                    n_qubits=n_qubits,
                    reps=reps,
                    seed=seed,
                    n_seeds=n_seeds,
                    lr=lr,
                    exact_E=exact_E,
                    method_name=f"{method_name} (lr={lr})",
                )
                if method_name == "SIGMA-QGD v9.0" and sigma_kwargs:
                    run_kw.update(sigma_kwargs)

                result, _, _ = opt.run(cf, theta0.copy(), **run_kw)
                gap = abs(result.energy_opt - exact_E)
                gaps.append(gap)
                detail_rows.append(dict(
                    method=method_name, lr=lr, seed=seed,
                    gap=round(gap, 6), n_qubits=n_qubits,
                    ham=ham_name,
                ))

            mean_gap = float(np.mean(gaps))
            std_gap  = float(np.std(gaps))
            lr_results[lr] = (mean_gap, std_gap)
            print(f"    lr={lr:.4f}  gap={mean_gap:.5f} ± {std_gap:.5f}")

        best_lr  = min(lr_results, key=lambda lr: lr_results[lr][0])
        best_gap = lr_results[best_lr][0]
        best_std = lr_results[best_lr][1]
        best_by_method[method_name] = dict(best_lr=best_lr,
                                            best_gap=best_gap, best_std=best_std)
        summary_rows.append(dict(
            method=method_name, best_lr=best_lr,
            best_gap=round(best_gap, 6), best_std=round(best_std, 6),
            ham=ham_name, n_qubits=n_qubits, n_seeds=n_seeds,
        ))
        print(f"  → Best LR={best_lr}  gap={best_gap:.5f} ± {best_std:.5f}")

    # Save detail CSV
    det_path = os.path.join(results_dir, f"lr_sweep_detail{sfx}.csv")
    with open(det_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "lr", "seed", "gap",
                                           "n_qubits", "ham"])
        w.writeheader(); w.writerows(detail_rows)
    print(f"  [LR Sweep] Detail → {det_path}")

    # Save summary CSV
    sum_path = os.path.join(results_dir, f"lr_sweep_summary{sfx}.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "best_lr", "best_gap",
                                           "best_std", "ham", "n_qubits", "n_seeds"])
        w.writeheader(); w.writerows(summary_rows)
    print(f"  [LR Sweep] Summary → {sum_path}")

    # Print comparison table
    print(f"\n  LR Sweep Results -- {ham_name}  (n_qubits={n_qubits}, {n_seeds} seeds)")
    print(f"  {'Method':<22} {'Best LR':>8} {'Best Gap':>10} {'Std':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*10} {'─'*8}")
    for row in sorted(summary_rows, key=lambda r: r["best_gap"]):
        print(f"  {row['method']:<22} {row['best_lr']:>8.4f} "
              f"{row['best_gap']:>10.5f} {row['best_std']:>8.5f}")

    return best_by_method
