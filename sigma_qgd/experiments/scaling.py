"""
System size scaling benchmark across qubit counts.
"""

from __future__ import annotations
import csv
import os
from typing import List

import numpy as np

try:
    from ..core.evaluator import CostEvaluator
    from ..core.utils import get_initial_theta
    from ..optimizers.sigma_qgd import SigmaQGD
    from ..optimizers.adam_family import Adam
    from ..problems.hamiltonians import build_tfim, exact_gs
    from ..analysis.stats import wilcoxon_pairwise
    from ..style import DEFAULT_LR
except ImportError:
    print("Could not import required modules. Please install sigma-qgd first.")


def run_scaling_study(
    results_dir: str,
    tag: str = "",
    max_steps: int = 150,
    n_seeds: int = 3,
    reps: int = 2,
    qubit_counts: List[int] = None,
) -> dict:
    """SIGMA-QGD vs Adam over increasing system sizes."""
    if qubit_counts is None:
        qubit_counts = [4, 6, 8]
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    scaling_results = {}
    sigma_opt = SigmaQGD()
    adam_opt  = Adam()

    

    for nq in qubit_counts:
        H = build_tfim(nq); exact_E = exact_gs(H)
        n_params = nq * reps
        sigma_gaps = []; adam_gaps = []

        for seed in range(n_seeds):
            theta0 = get_initial_theta(n_params, seed)

            cf  = CostEvaluator(nq, reps, H, exact_E=exact_E)
            r, _, _ = sigma_opt.run(cf, theta0.copy(), max_steps=max_steps,
                                     ham_type="tfim", n_qubits=nq, reps=reps, seed=seed,
                                     n_seeds=n_seeds, exact_E=exact_E,
                                     method_name=f"Scaling {nq}q (SIGMA)")
            sigma_gaps.append(abs(r.energy_opt - exact_E))

            cf2 = CostEvaluator(nq, reps, H, exact_E=exact_E)
            r2, _, _ = adam_opt.run(cf2, theta0.copy(), max_steps=max_steps, lr=DEFAULT_LR,
                                    seed=seed, n_seeds=n_seeds, exact_E=exact_E,
                                    method_name=f"Scaling {nq}q (Adam)")
            adam_gaps.append(abs(r2.energy_opt - exact_E))

        stat = wilcoxon_pairwise(sigma_gaps, adam_gaps, "Adam")
        wp   = stat["p_value"] if stat else float("nan")
        direction = stat["direction"] if stat else "SKIP (n<3)"

        scaling_results[nq] = dict(
            sigma_gap_mean=float(np.mean(sigma_gaps)),
            sigma_gap_std=float(np.std(sigma_gaps)),
            adam_gap_mean=float(np.mean(adam_gaps)),
            adam_gap_std=float(np.std(adam_gaps)),
            p=nq * reps, n_seeds=n_seeds,
            wilcoxon_p=wp, wilcoxon_direction=direction,
        )
        sig_tag = " *" if (wp == wp and wp < 0.05) else ""
        print(f"  n_qubits={nq}  p={nq*reps}  "
              f"SIGMA={np.mean(sigma_gaps):.5f}±{np.std(sigma_gaps):.5f}  "
              f"Adam={np.mean(adam_gaps):.5f}±{np.std(adam_gaps):.5f}  "
              f"p={wp:.4f}  {direction}{sig_tag}")

    sc_path = os.path.join(results_dir, f"scaling_results{sfx}.csv")
    with open(sc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_qubits", "p", "n_seeds",
                                           "sigma_gap_mean", "sigma_gap_std",
                                           "adam_gap_mean", "adam_gap_std",
                                           "wilcoxon_p", "wilcoxon_direction"])
        w.writeheader()
        for nq, d in scaling_results.items():
            w.writerow({"n_qubits": nq, **d})
    print(f"  [Scaling] Saved → {sc_path}")
    return scaling_results
