"""
Budget-matched comparison between CMA-ES and SIGMA-QGD under finite shot noise.
"""

from __future__ import annotations
import csv
import os

import numpy as np

try:
    from ..core.evaluator import CostEvaluator, ShotNoiseCostEvaluator
    from ..optimizers.sigma_qgd import SigmaQGD
    from ..optimizers.derivative_free import CMAES
    from ..problems.hamiltonians import exact_gs
    from ..analysis.stats import wilcoxon_pairwise
    from ..style import ABLATION_MAX_SEEDS
    from ..core.utils import get_initial_theta
except ImportError:
    raise ImportError("Could not import required modules. Please install sigma-qgd first.  ")


def run_cma_sigma_experiment(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 200,
    n_seeds: int = 15,
    n_shots: int = 1024,
    cma_sigma0: float = 0.3,
) -> dict:
    """
    Budget-matched CMA-ES vs SIGMA-QGD under shot noise.
    Evaluates best parameters on the exact noiseless Hamiltonian.
    """
    n_seeds = min(n_seeds, ABLATION_MAX_SEEDS)
    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    sigma_opt = SigmaQGD()
    cma_opt   = CMAES()

    sigma_gaps = []; cma_gaps = []; per_seed = []

    

    for seed in range(n_seeds):
        # Both optimizers use the exact same starting theta0 for this seed.
        # get_initial_theta(seed=k) is always deterministic and independent of n_seeds.
        theta0 = get_initial_theta(n_params, seed)

        cf_sigma = ShotNoiseCostEvaluator(n_qubits, reps, H, n_shots=n_shots, exact_E=exact_E)
        r_sigma, _, bt_sigma = sigma_opt.run(cf_sigma, theta0.copy(),
                                              max_steps=max_steps, ham_type=ham_type,
                                              n_qubits=n_qubits, reps=reps, seed=seed,
                                              n_seeds=n_seeds, exact_E=exact_E,
                                              method_name="SIGMA-QGD (shots)")
        sigma_budget = cf_sigma.n_calls
        cf_ex1  = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
        sg      = abs(cf_ex1(bt_sigma) - exact_E)
        sigma_gaps.append(sg)

        # CMA-ES: same theta0, same shot evaluator, strict budget-match.
        # No seed offset — CMA internal seed is set via the `seed` kwarg directly.
        cf_cma = ShotNoiseCostEvaluator(n_qubits, reps, H, n_shots=n_shots, exact_E=exact_E)
        r_cma, _, bt_cma = cma_opt.run(cf_cma, theta0.copy(),
                                        total_circuit_budget=sigma_budget,
                                        sigma0=cma_sigma0, seed=seed,
                                        n_seeds=n_seeds, exact_E=exact_E,
                                        method_name="CMA-ES (shots)")
        cf_ex2 = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
        cg     = abs(cf_ex2(bt_cma) - exact_E)
        cma_gaps.append(cg)

        print(f"  seed={seed}  SIGMA gap={sg:.5f} ({sigma_budget} circ)  "
              f"CMA-ES gap={cg:.5f} ({cf_cma.n_calls} circ)")
        per_seed.append(dict(seed=seed,
                              sigma_gap=round(sg, 6), cma_gap=round(cg, 6),
                              sigma_circuits=sigma_budget, cma_circuits=cf_cma.n_calls))

    stat = wilcoxon_pairwise(sigma_gaps, cma_gaps, "CMA-ES")
    wp   = stat["p_value"] if stat else float("nan")
    direction = stat["direction"] if stat else "SKIP"

    print(f"\n  SIGMA gap: {np.mean(sigma_gaps):.5f} ± {np.std(sigma_gaps):.5f}")
    print(f"  CMA-ES gap: {np.mean(cma_gaps):.5f} ± {np.std(cma_gaps):.5f}")
    print(f"  Wilcoxon p={wp:.4f}  {direction}")

    cma_path = os.path.join(results_dir, f"cma_vs_sigma{sfx}.csv")
    with open(cma_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "sigma_gap", "cma_gap",
                                           "sigma_circuits", "cma_circuits"])
        w.writeheader(); w.writerows(per_seed)
    print(f"  [CMA-ES] Saved → {cma_path}")
    return {"sigma_gaps": sigma_gaps, "cma_gaps": cma_gaps,
            "wilcoxon_p": wp, "direction": direction}
