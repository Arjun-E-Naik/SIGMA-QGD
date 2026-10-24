"""
Shot-noise robustness experiments and James-Stein shrinkage evaluations.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

try:
    from ..core.evaluator import CostEvaluator, ShotNoiseCostEvaluator
    from ..core.utils import get_initial_theta
    from ..optimizers.sigma_qgd import SigmaQGD
    from ..problems.hamiltonians import exact_gs
    from ..analysis.stats import wilcoxon_pairwise
except ImportError:
    raise ImportError("Failed to import required modules from sigma_qgd.")


def run_shot_noise_experiment(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 100,
    n_seeds: int = 15,
    shot_counts: Optional[list] = None,
) -> dict:
    """
    Evaluate optimizer convergence under varying finite shot noise levels.
    """
    if shot_counts is None:
        shot_counts = [100, 500, 1000, 5000, 10000]

    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    results = {ns: [] for ns in shot_counts}
    results["noiseless"] = []
    exact_gaps = {ns: [] for ns in shot_counts}
    exact_gaps["noiseless"] = []

    print(
        f"\nShot-noise evaluation: {ham_name} (qubits={n_qubits}, seeds={n_seeds})\n"
        f"Shot budgets: {shot_counts}"
    )

    opt = SigmaQGD()

    for seed in range(n_seeds):
        theta0 = get_initial_theta(n_params, seed)

        # Noiseless reference evaluation
        cf_nl = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
        r_nl, _, bt_nl = opt.run(
            cf_nl,
            theta0.copy(),
            max_steps=max_steps,
            ham_type=ham_type,
            n_qubits=n_qubits,
            reps=reps,
            seed=seed,
            n_seeds=n_seeds,
            exact_E=exact_E,
            method_name="ShotNoise: noiseless",
        )
        cf_exact = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
        gap_nl = abs(cf_exact(bt_nl) - exact_E)
        results["noiseless"].append(r_nl)
        exact_gaps["noiseless"].append(gap_nl)

        for ns in shot_counts:
            cf_noisy = ShotNoiseCostEvaluator(n_qubits, reps, H, n_shots=ns, exact_E=exact_E)
            r_noisy, _, bt_noisy = opt.run(
                cf_noisy,
                theta0.copy(),
                max_steps=max_steps,
                ham_type=ham_type,
                n_qubits=n_qubits,
                reps=reps,
                seed=seed,
                n_seeds=n_seeds,
                exact_E=exact_E,
                method_name=f"ShotNoise: {ns} shots",
            )
            cf_exact2 = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
            exact_e_noisy = cf_exact2(bt_noisy)
            gap_exact = abs(exact_e_noisy - exact_E)
            results[ns].append(r_noisy)
            exact_gaps[ns].append(gap_exact)

    noiseless_exact = np.array(exact_gaps["noiseless"])
    all_results = {}

    print(f"\n{'Shots':<12} {'Exact Gap':>10} {'Gap std':>10} {'Wilcoxon p':>11}  Direction")

    for key in ["noiseless"] + shot_counts:
        gaps = np.array(exact_gaps[key])
        label = str(key) if key != "noiseless" else "Noiseless"
        wilcoxon_p = float("nan")
        direction = "-- (reference)"
        if key != "noiseless" and len(gaps) >= 3:
            stat = wilcoxon_pairwise(list(noiseless_exact), list(gaps), str(key))
            if stat:
                wilcoxon_p = stat["p_value"]
                direction = "fewer shots better" if stat["mean_diff"] > 0 else "noiseless better"
        sig_tag = " *" if (wilcoxon_p == wilcoxon_p and wilcoxon_p < 0.05) else ""
        p_str = f"{wilcoxon_p:.4f}" if wilcoxon_p == wilcoxon_p else "--"
        print(f"  {label:<12} {np.mean(gaps):>10.5f} {np.std(gaps):>10.5f}  {p_str:>11}  {direction}{sig_tag}")
        all_results[key] = dict(
            gap_mean=float(np.mean(gaps)),
            gap_std=float(np.std(gaps)),
            n_seeds=len(gaps),
            n_qubits=n_qubits,
            wilcoxon_p=wilcoxon_p,
            direction=direction,
        )

    sn_path = os.path.join(results_dir, f"shot_noise_results{sfx}.csv")
    with open(sn_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_shots", "gap_mean", "gap_std", "n_seeds", "n_qubits", "wilcoxon_p", "direction"])
        for key in ["noiseless"] + shot_counts:
            d = all_results[key]
            w.writerow([
                "inf" if key == "noiseless" else key,
                round(d["gap_mean"], 6),
                round(d["gap_std"], 6),
                d["n_seeds"],
                d["n_qubits"],
                round(d["wilcoxon_p"], 6) if d["wilcoxon_p"] == d["wilcoxon_p"] else "",
                d["direction"],
            ])
    print(f"  [Shot-noise] Saved -> {sn_path}")
    return all_results


def run_js_shrinkage_under_noise(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    max_steps: int = 100,
    n_seeds: int = 15,
    n_shots: int = 1024,
) -> dict:
    """Compare raw vs James-Stein shrunk preconditioner under shot noise."""
    exact_E = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    raw_gaps = []
    shrunk_gaps = []
    per_seed_rows = []

    for seed in range(n_seeds):
        theta0 = get_initial_theta(n_params, seed)

        for pv, gap_list in [("raw", raw_gaps), ("shrunk", shrunk_gaps)]:
            cf = ShotNoiseCostEvaluator(n_qubits, reps, H, n_shots=n_shots, exact_E=exact_E)
            opt = SigmaQGD(precond_variance=pv)
            r, _, bt = opt.run(
                cf,
                theta0.copy(),
                max_steps=max_steps,
                ham_type=ham_type,
                n_qubits=n_qubits,
                reps=reps,
                seed=seed,
                n_seeds=n_seeds,
                exact_E=exact_E,
                method_name=f"JS-Shrinkage: {pv}",
            )
            cf_ex = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
            gap = abs(cf_ex(bt) - exact_E)
            gap_list.append(gap)

        per_seed_rows.append(dict(
            seed=seed,
            raw_gap=round(raw_gaps[-1], 6),
            shrunk_gap=round(shrunk_gaps[-1], 6),
        ))
        print(f"  seed={seed}  raw_gap={raw_gaps[-1]:.5f}  shrunk_gap={shrunk_gaps[-1]:.5f}")

    stat = wilcoxon_pairwise(raw_gaps, shrunk_gaps, "shrunk")
    wp = stat["p_value"] if stat else float("nan")
    direction = ("shrunk better" if stat and stat["mean_diff"] > 0 else "raw better") if stat else "N/A"

    js_path = os.path.join(results_dir, f"js_shrinkage_noise{sfx}.csv")
    with open(js_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "raw_gap", "shrunk_gap"])
        w.writeheader()
        w.writerows(per_seed_rows)
    print(f"  [JS-Shrinkage] Saved -> {js_path}")
    return {
        "raw_gaps": raw_gaps,
        "shrunk_gaps": shrunk_gaps,
        "wilcoxon_p": wp,
        "direction": direction,
    }
