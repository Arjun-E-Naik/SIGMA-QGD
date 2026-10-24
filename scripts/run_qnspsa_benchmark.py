"""
Independent QN-SPSA Benchmark Runner & Updater.

Runs ONLY QN-SPSA across the exact 20 seeds (0 to 19) for:
  - TFIM 10q
  - XXZ 10q
  - TFIM 12q
  - XXZ 12q

Then cleanly updates:
  1. run_summaries_benchmark_both_{10q,12q}.csv
  2. benchmark_{TFIM,XXZ}_..._{10q,12q}.csv
  3. significance_tests_benchmark_both_{10q,12q}.csv
  4. Regenerates all consolidated figures (PDF/PNG) and LaTeX tables!

Usage:
  python scripts/run_qnspsa_benchmark.py                # Runs all 4 regimes (20 seeds each)
  python scripts/run_qnspsa_benchmark.py --qubits 10    # Runs only 10q (TFIM + XXZ)
  python scripts/run_qnspsa_benchmark.py --test         # Quick 1-seed smoke test
"""

import os
import sys
import time
import argparse
import shutil
import csv

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats

from sigma_qgd.problems.hamiltonians import build_tfim, build_xxz
from sigma_qgd.core.evaluator import CostEvaluator
from sigma_qgd.core.utils import get_initial_theta
from sigma_qgd.optimizers.qng_methods import QNSPSA
from scripts.generate_consolidated_benchmark import main as regenerate_benchmark_artifacts


EXACT_ENERGIES = {
    ("tfim", 10): -10.635604,
    ("tfim", 12): -12.762569,
    ("xxz", 10):  -15.276131,
    ("xxz", 12):  -18.229090,
}

HAM_BUILDERS = {
    "tfim": lambda n: build_tfim(n, J=1.0, h=0.5),
    "xxz":  lambda n: build_xxz(n, Jxy=1.0, Jz=0.5, h=0.1),
}

HAM_NAMES = {
    "tfim": "TFIM (J=1, h=0.5)",
    "xxz":  "XXZ (Jxy=1, Jz=0.5, h=0.1)",
}


def run_qnspsa_regime(ham_type: str, n_qubits: int, n_seeds: int = 20, max_steps: int = 200, reps: int = 3):
    exact_E = EXACT_ENERGIES[(ham_type, n_qubits)]
    ham_name = HAM_NAMES[ham_type]
    H = HAM_BUILDERS[ham_type](n_qubits)
    n_params = n_qubits * reps
    
    print(f"\n========================================================")
    print(f"RUNNING QN-SPSA: {ham_name} | {n_qubits} Qubits ({n_params} params) | {n_seeds} Seeds")
    print(f"========================================================")
    
    opt = QNSPSA()
    results = []
    
    for seed in range(n_seeds):
        t0 = time.time()
        theta0 = get_initial_theta(n_params, seed)
        cf = CostEvaluator(n_qubits, reps, H, exact_E=exact_E)
        
        # Run standard Gacon et al. (2021) rank-2 formulation with blocking
        result, _, _ = opt.run(
            cf, theta0.copy(),
            max_steps=max_steps,
            lr=0.05,
            lambda_reg=0.001,
            mode="full",
            blocking=True,
            silent=True
        )
        elapsed = time.time() - t0
        gap = abs(result.energy_opt - exact_E)
        
        results.append({
            "ham": ham_name,
            "method": "QN-SPSA",
            "seed": seed,
            "n_qubits": n_qubits,
            "n_seeds": n_seeds,
            "lr": "N/A",
            "energy_opt": round(result.energy_opt, 6),
            "exact_E": round(exact_E, 6),
            "gap": round(gap, 6),
            "n_circuits": result.n_circuits,
            "n_steps": result.n_steps,
            "wall_time": round(elapsed, 3),
        })
        print(f"  Seed {seed:2d}/20: E_opt = {result.energy_opt:10.6f} | Gap = {gap:8.6f} | Circuits = {result.n_circuits} | Time = {elapsed:5.2f}s")
        
    return results


def update_regime_csvs(q_str: str, new_results: list, base_dir: str = "sigma_data"):
    dir_path = os.path.join(base_dir, f"benchmark_{q_str}")
    run_sum_path = os.path.join(dir_path, f"run_summaries_benchmark_both_{q_str}.csv")
    
    if not os.path.exists(run_sum_path):
        print(f"  [ERROR] {run_sum_path} does not exist!")
        return

    # 1. Update run_summaries CSV
    # Backup first
    bak_path = run_sum_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copyfile(run_sum_path, bak_path)

    existing_rows = []
    with open(run_sum_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            existing_rows.append(r)

    # Filter out old QN-SPSA rows for the tested regimes
    new_regimes = set((r["ham"], r["n_qubits"]) for r in new_results)
    filtered_rows = [
        r for r in existing_rows
        if not (r["method"] == "QN-SPSA" and (r["ham"], int(r["n_qubits"])) in new_regimes)
    ]
    
    # Append updated QN-SPSA rows
    filtered_rows.extend([
        {
            "ham": r["ham"], "method": r["method"], "seed": str(r["seed"]),
            "n_qubits": str(r["n_qubits"]), "n_seeds": str(r["n_seeds"]),
            "lr": r["lr"], "energy_opt": str(r["energy_opt"]),
            "exact_E": str(r["exact_E"]), "gap": str(r["gap"]),
            "n_circuits": str(r["n_circuits"]), "n_steps": str(r["n_steps"]),
            "wall_time": str(r["wall_time"]),
        }
        for r in new_results
    ])

    fieldnames = ["ham", "method", "seed", "n_qubits", "n_seeds", "lr", "energy_opt", "exact_E", "gap", "n_circuits", "n_steps", "wall_time"]
    with open(run_sum_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    print(f"  Updated: {run_sum_path}")

    # Also update run_summary (singular) if present
    run_sum_sing = os.path.join(dir_path, f"run_summary_benchmark_both_{q_str}.csv")
    if os.path.exists(run_sum_sing):
        shutil.copyfile(run_sum_path, run_sum_sing)

    # 2. Update benchmark summary CSVs (benchmark_TFIM_... and benchmark_XXZ_...)
    for h_type in ["tfim", "xxz"]:
        h_name = HAM_NAMES[h_type]
        reg_new = [r for r in new_results if r["ham"] == h_name]
        if not reg_new:
            continue
            
        gaps = [r["gap"] for r in reg_new]
        energies = [r["energy_opt"] for r in reg_new]
        circuits = [r["n_circuits"] for r in reg_new]
        times = [r["wall_time"] for r in reg_new]
        steps = [r["n_steps"] for r in reg_new]
        
        # Locate corresponding benchmark CSV file
        pattern_prefix = f"benchmark_TFIM" if h_type == "tfim" else "benchmark_XXZ"
        for fname in os.listdir(dir_path):
            if fname.startswith(pattern_prefix) and fname.endswith(f"_{q_str}.csv"):
                bench_csv_path = os.path.join(dir_path, fname)
                bench_rows = []
                with open(bench_csv_path, "r", encoding="utf-8") as f:
                    b_reader = csv.DictReader(f)
                    b_fields = b_reader.fieldnames
                    for brow in b_reader:
                        if brow["method"] == "QN-SPSA":
                            brow["energy_mean"] = str(round(float(np.mean(energies)), 6))
                            brow["energy_std"] = str(round(float(np.std(energies, ddof=1)), 6))
                            brow["gap_mean"] = str(round(float(np.mean(gaps)), 6))
                            brow["gap_std"] = str(round(float(np.std(gaps, ddof=1)), 6))
                            brow["circuits_mean"] = str(round(float(np.mean(circuits)), 1))
                            brow["time_mean"] = str(round(float(np.mean(times)), 3))
                            brow["steps_mean"] = str(round(float(np.mean(steps)), 1))
                            brow["n_seeds"] = str(len(gaps))
                        bench_rows.append(brow)
                
                with open(bench_csv_path, "w", newline="", encoding="utf-8") as f:
                    b_writer = csv.DictWriter(f, fieldnames=b_fields)
                    b_writer.writeheader()
                    b_writer.writerows(bench_rows)
                print(f"  Updated: {bench_csv_path}")

    # 3. Update significance tests for QN-SPSA vs SIGMA-QGD
    sig_path = os.path.join(dir_path, f"significance_tests_benchmark_both_{q_str}.csv")
    if os.path.exists(sig_path):
        sig_rows = []
        with open(sig_path, "r", encoding="utf-8") as f:
            s_reader = csv.DictReader(f)
            s_fields = s_reader.fieldnames
            for srow in s_reader:
                h_target = srow["ham"]
                if srow["baseline"] == "QN-SPSA":
                    # Recompute Wilcoxon signed-rank vs SIGMA-QGD
                    sigma_gaps = [
                        float(r["gap"]) for r in filtered_rows
                        if r["method"] == "SIGMA-QGD v9.0" and r["ham"] == h_target
                    ]
                    qn_gaps = [
                        float(r["gap"]) for r in filtered_rows
                        if r["method"] == "QN-SPSA" and r["ham"] == h_target
                    ]
                    if len(sigma_gaps) == len(qn_gaps) and len(sigma_gaps) > 0:
                        diffs = np.array(sigma_gaps) - np.array(qn_gaps)
                        try:
                            stat, p_val = stats.wilcoxon(diffs, alternative="two-sided")
                        except Exception:
                            stat, p_val = 0.0, 1.0
                        srow["statistic"] = str(round(float(stat), 1))
                        srow["p_value"] = f"{p_val:.2e}"
                        srow["p_value_holm"] = f"{p_val:.2e}"
                        srow["mean_diff"] = str(round(float(np.mean(diffs)), 6))
                        srow["direction"] = "SIGMA better" if np.mean(diffs) < 0 else "QN-SPSA better"
                sig_rows.append(srow)

        with open(sig_path, "w", newline="", encoding="utf-8") as f:
            s_writer = csv.DictWriter(f, fieldnames=s_fields)
            s_writer.writeheader()
            s_writer.writerows(sig_rows)
        print(f"  Updated: {sig_path}")


def main():
    parser = argparse.ArgumentParser(description="Run QN-SPSA benchmark independently and update paper results.")
    parser.add_argument("--qubits", choices=["10", "12", "all"], default="all", help="Target qubit size")
    parser.add_argument("--ham", choices=["tfim", "xxz", "both"], default="both", help="Target Hamiltonian")
    parser.add_argument("--seeds", type=int, default=20, help="Number of random seeds (default: 20)")
    parser.add_argument("--steps", type=int, default=200, help="Iterations per run (default: 200)")
    parser.add_argument("--test", action="store_true", help="Quick 1-seed test mode")
    args = parser.parse_args()

    n_seeds = 1 if args.test else args.seeds
    max_steps = 50 if args.test else args.steps

    qubit_list = [10, 12] if args.qubits == "all" else [int(args.qubits)]
    ham_list = ["tfim", "xxz"] if args.ham == "both" else [args.ham]

    all_runs = {}
    for nq in qubit_list:
        q_str = f"{nq}q"
        all_runs[q_str] = []
        for ht in ham_list:
            res = run_qnspsa_regime(ht, nq, n_seeds=n_seeds, max_steps=max_steps)
            all_runs[q_str].extend(res)
        
        if not args.test:
            update_regime_csvs(q_str, all_runs[q_str])

    if not args.test:
        print("\nRegenerating consolidated publication figures and LaTeX tables...")
        regenerate_benchmark_artifacts()
        print("\nAll benchmark artifacts successfully refreshed with updated QN-SPSA!")


if __name__ == "__main__":
    main()
