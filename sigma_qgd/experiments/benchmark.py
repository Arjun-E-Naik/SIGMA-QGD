"""
Main benchmark runner.

Runs a list of optimizers over N seeds on a given Hamiltonian and
writes raw CSVs to results_dir. Does NOT produce any plots.

Plots are generated separately by scripts/generate_plots.py reading the CSVs.
"""

from __future__ import annotations

import io
import sys
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper) and stream.encoding and stream.encoding.lower() != "utf-8":
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

import csv
import os
from typing import Dict, List, Optional

import numpy as np
try:
    from ..core.config import derive_config
    from ..core.device import make_device
    from ..core.evaluator import CostEvaluator
    from ..core.logging import ParameterLogger
    from ..optimizers.registry import get_optimizer, KNOWN_METHODS
    from ..analysis.stats import wilcoxon_pairwise, holm_bonferroni_correction
    from ..problems.hamiltonians import exact_gs
    from ..style import DEFAULT_LR, DEFAULT_SEEDS, BENCHMARK_METHODS, TOP2_BASELINE_SET, get_optimal_lr, OPTIMAL_LRS
    from ..core.utils import get_initial_theta
except ImportError:
    raise ImportError("Could not import required modules. Please install sigma-qgd first.")


def resolve_optimizer_lr(name: str, ham_type: str, n_qubits: int, baseline_lr=None, use_optimal_lrs: bool = True) -> float:
    """Resolve the exact optimal learning rate for a method given the Hamiltonian and system size.
    Guarantees no mismatch when running both TFIM and XXZ at 10q and 12q:
      - TFIM (all sizes): RMSProp=0.01, AdaBelief=0.1, Adam=0.1, Yogi=0.1, Diag-Hess=0.1, H-QNG=0.001
      - XXZ 10 qubits:   RMSProp=0.05, AdaBelief=0.1, Adam=0.1, Yogi=0.1, Diag-Hess=0.1, H-QNG=0.001
      - XXZ 12 qubits:   RMSProp=0.05, AdaBelief=0.01, Adam=0.1, Yogi=0.1, Diag-Hess=0.1, H-QNG=0.001
      - SIGMA-QGD v9.0:  0.1 across all configurations
    """
    if name == "SIGMA-QGD v9.0":
        return 0.1

    if use_optimal_lrs:
        return get_optimal_lr(name, ham_type, n_qubits)

    if isinstance(baseline_lr, dict):
        ht = "xxz" if ("xxz" in str(ham_type).lower() or "heis" in str(ham_type).lower()) else "tfim"
        if ht in baseline_lr and isinstance(baseline_lr[ht], dict):
            return baseline_lr[ht].get(name, get_optimal_lr(name, ht, n_qubits))
        return baseline_lr.get(name, get_optimal_lr(name, ht, n_qubits))

    if baseline_lr is not None and baseline_lr not in (DEFAULT_LR, 0.01, None):
        return baseline_lr

    return get_optimal_lr(name, ham_type, n_qubits)


def run_benchmark(
    H,
    ham_name: str,
    ham_type: str,
    n_qubits: int,
    reps: int,
    results_dir: str,
    tag: str = "",
    n_seeds: int = DEFAULT_SEEDS,
    max_steps: int = 200,
    methods: Optional[List[str]] = None,
    sigma_kwargs: Optional[dict] = None,
    baseline_lr: float = DEFAULT_LR,
    use_optimal_lrs: bool = True,
    logger: Optional[ParameterLogger] = None,
    device=None,
) -> tuple:
    """
    Run all specified methods for n_seeds seeds and save results.

    Parameters
    ----------
    H            : qml.Hamiltonian
    ham_name     : str  Human-readable name for plots/CSV ("TFIM (J=1, h=0.5)")
    ham_type     : str  Internal type ("tfim" or "xxz")
    n_qubits     : int
    reps         : int
    results_dir  : str  Output directory for CSVs.
    tag          : str  Suffix for output filenames.
    n_seeds      : int
    max_steps    : int
    methods      : list[str] or None   None → use BENCHMARK_METHODS
    sigma_kwargs : dict or None        Extra kwargs forwarded to SIGMA-QGD
    baseline_lr  : float or dict       LR for all gradient-based baselines
    use_optimal_lrs : bool             If True, automatically resolve optimal LRs per (ham_type, n_qubits)
    logger       : ParameterLogger     Optional per-step logger
    device       : pennylane.Device    Optional pre-allocated device instance

    Returns
    -------
    all_res  : dict[str, list[Result]]
    summary  : dict[str, dict]
    exact_E  : float
    """
    os.makedirs(results_dir, exist_ok=True)
    exact_E  = exact_gs(H)
    methods  = methods or list(BENCHMARK_METHODS)
    sigma_kwargs = sigma_kwargs or {}
    sfx = f"_{tag}" if tag else ""

    if device is None:
        device, _ = make_device(n_qubits)

    cfg = derive_config(n_qubits, reps, max_steps, ham_type)
    print(f"\n{'='*78}")
    print(f"  {ham_name}  |  n_qubits={n_qubits}  p={cfg['p']}  |  Exact: {exact_E:.4f} H")
    print(f"  W={cfg['W']}  t_qfim={cfg['t_qfim']}  beta={cfg['beta']:.3f}  "
          f"clip={cfg['clip_norm']:.2f}  warmup={cfg['warmup']}  |  n_seeds={n_seeds}")
    print(f"  methods={methods}")
    print(f"{'-'*78}")
    print(f"  Assigned Optimizer Learning Rates (Resolved for {ham_type.upper()} at {n_qubits} qubits):")
    
    # Pre-resolve and verify learning rates for every optimizer
    resolved_lrs = {}
    for name in methods:
        opt = get_optimizer(name)
        if opt.supports_lr:
            m_lr = resolve_optimizer_lr(name, ham_type, n_qubits, baseline_lr, use_optimal_lrs)
            resolved_lrs[name] = m_lr
            if name == "SIGMA-QGD v9.0":
                skw = dict(eta_0=m_lr, multiplier=1.0, lambda_reg=0.005)
                skw.update(sigma_kwargs)
                print(f"    ✓ {name:20s}: lr = {m_lr:<6} (c={skw.get('multiplier', 1.0)}, λ_reg={skw.get('lambda_reg', 0.005)})")
            else:
                print(f"    ✓ {name:20s}: lr = {m_lr:<6} [Optimal for {ham_type.upper()} {n_qubits}q]")
        else:
            resolved_lrs[name] = "N/A"
            print(f"    ✓ {name:20s}: [Derivative-free / Population / Schedule]")
    print(f"{'='*78}")

    all_res: Dict[str, List] = {m: [] for m in methods}
    run_summaries = []
    n_params = n_qubits * reps

    for seed in range(n_seeds):
        theta0 = get_initial_theta(n_params, seed)
        print(f"\n  Seed {seed}  |theta0|={np.linalg.norm(theta0):.3f}")

        for name in methods:
            cf  = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)
            opt = get_optimizer(name)

            # Build kwargs for this optimizer
            run_kw = dict(
                max_steps=max_steps,
                ham_type=ham_type,
                n_qubits=n_qubits,
                reps=reps,
                seed=seed,
                n_seeds=n_seeds,
                exact_E=exact_E,
                method_name=name,
            )
            if opt.supports_lr:
                if name == "SIGMA-QGD v9.0":
                    skw = dict(eta_0=resolved_lrs[name], multiplier=1.0, lambda_reg=0.005)
                    skw.update(sigma_kwargs)
                    run_kw.update(skw)
                else:
                    run_kw["lr"] = resolved_lrs[name]

            result, recs, _ = opt.run(cf, theta0.copy(), **run_kw)

            final_gap = abs(result.energy_opt - exact_E)
            if recs and logger:
                logger.finalise_run(recs, final_gap, result.n_steps)
            all_res[name].append(result)

            run_summaries.append(dict(
                ham=ham_name, method=name, seed=seed,
                n_qubits=n_qubits, n_seeds=n_seeds,
                lr=resolved_lrs.get(name, "N/A"),
                energy_opt=round(result.energy_opt, 6),
                exact_E=round(exact_E, 6),
                gap=round(final_gap, 6),
                n_circuits=result.n_circuits,
                n_steps=result.n_steps,
                wall_time=round(result.wall_time, 3),
            ))

    if logger:
        logger.run_summary_csv(run_summaries, tag=tag)

    # Aggregate summary
    summary = {}
    for name in methods:
        if not all_res[name]:
            continue
        energies = [r.energy_opt for r in all_res[name]]
        gaps     = [abs(e - exact_E) for e in energies]
        summary[name] = dict(
            lr=resolved_lrs.get(name, "N/A"),
            energy_mean=float(np.mean(energies)),
            energy_std=float(np.std(energies)),
            energy_all=energies,
            gap_mean=float(np.mean(gaps)),
            gap_std=float(np.std(gaps)),
            gap_all=gaps,
            steps_mean=float(np.mean([r.n_steps for r in all_res[name]])),
            circuits_mean=float(np.mean([r.n_circuits for r in all_res[name]])),
            circuits_all=[r.n_circuits for r in all_res[name]],
            time_mean=float(np.mean([r.wall_time for r in all_res[name]])),
        )

    _save_benchmark_csv(summary, exact_E, ham_name, n_seeds, n_qubits,
                        results_dir, sfx)
    _save_run_summary_csv(run_summaries, results_dir, sfx)
    _run_and_save_significance_tests(all_res, exact_E, ham_name, n_qubits,
                                     n_seeds, results_dir, sfx)
    _print_results_table(summary, exact_E, ham_name, n_seeds, n_qubits)

    return all_res, summary, exact_E






def _save_benchmark_csv(summary, exact_E, ham_name, n_seeds, n_qubits,
                         results_dir, sfx):
    safe = (ham_name.replace(" ", "_").replace("(", "").replace(")", "")
            .replace(",", "").replace("=", ""))
    path = os.path.join(results_dir, f"benchmark_{safe}{sfx}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "lr", "energy_mean", "energy_std", "gap_mean",
                    "gap_std", "steps_mean", "circuits_mean", "time_mean",
                    "n_seeds", "n_qubits", "exact_E"])
        for name, s in summary.items():
            w.writerow([name,
                        s.get("lr", "N/A"),
                        round(s["energy_mean"], 6), round(s["energy_std"], 6),
                        round(s["gap_mean"], 6),    round(s["gap_std"], 6),
                        round(s["steps_mean"], 1),  round(s["circuits_mean"], 1),
                        round(s["time_mean"], 3),   n_seeds, n_qubits,
                        round(exact_E, 6)])
    print(f"  [Benchmark] Saved → {path}")


def _save_run_summary_csv(summaries, results_dir, sfx):
    if not summaries:
        return
    path = os.path.join(results_dir, f"run_summary{sfx}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    print(f"  [Benchmark] Run summary → {path}")


def _run_and_save_significance_tests(all_res, exact_E, ham_name, n_qubits,
                                      n_seeds, results_dir, sfx):
    sigma_key = "SIGMA-QGD v9.0"
    if sigma_key not in all_res or not all_res[sigma_key]:
        return
    sigma_gaps = [abs(r.energy_opt - exact_E) for r in all_res[sigma_key]]
    rows = []
    raw_pvals = []
    names_for_correction = []
    for name, res_list in all_res.items():
        if name == sigma_key or not res_list:
            continue
        base_gaps = [abs(r.energy_opt - exact_E) for r in res_list]
        result = wilcoxon_pairwise(sigma_gaps, base_gaps, name)
        if result:
            rows.append({**result, "ham": ham_name, "n_qubits": n_qubits, "n_seeds": n_seeds})
            raw_pvals.append(result["p_value"])
            names_for_correction.append(name)

    # Holm-Bonferroni step-down correction
    if raw_pvals:
        corrected = holm_bonferroni_correction(raw_pvals)
        for row, cp in zip(rows, corrected):
            row["p_value_holm"] = round(cp, 6)

    if not rows:
        return
    path = os.path.join(results_dir, f"significance_tests{sfx}.csv")
    fields = ["ham", "baseline", "n_qubits", "n_seeds", "statistic",
              "p_value", "p_value_holm", "mean_diff", "n", "direction"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"  [Significance] Saved (with Holm correction) → {path}")


def _print_results_table(summary, exact_E, ham_name, n_seeds, n_qubits):
    print(f"\n{'='*78}")
    print(f"  RESULTS -- {ham_name}  |  n_qubits={n_qubits}  |  "
          f"Exact: {exact_E:.6f} H  |  {n_seeds} seeds")
    print(f"  {'Method':<22} {'LR':<8} {'E mean±std':>22}  {'Gap':>9}  {'Steps':>7}  {'Circuits':>10}")
    print(f"  {'─'*22} {'─'*8} {'─'*22}  {'─'*9}  {'─'*7}  {'─'*10}")
    for name, s in sorted(summary.items(), key=lambda x: x[1]["gap_mean"]):
        tag = " *" if name == "SIGMA-QGD v9.0" else ""
        lr_val = str(s.get("lr", "N/A"))
        print(f"  {name:<22} {lr_val:<8} {s['energy_mean']:>10.4f}±{s['energy_std']:<8.4f}"
              f"  {s['gap_mean']:>9.5f}  {s['steps_mean']:>7.1f}"
              f"  {s['circuits_mean']:>10.0f}{tag}")

