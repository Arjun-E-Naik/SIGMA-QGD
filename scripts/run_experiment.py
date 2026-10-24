"""
Main CLI entry point for running experiments.

Usage:
    python scripts/run_experiment.py --config configs/benchmark_tfim_10q.yaml
    python scripts/run_experiment.py --config configs/lr_sweep.yaml
    python scripts/run_experiment.py --config configs/ablation.yaml
"""

import argparse
import io
import os
import sys

import yaml

# Ensure UTF-8 encoding for console output on Windows
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper) and stream.encoding and stream.encoding.lower() != "utf-8":
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sigma_qgd.problems.hamiltonians import build_tfim, build_xxz, exact_gs
from sigma_qgd.problems.ansatz import build_ansatz
from sigma_qgd.core.logging import ParameterLogger
from sigma_qgd.core.device import make_device
from sigma_qgd.optimizers.registry import OPTIMIZER_REGISTRY, KNOWN_METHODS
from sigma_qgd.style import BENCHMARK_METHODS, DEFAULT_LR, DEFAULT_SEEDS

# Experiment runners
from sigma_qgd.experiments.benchmark     import run_benchmark
from sigma_qgd.experiments.ablation      import run_ablation, run_precond_variance_ablation
from sigma_qgd.experiments.sensitivity   import run_sensitivity_sweep
from sigma_qgd.experiments.shot_noise    import run_shot_noise_experiment, run_js_shrinkage_under_noise
from sigma_qgd.experiments.scaling       import run_scaling_study
from sigma_qgd.experiments.hessian_corr  import run_hessian_correlation_experiment
from sigma_qgd.experiments.cma_sigma     import run_cma_sigma_experiment
from sigma_qgd.experiments.lr_sweep      import run_lr_sweep

# Plot generators
from sigma_qgd.analysis.plots import (
    plot_convergence, plot_boxplots, plot_efficiency,
    plot_ablation_heatmap, plot_shot_noise, plot_scaling,
    plot_lr_sweep, plot_sensitivity_sweep,
)

# Results directory: auto-detect Kaggle vs local environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_KAGGLE = (
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    or os.path.exists("/kaggle/working")
)
if IS_KAGGLE:
    DEFAULT_RESULTS_DIR = "/kaggle/working/sigma_qgd_results"
else:
    DEFAULT_RESULTS_DIR = os.path.join(
        os.path.dirname(PROJECT_ROOT), "sigma_qgd_results"
    )


def _build_hamiltonian(ham_type: str, n_qubits: int):
    if ham_type.lower() == "tfim":
        return build_tfim(n_qubits)
    if ham_type.lower() in ("xxz", "heisenberg"):
        return build_xxz(n_qubits)
    raise ValueError(f"Unknown hamiltonian type: {ham_type!r}. Use 'tfim' or 'xxz'.")


def _resolve_tag(tag: str, ht: str, is_both: bool) -> str:
    if not is_both:
        return tag
    if "tfim" in tag:
        return tag.replace("tfim", ht)
    if "xxz" in tag:
        return tag.replace("xxz", ht)
    if "both" in tag:
        return tag.replace("both", ht)
    if tag.endswith(f"_{ht}"):
        return tag
    return f"{tag}_{ht}"


def _validate_methods(methods):
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Valid: {sorted(KNOWN_METHODS)}"
        )
    return methods


def main():
    parser = argparse.ArgumentParser(
        description="SIGMA-QGD — run an experiment from a YAML config."
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to a YAML config file, e.g. configs/benchmark_tfim_10q.yaml")
    parser.add_argument("--results-dir", "-r", default=None,
                        help=f"Override results directory (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("--device", "-d", default=None,
                        choices=["auto", "gpu", "cpu", "qrack", "lightning.gpu", "lightning.qubit", "default.qubit"],
                        help="Force simulation backend (default: auto)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating figures (save CSVs only)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp_type    = cfg.get("experiment", "benchmark").lower()
    n_qubits    = cfg.get("n_qubits", 10)
    reps        = cfg.get("reps", 3)
    n_seeds     = cfg.get("n_seeds", DEFAULT_SEEDS)
    max_steps   = cfg.get("max_steps", 200)
    ham_type    = cfg.get("hamiltonian", "tfim").lower()
    tag         = cfg.get("out_tag", f"{ham_type}_{n_qubits}q")
    methods       = cfg.get("methods", None)
    baselines_cfg   = cfg.get("baselines", {})
    use_optimal_lrs = baselines_cfg.get("use_optimal_lrs", True)
    baseline_lr     = baselines_cfg.get("lr_by_method", baselines_cfg.get("lr", None))
    sigma_cfg       = cfg.get("sigma_qgd", {})

    results_dir = args.results_dir or cfg.get("results_dir", DEFAULT_RESULTS_DIR)
    data_dir    = os.path.join(results_dir, "data")
    figs_dir    = os.path.join(results_dir, "figures")

    try:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)
    except (PermissionError, OSError) as e:
        fallback_dir = "/kaggle/working/sigma_qgd_results" if os.path.exists("/kaggle/working") else os.path.join(PROJECT_ROOT, "sigma_qgd_results")
        print(f"  [Notice] Output directory '{results_dir}' is read-only ({e}). Automatically redirecting to: {fallback_dir}")
        results_dir = fallback_dir
        data_dir    = os.path.join(results_dir, "data")
        figs_dir    = os.path.join(results_dir, "figures")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(figs_dir, exist_ok=True)

    if methods:
        _validate_methods(methods)

    # Probe backend with device preference
    device_pref = args.device or cfg.get("device", "auto")
    dev, gpu_status = make_device(max(n_qubits, 1), device_pref=device_pref, verbose=True)

        print(f"  SIGMA-QGD v9.0 — {exp_type.upper()}")
    print(f"  Config: {args.config}")
    print(f"  Backend: {gpu_status}")
    print(f"  n_qubits={n_qubits}  reps={reps}  n_seeds={n_seeds}  max_steps={max_steps}")
    print(f"  hamiltonian={ham_type}  use_optimal_lrs={use_optimal_lrs}")
    print(f"  Results → {results_dir}")
    
    logger = ParameterLogger(data_dir)


    
    if exp_type in ("benchmark", "baseline"):
        ham_types  = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        all_results = {}

        for ht in ham_types:
            H        = _build_hamiltonian(ht, n_qubits)
            ham_name = "TFIM (J=1, h=0.5)" if ht == "tfim" else "XXZ (Jxy=1, Jz=0.5, h=0.1)"
            all_res, summary, exact_E = run_benchmark(
                H, ham_name, ht, n_qubits, reps,
                data_dir, tag=tag, n_seeds=n_seeds, max_steps=max_steps,
                methods=methods, sigma_kwargs=sigma_cfg,
                baseline_lr=baseline_lr, use_optimal_lrs=use_optimal_lrs,
                logger=logger, device=dev,
            )
            all_results[ht] = (all_res, summary, exact_E)

        if not args.no_plots:
            r_t, s_t, e_t = all_results.get("tfim", (None, None, None))
            r_x, s_x, e_x = all_results.get("xxz",  (None, None, None))
            plot_convergence(r_t, e_t, r_x, e_x, n_seeds, n_qubits, figs_dir, tag)
            plot_boxplots(s_t, e_t, s_x, e_x, n_seeds, n_qubits, figs_dir, tag)
            if r_t and r_x:
                plot_efficiency(s_t, e_t, s_x, e_x, n_qubits, figs_dir, tag)

    elif exp_type == "ablation":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H = _build_hamiltonian(ht, n_qubits)
            exact_E = exact_gs(H)
            cur_tag = _resolve_tag(tag, ht, ham_type == "both")
            abl_results = run_ablation(
                H, ht.upper(), ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                eta_0=sigma_cfg.get("eta_0", 0.1),
                multiplier=sigma_cfg.get("multiplier", 1.0),
                lambda_reg=sigma_cfg.get("lambda_reg", 0.001),
                logger=logger, device=dev,
            )
            if not args.no_plots:
                plot_ablation_heatmap(abl_results, exact_E, n_qubits, figs_dir, cur_tag)

    elif exp_type == "precond_ablation":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H = _build_hamiltonian(ht, n_qubits)
            cur_tag = _resolve_tag(tag, ht, ham_type == "both")
            run_precond_variance_ablation(
                H, ht.upper(), ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                eta_0=sigma_cfg.get("eta_0", 0.1),
                multiplier=sigma_cfg.get("multiplier", 1.0),
                lambda_reg=sigma_cfg.get("lambda_reg", 0.001),
                logger=logger, device=dev,
            )

    elif exp_type == "sensitivity":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H = _build_hamiltonian(ht, n_qubits)
            h_name = "TFIM (J=1, h=0.5)" if ht == "tfim" else "XXZ (Jxy=1, Jz=0.5, h=0.1)"
            cur_tag = _resolve_tag(tag, ht, ham_type == "both")
            rows = run_sensitivity_sweep(
                H, h_name, ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                base_eta=sigma_cfg.get("eta_0", 0.05),
                base_lambda=sigma_cfg.get("lambda_reg", 0.001),
                base_multiplier=sigma_cfg.get("multiplier", 1.0),
                device=dev,
            )
            if not args.no_plots:
                plot_sensitivity_sweep(rows, h_name, n_qubits, figs_dir, cur_tag)

    elif exp_type == "shot_noise":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H = _build_hamiltonian(ht, n_qubits)
            cur_tag = _resolve_tag(tag, ht, ham_type == "both")
            shot_results = run_shot_noise_experiment(
                H, ht.upper(), ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                shot_counts=cfg.get("shot_counts", None),
            )
            if not args.no_plots:
                plot_shot_noise(shot_results, n_qubits, figs_dir, cur_tag)

    elif exp_type == "scaling":
        qubit_counts = cfg.get("scaling_qubits", [4, 6, 8, 10, 12])
        scaling_res  = run_scaling_study(
            data_dir, tag=tag, max_steps=max_steps, n_seeds=n_seeds,
            reps=reps, qubit_counts=qubit_counts,
        )
        if not args.no_plots:
            plot_scaling(scaling_res, figs_dir, tag)

    elif exp_type == "hessian_correlation":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H = _build_hamiltonian(ht, n_qubits)
            cur_tag = _resolve_tag(tag, ht, ham_type == "both")
            run_hessian_correlation_experiment(
                H, ht.upper(), ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                eta_0=sigma_cfg.get("eta_0", 0.1),
                multiplier=sigma_cfg.get("multiplier", 1.0),
                lambda_reg=sigma_cfg.get("lambda_reg", 0.001),
                device=dev,
            )

    elif exp_type == "cma_sigma":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        for ht in ham_types:
            H      = _build_hamiltonian(ht, n_qubits)
            h_name = "TFIM (J=1, h=0.5)" if ht == "tfim" else "XXZ (Jxy=1, Jz=0.5)"
            run_cma_sigma_experiment(
                H, h_name, ht, n_qubits, reps,
                data_dir, tag=f"{tag}_{ht}", max_steps=max_steps, n_seeds=n_seeds,
                n_shots=cfg.get("cma_shots", 1024),
                cma_sigma0=cfg.get("cma_sigma0", 0.3),
            )
            run_js_shrinkage_under_noise(
                H, h_name, ht, n_qubits, reps,
                data_dir, tag=f"{tag}_{ht}", max_steps=max_steps, n_seeds=n_seeds,
                n_shots=cfg.get("cma_shots", 1024),
            )

    elif exp_type == "lr_sweep":
        ham_types = ["tfim", "xxz"] if ham_type == "both" else [ham_type]
        lr_grid  = cfg.get("lr_values", None)
        for ht in ham_types:
            H        = _build_hamiltonian(ht, n_qubits)
            h_name   = "TFIM (J=1, h=0.5)" if ht == "tfim" else "XXZ (Jxy=1, Jz=0.5, h=0.1)"
            cur_tag  = _resolve_tag(tag, ht, ham_type == "both")
            summary  = run_lr_sweep(
                H, h_name, ht, n_qubits, reps,
                data_dir, tag=cur_tag, max_steps=max_steps, n_seeds=n_seeds,
                lr_grid=lr_grid, methods=methods, sigma_kwargs=sigma_cfg,
                device=dev,
            )
            if not args.no_plots:
                plot_lr_sweep(list(summary.values()) if not isinstance(
                    list(summary.values())[0], dict
                        ) else [
                            {"method": k, **v} for k, v in summary.items()
                        ], figs_dir, cur_tag)

    else:
        print(f"  [ERROR] Unknown experiment type: {exp_type!r}")
        sys.exit(1)

    logger.save(tag=tag)
    print(f"\n  Done. Results in {results_dir}/")


if __name__ == "__main__":
    main()
