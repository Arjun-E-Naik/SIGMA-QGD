"""
Ablation study experiments assessing individual algorithmic components.
"""

from __future__ import annotations
import csv
import os
from typing import Dict, List, Optional

import numpy as np
try:
    from ..core.config import derive_config
    from ..core.device import make_device
    from ..core.evaluator import CostEvaluator
    from ..core.logging import ParameterLogger
    from ..optimizers.sigma_qgd import SigmaQGD
    from ..problems.hamiltonians import exact_gs
    from ..analysis.stats import wilcoxon_pairwise
    from ..style import ABLATION_MAX_SEEDS
    from ..core.utils import get_initial_theta
except ImportError:
    raise ImportError("Could not import required modules. Please install sigma-qgd first.")

def run_ablation(
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
    logger: Optional[ParameterLogger] = None,
    device=None,
) -> Dict:
    """
    Component ablation: Full v9.0 vs No-preconditioner vs No-momentum vs
    No-clipping vs With-drift-correction.

    Returns dict mapping config_name → list[Result].
    """
    n_seeds  = min(n_seeds, ABLATION_MAX_SEEDS)
    exact_E  = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    if device is None:
        device, _ = make_device(n_qubits)

    configs = {
        "Full v9.0":           dict(use_precond=True,  use_momentum=True,
                                    use_clip=True,  use_drift_correction=False),
        "No preconditioner":   dict(use_precond=False, use_momentum=True,
                                    use_clip=True,  use_drift_correction=False),
        "No momentum":         dict(use_precond=True,  use_momentum=False,
                                    use_clip=True,  use_drift_correction=False),
        "No clipping":         dict(use_precond=True,  use_momentum=True,
                                    use_clip=False, use_drift_correction=False),
        "With drift correct.": dict(use_precond=True,  use_momentum=True,
                                    use_clip=True,  use_drift_correction=True),
    }
    results = {k: [] for k in configs}

    

    for seed in range(n_seeds):
        theta0 = get_initial_theta(n_params, seed)
        for cname, cflags in configs.items():
            cf  = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)
            opt = SigmaQGD(precond_multiplier=multiplier, **cflags)
            res, recs, _ = opt.run(cf, theta0.copy(),
                                   max_steps=max_steps, ham_type=ham_type,
                                   n_qubits=n_qubits, reps=reps, seed=seed,
                                   n_seeds=n_seeds, exact_E=exact_E,
                                   eta_0=eta_0, lambda_reg=lambda_reg, multiplier=multiplier,
                                   method_name=f"Ablation: {cname}")
            if recs and logger:
                logger.finalise_run(recs, abs(res.energy_opt - exact_E), res.n_steps)
            results[cname].append(res)

    full_gaps = [abs(r.energy_opt - exact_E) for r in results["Full v9.0"]]
    full_gap  = float(np.mean(full_gaps))

    csv_rows  = []
    sig_rows  = []
    print(f"\n  {'Config':<25} {'Gap mean':>10} {'Gap std':>10} "
          f"{'Circuits':>10} {'Ratio':>8} {'p-raw':>8} {'p-holm':>8}  Interpretation")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*28}")

    raw_pvals = []
    for cname, rs in results.items():
        gaps  = [abs(r.energy_opt - exact_E) for r in rs]
        circs = [r.n_circuits for r in rs]
        ratio = float(np.mean(gaps)) / max(full_gap, 1e-12)
        interp = "(base)" if cname == "Full v9.0" else _ratio_interpretation(ratio)
        pval_str = "  (base)"
        pval = None
        if cname != "Full v9.0" and len(full_gaps) >= 3 and len(gaps) >= 3:
            res_stat = wilcoxon_pairwise(full_gaps, gaps, cname)
            if res_stat:
                pval = res_stat["p_value"]
                pval_str = f"  {pval:.4f}"
                raw_pvals.append(pval)
                sig_rows.append(dict(ham=ham_name, config=cname, n_qubits=n_qubits,
                                     statistic=res_stat["statistic"],
                                     p_value=pval, ratio=round(ratio, 4),
                                     n=res_stat["n"], interpretation=interp))
        print(f"  {cname:<25} {np.mean(gaps):>10.5f} {np.std(gaps):>10.5f} "
              f"{np.mean(circs):>10.0f}  {ratio:>6.2f}x{pval_str}  {interp}")
        csv_rows.append(dict(config=cname, gap_mean=round(float(np.mean(gaps)), 6),
                             gap_std=round(float(np.std(gaps)), 6),
                             circuits_mean=round(float(np.mean(circs)), 1),
                             ratio=round(ratio, 2), n_seeds=len(rs),
                             n_qubits=n_qubits, interpretation=interp))

    # Holm correction on ablation p-values
    if raw_pvals:
        from ..analysis.stats import holm_bonferroni_correction
        corrected = holm_bonferroni_correction(raw_pvals)
        for row, cp in zip(sig_rows, corrected):
            row["p_value_holm"] = round(cp, 6)

    # Save CSVs
    abl_path = os.path.join(results_dir, f"ablation_results{sfx}.csv")
    with open(abl_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "gap_mean", "gap_std",
                                           "circuits_mean", "ratio",
                                           "n_seeds", "n_qubits", "interpretation"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"  [Ablation] Saved → {abl_path}")

    if sig_rows:
        sig_path = os.path.join(results_dir, f"ablation_significance{sfx}.csv")
        with open(sig_path, "w", newline="") as f:
            fields = ["ham", "config", "n_qubits", "statistic", "p_value",
                      "p_value_holm", "ratio", "n", "interpretation"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(sig_rows)
        print(f"  [Ablation] Significance → {sig_path}")

    return results


def run_precond_variance_ablation(
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
    logger: Optional[ParameterLogger] = None,
    device=None,
) -> Dict:
    """Raw sigma_hat^2 vs JS-shrunk sigma_tilde^2 preconditioner ablation."""
    n_seeds  = min(n_seeds, ABLATION_MAX_SEEDS)
    exact_E  = exact_gs(H)
    n_params = n_qubits * reps
    os.makedirs(results_dir, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    if device is None:
        device, _ = make_device(n_qubits)

    configs = {
        "Precond: raw (sigma_hat^2)":      dict(precond_variance="raw"),
        "Precond: shrunk (sigma_tilde^2)": dict(precond_variance="shrunk"),
    }
    base_flags = dict(use_precond=True, use_momentum=True,
                      use_clip=True, use_drift_correction=False)
    results = {k: [] for k in configs}

    for seed in range(n_seeds):
        theta0 = get_initial_theta(n_params, seed)
        for cname, extra in configs.items():
            cf  = CostEvaluator(n_qubits, reps, H, device=device, exact_E=exact_E)
            opt = SigmaQGD(precond_multiplier=multiplier, **base_flags, **extra)
            res, recs, _ = opt.run(cf, theta0.copy(),
                                   max_steps=max_steps, ham_type=ham_type,
                                   n_qubits=n_qubits, reps=reps, seed=seed,
                                   n_seeds=n_seeds, exact_E=exact_E,
                                   eta_0=eta_0, lambda_reg=lambda_reg, multiplier=multiplier,
                                   method_name=f"Ablation: {cname}")
            results[cname].append(res)

    base_key  = "Precond: raw (sigma_hat^2)"
    base_gaps = [abs(r.energy_opt - exact_E) for r in results[base_key]]
    base_gap  = float(np.mean(base_gaps))

    csv_rows = []; sig_rows = []
    for cname, rs in results.items():
        gaps  = [abs(r.energy_opt - exact_E) for r in rs]
        circs = [r.n_circuits for r in rs]
        ratio = float(np.mean(gaps)) / max(base_gap, 1e-12)
        pval  = None
        if cname != base_key and len(base_gaps) >= 3 and len(gaps) >= 3:
            res_stat = wilcoxon_pairwise(base_gaps, gaps, cname)
            if res_stat:
                pval = res_stat["p_value"]
                sig_rows.append(dict(ham=ham_name, config=cname, n_qubits=n_qubits,
                                     statistic=res_stat["statistic"],
                                     p_value=pval, ratio=round(ratio, 4), n=res_stat["n"]))
        csv_rows.append(dict(config=cname, gap_mean=round(float(np.mean(gaps)), 6),
                             gap_std=round(float(np.std(gaps)), 6),
                             circuits_mean=round(float(np.mean(circs)), 1),
                             ratio=round(ratio, 4), n_seeds=len(rs), n_qubits=n_qubits))

    path = os.path.join(results_dir, f"precond_variance_ablation{sfx}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "gap_mean", "gap_std",
                                           "circuits_mean", "ratio", "n_seeds", "n_qubits"])
        w.writeheader(); w.writerows(csv_rows)
    print(f"  [Precond Ablation] Saved → {path}")

    if sig_rows:
        sp = os.path.join(results_dir, f"precond_variance_significance{sfx}.csv")
        with open(sp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ham", "config", "n_qubits",
                                               "statistic", "p_value", "ratio", "n"])
            w.writeheader(); w.writerows(sig_rows)
        print(f"  [Precond Ablation] Significance → {sp}")

    return results


def _ratio_interpretation(ratio: float) -> str:
    if ratio > 1.1:
        return "removed component was helpful"
    elif ratio < 0.9:
        return "removed component was harmful"
    return "no measurable effect"
