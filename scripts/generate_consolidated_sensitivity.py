"""
Generate publication-quality consolidated sensitivity figure and table
combining TFIM 10q, TFIM 12q, XXZ 10q, and XXZ 12q into:
  1. A single 2x2 publication figure (PDF + PNG)
  2. A single clean LaTeX / Markdown table
Uses only Python standard library + numpy + matplotlib (no pandas dependency).
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_sweep_csv(filepath):
    """Load sensitivity CSV into structured dict keyed by (sweep, value)."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sweep = row["sweep"].strip()
            if sweep == "ema_window":
                continue  # drop dead sweep
            val = float(row["value"])
            mean = float(row["gap_mean"])
            std = float(row["gap_std"])
            data[(sweep, val)] = (mean, std)
    return data

def generate_consolidated_sensitivity(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    # Load 4 CSVs
    t10 = load_sweep_csv(os.path.join(data_dir, "sensitivity_sweep_sensitivity_tfim_10q.csv"))
    t12 = load_sweep_csv(os.path.join(data_dir, "sensitivity_sweep_sensitivity_tfim_12q.csv"))
    x10 = load_sweep_csv(os.path.join(data_dir, "sensitivity_sweep_sensitivity_xxz_10q.csv"))
    x12 = load_sweep_csv(os.path.join(data_dir, "sensitivity_sweep_sensitivity_xxz_12q.csv"))

    mult_vals = [0.5, 1.0, 2.0, 4.0, 6.0]
    lam_vals  = [0.0005, 0.001, 0.005, 0.01, 0.05]

    # 1. Plotting: Publication-Quality 2x2 Grid (Springer / QIP style)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex="col")

    # Colors and markers matching publication theme
    c_10q = "#008080"  # Teal / Dark Cyan
    c_12q = "#D95F02"  # Terracotta / Burnt Orange
    m_10q = "o"
    m_12q = "s"

    # Helper to extract points
    def get_points(d, sweep, vals):
        means = [d[(sweep, v)][0] for v in vals]
        stds  = [d[(sweep, v)][1] for v in vals]
        return means, stds

    # (a) TFIM Multiplier
    ax = axes[0, 0]
    t10_m, t10_s = get_points(t10, "multiplier", mult_vals)
    t12_m, t12_s = get_points(t12, "multiplier", mult_vals)
    ax.errorbar(mult_vals, t10_m, yerr=t10_s, fmt=f"-{m_10q}", color=c_10q, label="10 qubits", capsize=3.5)
    ax.errorbar(mult_vals, t12_m, yerr=t12_s, fmt=f"-{m_12q}", color=c_12q, label="12 qubits", capsize=3.5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7, label="Deployed ($c=1.0$)")
    ax.set_ylabel("Energy Gap $\\Delta E$ (TFIM)")
    ax.set_title("(a) TFIM: Preconditioner Multiplier $c$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper left")

    # (b) TFIM Lambda_reg
    ax = axes[0, 1]
    t10_lm, t10_ls = get_points(t10, "lambda_reg", lam_vals)
    t12_lm, t12_ls = get_points(t12, "lambda_reg", lam_vals)
    ax.errorbar(lam_vals, t10_lm, yerr=t10_ls, fmt=f"-{m_10q}", color=c_10q, label="10 qubits", capsize=3.5)
    ax.errorbar(lam_vals, t12_lm, yerr=t12_ls, fmt=f"-{m_12q}", color=c_12q, label="12 qubits", capsize=3.5)
    ax.axvline(0.005, color="gray", linestyle=":", alpha=0.7, label="Deployed ($\\lambda_{\\mathrm{reg}}=0.005$)")
    ax.set_xscale("log")
    ax.set_title("(b) TFIM: Regularizer $\\lambda_{\\mathrm{reg}}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper left")

    # (c) XXZ Multiplier
    ax = axes[1, 0]
    x10_m, x10_s = get_points(x10, "multiplier", mult_vals)
    x12_m, x12_s = get_points(x12, "multiplier", mult_vals)
    ax.errorbar(mult_vals, x10_m, yerr=x10_s, fmt=f"-{m_10q}", color=c_10q, label="10 qubits", capsize=3.5)
    ax.errorbar(mult_vals, x12_m, yerr=x12_s, fmt=f"-{m_12q}", color=c_12q, label="12 qubits", capsize=3.5)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7, label="Deployed ($c=1.0$)")
    ax.set_xlabel("Multiplier $c$ (in $F_i = c\\sigma_i^2 + \\lambda_{\\mathrm{reg}}$)")
    ax.set_ylabel("Energy Gap $\\Delta E$ (XXZ)")
    ax.set_title("(c) XXZ: Preconditioner Multiplier $c$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper left")

    # (d) XXZ Lambda_reg
    ax = axes[1, 1]
    x10_lm, x10_ls = get_points(x10, "lambda_reg", lam_vals)
    x12_lm, x12_ls = get_points(x12, "lambda_reg", lam_vals)
    ax.errorbar(lam_vals, x10_lm, yerr=x10_ls, fmt=f"-{m_10q}", color=c_10q, label="10 qubits", capsize=3.5)
    ax.errorbar(lam_vals, x12_lm, yerr=x12_ls, fmt=f"-{m_12q}", color=c_12q, label="12 qubits", capsize=3.5)
    ax.axvline(0.005, color="gray", linestyle=":", alpha=0.7, label="Deployed ($\\lambda_{\\mathrm{reg}}=0.005$)")
    ax.set_xscale("log")
    ax.set_xlabel("Regularization parameter $\\lambda_{\\mathrm{reg}}$")
    ax.set_title("(d) XXZ: Regularizer $\\lambda_{\\mathrm{reg}}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    png_path = os.path.join(out_dir, "fig_sensitivity_consolidated.png")
    pdf_path = os.path.join(out_dir, "fig_sensitivity_consolidated.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300)
    plt.close()
    print(f"Generated Figures:\n  {png_path}\n  {pdf_path}")

    # 2. Markdown & LaTeX Table Output
    csv_table_path = os.path.join(out_dir, "table_sensitivity_consolidated.csv")
    with open(csv_table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value", "TFIM 10q", "TFIM 12q", "XXZ 10q", "XXZ 12q"])
        
        for v in mult_vals:
            writer.writerow([
                "Multiplier c", f"{v:.1f}",
                f"{t10[('multiplier', v)][0]:.4f} +/- {t10[('multiplier', v)][1]:.4f}",
                f"{t12[('multiplier', v)][0]:.4f} +/- {t12[('multiplier', v)][1]:.4f}",
                f"{x10[('multiplier', v)][0]:.4f} +/- {x10[('multiplier', v)][1]:.4f}",
                f"{x12[('multiplier', v)][0]:.4f} +/- {x12[('multiplier', v)][1]:.4f}",
            ])
        for v in lam_vals:
            writer.writerow([
                "Regularizer lambda_reg", f"{v:g}",
                f"{t10[('lambda_reg', v)][0]:.4f} +/- {t10[('lambda_reg', v)][1]:.4f}",
                f"{t12[('lambda_reg', v)][0]:.4f} +/- {t12[('lambda_reg', v)][1]:.4f}",
                f"{x10[('lambda_reg', v)][0]:.4f} +/- {x10[('lambda_reg', v)][1]:.4f}",
                f"{x12[('lambda_reg', v)][0]:.4f} +/- {x12[('lambda_reg', v)][1]:.4f}",
            ])
    print(f"Generated Table CSV:\n  {csv_table_path}")

if __name__ == "__main__":
    data_dir = "sigma_data/sensitivity_sweep_results"
    generate_consolidated_sensitivity(data_dir, "figures")
    generate_consolidated_sensitivity(data_dir, "sigma_data/sensitivity_sweep_results")
