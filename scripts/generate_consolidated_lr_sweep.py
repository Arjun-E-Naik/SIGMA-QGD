"""
Generate publication-quality consolidated Learning Rate (LR) sweep figure and table
combining TFIM 10q (10 seeds), TFIM 12q (5 seeds), XXZ 10q (10 seeds), and XXZ 12q (5 seeds) into:
  1. A single 2x2 publication figure (PDF + PNG at 300 DPI)
  2. A single clean LaTeX / Markdown / CSV table

Uses only Python standard library + numpy + matplotlib (no pandas dependency).
"""

import os
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Try importing style definitions from sigma_qgd if available
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sigma_qgd.style import COLOURS, LINES, MARKERS, LW
except Exception:
    COLOURS = {
        "SIGMA-QGD v9.0": "#00C9A7",
        "Adam":           "#E24B4A",
        "AdaBelief":      "#1F77B4",
        "RMSProp":        "#BCBD22",
        "Yogi":           "#17BECF",
        "H-QNG (Shi&Wang)":"#D62728",
        "Diag-Hessian":   "#8C564B",
    }
    LINES = {
        "SIGMA-QGD v9.0": "-",
        "Adam": "-.",
        "AdaBelief": "-.",
        "RMSProp": ":",
        "Yogi": "--",
        "H-QNG (Shi&Wang)": "-.",
        "Diag-Hessian": "--",
    }
    MARKERS = {
        "SIGMA-QGD v9.0": "D",
        "Adam": "X",
        "AdaBelief": "P",
        "RMSProp": "s",
        "Yogi": "*",
        "H-QNG (Shi&Wang)": "<",
        "Diag-Hessian": "h",
    }
    LW = {
        "SIGMA-QGD v9.0": 2.5,
        "Adam": 1.8,
        "AdaBelief": 1.8,
        "RMSProp": 1.5,
        "Yogi": 1.5,
        "H-QNG (Shi&Wang)": 1.8,
        "Diag-Hessian": 1.8,
    }


def find_data_file(data_dir, filename_pattern):
    """Search for data file in data_dir or fallback directories."""
    candidates = [
        os.path.join(data_dir, filename_pattern),
        os.path.join("sigma_data", "lr_sweep_results", filename_pattern),
        os.path.join("sigma_results", "lr_sweep_results", filename_pattern),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find {filename_pattern} in any of: {candidates}")


def load_detail_csv(filepath):
    """
    Load per-seed LR sweep detail CSV.
    Returns:
      data[method][lr] = list of gap values across seeds
      n_seeds: number of distinct seeds found
      n_qubits: system size
      ham_name: Hamiltonian description string
    """
    data = defaultdict(lambda: defaultdict(list))
    seeds = set()
    n_qubits = None
    ham_name = ""

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["method"].strip()
            # Deduplicate: treat 'H-QNG' as canonical 'H-QNG (Shi&Wang)'
            if m == "H-QNG":
                continue
            lr = float(row["lr"])
            seed = int(row["seed"])
            gap = float(row["gap"])
            seeds.add(seed)
            if n_qubits is None and "n_qubits" in row:
                n_qubits = int(row["n_qubits"])
            if not ham_name and "ham" in row:
                ham_name = row["ham"]
            data[m][lr].append(gap)

    return data, len(seeds), n_qubits, ham_name


def generate_consolidated_lr_sweep(data_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load the 4 datasets
    f_t10 = find_data_file(data_dir, "lr_sweep_detail_lr_sweep_tfim_10q.csv")
    f_t12 = find_data_file(data_dir, "lr_sweep_detail_lr_sweep_tfim_12q.csv")
    f_x10 = find_data_file(data_dir, "lr_sweep_detail_lr_sweep_xxz_10q.csv")
    f_x12 = find_data_file(data_dir, "lr_sweep_detail_lr_sweep_xxz_12q.csv")

    t10_data, t10_seeds, _, _ = load_detail_csv(f_t10)
    t12_data, t12_seeds, _, _ = load_detail_csv(f_t12)
    x10_data, x10_seeds, _, _ = load_detail_csv(f_x10)
    x12_data, x12_seeds, _, _ = load_detail_csv(f_x12)

    datasets = [
        ("TFIM 10-Qubit", t10_data, t10_seeds, "(a) TFIM 10-Qubit ($N=10$ seeds)", True),
        ("TFIM 12-Qubit", t12_data, t12_seeds, "(b) TFIM 12-Qubit ($N=5$ seeds)", True),
        ("XXZ 10-Qubit",  x10_data, x10_seeds, "(c) XXZ 10-Qubit ($N=10$ seeds)", False),
        ("XXZ 12-Qubit",  x12_data, x12_seeds, "(d) XXZ 12-Qubit ($N=5$ seeds)", False),
    ]

    method_order = [
        "SIGMA-QGD v9.0",
        "Adam",
        "AdaBelief",
        "Yogi",
        "RMSProp",
        "Diag-Hessian",
        "H-QNG (Shi&Wang)",
    ]

    lr_grid = [0.001, 0.005, 0.01, 0.05, 0.1]

    # Publication-Quality 2x2 Plot (Springer Nature / QIP style)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)

    for idx, (label, d_data, n_seeds, panel_title, is_log_y) in enumerate(datasets):
        ax = axes[idx // 2, idx % 2]
        
        for m in method_order:
            if m not in d_data:
                continue
            
            means = []
            stds  = []
            for lr in lr_grid:
                gaps = d_data[m].get(lr, [])
                means.append(float(np.mean(gaps)) if gaps else np.nan)
                stds.append(float(np.std(gaps)) if gaps else np.nan)

            means = np.array(means)
            stds  = np.array(stds)

            color  = COLOURS.get(m, "#555555")
            ls     = LINES.get(m, "-")
            marker = MARKERS.get(m, "o")
            lw     = LW.get(m, 1.8)

            # Plot curve with errorbars
            line = ax.errorbar(
                lr_grid, means, yerr=stds,
                fmt=f"{ls}{marker}", color=color, lw=lw,
                markersize=6 if m != "SIGMA-QGD v9.0" else 7.5,
                capsize=3.5, capthick=1.0, elinewidth=1.0,
                label="Hess-Mag" if "H-QNG" in m else m, alpha=0.92, zorder=5 if m == "SIGMA-QGD v9.0" else 3
            )

            # Highlight minimum with a star
            valid_idx = np.where(~np.isnan(means))[0]
            if len(valid_idx) > 0:
                best_i = valid_idx[np.argmin(means[valid_idx])]
                ax.plot(
                    lr_grid[best_i], means[best_i],
                    marker="*", markersize=10 if m != "SIGMA-QGD v9.0" else 12,
                    color=color, markeredgecolor="black", markeredgewidth=0.7,
                    zorder=6
                )

        ax.set_xscale("log")
        ax.set_xticks(lr_grid)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(panel_title, fontweight="bold", pad=7)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        if is_log_y:
            ax.set_yscale("log")
            ax.set_ylabel("Energy Gap $\\Delta E$ (log scale)")
        else:
            ax.set_ylabel("Energy Gap $\\Delta E$")

        if idx // 2 == 1:
            ax.set_xlabel("Learning Rate $\\eta$")

    # Shared legend placed cleanly between suptitle and top subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=4,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=9
    )

    plt.suptitle("Consolidated Learning Rate Sweep Across System Sizes and Models\n"
                 "(Stars denote independently identified optimal $\\eta^*$ for each method)",
                 fontsize=12, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.89])

    png_path = os.path.join(out_dir, "fig_lr_sweep_consolidated.png")
    pdf_path = os.path.join(out_dir, "fig_lr_sweep_consolidated.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, dpi=300)
    plt.close()
    print(f"Generated Figures:\n  {png_path}\n  {pdf_path}")

    # 2. Extract best LR summary per method and dataset
    table_data = defaultdict(dict)
    min_mean_per_dataset = {}

    for label, d_data, n_seeds, _, _ in datasets:
        best_overall_gap = float("inf")
        for m in method_order:
            best_lr = None
            best_mean = float("inf")
            best_std = 0.0
            for lr in lr_grid:
                gaps = d_data[m].get(lr, [])
                if gaps:
                    mean = float(np.mean(gaps))
                    std  = float(np.std(gaps))
                    if mean < best_mean:
                        best_mean = mean
                        best_std  = std
                        best_lr   = lr
            table_data[m][label] = (best_lr, best_mean, best_std)
            if best_mean < best_overall_gap:
                best_overall_gap = best_mean
        min_mean_per_dataset[label] = best_overall_gap

    # 3. Save CSV Table
    csv_path = os.path.join(out_dir, "table_lr_sweep_consolidated.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "Method",
            "TFIM 10q (10s) Best LR", "TFIM 10q Gap Mean", "TFIM 10q Gap Std",
            "TFIM 12q (5s) Best LR",  "TFIM 12q Gap Mean", "TFIM 12q Gap Std",
            "XXZ 10q (10s) Best LR",  "XXZ 10q Gap Mean",  "XXZ 10q Gap Std",
            "XXZ 12q (5s) Best LR",   "XXZ 12q Gap Mean",   "XXZ 12q Gap Std",
        ])
        for m in method_order:
            t10 = table_data[m]["TFIM 10-Qubit"]
            t12 = table_data[m]["TFIM 12-Qubit"]
            x10 = table_data[m]["XXZ 10-Qubit"]
            x12 = table_data[m]["XXZ 12-Qubit"]
            w.writerow([
                m,
                t10[0], f"{t10[1]:.6f}", f"{t10[2]:.6f}",
                t12[0], f"{t12[1]:.6f}", f"{t12[2]:.6f}",
                x10[0], f"{x10[1]:.6f}", f"{x10[2]:.6f}",
                x12[0], f"{x12[1]:.6f}", f"{x12[2]:.6f}",
            ])
    print(f"Generated Table CSV:\n  {csv_path}")

    # 4. Save Markdown Table
    md_path = os.path.join(out_dir, "table_lr_sweep_consolidated.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Consolidated Learning Rate Sweep Summary\n\n")
        f.write("| Method | TFIM 10q (10 seeds) $\\eta^*$ | TFIM 10q $\\Delta E$ | "
                "TFIM 12q (5 seeds) $\\eta^*$ | TFIM 12q $\\Delta E$ | "
                "XXZ 10q (10 seeds) $\\eta^*$ | XXZ 10q $\\Delta E$ | "
                "XXZ 12q (5 seeds) $\\eta^*$ | XXZ 12q $\\Delta E$ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m in method_order:
            t10 = table_data[m]["TFIM 10-Qubit"]
            t12 = table_data[m]["TFIM 12-Qubit"]
            x10 = table_data[m]["XXZ 10-Qubit"]
            x12 = table_data[m]["XXZ 12-Qubit"]
            
            # Highlight SIGMA-QGD in bold
            is_sigma = (m == "SIGMA-QGD v9.0")
            fmt = "**{}**" if is_sigma else "{}"
            f.write(f"| {fmt.format(m)} "
                    f"| {t10[0]} | {fmt.format(f'{t10[1]:.4f} +/- {t10[2]:.4f}')} "
                    f"| {t12[0]} | {fmt.format(f'{t12[1]:.4f} +/- {t12[2]:.4f}')} "
                    f"| {x10[0]} | {fmt.format(f'{x10[1]:.4f} +/- {x10[2]:.4f}')} "
                    f"| {x12[0]} | {fmt.format(f'{x12[1]:.4f} +/- {x12[2]:.4f}')} |\n")
    print(f"Generated Markdown Table:\n  {md_path}")

    # 5. Save LaTeX Table
    tex_path = os.path.join(out_dir, "table_lr_sweep_consolidated.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Consolidated Learning Rate Sweep Table for Springer LNCS / QIP\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\caption{Predeclared learning rate sweep results across 7 gradient-based optimizers ($200$ steps). "
                "10-qubit experiments evaluated over $N=10$ independent random seeds; 12-qubit experiments evaluated over $N=5$ seeds. "
                "Reported values indicate the optimal step size $\\eta^*$ and resulting mean ground-state energy gap $\\Delta E \\pm \\sigma$.}\n")
        f.write("\\label{tab:lr_sweep_summary}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("\\multirow{2}{*}{\\textbf{Method}} & "
                "\\multicolumn{2}{c}{\\textbf{TFIM 10q ($N=10$)}} & "
                "\\multicolumn{2}{c}{\\textbf{TFIM 12q ($N=5$)}} & "
                "\\multicolumn{2}{c}{\\textbf{XXZ 10q ($N=10$)}} & "
                "\\multicolumn{2}{c}{\\textbf{XXZ 12q ($N=5$)}} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n")
        f.write("& $\\eta^*$ & $\\Delta E$ (Ha) & $\\eta^*$ & $\\Delta E$ (Ha) & $\\eta^*$ & $\\Delta E$ (Ha) & $\\eta^*$ & $\\Delta E$ (Ha) \\\\\n")
        f.write("\\midrule\n")
        for m in method_order:
            t10 = table_data[m]["TFIM 10-Qubit"]
            t12 = table_data[m]["TFIM 12-Qubit"]
            x10 = table_data[m]["XXZ 10-Qubit"]
            x12 = table_data[m]["XXZ 12-Qubit"]

            def fmt_tex(val_tuple, ds_name):
                lr, mean, std = val_tuple
                s = f"{mean:.4f} $\\pm$ {std:.4f}"
                if abs(mean - min_mean_per_dataset[ds_name]) < 1e-6:
                    return f"\\textbf{{{lr}}} & \\textbf{{{s}}}"
                return f"{lr} & {s}"

            name_tex = m.replace("&", "\\&")
            if m == "SIGMA-QGD v9.0":
                name_tex = f"\\textbf{{{name_tex}}}"

            s_t10 = fmt_tex(t10, "TFIM 10-Qubit")
            s_t12 = fmt_tex(t12, "TFIM 12-Qubit")
            s_x10 = fmt_tex(x10, "XXZ 10-Qubit")
            s_x12 = fmt_tex(x12, "XXZ 12-Qubit")

            f.write(f"{name_tex} & {s_t10} & {s_t12} & {s_x10} & {s_x12} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    print(f"Generated LaTeX Table:\n  {tex_path}")


if __name__ == "__main__":
    d_dir = sys.argv[1] if len(sys.argv) > 1 else "sigma_data/lr_sweep_results"
    o_dir = sys.argv[2] if len(sys.argv) > 2 else d_dir
    generate_consolidated_lr_sweep(d_dir, o_dir)
