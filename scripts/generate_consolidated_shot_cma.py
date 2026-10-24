"""
Generate publication-quality consolidated Shot-Noise and CMA-ES figures and tables
combining TFIM 10q, TFIM 12q, XXZ 10q, and XXZ 12q (15 seeds each):
  1. Consolidated Multi-Panel Publication Figure (PDF + PNG at 300 DPI):
     - Panel A: Shot-Noise Error Scaling vs. Number of Shots M in {100, 500, 1000, 5000, 10000, inf}
     - Panel B: Budget-Matched Head-to-Head Comparison: SIGMA-QGD vs. CMA-ES at 1024 shots
  2. Consolidated Unified LaTeX, Markdown, and CSV tables

Outputs are written to:
  - figures/fig_shot_noise_cma_consolidated.pdf / .png
  - sigma_data/shot_noise_results/
  - sigma_data/sigma_cma_results/
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sigma_qgd.style import COLOURS, LW
except ImportError:
    COLOURS = {
        "SIGMA-QGD v9.0": "#00C9A7",
        "CMA-ES": "#D95F02",
        "TFIM": "#1F77B4",
        "XXZ": "#9467BD",
    }
    LW = {"SIGMA-QGD v9.0": 2.5, "CMA-ES": 2.0}


def load_shot_noise_data(base_dir="sigma_data/shot_noise_results"):
    """Load shot noise sweep data for TFIM and XXZ at 10q and 12q."""
    regimes = [
        ("TFIM", 10, os.path.join(base_dir, "shot_noise_results_shotnoise_tfim_10q.csv")),
        ("TFIM", 12, os.path.join(base_dir, "shot_noise_results_shotnoise_tfim_12q.csv")),
        ("XXZ",  10, os.path.join(base_dir, "shot_noise_results_shotnoise_xxz_10q.csv")),
        ("XXZ",  12, os.path.join(base_dir, "shot_noise_results_shotnoise_xxz_12q.csv")),
    ]
    data = {}
    for ham, nq, path in regimes:
        key = f"{ham}_{nq}q"
        rows = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append({
                        "n_shots": row["n_shots"],
                        "gap_mean": float(row["gap_mean"]),
                        "gap_std": float(row["gap_std"]),
                        "n_seeds": int(row["n_seeds"]),
                        "wilcoxon_p": row.get("wilcoxon_p", ""),
                        "direction": row.get("direction", ""),
                    })
        data[key] = {"ham": ham, "n_qubits": nq, "rows": rows}
    return data


def load_cma_data(base_dir="sigma_data/sigma_cma_results"):
    """Load budget-matched CMA-ES vs SIGMA-QGD seed-level data."""
    regimes = [
        ("TFIM", 10, 30, 12200, os.path.join(base_dir, "cma_vs_sigma_cma_both_10q_tfim.csv")),
        ("TFIM", 12, 36, 14600, os.path.join(base_dir, "cma_vs_sigma_cma_both_12q_tfim.csv")),
        ("XXZ",  10, 30, 12200, os.path.join(base_dir, "cma_vs_sigma_cma_both_10q_xxz.csv")),
        ("XXZ",  12, 36, 14600, os.path.join(base_dir, "cma_vs_sigma_cma_both_12q_xxz.csv")),
    ]
    data = {}
    for ham, nq, p, circ, path in regimes:
        key = f"{ham}_{nq}q"
        sigma_gaps, cma_gaps = [], []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    sigma_gaps.append(float(row["sigma_gap"]))
                    cma_gaps.append(float(row["cma_gap"]))
        
        # Calculate Wilcoxon test
        if len(sigma_gaps) > 0 and len(cma_gaps) > 0:
            stat = stats.wilcoxon(sigma_gaps, cma_gaps, alternative="two-sided")
            p_val = stat.pvalue
        else:
            p_val = float("nan")

        data[key] = {
            "ham": ham,
            "n_qubits": nq,
            "params": p,
            "circuits": circ,
            "sigma_gaps": np.array(sigma_gaps),
            "cma_gaps": np.array(cma_gaps),
            "wilcoxon_p": p_val,
        }
    return data


def plot_consolidated_figure(shot_data, cma_data, out_pdf, out_png):
    """
    Produce a 2-row x 2-column or 1-row x 2-panel figure.
    Panel A (Left): Finite-shot convergence scaling across shot budgets.
    Panel B (Right): Boxplots comparing SIGMA-QGD vs CMA-ES under shot noise.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Panel A: Shot Noise Error Scaling
    styles = {
        "TFIM_10q": {"color": "#1f77b4", "marker": "o", "label": "TFIM 10q ($p=30$)", "ls": "-"},
        "TFIM_12q": {"color": "#2ca02c", "marker": "s", "label": "TFIM 12q ($p=36$)", "ls": "--"},
        "XXZ_10q":  {"color": "#d62728", "marker": "^", "label": "XXZ 10q ($p=30$)",  "ls": "-."},
        "XXZ_12q":  {"color": "#9467bd", "marker": "D", "label": "XXZ 12q ($p=36$)",  "ls": ":"},
    }

    shot_order = [100, 500, 1000, 5000, 10000]

    for key, st in styles.items():
        if key not in shot_data:
            continue
        rows = shot_data[key]["rows"]
        # extract points for finite shots
        shot_map = {int(r["n_shots"]): (r["gap_mean"], r["gap_std"]) for r in rows if r["n_shots"] != "inf"}
        inf_row = next((r for r in rows if r["n_shots"] == "inf"), None)

        x_vals = [s for s in shot_order if s in shot_map]
        y_means = [shot_map[s][0] for s in x_vals]
        y_stds = [shot_map[s][1] for s in x_vals]

        # Plot error bar curve
        ax1.errorbar(
            x_vals, y_means, yerr=y_stds,
            color=st["color"], marker=st["marker"], linestyle=st["ls"],
            linewidth=2.0, capsize=4, capthick=1.2, markersize=6,
            label=st["label"], alpha=0.9
        )

        # Plot noiseless reference as dashed horizontal guide
        if inf_row:
            inf_mean = inf_row["gap_mean"]
            ax1.axhline(inf_mean, color=st["color"], linestyle=":", alpha=0.45, linewidth=1.2)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Shot Budget per Circuit $M$ (log scale)")
    ax1.set_ylabel(r"Exact Ground-State Energy Gap $|E(\boldsymbol{\theta}^*) - E_0|$")
    ax1.set_title("(a) Finite-Shot Noise Robustness ($N=15$ seeds)", fontweight="bold")
    ax1.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.95)

    # Panel B: Budget-Matched CMA-ES vs. SIGMA-QGD (1024 shots)
    regime_order = ["TFIM_10q", "TFIM_12q", "XXZ_10q", "XXZ_12q"]
    labels = ["TFIM 10q\n(12,200 circ)", "TFIM 12q\n(14,600 circ)", "XXZ 10q\n(12,200 circ)", "XXZ 12q\n(14,600 circ)"]
    
    x = np.arange(len(regime_order))
    width = 0.32

    sigma_color = "#00C9A7"  # Brand teal
    cma_color = "#E24B4A"    # Contrasting coral/red

    # Plot paired boxplots
    for idx, key in enumerate(regime_order):
        if key not in cma_data:
            continue
        sig_pts = cma_data[key]["sigma_gaps"]
        cma_pts = cma_data[key]["cma_gaps"]

        # Boxplot for SIGMA-QGD
        bp_sig = ax2.boxplot(
            [sig_pts], positions=[x[idx] - width/2], widths=width * 0.85,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=sigma_color, color="#007A65", alpha=0.75, lw=1.2),
            medianprops=dict(color="#004D40", lw=2.0),
            whiskerprops=dict(color="#007A65", lw=1.2),
            capprops=dict(color="#007A65", lw=1.2),
            flierprops=dict(marker="o", markerfacecolor=sigma_color, markersize=4, alpha=0.6)
        )

        # Boxplot for CMA-ES
        bp_cma = ax2.boxplot(
            [cma_pts], positions=[x[idx] + width/2], widths=width * 0.85,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=cma_color, color="#A82322", alpha=0.75, lw=1.2),
            medianprops=dict(color="#680000", lw=2.0),
            whiskerprops=dict(color="#A82322", lw=1.2),
            capprops=dict(color="#A82322", lw=1.2),
            flierprops=dict(marker="o", markerfacecolor=cma_color, markersize=4, alpha=0.6)
        )

        # Annotate statistical significance above pairs
        p_val = cma_data[key]["wilcoxon_p"]
        max_y = max(np.max(sig_pts), np.max(cma_pts))
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
        ax2.text(x[idx], max_y + 0.15, f"{star}\n($p={p_val:.1e}$)",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333333")

    # Dummy artists for clean legend
    patch_sig = plt.Rectangle((0, 0), 1, 1, facecolor=sigma_color, edgecolor="#007A65", alpha=0.75, label="SIGMA-QGD v9.0")
    patch_cma = plt.Rectangle((0, 0), 1, 1, facecolor=cma_color, edgecolor="#A82322", alpha=0.75, label="CMA-ES (Budget-Matched)")
    ax2.legend(handles=[patch_sig, patch_cma], loc="upper left", framealpha=0.95)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel(r"Exact Ground-State Energy Gap $|E(\boldsymbol{\theta}^*) - E_0|$")
    ax2.set_title("(b) CMA-ES vs. SIGMA-QGD ($M=1024$ shots, $N=15$)", fontweight="bold")
    ax2.set_ylim(-0.1, 4.45)
    ax2.grid(True, axis="y", ls="--", lw=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [Saved Figure] -> {out_pdf}")
    print(f"  [Saved Figure] -> {out_png}")


def generate_consolidated_table(shot_data, cma_data, out_tex, out_csv, out_md):
    """
    Generate unified LaTeX, Markdown, and CSV tables combining
    both Shot-Noise Robustness and CMA-ES Benchmark.
    """
    # 1. Prepare Data for CMA-ES Comparison Table
    cma_rows = []
    regime_order = [
        ("TFIM", 10, "TFIM ($n=10$, $p=30$)", 12200),
        ("TFIM", 12, "TFIM ($n=12$, $p=36$)", 14600),
        ("XXZ",  10, "XXZ ($n=10$, $p=30$)",  12200),
        ("XXZ",  12, "XXZ ($n=12$, $p=36$)",  14600),
    ]

    for ham, nq, label, circ in regime_order:
        key = f"{ham}_{nq}q"
        d = cma_data[key]
        sig_m, sig_s = np.mean(d["sigma_gaps"]), np.std(d["sigma_gaps"])
        cma_m, cma_s = np.mean(d["cma_gaps"]), np.std(d["cma_gaps"])
        p_val = d["wilcoxon_p"]
        cma_rows.append({
            "regime": label,
            "circuits": circ,
            "sigma_mean_std": f"{sig_m:.4f} ± {sig_s:.4f}",
            "cma_mean_std": f"{cma_m:.4f} ± {cma_s:.4f}",
            "p_val": f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}",
            "advantage": f"{cma_m / sig_m:.2f}×" if sig_m > 0 else "---",
        })

    # 2. Prepare Data for Shot-Noise Decay Table
    shot_rows = []
    shot_budgets = ["inf", "100", "500", "1000", "5000", "10000"]
    for sb in shot_budgets:
        sb_name = "Noiseless" if sb == "inf" else f"{int(sb):,} shots"
        row_dict = {"shot_budget": sb_name}
        for ham in ["TFIM", "XXZ"]:
            for nq in [10, 12]:
                k = f"{ham}_{nq}q"
                r = next((x for x in shot_data[k]["rows"] if x["n_shots"] == sb), None)
                if r:
                    row_dict[f"{ham}_{nq}q"] = f"{r['gap_mean']:.4f} ± {r['gap_std']:.4f}"
                else:
                    row_dict[f"{ham}_{nq}q"] = "---"
        shot_rows.append(row_dict)

    # Write LaTeX Table
    tex_content = r"""% Consolidated Finite-Shot Noise and CMA-ES Comparison Table
\begin{table*}[t]
\centering
\small
\caption{\textbf{Comprehensive finite-shot robustness and derivative-free metaheuristic comparison ($N=15$ independent seeds).} 
\textbf{Top:} Exact ground-state energy gap $|E(\theta^*) - E_0|$ across finite shot budgets $M \in \{100, 500, 1000, 5000, 10000, \infty\}$. All parameter vectors are re-evaluated on a noiseless simulator to prevent downward noise-selection bias. 
\textbf{Bottom:} Strict circuit-budget-matched comparison against CMA-ES under finite shot noise ($M=1024$ shots). Both algorithms run for identical cumulative quantum circuit budgets ($200 \times (2p+1)$ evaluations).}
\label{tab:shot_noise_cma_consolidated}
\begin{tabular}{lcccc}
\toprule
\multicolumn{5}{c}{\textbf{Part I: SIGMA-QGD Finite-Shot Scaling ($|E(\theta^*) - E_0|$ in Hartree)}} \\
\midrule
\textbf{Shot Budget ($M$)} & \textbf{TFIM ($n=10$, $p=30$)} & \textbf{TFIM ($n=12$, $p=36$)} & \textbf{XXZ ($n=10$, $p=30$)} & \textbf{XXZ ($n=12$, $p=36$)} \\
\midrule
"""
    for sr in shot_rows:
        tex_content += f"{sr['shot_budget']} & {sr['TFIM_10q']} & {sr['TFIM_12q']} & {sr['XXZ_10q']} & {sr['XXZ_12q']} \\\\\n"

    tex_content += r"""\midrule
\multicolumn{5}{c}{\textbf{Part II: Budget-Matched Head-to-Head Comparison: SIGMA-QGD vs. CMA-ES ($M=1024$ Shots)}} \\
\midrule
\textbf{Regime} & \textbf{Circuit Budget} & \textbf{SIGMA-QGD Gap} & \textbf{CMA-ES Gap} & \textbf{Wilcoxon $p$ (Advantage)} \\
\midrule
"""
    for cr in cma_rows:
        tex_content += f"{cr['regime']} & {cr['circuits']:,} circ & {cr['sigma_mean_std']} & {cr['cma_mean_std']} & $p={cr['p_val']}$ ({cr['advantage']}) \\\\\n"

    tex_content += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"  [Saved Table]  -> {out_tex}")

    # Write CSV Table
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric", "TFIM_10q", "TFIM_12q", "XXZ_10q", "XXZ_12q", "Details"])
        for sr in shot_rows:
            writer.writerow(["Shot_Noise", sr["shot_budget"], sr["TFIM_10q"], sr["TFIM_12q"], sr["XXZ_10q"], sr["XXZ_12q"], "SIGMA-QGD"])
        for cr in cma_rows:
            writer.writerow(["CMA_Benchmark", cr["regime"], cr["sigma_mean_std"], cr["cma_mean_std"], cr["circuits"], cr["p_val"], f"Advantage: {cr['advantage']}"])
    print(f"  [Saved Table]  -> {out_csv}")

    # Write Markdown Table
    md_content = "# Consolidated Finite-Shot Noise & CMA-ES Comparison\n\n"
    md_content += "### Part I: Finite-Shot Scaling (SIGMA-QGD)\n\n"
    md_content += "| Shot Budget ($M$) | TFIM (10q) | TFIM (12q) | XXZ (10q) | XXZ (12q) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: |\n"
    for sr in shot_rows:
        md_content += f"| {sr['shot_budget']} | {sr['TFIM_10q']} | {sr['TFIM_12q']} | {sr['XXZ_10q']} | {sr['XXZ_12q']} |\n"

    md_content += "\n### Part II: Budget-Matched CMA-ES vs. SIGMA-QGD ($M=1024$ Shots)\n\n"
    md_content += "| Regime | Circuit Budget | SIGMA-QGD Gap | CMA-ES Gap | Wilcoxon $p$ | Ratio (CMA/SIGMA) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for cr in cma_rows:
        md_content += f"| {cr['regime']} | {cr['circuits']:,} | {cr['sigma_mean_std']} | {cr['cma_mean_std']} | {cr['p_val']} | {cr['advantage']} |\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"  [Saved Table]  -> {out_md}")


def main():
    base_shot_dir = "sigma_data/shot_noise_results"
    base_cma_dir  = "sigma_data/sigma_cma_results"
    figures_dir   = "figures"
    os.makedirs(figures_dir, exist_ok=True)

    print("="*70)
    print("Generating Consolidated Shot-Noise & CMA-ES Figures and Tables...")
    print("="*70)

    shot_data = load_shot_noise_data(base_shot_dir)
    cma_data  = load_cma_data(base_cma_dir)

    out_pdf = os.path.join(figures_dir, "fig_shot_noise_cma_consolidated.pdf")
    out_png = os.path.join(figures_dir, "fig_shot_noise_cma_consolidated.png")
    plot_consolidated_figure(shot_data, cma_data, out_pdf, out_png)

    # Also save a copy inside sigma_data/shot_noise_results for local inspection
    plot_consolidated_figure(shot_data, cma_data,
                            os.path.join(base_shot_dir, "fig_shot_noise_cma_consolidated.pdf"),
                            os.path.join(base_shot_dir, "fig_shot_noise_cma_consolidated.png"))

    out_tex = os.path.join(base_shot_dir, "table_shot_noise_cma_consolidated.tex")
    out_csv = os.path.join(base_shot_dir, "table_shot_noise_cma_consolidated.csv")
    out_md  = os.path.join(base_shot_dir, "table_shot_noise_cma_consolidated.md")
    generate_consolidated_table(shot_data, cma_data, out_tex, out_csv, out_md)

    print("="*70)
    print("Successfully generated all consolidated artifacts!")
    print("="*70)


if __name__ == "__main__":
    main()
