"""
Generate publication-quality consolidated Ablation Study figure and table
combining TFIM 10q (15 seeds), TFIM 12q (15 seeds), XXZ 10q (15 seeds), and XXZ 12q (15 seeds) into:
  1. A single 2x2 publication figure (PDF + PNG at 300 DPI)
  2. A single clean LaTeX / Markdown / CSV table

Uses only Python standard library + numpy + matplotlib + scipy.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_ablation_data(data_dir):
    """
    Load all 4 ablation results and significance files.
    Returns structured dictionary with metrics per regime and config.
    """
    regimes = [
        ("tfim_10q", "TFIM 10-Qubit ($p=30$)", "ablation_results_ablation_tfim_10q.csv", "ablation_significance_ablation_tfim_10q.csv"),
        ("tfim_12q", "TFIM 12-Qubit ($p=36$)", "ablation_results_ablation_tfim_12q.csv", "ablation_significance_ablation_tfim_12q.csv"),
        ("xxz_10q",  "XXZ 10-Qubit ($p=30$)",  "ablation_results_ablation_xxz_10q.csv",  "ablation_significance_ablation_xxz_10q.csv"),
        ("xxz_12q",  "XXZ 12-Qubit ($p=36$)",  "ablation_results_ablation_xxz_12q.csv",  "ablation_significance_ablation_xxz_12q.csv"),
    ]
    
    data = {}
    for key, title, res_file, sig_file in regimes:
        res_path = os.path.join(data_dir, res_file)
        sig_path = os.path.join(data_dir, sig_file)
        
        # Load significance mapping
        sig_map = {}
        if os.path.exists(sig_path):
            with open(sig_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sig_map[r["config"].strip()] = r
                    
        configs_data = []
        with open(res_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cfg = r["config"].strip()
                sig = sig_map.get(cfg, {})
                configs_data.append({
                    "config": cfg,
                    "gap_mean": float(r["gap_mean"]),
                    "gap_std": float(r["gap_std"]),
                    "ratio": float(r["ratio"]),
                    "circuits_mean": float(r.get("circuits_mean", 0)),
                    "n_seeds": int(r.get("n_seeds", 15)),
                    "p_value": sig.get("p_value", "---"),
                    "p_value_holm": sig.get("p_value_holm", "---"),
                    "interpretation": r.get("interpretation", "")
                })
        data[key] = {
            "title": title,
            "configs": configs_data
        }
    return data

def generate_consolidated_ablation(data_dir: str, out_dirs: list):
    data = load_ablation_data(data_dir)
    
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    # Publication-quality plot style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    
    # Palette
    c_base = "#00C9A7"      # Teal for full model
    c_critical = "#D95F02"  # Terracotta / Coral for critical components
    c_neutral = "#7570B3"   # Slate purple for invariant components
    
    config_labels_clean = {
        "Full v9.0": "Full SIGMA-QGD (Default)",
        "No preconditioner": "No Preconditioner ($F=1$)",
        "No momentum": "No Momentum ($\\beta=0$)",
        "No clipping": "No Clipping ($c_{\\rm clip}=\\infty$)",
        "With drift correct.": "With Drift Correction",
    }
    
    panels = [
        ("tfim_10q", axes[0, 0], "(a) TFIM: 10 Qubits ($p=30$)"),
        ("tfim_12q", axes[0, 1], "(b) TFIM: 12 Qubits ($p=36$)"),
        ("xxz_10q",  axes[1, 0], "(c) XXZ: 10 Qubits ($p=30$)"),
        ("xxz_12q",  axes[1, 1], "(d) XXZ: 12 Qubits ($p=36$)"),
    ]
    
    for key, ax, panel_title in panels:
        c_list = data[key]["configs"]
        y_pos = np.arange(len(c_list))
        
        labels = [config_labels_clean.get(c["config"], c["config"]) for c in c_list]
        means = [c["gap_mean"] for c in c_list]
        stds = [c["gap_std"] for c in c_list]
        ratios = [c["ratio"] for c in c_list]
        
        colors = []
        for c in c_list:
            cfg_name = c["config"]
            if "Full" in cfg_name:
                colors.append(c_base)
            elif "No precond" in cfg_name or "No momentum" in cfg_name:
                colors.append(c_critical)
            else:
                colors.append(c_neutral)
                
        bars = ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.85,
                       edgecolor="#333333", linewidth=0.8, capsize=4, height=0.6)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Top-down order
        ax.set_title(panel_title, fontweight="bold", pad=8)
        ax.set_xlabel("Ground-State Energy Gap $\\Delta E$")
        ax.grid(True, linestyle="--", alpha=0.35, axis="x")
        
        # Add ratio annotations
        max_x = max([m + s for m, s in zip(means, stds)])
        x_limit = max_x * 1.28
        ax.set_xlim(0, x_limit)
        
        for i, (m, s, r, c) in enumerate(zip(means, stds, ratios, c_list)):
            if "Full" in c["config"]:
                txt = "1.00$\\times$ (Base)"
            else:
                txt = f"{r:.2f}$\\times$"
                # append asterisks if significant
                p_h = c["p_value_holm"]
                try:
                    p_val = float(p_h)
                    if p_val < 0.001:
                        txt += " $^{***}$"
                    elif p_val < 0.01:
                        txt += " $^{**}$"
                    elif p_val < 0.05:
                        txt += " $^{*}$"
                except:
                    pass
            ax.text(m + s + (0.01 * x_limit), i, txt, va="center", ha="left",
                    fontsize=8.5, fontweight="bold", color="#222222")

    # Legend at the top/bottom
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c_base, edgecolor="#333333", label="Full Default Pipeline"),
        Patch(facecolor=c_critical, edgecolor="#333333", label="Ablated Critical Component ($p < 0.05$)"),
        Patch(facecolor=c_neutral, edgecolor="#333333", label="Invariant Safeguard / Extension ($p > 0.05$)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    
    for out_dir in out_dirs:
        png_path = os.path.join(out_dir, "fig_ablation_consolidated.png")
        pdf_path = os.path.join(out_dir, "fig_ablation_consolidated.pdf")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        print(f"  Saved figure: {png_path} & {pdf_path}")
    plt.close()

    # 2. LaTeX Table Generation (Springer / LNCS multi-column)
    tex_content = []
    tex_content.append("% Consolidated Component Ablation Table")
    tex_content.append("\\begin{table*}[htbp]")
    tex_content.append("\\centering")
    tex_content.append("\\small")
    tex_content.append("\\caption{Consolidated component-level ablation study across TFIM and XXZ models at 10 and 12 qubits ($N=15$ seeds, $200$ steps). Reported metrics include mean ground-state energy gap $\\Delta E \\pm \\sigma$, degradation ratio relative to Full SIGMA-QGD, and Holm-Bonferroni corrected Wilcoxon signed-rank $p$-value.}")
    tex_content.append("\\label{tab:ablation_consolidated}")
    tex_content.append("\\begin{tabular}{lcccccccc}")
    tex_content.append("\\toprule")
    tex_content.append("\\multirow{2}{*}{\\textbf{Configuration}} & \\multicolumn{2}{c}{\\textbf{TFIM 10q ($p=30$)}} & \\multicolumn{2}{c}{\\textbf{TFIM 12q ($p=36$)}} & \\multicolumn{2}{c}{\\textbf{XXZ 10q ($p=30$)}} & \\multicolumn{2}{c}{\\textbf{XXZ 12q ($p=36$)}} \\\\")
    tex_content.append("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}")
    tex_content.append("& $\\Delta E$ (Ha) & Ratio ($p_{\\text{Holm}}$) & $\\Delta E$ (Ha) & Ratio ($p_{\\text{Holm}}$) & $\\Delta E$ (Ha) & Ratio ($p_{\\text{Holm}}$) & $\\Delta E$ (Ha) & Ratio ($p_{\\text{Holm}}$) \\\\")
    tex_content.append("\\midrule")
    
    # Rows: Full, No precond, No momentum, No clipping, With drift correct
    ordered_cfgs = [
        ("Full v9.0", "\\textbf{Full SIGMA-QGD v9.0}"),
        ("No preconditioner", "No Preconditioner ($\\mathbf{F}=\\mathbf{I}$)"),
        ("No momentum", "No Momentum ($\\beta=0$)"),
        ("No clipping", "No Clipping ($c_{\\text{clip}}=\\infty$)"),
        ("With drift correct.", "With Drift Correction"),
    ]
    
    def format_cell(c):
        mean = c["gap_mean"]
        std = c["gap_std"]
        ratio = c["ratio"]
        p_h = c["p_value_holm"]
        
        val_str = f"{mean:.4f} $\\pm$ {std:.4f}"
        if "Full" in c["config"]:
            ratio_str = "$1.00\\times$ (Base)"
        else:
            try:
                p_val = float(p_h)
                if p_val < 0.001:
                    p_str = "$< 10^{-3}$"
                else:
                    p_str = f"${p_val:.3f}$"
            except:
                p_str = "n.s."
            ratio_str = f"${ratio:.2f}\\times$ ({p_str})"
        return val_str, ratio_str

    for cfg_key, tex_name in ordered_cfgs:
        row_cells = [tex_name]
        for reg_key in ["tfim_10q", "tfim_12q", "xxz_10q", "xxz_12q"]:
            c_dict = next(item for item in data[reg_key]["configs"] if item["config"] == cfg_key)
            v_s, r_s = format_cell(c_dict)
            row_cells.append(v_s)
            row_cells.append(r_s)
        tex_content.append(" & ".join(row_cells) + " \\\\")
        
    tex_content.append("\\bottomrule")
    tex_content.append("\\end{tabular}")
    tex_content.append("\\end{table*}")
    
    tex_str = "\n".join(tex_content)
    
    # 3. Markdown and CSV Table
    md_content = []
    md_content.append("# Consolidated Component Ablation Study\n")
    md_content.append("| Configuration | TFIM 10q (Gap ± Std) | TFIM 10q Ratio (p_Holm) | TFIM 12q (Gap ± Std) | TFIM 12q Ratio (p_Holm) | XXZ 10q (Gap ± Std) | XXZ 10q Ratio (p_Holm) | XXZ 12q (Gap ± Std) | XXZ 12q Ratio (p_Holm) |")
    md_content.append("|---|---|---|---|---|---|---|---|---|")
    
    csv_rows = [["Configuration", "TFIM_10q_Gap", "TFIM_10q_Std", "TFIM_10q_Ratio", "TFIM_10q_pHolm",
                 "TFIM_12q_Gap", "TFIM_12q_Std", "TFIM_12q_Ratio", "TFIM_12q_pHolm",
                 "XXZ_10q_Gap", "XXZ_10q_Std", "XXZ_10q_Ratio", "XXZ_10q_pHolm",
                 "XXZ_12q_Gap", "XXZ_12q_Std", "XXZ_12q_Ratio", "XXZ_12q_pHolm"]]
                 
    for cfg_key, tex_name in ordered_cfgs:
        md_row = [cfg_key]
        csv_r = [cfg_key]
        for reg_key in ["tfim_10q", "tfim_12q", "xxz_10q", "xxz_12q"]:
            c_dict = next(item for item in data[reg_key]["configs"] if item["config"] == cfg_key)
            m = c_dict["gap_mean"]
            s = c_dict["gap_std"]
            r = c_dict["ratio"]
            p = c_dict["p_value_holm"]
            md_row.append(f"{m:.6f} ± {s:.6f}")
            md_row.append(f"{r:.2f}x ({p})")
            csv_r.extend([m, s, r, p])
        md_content.append("| " + " | ".join(md_row) + " |")
        csv_rows.append(csv_r)
        
    for out_dir in out_dirs:
        with open(os.path.join(out_dir, "table_ablation_consolidated.tex"), "w", encoding="utf-8") as f:
            f.write(tex_str)
        with open(os.path.join(out_dir, "table_ablation_consolidated.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
        with open(os.path.join(out_dir, "table_ablation_consolidated.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"  Saved tables in: {out_dir}")

if __name__ == "__main__":
    data_directory = "sigma_data/ablation_results"
    output_directories = [
        "sigma_data/ablation_results",
        "sigma_results/ablation_results"
    ]
    generate_consolidated_ablation(data_directory, output_directories)
