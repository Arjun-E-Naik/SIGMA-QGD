"""
Plot generators for experimental evaluation and convergence visualization.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..style import COLOURS, LINES, LW, MARKERS


def _set_rcparams():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "legend.fontsize": 8, "xtick.labelsize": 10, "ytick.labelsize": 10,
    })



def plot_convergence(
    all_res_t, exact_E_t,
    all_res_x, exact_E_x,
    n_seeds: int, n_qubits: int,
    figures_dir: str, tag: str = "",
):
    """Energy convergence curves for TFIM and XXZ."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""

    def _conv_axes(ax, all_res, exact_E, title, log_scale):
        for name in COLOURS:
            if name not in all_res or not all_res[name]:
                continue
            hists = [r.energy_hist for r in all_res[name]]
            mn    = min(len(h) for h in hists)
            if mn == 0:
                continue
            arr = np.array([h[:mn] for h in hists])
            mu  = arr.mean(0); sd = arr.std(0); xs = np.arange(mn)
            c   = COLOURS[name]; lw = LW.get(name, 1.5); ls = LINES.get(name, "-")
            zo  = 12 if "SIGMA" in name else 5
            if log_scale:
                ax.semilogy(xs, np.maximum(np.abs(mu - exact_E), 1e-10),
                            color=c, lw=lw, ls=ls, label=name, zorder=zo)
            else:
                ax.plot(xs, mu, color=c, lw=lw, ls=ls, label=name, zorder=zo)
                ax.fill_between(xs, mu - sd, mu + sd, alpha=0.1, color=c)
        if not log_scale:
            ax.axhline(exact_E, color="k", ls=":", lw=1.2, label=f"Exact ({exact_E:.3f})")
            ax.set_ylabel("Energy (H)")
        else:
            ax.set_ylabel("|E−E_exact| (log)")
        ax.set_xlabel("Step"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)

    if all_res_t:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v9.0 — TFIM (n_qubits={n_qubits}, {n_seeds} seeds)",
                     fontsize=13, fontweight="bold")
        _conv_axes(axes[0], all_res_t, exact_E_t, "Energy convergence — TFIM", False)
        _conv_axes(axes[1], all_res_t, exact_E_t, "Energy gap — TFIM (log)", True)
        plt.tight_layout()
        path = os.path.join(figures_dir, f"fig1_tfim_convergence{sfx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {path}")

    if all_res_x:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"SIGMA-QGD v9.0 — XXZ (n_qubits={n_qubits}, {n_seeds} seeds)",
                     fontsize=13, fontweight="bold")
        _conv_axes(axes[0], all_res_x, exact_E_x, "Energy convergence — XXZ", False)
        _conv_axes(axes[1], all_res_x, exact_E_x, "Energy gap — XXZ (log)", True)
        plt.tight_layout()
        path = os.path.join(figures_dir, f"fig2_xxz_convergence{sfx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {path}")


def plot_boxplots(summary_t, exact_E_t, summary_x, exact_E_x,
                   n_seeds: int, n_qubits: int,
                   figures_dir: str, tag: str = ""):
    """Final energy distributions as box plots."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)

    summaries = [(s, e, t) for s, e, t in [
        (summary_t, exact_E_t, "TFIM"), (summary_x, exact_E_x, "XXZ")
    ] if s is not None]

    if not summaries:
        return

    fig, axes = plt.subplots(1, len(summaries), figsize=(8 * len(summaries), 6))
    if len(summaries) == 1:
        axes = [axes]

    for ax, (smry, exact, title) in zip(axes, summaries):
        labels = [m for m in COLOURS if m in smry]
        groups = [smry[m]["energy_all"] for m in labels]
        colors = [COLOURS[m] for m in labels]
        tick_labels = [l.replace(" ", "\n") for l in labels]
        bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color="k", lw=2))
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        ax.axhline(exact, color="k", ls=":", lw=1.5, label=f"Exact ({exact:.3f})")
        ax.set_ylabel("Final energy (H)")
        ax.set_title(f"{title} — n_qubits={n_qubits}, {n_seeds} seeds")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)

    fig.suptitle("Final energy distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig3_boxplots{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_efficiency(summary_t, exact_E_t, summary_x, exact_E_x,
                     n_qubits: int, figures_dir: str, tag: str = ""):
    """Circuit efficiency scatter: circuits used vs gap achieved."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    if not (summary_t and summary_x):
        print("  [WARNING] fig4_efficiency requires both TFIM and XXZ — skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (smry, exact, title) in zip(axes,
            [(summary_t, exact_E_t, "TFIM"), (summary_x, exact_E_x, "XXZ")]):
        for name in COLOURS:
            if name not in smry:
                continue
            s   = smry[name]; c = COLOURS[name]; ms = 200 if "SIGMA" in name else 80
            for e, ci in zip(s["energy_all"], s["circuits_all"]):
                ax.scatter(ci, abs(e - exact), color=c, alpha=0.2, s=25,
                           marker=MARKERS.get(name, "o"))
            ax.scatter(s["circuits_mean"], s["gap_mean"], color=c, s=ms,
                       marker=MARKERS.get(name, "o"), zorder=10, label=name,
                       edgecolors="k", linewidths=1.5)
        ax.set_xlabel("Circuit evaluations"); ax.set_ylabel("|E−E_exact| (log)")
        ax.set_title(f"Circuit efficiency — {title} (n_qubits={n_qubits})")
        ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Circuit efficiency: bottom-left = best", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig4_efficiency{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_ablation_heatmap(ablation_results: dict, exact_E: float,
                           n_qubits: int, figures_dir: str, tag: str = ""):
    """Horizontal bar chart of ablation gaps."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    configs   = list(ablation_results.keys())
    gaps_mean = [np.mean([abs(r.energy_opt - exact_E) for r in ablation_results[c]])
                 for c in configs]
    gaps_std  = [np.std([abs(r.energy_opt - exact_E) for r in ablation_results[c]])
                 for c in configs]
    fig, ax   = plt.subplots(figsize=(10, max(6, len(configs) * 0.5 + 2)))
    colors    = ["#00C9A7" if "Full" in c else "#E24B4A" for c in configs]
    ax.barh(range(len(configs)), gaps_mean, xerr=gaps_std,
            color=colors, alpha=0.8, edgecolor="k", lw=0.5, capsize=4)
    base_gap = gaps_mean[0]
    for i, (g, gs, c) in enumerate(zip(gaps_mean, gaps_std, configs)):
        ratio = g / max(base_gap, 1e-12)
        label = "1.0x (base)" if "Full" in c else f"{ratio:.1f}x"
        ax.text(g + gs + 0.0001, i, label, va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(configs))); ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel("Energy gap |E − E_exact| (H)", fontsize=11)
    n_seeds_abl = len(list(ablation_results.values())[0]) if ablation_results else 0
    ax.set_title(f"Ablation Study (n_qubits={n_qubits}, {n_seeds_abl} seeds)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig8_ablation{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_shot_noise(shot_results: dict, n_qubits: int,
                    figures_dir: str, tag: str = ""):
    """Shot-noise robustness curve."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    if not shot_results:
        return
    shot_keys = sorted(k for k in shot_results.keys() if k != "noiseless")
    gaps = [shot_results[k]["gap_mean"] for k in shot_keys]
    stds = [shot_results[k]["gap_std"]  for k in shot_keys]
    n_seeds = shot_results.get(shot_keys[0], {}).get("n_seeds", "?") if shot_keys else "?"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(shot_keys, gaps, yerr=stds, fmt="D-", color="#00C9A7",
                lw=2, markersize=8, capsize=5, markeredgecolor="k",
                markeredgewidth=1.5, label="SIGMA-QGD v9.0")
    if "noiseless" in shot_results:
        nl_gap = shot_results["noiseless"]["gap_mean"]
        ax.axhline(nl_gap, color="k", ls=":", lw=1.5, label=f"Noiseless ({nl_gap:.4f})")
    ax.set_xlabel("Number of shots", fontsize=11)
    ax.set_ylabel("Energy gap |E − E_exact| (exact eval)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title(f"Shot-Noise Robustness (n_qubits={n_qubits}, {n_seeds} seeds)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig10_shot_noise{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_scaling(scaling_results: dict, figures_dir: str, tag: str = ""):
    """Bar chart of SIGMA-QGD vs Adam across qubit counts."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    if not scaling_results:
        return
    qubits     = sorted(scaling_results.keys())
    sigma_gaps = [scaling_results[nq]["sigma_gap_mean"] for nq in qubits]
    sigma_stds = [scaling_results[nq]["sigma_gap_std"]  for nq in qubits]
    adam_gaps  = [scaling_results[nq]["adam_gap_mean"]  for nq in qubits]
    adam_stds  = [scaling_results[nq]["adam_gap_std"]   for nq in qubits]
    n_seeds    = scaling_results[qubits[0]].get("n_seeds", "?")
    x = np.arange(len(qubits)); w = 0.3
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, sigma_gaps, w, yerr=sigma_stds, color="#00C9A7",
           alpha=0.85, edgecolor="k", lw=0.5, capsize=5, label="SIGMA-QGD v9.0")
    ax.bar(x + w/2, adam_gaps, w, yerr=adam_stds, color="#E24B4A",
           alpha=0.85, edgecolor="k", lw=0.5, capsize=5, label="Adam")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nq}q (p={scaling_results[nq]['p']})" for nq in qubits], fontsize=9)
    ax.set_xlabel("System size", fontsize=11)
    ax.set_ylabel("Energy gap |E − E_exact| (H)", fontsize=11)
    ax.set_title(f"Scaling: SIGMA-QGD v9.0 vs Adam (TFIM, {n_seeds} seeds)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig11_scaling{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_lr_sweep(lr_sweep_summary: List[dict], figures_dir: str, tag: str = ""):
    """
    Bar chart of best-LR gap per method from the LR sweep.
    Reads from lr_sweep_summary data (list of dicts).
    """
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    if not lr_sweep_summary:
        return

    methods = [r["method"] for r in lr_sweep_summary]
    gaps    = [r["best_gap"] for r in lr_sweep_summary]
    stds    = [r["best_std"] for r in lr_sweep_summary]
    best_lrs = [r["best_lr"] for r in lr_sweep_summary]
    colors  = [COLOURS.get(m, "#888888") for m in methods]
    x       = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, gaps, yerr=stds, color=colors, alpha=0.85,
                  edgecolor="k", lw=0.5, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(lr={lr})" for m, lr in zip(methods, best_lrs)],
                       fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Energy gap |E − E_exact| at best LR", fontsize=11)
    ax.set_title("LR Sweep — Best Gap per Method (fair comparison)\n"
                 "Each method tuned independently on the same seeds",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig_lr_sweep{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_sensitivity_sweep(rows: List[dict], ham_name: str,
                            n_qubits: int, figures_dir: str, tag: str = ""):
    """Two-panel sensitivity sweep plot (multiplier, lambda_reg)."""
    _set_rcparams()
    sfx = f"_{tag}" if tag else ""
    os.makedirs(figures_dir, exist_ok=True)
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for ax, sweep_name, xlabel, default_val in zip(
            axes, ["multiplier", "lambda_reg"],
            ["Multiplier $c$ (in $F_i = c\\sigma_i^2 + \\lambda_{\\mathrm{reg}}$)",
             "Regularization parameter $\\lambda_{\\mathrm{reg}}$"],
            [1.0, 0.001]):
        sub = [r for r in rows if r["sweep"] == sweep_name]
        if not sub:
            continue
        xs  = [r["value"]    for r in sub]
        ys  = [r["gap_mean"] for r in sub]
        es  = [r["gap_std"]  for r in sub]
        ax.errorbar(xs, ys, yerr=es, marker="o", color="#00C9A7", capsize=4,
                    label="Sensitivity profile")
        if default_val is not None:
            ax.axvline(default_val, color="gray", linestyle=":", alpha=0.7,
                       label=f"Deployed default ({default_val})")
        if sweep_name == "lambda_reg":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("Energy gap (mean ± std)")
        ax.set_title(f"Sensitivity: {sweep_name}"); ax.grid(alpha=0.3)
        ax.legend(frameon=True, fontsize=9)
    fig.suptitle(f"Sensitivity sweep — {ham_name}, n_qubits={n_qubits}",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"fig_sensitivity{sfx}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
