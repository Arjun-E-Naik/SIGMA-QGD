
"""
Re-plot from saved CSVs without re-running any simulation.

Usage:
    python scripts/generate_plots.py --results-dir /path/to/sigma_qgd_results --tag tfim_10q_v1
"""

import argparse
import csv
import io
import os
import sys

# Ensure UTF-8 encoding for console output on Windows
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper) and stream.encoding and stream.encoding.lower() != "utf-8":
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma_qgd.analysis.plots import (
    plot_shot_noise, plot_scaling, plot_lr_sweep, plot_sensitivity_sweep
)

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


def _read_csv_dicts(path):
    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path}")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description="Re-generate all SIGMA-QGD plots from existing CSVs."
    )
    parser.add_argument("--results-dir", "-r", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tag",  "-t", default="", help="Run tag used in filenames")
    parser.add_argument("--plot", "-p", default="all",
                        choices=["all", "shot_noise", "scaling", "lr_sweep", "sensitivity"],
                        help="Which plot to generate")
    args = parser.parse_args()

    data_dir = os.path.join(args.results_dir, "data")
    figs_dir = os.path.join(args.results_dir, "figures")
    sfx = f"_{args.tag}" if args.tag else ""

    def path(name): return os.path.join(data_dir, f"{name}{sfx}.csv")

    # Shot noise
    if args.plot in ("all", "shot_noise"):
        rows = _read_csv_dicts(path("shot_noise_results"))
        if rows:
            shot_results = {}
            for row in rows:
                key = "noiseless" if row["n_shots"] == "inf" else int(row["n_shots"])
                shot_results[key] = {
                    "gap_mean": float(row["gap_mean"]),
                    "gap_std":  float(row["gap_std"]),
                    "n_seeds":  int(row["n_seeds"]),
                    "n_qubits": int(row["n_qubits"]),
                }
            n_qubits = int(rows[0]["n_qubits"])
            plot_shot_noise(shot_results, n_qubits, figs_dir, args.tag)

    # Scaling
    if args.plot in ("all", "scaling"):
        rows = _read_csv_dicts(path("scaling_results"))
        if rows:
            scaling_results = {}
            for row in rows:
                nq = int(row["n_qubits"])
                scaling_results[nq] = {k: float(v) if k not in ("n_qubits", "n_seeds")
                                        else int(v) for k, v in row.items()}
            plot_scaling(scaling_results, figs_dir, args.tag)

    # LR sweep
    if args.plot in ("all", "lr_sweep"):
        rows = _read_csv_dicts(path("lr_sweep_summary"))
        if rows:
            summary = [{**r, "best_lr": float(r["best_lr"]),
                        "best_gap": float(r["best_gap"]),
                        "best_std": float(r["best_std"])} for r in rows]
            plot_lr_sweep(summary, figs_dir, args.tag)

    # Sensitivity
    if args.plot in ("all", "sensitivity"):
        rows = _read_csv_dicts(path("sensitivity_sweep"))
        if rows:
            cleaned = [{**r, "value": float(r["value"]),
                        "gap_mean": float(r["gap_mean"]),
                        "gap_std": float(r["gap_std"])} for r in rows]
            n_qubits = int(cleaned[0]["n_qubits"]) if cleaned else 0
            plot_sensitivity_sweep(cleaned, "TFIM", n_qubits, figs_dir, args.tag)

    print(f"  Done. Figures → {figs_dir}/")


if __name__ == "__main__":
    main()
