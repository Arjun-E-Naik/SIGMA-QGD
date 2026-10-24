"""
Logging utilities for step-level diagnostics and experiment summaries.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import List

try:
    from .config import StepRecord
except Exception as e:
    raise ImportError(f"Error importing StepRecord: {e}")


class ParameterLogger:
    """Collects and writes StepRecords during an optimization run."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.records: List[StepRecord] = []

    def finalise_run(self, run_records, final_gap: float, n_steps: int):
        for r in run_records:
            r.final_gap = final_gap
            r.n_steps_total = n_steps
        self.records.extend(run_records)

    def save(self, tag: str = "") -> None:
        if not self.records:
            return

        suffix = f"_{tag}" if tag else ""

        jpath = os.path.join(self.results_dir, f"step_records{suffix}.json")
        with open(jpath, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

        cpath = os.path.join(self.results_dir, f"step_records{suffix}.csv")
        keys = list(asdict(self.records[0]).keys())
        with open(cpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))
        print(f"    [Logger] {len(self.records)} records -> {self.results_dir}/")

    def run_summary_csv(self, summaries: List[dict], tag: str = "") -> None:
        if not summaries:
            return

        suffix = f"_{tag}" if tag else ""
        spath = os.path.join(self.results_dir, f"run_summaries{suffix}.csv")
        keys = list(summaries[0].keys())

        with open(spath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summaries)

        print(f"    [Logger] Run summaries -> {spath}")
