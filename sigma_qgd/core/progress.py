"""
Terminal progress reporter for VQE optimization routines.
"""

from __future__ import annotations

import io
import shutil
import sys
import time
from typing import Optional, Sequence

import numpy as np

# Ensure UTF-8 where possible on Windows
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper) and _stream.encoding and _stream.encoding.lower() != "utf-8":
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, io.UnsupportedOperation):
            pass


def _supports_unicode() -> bool:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    encoding = encoding.lower()
    if "utf" in encoding:
        return True
    try:
        "█─‖θ✓".encode(sys.stdout.encoding or "ascii")
        return True
    except Exception:
        return False


_UNICODE_SUPPORTED = _supports_unicode()

if _UNICODE_SUPPORTED:
    CHAR_FILL = "█"
    CHAR_EMPTY = "─"
    CHAR_CHECK = "✓"
    CHAR_THETA_NORM = "‖θ‖"
    CHAR_GRAD_NORM = "‖g‖"
    CHAR_THETA = "θ"
else:
    CHAR_FILL = "="
    CHAR_EMPTY = "-"
    CHAR_CHECK = "[OK]"
    CHAR_THETA_NORM = "|th|"
    CHAR_GRAD_NORM = "|g|"
    CHAR_THETA = "th"


class ProgressReporter:
    """
    Live terminal progress reporter for VQE optimization loops.
    """

    def __init__(
        self,
        total_steps: int,
        desc: str = "Optimizing",
        exact_E: Optional[float] = None,
        seed: Optional[int] = None,
        n_seeds: Optional[int] = None,
        show_progress: bool = True,
        min_interval: float = 0.05,
    ):
        self.total_steps = max(int(total_steps), 1)
        self.desc = desc
        self.exact_E = exact_E
        self.seed = seed
        self.n_seeds = n_seeds
        self.show_progress = show_progress
        self.min_interval = min_interval

        self.t0 = time.time()
        self.last_update_time = 0.0
        self.last_printed_len = 0
        self.best_energy = float("inf")
        self.is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.last_step_reported = 0

    def update(
        self,
        step: int,
        energy: float,
        theta: Optional[Sequence[float] | np.ndarray] = None,
        gnorm: Optional[float] = None,
        circuits: Optional[int] = None,
        extra: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Update live progress display in terminal."""
        if not self.show_progress:
            return

        now = time.time()
        is_last = step >= self.total_steps
        step = min(step, self.total_steps)

        if not self.is_tty:
            step_interval = max(1, self.total_steps // 10)
            if force or is_last or step == 1 or (step - self.last_step_reported) >= step_interval:
                self.last_step_reported = step
                self._print_log_line(step, energy, theta, gnorm, circuits, extra)
            return

        if not force and not is_last and (now - self.last_update_time) < self.min_interval:
            return

        self.last_update_time = now
        if energy is not None and energy < self.best_energy:
            self.best_energy = energy

        elapsed = now - self.t0
        rate = step / elapsed if elapsed > 0 else 0.0
        eta = (self.total_steps - step) / rate if rate > 0 else 0.0

        pct = min(100.0, 100.0 * step / self.total_steps)
        cols = shutil.get_terminal_size((100, 24)).columns

        bar_len = 12 if cols >= 120 else (8 if cols >= 100 else 6)
        filled = int(round(bar_len * step / self.total_steps))
        bar = CHAR_FILL * filled + CHAR_EMPTY * (bar_len - filled)

        if self.exact_E is not None and energy is not None:
            gap = abs(energy - self.exact_E)
            if gap < 0.001:
                gap_str = f"Gap={gap:.2e}"
            else:
                gap_str = f"Gap={gap:.5f}"
        elif energy is not None:
            gap_str = f"Loss={energy:.4f}"
        else:
            gap_str = ""

        seed_label = ""
        if self.seed is not None:
            if self.n_seeds is not None:
                seed_label = f" (S{self.seed}/{self.n_seeds})"
            else:
                seed_label = f" (S{self.seed})"
        header = f"[{self.desc}{seed_label}]"

        e_str = f"E={energy:.4f}" if energy is not None else ""

        gnorm_str = ""
        if gnorm is not None:
            gnorm_str = f"{CHAR_GRAD_NORM}={gnorm:.2e}" if gnorm < 0.01 else f"{CHAR_GRAD_NORM}={gnorm:.4f}"

        circ_str = f"{circuits:,}c" if circuits is not None else ""

        if elapsed < 60:
            time_str = f"t={elapsed:4.1f}s"
        else:
            m = int(elapsed // 60)
            s = int(elapsed % 60)
            time_str = f"t={m}m{s:02d}s"

        eta_str = f"ETA {eta:2.0f}s" if eta < 60 else f"ETA {int(eta // 60)}m{int(eta % 60):02d}s"

        if cols >= 140:
            parts = [
                f"\r  {header}",
                f"{step:3d}/{self.total_steps} [{bar} {pct:4.1f}%]",
                f"{time_str} ({eta_str})",
                e_str,
                gap_str,
                gnorm_str,
                circ_str,
                extra or "",
            ]
        elif cols >= 115:
            parts = [
                f"\r  {header}",
                f"{step:3d}/{self.total_steps} [{bar} {pct:4.1f}%]",
                f"{time_str}",
                e_str,
                gap_str,
                circ_str,
            ]
        elif cols >= 90:
            parts = [
                f"\r  {header}",
                f"{step:3d}/{self.total_steps} [{bar}]",
                f"{time_str}",
                e_str,
                gap_str,
            ]
        else:
            parts = [
                f"\r  {header[:14]}",
                f"{step:2d}/{self.total_steps}",
                e_str,
                gap_str,
            ]

        line = " | ".join(p for p in parts if p)
        if len(line) > cols - 1:
            line = line[: cols - 2]

        pad = max(0, self.last_printed_len - len(line))
        self.last_printed_len = len(line)
        sys.stdout.write(line + " " * pad)
        sys.stdout.flush()

    def _print_log_line(
        self,
        step: int,
        energy: float,
        theta: Optional[Sequence[float] | np.ndarray],
        gnorm: Optional[float],
        circuits: Optional[int],
        extra: Optional[str],
    ) -> None:
        """Formatted line print for non-interactive log files."""
        elapsed = time.time() - self.t0
        gap_str = ""
        if self.exact_E is not None and energy is not None:
            gap_str = f"  gap={abs(energy - self.exact_E):.5f}"
        circ_str = f"  circ={circuits:,}" if circuits else ""
        gnorm_str = f"  {CHAR_GRAD_NORM}={gnorm:.2e}" if gnorm is not None else ""
        seed_label = f" (Seed {self.seed})" if self.seed is not None else ""
        extra_str = f"  {extra}" if extra else ""

        sys.stdout.write(
            f"  [{self.desc}{seed_label}] Step {step}/{self.total_steps}  "
            f"E={energy:.4f}{gap_str}{gnorm_str}{circ_str}{extra_str}  t={elapsed:.1f}s\n"
        )
        sys.stdout.flush()

    def finish(
        self,
        final_energy: Optional[float] = None,
        final_theta: Optional[Sequence[float] | np.ndarray] = None,
        circuits: Optional[int] = None,
        steps: Optional[int] = None,
        extra: Optional[str] = None,
    ) -> None:
        """Clear progress line and print completion summary."""
        if not self.show_progress:
            return

        elapsed = time.time() - self.t0
        e_val = final_energy if final_energy is not None else self.best_energy

        gap_str = ""
        if self.exact_E is not None and e_val != float("inf"):
            gap_str = f"  gap={abs(e_val - self.exact_E):.5f}"

        circ_str = f"  circ={circuits:,}" if circuits is not None else ""
        step_str = f"  steps={steps}" if steps is not None else f"  steps={self.total_steps}"
        seed_label = f" [Seed {self.seed}]" if self.seed is not None else ""
        extra_str = f"  {extra}" if extra else ""

        if self.is_tty:
            clear_pad = " " * max(0, self.last_printed_len)
            sys.stdout.write(f"\r{clear_pad}\r")

        sys.stdout.write(
            f"  {CHAR_CHECK} {self.desc}{seed_label}  "
            f"E={e_val:.4f}{gap_str}{circ_str}{step_str}{extra_str}  t={elapsed:.1f}s\n"
        )
        sys.stdout.flush()


def make_progress_reporter(
    total_steps: int,
    desc: str = "Optimizing",
    exact_E: Optional[float] = None,
    seed: Optional[int] = None,
    n_seeds: Optional[int] = None,
    show_progress: bool = True,
    **kwargs,
) -> ProgressReporter:
    """Helper factory for ProgressReporter."""
    return ProgressReporter(
        total_steps=total_steps,
        desc=desc,
        exact_E=exact_E,
        seed=seed,
        n_seeds=n_seeds,
        show_progress=show_progress,
    )
