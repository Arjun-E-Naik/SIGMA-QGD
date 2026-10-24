"""
Base optimizer abstract class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

try:
    from ..core.config import Result
except ImportError:
    raise RuntimeError("Failed to import Result from core config.")


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""

    @abstractmethod
    def run(self, cost_fn, theta_init, max_steps: int = 200, **kwargs) -> tuple:
        """
        Execute optimization.

        Parameters
        ----------
        cost_fn : CostEvaluator or ShotNoiseCostEvaluator
            Cost evaluation callable.
        theta_init : np.ndarray
            Initial parameter vector.
        max_steps : int
            Maximum number of optimization iterations.
        **kwargs
            Optimizer-specific keyword arguments.

        Returns
        -------
        result : Result
            Container holding loss history, metrics, and call counts.
        run_records : list
            Per-step diagnostic records.
        best_theta : np.ndarray
            Best parameter vector found during optimization.
        """

    @property
    def supports_lr(self) -> bool:
        """Indicates if the optimizer uses a standard learning rate."""
        return False

    @property
    def default_lr(self) -> float:
        """Default learning rate value."""
        return 0.1

    def _create_progress_reporter(
        self,
        cost_fn,
        max_steps: int,
        kwargs: dict,
        default_desc: str = None,
    ):
        """Initialize progress reporter if enabled."""
        if "progress_reporter" in kwargs and kwargs["progress_reporter"] is not None:
            return kwargs["progress_reporter"]
        try:
            from ..core.progress import ProgressReporter

            exact_E = kwargs.get("exact_E", getattr(cost_fn, "exact_E", None))
            desc = kwargs.get("method_name", default_desc or self.__class__.__name__)
            seed = kwargs.get("seed", None)
            n_seeds = kwargs.get("n_seeds", None)
            show_progress = kwargs.get("show_progress", True)
            return ProgressReporter(
                total_steps=max_steps,
                desc=desc,
                exact_E=exact_E,
                seed=seed,
                n_seeds=n_seeds,
                show_progress=show_progress,
            )
        except Exception:
            class _DummyReporter:
                def update(self, *args, **kw):
                    pass

                def finish(self, *args, **kw):
                    pass

            return _DummyReporter()
