"""Custom training callbacks."""

from __future__ import annotations


class BestModelCallback:
    """Saves and symlinks best_model/ on validation loss improvement.

    Tracks the best validation loss seen so far and saves a checkpoint
    whenever it improves. Maintains a `best_model/` symlink in the run directory.
    """

    def __init__(self, output_dir: str, metric: str = "eval_loss", mode: str = "min") -> None:
        self.output_dir = output_dir
        self.metric = metric
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.mode = mode

    def on_evaluate(self, metrics: dict, step: int, checkpoint_dir: str) -> None:
        """Called after each evaluation. Saves best checkpoint if improved.

        Args:
            metrics: Dict of evaluation metrics.
            step: Current training step.
            checkpoint_dir: Path to the latest checkpoint directory.
        """
        raise NotImplementedError("BestModelCallback.on_evaluate() not yet implemented")


class FrozenParamCallback:
    """Asserts that frozen parameters remain frozen throughout training.

    Guards against accidental gradient flow into frozen components (e.g., encoder
    parameters being updated during Stage 1 when only the adapter should train).
    """

    def __init__(self, frozen_module_names: list[str]) -> None:
        self.frozen_module_names = frozen_module_names
        self._initial_checksums: dict[str, float] = {}

    def on_train_begin(self, model) -> None:
        """Record initial parameter checksums for frozen modules."""
        raise NotImplementedError

    def on_step_end(self, model, step: int) -> None:
        """Verify frozen parameters are unchanged."""
        raise NotImplementedError


class MetricsLoggerCallback:
    """Logs per-step metrics to W&B and a local metrics.jsonl file."""

    def __init__(self, run_dir: str, use_wandb: bool = True) -> None:
        self.run_dir = run_dir
        self.use_wandb = use_wandb

    def on_log(self, metrics: dict, step: int) -> None:
        """Log metrics dict at the given step."""
        raise NotImplementedError
