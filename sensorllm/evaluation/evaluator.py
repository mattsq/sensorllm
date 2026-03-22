"""Evaluation pipeline runner."""

from __future__ import annotations

from typing import Any


class SensorLLMEvaluator:
    """Runs evaluation over a dataset split and computes configured metrics.

    Args:
        model: SensorLLMModel to evaluate.
        dataset: Evaluation dataset.
        metrics: List of metric names from METRIC_REGISTRY.
        generation_max_new_tokens: Max tokens to generate per sample.
        batch_size: Evaluation batch size.
    """

    def __init__(
        self,
        model,
        dataset,
        metrics: list[str],
        generation_max_new_tokens: int = 128,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.generation_max_new_tokens = generation_max_new_tokens
        self.batch_size = batch_size

    def evaluate(self) -> dict[str, Any]:
        """Run evaluation and return metrics dict.

        Returns:
            Dict mapping metric name to computed value.
        """
        raise NotImplementedError("SensorLLMEvaluator.evaluate() not yet implemented")
