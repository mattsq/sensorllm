"""Anomaly detection benchmark for aircraft sensor data."""

from __future__ import annotations

from typing import Any


class AnomalyDetectionBenchmark:
    """Benchmark for evaluating anomaly detection from sensor LLM outputs.

    Evaluates whether the model correctly identifies and describes anomalous
    sensor patterns versus normal operation. Supports both binary classification
    (anomaly/normal) and free-text anomaly description evaluation.

    Args:
        dataset_path: Path to the benchmark dataset (JSONL format).
        metrics: List of metrics to compute ('f1', 'bleu', 'rouge').
    """

    def __init__(self, dataset_path: str, metrics: list[str] | None = None) -> None:
        self.dataset_path = dataset_path
        self.metrics = metrics or ["f1", "bleu"]

    def run(self, model, tokenizer) -> dict[str, Any]:
        """Run the benchmark against a model.

        Args:
            model: SensorLLMModel to evaluate.
            tokenizer: Tokenizer matching the LLM backbone.

        Returns:
            Dict of metric name → score.
        """
        raise NotImplementedError("AnomalyDetectionBenchmark.run() not yet implemented")
