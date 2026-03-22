"""Integration smoke test: evaluation pipeline."""

from __future__ import annotations

import pytest


@pytest.mark.slow
class TestEvaluationPipelineSmoke:
    """Placeholder for evaluation pipeline integration tests.

    These will be implemented once SensorLLMEvaluator.evaluate() is complete.
    """

    def test_evaluator_instantiation(self):
        from sensorllm.evaluation.evaluator import SensorLLMEvaluator

        evaluator = SensorLLMEvaluator(
            model=None,
            dataset=None,
            metrics=["bleu", "rouge"],
            generation_max_new_tokens=32,
        )
        assert evaluator.metrics == ["bleu", "rouge"]
        assert evaluator.generation_max_new_tokens == 32
