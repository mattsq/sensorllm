"""Main Trainer class for SensorLLM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    Fields mirror the `training:` block in experiment YAML configs.
    """

    stage: int = 1
    max_steps: int = 10000
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    bf16: bool = True
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 50
    output_dir: str = "outputs/runs"
    seed: int = 42


class SensorLLMTrainer:
    """Trainer for SensorLLM two-stage training.

    Wraps HuggingFace Trainer with sensor-LLM-specific logic:
    - Stage-based parameter freezing
    - Sensor token injection into input sequences
    - Structured W&B logging

    Args:
        model: SensorLLMModel instance.
        config: TrainingConfig dataclass.
        train_dataset: PyTorch Dataset for training.
        eval_dataset: PyTorch Dataset for evaluation.
    """

    def __init__(
        self,
        model,
        config: TrainingConfig,
        train_dataset,
        eval_dataset=None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self) -> dict[str, Any]:
        """Run the training loop.

        Returns:
            Dict of training metrics and best checkpoint path.
        """
        raise NotImplementedError("SensorLLMTrainer.train() not yet implemented")

    def _apply_stage_freezing(self) -> None:
        """Freeze/unfreeze model components according to the training stage."""
        raise NotImplementedError
