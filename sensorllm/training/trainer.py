"""Main Trainer class for SensorLLM experiments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    Fields mirror the `training:` block in experiment YAML configs.
    """

    stage: int = 1
    max_steps: int = 10000
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    batch_size: int = 4
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

    Uses a plain PyTorch training loop with stage-based parameter freezing.

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

    def _get_device(self) -> torch.device:
        """Infer the device from model parameters."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def train(self) -> dict[str, Any]:
        """Run the training loop.

        Returns:
            Dict of training metrics and step count.
        """
        self._apply_stage_freezing()

        device = self._get_device()
        logger.info("Training on device: %s", device)

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config.learning_rate)

        self.model.train()
        step = 0
        total_loss = 0.0
        first_loss = None

        while step < self.config.max_steps:
            for batch in dataloader:
                if step >= self.config.max_steps:
                    break

                # Move batch tensors to the model's device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                logits, loss = self.model(
                    sensor_signals=batch["sensor_signal"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_val = loss.item()
                total_loss += loss_val
                if first_loss is None:
                    first_loss = loss_val
                step += 1

                if step % max(1, self.config.logging_steps) == 0:
                    logger.info("Step %d / %d  loss=%.4f", step, self.config.max_steps, loss_val)

                if self.config.save_steps > 0 and step % self.config.save_steps == 0:
                    self._save_checkpoint(step)

            if step >= self.config.max_steps:
                break

        # Always save final checkpoint
        self._save_checkpoint(step, label="final")

        avg_loss = total_loss / max(step, 1)
        logger.info(
            "Training complete: %d steps, avg_loss=%.4f", step, avg_loss
        )
        return {
            "steps_completed": step,
            "first_loss": first_loss,
            "final_loss": loss_val if step > 0 else None,
            "avg_loss": avg_loss,
        }

    def _save_checkpoint(self, step: int, label: str | None = None) -> None:
        """Save model state_dict to a checkpoint directory."""
        dirname = label if label else f"checkpoint-{step}"
        ckpt_dir = Path(self.config.output_dir) / dirname
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / "model.pt"
        torch.save(self.model.state_dict(), path)
        logger.info("Saved checkpoint: %s", path)

    def _apply_stage_freezing(self) -> None:
        """Freeze/unfreeze model components according to the training stage."""
        if self.config.stage == 1:
            # Stage 1: freeze LLM, train encoder + adapter
            self.model.encoder.requires_grad_(True)
            self.model.adapter.requires_grad_(True)
            self.model.llm.requires_grad_(False)
            logger.info("Stage 1 freezing: LLM frozen, encoder+adapter trainable")
        elif self.config.stage == 2:
            # Stage 2: freeze encoder, train adapter + LLM (LoRA)
            self.model.encoder.requires_grad_(False)
            self.model.adapter.requires_grad_(True)
            self.model.llm.requires_grad_(True)
            logger.info("Stage 2 freezing: encoder frozen, adapter+LLM trainable")
