"""Checkpoint save/load and artifact I/O helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model,
    optimizer,
    step: int,
    metrics: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Save a training checkpoint.

    Saves model state dict, optimizer state, step count, and metrics.
    Creates a directory: `output_dir/checkpoint-{step}/`.

    Args:
        model: SensorLLMModel to save.
        optimizer: Optimizer whose state to save.
        step: Current training step.
        metrics: Dict of current metrics to record alongside the checkpoint.
        output_dir: Base directory for checkpoints.

    Returns:
        Path to the saved checkpoint directory.
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump({"step": step, "metrics": metrics}, f, indent=2)
    return ckpt_dir


def load_checkpoint(model, checkpoint_dir: str | Path, optimizer=None) -> dict[str, Any]:
    """Load a training checkpoint into model (and optionally optimizer).

    Args:
        model: SensorLLMModel to load weights into.
        checkpoint_dir: Path to checkpoint directory.
        optimizer: If provided, load optimizer state as well.

    Returns:
        Metadata dict (contains 'step' and 'metrics').
    """
    ckpt_dir = Path(checkpoint_dir)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu"))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location="cpu"))
    with open(ckpt_dir / "metadata.json") as f:
        return json.load(f)


def symlink_best_model(checkpoint_dir: Path, run_dir: Path) -> None:
    """Create or update the best_model/ symlink in the run directory.

    Args:
        checkpoint_dir: Checkpoint directory to symlink to.
        run_dir: Run output directory where best_model/ link will be created.
    """
    link = run_dir / "best_model"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(checkpoint_dir.resolve())


def append_metrics_jsonl(metrics: dict[str, Any], path: str | Path) -> None:
    """Append a metrics dict as a JSON line to a .jsonl file.

    Args:
        metrics: Metrics dict to serialize.
        path: Path to the .jsonl file (created if not exists).
    """
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
