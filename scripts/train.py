#!/usr/bin/env python3
"""Training entry point for SensorLLM experiments.

Usage:
    python scripts/train.py --config configs/experiments/exp004_cnn1d_linear_gpt2.yaml
    python scripts/train.py --config configs/experiments/exp004_cnn1d_linear_gpt2.yaml \
        --override training.learning_rate=5e-5 training.max_steps=20000
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SensorLLM model")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-notation config overrides (e.g. training.learning_rate=1e-4)",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    from sensorllm.utils.config import load_config
    from sensorllm.utils.logging import get_logger, init_wandb
    from sensorllm.utils.reproducibility import set_seed
    from sensorllm.models.encoders import ENCODER_REGISTRY
    from sensorllm.models.adapters import ADAPTER_REGISTRY
    from sensorllm.models.llm.hf_causal_lm import HFCausalLMBackbone
    from sensorllm.models.sensorllm_model import SensorLLMModel
    from sensorllm.data.datasets.pretrain import SensorPretrainDataset
    from sensorllm.training.trainer import SensorLLMTrainer, TrainingConfig

    overrides = {}
    for item in args.override:
        key, _, val = item.partition("=")
        overrides[key] = val

    config = load_config(args.config, overrides=overrides if overrides else None)
    logger = get_logger(__name__)
    logger.info("Starting experiment: %s", config.get("experiment_name"))
    logger.info("Config loaded from: %s", args.config)

    set_seed(config.get("seed", 42))
    init_wandb(config)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Build encoder
    enc_cfg = model_cfg["encoder"]
    enc_cls = ENCODER_REGISTRY[enc_cfg["name"]]
    enc_kwargs = {k: v for k, v in enc_cfg.items() if k not in ("name", "freeze")}
    # Filter to only params accepted by the encoder constructor
    enc_params = set(inspect.signature(enc_cls.__init__).parameters.keys()) - {"self"}
    enc_kwargs = {k: v for k, v in enc_kwargs.items() if k in enc_params}
    encoder = enc_cls(**enc_kwargs)
    logger.info("Encoder: %s (output_dim=%d)", enc_cfg["name"], encoder.output_dim)

    # Build LLM backbone
    llm_cfg = model_cfg["llm"]
    llm = HFCausalLMBackbone(
        model_name_or_path=llm_cfg["name"],
        freeze=llm_cfg.get("freeze", True),
        lora_config=llm_cfg.get("lora"),
        torch_dtype=llm_cfg.get("torch_dtype", "float32"),
        device_map=llm_cfg.get("device_map", "cpu"),
    )
    llm.load()
    logger.info("LLM: %s (hidden_size=%d)", llm_cfg["name"], llm.hidden_size)

    # Build adapter (needs encoder output_dim and llm hidden_size)
    adp_cfg = model_cfg["adapter"]
    adp_cls = ADAPTER_REGISTRY[adp_cfg["name"]]
    adp_kwargs = {k: v for k, v in adp_cfg.items() if k != "name"}
    adp_kwargs["input_dim"] = encoder.output_dim
    adp_kwargs["output_dim"] = llm.hidden_size
    # Map config keys to constructor parameter names
    if "n_output_tokens" in adp_kwargs:
        adp_kwargs["n_tokens"] = adp_kwargs.pop("n_output_tokens")
    # Filter to only params accepted by the adapter constructor
    adp_params = set(inspect.signature(adp_cls.__init__).parameters.keys()) - {"self"}
    adp_kwargs = {k: v for k, v in adp_kwargs.items() if k in adp_params}
    adapter = adp_cls(**adp_kwargs)
    logger.info("Adapter: %s (n_output_tokens=%d)", adp_cfg["name"], adapter.n_output_tokens)

    # Build top-level model
    sensor_token_id = model_cfg.get("sensor_token_id", 50257)
    model = SensorLLMModel(encoder, adapter, llm, sensor_token_id=sensor_token_id)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build datasets
    data_root = data_cfg.get("data_root", "data")
    train_dataset = SensorPretrainDataset(
        data_root=data_root,
        split="train",
        tokenizer=tokenizer,
        window_size=data_cfg.get("window_size", 4096),
        n_channels=data_cfg.get("n_channels", 1),
    )
    val_dataset = SensorPretrainDataset(
        data_root=data_root,
        split="val",
        tokenizer=tokenizer,
        window_size=data_cfg.get("window_size", 4096),
        n_channels=data_cfg.get("n_channels", 1),
    )
    logger.info("Train samples: %d, Val samples: %d", len(train_dataset), len(val_dataset))

    if len(train_dataset) == 0:
        logger.error(
            "No training samples found. Generate synthetic data first:\n"
            "  python scripts/generate_synthetic_data.py --data-root %s --samples 20",
            data_root,
        )
        sys.exit(1)

    # Build trainer
    experiment_name = config.get("experiment_name", "run")
    output_dir = train_cfg.get("output_dir", "outputs/runs")
    run_output_dir = str(Path(output_dir) / experiment_name)

    training_config = TrainingConfig(
        stage=int(train_cfg.get("stage", 1)),
        max_steps=int(train_cfg.get("max_steps", 10000)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        warmup_steps=int(train_cfg.get("warmup_steps", 500)),
        batch_size=int(data_cfg.get("batch_size", 4)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", False)),
        save_steps=int(train_cfg.get("save_steps", 1000)),
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        output_dir=run_output_dir,
        seed=int(config.get("seed", 42)),
    )

    trainer = SensorLLMTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
    )

    results = trainer.train()
    logger.info("Training complete. Results: %s", results)


if __name__ == "__main__":
    main()
