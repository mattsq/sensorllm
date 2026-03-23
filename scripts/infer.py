#!/usr/bin/env python3
"""Single-sample inference for SensorLLM.

Builds the model from a config, optionally loads a trained checkpoint,
reads a sensor file, and generates text.

Usage:
    # With a trained checkpoint
    python scripts/infer.py \\
        --config configs/experiments/exp005_cnn1d_linear_gpt2_real.yaml \\
        --checkpoint outputs/runs/exp005_cnn1d_linear_gpt2_real/final/model.pt \\
        --sensor-file data/raw/synthetic/vibration_bearing_fault_0000.h5 \\
        --prompt "Describe the sensor data."

    # Without checkpoint (uses pretrained LLM + random encoder/adapter)
    python scripts/infer.py \\
        --config configs/experiments/exp005_cnn1d_linear_gpt2_real.yaml \\
        --sensor-file data/raw/synthetic/vibration_normal_0000.h5 \\
        --prompt "What do you observe in this vibration signal?"
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from pathlib import Path

import h5py
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SensorLLM inference on a single sample")
    parser.add_argument("--config", required=True, help="Experiment config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to model.pt checkpoint")
    parser.add_argument("--sensor-file", required=True, help="Path to .h5 sensor data file")
    parser.add_argument("--prompt", default="Describe the sensor data.", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def load_sensor(path: str, window_size: int, n_channels: int) -> torch.Tensor:
    """Load a sensor signal from an HDF5 file and return as (1, C, L) tensor."""
    with h5py.File(path, "r") as f:
        signal = f["signal"][:]  # (L, C) or (L,)
        label = f["metadata"].attrs.get("label", "unknown")

    signal = torch.tensor(signal, dtype=torch.float32)
    if signal.ndim == 1:
        signal = signal.unsqueeze(-1)  # (L,) -> (L, 1)

    # Transpose to (C, L) and truncate/pad to window_size
    signal = signal.T  # (C, L)
    C = signal.shape[0]
    L = signal.shape[1]

    if L > window_size:
        signal = signal[:, :window_size]
    elif L < window_size:
        pad = torch.zeros(C, window_size - L)
        signal = torch.cat([signal, pad], dim=1)

    # Select channels
    if C > n_channels:
        signal = signal[:n_channels]

    return signal.unsqueeze(0), label  # (1, C, L)


def build_model(config: dict):
    """Reconstruct the SensorLLM model from config (same as train.py)."""
    from transformers import AutoTokenizer

    from sensorllm.models.adapters import ADAPTER_REGISTRY
    from sensorllm.models.encoders import ENCODER_REGISTRY
    from sensorllm.models.llm.hf_causal_lm import HFCausalLMBackbone
    from sensorllm.models.sensorllm_model import SensorLLMModel

    model_cfg = config["model"]
    enc_cfg = model_cfg["encoder"]
    adp_cfg = model_cfg["adapter"]
    llm_cfg = model_cfg["llm"]

    # Encoder
    enc_cls = ENCODER_REGISTRY[enc_cfg["name"]]
    enc_kwargs = {k: v for k, v in enc_cfg.items() if k not in ("name", "freeze")}
    enc_params = set(inspect.signature(enc_cls.__init__).parameters.keys()) - {"self"}
    enc_kwargs = {k: v for k, v in enc_kwargs.items() if k in enc_params}
    encoder = enc_cls(**enc_kwargs)
    log.info("Encoder: %s (output_dim=%d)", enc_cfg["name"], encoder.output_dim)

    # LLM
    llm = HFCausalLMBackbone(
        model_name_or_path=llm_cfg["name"],
        freeze=True,
        torch_dtype=llm_cfg.get("torch_dtype", "float32"),
        device_map=llm_cfg.get("device_map", "cpu"),
    )
    llm.load()
    log.info("LLM: %s (hidden_size=%d)", llm_cfg["name"], llm.hidden_size)

    # Adapter
    adp_cls = ADAPTER_REGISTRY[adp_cfg["name"]]
    adp_kwargs = {k: v for k, v in adp_cfg.items() if k != "name"}
    adp_kwargs["input_dim"] = encoder.output_dim
    adp_kwargs["output_dim"] = llm.hidden_size
    if "n_output_tokens" in adp_kwargs:
        adp_kwargs["n_tokens"] = adp_kwargs.pop("n_output_tokens")
    adp_params = set(inspect.signature(adp_cls.__init__).parameters.keys()) - {"self"}
    adp_kwargs = {k: v for k, v in adp_kwargs.items() if k in adp_params}
    adapter = adp_cls(**adp_kwargs)
    log.info("Adapter: %s (n_output_tokens=%d)", adp_cfg["name"], adapter.n_output_tokens)

    # Top-level model
    sensor_token_id = model_cfg.get("sensor_token_id", 50257)
    model = SensorLLMModel(encoder, adapter, llm, sensor_token_id=sensor_token_id)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main() -> None:
    args = parse_args()

    from sensorllm.utils.config import load_config

    config = load_config(args.config)
    data_cfg = config.get("data", {})

    # Build model
    model, tokenizer = build_model(config)

    # Load checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            log.error("Checkpoint not found: %s", ckpt_path)
            sys.exit(1)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        log.info("Loaded checkpoint: %s", ckpt_path)
    else:
        log.warning("No checkpoint specified -- using untrained encoder/adapter")

    model.eval()

    # Load sensor data
    window_size = data_cfg.get("window_size", 4096)
    n_channels = data_cfg.get("n_channels", 1)
    sensor_signal, label = load_sensor(args.sensor_file, window_size, n_channels)
    log.info("Sensor: %s (label=%s, shape=%s)", args.sensor_file, label, list(sensor_signal.shape))

    # Tokenize prompt
    prompt_tokens = tokenizer(args.prompt, return_tensors="pt", padding=True)
    prompt_ids = prompt_tokens["input_ids"]
    prompt_mask = prompt_tokens["attention_mask"]

    log.info("Prompt: %s", args.prompt)
    log.info("Generating (max_new_tokens=%d, temperature=%.2f) ...", args.max_new_tokens, args.temperature)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            sensor_signals=sensor_signal,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and print
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print()
    print("=" * 60)
    print(f"Sensor: {args.sensor_file}")
    print(f"Label:  {label}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
