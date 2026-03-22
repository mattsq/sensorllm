"""End-to-end integration tests for the full SensorLLM pipeline.

Tests cover:
- Forward + backward pass through encoder -> adapter -> LLM
- Training loop with SensorLLMTrainer
- Text generation conditioned on sensor signals

All tests use a tiny randomly-initialized GPT2 model (no download, no auth)
and synthetic sensor data generated on the fly. A simple local tokenizer is
created from scratch to avoid network dependencies.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sensorllm.data.synthetic.dataset_builder import SyntheticDatasetBuilder
from sensorllm.data.synthetic.sensor_generator import SensorType, SyntheticSensorConfig
from sensorllm.data.datasets.pretrain import SensorPretrainDataset
from sensorllm.models.encoders.cnn1d_encoder import CNN1DSensorEncoder
from sensorllm.models.adapters.linear_projection import LinearProjectionAdapter
from sensorllm.models.llm.hf_causal_lm import HFCausalLMBackbone
from sensorllm.models.sensorllm_model import SensorLLMModel
from sensorllm.training.trainer import SensorLLMTrainer, TrainingConfig


# -- Shared constants for tiny model config --
_LLM_HIDDEN = 64
_ENCODER_HIDDEN = 32
_N_ADAPTER_TOKENS = 4
_WINDOW_SIZE = 256
_MAX_SEQ_LEN = 64
_SENSOR_SAMPLE_RATE = 256.0
_VOCAB_SIZE = 500


def _make_local_tokenizer() -> PreTrainedTokenizerFast:
    """Build a simple whitespace-split tokenizer locally (no network)."""
    # Build a vocabulary of common English words + special chars
    words = (
        list(string.ascii_lowercase)
        + list(string.digits)
        + list(string.punctuation)
        + ["the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
           "sensor", "vibration", "signal", "normal", "fault", "bearing",
           "temperature", "pressure", "imu", "reading", "data", "anomaly",
           "frequency", "amplitude", "rms", "peak", "Hz", "degrees",
           "describe", "what", "how", "does", "this", "that", "with", "from",
           "and", "or", "not", "no", "yes", "in", "of", "to", "for", "on",
           "at", "by", "it", "be", "as", "do", "if", "so", "up", "out",
           "all", "one", "two", "can", "may", "will", "than", "more", "very",
           "show", "shows", "indicates", "detected", "operating", "condition",
           "aircraft", "engine", "analyzing", "you", "analyzing",
           "component", "harmonic", "rotation", "imbalance", "misalignment",
           "spectral", "noise", "elevated", "baseline", "dominant",
           "approximately", "consistent", "within", "acceptable", "limits"]
    )
    # Pad vocab to _VOCAB_SIZE with numbered tokens
    while len(words) < _VOCAB_SIZE - 2:  # reserve 2 for special tokens
        words.append(f"_t{len(words)}")

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for i, w in enumerate(words):
        if w not in vocab:
            vocab[w] = len(vocab)

    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[UNK]",
        eos_token="[PAD]",
    )
    return tok


@pytest.fixture
def tiny_gpt2():
    """Create a tiny GPT2 model from random weights (no download)."""
    config = GPT2Config(
        vocab_size=_VOCAB_SIZE,
        n_positions=256,
        n_embd=_LLM_HIDDEN,
        n_layer=2,
        n_head=2,
        n_inner=128,
    )
    return GPT2LMHeadModel(config)


@pytest.fixture
def tokenizer():
    """Create a simple local tokenizer (no network access needed)."""
    return _make_local_tokenizer()


@pytest.fixture
def synthetic_data(tmp_path):
    """Generate a tiny synthetic dataset to tmp_path."""
    small_config = SyntheticSensorConfig(
        sample_rate=_SENSOR_SAMPLE_RATE,
        duration_s=1.0,
        n_channels=1,
        noise_std=0.05,
        rng_seed=0,
    )
    builder = SyntheticDatasetBuilder(
        data_root=tmp_path,
        samples_per_class=2,
        sensor_types=[SensorType.VIBRATION],
        config_overrides={SensorType.VIBRATION: small_config},
        seed=0,
    )
    builder.build()
    return tmp_path


def _build_model(tiny_gpt2):
    """Assemble encoder + adapter + LLM into a SensorLLMModel."""
    encoder = CNN1DSensorEncoder(
        in_channels=1,
        hidden_dim=_ENCODER_HIDDEN,
        n_res_blocks=1,
        n_stride_layers=2,
        kernel_size=3,
        stride=4,
    )
    adapter = LinearProjectionAdapter(
        input_dim=_ENCODER_HIDDEN,
        output_dim=_LLM_HIDDEN,
        n_tokens=_N_ADAPTER_TOKENS,
    )
    llm_backbone = HFCausalLMBackbone.from_model(tiny_gpt2, freeze=False)

    return SensorLLMModel(encoder=encoder, adapter=adapter, llm=llm_backbone)


@pytest.mark.slow
class TestEndToEndPipeline:
    """Full pipeline: synthetic data -> encoder -> adapter -> LLM -> text."""

    def test_forward_backward_with_llm(self, tiny_gpt2, tokenizer, synthetic_data):
        """Forward + backward pass through the full pipeline produces gradients."""
        model = _build_model(tiny_gpt2)

        dataset = SensorPretrainDataset(
            data_root=synthetic_data,
            split="train",
            tokenizer=tokenizer,
            window_size=_WINDOW_SIZE,
            n_channels=1,
            max_length=_MAX_SEQ_LEN,
        )
        assert len(dataset) > 0, "Dataset should have samples"

        sample = dataset[0]
        # Add batch dimension
        sensor_signals = sample["sensor_signal"].unsqueeze(0)
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        labels = sample["labels"].unsqueeze(0)

        logits, loss = model(
            sensor_signals=sensor_signals,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        assert loss is not None, "Loss should not be None when labels are provided"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

        loss.backward()

        # Verify gradients flow to encoder and adapter
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.encoder.parameters()
        )
        adapter_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.adapter.parameters()
        )
        assert encoder_has_grad, "Encoder should have gradients"
        assert adapter_has_grad, "Adapter should have gradients"

    def test_training_loop_runs(self, tiny_gpt2, tokenizer, synthetic_data):
        """SensorLLMTrainer.train() completes and returns metrics."""
        model = _build_model(tiny_gpt2)

        dataset = SensorPretrainDataset(
            data_root=synthetic_data,
            split="train",
            tokenizer=tokenizer,
            window_size=_WINDOW_SIZE,
            n_channels=1,
            max_length=_MAX_SEQ_LEN,
        )

        config = TrainingConfig(
            stage=1,
            max_steps=3,
            learning_rate=1e-3,
            batch_size=2,
            logging_steps=1,
        )
        trainer = SensorLLMTrainer(
            model=model,
            config=config,
            train_dataset=dataset,
        )

        metrics = trainer.train()

        assert isinstance(metrics, dict)
        assert metrics["steps_completed"] == 3
        assert metrics["first_loss"] is not None
        assert metrics["final_loss"] is not None

    def test_generate_produces_tokens(self, tiny_gpt2, tokenizer, synthetic_data):
        """model.generate() produces a non-empty tensor of token IDs."""
        model = _build_model(tiny_gpt2)
        model.eval()

        # Build a short prompt
        prompt = "describe the sensor reading"
        enc = tokenizer(prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"]
        prompt_mask = enc["attention_mask"]

        # Clamp token IDs to be within model vocab
        prompt_ids = prompt_ids.clamp(max=_VOCAB_SIZE - 1)

        # Create a single synthetic sensor signal
        sensor_signals = torch.randn(1, 1, _WINDOW_SIZE)

        with torch.no_grad():
            output_ids = model.generate(
                sensor_signals=sensor_signals,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                max_new_tokens=10,
                do_sample=False,
            )

        assert output_ids.dim() == 2, "Output should be (B, seq_len)"
        assert output_ids.shape[0] == 1, "Batch size should be 1"
        assert output_ids.shape[1] > 0, "Should generate at least one token"

    def test_stage1_freezing(self, tiny_gpt2):
        """Stage 1 should freeze LLM and keep encoder+adapter trainable."""
        model = _build_model(tiny_gpt2)

        config = TrainingConfig(stage=1)
        trainer = SensorLLMTrainer(model=model, config=config, train_dataset=[])
        trainer._apply_stage_freezing()

        assert all(p.requires_grad for p in model.encoder.parameters())
        assert all(p.requires_grad for p in model.adapter.parameters())
        # LLM base model params should be frozen
        llm_params = list(model.llm.parameters())
        assert len(llm_params) > 0
        assert not any(p.requires_grad for p in llm_params)
