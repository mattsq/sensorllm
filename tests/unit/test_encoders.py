"""Unit tests for time-series sensor encoders."""

from __future__ import annotations

import pytest
import torch

from sensorllm.models.encoders import ENCODER_REGISTRY
from sensorllm.models.encoders.base import SensorEncoder
from sensorllm.models.encoders.cnn1d_encoder import CNN1DSensorEncoder
from sensorllm.models.encoders.transformer_encoder import TransformerSensorEncoder
from sensorllm.models.encoders.patchtst_encoder import PatchTSTSensorEncoder


class TestCNN1DSensorEncoder:
    def test_output_shape(self):
        enc = CNN1DSensorEncoder(in_channels=1, hidden_dim=32, n_res_blocks=1, n_stride_layers=2, stride=4)
        x = torch.randn(2, 1, 256)  # (B, C, L)
        out = enc(x)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[2] == 32  # hidden_dim

    def test_output_dim_property(self):
        enc = CNN1DSensorEncoder(in_channels=1, hidden_dim=64)
        assert enc.output_dim == 64

    def test_multichannel_input(self):
        enc = CNN1DSensorEncoder(in_channels=3, hidden_dim=32, n_stride_layers=2, stride=4)
        x = torch.randn(2, 3, 256)
        out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 32

    def test_inherits_sensor_encoder(self):
        assert issubclass(CNN1DSensorEncoder, SensorEncoder)

    def test_output_dtype_float32(self):
        enc = CNN1DSensorEncoder(in_channels=1, hidden_dim=16, n_stride_layers=1, stride=4)
        x = torch.randn(1, 1, 64)
        out = enc(x)
        assert out.dtype == torch.float32


class TestTransformerSensorEncoder:
    def test_output_shape(self):
        enc = TransformerSensorEncoder(in_channels=1, patch_size=16, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(2, 1, 64)   # L=64, patch_size=16 → N=4
        out = enc(x)
        assert out.shape == (2, 4, 32)

    def test_output_dim_property(self):
        enc = TransformerSensorEncoder(in_channels=1, patch_size=16, d_model=64, n_heads=2, n_layers=1)
        assert enc.output_dim == 64

    def test_patch_size_must_divide_length(self):
        enc = TransformerSensorEncoder(in_channels=1, patch_size=16, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(1, 1, 65)   # 65 not divisible by 16
        with pytest.raises(ValueError, match="not divisible by patch_size"):
            enc(x)

    def test_multichannel_input(self):
        enc = TransformerSensorEncoder(in_channels=3, patch_size=16, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(2, 3, 64)
        out = enc(x)
        assert out.shape == (2, 4, 32)

    def test_learned_positional_encoding(self):
        enc = TransformerSensorEncoder(
            in_channels=1, patch_size=16, d_model=32, n_heads=2, n_layers=1,
            positional_encoding="learned"
        )
        x = torch.randn(1, 1, 64)
        out = enc(x)
        assert out.shape == (1, 4, 32)

    def test_inherits_sensor_encoder(self):
        assert issubclass(TransformerSensorEncoder, SensorEncoder)


class TestPatchTSTSensorEncoder:
    def test_output_shape(self):
        enc = PatchTSTSensorEncoder(in_channels=1, patch_len=16, stride=8, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(2, 1, 64)
        # N = floor((64 - 16) / 8) + 1 = 7
        out = enc(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 7
        assert out.shape[2] == 32

    def test_output_dim_property(self):
        enc = PatchTSTSensorEncoder(in_channels=1, patch_len=16, stride=8, d_model=48, n_heads=2, n_layers=1)
        assert enc.output_dim == 48

    def test_multichannel_aggregation(self):
        enc = PatchTSTSensorEncoder(in_channels=3, patch_len=16, stride=16, d_model=32, n_heads=2, n_layers=1)
        x = torch.randn(2, 3, 64)
        out = enc(x)
        # multi-channel → mean aggregation → same shape as single channel
        assert out.shape[0] == 2
        assert out.shape[2] == 32

    def test_inherits_sensor_encoder(self):
        assert issubclass(PatchTSTSensorEncoder, SensorEncoder)


class TestEncoderRegistry:
    def test_all_keys_registered(self):
        expected = {"cnn1d", "transformer", "patchtst"}
        assert expected == set(ENCODER_REGISTRY.keys())

    def test_all_classes_inherit_base(self):
        for name, cls in ENCODER_REGISTRY.items():
            assert issubclass(cls, SensorEncoder), f"{name} does not inherit SensorEncoder"

    def test_no_image_encoders_in_registry(self):
        assert "vit_b16" not in ENCODER_REGISTRY
        assert "resnet50" not in ENCODER_REGISTRY
