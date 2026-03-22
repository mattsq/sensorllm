"""Unit tests for signal-to-image transforms."""

from __future__ import annotations

import numpy as np
import pytest

from sensorllm.data.transforms import TRANSFORM_REGISTRY
from sensorllm.data.transforms.base import BaseTransform
from sensorllm.data.transforms.raw_image import RawImageTransform
from sensorllm.data.transforms.recurrence import RecurrencePlotTransform


class TestRawImageTransform:
    def test_output_shape(self, synthetic_vibration_signal):
        t = RawImageTransform(height=32, width=32)
        out = t(synthetic_vibration_signal[:1024])
        assert out.shape == (32, 32)

    def test_output_range(self, synthetic_vibration_signal):
        t = RawImageTransform(height=32, width=32)
        out = t(synthetic_vibration_signal[:1024])
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_multichannel_output_shape(self, synthetic_multichannel_signal):
        t = RawImageTransform(height=16, width=16)
        out = t(synthetic_multichannel_signal[:256])
        assert out.shape == (3, 16, 16)

    def test_padding_short_signal(self):
        t = RawImageTransform(height=64, width=64)
        short_signal = np.ones(100, dtype=np.float32)
        out = t(short_signal)
        assert out.shape == (64, 64)


class TestRecurrencePlotTransform:
    def test_output_shape(self, synthetic_vibration_signal):
        t = RecurrencePlotTransform(dimension=2, time_delay=1, image_size=32)
        out = t(synthetic_vibration_signal[:200])
        assert out.shape == (32, 32)

    def test_output_range(self, synthetic_vibration_signal):
        t = RecurrencePlotTransform(dimension=2, time_delay=1, image_size=64)
        out = t(synthetic_vibration_signal[:200])
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestTransformRegistry:
    def test_all_keys_registered(self):
        expected_keys = {"spectrogram", "cwt", "recurrence", "raw_image"}
        assert expected_keys == set(TRANSFORM_REGISTRY.keys())

    def test_all_classes_inherit_base(self):
        for name, cls in TRANSFORM_REGISTRY.items():
            assert issubclass(cls, BaseTransform), f"{name} does not inherit BaseTransform"
