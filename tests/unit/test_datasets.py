"""Unit tests for sensor dataset classes."""

from __future__ import annotations

import pytest

from sensorllm.data.datasets import DATASET_REGISTRY
from sensorllm.data.datasets.base import BaseSensorDataset


class TestDatasetRegistry:
    def test_all_keys_registered(self):
        expected = {"aircraft_qa", "pretrain"}
        assert expected == set(DATASET_REGISTRY.keys())

    def test_all_classes_inherit_base(self):
        for name, cls in DATASET_REGISTRY.items():
            assert issubclass(cls, BaseSensorDataset), f"{name} does not inherit BaseSensorDataset"


class TestBaseSensorDatasetContract:
    """Verify that all registered datasets expose the required interface."""

    def test_len_defined(self):
        for name, cls in DATASET_REGISTRY.items():
            assert hasattr(cls, "__len__"), f"{name} missing __len__"

    def test_getitem_defined(self):
        for name, cls in DATASET_REGISTRY.items():
            assert hasattr(cls, "__getitem__"), f"{name} missing __getitem__"
