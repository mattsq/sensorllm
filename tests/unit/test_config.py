"""Unit tests for config loading and composition."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml


class TestLoadConfig:
    def test_load_simple_yaml(self, tmp_path):
        from sensorllm.utils.config import load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("experiment_name: test\nseed: 7\n")
        config = load_config(cfg_file)
        assert config["experiment_name"] == "test"
        assert config["seed"] == 7

    def test_deep_merge(self, tmp_path):
        from sensorllm.utils.config import load_config

        base_file = tmp_path / "base.yaml"
        base_file.write_text("training:\n  lr: 1.0e-4\n  steps: 1000\n")

        exp_file = tmp_path / "exp.yaml"
        exp_file.write_text(
            f"_base_:\n  - {base_file}\ntraining:\n  steps: 5000\n"
        )
        config = load_config(exp_file)
        assert config["training"]["lr"] == 1e-4
        assert config["training"]["steps"] == 5000  # override applied

    def test_override_application(self, tmp_path):
        from sensorllm.utils.config import load_config

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("training:\n  lr: 1e-4\n")
        config = load_config(cfg_file, overrides={"training.lr": 5e-5})
        assert config["training"]["lr"] == 5e-5

    def test_missing_env_var_raises(self, tmp_path, monkeypatch):
        from sensorllm.utils.config import load_config

        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $NONEXISTENT_VAR_XYZ\n")
        with pytest.raises(EnvironmentError, match="NONEXISTENT_VAR_XYZ"):
            load_config(cfg_file)

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        from sensorllm.utils.config import load_config

        monkeypatch.setenv("TEST_DATA_ROOT", "/tmp/data")
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("data_root: $TEST_DATA_ROOT\n")
        config = load_config(cfg_file)
        assert config["data_root"] == "/tmp/data"
