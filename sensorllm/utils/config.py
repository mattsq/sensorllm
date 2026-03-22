"""YAML config loading with _base_ inheritance composition."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML experiment config with base config inheritance.

    Processes the `_base_` key in the config: loads each base config in order
    and deep-merges them, then merges the current config on top.

    Environment variable substitution: values of the form `$ENV_VAR` are
    replaced with the corresponding environment variable value.

    Args:
        path: Path to the YAML config file.
        overrides: Optional dict of dot-notation key=value overrides
            (e.g., {'training.learning_rate': 5e-5}).

    Returns:
        Merged config dict with all base configs applied and env vars substituted.

    Example:
        config = load_config("configs/experiments/exp001_linear_proj_vit.yaml")
        config = load_config(
            "configs/experiments/exp001_linear_proj_vit.yaml",
            overrides={"training.max_steps": 20000}
        )
    """
    path = Path(path)
    raw = _load_yaml(path)
    config = _resolve_bases(raw, base_dir=path.parent)
    config = _substitute_env_vars(config)
    if overrides:
        config = _apply_overrides(config, overrides)
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_bases(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base_path in bases:
        base = _load_yaml(Path(base_path) if Path(base_path).is_absolute() else Path.cwd() / base_path)
        base = _resolve_bases(base, base_dir=Path(base_path).parent)
        merged = _deep_merge(merged, base)
    return _deep_merge(merged, config)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _substitute_env_vars(config: Any) -> Any:
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    if isinstance(config, list):
        return [_substitute_env_vars(v) for v in config]
    if isinstance(config, str) and config.startswith("$"):
        env_key = config[1:]
        val = os.environ.get(env_key)
        if val is None:
            raise EnvironmentError(f"Required environment variable '{env_key}' is not set")
        return val
    return config


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for dot_key, value in overrides.items():
        keys = dot_key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config
