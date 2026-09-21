"""Config loading: YAML file + dotted-path overrides (for the hyperparameter sweep)."""
from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and apply ``key.path=value`` overrides in order.

    Overrides are strings like ``"quantizer.W=1024"`` or ``"training.lr=5e-4"``;
    values are parsed with ``yaml.safe_load`` so ints/floats/bools/null work naturally.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    for override in overrides or []:
        key_path, _, raw_value = override.partition("=")
        _set_dotted(config, key_path.strip(), yaml.safe_load(raw_value.strip()))
    return config


def _set_dotted(config: dict[str, Any], key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    node = config
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
