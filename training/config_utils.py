from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch may be optional in some environments
    torch = None

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 1337,
    "experiment_log_dir": "logs/experiments",
    "output_dir": "artifacts",
    "train_fusion": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_workers": 4,
    },
    "train_liveness": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_workers": 4,
    },
    "train_temporal": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_workers": 4,
    },
}


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(value, dict)
        ):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_training_config(config_path: Optional[str]) -> Dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yml", ".yaml"}:
            overrides = yaml.safe_load(fh) or {}
        else:
            overrides = json.load(fh)

    if not isinstance(overrides, dict):
        raise ValueError("Config file must define a mapping at the top level")

    return _deep_update(config, overrides)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on runtime
            torch.cuda.manual_seed_all(seed)
