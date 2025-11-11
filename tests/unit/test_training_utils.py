import io
import json
from pathlib import Path

import numpy as np
import pytest

from training.config_utils import load_training_config, set_global_seed
from training.dataset_loaders import TemporalLivenessDataset


def test_load_training_config_overrides(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\ntrain_fusion:\n  epochs: 10\n", encoding="utf-8")

    config = load_training_config(str(config_path))

    assert config["seed"] == 42
    assert config["train_fusion"]["epochs"] == 10
    # Defaults should remain present
    assert "learning_rate" in config["train_fusion"]


def test_set_global_seed_reproducibility():
    set_global_seed(1234)
    a = np.random.rand()

    set_global_seed(1234)
    b = np.random.rand()

    assert a == pytest.approx(b)


def test_temporal_dataset_raises_when_empty(tmp_path: Path):
    dataset_root = tmp_path / "temporal"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "val").mkdir(parents=True)

    with pytest.raises(RuntimeError):
        TemporalLivenessDataset(str(dataset_root), split="train", sequence_length=10)
