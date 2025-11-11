"""Hyperparameter optimization utilities using Optuna."""

from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Optional

import optuna

from training.config_utils import load_training_config, set_global_seed
from training.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter tuning with Optuna and experiment tracking."""

    def __init__(
        self,
        study_name: str = "facial_auth_study",
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        experiment_log_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.study_name = study_name
        self.config = load_training_config(config_path)
        self.seed = seed if seed is not None else self.config["seed"]
        set_global_seed(self.seed)

        tracker_name = experiment_name or f"{study_name}_optuna"
        log_dir = experiment_log_dir or self.config["experiment_log_dir"]
        self.tracker = ExperimentTracker(tracker_name, log_dir, {"seed": self.seed})

        self.study: Optional[optuna.Study] = None

    def create_study(self, direction: str = "maximize", storage: Optional[str] = None) -> None:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner()

        self.study = optuna.create_study(
            study_name=self.study_name,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            storage=storage,
            load_if_exists=True,
        )
        logger.info("Created study: %s", self.study_name)

    def optimize(
        self,
        objective_func: Callable[[optuna.trial.Trial], float],
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, float]:
        if self.study is None:
            self.create_study()

        assert self.study is not None  # for type checkers

        def tracked_objective(trial: optuna.trial.Trial) -> float:
            value = float(objective_func(trial))
            self.tracker.log_epoch(
                epoch=trial.number + 1,
                metrics={**trial.params, "objective": value},
            )
            return value

        self.study.optimize(tracked_objective, n_trials=n_trials, timeout=timeout)

        logger.info("Optimization complete. Best value: %.4f", self.study.best_value)
        self.tracker.finalize(
            {
                "best_params": self.study.best_params,
                "best_value": float(self.study.best_value),
                "trial_count": len(self.study.trials),
            }
        )
        return self.study.best_params

    def get_best_params(self) -> Dict[str, float]:
        if self.study is None:
            raise RuntimeError("Study not initialized")
        return self.study.best_params

    def get_best_value(self) -> float:
        if self.study is None:
            raise RuntimeError("Study not initialized")
        return float(self.study.best_value)


def tune_liveness_model_params(config_path: Optional[str] = None) -> Dict[str, float]:
    """Example Optuna study for the liveness detector."""

    tuner = HyperparameterTuner(
        study_name="liveness_tuning",
        config_path=config_path,
        experiment_name="liveness_hpo",
    )

    def objective(trial: optuna.trial.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)

        return simulate_liveness_training(lr, batch_size, dropout)

    return tuner.optimize(objective, n_trials=30)


def simulate_liveness_training(lr: float, batch_size: int, dropout: float) -> float:
    """Deterministic surrogate for liveness model accuracy.

    This avoids running full training loops during HPO while still providing a
    reproducible surface for Optuna to explore.
    """

    lr_term = math.exp(-abs(math.log10(lr) - 3))
    batch_term = 1.0 - abs(batch_size - 32) / 128
    dropout_term = 1.0 - abs(dropout - 0.3)

    score = 0.75 + 0.1 * lr_term + 0.1 * batch_term + 0.05 * dropout_term
    return max(0.0, min(0.99, score))