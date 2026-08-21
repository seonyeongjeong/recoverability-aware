from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import CostConfig
from .domain import InterventionOption, Prediction


class Predictor(Protocol):
    def predict_options(
        self,
        state: np.ndarray,
        budget: int,
        options: tuple[InterventionOption, ...],
    ) -> dict[InterventionOption, Prediction]: ...


def feasible_options(budget: int, costs: CostConfig) -> tuple[InterventionOption, ...]:
    options = [InterventionOption.CONTINUE]
    if budget >= costs.recovery_budget:
        options.append(InterventionOption.RECOVERY)
    if budget >= costs.reset_budget:
        options.append(InterventionOption.RESET)
    return tuple(options)


class InterventionPolicy(Protocol):
    def choose(
        self,
        state: np.ndarray,
        budget: int,
    ) -> tuple[InterventionOption, dict[InterventionOption, Prediction]]: ...


@dataclass
class RecoverabilityPolicy:
    estimator: Predictor
    costs: CostConfig
    cost_weight: float

    def choose(
        self, state: np.ndarray, budget: int
    ) -> tuple[InterventionOption, dict[InterventionOption, Prediction]]:
        options = feasible_options(budget, self.costs)
        predictions = self.estimator.predict_options(state, budget, options)
        utility = {
            option: prediction.success_probability
            - self.cost_weight * prediction.expected_cost
            for option, prediction in predictions.items()
        }
        choice = max(options, key=lambda option: (utility[option], -int(option)))
        return choice, predictions


@dataclass
class ThresholdRecoveryPolicy:
    estimator: Predictor
    costs: CostConfig
    threshold: float

    def choose(
        self, state: np.ndarray, budget: int
    ) -> tuple[InterventionOption, dict[InterventionOption, Prediction]]:
        options = feasible_options(budget, self.costs)
        predictions = self.estimator.predict_options(state, budget, options)
        continue_risk = predictions[InterventionOption.CONTINUE].success_probability
        if continue_risk < self.threshold and InterventionOption.RECOVERY in options:
            return InterventionOption.RECOVERY, predictions
        return InterventionOption.CONTINUE, predictions


@dataclass
class NoInterventionPolicy:
    estimator: Predictor
    costs: CostConfig

    def choose(
        self, state: np.ndarray, budget: int
    ) -> tuple[InterventionOption, dict[InterventionOption, Prediction]]:
        options = feasible_options(budget, self.costs)
        predictions = self.estimator.predict_options(state, budget, options)
        return InterventionOption.CONTINUE, predictions

