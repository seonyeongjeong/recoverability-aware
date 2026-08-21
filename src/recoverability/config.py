from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class EnvironmentConfig:
    env_id: str = "PickCube-v1"
    obs_mode: str = "state_dict"
    control_mode: str = "pd_ee_delta_pos"
    sim_backend: str = "physx_cpu"
    max_episode_steps: int = 100
    seed: int = 7


@dataclass(frozen=True)
class ControllerConfig:
    max_translation_action: float = 0.65
    position_action_scale_m: float = 0.10
    position_tolerance: float = 0.018
    hover_height: float = 0.080
    grasp_offset: float = 0.005
    close_steps: int = 6
    grasp_confirmation_steps: int = 3
    recovery_retreat_steps: int = 5


@dataclass(frozen=True)
class PerturbationConfig:
    kind: str
    weight: float = 1.0
    magnitude: float = 0.25
    duration: int = 5
    start_step_min: int | None = None
    start_step_max: int | None = None


@dataclass(frozen=True)
class CollectionConfig:
    episodes: int = 40
    checkpoint_start: int = 8
    checkpoint_interval: int = 8
    max_checkpoints_per_episode: int = 8
    counterfactual_repeats: int = 2
    rollout_action_noise: float = 0.02
    rollout_horizon: int = 100
    fallback_recovery_interval: int = 0
    remaining_budgets: tuple[int, ...] = (0, 1, 2)
    perturbation_probability: float = 0.75
    perturbations: tuple[PerturbationConfig, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CostConfig:
    step: float = 1.0
    recovery: float = 8.0
    reset: float = 20.0
    recovery_budget: int = 1
    reset_budget: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    batch_size: int = 256
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cost_loss_weight: float = 0.25
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 7


@dataclass(frozen=True)
class DecisionConfig:
    cost_weight: float = 0.015
    decision_interval: int = 5
    risk_threshold: float = 0.45
    evaluation_episodes: int = 30
    initial_budget: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)


def _reject_unknown(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown key(s) in [{section}]: {names}")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    allowed_sections = {
        "environment", "controller", "collection", "costs", "training", "decision"
    }
    _reject_unknown("root", raw, allowed_sections)

    env_values = raw.get("environment", {})
    controller_values = raw.get("controller", {})
    collection_values = dict(raw.get("collection", {}))
    cost_values = raw.get("costs", {})
    training_values = dict(raw.get("training", {}))
    decision_values = raw.get("decision", {})

    _reject_unknown("environment", env_values, set(EnvironmentConfig.__dataclass_fields__))
    _reject_unknown("controller", controller_values, set(ControllerConfig.__dataclass_fields__))
    _reject_unknown("collection", collection_values, set(CollectionConfig.__dataclass_fields__))
    _reject_unknown("costs", cost_values, set(CostConfig.__dataclass_fields__))
    _reject_unknown("training", training_values, set(TrainingConfig.__dataclass_fields__))
    _reject_unknown("decision", decision_values, set(DecisionConfig.__dataclass_fields__))

    perturbation_values = collection_values.pop("perturbations", [])
    perturbations = tuple(PerturbationConfig(**value) for value in perturbation_values)
    if "remaining_budgets" in collection_values:
        collection_values["remaining_budgets"] = tuple(collection_values["remaining_budgets"])
    collection_values["perturbations"] = perturbations
    if "hidden_sizes" in training_values:
        training_values["hidden_sizes"] = tuple(training_values["hidden_sizes"])

    config = ExperimentConfig(
        environment=EnvironmentConfig(**env_values),
        controller=ControllerConfig(**controller_values),
        collection=CollectionConfig(**collection_values),
        costs=CostConfig(**cost_values),
        training=TrainingConfig(**training_values),
        decision=DecisionConfig(**decision_values),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.environment.obs_mode != "state_dict":
        raise ValueError("This implementation requires privileged obs_mode='state_dict'.")
    if config.environment.control_mode != "pd_ee_delta_pos":
        raise ValueError("The scripted PickCube controller requires control_mode='pd_ee_delta_pos'.")
    if not config.collection.remaining_budgets:
        raise ValueError("At least one remaining budget is required.")
    if min(config.collection.remaining_budgets) < 0:
        raise ValueError("Remaining budgets cannot be negative.")
    if not 0.0 <= config.collection.perturbation_probability <= 1.0:
        raise ValueError("perturbation_probability must be in [0, 1].")
    split_fraction = config.training.validation_fraction + config.training.test_fraction
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("Validation and test fractions must sum to a value in (0, 1).")
    if any(item.weight <= 0 for item in config.collection.perturbations):
        raise ValueError("Perturbation weights must be positive.")
    if config.collection.fallback_recovery_interval < 0:
        raise ValueError("fallback_recovery_interval cannot be negative.")
    for item in config.collection.perturbations:
        if item.duration <= 0:
            raise ValueError("Perturbation durations must be positive.")
        if (item.start_step_min is None) != (item.start_step_max is None):
            raise ValueError(
                "Perturbation start_step_min and start_step_max must be set together."
            )
        if item.start_step_min is not None:
            if item.start_step_min < 0 or item.start_step_max < item.start_step_min:
                raise ValueError("Invalid perturbation start-step range.")
