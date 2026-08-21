from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .controllers import RecoveryThenNominalController, ScriptedPickCubeController
from .dataset import CounterfactualDataset
from .domain import Checkpoint, InterventionOption, RolloutOutcome
from .environment import ManiSkillPickCube
from .features import flatten_observation, get_extra
from .perturbations import PerturbationSchedule, sample_perturbation


def option_budget_cost(option: InterventionOption, config: ExperimentConfig) -> int:
    if option is InterventionOption.RECOVERY:
        return config.costs.recovery_budget
    if option is InterventionOption.RESET:
        return config.costs.reset_budget
    return 0


def option_execution_cost(option: InterventionOption, config: ExperimentConfig) -> float:
    if option is InterventionOption.RECOVERY:
        return config.costs.recovery
    if option is InterventionOption.RESET:
        return config.costs.reset
    return 0.0


class CounterfactualRollout:
    def __init__(self, env: ManiSkillPickCube, config: ExperimentConfig):
        self.env = env
        self.config = config

    def run(
        self,
        checkpoint: Checkpoint,
        option: InterventionOption,
        remaining_budget: int,
        rng: np.random.Generator,
    ) -> RolloutOutcome:
        required_budget = option_budget_cost(option, self.config)
        if required_budget > remaining_budget:
            raise ValueError(f"{option.name} is infeasible with budget {remaining_budget}.")

        if option is InterventionOption.RESET:
            observation = self.env.reset(
                seed=self.config.environment.seed + checkpoint.episode_id,
                episode_id=checkpoint.episode_id,
            )
            controller: Any = ScriptedPickCubeController(
                self.config.controller, observation
            )
            horizon = self.config.collection.rollout_horizon
        else:
            observation = self.env.restore(checkpoint)
            if option is InterventionOption.RECOVERY:
                controller = RecoveryThenNominalController(
                    self.config.controller, observation
                )
            else:
                controller = ScriptedPickCubeController(
                    self.config.controller, observation
                )
                controller.load_state_dict(checkpoint.controller_state)
            remaining_episode_steps = max(
                1, self.config.environment.max_episode_steps - checkpoint.step
            )
            horizon = min(self.config.collection.rollout_horizon, remaining_episode_steps)

        fixed_cost = option_execution_cost(option, self.config)
        available_budget = remaining_budget - required_budget
        if self.env.is_success():
            return RolloutOutcome(True, fixed_cost, 0, option, remaining_budget)

        success = False
        steps = 0
        for _ in range(horizon):
            retry_interval = self.config.collection.fallback_recovery_interval
            grasped = bool(np.asarray(get_extra(observation, "is_grasped")).reshape(-1)[0])
            if (
                retry_interval > 0
                and steps > 0
                and steps % retry_interval == 0
                and not grasped
                and available_budget >= self.config.costs.recovery_budget
            ):
                controller = RecoveryThenNominalController(
                    self.config.controller, observation
                )
                available_budget -= self.config.costs.recovery_budget
                fixed_cost += self.config.costs.recovery
            action = controller.act(observation)
            noise = self.config.collection.rollout_action_noise
            if noise > 0:
                action = action.copy()
                action[:3] += rng.normal(0.0, noise, size=3).astype(np.float32)
            observation, _, terminated, truncated, info = self.env.step(action)
            steps += 1
            success = self.env.is_success(info)
            if success or terminated or truncated:
                break
        cost = fixed_cost + steps * self.config.costs.step
        return RolloutOutcome(success, cost, steps, option, remaining_budget)


@dataclass
class _Rows:
    states: list[np.ndarray] = field(default_factory=list)
    option_ids: list[int] = field(default_factory=list)
    budgets: list[float] = field(default_factory=list)
    successes: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    episode_ids: list[int] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    perturbations: list[str] = field(default_factory=list)


class CounterfactualCollector:
    def __init__(
        self,
        config: ExperimentConfig,
        progress: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.progress = progress

    def _collect_checkpoints(
        self,
        env: ManiSkillPickCube,
        episode_id: int,
        rng: np.random.Generator,
    ) -> tuple[list[Checkpoint], str]:
        observation = env.reset(
            seed=self.config.environment.seed + episode_id,
            episode_id=episode_id,
        )
        controller = ScriptedPickCubeController(self.config.controller, observation)
        collection = self.config.collection
        schedule: PerturbationSchedule | None = None
        if rng.random() < collection.perturbation_probability:
            low = max(1, collection.checkpoint_start // 2)
            high = max(low + 1, self.config.environment.max_episode_steps // 2)
            schedule = sample_perturbation(
                collection.perturbations,
                start_step=int(rng.integers(low, high)),
                rng=rng,
            )
        perturbation_name = schedule.config.kind if schedule else "none"
        checkpoints: list[Checkpoint] = []

        for step in range(self.config.environment.max_episode_steps):
            action = controller.act(observation)
            if schedule is not None:
                action = schedule.before_action(env, action, step, rng)
            observation, _, terminated, truncated, info = env.step(action)
            current_step = step + 1
            periodic = (
                current_step >= collection.checkpoint_start
                and (current_step - collection.checkpoint_start)
                % collection.checkpoint_interval
                == 0
            )
            near_perturbation = schedule is not None and current_step in {
                schedule.start_step,
                schedule.end_step,
            }
            if periodic or near_perturbation:
                checkpoints.append(
                    env.checkpoint(
                        episode_id,
                        controller_state=controller.state_dict(),
                    )
                )
            if len(checkpoints) >= collection.max_checkpoints_per_episode:
                break
            if env.is_success(info) or terminated or truncated:
                break
        return checkpoints, perturbation_name

    def collect(self) -> CounterfactualDataset:
        rows = _Rows()
        feature_names: tuple[str, ...] | None = None
        rng = np.random.default_rng(self.config.collection.episodes + self.config.environment.seed)
        env = ManiSkillPickCube(self.config.environment)
        rollout = CounterfactualRollout(env, self.config)
        try:
            if env.action_shape != (4,):
                raise RuntimeError(
                    "The scripted controller expects a four-dimensional "
                    "pd_ee_delta_pos action (XYZ + gripper)."
                )
            for episode_id in range(self.config.collection.episodes):
                samples_before = len(rows.successes)
                checkpoints, perturbation_name = self._collect_checkpoints(
                    env, episode_id, rng
                )
                if env.initial_checkpoint is None:  # pragma: no cover - reset guarantees this
                    raise RuntimeError("Missing initial checkpoint.")
                for checkpoint in checkpoints:
                    checkpoint_controller = ScriptedPickCubeController(
                        self.config.controller,
                        checkpoint.observation,
                    )
                    checkpoint_controller.load_state_dict(checkpoint.controller_state)
                    state, names = flatten_observation(
                        checkpoint.observation,
                        checkpoint_controller.state_features(),
                    )
                    if feature_names is None:
                        feature_names = names
                    elif feature_names != names:
                        raise RuntimeError("Observation feature schema changed between episodes.")
                    for budget in self.config.collection.remaining_budgets:
                        for option in InterventionOption:
                            if option_budget_cost(option, self.config) > budget:
                                continue
                            for _ in range(self.config.collection.counterfactual_repeats):
                                outcome = rollout.run(
                                    checkpoint,
                                    option,
                                    budget,
                                    rng,
                                )
                                rows.states.append(state)
                                rows.option_ids.append(int(option))
                                rows.budgets.append(float(budget))
                                rows.successes.append(float(outcome.success))
                                rows.costs.append(outcome.execution_cost)
                                rows.episode_ids.append(episode_id)
                                rows.steps.append(checkpoint.step)
                                rows.checkpoint_ids.append(checkpoint.checkpoint_id)
                                rows.perturbations.append(perturbation_name)
                if self.progress is not None:
                    failures = sum(success < 0.5 for success in rows.successes)
                    self.progress(
                        "collection episode "
                        f"{episode_id + 1}/{self.config.collection.episodes}: "
                        f"+{len(rows.successes) - samples_before} samples, "
                        f"{failures} failures total"
                    )
        finally:
            env.close()

        if not rows.states or feature_names is None:
            raise RuntimeError("No counterfactual samples were collected.")
        return CounterfactualDataset(
            states=np.stack(rows.states).astype(np.float32),
            option_ids=np.asarray(rows.option_ids, dtype=np.int64),
            remaining_budgets=np.asarray(rows.budgets, dtype=np.float32),
            successes=np.asarray(rows.successes, dtype=np.float32),
            costs=np.asarray(rows.costs, dtype=np.float32),
            episode_ids=np.asarray(rows.episode_ids, dtype=np.int64),
            steps=np.asarray(rows.steps, dtype=np.int64),
            checkpoint_ids=np.asarray(rows.checkpoint_ids, dtype=str),
            perturbations=np.asarray(rows.perturbations, dtype=str),
            feature_names=feature_names,
        )
