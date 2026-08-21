from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable
from typing import Any

import numpy as np

from .collection import option_budget_cost, option_execution_cost
from .config import ExperimentConfig
from .controllers import RecoveryThenNominalController, ScriptedPickCubeController
from .domain import InterventionOption
from .environment import ManiSkillPickCube
from .features import flatten_observation
from .perturbations import PerturbationSchedule, sample_perturbation
from .policies import InterventionPolicy


@dataclass(frozen=True)
class EpisodeMetrics:
    policy: str
    episode_id: int
    success: bool
    steps: int
    interventions: int
    decisions: int
    execution_cost: float
    final_budget: int
    perturbation: str


@dataclass(frozen=True)
class DecisionTrace:
    policy: str
    episode_id: int
    decision: int
    step: int
    controller_phase: str
    budget_before: int
    chosen_option: str
    perturbation: str
    perturbation_started: bool
    continue_success: float | None
    continue_cost: float | None
    recovery_success: float | None
    recovery_cost: float | None
    reset_success: float | None
    reset_cost: float | None
    success_after_window: bool
    steps_after_window: int


def _evaluation_perturbation(
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> PerturbationSchedule | None:
    if rng.random() >= config.collection.perturbation_probability:
        return None
    low = max(1, config.collection.checkpoint_start // 2)
    high = max(low + 1, config.environment.max_episode_steps // 2)
    return sample_perturbation(
        config.collection.perturbations,
        start_step=int(rng.integers(low, high)),
        rng=rng,
    )


def _run_episode(
    env: ManiSkillPickCube,
    config: ExperimentConfig,
    policy_name: str,
    policy: InterventionPolicy,
    episode_id: int,
    traces: list[DecisionTrace] | None = None,
) -> EpisodeMetrics:
    seed = config.environment.seed + 10_000 + episode_id
    rng = np.random.default_rng(seed)
    observation = env.reset(seed=seed, episode_id=episode_id)
    if env.initial_checkpoint is None:  # pragma: no cover - reset guarantees this
        raise RuntimeError("Missing initial checkpoint.")
    controller: Any = ScriptedPickCubeController(config.controller, observation)
    schedule = _evaluation_perturbation(config, rng)
    budget = config.decision.initial_budget
    total_steps = 0
    decisions = 0
    interventions = 0
    execution_cost = 0.0
    success = False

    while total_steps < config.environment.max_episode_steps:
        state, _ = flatten_observation(observation, controller.state_features())
        option, predictions = policy.choose(state, budget)
        decisions += 1
        decision_step = total_steps
        budget_before = budget
        if isinstance(controller, ScriptedPickCubeController):
            controller_phase = controller.phase.name
        elif controller.nominal is None:
            controller_phase = "RECOVERY_RETREAT"
        else:
            controller_phase = f"RECOVERY_{controller.nominal.phase.name}"
        if option is InterventionOption.RECOVERY:
            controller = RecoveryThenNominalController(config.controller, observation)
            interventions += 1
        elif option is InterventionOption.RESET:
            observation = env.reset(seed=seed, episode_id=episode_id)
            controller = ScriptedPickCubeController(config.controller, observation)
            interventions += 1
        budget -= option_budget_cost(option, config)
        execution_cost += option_execution_cost(option, config)

        macro_steps = min(
            config.decision.decision_interval,
            config.environment.max_episode_steps - total_steps,
        )
        for _ in range(macro_steps):
            action = controller.act(observation)
            if schedule is not None:
                action = schedule.before_action(env, action, total_steps, rng)
            observation, _, terminated, truncated, info = env.step(action)
            total_steps += 1
            execution_cost += config.costs.step
            success = env.is_success(info)
            if success or terminated or truncated:
                break
        if success or terminated or truncated:
            finished = True
        else:
            finished = False

        if traces is not None:
            def prediction_value(
                selected: InterventionOption,
                field: str,
            ) -> float | None:
                prediction = predictions.get(selected)
                if prediction is None:
                    return None
                return float(getattr(prediction, field))

            traces.append(
                DecisionTrace(
                    policy=policy_name,
                    episode_id=episode_id,
                    decision=decisions,
                    step=decision_step,
                    controller_phase=controller_phase,
                    budget_before=budget_before,
                    chosen_option=option.name,
                    perturbation=schedule.config.kind if schedule else "none",
                    perturbation_started=(
                        schedule is not None and decision_step >= schedule.start_step
                    ),
                    continue_success=prediction_value(
                        InterventionOption.CONTINUE, "success_probability"
                    ),
                    continue_cost=prediction_value(
                        InterventionOption.CONTINUE, "expected_cost"
                    ),
                    recovery_success=prediction_value(
                        InterventionOption.RECOVERY, "success_probability"
                    ),
                    recovery_cost=prediction_value(
                        InterventionOption.RECOVERY, "expected_cost"
                    ),
                    reset_success=prediction_value(
                        InterventionOption.RESET, "success_probability"
                    ),
                    reset_cost=prediction_value(
                        InterventionOption.RESET, "expected_cost"
                    ),
                    success_after_window=success,
                    steps_after_window=total_steps,
                )
            )
        if finished:
            break

    return EpisodeMetrics(
        policy=policy_name,
        episode_id=episode_id,
        success=success,
        steps=total_steps,
        interventions=interventions,
        decisions=decisions,
        execution_cost=execution_cost,
        final_budget=budget,
        perturbation=schedule.config.kind if schedule else "none",
    )


def evaluate_policies(
    config: ExperimentConfig,
    policies: dict[str, InterventionPolicy],
    progress: Callable[[str], None] | None = None,
    traces: list[DecisionTrace] | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    env = ManiSkillPickCube(config.environment)
    rows: list[EpisodeMetrics] = []
    try:
        for policy_name, policy in policies.items():
            for episode_id in range(config.decision.evaluation_episodes):
                rows.append(
                        _run_episode(
                            env,
                            config,
                            policy_name,
                            policy,
                            episode_id,
                            traces,
                        )
                )
            if progress is not None:
                successes = sum(
                    row.success for row in rows if row.policy == policy_name
                )
                progress(
                    f"evaluation {policy_name}: {successes}/"
                    f"{config.decision.evaluation_episodes} successes"
                )
    finally:
        env.close()

    def aggregate(selected: list[EpisodeMetrics]) -> dict[str, float]:
        intervention_fraction = sum(row.interventions for row in selected) / max(
            1, sum(row.decisions for row in selected)
        )
        return {
            "episodes": float(len(selected)),
            "success_rate": float(np.mean([row.success for row in selected])),
            "mean_steps": float(np.mean([row.steps for row in selected])),
            "mean_interventions": float(
                np.mean([row.interventions for row in selected])
            ),
            "intervention_frequency": float(intervention_fraction),
            "mean_execution_cost": float(
                np.mean([row.execution_cost for row in selected])
            ),
        }

    summaries: dict[str, dict[str, float]] = {}
    robustness: dict[str, dict[str, dict[str, float]]] = {}
    for policy_name in policies:
        selected = [row for row in rows if row.policy == policy_name]
        summaries[policy_name] = aggregate(selected)
        robustness[policy_name] = {
            perturbation: aggregate(
                [row for row in selected if row.perturbation == perturbation]
            )
            for perturbation in sorted({row.perturbation for row in selected})
        }

    paired: dict[str, dict[str, dict[str, float]]] = {}
    if "recoverability" in policies:
        target = sorted(
            (row for row in rows if row.policy == "recoverability"),
            key=lambda row: row.episode_id,
        )
        rng = np.random.default_rng(config.environment.seed + 50_000)
        for baseline_name in policies:
            if baseline_name == "recoverability":
                continue
            baseline = sorted(
                (row for row in rows if row.policy == baseline_name),
                key=lambda row: row.episode_id,
            )
            if [row.episode_id for row in target] != [row.episode_id for row in baseline]:
                raise RuntimeError("Policy evaluation episodes are not paired.")
            success_delta = np.asarray(
                [float(a.success) - float(b.success) for a, b in zip(target, baseline, strict=True)]
            )
            cost_delta = np.asarray(
                [a.execution_cost - b.execution_cost for a, b in zip(target, baseline, strict=True)]
            )

            def bootstrap(values: np.ndarray) -> dict[str, float]:
                sample_indices = rng.integers(0, len(values), size=(2000, len(values)))
                estimates = values[sample_indices].mean(axis=1)
                return {
                    "mean": float(values.mean()),
                    "ci95_low": float(np.quantile(estimates, 0.025)),
                    "ci95_high": float(np.quantile(estimates, 0.975)),
                }

            paired[baseline_name] = {
                "success_rate_delta": bootstrap(success_delta),
                "execution_cost_delta": bootstrap(cost_delta),
            }
    return rows, {
        "overall": summaries,
        "by_perturbation": robustness,
        "paired_bootstrap_vs_baselines": paired,
    }


def rows_as_dicts(rows: list[EpisodeMetrics]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def traces_as_dicts(rows: list[DecisionTrace]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def failure_case_report(
    episodes: list[EpisodeMetrics],
    traces: list[DecisionTrace],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.success:
            continue
        selected = [
            row
            for row in traces
            if row.policy == episode.policy and row.episode_id == episode.episode_id
        ]
        continue_predictions = [
            row.continue_success
            for row in selected
            if row.continue_success is not None
        ]
        interventions = [
            row for row in selected if row.chosen_option != InterventionOption.CONTINUE.name
        ]
        failures.append(
            {
                **asdict(episode),
                "first_intervention_step": (
                    interventions[0].step if interventions else None
                ),
                "chosen_options": [row.chosen_option for row in selected],
                "controller_phases": [row.controller_phase for row in selected],
                "minimum_predicted_continue_success": (
                    min(continue_predictions) if continue_predictions else None
                ),
            }
        )
    return failures
