from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .collection import CounterfactualCollector
from .config import ExperimentConfig
from .dataset import CounterfactualDataset
from .domain import InterventionOption
from .evaluation import (
    DecisionTrace,
    EpisodeMetrics,
    evaluate_policies,
    failure_case_report,
    rows_as_dicts,
    traces_as_dicts,
)
from .model import (
    FeatureSubsetEstimator,
    RecoverabilityEstimator,
    train_estimator,
)
from .policies import (
    NoInterventionPolicy,
    RecoverabilityPolicy,
    ThresholdRecoveryPolicy,
)


Progress = Callable[[str], None]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def tune_cost_head(
    dataset: CounterfactualDataset,
    config: ExperimentConfig,
    weights: Iterable[float],
    output_dir: Path,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Tune loss weight using validation metrics only; test metrics are reporting-only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = tuple(float(weight) for weight in weights)
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError(
            "Cost-head tuning weights must be positive; use the no_cost ablation "
            "to evaluate an untrained cost head."
        )
    candidates: list[dict[str, Any]] = []
    validation_mean_cost = float(
        dataset.split_by_episode(
            config.training.validation_fraction,
            config.training.test_fraction,
            config.training.seed,
        )[1].costs.mean()
    )
    for weight in weights:
        if progress is not None:
            progress(f"cost-head tuning: training cost_loss_weight={weight:g}")
        training = replace(config.training, cost_loss_weight=float(weight))
        model_path = output_dir / f"model_cost_{_tag(float(weight))}.pt"
        metrics = train_estimator(dataset, training, model_path, progress=progress)
        validation = metrics["validation_calibrated"]
        # Scale MAE by the validation cost magnitude so NLL and cost error both
        # influence selection without consulting test outcomes.
        selection_score = float(validation["nll"]) + 0.25 * float(
            validation["cost_mae"]
        ) / max(validation_mean_cost, 1.0)
        candidates.append(
            {
                "cost_loss_weight": float(weight),
                "selection_score": selection_score,
                "model": str(model_path),
                "metrics": metrics,
            }
        )
    best = min(candidates, key=lambda item: item["selection_score"])
    best_path = output_dir / "best_model.pt"
    shutil.copyfile(best["model"], best_path)
    result = {
        "selection_rule": "validation_nll + 0.25 * validation_cost_mae / validation_mean_cost",
        "validation_mean_cost": validation_mean_cost,
        "best_cost_loss_weight": best["cost_loss_weight"],
        "best_model": str(best_path),
        "candidates": candidates,
    }
    _write_json(output_dir / "results.json", result)
    return result


def _validation_decision_score(
    dataset: CounterfactualDataset,
    estimator: Any,
    mode: str,
    candidate: float,
    reference_cost_weight: float,
) -> dict[str, float]:
    grouped: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, (checkpoint, budget) in enumerate(
        zip(dataset.checkpoint_ids, dataset.remaining_budgets, strict=True)
    ):
        grouped[(str(checkpoint), float(budget))].append(index)

    successes: list[float] = []
    costs: list[float] = []
    interventions: list[float] = []
    for indices in grouped.values():
        selected = np.asarray(indices, dtype=np.int64)
        state = dataset.states[selected[0]]
        budget = int(dataset.remaining_budgets[selected[0]])
        options = tuple(
            InterventionOption(int(value))
            for value in sorted(np.unique(dataset.option_ids[selected]).tolist())
        )
        predictions = estimator.predict_options(state, budget, options)
        if mode == "cost_weight":
            choice = max(
                options,
                key=lambda option: (
                    predictions[option].success_probability
                    - candidate * predictions[option].expected_cost,
                    -int(option),
                ),
            )
        elif mode == "threshold":
            if (
                predictions[InterventionOption.CONTINUE].success_probability < candidate
                and InterventionOption.RECOVERY in options
            ):
                choice = InterventionOption.RECOVERY
            else:
                choice = InterventionOption.CONTINUE
        else:  # pragma: no cover - internal callers use the two modes above
            raise ValueError(f"Unknown decision tuning mode: {mode}")
        outcome_indices = selected[dataset.option_ids[selected] == int(choice)]
        successes.append(float(dataset.successes[outcome_indices].mean()))
        costs.append(float(dataset.costs[outcome_indices].mean()))
        interventions.append(float(choice is not InterventionOption.CONTINUE))

    mean_success = float(np.mean(successes))
    mean_cost = float(np.mean(costs))
    return {
        "candidate": float(candidate),
        "success_rate": mean_success,
        "mean_cost": mean_cost,
        "intervention_rate": float(np.mean(interventions)),
        "selection_utility": mean_success - reference_cost_weight * mean_cost,
    }


def tune_decision_rules(
    dataset: CounterfactualDataset,
    estimator: Any,
    config: ExperimentConfig,
    cost_weights: Iterable[float],
    thresholds: Iterable[float],
    output_dir: Path,
) -> dict[str, Any]:
    """Choose policy hyperparameters on episode-disjoint validation data only."""
    validation = dataset.split_by_episode(
        config.training.validation_fraction,
        config.training.test_fraction,
        config.training.seed,
    )[1]
    cost_results = [
        _validation_decision_score(
            validation,
            estimator,
            "cost_weight",
            float(candidate),
            config.decision.cost_weight,
        )
        for candidate in cost_weights
    ]
    threshold_results = [
        _validation_decision_score(
            validation,
            estimator,
            "threshold",
            float(candidate),
            config.decision.cost_weight,
        )
        for candidate in thresholds
    ]
    best_cost = max(
        cost_results,
        key=lambda item: (item["selection_utility"], item["success_rate"]),
    )
    best_threshold = max(
        threshold_results,
        key=lambda item: (item["selection_utility"], item["success_rate"]),
    )
    result = {
        "selection_rule": (
            "maximize validation success_rate - reference_cost_weight * mean_cost"
        ),
        "reference_cost_weight": config.decision.cost_weight,
        "best_cost_weight": best_cost["candidate"],
        "best_threshold": best_threshold["candidate"],
        "cost_weight_candidates": cost_results,
        "threshold_candidates": threshold_results,
    }
    _write_json(output_dir / "decision_tuning.json", result)
    return result


def _policies(
    estimator: Any,
    config: ExperimentConfig,
    cost_weight: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "recoverability": RecoverabilityPolicy(estimator, config.costs, cost_weight),
        "success_only_ablation": RecoverabilityPolicy(
            estimator, config.costs, cost_weight=0.0
        ),
        "no_intervention": NoInterventionPolicy(estimator, config.costs),
        "threshold_recovery": ThresholdRecoveryPolicy(
            estimator, config.costs, threshold
        ),
    }


def evaluate_and_write(
    config: ExperimentConfig,
    policies: dict[str, Any],
    output_dir: Path,
    progress: Progress | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    traces: list[DecisionTrace] = []
    rows, summary = evaluate_policies(
        config,
        policies,
        progress=progress,
        traces=traces,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "episodes.csv", rows_as_dicts(rows))
    _write_csv(output_dir / "decisions.csv", traces_as_dicts(traces))
    _write_json(output_dir / "failure_cases.json", failure_case_report(rows, traces))
    return rows, summary


def run_seed(
    base_config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    cost_loss_weight: float,
    decision_cost_weight: float,
    threshold: float,
    progress: Progress | None = None,
    existing_dataset: Path | None = None,
) -> dict[str, Any]:
    seed_config = replace(
        base_config,
        environment=replace(base_config.environment, seed=seed),
        training=replace(
            base_config.training,
            seed=seed,
            cost_loss_weight=cost_loss_weight,
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "counterfactual.npz"
    if existing_dataset is None:
        if progress is not None:
            progress(f"seed {seed}: collecting counterfactual dataset")
        dataset = CounterfactualCollector(seed_config, progress=progress).collect()
        dataset.save(dataset_path)
    else:
        shutil.copyfile(existing_dataset, dataset_path)
        dataset = CounterfactualDataset.load(dataset_path)
        if progress is not None:
            progress(f"seed {seed}: reusing existing core dataset")

    model_path = output_dir / "recoverability.pt"
    if progress is not None:
        progress(f"seed {seed}: training estimator")
    metrics = train_estimator(
        dataset,
        seed_config.training,
        model_path,
        progress=progress,
    )
    _write_json(output_dir / "training_metrics.json", metrics)
    estimator = RecoverabilityEstimator.load(model_path)
    if progress is not None:
        progress(f"seed {seed}: evaluating policies")
    rows, summary = evaluate_and_write(
        seed_config,
        _policies(
            estimator,
            seed_config,
            decision_cost_weight,
            threshold,
        ),
        output_dir / "evaluation",
        progress,
    )
    result = {
        "seed": seed,
        "config": asdict(seed_config),
        "dataset": dataset.quality_report(),
        "training": metrics,
        "evaluation": summary,
    }
    _write_json(output_dir / "run.json", result)
    return {**result, "episode_rows": rows}


def _hierarchical_bootstrap(
    groups: dict[int, list[float]],
    rng: np.random.Generator,
    samples: int = 5_000,
) -> dict[str, float]:
    seeds = np.asarray(sorted(groups), dtype=np.int64)
    observed = np.concatenate(
        [np.asarray(groups[int(seed)], dtype=np.float64) for seed in seeds]
    )
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values: list[np.ndarray] = []
        for selected_seed in selected_seeds:
            source = np.asarray(groups[int(selected_seed)], dtype=np.float64)
            values.append(rng.choice(source, size=len(source), replace=True))
        estimates[index] = float(np.concatenate(values).mean())
    return {
        "mean": float(observed.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def aggregate_seed_runs(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(91_337)
    policy_names = sorted(
        {row.policy for run in seed_runs for row in run["episode_rows"]}
    )
    overall: dict[str, Any] = {}
    for policy in policy_names:
        success_groups: dict[int, list[float]] = {}
        cost_groups: dict[int, list[float]] = {}
        intervention_groups: dict[int, list[float]] = {}
        for run in seed_runs:
            selected = [row for row in run["episode_rows"] if row.policy == policy]
            seed = int(run["seed"])
            success_groups[seed] = [float(row.success) for row in selected]
            cost_groups[seed] = [row.execution_cost for row in selected]
            intervention_groups[seed] = [float(row.interventions) for row in selected]
        overall[policy] = {
            "success_rate": _hierarchical_bootstrap(success_groups, rng),
            "execution_cost": _hierarchical_bootstrap(cost_groups, rng),
            "interventions": _hierarchical_bootstrap(intervention_groups, rng),
        }

    robustness: dict[str, Any] = {}
    for policy in policy_names:
        perturbations = sorted(
            {
                row.perturbation
                for run in seed_runs
                for row in run["episode_rows"]
                if row.policy == policy
            }
        )
        robustness[policy] = {}
        for perturbation in perturbations:
            success_groups: dict[int, list[float]] = {}
            cost_groups: dict[int, list[float]] = {}
            for run in seed_runs:
                selected = [
                    row
                    for row in run["episode_rows"]
                    if row.policy == policy and row.perturbation == perturbation
                ]
                if not selected:
                    continue
                seed = int(run["seed"])
                success_groups[seed] = [float(row.success) for row in selected]
                cost_groups[seed] = [row.execution_cost for row in selected]
            robustness[policy][perturbation] = {
                "success_rate": _hierarchical_bootstrap(success_groups, rng),
                "execution_cost": _hierarchical_bootstrap(cost_groups, rng),
            }

    paired: dict[str, Any] = {}
    target_name = "recoverability"
    if target_name in policy_names:
        for baseline in policy_names:
            if baseline == target_name:
                continue
            success_groups: dict[int, list[float]] = {}
            cost_groups: dict[int, list[float]] = {}
            for run in seed_runs:
                target = sorted(
                    (row for row in run["episode_rows"] if row.policy == target_name),
                    key=lambda row: row.episode_id,
                )
                comparison = sorted(
                    (row for row in run["episode_rows"] if row.policy == baseline),
                    key=lambda row: row.episode_id,
                )
                seed = int(run["seed"])
                success_groups[seed] = [
                    float(a.success) - float(b.success)
                    for a, b in zip(target, comparison, strict=True)
                ]
                cost_groups[seed] = [
                    a.execution_cost - b.execution_cost
                    for a, b in zip(target, comparison, strict=True)
                ]
            paired[baseline] = {
                "success_rate_delta": _hierarchical_bootstrap(success_groups, rng),
                "execution_cost_delta": _hierarchical_bootstrap(cost_groups, rng),
            }
    return {
        "seeds": [int(run["seed"]) for run in seed_runs],
        "overall": overall,
        "by_perturbation": robustness,
        "paired_vs_recoverability": paired,
    }


def _ablation_dataset(
    dataset: CounterfactualDataset,
    name: str,
) -> CounterfactualDataset:
    if name == "no_controller_phase":
        names = tuple(
            feature for feature in dataset.feature_names if not feature.startswith("controller.")
        )
        return dataset.select_features(names)
    if name.startswith("without_"):
        return dataset.without_perturbation(name.removeprefix("without_"))
    return dataset


def run_ablation(
    name: str,
    dataset: CounterfactualDataset,
    config: ExperimentConfig,
    output_dir: Path,
    cost_loss_weight: float,
    decision_cost_weight: float,
    progress: Progress | None = None,
) -> dict[str, Any]:
    selected = _ablation_dataset(dataset, name)
    training = replace(
        config.training,
        cost_loss_weight=(0.0 if name == "no_cost" else cost_loss_weight),
    )
    model_path = output_dir / "recoverability.pt"
    if progress is not None:
        progress(f"ablation {name}: training on {len(selected)} samples")
    metrics = train_estimator(selected, training, model_path, progress=progress)
    _write_json(output_dir / "training_metrics.json", metrics)
    base_estimator = RecoverabilityEstimator.load(model_path)
    if selected.feature_names != dataset.feature_names:
        estimator: Any = FeatureSubsetEstimator(base_estimator, dataset.feature_names)
    else:
        estimator = base_estimator
    policy_cost_weight = 0.0 if name == "no_cost" else decision_cost_weight
    policies = {
        "recoverability": RecoverabilityPolicy(
            estimator,
            config.costs,
            policy_cost_weight,
        ),
        "no_intervention": NoInterventionPolicy(estimator, config.costs),
    }
    rows, summary = evaluate_and_write(
        config,
        policies,
        output_dir / "evaluation",
        progress,
    )
    result = {
        "name": name,
        "dataset": selected.quality_report(),
        "training": metrics,
        "evaluation": summary,
    }
    _write_json(output_dir / "result.json", result)
    return {**result, "episode_rows": rows}


def _prepare_matplotlib(output_dir: Path) -> Any:
    import os

    cache = output_dir / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_multiseed_summary(aggregate: dict[str, Any], output_path: Path) -> None:
    plt = _prepare_matplotlib(output_path.parent)
    policies = list(aggregate["overall"])
    labels = [name.replace("_", "\n") for name in policies]
    successes = [aggregate["overall"][name]["success_rate"] for name in policies]
    costs = [aggregate["overall"][name]["execution_cost"] for name in policies]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(policies))
    success_means = [value["mean"] for value in successes]
    success_error = np.asarray(
        [
            [value["mean"] - value["ci95_low"] for value in successes],
            [value["ci95_high"] - value["mean"] for value in successes],
        ]
    )
    axes[0].bar(x, success_means, color="#3b82f6", alpha=0.85)
    axes[0].errorbar(x, success_means, yerr=success_error, fmt="none", color="black", capsize=4)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Multi-seed policy success (95% CI)")

    cost_means = [value["mean"] for value in costs]
    cost_error = np.asarray(
        [
            [value["mean"] - value["ci95_low"] for value in costs],
            [value["ci95_high"] - value["mean"] for value in costs],
        ]
    )
    axes[1].bar(x, cost_means, color="#f59e0b", alpha=0.85)
    axes[1].errorbar(x, cost_means, yerr=cost_error, fmt="none", color="black", capsize=4)
    axes[1].set_ylabel("Mean execution cost")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Multi-seed execution cost (95% CI)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_robustness(aggregate: dict[str, Any], output_path: Path) -> None:
    plt = _prepare_matplotlib(output_path.parent)
    policies = list(aggregate["by_perturbation"])
    perturbations = sorted(
        {
            kind
            for values in aggregate["by_perturbation"].values()
            for kind in values
        }
    )
    x = np.arange(len(perturbations))
    width = 0.8 / max(1, len(policies))
    figure, axis = plt.subplots(figsize=(11, 5))
    for index, policy in enumerate(policies):
        statistics = [
            aggregate["by_perturbation"][policy].get(kind, {}).get("success_rate")
            for kind in perturbations
        ]
        rates = [item["mean"] if item is not None else 0.0 for item in statistics]
        errors = np.asarray(
            [
                [
                    item["mean"] - item["ci95_low"] if item is not None else 0.0
                    for item in statistics
                ],
                [
                    item["ci95_high"] - item["mean"] if item is not None else 0.0
                    for item in statistics
                ],
            ]
        )
        axis.bar(
            x + (index - (len(policies) - 1) / 2) * width,
            rates,
            width,
            yerr=errors,
            capsize=2,
            label=policy.replace("_", " "),
        )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Success rate")
    axis.set_xticks(x, [kind.replace("_", "\n") for kind in perturbations])
    axis.set_title("Multi-seed robustness by perturbation family (95% CI)")
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_calibration(
    dataset: CounterfactualDataset,
    estimator: RecoverabilityEstimator,
    config: ExperimentConfig,
    output_path: Path,
) -> None:
    plt = _prepare_matplotlib(output_path.parent)
    test = dataset.split_by_episode(
        config.training.validation_fraction,
        config.training.test_fraction,
        config.training.seed,
    )[2]
    probabilities, _ = estimator.predict_many(
        test.states,
        test.option_ids,
        test.remaining_budgets,
    )
    boundaries = np.linspace(0.0, 1.0, 11)
    confidence: list[float] = []
    accuracy: list[float] = []
    counts: list[int] = []
    for index in range(10):
        if index == 9:
            selected = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            selected = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if selected.any():
            confidence.append(float(probabilities[selected].mean()))
            accuracy.append(float(test.successes[selected].mean()))
            counts.append(int(selected.sum()))
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    axis.plot(confidence, accuracy, marker="o", color="#2563eb", label="estimator")
    for x, y, count in zip(confidence, accuracy, counts, strict=True):
        axis.annotate(str(count), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Predicted success probability")
    axis.set_ylabel("Observed success frequency")
    axis.set_title("Test-set reliability diagram (labels show bin count)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_ablation_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    plt = _prepare_matplotlib(output_path.parent)
    names = [result["name"] for result in results]
    success = [
        result["evaluation"]["overall"]["recoverability"]["success_rate"]
        for result in results
    ]
    cost = [
        result["evaluation"]["overall"]["recoverability"]["mean_execution_cost"]
        for result in results
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    axes[0].bar(x, success, color="#10b981")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Success rate")
    axes[0].set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    axes[0].set_title("Ablation success")
    axes[1].bar(x, cost, color="#f97316")
    axes[1].set_ylabel("Mean execution cost")
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    axes[1].set_title("Ablation cost")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _suite_report(
    tuning: dict[str, Any],
    aggregate: dict[str, Any],
    ablations: list[dict[str, Any]],
) -> str:
    recoverability = aggregate["overall"]["recoverability"]
    threshold = aggregate["overall"].get("threshold_recovery")
    lines = [
        "# Multi-seed recoverability research suite",
        "",
        "## Validation-only tuning",
        "",
        f"- Selected cost-head loss weight: {tuning['cost_head']['best_cost_loss_weight']}",
        f"- Selected decision cost weight: {tuning['decision']['best_cost_weight']}",
        f"- Selected threshold baseline value: {tuning['decision']['best_threshold']}",
        "",
        "No test or closed-loop evaluation outcomes were used for these selections.",
        "",
        "## Multi-seed results",
        "",
        f"Seeds: {', '.join(map(str, aggregate['seeds']))}",
        "",
        (
            "Recoverability success: "
            f"{recoverability['success_rate']['mean']:.3f} "
            f"(95% CI {recoverability['success_rate']['ci95_low']:.3f}, "
            f"{recoverability['success_rate']['ci95_high']:.3f})"
        ),
        (
            "Recoverability cost: "
            f"{recoverability['execution_cost']['mean']:.2f} "
            f"(95% CI {recoverability['execution_cost']['ci95_low']:.2f}, "
            f"{recoverability['execution_cost']['ci95_high']:.2f})"
        ),
    ]
    if threshold is not None:
        lines.extend(
            [
                (
                    "Threshold success: "
                    f"{threshold['success_rate']['mean']:.3f} "
                    f"(95% CI {threshold['success_rate']['ci95_low']:.3f}, "
                    f"{threshold['success_rate']['ci95_high']:.3f})"
                ),
                "",
            ]
        )
    paired = aggregate["paired_vs_recoverability"]
    lines.extend(["## Paired comparisons", ""])
    for baseline_name in ("no_intervention", "threshold_recovery"):
        if baseline_name not in paired:
            continue
        success = paired[baseline_name]["success_rate_delta"]
        cost = paired[baseline_name]["execution_cost_delta"]
        lines.append(
            f"- Versus {baseline_name.replace('_', ' ')}: success delta "
            f"{success['mean']:+.3f} (95% CI {success['ci95_low']:+.3f}, "
            f"{success['ci95_high']:+.3f}); cost delta {cost['mean']:+.2f} "
            f"(95% CI {cost['ci95_low']:+.2f}, {cost['ci95_high']:+.2f})"
        )
    lines.extend(
        [
            "",
            (
                "The success intervals include zero, so this run does not establish "
                "that recoverability outperforms either baseline."
            ),
            "",
            "## Ablations",
            "",
            "Ablations use the representative seed only and are diagnostic, not "
            "multi-seed causal estimates.",
            "",
            "| Ablation | Success | Mean cost |",
            "|---|---:|---:|",
        ]
    )
    for result in ablations:
        metrics = result["evaluation"]["overall"]["recoverability"]
        lines.append(
            f"| {result['name']} | {metrics['success_rate']:.3f} | "
            f"{metrics['mean_execution_cost']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Generated evidence",
            "",
            "- `aggregate.json`: hierarchical bootstrap estimates across seeds and episodes",
            "- `seeds/*/evaluation/decisions.csv`: decision-level traces",
            "- `seeds/*/evaluation/failure_cases.json`: failure-case diagnostics",
            "- `ablations/*`: trained ablation models and paired evaluations",
            "- `plots/*`: multi-seed, robustness, calibration, and ablation figures",
            "",
        ]
    )
    return "\n".join(lines)


def run_research_suite(
    config: ExperimentConfig,
    base_dataset_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...],
    cost_loss_weights: tuple[float, ...],
    decision_cost_weights: tuple[float, ...],
    thresholds: tuple[float, ...],
    progress: Progress | None = None,
    run_ablations: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CounterfactualDataset.load(base_dataset_path)
    cost_tuning = tune_cost_head(
        dataset,
        config,
        cost_loss_weights,
        output_dir / "tuning" / "cost_head",
        progress,
    )
    tuning_estimator = RecoverabilityEstimator.load(cost_tuning["best_model"])
    decision_tuning = tune_decision_rules(
        dataset,
        tuning_estimator,
        config,
        decision_cost_weights,
        thresholds,
        output_dir / "tuning",
    )
    tuning = {"cost_head": cost_tuning, "decision": decision_tuning}
    _write_json(output_dir / "tuning" / "summary.json", tuning)

    seed_runs: list[dict[str, Any]] = []
    for seed in seeds:
        existing = (
            base_dataset_path if seed == config.environment.seed else None
        )
        seed_runs.append(
            run_seed(
                config,
                seed,
                output_dir / "seeds" / str(seed),
                float(cost_tuning["best_cost_loss_weight"]),
                float(decision_tuning["best_cost_weight"]),
                float(decision_tuning["best_threshold"]),
                progress,
                existing,
            )
        )
    aggregate = aggregate_seed_runs(seed_runs)
    _write_json(output_dir / "aggregate.json", aggregate)

    ablation_results: list[dict[str, Any]] = []
    if run_ablations:
        ablation_names = ["no_controller_phase", "no_cost"] + [
            f"without_{item.kind}" for item in config.collection.perturbations
        ]
        for name in ablation_names:
            ablation_results.append(
                run_ablation(
                    name,
                    dataset,
                    config,
                    output_dir / "ablations" / name,
                    float(cost_tuning["best_cost_loss_weight"]),
                    float(decision_tuning["best_cost_weight"]),
                    progress,
                )
            )

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plot_multiseed_summary(aggregate, plots / "multi_seed_summary.png")
    representative = seed_runs[0]
    plot_robustness(aggregate, plots / "robustness_by_perturbation.png")
    representative_model = RecoverabilityEstimator.load(
        output_dir / "seeds" / str(representative["seed"]) / "recoverability.pt"
    )
    representative_dataset = CounterfactualDataset.load(
        output_dir / "seeds" / str(representative["seed"]) / "counterfactual.npz"
    )
    representative_config = replace(
        config,
        environment=replace(config.environment, seed=int(representative["seed"])),
        training=replace(config.training, seed=int(representative["seed"])),
    )
    plot_calibration(
        representative_dataset,
        representative_model,
        representative_config,
        plots / "calibration.png",
    )
    if ablation_results:
        plot_ablation_summary(ablation_results, plots / "ablations.png")

    report = _suite_report(tuning, aggregate, ablation_results)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    result = {
        "tuning": tuning,
        "aggregate": aggregate,
        "ablations": [
            {key: value for key, value in item.items() if key != "episode_rows"}
            for item in ablation_results
        ],
        "report": str(output_dir / "report.md"),
    }
    _write_json(output_dir / "suite.json", result)
    return result
