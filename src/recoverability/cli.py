from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

from .collection import CounterfactualCollector
from .config import ExperimentConfig, load_config
from .controllers import ScriptedPickCubeController
from .dataset import CounterfactualDataset
from .environment import ManiSkillPickCube
from .features import flatten_observation


def _progress(message: str) -> None:
    print(f"[recoverability] {message}", file=sys.stderr, flush=True)


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _inspect(config: ExperimentConfig) -> dict[str, Any]:
    env = ManiSkillPickCube(config.environment)
    try:
        observation = env.reset(config.environment.seed)
        controller = ScriptedPickCubeController(config.controller, observation)
        state, names = flatten_observation(observation, controller.state_features())
        success = False
        steps = 0
        for _ in range(config.environment.max_episode_steps):
            observation, _, terminated, truncated, info = env.step(
                controller.act(observation)
            )
            steps += 1
            success = env.is_success(info)
            if success or terminated or truncated:
                break
        return {
            "env_id": config.environment.env_id,
            "action_shape": env.action_shape,
            "state_features": len(state),
            "first_features": list(names[:10]),
            "nominal_success": success,
            "nominal_steps": steps,
        }
    finally:
        env.close()


def _collect(config: ExperimentConfig, output: str | Path) -> dict[str, Any]:
    dataset = CounterfactualCollector(config, progress=_progress).collect()
    dataset.save(output)
    return dataset.quality_report()


def _train(
    config: ExperimentConfig,
    dataset_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None,
) -> dict[str, Any]:
    from .model import train_estimator

    dataset = CounterfactualDataset.load(dataset_path)
    metrics = train_estimator(dataset, config.training, model_path, progress=_progress)
    if metrics_path is not None:
        _write_json(metrics_path, metrics)
    return metrics


def _evaluate(
    config: ExperimentConfig,
    model_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from .evaluation import (
        evaluate_policies,
        failure_case_report,
        rows_as_dicts,
        traces_as_dicts,
    )
    from .model import RecoverabilityEstimator
    from .policies import NoInterventionPolicy, RecoverabilityPolicy, ThresholdRecoveryPolicy

    estimator = RecoverabilityEstimator.load(model_path)
    policies = {
        "recoverability": RecoverabilityPolicy(
            estimator, config.costs, config.decision.cost_weight
        ),
        "success_only_ablation": RecoverabilityPolicy(
            estimator, config.costs, cost_weight=0.0
        ),
        "no_intervention": NoInterventionPolicy(estimator, config.costs),
        "threshold_recovery": ThresholdRecoveryPolicy(
            estimator, config.costs, config.decision.risk_threshold
        ),
    }
    traces = []
    rows, summaries = evaluate_policies(
        config,
        policies,
        progress=_progress,
        traces=traces,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summaries)
    dictionaries = rows_as_dicts(rows)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)
    trace_dictionaries = traces_as_dicts(traces)
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_dictionaries[0]))
        writer.writeheader()
        writer.writerows(trace_dictionaries)
    _write_json(output / "failure_cases.json", failure_case_report(rows, traces))
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recoverability",
        description="Recoverability-aware ManiSkill PickCube experiments",
    )
    parser.add_argument(
        "--config", default="configs/pick_cube.toml", help="Experiment TOML file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Check the environment and nominal controller")

    collect = subparsers.add_parser("collect", help="Generate counterfactual rollouts")
    collect.add_argument("--output", default="data/counterfactual.npz")

    train = subparsers.add_parser("train", help="Train and calibrate the estimator")
    train.add_argument("--dataset", default="data/counterfactual.npz")
    train.add_argument("--model", default="models/recoverability.pt")
    train.add_argument("--metrics", default="artifacts/training_metrics.json")

    evaluate = subparsers.add_parser("evaluate", help="Compare closed-loop policies")
    evaluate.add_argument("--model", default="models/recoverability.pt")
    evaluate.add_argument("--output-dir", default="artifacts/evaluation")

    run_all = subparsers.add_parser("run-all", help="Collect, train, and evaluate")
    run_all.add_argument("--dataset", default="data/counterfactual.npz")
    run_all.add_argument("--model", default="models/recoverability.pt")
    run_all.add_argument("--output-dir", default="artifacts/evaluation")

    research = subparsers.add_parser(
        "research-suite",
        help="Run validation tuning, multi-seed evaluation, ablations, and plots",
    )
    research.add_argument("--dataset", default="data/counterfactual.npz")
    research.add_argument("--output-dir", default="artifacts/research_suite")
    research.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    research.add_argument(
        "--cost-loss-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.1, 0.25, 0.5],
    )
    research.add_argument(
        "--decision-cost-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03],
    )
    research.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
    )
    research.add_argument("--episodes", type=int)
    research.add_argument("--evaluation-episodes", type=int)
    research.add_argument("--epochs", type=int)
    research.add_argument("--no-ablations", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "inspect":
        result = _inspect(config)
    elif args.command == "collect":
        result = _collect(config, args.output)
    elif args.command == "train":
        result = _train(config, args.dataset, args.model, args.metrics)
    elif args.command == "evaluate":
        result = _evaluate(config, args.model, args.output_dir)
    elif args.command == "run-all":
        _progress("stage 1/3: collecting counterfactual data")
        collection_summary = _collect(config, args.dataset)
        _progress("stage 2/3: training recoverability estimator")
        training_metrics = _train(
            config,
            args.dataset,
            args.model,
            Path(args.output_dir) / "training_metrics.json",
        )
        _progress("stage 3/3: evaluating closed-loop policies")
        evaluation_summary = _evaluate(config, args.model, args.output_dir)
        result = {
            "collection": collection_summary,
            "training": training_metrics,
            "evaluation": evaluation_summary,
        }
    elif args.command == "research-suite":
        from .research import run_research_suite

        if args.episodes is not None:
            config = replace(
                config,
                collection=replace(config.collection, episodes=args.episodes),
            )
        if args.evaluation_episodes is not None:
            config = replace(
                config,
                decision=replace(
                    config.decision,
                    evaluation_episodes=args.evaluation_episodes,
                ),
            )
        if args.epochs is not None:
            config = replace(
                config,
                training=replace(config.training, epochs=args.epochs),
            )
        result = run_research_suite(
            config,
            Path(args.dataset),
            Path(args.output_dir),
            tuple(args.seeds),
            tuple(args.cost_loss_weights),
            tuple(args.decision_cost_weights),
            tuple(args.thresholds),
            _progress,
            run_ablations=not args.no_ablations,
        )
    else:  # pragma: no cover - argparse enforces commands
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
