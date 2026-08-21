import unittest

import numpy as np

from recoverability.config import CostConfig, DecisionConfig, ExperimentConfig
from recoverability.domain import InterventionOption, Prediction
from recoverability.evaluation import (
    DecisionTrace,
    EpisodeMetrics,
    failure_case_report,
)
from recoverability.research import aggregate_seed_runs, tune_decision_rules
from test_dataset import make_dataset


class _Estimator:
    def predict_options(self, state, budget, options):
        del state, budget
        values = {
            InterventionOption.CONTINUE: Prediction(0.4, 10.0),
            InterventionOption.RECOVERY: Prediction(0.8, 15.0),
            InterventionOption.RESET: Prediction(0.95, 40.0),
        }
        return {option: values[option] for option in options}


def _row(policy: str, episode: int, success: bool, cost: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        policy=policy,
        episode_id=episode,
        success=success,
        steps=30,
        interventions=0,
        decisions=6,
        execution_cost=cost,
        final_budget=2,
        perturbation="none",
    )


class ResearchTests(unittest.TestCase):
    def test_decision_tuning_uses_validation_counterfactuals(self) -> None:
        dataset = make_dataset()
        dataset.checkpoint_ids = np.asarray(
            [f"cp-{index // 3}" for index in range(len(dataset))]
        )
        config = ExperimentConfig(
            costs=CostConfig(),
            decision=DecisionConfig(cost_weight=0.01),
        )
        # A temporary-like object is sufficient because the function only writes
        # one JSON file after computing the validation sweep.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            result = tune_decision_rules(
                dataset,
                _Estimator(),
                config,
                cost_weights=(0.0, 0.1),
                thresholds=(0.3, 0.5),
                output_dir=Path(directory),
            )
        self.assertIn(result["best_cost_weight"], (0.0, 0.1))
        self.assertIn(result["best_threshold"], (0.3, 0.5))

    def test_multiseed_aggregate_contains_paired_deltas(self) -> None:
        runs = []
        for seed in (1, 2):
            rows = [
                _row("recoverability", 0, True, 30.0),
                _row("recoverability", 1, True, 35.0),
                _row("no_intervention", 0, False, 40.0),
                _row("no_intervention", 1, True, 35.0),
            ]
            runs.append({"seed": seed, "episode_rows": rows})
        aggregate = aggregate_seed_runs(runs)
        self.assertEqual(aggregate["seeds"], [1, 2])
        delta = aggregate["paired_vs_recoverability"]["no_intervention"]
        self.assertAlmostEqual(delta["success_rate_delta"]["mean"], 0.5)
        robustness = aggregate["by_perturbation"]["recoverability"]["none"]
        self.assertAlmostEqual(robustness["success_rate"]["mean"], 1.0)

    def test_failure_report_summarizes_decision_trace(self) -> None:
        episode = _row("recoverability", 4, False, 50.0)
        trace = DecisionTrace(
            policy="recoverability",
            episode_id=4,
            decision=1,
            step=10,
            controller_phase="LIFT",
            budget_before=2,
            chosen_option="RECOVERY",
            perturbation="object_shift",
            perturbation_started=True,
            continue_success=0.2,
            continue_cost=50.0,
            recovery_success=0.7,
            recovery_cost=35.0,
            reset_success=0.9,
            reset_cost=60.0,
            success_after_window=False,
            steps_after_window=15,
        )
        report = failure_case_report([episode], [trace])
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["first_intervention_step"], 10)
        self.assertEqual(report[0]["chosen_options"], ["RECOVERY"])
        self.assertEqual(report[0]["controller_phases"], ["LIFT"])
        self.assertEqual(report[0]["minimum_predicted_continue_success"], 0.2)


if __name__ == "__main__":
    unittest.main()
