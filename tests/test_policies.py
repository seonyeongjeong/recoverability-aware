import unittest

import numpy as np

from recoverability.config import CostConfig
from recoverability.domain import InterventionOption, Prediction
from recoverability.policies import RecoverabilityPolicy, feasible_options


class FakeEstimator:
    def predict_options(self, state, budget, options):
        del state, budget
        values = {
            InterventionOption.CONTINUE: Prediction(0.40, 10.0),
            InterventionOption.RECOVERY: Prediction(0.85, 15.0),
            InterventionOption.RESET: Prediction(0.95, 50.0),
        }
        return {option: values[option] for option in options}


class PolicyTests(unittest.TestCase):
    def test_budget_masks_infeasible_options(self) -> None:
        costs = CostConfig(recovery_budget=1, reset_budget=2)
        self.assertEqual(feasible_options(0, costs), (InterventionOption.CONTINUE,))
        self.assertNotIn(InterventionOption.RESET, feasible_options(1, costs))

    def test_joint_success_cost_utility(self) -> None:
        policy = RecoverabilityPolicy(FakeEstimator(), CostConfig(), cost_weight=0.01)
        choice, _ = policy.choose(np.zeros(3, dtype=np.float32), budget=2)
        self.assertEqual(choice, InterventionOption.RECOVERY)


if __name__ == "__main__":
    unittest.main()

