import unittest

import numpy as np

from recoverability.config import PerturbationConfig
from recoverability.perturbations import PerturbationSchedule, sample_perturbation


class PerturbationTests(unittest.TestCase):
    def test_gripper_release_preserves_translation_and_opens_gripper(self) -> None:
        schedule = PerturbationSchedule(
            PerturbationConfig(kind="gripper_release", duration=2),
            start_step=3,
        )
        action = np.asarray([0.1, -0.2, 0.3, -1.0], dtype=np.float32)
        modified = schedule.before_action(None, action, step=3, rng=np.random.default_rng(1))
        np.testing.assert_allclose(modified[:3], action[:3])
        self.assertEqual(float(modified[3]), 1.0)

    def test_configured_start_window_overrides_fallback_window(self) -> None:
        config = PerturbationConfig(
            kind="action_dropout",
            start_step_min=14,
            start_step_max=14,
        )
        schedule = sample_perturbation((config,), start_step=2, rng=np.random.default_rng(1))
        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.start_step, 14)


if __name__ == "__main__":
    unittest.main()
