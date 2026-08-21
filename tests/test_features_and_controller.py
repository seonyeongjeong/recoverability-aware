import unittest

import numpy as np

from recoverability.config import ControllerConfig
from recoverability.controllers import ControllerPhase, ScriptedPickCubeController
from recoverability.features import flatten_observation


def observation(tcp=(0.0, 0.0, 0.2), obj=(0.1, 0.0, 0.02), grasped=False):
    return {
        "agent": {
            "qvel": np.asarray([2.0, 3.0], dtype=np.float32),
            "qpos": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        "extra": {
            "tcp_pose": np.asarray([*tcp, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "obj_pose": np.asarray([*obj, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "goal_pos": np.asarray([0.2, 0.1, 0.2], dtype=np.float32),
            "is_grasped": np.asarray(grasped),
        },
    }


class FeatureAndControllerTests(unittest.TestCase):
    def test_flattening_is_sorted_and_stable(self) -> None:
        state, names = flatten_observation(observation())
        self.assertEqual(len(state), len(names))
        self.assertEqual(names[0], "agent.qpos[0]")
        self.assertIn("extra.goal_pos[0]", names)

    def test_controller_moves_toward_object_hover(self) -> None:
        controller = ScriptedPickCubeController(ControllerConfig())
        action = controller.act(observation())
        self.assertEqual(action.shape, (4,))
        self.assertGreater(action[0], 0.0)
        self.assertEqual(action[-1], 1.0)

    def test_lift_target_does_not_move_up_with_grasped_object(self) -> None:
        controller = ScriptedPickCubeController(
            ControllerConfig(),
            observation(tcp=(0.0, 0.0, 0.28), obj=(0.0, 0.0, 0.275), grasped=True),
        )
        action = controller.act(
            observation(tcp=(0.0, 0.0, 0.28), obj=(0.0, 0.0, 0.275), grasped=True)
        )
        self.assertEqual(controller.phase, ControllerPhase.TRANSFER)
        self.assertAlmostEqual(float(action[2]), 0.0, places=5)
        self.assertEqual(action[-1], -1.0)

    def test_nominal_continue_does_not_hide_a_recovery_after_drop(self) -> None:
        controller = ScriptedPickCubeController(ControllerConfig())
        controller.phase = ControllerPhase.TRANSFER
        action = controller.act(observation(grasped=False))
        self.assertEqual(controller.phase, ControllerPhase.TRANSFER)
        self.assertEqual(action[-1], -1.0)

    def test_controller_state_round_trip(self) -> None:
        original = ScriptedPickCubeController(ControllerConfig())
        original.phase = ControllerPhase.CLOSE
        original.close_counter = 3
        restored = ScriptedPickCubeController(ControllerConfig())
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.phase, ControllerPhase.CLOSE)
        self.assertEqual(restored.close_counter, 3)
        self.assertEqual(len(restored.state_features()["phase"]), len(ControllerPhase))


if __name__ == "__main__":
    unittest.main()
