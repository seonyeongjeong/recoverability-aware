from pathlib import Path
import unittest

from recoverability.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_reference_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "pick_cube.toml")
        self.assertEqual(config.environment.env_id, "PickCube-v1")
        self.assertEqual(config.environment.obs_mode, "state_dict")
        self.assertEqual(config.collection.remaining_budgets, (0, 1, 2))
        self.assertEqual(len(config.collection.perturbations), 4)
        self.assertEqual(config.collection.fallback_recovery_interval, 0)

    def test_pilot_config_loads(self) -> None:
        config = load_config(ROOT / "configs" / "pick_cube_pilot.toml")
        self.assertEqual(config.collection.episodes, 18)
        self.assertEqual(config.training.epochs, 30)
        self.assertTrue(
            any(item.kind == "gripper_release" for item in config.collection.perturbations)
        )


if __name__ == "__main__":
    unittest.main()
