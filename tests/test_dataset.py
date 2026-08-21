import tempfile
from pathlib import Path
import unittest

import numpy as np

from recoverability.dataset import CounterfactualDataset


def make_dataset() -> CounterfactualDataset:
    episodes = np.repeat(np.arange(10), 6)
    samples = len(episodes)
    return CounterfactualDataset(
        states=np.arange(samples * 3, dtype=np.float32).reshape(samples, 3),
        option_ids=np.tile(np.arange(3), samples // 3),
        remaining_budgets=np.ones(samples, dtype=np.float32),
        successes=(np.arange(samples) % 2).astype(np.float32),
        costs=np.arange(samples, dtype=np.float32),
        episode_ids=episodes,
        steps=np.tile(np.arange(6), 10),
        checkpoint_ids=np.asarray([f"cp-{index}" for index in range(samples)]),
        perturbations=np.asarray(["none"] * samples),
        feature_names=("a", "b", "c"),
    )


class DatasetTests(unittest.TestCase):
    def test_group_split_has_no_episode_leakage(self) -> None:
        train, validation, test = make_dataset().split_by_episode(0.2, 0.2, seed=3)
        train_ids = set(train.episode_ids.tolist())
        validation_ids = set(validation.episode_ids.tolist())
        test_ids = set(test.episode_ids.tolist())
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_round_trip_without_pickle(self) -> None:
        original = make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.npz"
            original.save(path)
            loaded = CounterfactualDataset.load(path)
        np.testing.assert_array_equal(loaded.states, original.states)
        self.assertEqual(loaded.feature_names, original.feature_names)

    def test_split_rejects_failures_from_only_one_episode(self) -> None:
        dataset = make_dataset()
        dataset.successes[:] = 1.0
        dataset.successes[dataset.episode_ids == 0] = 0.0
        with self.assertRaisesRegex(ValueError, "at least three distinct episodes"):
            dataset.split_by_episode(0.2, 0.2, seed=3)

    def test_feature_and_perturbation_ablation_helpers(self) -> None:
        dataset = make_dataset()
        selected = dataset.select_features(("a", "c"))
        self.assertEqual(selected.feature_names, ("a", "c"))
        np.testing.assert_array_equal(selected.states, dataset.states[:, [0, 2]])
        dataset.perturbations = dataset.perturbations.astype("<U16")
        dataset.perturbations[:6] = "object_shift"
        filtered = dataset.without_perturbation("object_shift")
        self.assertEqual(len(filtered), len(dataset) - 6)


if __name__ == "__main__":
    unittest.main()
