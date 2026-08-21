from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


@dataclass
class CounterfactualDataset:
    states: np.ndarray
    option_ids: np.ndarray
    remaining_budgets: np.ndarray
    successes: np.ndarray
    costs: np.ndarray
    episode_ids: np.ndarray
    steps: np.ndarray
    checkpoint_ids: np.ndarray
    perturbations: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        arrays = (
            self.option_ids,
            self.remaining_budgets,
            self.successes,
            self.costs,
            self.episode_ids,
            self.steps,
            self.checkpoint_ids,
            self.perturbations,
        )
        if self.states.ndim != 2:
            raise ValueError("states must have shape [samples, features].")
        if any(len(array) != len(self.states) for array in arrays):
            raise ValueError("Every dataset field must have the same sample count.")
        if len(self.feature_names) != self.states.shape[1]:
            raise ValueError("feature_names must match the state feature dimension.")
        if len(self.states) and not np.isfinite(self.states).all():
            raise ValueError("states contain NaN or infinite values.")
        if len(self.states) and not np.isfinite(self.costs).all():
            raise ValueError("costs contain NaN or infinite values.")

    def __len__(self) -> int:
        return len(self.states)

    def subset(self, indices: np.ndarray) -> "CounterfactualDataset":
        return CounterfactualDataset(
            states=self.states[indices],
            option_ids=self.option_ids[indices],
            remaining_budgets=self.remaining_budgets[indices],
            successes=self.successes[indices],
            costs=self.costs[indices],
            episode_ids=self.episode_ids[indices],
            steps=self.steps[indices],
            checkpoint_ids=self.checkpoint_ids[indices],
            perturbations=self.perturbations[indices],
            feature_names=self.feature_names,
        )

    def select_features(self, names: tuple[str, ...]) -> "CounterfactualDataset":
        positions = {name: index for index, name in enumerate(self.feature_names)}
        missing = [name for name in names if name not in positions]
        if missing:
            raise ValueError(f"Unknown dataset feature(s): {', '.join(missing)}")
        columns = np.asarray([positions[name] for name in names], dtype=np.int64)
        return CounterfactualDataset(
            states=self.states[:, columns],
            option_ids=self.option_ids.copy(),
            remaining_budgets=self.remaining_budgets.copy(),
            successes=self.successes.copy(),
            costs=self.costs.copy(),
            episode_ids=self.episode_ids.copy(),
            steps=self.steps.copy(),
            checkpoint_ids=self.checkpoint_ids.copy(),
            perturbations=self.perturbations.copy(),
            feature_names=names,
        )

    def without_perturbation(self, kind: str) -> "CounterfactualDataset":
        return self.subset(np.flatnonzero(self.perturbations != kind))

    def split_by_episode(
        self,
        validation_fraction: float,
        test_fraction: float,
        seed: int,
    ) -> tuple["CounterfactualDataset", "CounterfactualDataset", "CounterfactualDataset"]:
        """Group split prevents counterfactual variants leaking across splits."""

        episodes = np.unique(self.episode_ids)
        if len(episodes) < 3:
            raise ValueError("At least three source episodes are required for train/val/test splits.")
        validation_count = max(1, int(round(len(episodes) * validation_fraction)))
        test_count = max(1, int(round(len(episodes) * test_fraction)))
        if validation_count + test_count >= len(episodes):
            overflow = validation_count + test_count - len(episodes) + 1
            if test_count > 1:
                reduction = min(overflow, test_count - 1)
                test_count -= reduction
                overflow -= reduction
            validation_count -= overflow
        def indices(group: np.ndarray) -> np.ndarray:
            return np.flatnonzero(np.isin(self.episode_ids, group))

        # Keep counterfactuals from one source episode together, but retry the
        # group shuffle until every split contains both successes and failures.
        # Without this guard, calibration on an all-success split looks
        # artificially perfect and cannot measure failure discrimination.
        for attempt in range(1_000):
            shuffled = episodes.copy()
            np.random.default_rng(seed + attempt).shuffle(shuffled)
            validation_episodes = shuffled[:validation_count]
            test_episodes = shuffled[validation_count : validation_count + test_count]
            train_episodes = shuffled[validation_count + test_count :]
            split_indices = (
                indices(train_episodes),
                indices(validation_episodes),
                indices(test_episodes),
            )
            if all(np.unique(self.successes[selected]).size == 2 for selected in split_indices):
                return tuple(self.subset(selected) for selected in split_indices)

        failure_episodes = np.unique(self.episode_ids[self.successes < 0.5])
        raise ValueError(
            "Could not create episode-disjoint train/validation/test splits with "
            "both success classes. Collect failures in at least three distinct "
            f"episodes; this dataset has {len(failure_episodes)} failure episode(s)."
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int64),
            states=self.states.astype(np.float32),
            option_ids=self.option_ids.astype(np.int64),
            remaining_budgets=self.remaining_budgets.astype(np.float32),
            successes=self.successes.astype(np.float32),
            costs=self.costs.astype(np.float32),
            episode_ids=self.episode_ids.astype(np.int64),
            steps=self.steps.astype(np.int64),
            checkpoint_ids=self.checkpoint_ids.astype(str),
            perturbations=self.perturbations.astype(str),
            feature_names=np.asarray(self.feature_names, dtype=str),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CounterfactualDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["schema_version"][0])
            if version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported dataset schema version: {version}")
            return cls(
                states=data["states"],
                option_ids=data["option_ids"],
                remaining_budgets=data["remaining_budgets"],
                successes=data["successes"],
                costs=data["costs"],
                episode_ids=data["episode_ids"],
                steps=data["steps"],
                checkpoint_ids=data["checkpoint_ids"],
                perturbations=data["perturbations"],
                feature_names=tuple(data["feature_names"].tolist()),
            )

    def summary(self) -> dict[str, float | int]:
        failures = self.successes < 0.5
        return {
            "samples": len(self),
            "features": self.states.shape[1],
            "episodes": len(np.unique(self.episode_ids)),
            "checkpoints": len(np.unique(self.checkpoint_ids)),
            "successes": int((~failures).sum()),
            "failures": int(failures.sum()),
            "failure_episodes": int(len(np.unique(self.episode_ids[failures]))),
            "success_rate": float(self.successes.mean()) if len(self) else float("nan"),
            "mean_cost": float(self.costs.mean()) if len(self) else float("nan"),
        }

    def quality_report(self) -> dict[str, Any]:
        def breakdown(values: np.ndarray) -> dict[str, dict[str, float | int]]:
            result: dict[str, dict[str, float | int]] = {}
            for value in sorted(np.unique(values).tolist(), key=str):
                selected = values == value
                failures = self.successes[selected] < 0.5
                result[str(value)] = {
                    "samples": int(selected.sum()),
                    "failures": int(failures.sum()),
                    "success_rate": float(self.successes[selected].mean()),
                }
            return result

        return {
            **self.summary(),
            "by_option": breakdown(self.option_ids),
            "by_perturbation": breakdown(self.perturbations),
        }
