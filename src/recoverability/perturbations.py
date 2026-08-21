from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PerturbationConfig
from .environment import ManiSkillPickCube


SUPPORTED_PERTURBATIONS = {
    "action_noise",
    "action_dropout",
    "gripper_release",
    "object_shift",
}


@dataclass
class PerturbationSchedule:
    config: PerturbationConfig
    start_step: int
    _state_applied: bool = False

    @property
    def end_step(self) -> int:
        return self.start_step + self.config.duration

    def before_action(
        self,
        env: ManiSkillPickCube,
        action: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if step < self.start_step or step >= self.end_step:
            return action
        if self.config.kind == "object_shift":
            if not self._state_applied:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                delta = np.asarray(
                    [np.cos(angle), np.sin(angle), 0.0], dtype=np.float32
                ) * self.config.magnitude
                env.shift_object(delta)
                self._state_applied = True
            return action
        modified = np.asarray(action, dtype=np.float32).copy()
        if self.config.kind == "action_noise":
            modified[:3] += rng.normal(
                0.0, self.config.magnitude, size=3
            ).astype(np.float32)
        elif self.config.kind == "action_dropout":
            modified[:3] *= max(0.0, 1.0 - self.config.magnitude)
        elif self.config.kind == "gripper_release":
            modified[3] = 1.0
        else:
            raise ValueError(f"Unsupported perturbation kind: {self.config.kind}")
        return np.clip(modified, -1.0, 1.0)


def sample_perturbation(
    configs: tuple[PerturbationConfig, ...],
    start_step: int,
    rng: np.random.Generator,
) -> PerturbationSchedule | None:
    if not configs:
        return None
    for config in configs:
        if config.kind not in SUPPORTED_PERTURBATIONS:
            raise ValueError(f"Unsupported perturbation kind: {config.kind}")
    weights = np.asarray([config.weight for config in configs], dtype=np.float64)
    weights /= weights.sum()
    index = int(rng.choice(len(configs), p=weights))
    config = configs[index]
    if config.start_step_min is not None and config.start_step_max is not None:
        start_step = int(
            rng.integers(config.start_step_min, config.start_step_max + 1)
        )
    return PerturbationSchedule(config=config, start_step=start_step)
