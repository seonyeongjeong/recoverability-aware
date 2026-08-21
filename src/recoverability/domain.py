from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


class InterventionOption(IntEnum):
    """The three options named in the research plan."""

    CONTINUE = 0
    RECOVERY = 1
    RESET = 2

    @classmethod
    def parse(cls, value: str | int | "InterventionOption") -> "InterventionOption":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls[value.strip().upper()]
        return cls(value)


@dataclass
class Checkpoint:
    """Restorable simulator state plus experiment bookkeeping."""

    env_state: dict[str, Any]
    observation: Any
    episode_id: int
    step: int
    checkpoint_id: str
    controller_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class RolloutOutcome:
    success: bool
    execution_cost: float
    steps: int
    option: InterventionOption
    remaining_budget: int


@dataclass(frozen=True)
class Prediction:
    success_probability: float
    expected_cost: float


def scalar_bool(value: Any) -> bool:
    """Convert a scalar-like NumPy/Torch value to a Python bool."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size == 0:
        return False
    return bool(array.reshape(-1)[0])
