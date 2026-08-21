from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def unbatch(value: Any) -> Any:
    """Remove ManiSkill's singleton environment dimension recursively."""

    if isinstance(value, Mapping):
        return {key: unbatch(child) for key, child in value.items()}
    array = to_numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def flatten_observation(
    observation: Any,
    controller_features: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Deterministically flatten a privileged state_dict observation."""

    values: list[np.ndarray] = []
    names: list[str] = []

    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node):
                visit(node[key], f"{prefix}.{key}" if prefix else str(key))
            return
        array = to_numpy(node)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        flat = array.astype(np.float32, copy=False).reshape(-1)
        values.append(flat)
        if flat.size == 1:
            names.append(prefix)
        else:
            names.extend(f"{prefix}[{index}]" for index in range(flat.size))

    visit(observation, "")
    if controller_features is not None:
        visit(controller_features, "controller")
    if not values:
        raise ValueError("Observation contains no numeric leaves.")
    return np.concatenate(values, dtype=np.float32), tuple(names)


def get_extra(observation: Mapping[str, Any], key: str) -> np.ndarray:
    try:
        value = observation["extra"][key]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"PickCube observation is missing extra.{key}") from exc
    array = to_numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array
