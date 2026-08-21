from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np

from .config import ControllerConfig
from .features import get_extra


class ControllerPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    TRANSFER = auto()
    SETTLE = auto()


def _as_bool(value: np.ndarray) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


class ScriptedPickCubeController:
    """Privileged-state controller for PickCube using EE delta position control."""

    def __init__(self, config: ControllerConfig, observation: Any | None = None):
        self.config = config
        self.phase = ControllerPhase.APPROACH
        self.close_counter = 0
        if observation is not None and _as_bool(get_extra(observation, "is_grasped")):
            self.phase = ControllerPhase.LIFT

    def _translation_action(self, tcp: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = (target - tcp) / self.config.position_action_scale_m
        return np.clip(
            delta,
            -self.config.max_translation_action,
            self.config.max_translation_action,
        ).astype(np.float32)

    def _action(self, tcp: np.ndarray, target: np.ndarray, gripper: float) -> np.ndarray:
        return np.concatenate(
            [self._translation_action(tcp, target), np.asarray([gripper], dtype=np.float32)]
        )

    def state_dict(self) -> dict[str, int | str]:
        return {"phase": self.phase.name, "close_counter": self.close_counter}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if state is None:
            return
        self.phase = ControllerPhase[str(state["phase"])]
        self.close_counter = int(state.get("close_counter", 0))

    def state_features(self) -> dict[str, np.ndarray]:
        phase = np.zeros(len(ControllerPhase), dtype=np.float32)
        phase[self.phase.value - 1] = 1.0
        close_progress = min(1.0, self.close_counter / max(1, self.config.close_steps))
        return {
            "phase": phase,
            "close_progress": np.asarray(close_progress, dtype=np.float32),
        }

    def act(self, observation: Any) -> np.ndarray:
        tcp = get_extra(observation, "tcp_pose").astype(np.float32)[:3]
        obj = get_extra(observation, "obj_pose").astype(np.float32)[:3]
        goal = get_extra(observation, "goal_pos").astype(np.float32)[:3]
        grasped = _as_bool(get_extra(observation, "is_grasped"))

        hover = obj + np.asarray([0.0, 0.0, self.config.hover_height], dtype=np.float32)
        grasp = obj + np.asarray([0.0, 0.0, self.config.grasp_offset], dtype=np.float32)

        if self.phase is ControllerPhase.APPROACH:
            if np.linalg.norm(tcp - hover) <= self.config.position_tolerance:
                self.phase = ControllerPhase.DESCEND
            return self._action(tcp, hover, gripper=1.0)

        if self.phase is ControllerPhase.DESCEND:
            if np.linalg.norm(tcp - grasp) <= self.config.position_tolerance:
                self.phase = ControllerPhase.CLOSE
                self.close_counter = 0
            return self._action(tcp, grasp, gripper=1.0)

        if self.phase is ControllerPhase.CLOSE:
            self.close_counter += 1
            if grasped and self.close_counter >= self.config.grasp_confirmation_steps:
                self.phase = ControllerPhase.LIFT
            elif self.close_counter >= self.config.close_steps:
                self.phase = ControllerPhase.APPROACH
            return self._action(tcp, grasp, gripper=-1.0)

        if self.phase is ControllerPhase.LIFT:
            if not grasped:
                # Continuing the nominal plan is intentionally not a hidden
                # recovery. A separate RECOVERY option must be selected to
                # retreat, open the gripper, and reacquire a dropped object.
                lift_target = tcp.copy()
                lift_target[2] = float(goal[2] + self.config.hover_height)
                return self._action(tcp, lift_target, gripper=-1.0)
            lift_target = tcp.copy()
            # Use a fixed clearance above the goal. Basing this target on the
            # current object height makes it move upward with the grasped cube,
            # so the controller can never finish the lift phase.
            lift_target[2] = float(goal[2] + self.config.hover_height)
            if abs(float(tcp[2] - lift_target[2])) <= self.config.position_tolerance:
                self.phase = ControllerPhase.TRANSFER
            return self._action(tcp, lift_target, gripper=-1.0)

        target_at_goal = goal + np.asarray(
            [0.0, 0.0, self.config.grasp_offset], dtype=np.float32
        )
        if self.phase is ControllerPhase.TRANSFER:
            if not grasped:
                return self._action(tcp, target_at_goal, gripper=-1.0)
            if np.linalg.norm(tcp - target_at_goal) <= self.config.position_tolerance:
                self.phase = ControllerPhase.SETTLE
            return self._action(tcp, target_at_goal, gripper=-1.0)

        # SETTLE: keep holding the cube still in the goal region.
        return self._action(tcp, target_at_goal, gripper=-1.0)


class RecoveryThenNominalController:
    """Open/retreat/reacquire recovery followed by the nominal controller."""

    def __init__(self, config: ControllerConfig, observation: Any):
        self.config = config
        self.retreat_counter = 0
        self.nominal: ScriptedPickCubeController | None = None
        if _as_bool(get_extra(observation, "is_grasped")):
            self.nominal = ScriptedPickCubeController(config, observation)

    def act(self, observation: Any) -> np.ndarray:
        if self.nominal is not None:
            return self.nominal.act(observation)

        tcp = get_extra(observation, "tcp_pose").astype(np.float32)[:3]
        obj = get_extra(observation, "obj_pose").astype(np.float32)[:3]
        target = obj + np.asarray(
            [0.0, 0.0, self.config.hover_height], dtype=np.float32
        )
        delta = (target - tcp) / self.config.position_action_scale_m
        translation = np.clip(
            delta,
            -self.config.max_translation_action,
            self.config.max_translation_action,
        ).astype(np.float32)
        self.retreat_counter += 1
        reached = np.linalg.norm(tcp - target) <= self.config.position_tolerance
        if reached or self.retreat_counter >= self.config.recovery_retreat_steps:
            self.nominal = ScriptedPickCubeController(self.config, observation)
        return np.concatenate([translation, np.asarray([1.0], dtype=np.float32)])

    def state_features(self) -> dict[str, np.ndarray]:
        if self.nominal is not None:
            return self.nominal.state_features()
        phase = np.zeros(len(ControllerPhase), dtype=np.float32)
        phase[ControllerPhase.APPROACH.value - 1] = 1.0
        return {
            "phase": phase,
            "close_progress": np.asarray(0.0, dtype=np.float32),
        }
