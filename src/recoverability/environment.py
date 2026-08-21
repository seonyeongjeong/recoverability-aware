from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
import importlib.util
import platform
from typing import Any

import numpy as np

from .config import EnvironmentConfig
from .domain import Checkpoint, scalar_bool
from .features import unbatch


def _build_state_only_primitive(
    scene: Any,
    *,
    collision: str,
    size: float,
    name: str,
    body_type: str,
    add_collision: bool,
    scene_idxs: Any,
    initial_pose: Any,
) -> Any:
    """Build a primitive's physics actor without constructing Vulkan materials."""
    builder = scene.create_actor_builder()
    if add_collision:
        if collision == "box":
            builder.add_box_collision(half_size=[size] * 3)
        elif collision == "sphere":
            builder.add_sphere_collision(radius=size)
        else:  # pragma: no cover - only the two PickCube primitives use this helper
            raise ValueError(f"Unsupported collision primitive: {collision}")
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        return builder.build(name=name)
    if body_type == "static":
        return builder.build_static(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    raise ValueError(f"Unknown body type: {body_type}")


@contextmanager
def _state_only_pick_cube_builders(actors: Any) -> Iterator[None]:
    """Avoid SAPIEN material creation when ManiSkill rendering is disabled.

    ManiSkill 3.0.1's PickCube task constructs RenderMaterial objects for its
    cube and goal marker even with ``render_backend='none'``. SAPIEN then asks
    for a Vulkan device, which is unavailable under WSL. The visual shapes are
    irrelevant to this project's state-only observations, so temporarily use
    collision/state-only builders while the environment creates its scene.
    """
    original_cube = actors.build_cube
    original_sphere = actors.build_sphere

    def build_cube(
        scene: Any,
        half_size: float,
        color: Any,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
        scene_idxs: Any = None,
        initial_pose: Any = None,
    ) -> Any:
        del color
        return _build_state_only_primitive(
            scene,
            collision="box",
            size=half_size,
            name=name,
            body_type=body_type,
            add_collision=add_collision,
            scene_idxs=scene_idxs,
            initial_pose=initial_pose,
        )

    def build_sphere(
        scene: Any,
        radius: float,
        color: Any,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
        scene_idxs: Any = None,
        initial_pose: Any = None,
    ) -> Any:
        del color
        return _build_state_only_primitive(
            scene,
            collision="sphere",
            size=radius,
            name=name,
            body_type=body_type,
            add_collision=add_collision,
            scene_idxs=scene_idxs,
            initial_pose=initial_pose,
        )

    actors.build_cube = build_cube
    actors.build_sphere = build_sphere
    try:
        yield
    finally:
        actors.build_cube = original_cube
        actors.build_sphere = original_sphere


class ManiSkillPickCube:
    """Small CPU-state adapter around ManiSkill's PickCube environment.

    Imports are intentionally lazy: data/model utilities remain usable on machines
    that do not have the simulator installed.
    """

    def __init__(self, config: EnvironmentConfig):
        if (
            platform.system() == "Windows"
            and config.sim_backend == "physx_cpu"
            and config.control_mode.startswith("pd_ee")
            and importlib.util.find_spec("pinocchio") is None
        ):
            raise RuntimeError(
                "ManiSkill CPU end-effector controllers require Pinocchio, whose pip "
                "wheel is not available on native Windows. Run the default experiment "
                "inside WSL2/Linux, as anticipated by the research plan."
            )
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401 - registers ManiSkill environments
            from mani_skill.utils.building import actors
            from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
        except ImportError as exc:
            raise RuntimeError(
                "ManiSkill is not installed. Install the project with `pip install -e .`."
            ) from exc

        with _state_only_pick_cube_builders(actors):
            raw_env = gym.make(
                config.env_id,
                num_envs=1,
                obs_mode=config.obs_mode,
                control_mode=config.control_mode,
                sim_backend=config.sim_backend,
                # ManiSkill 3.0.1 documents None but its backend parser expects a
                # string on Windows; "none" is the equivalent headless setting.
                render_backend="none",
                render_mode=None,
                max_episode_steps=config.max_episode_steps,
            )
        self.env = CPUGymWrapper(raw_env)
        self.base = self.env.unwrapped
        self.config = config
        self.step_count = 0
        self.initial_checkpoint: Checkpoint | None = None
        self.last_observation: Any = None

    def reset(self, seed: int, episode_id: int = 0) -> Any:
        # ManiSkill 3.0.1's CPU PickCube scene can retain stale grasp/contact
        # state across ordinary episode resets. A full reconfiguration produces
        # repeatable initial conditions. Keep checkpoint restoration separate
        # and lightweight in ``restore`` below.
        from mani_skill.utils.building import actors

        with _state_only_pick_cube_builders(actors):
            observation, _ = self.env.reset(
                seed=seed,
                options={"reconfigure": True},
            )
        self.step_count = 0
        self.last_observation = observation
        self.initial_checkpoint = self.checkpoint(
            episode_id=episode_id,
            checkpoint_id=f"episode-{episode_id}-initial",
        )
        return observation

    def checkpoint(
        self,
        episode_id: int,
        checkpoint_id: str | None = None,
        controller_state: dict[str, Any] | None = None,
    ) -> Checkpoint:
        if self.last_observation is None:
            raise RuntimeError("The environment must be reset before checkpointing.")
        identifier = checkpoint_id or f"episode-{episode_id}-step-{self.step_count}"
        return Checkpoint(
            env_state=copy.deepcopy(self.base.get_state_dict()),
            observation=copy.deepcopy(self.last_observation),
            episode_id=episode_id,
            step=self.step_count,
            checkpoint_id=identifier,
            controller_state=copy.deepcopy(controller_state),
        )

    def restore(self, checkpoint: Checkpoint) -> Any:
        # reset_to_env_states also resets Gym's episode bookkeeping. The state is
        # batched because it comes directly from BaseEnv.get_state_dict().
        observation, _ = self.env.reset(
            options={
                "reset_to_env_states": {
                    "env_states": copy.deepcopy(checkpoint.env_state),
                }
            }
        )
        self.step_count = checkpoint.step
        self.last_observation = observation
        return observation

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        clipped = np.clip(
            np.asarray(action, dtype=np.float32),
            self.env.action_space.low,
            self.env.action_space.high,
        )
        observation, reward, terminated, truncated, info = self.env.step(clipped)
        self.step_count += 1
        self.last_observation = observation
        return observation, float(reward), bool(terminated), bool(truncated), info

    def is_success(self, info: dict[str, Any] | None = None) -> bool:
        if info is not None and "success" in info:
            return scalar_bool(info["success"])
        evaluation = self.base.evaluate()
        return scalar_bool(evaluation.get("success", False))

    def shift_object(self, delta_xyz: np.ndarray) -> None:
        """Apply an object-state perturbation without changing the task goal."""

        try:
            from mani_skill.utils.structs import Pose
        except ImportError as exc:  # pragma: no cover - guarded by __init__
            raise RuntimeError("ManiSkill is required to perturb simulator state.") from exc

        delta = np.asarray(delta_xyz, dtype=np.float32).reshape(3)
        pose = self.base.cube.pose
        position = pose.p.clone()
        delta_tensor = position.new_tensor(delta).reshape(1, 3)
        position = position + delta_tensor
        self.base.cube.set_pose(Pose.create_from_pq(position, pose.q.clone()))

    def close(self) -> None:
        self.env.close()

    @property
    def action_shape(self) -> tuple[int, ...]:
        return tuple(self.env.action_space.shape)

    def observation_unbatched(self) -> Any:
        return unbatch(self.last_observation)
