from types import SimpleNamespace
import unittest

from recoverability.environment import _state_only_pick_cube_builders


class _Builder:
    def __init__(self) -> None:
        self.collision = None
        self.initial_pose = None

    def add_box_collision(self, *, half_size):
        self.collision = ("box", half_size)

    def add_sphere_collision(self, *, radius):
        self.collision = ("sphere", radius)

    def set_scene_idxs(self, scene_idxs):
        self.scene_idxs = scene_idxs

    def set_initial_pose(self, initial_pose):
        self.initial_pose = initial_pose

    def build(self, *, name):
        return ("dynamic", name, self.collision, self.initial_pose)

    def build_static(self, *, name):
        return ("static", name, self.collision, self.initial_pose)

    def build_kinematic(self, *, name):
        return ("kinematic", name, self.collision, self.initial_pose)


class _Scene:
    def create_actor_builder(self):
        return _Builder()


class EnvironmentCompatibilityTests(unittest.TestCase):
    def test_state_only_builders_preserve_physics_and_restore_originals(self) -> None:
        original_cube = object()
        original_sphere = object()
        actors = SimpleNamespace(
            build_cube=original_cube,
            build_sphere=original_sphere,
        )

        with _state_only_pick_cube_builders(actors):
            cube = actors.build_cube(
                _Scene(), 0.02, [1, 0, 0, 1], "cube", initial_pose="cube-pose"
            )
            goal = actors.build_sphere(
                _Scene(),
                0.025,
                [0, 1, 0, 1],
                "goal",
                body_type="kinematic",
                add_collision=False,
                initial_pose="goal-pose",
            )

        self.assertEqual(cube, ("dynamic", "cube", ("box", [0.02] * 3), "cube-pose"))
        self.assertEqual(goal, ("kinematic", "goal", None, "goal-pose"))
        self.assertIs(actors.build_cube, original_cube)
        self.assertIs(actors.build_sphere, original_sphere)


if __name__ == "__main__":
    unittest.main()
