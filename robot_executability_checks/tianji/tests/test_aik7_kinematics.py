from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tianji_lerobot_replay.frames import load_scene_config
from tianji_lerobot_replay.kinematics import TianjiKinematicReplayer
from tianji_lerobot_replay.scene import build_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "models"
    / "ros_packages"
    / "daimon_tj_fx_description"
    / "urdf"
    / "daimon_tj_fx_dual_arm_with_gripper.urdf"
)
NATIVE_LIBRARY = PROJECT_ROOT / "native" / "build" / "libtianji_aik7.so"


class Aik7KinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        if not NATIVE_LIBRARY.is_file():
            self.skipTest("Aik7 native library is not built; run ./setup.sh first.")
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.scene = build_scene(
            URDF,
            self.temporary.name,
            load_scene_config(PROJECT_ROOT / "config" / "default_scene.json"),
            [PROJECT_ROOT / "models" / "ros_packages"],
        )
        self.solver = TianjiKinematicReplayer(self.scene.model)

    def _known_tcp_targets(self) -> tuple[np.ndarray, np.ndarray]:
        known_q = {
            "left": np.array([0.25, -0.55, 0.48, -1.00, 0.35, 0.15, -0.40]),
            "right": np.array([0.25, 0.55, -0.48, 0.50, -0.35, -0.15, 0.40]),
        }
        self.solver.reset_pose()
        for side, q in known_q.items():
            self.solver.data.qpos[self.solver.arms[side].qpos_addresses] = q
        self.solver.mujoco.mj_forward(self.solver.model, self.solver.data)
        positions: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for side in ("left", "right"):
            transform = self.solver._body_transform(self.solver.arms[side].tcp_body_id)
            positions.append(transform[:3, 3])
            rotations.append(transform[:3, :3])
        return np.asarray([positions]), np.asarray([rotations])

    def test_tcp_to_link7_to_ee_aik7_round_trip(self) -> None:
        positions, rotations = self._known_tcp_targets()
        track = self.solver.solve_episode(positions, rotations)
        self.assertTrue(np.all(track.analytic_branch_found))
        self.assertLess(float(np.max(track.position_residual_m)), 2e-4)
        self.assertLess(float(np.max(track.orientation_residual_rad)), np.deg2rad(0.1))
        # The Aik7 initial branch is allowed to differ from the q used to
        # generate this target; redundancy is selected by continuity, not by
        # hidden knowledge of the source q.
        self.assertEqual(track.qpos.shape, (1, self.scene.model.nq))

    def test_no_analytical_branch_keeps_previous_joint_state(self) -> None:
        self.solver.reset_pose()
        previous = self.solver.data.qpos.copy()
        positions = np.full((1, 2, 3), 10.0, dtype=float)
        rotations = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
        track = self.solver.solve_episode(positions, rotations)
        self.assertFalse(np.any(track.analytic_branch_found))
        np.testing.assert_allclose(track.qpos[0], previous, atol=1e-12)

    def test_q7_search_keeps_current_value_first(self) -> None:
        arm = self.solver.arms["left"]
        values = self.solver._ordered_q7_values(arm, 0.123)
        self.assertEqual(values[0], 0.123)
        self.assertGreater(len(values), 100)
        self.assertGreaterEqual(min(values), arm.lower[6])
        self.assertLessEqual(max(values), arm.upper[6])

    def test_right_arm_can_record_an_explicit_alternate_elbow_root(self) -> None:
        # q4<0 is a valid right-arm geometry that the historical WBC's
        # preferred right-positive elbow root cannot represent.  It must stay
        # inside the same Aik7 analytical solver, not fall back to DLS.
        self.solver.reset_pose()
        self.solver.data.qpos[self.solver.arms["left"].qpos_addresses] = np.array(
            [0.25, -0.55, 0.48, -1.00, 0.35, 0.15, -0.40]
        )
        self.solver.data.qpos[self.solver.arms["right"].qpos_addresses] = np.array(
            [0.25, -0.35, 0.20, -1.50, 0.30, 0.25, -0.15]
        )
        self.solver.mujoco.mj_forward(self.solver.model, self.solver.data)
        positions: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for side in ("left", "right"):
            transform = self.solver._body_transform(self.solver.arms[side].tcp_body_id)
            positions.append(transform[:3, 3])
            rotations.append(transform[:3, :3])
        track = self.solver.solve_episode(np.asarray([positions]), np.asarray([rotations]))
        self.assertTrue(track.analytic_branch_found[0, 1])
        self.assertEqual(int(track.analytic_elbow_sign[0, 1]), -1)
        self.assertLess(float(track.position_residual_m[0, 1]), 2e-4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
