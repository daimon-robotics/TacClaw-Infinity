from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tianji_lerobot_replay.arm_quality import (
    ArmQualityEvaluator,
    TianjiUrdfKinematics,
    Thresholds,
    _rotation_from_matrix,
)
from tianji_lerobot_replay.frames import load_scene_config
from tianji_lerobot_replay.quality_visualization import write_quality_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIANJI_URDF = (
    PROJECT_ROOT / "models" / "ros_packages" /
    "daimon_tj_fx_description" / "urdf" /
    "daimon_tj_fx_dual_arm_with_gripper.urdf"
)
TIANJI_JOINT_PARAMS = PROJECT_ROOT / "config" / "tianji_joints_params.xml"


def assert_jacobian(test: unittest.TestCase, model, side: str) -> None:
    lower, upper = model.bounds(side)
    q = 0.45 * lower + 0.55 * upper
    pose, jacobian = model.forward_jacobian(side, q, {})
    step = 1e-7
    numerical = np.zeros((6, 7))
    for index in range(7):
        moved_q = q.copy()
        moved_q[index] += step
        moved, _ = model.forward_jacobian(side, moved_q, {})
        numerical[:3, index] = (moved[:3, 3] - pose[:3, 3]) / step
        world_increment = moved[:3, :3] @ pose[:3, :3].T
        numerical[3:, index] = _rotation_from_matrix(
            world_increment).as_rotvec() / step
    np.testing.assert_allclose(jacobian, numerical, atol=2e-6)


class TianjiArmQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = TianjiUrdfKinematics(TIANJI_URDF, TIANJI_JOINT_PARAMS)
        cls.evaluator = ArmQualityEvaluator(cls.model, Thresholds())

    def test_tianji_uses_tcp_links_and_runtime_limits(self) -> None:
        self.assertEqual(self.model.target_links, {
            "left": "gripper_base_link_L",
            "right": "gripper_base_link_R",
        })
        self.assertEqual(self.model.arm_joints["left"][0], "Joint1_L")
        self.assertEqual(self.model.arm_joints["right"][-1], "Joint7_R")
        np.testing.assert_allclose(
            self.model.bounds("left")[0],
            [-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708],
        )

    def test_tianji_tcp_jacobians(self) -> None:
        for side in ("left", "right"):
            assert_jacobian(self, self.model, side)

    def test_tianji_solver_recovers_known_tcp_pose(self) -> None:
        for side in ("left", "right"):
            lower, upper = self.model.bounds(side)
            known = 0.49 * lower + 0.51 * upper
            target, _ = self.model.forward_jacobian(side, known, {})
            result = self.evaluator.solve_independent(side, target)
            self.assertTrue(result.reachable)

    def test_tianji_has_no_non_arm_upstream_joint(self) -> None:
        self.assertEqual(self.model.effective_fixed_joints(), {})

    def test_offline_dashboard_is_self_contained(self) -> None:
        thresholds = Thresholds().report()
        summary = {
            "robot_model": {"robot_name": self.model.robot_name},
            "comparison_conclusion": "PASS_BOTH",
            "thresholds": thresholds,
            "strategies": {
                strategy: {"both_reachable_rate": 1.0}
                for strategy in ("independent", "continuous_midpoint")
            },
        }
        frame = {
            "episode_index": 0, "frame_index": 0, "timestamp_s": 0.0,
            "left_reachable": True, "right_reachable": True,
            "left_isotropy_ratio": 0.1, "right_isotropy_ratio": 0.2,
            "left_joint_margin_fraction": 0.3,
            "right_joint_margin_fraction": 0.4,
            "left_position_error_m": 0.001, "right_position_error_m": 0.002,
            "left_orientation_error_deg": 0.5,
            "right_orientation_error_deg": 0.6,
            "left_target_x_m": 0.2, "left_target_y_m": 0.1,
            "left_target_z_m": -0.2, "right_target_x_m": 0.3,
            "right_target_y_m": -0.1, "right_target_z_m": -0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_quality_dashboard(
                Path(directory) / "quality_dashboard.html",
                summary,
                {"independent": [frame], "continuous_midpoint": [frame]},
            )
            html = path.read_text(encoding="utf-8")
            self.assertIn("Tianji fixed-base dual arm", html)
            self.assertIn("Jacobian isotropy ratio", html)
            self.assertIn('id="trajectory3D"', html)
            self.assertIn("drawTrajectory3D", html)

    def test_dashboard_parses_csv_boolean_strings(self) -> None:
        from tianji_lerobot_replay.quality_visualization import _required_bool

        self.assertTrue(_required_bool("True"))
        self.assertFalse(_required_bool("False"))
        self.assertTrue(_required_bool("1"))
        self.assertFalse(_required_bool("0"))


if __name__ == "__main__":
    unittest.main()
