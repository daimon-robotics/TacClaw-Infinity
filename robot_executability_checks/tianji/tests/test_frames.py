from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from tianji_lerobot_replay.frames import (
    SceneConfig,
    load_scene_config,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)
from tianji_lerobot_replay.sim_registration import calibrate_current_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrameTests(unittest.TestCase):
    def test_world_origin_maps_to_configured_table_point(self) -> None:
        config = SceneConfig()
        table_point = config.table_from_umi_positions(np.zeros(3))
        np.testing.assert_allclose(table_point, config.world_origin_in_table_m)

    def test_quaternion_round_trip(self) -> None:
        quaternion = np.array([0.2, -0.3, 0.4, 0.7], dtype=float)
        rotation = quaternion_xyzw_to_matrix(quaternion)
        recovered = matrix_to_quaternion_xyzw(rotation)
        np.testing.assert_allclose(
            quaternion_xyzw_to_matrix(recovered), rotation, atol=1e-12
        )

    def test_scene_transform_is_proper_rotation(self) -> None:
        config = SceneConfig()
        self.assertAlmostEqual(np.linalg.det(config.rotation_robot_from_umi), 1.0)

    def test_measured_three_side_board_geometry_closes(self) -> None:
        config = load_scene_config(PROJECT_ROOT / "config" / "default_scene.json")
        board = config.board
        triangle = config.support_triangle_vertices_table_m()
        self.assertAlmostEqual(
            np.linalg.norm(triangle["lower"] - triangle["rear_base"]), 0.080, places=9
        )
        self.assertAlmostEqual(
            np.linalg.norm(triangle["top"] - triangle["rear_base"]), 0.285, places=9
        )
        self.assertAlmostEqual(
            np.linalg.norm(triangle["top"] - triangle["lower"]), 0.302, places=9
        )
        self.assertAlmostEqual(
            np.degrees(board.support_base_to_side_angle_rad), 94.5015946679, places=7
        )
        self.assertAlmostEqual(
            np.degrees(board.panel_tilt_from_vertical_rad), 19.8140950174, places=7
        )
        np.testing.assert_allclose(
            triangle["lower"], [0.1028792219, 0.11, 0.8032356041], atol=1e-9
        )

    def test_virtual_registration_matches_direct_scene_chain(self) -> None:
        config = load_scene_config(PROJECT_ROOT / "config" / "default_scene.json")
        registration = calibrate_current_simulation(config)
        np.testing.assert_allclose(
            registration.rotation_base_from_umi, config.rotation_robot_from_umi, atol=1e-12
        )
        np.testing.assert_allclose(
            registration.translation_base_from_umi_m,
            config.translation_robot_from_umi_m,
            atol=1e-12,
        )
        self.assertLess(registration.fit_max_m, 1e-12)
        self.assertLess(registration.closure_rotation_error_deg, 1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
