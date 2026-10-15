from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from tianji_lerobot_replay.frames import load_scene_config
from tianji_lerobot_replay.scene import TABLETOP_RGBA, build_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "models"
    / "ros_packages"
    / "daimon_tj_fx_description"
    / "urdf"
    / "daimon_tj_fx_dual_arm_with_gripper.urdf"
)


class SceneTests(unittest.TestCase):
    def test_scene_has_white_visual_only_table_and_two_tcp_markers(self) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as temporary:
            scene = build_scene(
                URDF,
                temporary,
                load_scene_config(PROJECT_ROOT / "config" / "default_scene.json"),
                [PROJECT_ROOT / "models" / "ros_packages"],
            )
            model = scene.model
            table_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_visual"
            )
            board_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "umi_calibration_board_visual"
            )
            self.assertGreaterEqual(table_id, 0)
            self.assertGreaterEqual(board_id, 0)
            self.assertEqual(int(model.geom_contype[table_id]), 0)
            self.assertEqual(int(model.geom_conaffinity[table_id]), 0)
            np.testing.assert_allclose(model.geom_rgba[table_id], TABLETOP_RGBA, atol=1e-6)
            self.assertEqual(model.nmocap, 2)
            self.assertEqual(set(scene.target_mocap_ids), {"left", "right"})
            self.assertTrue(scene.xml_path.is_file())
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            config = load_scene_config(PROJECT_ROOT / "config" / "default_scene.json")
            np.testing.assert_allclose(
                data.geom_xpos[board_id],
                config.robot_from_umi_positions(config.board.panel_center_in_umi_m),
                atol=1e-6,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
