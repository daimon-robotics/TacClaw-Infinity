"""Keep the main branch's public robot entry points Tianji-only."""

from pathlib import Path
from contextlib import redirect_stdout
import io
import runpy
import sys
import unittest
from unittest.mock import patch

from tianji_lerobot_replay.workspace_specs import SPECS, get_workspace_spec


ROOT = Path(__file__).resolve().parents[1]


class MainBranchScopeTests(unittest.TestCase):
    def test_only_tianji_is_registered(self):
        self.assertEqual(set(SPECS), {"tianji"})
        self.assertTrue(get_workspace_spec("tianji").urdf.is_file())
        for other in ("humanoid", "realman"):
            with self.assertRaises(ValueError):
                get_workspace_spec(other)

    def test_evaluation_entry_point_is_tianji_only(self):
        namespace = runpy.run_path(str(ROOT / "scripts/evaluate_arm_quality.py"))
        self.assertEqual(set(namespace["ROBOT_DEFAULTS"]), {"tianji"})

    def test_specialized_models_are_not_packaged_on_main(self):
        packages = ROOT / "models/ros_packages"
        self.assertFalse((packages / "dm25_pr02_hr_pd04_eg_20260810").exists())
        self.assertFalse((packages / "dual_rm75_b_description").exists())
        for script in ("render_action_space_video", "render_reachable_workspace",
                       "render_workspace_visualization", "view_workspace"):
            with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
                namespace = runpy.run_path(str(ROOT / f"scripts/{script}.py"))
            with patch.object(sys, "argv", [script, "--help"]):
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                    namespace["parse_args"]()
            self.assertEqual(exited.exception.code, 0)
