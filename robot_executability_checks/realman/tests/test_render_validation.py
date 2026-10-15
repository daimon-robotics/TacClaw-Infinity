from dataclasses import replace
import copy
import unittest

import numpy as np

from tianji_lerobot_replay.errors import ReplayError
from tianji_lerobot_replay.frames import SceneConfig
from tianji_lerobot_replay.lerobot_v3 import TcpEpisode
from tianji_lerobot_replay.render_validation import validate_render_rows


class RenderValidationTests(unittest.TestCase):
    def setUp(self):
        self.scene = SceneConfig()
        self.episode = TcpEpisode(3, np.array([0., .1]), np.zeros((2, 2, 3)),
                                  np.tile([0., 0., 0., 1.], (2, 2, 1)))
        targets = self.scene.robot_from_umi_positions(self.episode.positions_world_m)
        self.rows = []
        for frame in range(2):
            row = dict(episode_index=3, frame_index=frame, timestamp_s=frame * .1)
            for arm, side in enumerate(("left", "right")):
                row.update({f"{side}_target_{axis}_m": targets[frame, arm, i]
                            for i, axis in enumerate("xyz")})
            self.rows.append(row)

    def test_matching_rows(self):
        result = validate_render_rows(self.rows, self.episode, self.scene)
        self.assertEqual(result["frames_checked"], 2)
        self.assertFalse(result["physical_registration_verified"])

    def test_other_scene_rejected(self):
        scene = replace(self.scene, robot_root_in_table_m=np.array([0., 0., 0.]))
        with self.assertRaises(ReplayError):
            validate_render_rows(self.rows, self.episode, scene)

    def test_same_length_other_trajectory_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["left_target_x_m"] += .02
        with self.assertRaises(ReplayError):
            validate_render_rows(rows, self.episode, self.scene)

    def test_duplicate_frames_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["frame_index"] = 0
        with self.assertRaises(ReplayError):
            validate_render_rows(rows, self.episode, self.scene)

    def test_time_mismatch_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["timestamp_s"] = 1.
        with self.assertRaises(ReplayError):
            validate_render_rows(rows, self.episode, self.scene)

    def test_full_body_moving_base_targets_are_compared_in_fixed_world(self):
        rows = copy.deepcopy(self.rows)
        direction = self.scene.rotation_robot_from_table[:, 1]
        for i, row in enumerate(rows):
            row["strategy"] = "adaptive_whole_body"
            row["base_root_y_table_m"] = -.4 + i * .05
            offset = self.scene.robot_root_in_table_m[1] - row["base_root_y_table_m"]
            for side in ("left", "right"):
                for j, axis in enumerate("xyz"):
                    row[f"{side}_target_{axis}_m"] += offset * direction[j]
        self.assertLess(validate_render_rows(rows, self.episode, self.scene)["max_target_difference_m"], 1e-12)
