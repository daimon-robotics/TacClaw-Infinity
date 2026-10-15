import unittest

import numpy as np

from tianji_lerobot_replay.pointcloud_visualization import (
    backproject_depth,
    world_from_rectified_camera,
    world_from_rectified_camera_via_calibrated_imu,
)


class PointCloudVisualizationTests(unittest.TestCase):
    def test_backproject_depth(self):
        depth = np.full((2, 3), 2.0)
        points = backproject_depth(depth, (2.0, 2.0, 1.0, 0.5))
        np.testing.assert_allclose(points[0, 1], [0.0, -0.5, 2.0])
        np.testing.assert_allclose(points[1, 2], [1.0, 0.5, 2.0])

    def test_world_from_rectified_camera_chain(self):
        world_from_head = np.eye(4)
        world_from_head[:3, 3] = [1.0, 2.0, 3.0]
        result = world_from_rectified_camera(
            world_from_head, np.eye(3)
        )
        np.testing.assert_allclose(result[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[:3, :3], np.diag([1.0, -1.0, -1.0]))

    def test_world_from_rectified_camera_uses_full_imu_extrinsic(self):
        world_from_head = np.eye(4)
        world_from_head[:3, 3] = [1.0, 2.0, 3.0]
        camera_from_imu = np.eye(4)
        camera_from_imu[:3, 3] = [0.05, -0.02, 0.01]
        result = world_from_rectified_camera_via_calibrated_imu(
            world_from_head, camera_from_imu, np.eye(3)
        )
        np.testing.assert_allclose(result[:3, 3], [0.95, 1.98, 3.01])
        np.testing.assert_allclose(
            result[:3, :3], np.diag([1.0, -1.0, -1.0])
        )


if __name__ == "__main__":
    unittest.main()
