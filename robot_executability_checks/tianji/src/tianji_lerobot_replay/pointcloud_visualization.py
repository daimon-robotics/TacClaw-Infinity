"""Frame-synchronized stereo point clouds for MuJoCo workspace videos."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .frames import SceneConfig, quaternion_xyzw_to_matrix


_DEPTH_FRAME = re.compile(r".+_frame(\d+)_depth\.npy$")

# ``head_*`` is the SXR/OpenXR head pose.  Its local axes are x-right,
# y-up, z-backward, whereas OpenCV optical coordinates are x-right, y-down,
# z-forward.  It must not be interpreted as the RGB-camera IMU pose.
OPENXR_HEAD_FROM_OPENCV_OPTICAL = np.diag([1.0, -1.0, -1.0])


def backproject_depth(
    depth_m: np.ndarray,
    fx_fy_cx_cy: tuple[float, float, float, float] | list[float],
) -> np.ndarray:
    """Back-project a metric depth image into its rectified camera frame."""
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    fx, fy, cx, cy = map(float, fx_fy_cx_cy)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    yy, xx = np.indices(depth.shape, dtype=float)
    return np.stack(
        ((xx - cx) * depth / fx, (yy - cy) * depth / fy, depth), axis=-1
    )


def world_from_rectified_camera(
    world_from_head: np.ndarray,
    rectified_from_camera_rotation: np.ndarray,
) -> np.ndarray:
    """Return world-from-rectified-camera for an SXR/OpenXR head pose.

    The dataset's ``head_*`` pose is an OpenXR head local coordinate system,
    not an IMU LCS.  Rectification first maps the raw OpenCV camera frame to
    the rectified frame; the fixed OpenCV-to-OpenXR axis conversion is then
    applied explicitly.  The head/camera origin offset is not present in the
    LeRobot metadata and is therefore kept at zero rather than guessed.
    """
    world_from_head = np.asarray(world_from_head, dtype=float)
    rotation = np.asarray(rectified_from_camera_rotation, dtype=float)
    if world_from_head.shape != (4, 4):
        raise ValueError("world_from_head must be 4x4")
    if rotation.shape != (3, 3):
        raise ValueError("rectification rotation must be 3x3")
    head_from_rectified = np.eye(4)
    head_from_rectified[:3, :3] = (
        OPENXR_HEAD_FROM_OPENCV_OPTICAL @ rotation.T
    )
    return world_from_head @ head_from_rectified


def world_from_rectified_camera_via_calibrated_imu(
    world_from_head: np.ndarray,
    camera_from_imu: np.ndarray,
    rectified_from_camera_rotation: np.ndarray,
) -> np.ndarray:
    """Attach the RGB camera to an OpenXR head pose through its IMU calibration.

    ``head_pose`` uses OpenXR local axes while the calibration matrix uses the
    hardware IMU axes. The head and IMU origins are treated as coincident, but
    their axes differ. Deriving ``head_from_imu`` from the calibrated camera
    rotation preserves the OpenXR optical-axis convention and retains the
    measured camera/IMU lever-arm translation.
    """
    world_from_head = np.asarray(world_from_head, dtype=float)
    camera_from_imu = np.asarray(camera_from_imu, dtype=float)
    rectified_from_camera_rotation = np.asarray(
        rectified_from_camera_rotation, dtype=float
    )
    if world_from_head.shape != (4, 4) or camera_from_imu.shape != (4, 4):
        raise ValueError("pose and camera extrinsic must both be 4x4")
    if rectified_from_camera_rotation.shape != (3, 3):
        raise ValueError("rectification rotation must be 3x3")
    camera_from_rectified = np.eye(4)
    camera_from_rectified[:3, :3] = rectified_from_camera_rotation.T
    head_from_imu = np.eye(4)
    head_from_imu[:3, :3] = (
        OPENXR_HEAD_FROM_OPENCV_OPTICAL @ camera_from_imu[:3, :3]
    )
    return (
        world_from_head
        @ head_from_imu
        @ np.linalg.inv(camera_from_imu)
        @ camera_from_rectified
    )


def _episode_states(dataset: Path, episode: int) -> tuple[dict[int, dict[str, float]], dict]:
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    states: dict[int, dict[str, float]] = {}
    for path in sorted((dataset / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "observation.state"]
        )
        for row in table.to_pylist():
            if int(row["episode_index"]) != episode:
                continue
            states[int(row["frame_index"])] = {
                name: float(value) for name, value in zip(names, row["observation.state"])
            }
    if not states:
        raise ValueError(f"dataset has no state rows for episode {episode}: {dataset}")
    return states, info


class StereoPointCloudOverlay:
    """Load sparse Fast-FoundationStereo frames and append RGB points to an MjvScene."""

    def __init__(
        self,
        depth_run: str | Path,
        dataset: str | Path,
        episode: int,
        config: SceneConfig,
        *,
        max_points: int = 2200,
        voxel_size_m: float = 0.015,
        point_radius_m: float = 0.005,
        min_depth_m: float = 0.18,
        max_depth_m: float = 1.80,
        color_saturation: float = 1.0,
        color_gamma: float = 1.0,
        color_priority_fraction: float = 0.0,
        min_color_luminance: float = 0.08,
        workcell_xy_margin_m: float = 0.03,
        max_height_above_table_m: float = 0.55,
        hide_table_surface: bool = False,
        tcp_neighborhood_radius_m: float | None = None,
        pose_file: str | Path | None = None,
        stabilize_static_surfaces: bool = False,
    ) -> None:
        self.depth_run = Path(depth_run).expanduser().resolve()
        self.dataset = Path(dataset).expanduser().resolve()
        self.episode = int(episode)
        self.config = config
        self.max_points = int(max_points)
        self.voxel_size_m = float(voxel_size_m)
        self.point_radius_m = float(point_radius_m)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.color_saturation = float(color_saturation)
        self.color_gamma = float(color_gamma)
        self.color_priority_fraction = float(color_priority_fraction)
        self.min_color_luminance = float(min_color_luminance)
        self.workcell_xy_margin_m = float(workcell_xy_margin_m)
        self.max_height_above_table_m = float(max_height_above_table_m)
        self.hide_table_surface = bool(hide_table_surface)
        self.tcp_neighborhood_radius_m = (
            None
            if tcp_neighborhood_radius_m is None
            else float(tcp_neighborhood_radius_m)
        )
        self.stabilize_static_surfaces = bool(stabilize_static_surfaces)
        if self.max_points <= 0 or self.voxel_size_m <= 0.0 or self.point_radius_m <= 0.0:
            raise ValueError("point-cloud size parameters must be positive")
        if self.color_saturation <= 0.0 or self.color_gamma <= 0.0:
            raise ValueError("point-cloud color parameters must be positive")
        if not 0.0 <= self.color_priority_fraction <= 1.0:
            raise ValueError("color_priority_fraction must be in [0, 1]")
        if not 0.0 <= self.min_color_luminance <= 1.0:
            raise ValueError("min_color_luminance must be in [0, 1]")
        if self.workcell_xy_margin_m < 0.0 or self.max_height_above_table_m <= 0.0:
            raise ValueError("point-cloud workcell bounds must be positive")
        if (
            self.tcp_neighborhood_radius_m is not None
            and self.tcp_neighborhood_radius_m <= 0.0
        ):
            raise ValueError("tcp_neighborhood_radius_m must be positive")

        report = json.loads((self.depth_run / "run_report.json").read_text(encoding="utf-8"))
        scale = float(report["scale"])
        self.intrinsics = np.asarray(report["rectified_roi_fx_fy_cx_cy"], dtype=float) * np.array(
            [scale, scale, scale, scale], dtype=float
        )
        self.rectified_from_camera = np.asarray(
            report["rectification_rotation_cam0_to_rectified"], dtype=float
        )
        self.states, dataset_info = _episode_states(self.dataset, self.episode)
        self.world_from_head: dict[int, np.ndarray] | None = None
        dataset_camera_from_imu = dataset_info.get("features", {}).get(
            "observation.images.ego_left", {}
        ).get("imu_to_rgb_camera")
        self.camera_from_imu: np.ndarray | None = (
            np.asarray(dataset_camera_from_imu, dtype=float)
            if dataset_camera_from_imu is not None
            else None
        )
        if pose_file is not None:
            pose_path = Path(pose_file).expanduser().resolve()
            pose_data = np.load(pose_path)
            frame_indices = np.asarray(pose_data["frame_index"], dtype=int)
            transforms = np.asarray(pose_data["world_from_head"], dtype=float)
            if transforms.shape != (len(frame_indices), 4, 4):
                raise ValueError(f"invalid world_from_head shape in {pose_path}")
            self.world_from_head = {
                int(frame): transform
                for frame, transform in zip(frame_indices, transforms)
            }
            calibration = json.loads(
                (pose_path.parent / "calibration.json").read_text(encoding="utf-8")
            )
            self.camera_from_imu = np.asarray(
                calibration["observation"]["images"]["rgb"]["extrinsics"][
                    "T_ic_imu0_cam0"
                ],
                dtype=float,
            )

            ego_world_to_table_path = pose_path.parent / "ego_world_to_table.json"
            self.ego_world_to_table: np.ndarray | None = None
            if ego_world_to_table_path.is_file():
                override = json.loads(ego_world_to_table_path.read_text(encoding="utf-8"))
                matrix = np.asarray(
                    override["table_from_ego_world_4x4"], dtype=float
                )
                if matrix.shape != (4, 4):
                    raise ValueError(
                        f"table_from_ego_world_4x4 in {ego_world_to_table_path} must be 4x4"
                    )
                self.ego_world_to_table = matrix
        else:
            self.ego_world_to_table = None

        self.depth_paths: dict[int, Path] = {}
        for path in sorted((self.depth_run / "depth_npy").glob("*_depth.npy")):
            match = _DEPTH_FRAME.match(path.name)
            if match:
                self.depth_paths[int(match.group(1)) - 1] = path
        if not self.depth_paths:
            raise FileNotFoundError(f"no depth NPY frames under {self.depth_run / 'depth_npy'}")
        self.available_frames = np.asarray(sorted(self.depth_paths), dtype=int)
        self._cache_frame = -1
        self._cache_points = np.empty((0, 3), dtype=np.float32)
        self._cache_rgba = np.empty((0, 4), dtype=np.float32)
        self._static_points_table = np.empty((0, 3), dtype=np.float32)
        self._static_colors = np.empty((0, 3), dtype=np.float32)
        if self.stabilize_static_surfaces:
            if self.world_from_head is None:
                raise ValueError("static-surface stabilization requires a raw Ego pose file")
            self._build_static_surface_map()

    @property
    def cadence_frames(self) -> int:
        if len(self.available_frames) < 2:
            return 1
        return int(np.median(np.diff(self.available_frames)))

    def nearest_frame(self, frame_index: int) -> int:
        index = int(np.searchsorted(self.available_frames, int(frame_index)))
        choices = []
        if index < len(self.available_frames):
            choices.append(int(self.available_frames[index]))
        if index > 0:
            choices.append(int(self.available_frames[index - 1]))
        return min(choices, key=lambda value: (abs(value - frame_index), value))

    def _frame_arrays(
        self, source_frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = np.load(self.depth_paths[source_frame])
        xyz_rectified = backproject_depth(depth, self.intrinsics)
        stem = f"rgb_frame{source_frame + 1:06d}"
        color_path = self.depth_run / "registered_rgb" / f"{stem}_registered.png"
        color_mask_path = self.depth_run / "registered_rgb_mask" / f"{stem}_valid.png"
        if not color_path.exists() or not color_mask_path.exists():
            raise FileNotFoundError(
                "point-cloud RGB must be explicitly registered to depth; missing "
                f"{color_path} or {color_mask_path}"
            )
        rgb = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.float32) / 255.0
        color_valid = np.asarray(Image.open(color_mask_path).convert("L")) > 0
        if rgb.shape[:2] != depth.shape or color_valid.shape != depth.shape:
            raise ValueError("registered RGB, mask and depth shapes must match")
        points = xyz_rectified.reshape(-1, 3)
        colors = rgb.reshape(-1, 3)
        valid = (
            np.isfinite(points).all(axis=1)
            & (points[:, 2] >= self.min_depth_m)
            & (points[:, 2] <= self.max_depth_m)
            & color_valid.reshape(-1)
        )
        return points[valid], colors[valid], depth

    def _camera_transform(self, source_frame: int) -> np.ndarray:
        if self.world_from_head is not None:
            if source_frame not in self.world_from_head:
                raise KeyError(f"raw Ego pose file has no frame {source_frame}")
            assert self.camera_from_imu is not None
            return world_from_rectified_camera_via_calibrated_imu(
                self.world_from_head[source_frame],
                self.camera_from_imu,
                self.rectified_from_camera,
            )
        state = self.states[source_frame]
        world_from_head = np.eye(4)
        world_from_head[:3, 3] = [
            state["head_x"], state["head_y"], state["head_z"]
        ]
        world_from_head[:3, :3] = quaternion_xyzw_to_matrix(
            np.asarray(
                [
                    state["head_qx"], state["head_qy"],
                    state["head_qz"], state["head_qw"],
                ]
            )
        )
        if self.camera_from_imu is not None:
            return world_from_rectified_camera_via_calibrated_imu(
                world_from_head,
                self.camera_from_imu,
                self.rectified_from_camera,
            )
        return world_from_rectified_camera(world_from_head, self.rectified_from_camera)

    def _table_from_world_points(self, points_world: np.ndarray) -> np.ndarray:
        """Map raw-world points to table coordinates.

        Uses a directly-fitted ego-camera-to-table transform when one is
        available (raw Ego head tracking has its own independent VIO world
        frame, distinct from the gripper's calibration-sheet-anchored world),
        falling back to the shared UMI-world convention otherwise.
        """
        if self.ego_world_to_table is not None:
            homogeneous = np.concatenate(
                (points_world, np.ones((len(points_world), 1))), axis=1
            )
            return (homogeneous @ self.ego_world_to_table.T)[:, :3]
        return self.config.table_from_umi_positions(points_world)

    def _static_masks(
        self,
        points_world: np.ndarray,
        points_table: np.ndarray,
        colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        chroma = np.ptp(colors, axis=1)
        luminance = colors.mean(axis=1)
        neutral_bright = (chroma < 0.18) & (luminance > 0.42)
        table = (
            neutral_bright
            & (points_table[:, 0] >= 0.0)
            & (points_table[:, 0] <= self.config.table_length_m)
            & (points_table[:, 1] >= 0.0)
            & (points_table[:, 1] <= self.config.table_width_m)
            & (np.abs(points_table[:, 2] - self.config.table_height_m) <= 0.10)
        )
        board = (
            neutral_bright
            & (np.abs(points_world[:, 0]) <= self.config.board.panel_width_m / 2.0 + 0.04)
            & (points_world[:, 1] >= -self.config.board.origin_to_top_edge_m - 0.04)
            & (points_world[:, 1] <= self.config.board.origin_to_bottom_edge_m + 0.04)
            & (np.abs(points_world[:, 2]) <= 0.08)
        )
        return table, board

    def _build_static_surface_map(self) -> None:
        points_parts: list[np.ndarray] = []
        color_parts: list[np.ndarray] = []
        for source_frame in self.available_frames:
            points, colors, _depth = self._frame_arrays(int(source_frame))
            transform = self._camera_transform(int(source_frame))
            points_world = points @ transform[:3, :3].T + transform[:3, 3]
            points_table = self._table_from_world_points(points_world)
            table_mask, board_mask = self._static_masks(
                points_world, points_table, colors
            )
            table_points = points_table[table_mask].copy()
            table_points[:, 2] = self.config.table_height_m
            board_world = points_world[board_mask].copy()
            board_world[:, 2] = 0.0
            board_points = self.config.table_from_umi_positions(board_world)
            frame_points = np.concatenate((table_points, board_points), axis=0)
            frame_colors = np.concatenate(
                (colors[table_mask], colors[board_mask]), axis=0
            )
            if len(frame_points) > 3000:
                keep = np.linspace(0, len(frame_points) - 1, 3000, dtype=int)
                frame_points = frame_points[keep]
                frame_colors = frame_colors[keep]
            points_parts.append(frame_points)
            color_parts.append(frame_colors)
        if not points_parts:
            return
        points = np.concatenate(points_parts, axis=0)
        colors = np.concatenate(color_parts, axis=0)
        voxels = np.floor(points / self.voxel_size_m).astype(np.int64)
        _, indices = np.unique(voxels, axis=0, return_index=True)
        static_budget = max(1, int(round(self.max_points * 0.45)))
        if len(indices) > static_budget:
            indices = indices[
                np.linspace(0, len(indices) - 1, static_budget, dtype=int)
            ]
        self._static_points_table = points[indices].astype(np.float32)
        self._static_colors = colors[indices].astype(np.float32)

    def points_for_frame(self, frame_index: int) -> tuple[np.ndarray, np.ndarray, int]:
        source_frame = self.nearest_frame(frame_index)
        if source_frame == self._cache_frame:
            return self._cache_points, self._cache_rgba, source_frame
        points, colors, _depth = self._frame_arrays(source_frame)
        transform = self._camera_transform(source_frame)
        points_world = points @ transform[:3, :3].T + transform[:3, 3]
        points_table = self._table_from_world_points(points_world)
        keep_dynamic = np.ones(len(points_world), dtype=bool)
        if self.tcp_neighborhood_radius_m is not None:
            state = self.states[source_frame]
            tcp_world = np.asarray(
                [
                    [state["left_x"], state["left_y"], state["left_z"]],
                    [state["right_x"], state["right_y"], state["right_z"]],
                ],
                dtype=float,
            )
            distance = np.linalg.norm(
                points_world[:, None, :] - tcp_world[None, :, :], axis=2
            ).min(axis=1)
            keep_dynamic &= distance <= self.tcp_neighborhood_radius_m
        if self.stabilize_static_surfaces:
            table_mask, board_mask = self._static_masks(
                points_world, points_table, colors
            )
            keep_dynamic &= ~(table_mask | board_mask)
        points_table = points_table[keep_dynamic]
        colors = colors[keep_dynamic]
        luminance = colors @ np.asarray([0.2126, 0.7152, 0.0722])
        margin = self.workcell_xy_margin_m
        # MuJoCo already renders the table.  Remove the complete stereo slab,
        # including colored registration fringes and dark disparity speckles,
        # instead of trying to recognize it from RGB alone.
        table_surface = (
            np.abs(points_table[:, 2] - self.config.table_height_m) <= 0.10
        )
        workcell = (
            (points_table[:, 0] >= -margin)
            & (points_table[:, 0] <= self.config.table_length_m + margin)
            & (points_table[:, 1] >= -margin)
            & (points_table[:, 1] <= self.config.table_width_m + margin)
            # The recorded workcell has no scene geometry below the tabletop.
            # A narrow tolerance retains stereo noise without rendering the
            # invalid black edge streaks below the simulated table.
            & (points_table[:, 2] >= self.config.table_height_m - 0.025)
            & (
                points_table[:, 2]
                <= self.config.table_height_m + self.max_height_above_table_m
            )
            # Reject the black enclosure, hair and invalid dark stereo speckles.
            & (luminance >= self.min_color_luminance)
        )
        if self.hide_table_surface:
            workcell &= ~table_surface
        points_table = points_table[workcell]
        colors = colors[workcell]
        if len(points_table):
            voxels = np.floor(points_table / self.voxel_size_m).astype(np.int64)
            _, indices = np.unique(voxels, axis=0, return_index=True)
            indices.sort()
            dynamic_budget = self.max_points - len(self._static_points_table)
            if len(indices) > dynamic_budget:
                priority_count = int(round(dynamic_budget * self.color_priority_fraction))
                if priority_count:
                    voxel_colors = colors[indices]
                    chroma = np.ptp(voxel_colors, axis=1)
                    priority_local = np.argpartition(chroma, -priority_count)[-priority_count:]
                    priority_local.sort()
                    remaining_mask = np.ones(len(indices), dtype=bool)
                    remaining_mask[priority_local] = False
                    remaining_local = np.flatnonzero(remaining_mask)
                    uniform_count = dynamic_budget - len(priority_local)
                    uniform_local = remaining_local[
                        np.linspace(
                            0, len(remaining_local) - 1, uniform_count, dtype=int
                        )
                    ]
                    indices = indices[np.sort(np.concatenate((priority_local, uniform_local)))]
                else:
                    indices = indices[
                        np.linspace(0, len(indices) - 1, dynamic_budget, dtype=int)
                    ]
            points_table = points_table[indices]
            colors = colors[indices]
        if len(self._static_points_table):
            points_table = np.concatenate(
                (self._static_points_table, points_table), axis=0
            )
            colors = np.concatenate((self._static_colors, colors), axis=0)
        if len(colors):
            # Display-only sRGB enhancement. It preserves hue and leaves the
            # registered RGB files untouched while making small colored objects
            # visible against the mostly black/white capture enclosure.
            luminance = colors @ np.asarray([0.2126, 0.7152, 0.0722])
            colors = np.clip(
                luminance[:, None]
                + self.color_saturation * (colors - luminance[:, None]),
                0.0,
                1.0,
            )
            colors = np.power(colors, self.color_gamma)
        points_robot = self.config.robot_from_table_positions(points_table).astype(np.float32)
        alpha = np.ones((len(colors), 1), dtype=np.float32)
        self._cache_frame = source_frame
        self._cache_points = points_robot
        self._cache_rgba = np.concatenate((colors.astype(np.float32), alpha), axis=1)
        return self._cache_points, self._cache_rgba, source_frame

    def update_user_scene(self, scene: object, frame_index: int) -> dict[str, int]:
        import mujoco

        points, rgba, source_frame = self.points_for_frame(frame_index)
        start = int(scene.ngeom)
        displayed = min(len(points), max(0, int(scene.maxgeom) - start))
        scene.ngeom = start + displayed
        size = np.full(3, self.point_radius_m, dtype=float)
        identity = np.eye(3).reshape(-1)
        for index in range(displayed):
            mujoco.mjv_initGeom(
                scene.geoms[start + index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                points[index],
                identity,
                rgba[index],
            )
            # Preserve the registered camera colors instead of allowing scene
            # lighting to turn most RGB samples into dark grey.
            scene.geoms[start + index].emission = 0.80
            scene.geoms[start + index].specular = 0.0
            scene.geoms[start + index].shininess = 0.0
        return {"requested": len(points), "displayed": displayed, "source_frame": source_frame}
