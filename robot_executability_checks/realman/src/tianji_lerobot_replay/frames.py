"""Measured calibration-board, table, and Tianji-base coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ReplayError


# The table and robot placement describe the current *simulation* scene.  The
# source LeRobot poses are UMI gripper-base/TCP poses in the board-centred
# world frame, not IMU poses and not flange poses.
DEFAULT_TABLE_LENGTH_M = 1.40
DEFAULT_TABLE_WIDTH_M = 0.70
DEFAULT_TABLE_HEIGHT_M = 0.80
DEFAULT_TABLETOP_THICKNESS_M = 0.04
DEFAULT_ROBOT_ROOT_IN_TABLE_M = np.array([0.70, -0.20, 1.1785001], dtype=float)
DEFAULT_WORLD_ORIGIN_IN_TABLE_M = np.array([0.05, 0.11, 0.95], dtype=float)
DEFAULT_ROTATION_ROBOT_FROM_TABLE = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float
)
DEFAULT_WORLD_X_DIRECTION_IN_TABLE = np.array([0.0, 1.0, 0.0], dtype=float)
DEFAULT_WORLD_Y_DOWNWARD_HORIZONTAL_DIRECTION_IN_TABLE = np.array(
    [1.0, 0.0, 0.0], dtype=float
)


def _finite_vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ReplayError(f"{name} 必须是三个有限数值。")
    return result


def _proper_rotation(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ReplayError(f"{name} 必须是一个 3×3 有限旋转矩阵。")
    if not np.allclose(result.T @ result, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(result), 1.0, atol=1e-6
    ):
        raise ReplayError(f"{name} 必须是合法 SO(3) 旋转（det=+1）。")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ReplayError(f"{name} 必须是正的有限数值。")
    return result


@dataclass(frozen=True)
class BoardGeometry:
    """The measured 21 cm × 30.2 cm board and its triangular support.

    The cross-section is deliberately modelled from all three measured sides:
    support base = 8 cm, support side = 28.5 cm, board/sloped edge = 30.2 cm.
    They form a small obtuse angle rather than a forced right triangle.  This
    avoids silently throwing away one of the user's measurements.
    """

    panel_width_m: float = 0.210
    sloped_edge_m: float = 0.302
    support_base_m: float = 0.080
    support_side_m: float = 0.285
    origin_to_top_edge_m: float = 0.146
    origin_to_bottom_edge_m: float = 0.156
    panel_thickness_m: float = 0.003

    def __post_init__(self) -> None:
        for name in (
            "panel_width_m",
            "sloped_edge_m",
            "support_base_m",
            "support_side_m",
            "origin_to_top_edge_m",
            "origin_to_bottom_edge_m",
            "panel_thickness_m",
        ):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        if not math.isclose(
            self.origin_to_top_edge_m + self.origin_to_bottom_edge_m,
            self.sloped_edge_m,
            abs_tol=5e-4,
        ):
            raise ReplayError(
                "标定板原点到上/下边距离之和必须等于板面长边："
                f"{self.origin_to_top_edge_m + self.origin_to_bottom_edge_m:.6f} m "
                f"!= {self.sloped_edge_m:.6f} m。"
            )
        # Three-side triangle inequality.  It also guarantees the cosine-law
        # calculation below stays in the physically valid domain.
        base, side, long_edge = (
            self.support_base_m,
            self.support_side_m,
            self.sloped_edge_m,
        )
        if not (base + side > long_edge and base + long_edge > side and side + long_edge > base):
            raise ReplayError("标定板支撑三角形的三条边不满足三角形不等式。")

    @property
    def support_top_horizontal_offset_m(self) -> float:
        """Horizontal position of the top vertex from the rear base vertex.

        In the measured cross-section A--B is the 8 cm base, A--C is the
        28.5 cm support side, and B--C is the 30.2 cm sloped board edge.
        """
        return (
            self.support_base_m**2
            + self.support_side_m**2
            - self.sloped_edge_m**2
        ) / (2.0 * self.support_base_m)

    @property
    def sloped_horizontal_run_m(self) -> float:
        """Horizontal run from the board top edge to its lower edge."""
        return self.support_base_m - self.support_top_horizontal_offset_m

    @property
    def sloped_vertical_run_m(self) -> float:
        value = self.support_side_m**2 - self.support_top_horizontal_offset_m**2
        if value <= 0.0:  # guarded by triangle inequality; protects round-off.
            raise ReplayError("标定板三角形无法计算正的竖直投影。")
        return math.sqrt(value)

    @property
    def support_base_to_side_angle_rad(self) -> float:
        numerator = (
            self.support_base_m**2
            + self.support_side_m**2
            - self.sloped_edge_m**2
        )
        denominator = 2.0 * self.support_base_m * self.support_side_m
        return math.acos(float(np.clip(numerator / denominator, -1.0, 1.0)))

    @property
    def panel_tilt_from_vertical_rad(self) -> float:
        return math.atan2(self.sloped_horizontal_run_m, self.sloped_vertical_run_m)

    @property
    def panel_center_in_umi_m(self) -> np.ndarray:
        # +Y points down the board.  The specified 14.6/15.6 cm offsets mean
        # the coordinate origin is 5 mm above the rectangle's geometric center.
        return np.array(
            [0.0, (self.origin_to_bottom_edge_m - self.origin_to_top_edge_m) / 2.0, 0.0],
            dtype=float,
        )

    @property
    def top_edge_center_in_umi_m(self) -> np.ndarray:
        return np.array([0.0, -self.origin_to_top_edge_m, 0.0], dtype=float)

    @property
    def bottom_edge_center_in_umi_m(self) -> np.ndarray:
        return np.array([0.0, self.origin_to_bottom_edge_m, 0.0], dtype=float)

    def report(self) -> dict[str, Any]:
        return {
            "interpretation": "measured_three_side_obtuse_wedge",
            "panel_width_m": self.panel_width_m,
            "sloped_edge_m": self.sloped_edge_m,
            "support_base_m": self.support_base_m,
            "support_side_m": self.support_side_m,
            "origin_to_top_edge_m": self.origin_to_top_edge_m,
            "origin_to_bottom_edge_m": self.origin_to_bottom_edge_m,
            "panel_thickness_m": self.panel_thickness_m,
            "derived": {
                "support_base_to_side_angle_deg": math.degrees(
                    self.support_base_to_side_angle_rad
                ),
                "sloped_horizontal_run_m": self.sloped_horizontal_run_m,
                "sloped_vertical_run_m": self.sloped_vertical_run_m,
                "panel_tilt_from_vertical_deg": math.degrees(self.panel_tilt_from_vertical_rad),
                "panel_center_in_umi_m": self.panel_center_in_umi_m.tolist(),
                "top_edge_center_in_umi_m": self.top_edge_center_in_umi_m.tolist(),
                "bottom_edge_center_in_umi_m": self.bottom_edge_center_in_umi_m.tolist(),
            },
        }


@dataclass(frozen=True)
class SceneConfig:
    """Current simulation scene and full recorded-world → Tianji-base chain.

    ``world`` is the UMI/LeRobot coordinate frame whose origin lies on the
    calibration board, not the board's geometric centre.  It has +X to the
    right across the 21 cm board, +Y downward along its 30.2 cm sloped edge,
    and +Z into the board.  TCP means the UMI gripper URDF ``base_link`` and
    maps directly to Tianji ``gripper_base_link`` on both sides.
    """

    table_length_m: float = DEFAULT_TABLE_LENGTH_M
    table_width_m: float = DEFAULT_TABLE_WIDTH_M
    table_height_m: float = DEFAULT_TABLE_HEIGHT_M
    tabletop_thickness_m: float = DEFAULT_TABLETOP_THICKNESS_M
    robot_root_in_table_m: np.ndarray = field(
        default_factory=lambda: DEFAULT_ROBOT_ROOT_IN_TABLE_M.copy()
    )
    world_origin_in_table_m: np.ndarray = field(
        default_factory=lambda: DEFAULT_WORLD_ORIGIN_IN_TABLE_M.copy()
    )
    world_x_direction_in_table: np.ndarray = field(
        default_factory=lambda: DEFAULT_WORLD_X_DIRECTION_IN_TABLE.copy()
    )
    world_y_downward_horizontal_direction_in_table: np.ndarray = field(
        default_factory=lambda: DEFAULT_WORLD_Y_DOWNWARD_HORIZONTAL_DIRECTION_IN_TABLE.copy()
    )
    board: BoardGeometry = field(default_factory=BoardGeometry)
    rotation_robot_from_table: np.ndarray = field(
        default_factory=lambda: DEFAULT_ROTATION_ROBOT_FROM_TABLE.copy()
    )

    def __post_init__(self) -> None:
        for name in (
            "table_length_m",
            "table_width_m",
            "table_height_m",
            "tabletop_thickness_m",
        ):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        if self.tabletop_thickness_m >= self.table_height_m:
            raise ReplayError("tabletop_thickness_m 必须小于 table_height_m。")
        object.__setattr__(
            self,
            "robot_root_in_table_m",
            _finite_vector(self.robot_root_in_table_m, "robot_root_in_table_m"),
        )
        object.__setattr__(
            self,
            "world_origin_in_table_m",
            _finite_vector(self.world_origin_in_table_m, "world_origin_in_table_m"),
        )
        world_x = _finite_vector(self.world_x_direction_in_table, "world_x_direction_in_table")
        world_y_horizontal = _finite_vector(
            self.world_y_downward_horizontal_direction_in_table,
            "world_y_downward_horizontal_direction_in_table",
        )
        if not math.isclose(float(np.linalg.norm(world_x)), 1.0, abs_tol=1e-6):
            raise ReplayError("world_x_direction_in_table 必须是单位向量。")
        if not math.isclose(float(np.linalg.norm(world_y_horizontal)), 1.0, abs_tol=1e-6):
            raise ReplayError("world_y_downward_horizontal_direction_in_table 必须是单位向量。")
        if abs(float(world_x[2])) > 1e-6 or abs(float(world_y_horizontal[2])) > 1e-6:
            raise ReplayError("世界 +X 和 +Y 的水平投影必须位于桌面平面（table z=0）。")
        if abs(float(np.dot(world_x, world_y_horizontal))) > 1e-6:
            raise ReplayError("world_x_direction_in_table 与世界 +Y 的水平投影必须正交。")
        object.__setattr__(self, "world_x_direction_in_table", world_x)
        object.__setattr__(
            self,
            "world_y_downward_horizontal_direction_in_table",
            world_y_horizontal,
        )
        if not isinstance(self.board, BoardGeometry):
            raise ReplayError("board 必须是 BoardGeometry。")
        object.__setattr__(
            self,
            "rotation_robot_from_table",
            _proper_rotation(self.rotation_robot_from_table, "rotation_robot_from_table"),
        )

    @property
    def rotation_table_from_umi(self) -> np.ndarray:
        """Measured board axes expressed in table coordinates.

        All three triangle side lengths determine the +Y-down vector.  The
        third world axis is derived by a right-handed cross product.
        """
        down = (
            self.board.sloped_horizontal_run_m / self.board.sloped_edge_m
        ) * self.world_y_downward_horizontal_direction_in_table + np.array(
            [0.0, 0.0, -self.board.sloped_vertical_run_m / self.board.sloped_edge_m],
            dtype=float,
        )
        inward = np.cross(self.world_x_direction_in_table, down)
        return _proper_rotation(
            np.column_stack((self.world_x_direction_in_table, down, inward)),
            "derived rotation_table_from_umi",
        )

    @property
    def translation_robot_from_table_m(self) -> np.ndarray:
        return -self.rotation_robot_from_table @ self.robot_root_in_table_m

    @property
    def rotation_robot_from_umi(self) -> np.ndarray:
        return self.rotation_robot_from_table @ self.rotation_table_from_umi

    @property
    def translation_robot_from_umi_m(self) -> np.ndarray:
        return (
            self.rotation_robot_from_table @ self.world_origin_in_table_m
            + self.translation_robot_from_table_m
        )

    def table_from_umi_positions(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ReplayError("UMI TCP 位置必须是最后一维为 3 的有限数组。")
        return (
            np.einsum("ij,...j->...i", self.rotation_table_from_umi, values)
            + self.world_origin_in_table_m
        )

    def robot_from_table_positions(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ReplayError("桌面位置必须是最后一维为 3 的有限数组。")
        return (
            np.einsum("ij,...j->...i", self.rotation_robot_from_table, values)
            + self.translation_robot_from_table_m
        )

    def robot_from_umi_positions(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ReplayError("UMI TCP 位置必须是最后一维为 3 的有限数组。")
        return (
            np.einsum("ij,...j->...i", self.rotation_robot_from_umi, values)
            + self.translation_robot_from_umi_m
        )

    def robot_from_umi_rotations(self, rotations: np.ndarray) -> np.ndarray:
        values = np.asarray(rotations, dtype=float)
        if values.shape[-2:] != (3, 3) or not np.all(np.isfinite(values)):
            raise ReplayError("UMI TCP 姿态必须是最后两维为 3×3 的有限数组。")
        return np.einsum("ij,...jk->...ik", self.rotation_robot_from_umi, values)

    def support_triangle_vertices_table_m(self) -> dict[str, np.ndarray]:
        """Return top/lower/rear vertices of the measured board support."""
        lower = self.table_from_umi_positions(self.board.bottom_edge_center_in_umi_m)
        top = self.table_from_umi_positions(self.board.top_edge_center_in_umi_m)
        rear_base = (
            lower
            - self.board.support_base_m * self.world_y_downward_horizontal_direction_in_table
        )
        return {"top": top, "lower": lower, "rear_base": rear_base}

    def report(self) -> dict[str, Any]:
        support = self.support_triangle_vertices_table_m()
        return {
            "input_pose_convention": "world_from_umi_gripper_base_tcp",
            "tcp_alignment_to_tianji_gripper_base": "identity_for_left_and_right",
            "table": {
                "length_m": self.table_length_m,
                "width_m": self.table_width_m,
                "height_m": self.table_height_m,
                "top_thickness_m": self.tabletop_thickness_m,
            },
            "world_origin_in_table_m": self.world_origin_in_table_m.tolist(),
            "rotation_table_from_umi": self.rotation_table_from_umi.tolist(),
            "rotation_robot_from_table": self.rotation_robot_from_table.tolist(),
            "translation_robot_from_table_m": self.translation_robot_from_table_m.tolist(),
            "robot_root_in_table_m": self.robot_root_in_table_m.tolist(),
            "rotation_robot_from_umi": self.rotation_robot_from_umi.tolist(),
            "translation_robot_from_umi_m": self.translation_robot_from_umi_m.tolist(),
            "calibration_board": self.board.report(),
            "support_triangle_vertices_table_m": {
                name: point.tolist() for name, point in support.items()
            },
            "axis_definition": {
                "umi_x": "right across the 21 cm board; configured as table +y",
                "umi_y": "down the measured 30.2 cm sloped board edge",
                "umi_z": "into the board; right-handed +x cross +y",
            },
        }


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized or unnormalized xyzw quaternions to rotations."""
    values = np.asarray(quaternion, dtype=float)
    if values.shape[-1:] != (4,):
        raise ReplayError(f"四元数最后一维必须为 4，实际为 {values.shape}。")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-10):
        raise ReplayError("TCP 轨迹含有无效四元数。")
    x, y, z, w = np.moveaxis(values / norm, -1, 0)
    result = np.empty(values.shape[:-1] + (3, 3), dtype=float)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to a normalized xyzw quaternion."""
    matrix = _proper_rotation(rotation, "rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ]
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ]
        )
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    return quat / np.linalg.norm(quat)


def load_scene_config(path: str | Path) -> SceneConfig:
    """Load the measured-board scene JSON and reject ambiguous legacy fields."""
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"无法读取场景配置 {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayError("场景配置根节点必须是 JSON 对象。")
    try:
        table = payload["table"]
        robot = payload["robot"]
        umi = payload["umi_world"]
        board = payload["calibration_board"]
        if not all(isinstance(item, dict) for item in (table, robot, umi, board)):
            raise TypeError("table/robot/umi_world/calibration_board 必须是对象")
        interpretation = board.get("interpretation")
        if interpretation != "measured_three_side_obtuse_wedge":
            raise ReplayError(
                "calibration_board.interpretation 必须为 "
                "'measured_three_side_obtuse_wedge'，避免把 8/28.5/30.2 cm "
                "误当成直角三角形。"
            )
        return SceneConfig(
            table_length_m=float(table["length_m"]),
            table_width_m=float(table["width_m"]),
            table_height_m=float(table["height_m"]),
            tabletop_thickness_m=float(table["top_thickness_m"]),
            robot_root_in_table_m=np.asarray(robot["root_in_table_m"], dtype=float),
            world_origin_in_table_m=np.asarray(umi["world_origin_in_table_m"], dtype=float),
            world_x_direction_in_table=np.asarray(umi["world_x_direction_in_table"], dtype=float),
            world_y_downward_horizontal_direction_in_table=np.asarray(
                umi["world_y_downward_horizontal_direction_in_table"], dtype=float
            ),
            board=BoardGeometry(
                panel_width_m=float(board["panel_width_m"]),
                sloped_edge_m=float(board["sloped_edge_m"]),
                support_base_m=float(board["support_base_m"]),
                support_side_m=float(board["support_side_m"]),
                origin_to_top_edge_m=float(board["origin_to_top_edge_m"]),
                origin_to_bottom_edge_m=float(board["origin_to_bottom_edge_m"]),
                panel_thickness_m=float(board.get("panel_thickness_m", 0.003)),
            ),
            rotation_robot_from_table=np.asarray(robot["rotation_robot_from_table"], dtype=float),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayError(
            f"场景配置 {source} 缺少或包含无效 table/robot/umi_world/calibration_board 字段：{exc}"
        ) from exc
