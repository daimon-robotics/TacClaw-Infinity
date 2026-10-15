"""Build the visual-only Tianji, white-table, and calibration-sheet MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import ReplayError
from .frames import SceneConfig, matrix_to_quaternion_xyzw
from .model import prepare_urdf


TABLETOP_RGBA = np.array([0.96, 0.96, 0.96, 1.0], dtype=float)
TABLE_LEG_RGBA = np.array([0.82, 0.82, 0.84, 1.0], dtype=float)
TARGET_RGBA = {
    "left": np.array([0.95, 0.20, 0.20, 0.72], dtype=float),
    "right": np.array([0.18, 0.42, 1.00, 0.72], dtype=float),
}


@dataclass(frozen=True)
class ReplayScene:
    xml_path: Path
    model: object
    target_mocap_ids: dict[str, int]


def build_scene(
    urdf: str | Path,
    output_dir: str | Path,
    config: SceneConfig,
    package_roots: Iterable[str | Path] = (),
) -> ReplayScene:
    """Compile a self-contained XML scene; all table geometry is visual only."""
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ReplayError("缺少 mujoco；请先运行 ./setup.sh。") from exc

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare_urdf(
        urdf,
        output / "tianji_prepared.urdf",
        package_roots,
    )
    spec = mujoco.MjSpec.from_file(str(prepared))

    table_quat_xyzw = matrix_to_quaternion_xyzw(config.rotation_robot_from_table)
    table_quat_wxyz = np.array(
        [table_quat_xyzw[3], table_quat_xyzw[0], table_quat_xyzw[1], table_quat_xyzw[2]]
    )

    def add_box(
        name: str,
        table_position: Iterable[float],
        half_size: Iterable[float],
        rgba: Iterable[float],
    ) -> None:
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = config.robot_from_table_positions(np.asarray(table_position, dtype=float))
        geom.quat = table_quat_wxyz
        geom.size = np.asarray(half_size, dtype=float)
        geom.rgba = np.asarray(rgba, dtype=float)
        # Replay-only: no collision contact or reachability classification.
        geom.contype = 0
        geom.conaffinity = 0

    top_half_z = config.tabletop_thickness_m / 2.0
    add_box(
        "table_top_visual",
        (
            config.table_length_m / 2.0,
            config.table_width_m / 2.0,
            config.table_height_m - top_half_z,
        ),
        (config.table_length_m / 2.0, config.table_width_m / 2.0, top_half_z),
        TABLETOP_RGBA,
    )
    leg_half = 0.03
    leg_half_z = (config.table_height_m - config.tabletop_thickness_m) / 2.0
    for x in (0.05, config.table_length_m - 0.05):
        for y in (0.05, config.table_width_m - 0.05):
            add_box(
                f"table_leg_visual_{x:.2f}_{y:.2f}",
                (x, y, leg_half_z),
                (leg_half, leg_half, leg_half_z),
                TABLE_LEG_RGBA,
            )

    floor = spec.worldbody.add_geom()
    floor.name = "floor_visual"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = config.robot_from_table_positions(np.zeros(3))
    floor.quat = table_quat_wxyz
    floor.size = np.array([5.0, 5.0, 0.1])
    floor.rgba = np.array([0.22, 0.24, 0.28, 1.0])
    floor.contype = 0
    floor.conaffinity = 0

    _add_calibration_board_visual(spec, mujoco, config)
    _add_target_mocaps(spec, mujoco)

    light = spec.worldbody.add_light()
    light.name = "scene_light"
    light.pos = np.array([0.0, 0.0, 2.0])
    light.dir = np.array([0.0, 0.0, -1.0])
    light.diffuse = np.array([0.85, 0.85, 0.85])

    # MjSpec writes XML only after a successful compile.
    spec.compile()
    xml_path = output / "scene.xml"
    spec.to_file(str(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    target_mocap_ids: dict[str, int] = {}
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"umi_target_{side}")
        if body_id < 0 or model.body_mocapid[body_id] < 0:
            raise ReplayError(f"生成的场景缺少 {side} TCP target mocap body。")
        target_mocap_ids[side] = int(model.body_mocapid[body_id])
    return ReplayScene(xml_path=xml_path, model=model, target_mocap_ids=target_mocap_ids)


def _add_calibration_board_visual(spec: object, mujoco: object, config: SceneConfig) -> None:
    """Render the measured 21 cm × 30.2 cm board and its wedge outline."""
    board = config.board
    board_rotation_robot = config.rotation_robot_from_umi
    board_quat_xyzw = matrix_to_quaternion_xyzw(board_rotation_robot)
    board_panel = spec.worldbody.add_geom()
    board_panel.name = "umi_calibration_board_visual"
    board_panel.type = mujoco.mjtGeom.mjGEOM_BOX
    board_panel.pos = config.robot_from_umi_positions(board.panel_center_in_umi_m)
    board_panel.quat = np.array(
        [board_quat_xyzw[3], board_quat_xyzw[0], board_quat_xyzw[1], board_quat_xyzw[2]]
    )
    board_panel.size = np.array(
        [board.panel_width_m / 2.0, board.sloped_edge_m / 2.0, board.panel_thickness_m / 2.0]
    )
    board_panel.rgba = np.array([1.0, 0.90, 0.20, 0.34])
    board_panel.contype = 0
    board_panel.conaffinity = 0

    origin = spec.worldbody.add_site()
    origin.name = "umi_world_origin"
    origin.type = mujoco.mjtGeom.mjGEOM_SPHERE
    origin.pos = config.robot_from_umi_positions(np.zeros(3))
    origin.size = np.array([0.018, 0.018, 0.018])
    origin.rgba = np.array([0.95, 0.08, 0.08, 1.0])
    for name, axis, color in (
        ("x", 0, (0.95, 0.12, 0.12, 1.0)),
        ("y", 1, (0.12, 0.85, 0.18, 1.0)),
        ("z", 2, (0.15, 0.35, 1.0, 1.0)),
    ):
        endpoint_umi = np.eye(3)[axis] * 0.10
        marker = spec.worldbody.add_site()
        marker.name = f"umi_world_axis_{name}"
        marker.type = mujoco.mjtGeom.mjGEOM_SPHERE
        marker.pos = config.robot_from_umi_positions(endpoint_umi)
        marker.size = np.array([0.012, 0.012, 0.012])
        marker.rgba = np.asarray(color, dtype=float)

    # The panel is a 21 cm-wide extrusion of the measured triangle.  Two
    # visual-only edge outlines make the 8 / 28.5 / 30.2 cm cross-section
    # inspectable in the MuJoCo window without adding collision geometry.
    triangle = config.support_triangle_vertices_table_m()
    width_axis_table = config.rotation_table_from_umi[:, 0]
    for side_name, sign in (("left", -1.0), ("right", 1.0)):
        offset = sign * (board.panel_width_m / 2.0) * width_axis_table
        vertices = {name: point + offset for name, point in triangle.items()}
        _add_visual_edge(
            spec, mujoco, config, f"board_support_{side_name}_base", vertices["rear_base"], vertices["lower"]
        )
        _add_visual_edge(
            spec, mujoco, config, f"board_support_{side_name}_side", vertices["rear_base"], vertices["top"]
        )
        _add_visual_edge(
            spec, mujoco, config, f"board_support_{side_name}_sloped", vertices["top"], vertices["lower"]
        )


def _add_visual_edge(
    spec: object,
    mujoco: object,
    config: SceneConfig,
    name: str,
    point_a_table: np.ndarray,
    point_b_table: np.ndarray,
) -> None:
    edge = spec.worldbody.add_geom()
    edge.name = name
    edge.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    edge.fromto = np.concatenate(
        (
            config.robot_from_table_positions(point_a_table),
            config.robot_from_table_positions(point_b_table),
        )
    )
    edge.size = np.array([0.0025, 0.0, 0.0])
    edge.rgba = np.array([0.92, 0.36, 0.08, 0.85])
    edge.contype = 0
    edge.conaffinity = 0


def _add_target_mocaps(spec: object, mujoco: object) -> None:
    """Add source-TCP markers and local RGB axes updated by the viewer."""
    for side in ("left", "right"):
        body = spec.worldbody.add_body()
        body.name = f"umi_target_{side}"
        body.mocap = True
        site = body.add_site()
        site.name = f"umi_target_{side}_marker"
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = np.array([0.018, 0.018, 0.018])
        site.rgba = TARGET_RGBA[side]
        for axis_name, axis, color in (
            ("x", 0, (0.95, 0.12, 0.12, 1.0)),
            ("y", 1, (0.12, 0.85, 0.18, 1.0)),
            ("z", 2, (0.15, 0.35, 1.0, 1.0)),
        ):
            endpoint = body.add_site()
            endpoint.name = f"umi_target_{side}_axis_{axis_name}"
            endpoint.type = mujoco.mjtGeom.mjGEOM_SPHERE
            endpoint.pos = np.eye(3)[axis] * 0.055
            endpoint.size = np.array([0.008, 0.008, 0.008])
            endpoint.rgba = np.asarray(color, dtype=float)
