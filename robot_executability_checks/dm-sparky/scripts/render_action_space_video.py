#!/usr/bin/env python3
"""Render a paper/demo-ready reachability and manipulability replay video."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.arm_quality import (  # noqa: E402
    HUMANOID_ARM_JOINTS,
    TIANJI_ARM_JOINTS,
)
from tianji_lerobot_replay.frames import (  # noqa: E402
    load_scene_config,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)
from tianji_lerobot_replay.kinematics import GRIPPER_CAD_QPOS  # noqa: E402
from tianji_lerobot_replay.lerobot_v3 import load_episode, open_dataset  # noqa: E402
from tianji_lerobot_replay.render_validation import validate_render_rows  # noqa: E402
from tianji_lerobot_replay.scene import (  # noqa: E402
    TABLETOP_RGBA,
    TABLE_LEG_RGBA,
    _add_calibration_board_visual,
    build_scene,
)


DEFAULT_TIANJI_URDF = (
    PROJECT_ROOT / "models/ros_packages/daimon_tj_fx_description/urdf/"
    "daimon_tj_fx_dual_arm_with_gripper.urdf"
)
DEFAULT_HUMANOID_URDF = (
    PROJECT_ROOT / "models/ros_packages/dm25_pr02_hr_pd04_eg_20260810/urdf/"
    "dm25_pr02_hr_pd04_eg_20260810.urdf"
)
DEFAULT_TIANJI_SCENE = PROJECT_ROOT / "config/base_up_60mm_farther_100mm_scene.json"
DEFAULT_HUMANOID_SCENE = PROJECT_ROOT / "config/mobile_humanoid_scene.json"
DEFAULT_TIANJI_CSV = (
    PROJECT_ROOT / "outputs/full_base_up_60mm_farther_100mm_arm_quality/"
    "continuous_midpoint_frames.csv"
)
DEFAULT_HUMANOID_CSV = (
    PROJECT_ROOT / "outputs/full_cup_humanoid_adaptive_mobility/"
    "adaptive_whole_body_frames.csv"
)
# One table-relative camera and lighting setup is used for every embodiment.
# The model origins differ, so the shared look-at and light positions are
# transformed through each scene configuration before rendering.
CAMERA_LOOKAT_TABLE_M = np.asarray([0.70, 0.35, 0.90], dtype=float)
CAMERA_DISTANCE_M = 2.60
CAMERA_AZIMUTH_DEG = 145.0
CAMERA_ELEVATION_DEG = -15.0
KEY_LIGHT_TABLE_M = np.asarray([0.35, 0.05, 2.65], dtype=float)
KEY_LIGHT_TARGET_TABLE_M = np.asarray([0.70, 0.35, 0.72], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot", required=True, choices=("tianji", "humanoid"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--quality-csv")
    parser.add_argument("--scene-config")
    parser.add_argument("--urdf")
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def _defaults(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.robot == "tianji":
        return (
            Path(args.urdf or DEFAULT_TIANJI_URDF),
            Path(args.scene_config or DEFAULT_TIANJI_SCENE),
            Path(args.quality_csv or DEFAULT_TIANJI_CSV),
        )
    if args.robot == "humanoid":
        return (
            Path(args.urdf or DEFAULT_HUMANOID_URDF),
            Path(args.scene_config or DEFAULT_HUMANOID_SCENE),
            Path(args.quality_csv or DEFAULT_HUMANOID_CSV),
        )
    raise ValueError(f"unsupported robot {args.robot!r}")


def _read_rows(path: Path, episode: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["episode_index"]) == episode
        ]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise RuntimeError(f"quality CSV has no episode {episode}: {path}")
    return rows


def _prepare_generic_urdf(source: Path, destination: Path) -> Path:
    """Resolve the packaged humanoid meshes without Tianji-specific edits."""
    tree = ET.parse(source)
    root = tree.getroot()
    package_dir = source.parents[1]
    package_name = package_dir.name
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        prefix = f"package://{package_name}/"
        if filename and filename.startswith(prefix):
            mesh.set("filename", str((package_dir / filename[len(prefix):]).resolve()))
    mujoco_node = root.find("mujoco")
    if mujoco_node is None:
        mujoco_node = ET.Element("mujoco")
        root.insert(0, mujoco_node)
    compiler = mujoco_node.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_node, "compiler")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _generic_scene(urdf: Path, output_dir: Path, config):
    import mujoco

    prepared = _prepare_generic_urdf(urdf.resolve(), output_dir / "humanoid_prepared.urdf")
    spec = mujoco.MjSpec.from_file(str(prepared))
    quat_xyzw = matrix_to_quaternion_xyzw(config.rotation_robot_from_table)
    quat_wxyz = np.asarray(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)

    def add_box(name, table_pos, half_size, rgba):
        geom = spec.worldbody.add_geom()
        geom.name = name
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.pos = config.robot_from_table_positions(np.asarray(table_pos, dtype=float))
        geom.quat = quat_wxyz
        geom.size = np.asarray(half_size, dtype=float)
        geom.rgba = np.asarray(rgba, dtype=float)
        geom.contype = 0
        geom.conaffinity = 0

    half_top = config.tabletop_thickness_m / 2.0
    add_box(
        "table_top_visual",
        (config.table_length_m / 2.0, config.table_width_m / 2.0,
         config.table_height_m - half_top),
        (config.table_length_m / 2.0, config.table_width_m / 2.0, half_top),
        TABLETOP_RGBA,
    )
    leg_half_z = (config.table_height_m - config.tabletop_thickness_m) / 2.0
    for x in (0.05, config.table_length_m - 0.05):
        for y in (0.05, config.table_width_m - 0.05):
            add_box(
                f"table_leg_{x}_{y}", (x, y, leg_half_z),
                (0.03, 0.03, leg_half_z), TABLE_LEG_RGBA)
    floor = spec.worldbody.add_geom()
    floor.name = "floor_visual"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.pos = config.robot_from_table_positions(np.zeros(3))
    floor.quat = quat_wxyz
    floor.size = np.asarray([5.0, 5.0, 0.1])
    floor.rgba = np.asarray([0.22, 0.24, 0.28, 1.0])
    floor.contype = 0
    floor.conaffinity = 0
    _add_calibration_board_visual(spec, mujoco, config)
    light = spec.worldbody.add_light()
    light.name = "scene_light"
    light.pos = np.asarray([0.2, 0.0, 2.3])
    light.dir = np.asarray([0.0, 0.0, -1.0])
    light.diffuse = np.asarray([0.9, 0.9, 0.9])
    spec.compile()
    xml_path = output_dir / "scene.xml"
    spec.to_file(str(xml_path))
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _qpos_address(mujoco, model, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"MuJoCo model is missing joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def _add_sphere(mujoco, scene, position, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def _add_visual_overlays(mujoco, renderer, target_positions, frame, row, robot):
    trail_colors = ((0.95, 0.16, 0.16, 0.30), (0.15, 0.38, 1.0, 0.30))
    for side_index in range(2):
        for trail_frame in range(0, len(target_positions), 6):
            _add_sphere(
                mujoco, renderer.scene, target_positions[trail_frame, side_index],
                0.005, trail_colors[side_index])
    for side_index, side in enumerate(("left", "right")):
        if robot != "tianji":
            _add_sphere(
                mujoco, renderer.scene, target_positions[frame, side_index],
                0.018,
                (0.95, 0.16, 0.16, 0.78) if side == "left"
                else (0.15, 0.38, 1.0, 0.78),
            )
        reachable = _bool(row[f"{side}_reachable"])
        metric = (
            f"{side}_whole_body_isotropy_ratio"
            if robot == "humanoid" else f"{side}_isotropy_ratio"
        )
        isotropy = float(row[metric]) if row.get(metric) else 0.0
        color = (
            (0.15, 0.95, 0.28, 1.0) if reachable and isotropy >= 0.04
            else (1.0, 0.72, 0.05, 1.0) if reachable
            else (0.95, 0.05, 0.05, 1.0)
        )
        _add_sphere(
            mujoco, renderer.scene,
            target_positions[frame, side_index] + np.asarray([0.0, 0.0, 0.045]),
            0.012, color)


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _write_ass(path: Path, rows, robot: str, fps: float, width: int, height: int):
    title = {
        "tianji": "Tianji",
        "humanoid": "DM-SPARKY (full-body)",
    }[robot]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H80000000,&H30000000,0,0,0,0,100,100,0,0,1,1,0,7,24,24,18,1
Style: Metric,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H80000000,&H30000000,0,0,0,0,100,100,0,0,1,1,0,8,24,24,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    duration = len(rows) / fps
    events = [
        "Dialogue: 0,0:00:00.00,"
        f"{_ass_timestamp(duration)},Label,,0,0,0,,{title}"
    ]
    metric_key = (
        "{side}_whole_body_isotropy_ratio"
        if robot == "humanoid" else "{side}_isotropy_ratio"
    )
    for frame, row in enumerate(rows):
        values = []
        for side in ("left", "right"):
            value = row.get(metric_key.format(side=side), "")
            values.append(f"{float(value):.3f}" if value else "--")
        start = _ass_timestamp(frame / fps)
        end = _ass_timestamp((frame + 1) / fps)
        events.append(
            f"Dialogue: 0,{start},{end},Metric,,0,0,0,,"
            f"Manipulability η (L/R): {values[0]} / {values[1]}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _apply_common_lighting(model, config) -> None:
    light_pos = config.robot_from_table_positions(KEY_LIGHT_TABLE_M)
    light_target = config.robot_from_table_positions(KEY_LIGHT_TARGET_TABLE_M)
    light_dir = light_target - light_pos
    light_dir /= np.linalg.norm(light_dir)
    for light_id in range(model.nlight):
        model.light_pos[light_id] = light_pos
        model.light_dir[light_id] = light_dir
        model.light_ambient[light_id] = np.asarray([0.16, 0.16, 0.16])
        model.light_diffuse[light_id] = np.asarray([0.82, 0.82, 0.82])
        model.light_specular[light_id] = np.asarray([0.24, 0.24, 0.24])
        model.light_castshadow[light_id] = True
        model.light_active[light_id] = True
    model.vis.headlight.ambient[:] = np.asarray([0.08, 0.08, 0.08])
    model.vis.headlight.diffuse[:] = np.asarray([0.28, 0.28, 0.28])
    model.vis.headlight.specular[:] = np.asarray([0.16, 0.16, 0.16])
    model.vis.headlight.active = 1


def _apply_visual_palette(mujoco, model, robot: str) -> None:
    if robot != "tianji":
        return
    left = np.asarray([0.18, 0.43, 0.60, 1.0], dtype=float)
    right = np.asarray([0.32, 0.52, 0.68, 1.0], dtype=float)
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name == "Base_L" or (
            body_name.startswith("Link") and body_name.endswith("_L")):
            model.geom_rgba[geom_id] = left
        elif body_name == "Base_R" or (
            body_name.startswith("Link") and body_name.endswith("_R")):
            model.geom_rgba[geom_id] = right


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    urdf, scene_path, csv_path = _defaults(args)
    config = load_scene_config(scene_path)
    rows = _read_rows(csv_path, args.episode)
    dataset = open_dataset(args.dataset)
    episode = load_episode(dataset, args.episode)
    validate_render_rows(rows, episode, config)
    target_positions = config.robot_from_umi_positions(episode.positions_world_m)
    target_rotations = config.robot_from_umi_rotations(
        quaternion_xyzw_to_matrix(episode.quaternions_world_xyzw))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}_render"
    work.mkdir(parents=True, exist_ok=True)
    if args.robot == "tianji":
        scene = build_scene(
            urdf, work / "scene", config,
            [PROJECT_ROOT / "models/ros_packages"])
        model = scene.model
        target_mocap_ids = scene.target_mocap_ids
        joint_names = TIANJI_ARM_JOINTS
    else:
        model = _generic_scene(urdf, work / "scene", config)
        target_mocap_ids = None
        joint_names = (
            HUMANOID_ARM_JOINTS if args.robot == "humanoid" else {
                side: tuple(f"{side}_joint_{index}" for index in range(1, 8))
                for side in ("left", "right")
            }
        )

    _apply_visual_palette(mujoco, model, args.robot)
    _apply_common_lighting(model, config)
    data = mujoco.MjData(model)
    addresses = {
        side: [_qpos_address(mujoco, model, name) for name in joint_names[side]]
        for side in ("left", "right")
    }
    torso_addresses = {}
    base_body_id = -1
    base_body_pos = None
    if args.robot == "humanoid":
        torso_addresses = {
            name: _qpos_address(mujoco, model, name)
            for name in ("knee_joint", "hip_pitch_joint", "hip_yaw_joint")
        }
        base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base_body_id < 0:
            raise RuntimeError("humanoid model is missing base_link body")
        base_body_pos = model.body_pos[base_body_id].copy()

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width, max_geom=10000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = config.robot_from_table_positions(CAMERA_LOOKAT_TABLE_M)
    camera.distance = CAMERA_DISTANCE_M
    camera.azimuth = CAMERA_AZIMUTH_DEG
    camera.elevation = CAMERA_ELEVATION_DEG

    raw_video = work / "raw.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", f"{args.fps:g}",
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", str(raw_video),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for frame, row in enumerate(rows):
        data.qpos[:] = 0.0
        for side in ("left", "right"):
            for joint_index, address in enumerate(addresses[side], start=1):
                data.qpos[address] = float(row[f"{side}_q{joint_index}"])
        if args.robot == "tianji":
            gripper_names = (
                "left_endeffector_joint", "left_endeffector_mimic_joint",
                "right_endeffector_joint", "right_endeffector_mimic_joint")
            for name in gripper_names:
                data.qpos[_qpos_address(mujoco, model, name)] = GRIPPER_CAD_QPOS
            for side_index, side in enumerate(("left", "right")):
                mocap_id = target_mocap_ids[side]
                data.mocap_pos[mocap_id] = target_positions[frame, side_index]
                quat_xyzw = matrix_to_quaternion_xyzw(target_rotations[frame, side_index])
                data.mocap_quat[mocap_id] = np.asarray(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        elif args.robot == "humanoid":
            for name, address in torso_addresses.items():
                data.qpos[address] = float(row[name])
            model.body_pos[base_body_id] = base_body_pos
            model.body_pos[base_body_id, 0] += (
                float(row["base_root_y_table_m"])
                - float(config.robot_root_in_table_m[1]))
        data.time = float(row["timestamp_s"])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        _add_visual_overlays(
            mujoco, renderer, target_positions, frame, row, args.robot)
        encoder.stdin.write(renderer.render().tobytes())
    encoder.stdin.close()
    return_code = encoder.wait()
    renderer.close()
    if return_code:
        raise RuntimeError(f"ffmpeg raw encoder failed with code {return_code}")

    ass_path = work / "overlay.ass"
    _write_ass(ass_path, rows, args.robot, args.fps, args.width, args.height)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-vf", f"ass={ass_path}", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output),
        ],
        check=True,
    )
    print(f"VIDEO_WRITTEN robot={args.robot} episode={args.episode} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
