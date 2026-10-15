#!/usr/bin/env python3
"""Render a joint-limit sampled TCP workspace for each supported embodiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.arm_quality import (  # noqa: E402
    DualArmUrdfKinematics,
    HUMANOID_ARM_JOINTS,
    HUMANOID_TARGET_LINKS,
    HumanoidUrdfKinematics,
    TIANJI_ARM_JOINTS,
    TIANJI_TARGET_LINKS,
    Thresholds,
    TianjiUrdfKinematics,
)
from tianji_lerobot_replay.humanoid_mobility import (  # noqa: E402
    HumanoidMobilityEvaluator,
    MobilityBounds,
)
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.kinematics import GRIPPER_CAD_QPOS  # noqa: E402
from tianji_lerobot_replay.scene import build_scene  # noqa: E402

from render_action_space_video import (  # noqa: E402
    CAMERA_DISTANCE_M,
    CAMERA_ELEVATION_DEG,
    CAMERA_LOOKAT_TABLE_M,
    DEFAULT_HUMANOID_CSV,
    DEFAULT_HUMANOID_SCENE,
    DEFAULT_HUMANOID_URDF,
    DEFAULT_REALMAN_CSV,
    DEFAULT_REALMAN_SCENE,
    DEFAULT_REALMAN_URDF,
    DEFAULT_TIANJI_CSV,
    DEFAULT_TIANJI_SCENE,
    DEFAULT_TIANJI_URDF,
    _add_sphere,
    _apply_common_lighting,
    _apply_visual_palette,
    _generic_scene,
    _qpos_address,
)


ROBOT_TITLES = {
    "tianji": "Tianji",
    "humanoid": "DM-SPARKY (full-body)",
    "realman": "RealMan",
}
DEFAULT_JOINT_PARAMS = PROJECT_ROOT / "config/tianji_joints_params.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot", required=True, choices=("tianji", "humanoid", "realman"))
    parser.add_argument("--urdf")
    parser.add_argument("--scene-config")
    parser.add_argument("--quality-csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--display-points", type=int, default=2200)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def _defaults(robot: str, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    values = {
        "tianji": (DEFAULT_TIANJI_URDF, DEFAULT_TIANJI_SCENE, DEFAULT_TIANJI_CSV),
        "humanoid": (
            DEFAULT_HUMANOID_URDF, DEFAULT_HUMANOID_SCENE, DEFAULT_HUMANOID_CSV),
        "realman": (DEFAULT_REALMAN_URDF, DEFAULT_REALMAN_SCENE, DEFAULT_REALMAN_CSV),
    }[robot]
    return (
        Path(args.urdf or values[0]).expanduser().resolve(),
        Path(args.scene_config or values[1]).expanduser().resolve(),
        Path(args.quality_csv or values[2]).expanduser().resolve(),
    )


def _quality_pose(csv_path: Path) -> dict[str, str]:
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"quality CSV is empty: {csv_path}")
    return rows[len(rows) // 2]


def _sobol(count: int, dimensions: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    exponent = int(np.ceil(np.log2(max(2, count))))
    values = qmc.Sobol(dimensions, scramble=True, seed=seed).random_base2(exponent)
    return values[:count]


def _scale_samples(unit: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + unit * (high - low)


def _fixed_arm_workspace(model, samples: int, seed: int) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(("left", "right")):
        low, high = model.bounds(side)
        q_values = _scale_samples(
            _sobol(samples, len(low), seed + side_index), low, high)
        side_points = np.empty((samples, 3), dtype=float)
        for index, q in enumerate(q_values):
            transform, _ = model.forward_jacobian(side, q, {})
            side_points[index] = transform[:3, 3]
        points[side] = side_points
    return points


def _humanoid_workspace(
    model: HumanoidUrdfKinematics,
    scene,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    mobility = HumanoidMobilityEvaluator(
        model, Thresholds(), scene, MobilityBounds())
    torso_low = np.asarray([model.all_limits[name][0] for name in (
        "knee_joint", "hip_pitch_joint", "hip_yaw_joint")])
    torso_high = np.asarray([model.all_limits[name][1] for name in (
        "knee_joint", "hip_pitch_joint", "hip_yaw_joint")])
    base_low = mobility.bounds_spec.base_y_min_m
    base_high = mobility.bounds_spec.base_y_max_m
    points: dict[str, np.ndarray] = {}
    for side_index, side in enumerate(("left", "right")):
        arm_low, arm_high = model.bounds(side)
        unit = _sobol(samples, 11, seed + side_index)
        base_values = base_low + unit[:, 0] * (base_high - base_low)
        torso_values = _scale_samples(unit[:, 1:4], torso_low, torso_high)
        arm_values = _scale_samples(unit[:, 4:], arm_low, arm_high)
        side_points = np.empty((samples, 3), dtype=float)
        for index in range(samples):
            transform, _ = mobility._forward_extended(
                side, arm_values[index], torso_values[index])
            side_points[index] = transform[:3, 3]
            side_points[index, 0] += (
                base_values[index] - mobility.reference_base_y_m)
        points[side] = side_points
    return points


def _downsample_voxels(
    points: np.ndarray,
    maximum: int,
    seed: int,
    voxel_size_m: float = 0.025,
) -> np.ndarray:
    voxels = np.floor(points / voxel_size_m).astype(np.int32)
    _, indices = np.unique(voxels, axis=0, return_index=True)
    selected = points[np.sort(indices)]
    if len(selected) <= maximum:
        return selected
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(selected), size=maximum, replace=False))
    return selected[keep]


def _table_bounds(points: np.ndarray, config) -> dict[str, list[float]]:
    table = (points - config.translation_robot_from_table_m) @ (
        config.rotation_robot_from_table)
    return {
        "minimum_m": np.min(table, axis=0).round(5).tolist(),
        "maximum_m": np.max(table, axis=0).round(5).tolist(),
    }


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_overlay(path: Path, title: str, duration: float, width: int, height: int) -> None:
    path.write_text(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H90000000,&H30000000,0,0,0,0,100,100,0,0,1,1,0,7,24,24,18,1
Style: Center,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H90000000,&H30000000,0,0,0,0,100,100,0,0,1,1,0,8,24,24,18,1
Style: Legend,DejaVu Sans,18,&H00FFFFFF,&H00FFFFFF,&H90000000,&H30000000,0,0,0,0,100,100,0,0,1,1,0,2,24,24,18,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Label,,0,0,0,,{title}
Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Center,,0,0,0,,Reachable TCP workspace
Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Legend,,0,0,0,,{{\\c&H003333F2&}}Left TCP{{\\c&H00FFFFFF&}}     {{\\c&H00FF6626&}}Right TCP
""", encoding="utf-8")


def _set_reference_pose(mujoco, model, data, robot: str, row: dict[str, str]) -> None:
    data.qpos[:] = 0.0
    joint_names = {
        "tianji": TIANJI_ARM_JOINTS,
        "humanoid": HUMANOID_ARM_JOINTS,
        "realman": {
            side: tuple(f"{side}_joint_{index}" for index in range(1, 8))
            for side in ("left", "right")
        },
    }[robot]
    for side in ("left", "right"):
        for index, name in enumerate(joint_names[side], 1):
            data.qpos[_qpos_address(mujoco, model, name)] = float(row[f"{side}_q{index}"])
    if robot == "tianji":
        for name in (
            "left_endeffector_joint", "left_endeffector_mimic_joint",
            "right_endeffector_joint", "right_endeffector_mimic_joint",
        ):
            data.qpos[_qpos_address(mujoco, model, name)] = GRIPPER_CAD_QPOS
    elif robot == "humanoid":
        for name in ("knee_joint", "hip_pitch_joint", "hip_yaw_joint"):
            data.qpos[_qpos_address(mujoco, model, name)] = float(row[name])


def main() -> int:
    args = parse_args()
    if args.samples < 256 or args.display_points < 128:
        raise ValueError("samples and display-points are too small")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    urdf, scene_path, quality_csv = _defaults(args.robot, args)
    config = load_scene_config(scene_path)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent / f".{output.stem}_render"
    work.mkdir(parents=True, exist_ok=True)

    if args.robot == "tianji":
        kinematic_model = TianjiUrdfKinematics(urdf, DEFAULT_JOINT_PARAMS)
        points = _fixed_arm_workspace(kinematic_model, args.samples, args.seed)
        scene = build_scene(
            urdf, work / "scene", config,
            [PROJECT_ROOT / "models/ros_packages"])
        model = scene.model
    elif args.robot == "humanoid":
        kinematic_model = HumanoidUrdfKinematics(urdf)
        points = _humanoid_workspace(
            kinematic_model, config, args.samples, args.seed)
        model = _generic_scene(urdf, work / "scene", config)
    else:
        arm_joints = {
            side: tuple(f"{side}_joint_{index}" for index in range(1, 8))
            for side in ("left", "right")
        }
        kinematic_model = DualArmUrdfKinematics(
            urdf, None, robot_name="RealMan RM75-B compact dual arm",
            arm_joints=arm_joints,
            target_links={"left": "left_tcp", "right": "right_tcp"},
            root_link="dual_base",
        )
        points = _fixed_arm_workspace(kinematic_model, args.samples, args.seed)
        model = _generic_scene(urdf, work / "scene", config)

    display = {
        side: _downsample_voxels(
            values, args.display_points, args.seed + index)
        for index, (side, values) in enumerate(points.items())
    }
    manifest = {
        "robot": ROBOT_TITLES[args.robot],
        "definition": (
            "Forward-kinematic TCP samples across effective joint limits. "
            + (
                "The full-body configuration additionally samples its bounded "
                "longitudinal base and knee/waist joints. "
                if args.robot == "humanoid" else ""
            )
            + "Collision and dynamics are not evaluated."
        ),
        "sampling": {
            "method": "scrambled Sobol joint-space sampling with 25-mm voxel display downsampling",
            "seed": args.seed,
            "raw_samples_per_arm": args.samples,
            "display_points": {side: len(values) for side, values in display.items()},
        },
        "workspace_bounds_table_frame": {
            side: _table_bounds(values, config) for side, values in points.items()
        },
        "scene_config": _portable_path(scene_path),
        "urdf": _portable_path(urdf),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _apply_visual_palette(mujoco, model, args.robot)
    _apply_common_lighting(model, config)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    row = _quality_pose(quality_csv)
    _set_reference_pose(mujoco, model, data, args.robot, row)
    if args.robot == "humanoid":
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        model.body_pos[body_id, 0] += (
            float(row["base_root_y_table_m"])
            - float(config.robot_root_in_table_m[1]))
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width, max_geom=10000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = config.robot_from_table_positions(CAMERA_LOOKAT_TABLE_M)
    camera.distance = CAMERA_DISTANCE_M
    camera.elevation = CAMERA_ELEVATION_DEG

    frame_count = max(2, int(round(args.duration * args.fps)))
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
    colors = {
        "left": (0.95, 0.16, 0.16, 0.24),
        "right": (0.15, 0.40, 1.00, 0.24),
    }
    for frame in range(frame_count):
        phase = frame / (frame_count - 1)
        camera.azimuth = 132.0 + 28.0 * phase
        renderer.update_scene(data, camera=camera)
        for side in ("left", "right"):
            for position in display[side]:
                _add_sphere(
                    mujoco, renderer.scene, position, 0.007, colors[side])
        encoder.stdin.write(renderer.render().tobytes())
    encoder.stdin.close()
    code = encoder.wait()
    renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg raw encoder failed with code {code}")

    overlay = work / "overlay.ass"
    _write_overlay(
        overlay, ROBOT_TITLES[args.robot], frame_count / args.fps,
        args.width, args.height)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-vf", f"ass={overlay}", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output),
        ],
        check=True,
    )
    print(
        f"WORKSPACE_VIDEO_WRITTEN robot={args.robot} "
        f"left={len(display['left'])} right={len(display['right'])} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
