#!/usr/bin/env python3
"""Render collision-free workspace/manipulability videos and evidence images."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.lerobot_v3 import open_dataset, load_episode  # noqa: E402
from tianji_lerobot_replay.render_validation import validate_render_rows  # noqa: E402
from tianji_lerobot_replay.kinematics import GRIPPER_CAD_QPOS  # noqa: E402
from tianji_lerobot_replay.workspace import build_workspace_envelopes  # noqa: E402
from tianji_lerobot_replay.workspace_specs import get_workspace_spec  # noqa: E402
from tianji_lerobot_replay.workspace_visualization import WorkspaceOverlay  # noqa: E402
from tianji_lerobot_replay.pointcloud_visualization import StereoPointCloudOverlay  # noqa: E402
from render_action_space_video import (  # noqa: E402
    CAMERA_LOOKAT_TABLE_M,
    _apply_common_lighting,
    _apply_visual_palette,
)


VIEWS = {
    "front_oblique": (145.0, -13.0, 2.10),
    "side": (92.0, -10.0, 2.18),
    "top_oblique": (145.0, -45.0, 2.28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("tianji", "humanoid"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--workspace-cache")
    parser.add_argument("--quality-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional resampled duration; omitted means every source trajectory frame",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=2.10)
    parser.add_argument("--label-fontsize", type=int, default=54)
    parser.add_argument("--manip-fontsize", type=int, default=50)
    parser.add_argument("--legend-fontsize", type=int, default=44)
    parser.add_argument("--max-shell-geoms", type=int, default=5000)
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="omit the bottom color-legend caption (used for multi-panel composites)",
    )
    parser.add_argument("--pointcloud-run")
    parser.add_argument("--pointcloud-pose-file")
    parser.add_argument("--dataset")
    parser.add_argument("--pointcloud-max-points", type=int, default=2200)
    parser.add_argument("--pointcloud-voxel-size", type=float, default=0.015)
    parser.add_argument("--pointcloud-point-radius", type=float, default=0.005)
    parser.add_argument("--pointcloud-color-saturation", type=float, default=1.0)
    parser.add_argument("--pointcloud-color-gamma", type=float, default=1.0)
    parser.add_argument("--pointcloud-color-priority", type=float, default=0.0)
    parser.add_argument("--pointcloud-min-luminance", type=float, default=0.08)
    parser.add_argument("--pointcloud-workcell-margin", type=float, default=0.03)
    parser.add_argument("--pointcloud-max-height-above-table", type=float, default=0.55)
    parser.add_argument("--pointcloud-hide-table-surface", action="store_true")
    parser.add_argument("--pointcloud-tcp-radius", type=float)
    parser.add_argument("--pointcloud-stabilize-static", action="store_true")
    return parser.parse_args()


def _read_rows(path: Path, episode: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["episode_index"]) == episode]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise RuntimeError(f"quality CSV has no episode {episode}: {path}")
    return rows


def _object_id(mujoco: object, model: object, obj: object, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise RuntimeError(f"model is missing {name!r}")
    return int(value)


class PoseWriter:
    def __init__(self, mujoco: object, model: object, robot: str, config: object) -> None:
        spec = get_workspace_spec(robot)
        self.mujoco = mujoco
        self.model = model
        self.robot = robot
        self.config = config
        self.addresses = {
            side: np.asarray([
                model.jnt_qposadr[_object_id(
                    mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in spec.joint_names[side]
            ], dtype=int)
            for side in ("left", "right")
        }
        self.gripper_addresses: list[int] = []
        if robot == "tianji":
            self.gripper_addresses = [
                int(model.jnt_qposadr[_object_id(
                    mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in (
                    "left_endeffector_joint", "left_endeffector_mimic_joint",
                    "right_endeffector_joint", "right_endeffector_mimic_joint",
                )
            ]
        self.torso_addresses: dict[str, int] = {}
        self.base_body_id = -1
        self.base_body_pos = None
        if robot == "humanoid":
            self.torso_addresses = {
                name: int(model.jnt_qposadr[_object_id(
                    mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in ("knee_joint", "hip_pitch_joint", "hip_yaw_joint")
            }
            self.base_body_id = _object_id(
                mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
            self.base_body_pos = model.body_pos[self.base_body_id].copy()

    def apply(self, data: object, row: dict[str, str]) -> None:
        data.qpos[:] = self.model.qpos0
        for side in ("left", "right"):
            data.qpos[self.addresses[side]] = [
                float(row[f"{side}_q{index}"]) for index in range(1, 8)
            ]
        if self.gripper_addresses:
            data.qpos[self.gripper_addresses] = GRIPPER_CAD_QPOS
        if self.robot == "humanoid":
            for name, address in self.torso_addresses.items():
                data.qpos[address] = float(row[name])
            self.model.body_pos[self.base_body_id] = self.base_body_pos
            self.model.body_pos[self.base_body_id, 0] += (
                float(row["base_root_y_table_m"])
                - float(self.config.robot_root_in_table_m[1])
            )
        data.time = float(row["timestamp_s"])
        self.mujoco.mj_forward(self.model, data)


def _format_isotropy_ratio(value: str | None) -> str:
    """Render an arm-quality isotropy (manipulability) ratio for on-screen display."""
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.2f}"


def _write_ass(
    path: Path,
    title: str,
    rows: list[dict[str, str]],
    fps: float,
    width: int,
    height: int,
    pointcloud_cadence: int | None = None,
    pointcloud_static_fused: bool = False,
    show_legend: bool = True,
    label_fontsize: int = 54,
    manip_fontsize: int = 50,
    legend_fontsize: int = 44,
) -> None:
    duration = len(rows) / fps
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,DejaVu Sans,{label_fontsize},&H00FFFFFF,&H00FFFFFF,&HA0000000,&H30000000,-1,0,0,0,100,100,0,0,1,2,0,7,22,22,16,1
Style: Manip,DejaVu Sans,{manip_fontsize},&H00FFFFFF,&H00FFFFFF,&HA0000000,&H30000000,-1,0,0,0,100,100,0,0,1,2,0,9,22,22,64,1
Style: Legend,DejaVu Sans,{legend_fontsize},&H00FFFFFF,&H00FFFFFF,&HA0000000,&H30000000,-1,0,0,0,100,100,0,0,1,2,0,2,22,22,18,1
Style: Note,DejaVu Sans,24,&H00FFFFFF,&H00FFFFFF,&HA0000000,&H30000000,-1,0,0,0,100,100,0,0,1,2,0,2,22,22,16,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Label,,0,0,0,,{title}
"""
    if show_legend:
        header += (
            f"Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Legend,,0,0,0,,"
            "{\\c&H00FF9A30&}Blue: left-arm workspace{\\c&H00FFFFFF&}    "
            "{\\c&H002D77FF&}Orange: right-arm workspace{\\c&H00FFFFFF&}    "
            "{\\c&H00F060B8&}Purple: spatial overlap\n"
        )
    if pointcloud_cadence is not None:
        stabilization = (
            " Table/board: world-frame multi-view fusion."
            if pointcloud_static_fused else ""
        )
        header += (
            f"Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Note,,0,0,30,,"
            "RGB-D points: raw Ego stereo with calibrated RGB/IMU extrinsics."
            f"{stabilization} Dynamic points update every "
            f"{pointcloud_cadence} source frame(s).\n"
        )
    events = []
    for frame, row in enumerate(rows):
        start = _ass_timestamp(frame / fps)
        end = _ass_timestamp((frame + 1) / fps)
        manipulability = "\\N".join(
            f"{side.capitalize()} manipulability {_format_isotropy_ratio(row.get(f'{side}_isotropy_ratio'))}"
            for side in ("left", "right")
        )
        events.append(
            f"Dialogue: 0,{start},{end},Manip,,0,0,0,,{manipulability}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _apply_presentation_lighting(model: object) -> None:
    for light_id in range(model.nlight):
        model.light_ambient[light_id] = np.asarray([0.22, 0.22, 0.22])
        model.light_diffuse[light_id] = np.asarray([0.76, 0.76, 0.76])
        model.light_specular[light_id] = np.asarray([0.18, 0.18, 0.18])
    model.vis.headlight.ambient[:] = np.asarray([0.14, 0.14, 0.14])
    model.vis.headlight.diffuse[:] = np.asarray([0.42, 0.42, 0.42])
    model.vis.headlight.specular[:] = np.asarray([0.12, 0.12, 0.12])
    model.vis.headlight.active = 1


def _render_scene(
    mujoco: object,
    renderer: object,
    camera: object,
    option: object,
    data: object,
    overlay: WorkspaceOverlay,
    pointcloud: StereoPointCloudOverlay | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera, scene_option=option)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    overlay.update_user_scene(renderer.scene, data, reset=False)
    if pointcloud is not None:
        # Append opaque RGB points after the translucent workspace shells so
        # shell colors do not visually wash out the registered camera colors.
        pointcloud.update_user_scene(renderer.scene, frame_index)
    return renderer.render().copy()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from PIL import Image

    spec = get_workspace_spec(args.robot)
    video_title = {"humanoid": "DM-SPARKY"}.get(args.robot, spec.title)
    cache = Path(
        args.workspace_cache or PROJECT_ROOT / ".workspace_cache" / args.robot / "workspace.npz"
    ).expanduser().resolve()
    csv_path = Path(args.quality_csv or spec.quality_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(csv_path, args.episode)
    config = load_scene_config(spec.scene_config)
    if args.dataset:
        dataset = open_dataset(args.dataset)
        validate_render_rows(rows, load_episode(dataset, args.episode), config)
    else:
        print("WARNING: no --dataset; CSV/source consistency and world registration are unverified.")
    envelope_xml = cache.parent / "envelope" / "workspace_envelope.xml"
    build_workspace_envelopes(
        cache,
        cache.parent / "workspace_model.xml",
        spec.reference_body,
        envelope_xml.parent,
    )
    model = mujoco.MjModel.from_xml_path(str(envelope_xml))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)
    data = mujoco.MjData(model)
    _apply_visual_palette(mujoco, model, args.robot)
    _apply_common_lighting(model, config)
    _apply_presentation_lighting(model)
    overlay = WorkspaceOverlay(model, cache, initial_mode=1)
    pointcloud = None
    if args.pointcloud_run:
        if not args.dataset:
            raise ValueError("--dataset is required with --pointcloud-run")
        pose_file = args.pointcloud_pose_file
        if pose_file is None:
            candidate = (
                Path(args.pointcloud_run).expanduser().resolve().parent
                / "stereo_input" / "world_from_head.npz"
            )
            if candidate.is_file():
                pose_file = str(candidate)
        pointcloud = StereoPointCloudOverlay(
            args.pointcloud_run,
            args.dataset,
            args.episode,
            config,
            max_points=args.pointcloud_max_points,
            voxel_size_m=args.pointcloud_voxel_size,
            point_radius_m=args.pointcloud_point_radius,
            color_saturation=args.pointcloud_color_saturation,
            color_gamma=args.pointcloud_color_gamma,
            color_priority_fraction=args.pointcloud_color_priority,
            min_color_luminance=args.pointcloud_min_luminance,
            workcell_xy_margin_m=args.pointcloud_workcell_margin,
            max_height_above_table_m=args.pointcloud_max_height_above_table,
            hide_table_surface=args.pointcloud_hide_table_surface,
            tcp_neighborhood_radius_m=args.pointcloud_tcp_radius,
            pose_file=pose_file,
            stabilize_static_surfaces=args.pointcloud_stabilize_static,
        )
        # The source-colored points remain opaque; slightly reducing only the
        # translucent workspace opacity prevents its blue/orange/purple tint
        # from obscuring the registered camera colors.
        overlay.set_envelope_alpha_scale(0.60)
    writer = PoseWriter(mujoco, model, args.robot, config)
    renderer = mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
        max_geom=(
            args.max_shell_geoms
            + args.pointcloud_max_points
            + model.ngeom
            + 16
        ),
    )
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3] = 1
    camera.lookat[:] = config.robot_from_table_positions(CAMERA_LOOKAT_TABLE_M)

    evidence = output_dir / "screenshots"
    evidence.mkdir(parents=True, exist_ok=True)
    middle = rows[len(rows) // 2]
    writer.apply(data, middle)
    overlay.mode = 2
    overlay.apply_envelope_visibility()
    for name, (azimuth, elevation, distance) in VIEWS.items():
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        image = _render_scene(
            mujoco, renderer, camera, option, data, overlay, pointcloud,
            int(middle["frame_index"]),
        )
        Image.fromarray(image).save(evidence / f"{args.robot}_{name}_heatmap.png")
    overlay.mode = 1
    overlay.apply_envelope_visibility()
    for name, (azimuth, elevation, distance) in VIEWS.items():
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.distance = distance
        image = _render_scene(
            mujoco, renderer, camera, option, data, overlay, pointcloud,
            int(middle["frame_index"]),
        )
        Image.fromarray(image).save(evidence / f"{args.robot}_{name}_envelope.png")
    overlay.mode = 1
    overlay.apply_envelope_visibility()

    if args.duration is None:
        video_rows = rows
        time_steps = np.diff([float(row["timestamp_s"]) for row in rows])
        positive_steps = time_steps[time_steps > 1e-7]
        video_fps = (
            float(1.0 / np.median(positive_steps))
            if len(positive_steps) else float(args.fps)
        )
    else:
        frame_count = max(2, int(round(args.duration * args.fps)))
        video_rows = [
            rows[int(round(frame / (frame_count - 1) * (len(rows) - 1)))]
            for frame in range(frame_count)
        ]
        video_fps = float(args.fps)
    frame_count = len(video_rows)
    raw_video = output_dir / f".{args.robot}_workspace_raw.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{args.width}x{args.height}", "-r", f"{video_fps:g}",
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", str(raw_video),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    camera.elevation = -13.0
    camera.distance = args.camera_distance
    for output_frame in range(frame_count):
        fraction = output_frame / (frame_count - 1)
        writer.apply(data, video_rows[output_frame])
        camera.azimuth = 142.0 + 6.0 * fraction
        image = _render_scene(
            mujoco, renderer, camera, option, data, overlay, pointcloud,
            int(video_rows[output_frame]["frame_index"]),
        )
        encoder.stdin.write(image.tobytes())
    encoder.stdin.close()
    code = encoder.wait()
    renderer.close()
    if code:
        raise RuntimeError(f"ffmpeg raw encoder failed with code {code}")

    ass = output_dir / f".{args.robot}_workspace.ass"
    _write_ass(
        ass, video_title, video_rows, video_fps, args.width, args.height,
        pointcloud.cadence_frames if pointcloud is not None else None,
        pointcloud.stabilize_static_surfaces if pointcloud is not None else False,
        show_legend=not args.no_legend,
        label_fontsize=args.label_fontsize,
        manip_fontsize=args.manip_fontsize,
        legend_fontsize=args.legend_fontsize,
    )
    suffix = "_with_pointcloud" if pointcloud is not None else ""
    output = output_dir / f"{args.robot}_collision_free_workspace{suffix}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
            "-vf", f"ass={ass}", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output),
        ],
        check=True,
    )
    raw_video.unlink(missing_ok=True)
    ass.unlink(missing_ok=True)
    print(f"WORKSPACE_VISUALIZATION_WRITTEN robot={args.robot} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
