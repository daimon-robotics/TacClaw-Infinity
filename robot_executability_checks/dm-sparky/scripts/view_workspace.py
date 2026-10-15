#!/usr/bin/env python3
"""Replay a solved trajectory with collision-free workspace overlays."""

from __future__ import annotations

import argparse
import csv
import queue
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.kinematics import GRIPPER_CAD_QPOS  # noqa: E402
from tianji_lerobot_replay.workspace_specs import get_workspace_spec  # noqa: E402
from tianji_lerobot_replay.workspace_visualization import WorkspaceOverlay  # noqa: E402
from render_action_space_video import (  # noqa: E402
    CAMERA_AZIMUTH_DEG,
    CAMERA_DISTANCE_M,
    CAMERA_ELEVATION_DEG,
    CAMERA_LOOKAT_TABLE_M,
    _apply_common_lighting,
    _apply_visual_palette,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("tianji", "humanoid"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--quality-csv")
    parser.add_argument("--workspace-cache")
    parser.add_argument("--play-speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--heatmap", action="store_true", help="start in Mode B")
    parser.add_argument("--overlap-only", action="store_true")
    return parser.parse_args()


def _rows(path: Path, episode: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if int(row["episode_index"]) == episode
        ]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise RuntimeError(f"quality CSV has no episode {episode}: {path}")
    return rows


def _id(mujoco: object, model: object, obj: object, name: str) -> int:
    value = mujoco.mj_name2id(model, obj, name)
    if value < 0:
        raise RuntimeError(f"workspace model is missing {name!r}")
    return int(value)


def _add_target_marker(mujoco: object, scene: object, position: np.ndarray, rgba: tuple) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, 0.018),
        position,
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    scene.ngeom += 1


def main() -> int:
    args = parse_args()
    if args.play_speed <= 0:
        raise ValueError("--play-speed must be positive")
    import mujoco
    import mujoco.viewer

    spec = get_workspace_spec(args.robot)
    cache = Path(
        args.workspace_cache or PROJECT_ROOT / ".workspace_cache" / args.robot / "workspace.npz"
    ).expanduser().resolve()
    quality_csv = Path(args.quality_csv or spec.quality_csv).expanduser().resolve()
    envelope_xml = cache.parent / "envelope" / "workspace_envelope.xml"
    if not cache.is_file() or not envelope_xml.is_file():
        raise RuntimeError(
            "workspace cache is incomplete; run scripts/compute_workspace.py first: "
            f"{cache}"
        )
    rows = _rows(quality_csv, args.episode)
    model = mujoco.MjModel.from_xml_path(str(envelope_xml))
    data = mujoco.MjData(model)
    overlay = WorkspaceOverlay(
        model,
        cache,
        initial_mode=2 if args.heatmap else 1,
        overlap_only=args.overlap_only,
    )
    config = load_scene_config(spec.scene_config)
    _apply_visual_palette(mujoco, model, args.robot)
    _apply_common_lighting(model, config)

    qpos_addresses = {
        side: np.asarray([
            model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in spec.joint_names[side]
        ], dtype=int)
        for side in ("left", "right")
    }
    gripper_addresses = []
    if args.robot == "tianji":
        gripper_addresses = [
            int(model.jnt_qposadr[_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in (
                "left_endeffector_joint", "left_endeffector_mimic_joint",
                "right_endeffector_joint", "right_endeffector_mimic_joint",
            )
        ]
    torso_addresses = {}
    base_body_id = -1
    base_body_pos = None
    if args.robot == "humanoid":
        torso_addresses = {
            name: int(model.jnt_qposadr[
                _id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in ("knee_joint", "hip_pitch_joint", "hip_yaw_joint")
        }
        base_body_id = _id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        base_body_pos = model.body_pos[base_body_id].copy()

    actions: queue.SimpleQueue[str] = queue.SimpleQueue()

    def key_callback(key: int) -> None:
        action = {
            32: "play",
            ord("r"): "restart", ord("R"): "restart",
            ord("w"): "workspace", ord("W"): "workspace",
            ord("o"): "scope", ord("O"): "scope",
        }.get(key)
        if action:
            actions.put(action)

    frame = 0
    playing = not args.paused
    accumulated = 0.0

    def apply_frame() -> None:
        row = rows[frame]
        data.qpos[:] = model.qpos0
        for side in ("left", "right"):
            data.qpos[qpos_addresses[side]] = [
                float(row[f"{side}_q{index}"]) for index in range(1, 8)
            ]
        if gripper_addresses:
            data.qpos[gripper_addresses] = GRIPPER_CAD_QPOS
        if args.robot == "humanoid":
            for name, address in torso_addresses.items():
                data.qpos[address] = float(row[name])
            model.body_pos[base_body_id] = base_body_pos
            model.body_pos[base_body_id, 0] += (
                float(row["base_root_y_table_m"])
                - float(config.robot_root_in_table_m[1])
            )
        data.time = float(row["timestamp_s"])
        mujoco.mj_forward(model, data)

    print("Workspace replay viewer opened.")
    print("Space: play/pause | R: first frame | W: OFF/Envelope/Heatmap | O: all/overlap")
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = config.robot_from_table_positions(CAMERA_LOOKAT_TABLE_M)
        viewer.cam.distance = CAMERA_DISTANCE_M
        viewer.cam.azimuth = CAMERA_AZIMUTH_DEG
        viewer.cam.elevation = CAMERA_ELEVATION_DEG
        previous = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            while not actions.empty():
                action = actions.get()
                if action == "play":
                    playing = not playing
                elif action == "restart":
                    frame = 0
                    accumulated = 0.0
                    playing = False
                elif action == "workspace":
                    overlay.cycle_mode()
                elif action == "scope":
                    overlay.toggle_scope()
            if playing and frame < len(rows) - 1:
                accumulated += elapsed * args.play_speed
                while frame < len(rows) - 1:
                    interval = float(rows[frame + 1]["timestamp_s"]) - float(rows[frame]["timestamp_s"])
                    interval = interval if interval > 1e-7 else 1.0 / 30.0
                    if accumulated < interval:
                        break
                    accumulated -= interval
                    frame += 1
                if frame == len(rows) - 1:
                    playing = False
            with viewer.lock():
                apply_frame()
                draw = overlay.update_user_scene(viewer.user_scn, data)
                row = rows[frame]
                for side, color in (
                    ("left", (0.95, 0.12, 0.12, 0.95)),
                    ("right", (0.12, 0.35, 1.0, 0.95)),
                ):
                    position = np.asarray([
                        float(row[f"{side}_target_x_m"]),
                        float(row[f"{side}_target_y_m"]),
                        float(row[f"{side}_target_z_m"]),
                    ])
                    _add_target_marker(mujoco, viewer.user_scn, position, color)
            eta_key = (
                "{side}_whole_body_isotropy_ratio"
                if args.robot == "humanoid" else "{side}_isotropy_ratio"
            )
            eta = [float(rows[frame][eta_key.format(side=side)] or "nan") for side in ("left", "right")]
            left_text = (
                f"{spec.title}\n"
                f"episode {args.episode} | frame {frame + 1}/{len(rows)} | "
                f"{'PLAY' if playing else 'PAUSE'}\n"
                f"Workspace: {overlay.mode_name} | {overlay.scope_name}\n"
                f"Manipulability eta L/R: {eta[0]:.4f} / {eta[1]:.4f}\n"
                f"Heatmap shell geoms: {draw['displayed']}/{draw['requested']}"
            )
            right_text = (
                "Space: play / pause\nR: first frame\n"
                "W: workspace mode\nO: all / overlap only\n"
                "viridis: weighted Jacobian isotropy"
            )
            viewer.set_texts((None, None, left_text, right_text))
            viewer.sync()
            time.sleep(0.001)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
