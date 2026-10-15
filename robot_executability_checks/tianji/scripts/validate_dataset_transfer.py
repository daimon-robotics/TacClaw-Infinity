#!/usr/bin/env python3
"""Cross-dataset smoke checks and source-axis/MuJoCo evidence images.

An explicit board-frame declaration permits sampled IK diagnostics only.
Unknown frames never receive a reachability verdict or an invented transform.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tianji_lerobot_replay.arm_quality import (
    ArmQualityEvaluator,
    TianjiUrdfKinematics, Thresholds, _target_matrix,
)
from tianji_lerobot_replay.frames import load_scene_config, quaternion_xyzw_to_matrix
from tianji_lerobot_replay.lerobot_v3 import open_dataset, load_episode
from tianji_lerobot_replay.scene import build_scene
from tianji_lerobot_replay.workspace_specs import get_workspace_spec
from render_action_space_video import (
    _generic_scene, _add_sphere, _apply_common_lighting, _apply_visual_palette,
    CAMERA_LOOKAT_TABLE_M,
)


def annotation(image, lines):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    draw.rectangle((0, 0, canvas.width, 28 * len(lines) + 10), fill=(12, 19, 26))
    for i, line in enumerate(lines):
        draw.text((12, 8 + 28 * i), line, font=font, fill="white")
    return canvas


def axis(mj, scene, origin, rotation, length=.10):
    for i, color in enumerate(((1., .15, .15, 1.), (.15, 1., .15, 1.), (.15, .4, 1., 1.))):
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                       np.eye(3).ravel(), np.array(color))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_ARROW, .003,
                         origin, origin + length * rotation[:, i])
        scene.ngeom += 1


def source_preview(episode, output, frame_status):
    import mujoco as mj
    model = mj.MjModel.from_xml_string('<mujoco><visual><global offwidth="960" offheight="540"/></visual><worldbody/></mujoco>')
    data = mj.MjData(model)
    model.vis.headlight.ambient[:] = [.6, .6, .6]
    points = episode.positions_world_m
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.lookat[:] = np.mean(points.reshape(-1, 3), axis=0)
    camera.distance = max(.65, 1.8 * np.ptp(points.reshape(-1, 3), axis=0).max())
    camera.azimuth, camera.elevation = 140., -25.
    rotations = quaternion_xyzw_to_matrix(episode.quaternions_world_xyzw)
    with mj.Renderer(model, height=540, width=960, max_geom=3000) as renderer:
        for f in (0, episode.frame_count // 2, episode.frame_count - 1):
            renderer.update_scene(data, camera=camera)
            for s, color in enumerate(((1., .3, .3, .8), (.2, .6, 1., .8))):
                for t in np.unique(np.linspace(0, episode.frame_count - 1, 400, dtype=int)):
                    _add_sphere(mj, renderer.scene, points[t, s], .004, color)
                _add_sphere(mj, renderer.scene, points[f, s], .014, color)
                axis(mj, renderer.scene, points[f, s], rotations[f, s])
            axis(mj, renderer.scene, np.zeros(3), np.eye(3), .15)
            annotation(renderer.render().copy(), [
                f"SOURCE FRAME ONLY - episode {episode.episode_index}, frame {f}",
                f"World registration: {frame_status}. No robot/table alignment claimed.",
                "Trails: left red / right blue. Local axes: X red / Y green / Z blue.",
            ]).save(output / f"source_ep{episode.episode_index}_frame{f}.png")


def model_for(robot):
    spec = get_workspace_spec(robot)
    if robot == "tianji":
        return TianjiUrdfKinematics(spec.urdf, ROOT / "config/tianji_joints_params.xml")
    raise ValueError(f"unsupported robot {robot!r}")


def robot_checks(robot, episodes, samples, output, board_confirmed):
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    spec = get_workspace_spec(robot)
    kin = model_for(robot)
    config = load_scene_config(spec.scene_config)
    evaluator = ArmQualityEvaluator(kin, Thresholds())
    report = {"robot": robot, "scene_config": str(spec.scene_config.relative_to(ROOT)),
              "sampled_independent_ik": [], "physical_registration_verified": False}
    with tempfile.TemporaryDirectory(prefix="replay_transfer_") as tmp:
        if robot == "tianji":
            model = build_scene(spec.urdf, tmp, config, [ROOT / "models/ros_packages"]).model
        else:
            model = _generic_scene(kin.urdf_path, Path(tmp), config)
        data = mj.MjData(model)
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 540
        addresses = {s: np.array([model.jnt_qposadr[mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_JOINT, j)] for j in spec.joint_names[s]])
                     for s in ("left", "right")}
        dofs = {s: np.array([model.jnt_dofadr[mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_JOINT, j)] for j in spec.joint_names[s]])
                for s in ("left", "right")}
        body = {s: mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, spec.ee_bodies[s])
                for s in ("left", "right")}
        # Independent MuJoCo FK and Jacobian checks at deterministic joint samples.
        max_position = max_angle = max_jacobian = 0.
        rng = np.random.default_rng(20261012)
        for _ in range(6):
            data.qpos[:] = model.qpos0
            for s in ("left", "right"):
                low, high = kin.bounds(s)
                data.qpos[addresses[s]] = low + rng.uniform(.2, .8, 7) * (high - low)
            mj.mj_forward(model, data)
            for s in ("left", "right"):
                fk, jac = kin.forward_jacobian(s, data.qpos[addresses[s]], {})
                max_position = max(max_position, float(np.linalg.norm(fk[:3, 3] - data.xpos[body[s]])))
                max_angle = max(max_angle, float(Rotation.from_matrix(
                    fk[:3, :3].T @ data.xmat[body[s]].reshape(3, 3)).magnitude()))
                jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mj.mj_jacBody(model, data, jp, jr, body[s])
                max_jacobian = max(max_jacobian, float(np.abs(jac - np.vstack((jp, jr))[:, dofs[s]]).max()))
        report["mujoco_parity"] = {"arm_configurations": 12,
            "max_position_difference_m": max_position, "max_rotation_difference_rad": max_angle,
            "max_jacobian_element_difference": max_jacobian,
            # MuJoCo's serialized derived XML rounds joint transforms. These
            # guards allow micrometre-level serialization error, not task error.
            "position_tolerance_m": 5e-6, "rotation_tolerance_rad": 5e-6,
            "jacobian_element_tolerance": 5e-6,
            "pass": max_position < 5e-6 and max_angle < 5e-6 and max_jacobian < 5e-6}
        if not board_confirmed:
            report["ik_status"] = "BLOCKED_UNKNOWN_SOURCE_WORLD"
            return report
        _apply_visual_palette(mj, model, robot)
        _apply_common_lighting(model, config)
        for episode in episodes:
            positions = config.robot_from_umi_positions(episode.positions_world_m)
            rotations = config.robot_from_umi_rotations(quaternion_xyzw_to_matrix(episode.quaternions_world_xyzw))
            indices = np.unique(np.linspace(0, episode.frame_count - 1, samples, dtype=int))
            rows = []
            for f in indices:
                row = {"episode": episode.episode_index, "frame": int(f)}
                data.qpos[:] = model.qpos0
                for a, s in enumerate(("left", "right")):
                    result = evaluator.solve_independent(s, _target_matrix(positions[f, a], rotations[f, a]))
                    metrics = evaluator.metrics(s, result)
                    row[s] = metrics
                    data.qpos[addresses[s]] = result.q
                rows.append(row)
                if f == indices[len(indices) // 2]:
                    mj.mj_forward(model, data)
                    camera = mj.MjvCamera()
                    mj.mjv_defaultCamera(camera)
                    camera.lookat[:] = config.robot_from_table_positions(CAMERA_LOOKAT_TABLE_M)
                    camera.distance, camera.azimuth, camera.elevation = 2.1, 145., -20.
                    with mj.Renderer(model, height=540, width=960, max_geom=3000) as renderer:
                        renderer.update_scene(data, camera=camera)
                        for a, s in enumerate(("left", "right")):
                            color = (1., .2, .2, .85) if a == 0 else (.2, .5, 1., .85)
                            for t in np.unique(np.linspace(0, episode.frame_count - 1, 150, dtype=int)):
                                _add_sphere(mj, renderer.scene, positions[t, a], .004, color)
                            _add_sphere(mj, renderer.scene, positions[f, a], .016, color)
                            axis(mj, renderer.scene, positions[f, a], rotations[f, a], .08)
                        labels = [f"{robot}: episode {episode.episode_index}, frame {f} - sampled IK diagnostic",
                                  "Prior board registration assumed; fixed torso. NOT full-trajectory certification."]
                        for s in ("left", "right"):
                            m = row[s]
                            eta = m["isotropy_ratio"]
                            labels.append(f"{s}: reachable={m['reachable']}, error={1000*m['position_error_m']:.2f} mm / "
                                          f"{m['orientation_error_deg']:.2f} deg, eta={eta if eta is None else round(eta, 4)}")
                        annotation(renderer.render().copy(), labels).save(output / f"{robot}_ep{episode.episode_index}.png")
            report["sampled_independent_ik"].extend(rows)
            print(f"{robot} episode {episode.episode_index}: {len(rows)} sampled frames", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", default="representative", help="representative or comma-separated indices")
    parser.add_argument("--samples-per-episode", type=int, default=12)
    parser.add_argument("--board-frame-confirmed", action="store_true",
                        help="explicit declaration that the prior board/TCP convention applies; not estimated")
    args = parser.parse_args()
    if args.samples_per_episode < 2:
        parser.error("--samples-per-episode must be >= 2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = open_dataset(args.dataset)
    ids = dataset.episode_ids
    selected = sorted(set((ids[0], ids[len(ids)//2], ids[-1]))) if args.episodes == "representative" else [int(v) for v in args.episodes.split(",")]
    episodes = [load_episode(dataset, i) for i in selected]
    status = "PRIOR_BOARD_CONVENTION_DECLARED" if args.board_frame_confirmed else "UNKNOWN / BLOCKED"
    report = {"dataset": dataset.report(), "world_status": status,
              "note": "Sampled independent IK only. No continuity, collision or true-world registration certification.",
              "episodes": [], "robots": []}
    for episode in episodes:
        p = episode.positions_world_m
        table_p = load_scene_config(ROOT / "config/default_scene.json").table_from_umi_positions(p)
        report["episodes"].append({"episode": episode.episode_index, "frames": episode.frame_count,
            "quaternion_max_norm_error": float(np.abs(np.linalg.norm(episode.quaternions_world_xyzw, axis=-1)-1).max()),
            "source_position_min_m": p.min(axis=0).tolist(), "source_position_max_m": p.max(axis=0).tolist(),
            "counterfactual_old_mapping_tcp_height_above_table_range_m": [float(table_p[..., 2].min()-.8), float(table_p[..., 2].max()-.8)]})
        source_preview(episode, args.output, status)
    for robot in ("tianji",):
        report["robots"].append(robot_checks(robot, episodes, args.samples_per_episode, args.output, args.board_frame_confirmed))
        # Checkpoint preserves completed checks if subsequent rendering fails.
        (args.output / "validation.json").write_text(json.dumps(report, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist()) + "\n")
    print(f"Saved {args.output / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
