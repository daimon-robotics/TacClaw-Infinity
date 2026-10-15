#!/usr/bin/env python3
"""Prepare a smooth full-body humanoid replay by solving sampled TCP frames.

This is a visualization helper, not a replacement for the exhaustive action-
space evaluator.  It runs coupled full-body IK on regularly spaced frames,
keeps one continuity seed, and interpolates the resulting joint path to the
original dataset rate before recomputing per-frame pose errors and Jacobians.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.arm_quality import (  # noqa: E402
    HumanoidUrdfKinematics,
    Thresholds,
    _pose_error,
    _target_matrix,
)
from tianji_lerobot_replay.frames import (  # noqa: E402
    load_scene_config,
    quaternion_xyzw_to_matrix,
)
from tianji_lerobot_replay.humanoid_mobility import (  # noqa: E402
    HumanoidMobilityEvaluator,
    WholeBodyIkResult,
)
from tianji_lerobot_replay.lerobot_v3 import open_dataset, load_episode  # noqa: E402


DEFAULT_URDF = (
    PROJECT_ROOT / "models/ros_packages/dm25_pr02_hr_pd04_eg_20260810/urdf/"
    "dm25_pr02_hr_pd04_eg_20260810.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scene-config", default=str(PROJECT_ROOT / "config/mobile_humanoid_scene.json"))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--joint-params", default=str(PROJECT_ROOT / "config/dm25_joints_params.xml"))
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--ik-max-nfev", type=int, default=30)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _targets(
    frame: int, positions: np.ndarray, rotations: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        side: _target_matrix(positions[frame, side_index], rotations[frame, side_index])
        for side_index, side in enumerate(("left", "right"))
    }


def _result_from_x(
    evaluator: HumanoidMobilityEvaluator,
    targets: dict[str, np.ndarray],
    x: np.ndarray,
) -> WholeBodyIkResult:
    errors: dict[str, tuple[float, float]] = {}
    reachable: dict[str, bool] = {}
    for side in ("left", "right"):
        current, _ = evaluator._forward_extended(
            side, x[evaluator._arm_slice(side)], x[1:4]
        )
        errors[side] = _pose_error(
            current, evaluator.adjusted_target(targets[side], x[0])
        )[:2]
        reachable[side] = bool(
            errors[side][0] <= evaluator.thresholds.position_tolerance_m
            and errors[side][1] <= evaluator.thresholds.orientation_tolerance_rad
        )
    return WholeBodyIkResult(
        x=x.copy(),
        left_position_error_m=errors["left"][0],
        left_orientation_error_rad=errors["left"][1],
        right_position_error_m=errors["right"][0],
        right_orientation_error_rad=errors["right"][1],
        left_reachable=reachable["left"],
        right_reachable=reachable["right"],
        seed_name="sampled_continuous_interpolation",
        evaluations=0,
    )


def main() -> int:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    dataset = open_dataset(args.dataset)
    source = load_episode(dataset, args.episode)
    scene = load_scene_config(args.scene_config)
    model = HumanoidUrdfKinematics(args.urdf, args.joint_params)
    thresholds = Thresholds(ik_max_nfev=args.ik_max_nfev)
    evaluator = HumanoidMobilityEvaluator(model, thresholds, scene)
    positions = scene.robot_from_umi_positions(source.positions_world_m)
    rotations = scene.robot_from_umi_rotations(
        quaternion_xyzw_to_matrix(source.quaternions_world_xyzw)
    )

    sampled_frames = list(range(0, source.frame_count, args.stride))
    if sampled_frames[-1] != source.frame_count - 1:
        sampled_frames.append(source.frame_count - 1)
    sampled_x: list[np.ndarray] = []
    previous = None
    for sample_index, frame in enumerate(sampled_frames):
        targets = _targets(frame, positions, rotations)
        if previous is None:
            result = evaluator.solve(targets, previous)
        else:
            result = evaluator._solve_once(
                targets,
                previous,
                "sampled_continuous",
                regularization=7e-4,
            )
        previous = result.x.copy()
        sampled_x.append(previous)
        if sample_index % 25 == 0 or sample_index + 1 == len(sampled_frames):
            print(
                f"sampled IK {sample_index + 1}/{len(sampled_frames)} "
                f"(source frame {frame}/{source.frame_count - 1})",
                flush=True,
            )

    sample_axis = np.asarray(sampled_frames, dtype=float)
    full_axis = np.arange(source.frame_count, dtype=float)
    samples = np.asarray(sampled_x, dtype=float)
    full_x = np.column_stack(
        [np.interp(full_axis, sample_axis, samples[:, index]) for index in range(samples.shape[1])]
    )

    rows = []
    for frame, x in enumerate(full_x):
        targets = _targets(frame, positions, rotations)
        result = _result_from_x(evaluator, targets, x)
        rows.append(
            evaluator.row(
                args.episode,
                frame,
                float(source.timestamps_s[frame]),
                targets,
                result,
            )
        )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "adaptive_whole_body_frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "full-body replay visualization",
        "dataset": str(dataset.root),
        "episode": args.episode,
        "source_frames": source.frame_count,
        "sample_stride_frames": args.stride,
        "sampled_ik_frames": len(sampled_frames),
        "ik_max_nfev": args.ik_max_nfev,
        "interpolation": "linear in full-body joint coordinates",
        "per_frame_validation": "pose error and Jacobian recomputed after interpolation",
        "scope": "not an exhaustive reachability evaluation",
        "left_reachable_rate": float(np.mean([row["left_reachable"] for row in rows])),
        "right_reachable_rate": float(np.mean([row["right_reachable"] for row in rows])),
        "both_reachable_rate": float(np.mean([row["both_reachable"] for row in rows])),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"HUMANOID_REPLAY_ROWS_WRITTEN output={csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
