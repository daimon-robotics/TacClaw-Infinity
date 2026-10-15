#!/usr/bin/env python3
"""Evaluate a mobile humanoid with longitudinal base and torso motion enabled."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    summarize,
)
from tianji_lerobot_replay.errors import ReplayError  # noqa: E402
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.humanoid_mobility import (  # noqa: E402
    MobilityBounds,
    evaluate_mobility_dataset,
)
from tianji_lerobot_replay.lerobot_v3 import open_dataset  # noqa: E402
from tianji_lerobot_replay.quality_visualization import (  # noqa: E402
    write_quality_dashboard,
)


DEFAULT_URDF = (
    PROJECT_ROOT / "models" / "ros_packages" /
    "dm25_pr02_hr_pd04_eg_20260810" / "urdf" /
    "dm25_pr02_hr_pd04_eg_20260810.urdf"
)
DEFAULT_JOINT_PARAMS = PROJECT_ROOT / "config" / "dm25_joints_params.xml"
DEFAULT_SCENE = PROJECT_ROOT / "config" / "mobile_humanoid_scene.json"


def _episode_ids(text: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if text == "all":
        return available
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            selected.update(range(first, last + 1))
        else:
            selected.add(int(token))
    missing = sorted(selected.difference(available))
    if missing:
        raise ReplayError(f"数据集没有这些 episode：{missing}")
    return tuple(sorted(selected))


def _partitions(dataset, episodes: tuple[int, ...], jobs: int) -> list[tuple[int, ...]]:
    groups = [[] for _ in range(min(jobs, len(episodes)))]
    loads = [0 for _ in groups]
    for episode in sorted(
        episodes, key=lambda value: dataset.episodes[value].length, reverse=True
    ):
        target = min(range(len(groups)), key=loads.__getitem__)
        groups[target].append(episode)
        loads[target] += dataset.episodes[episode].length
    return [tuple(sorted(group)) for group in groups if group]


def _worker(payload: dict) -> tuple[tuple[int, ...], list[dict]]:
    dataset = open_dataset(
        payload["dataset"],
        pose_feature=payload["pose_feature"],
        pose_layout_path=payload["pose_layout"],
    )
    scene = load_scene_config(payload["scene"])
    model = HumanoidUrdfKinematics(payload["urdf"], payload["joint_params"])
    thresholds = Thresholds(**payload["thresholds"])
    bounds = MobilityBounds(**payload["bounds"])
    rows = evaluate_mobility_dataset(
        dataset, payload["episodes"], scene, model, thresholds, bounds)
    return tuple(payload["episodes"]), rows


def _range(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _optional_range(rows: list[dict], key: str) -> dict[str, float] | None:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) is not None], dtype=float)
    if not len(values):
        return None
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--pose-feature", default="observation.state")
    parser.add_argument("--pose-layout", default=None)
    parser.add_argument("--scene-config", default=str(DEFAULT_SCENE))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--joint-params", default=str(DEFAULT_JOINT_PARAMS))
    parser.add_argument("--base-y-min", type=float, default=-0.50)
    parser.add_argument("--base-y-max", type=float, default=-0.05)
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--orientation-weight", type=float, default=0.30)
    parser.add_argument("--isotropy-threshold", type=float, default=0.04)
    parser.add_argument("--minimum-reachable-rate", type=float, default=0.99)
    parser.add_argument("--maximum-low-manipulability-rate", type=float, default=0.05)
    parser.add_argument("--ik-max-nfev", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "humanoid_adaptive_mobility"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.jobs <= 0:
            raise ReplayError("--jobs 必须大于 0。")
        dataset = open_dataset(
            args.dataset,
            pose_feature=args.pose_feature,
            pose_layout_path=args.pose_layout,
        )
        episodes = _episode_ids(args.episodes, dataset.episode_ids)
        scene = load_scene_config(args.scene_config)
        model = HumanoidUrdfKinematics(args.urdf, args.joint_params)
        thresholds = Thresholds(
            position_tolerance_m=args.position_tolerance_mm / 1000.0,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
            orientation_weight_m_per_rad=args.orientation_weight,
            isotropy_ratio_low=args.isotropy_threshold,
            minimum_both_reachable_rate=args.minimum_reachable_rate,
            maximum_low_manipulability_rate=args.maximum_low_manipulability_rate,
            ik_max_nfev=args.ik_max_nfev,
        )
        bounds = MobilityBounds(args.base_y_min, args.base_y_max)
        partitions = _partitions(dataset, episodes, args.jobs)
        payload_base = {
            "dataset": str(dataset.root),
            "pose_feature": args.pose_feature,
            "pose_layout": args.pose_layout,
            "scene": str(Path(args.scene_config).resolve()),
            "urdf": str(Path(args.urdf).resolve()),
            "joint_params": str(Path(args.joint_params).resolve()),
            "thresholds": thresholds.report(),
            "bounds": {
                "base_y_min_m": bounds.base_y_min_m,
                "base_y_max_m": bounds.base_y_max_m,
            },
        }
        rows: list[dict] = []
        if len(partitions) == 1:
            _, rows = _worker({**payload_base, "episodes": partitions[0]})
        else:
            print(
                f"并行全身评估：{len(episodes)} episodes / "
                f"{sum(dataset.episodes[e].length for e in episodes)} frames / "
                f"{len(partitions)} workers",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
                futures = [
                    executor.submit(_worker, {**payload_base, "episodes": part})
                    for part in partitions
                ]
                completed = 0
                for future in as_completed(futures):
                    part, partial = future.result()
                    rows.extend(partial)
                    completed += len(part)
                    print(
                        f"已完成 {completed}/{len(episodes)} episodes "
                        f"(本分片 {part[0]}..{part[-1]})",
                        flush=True,
                    )
        rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
        result = summarize("adaptive_whole_body", rows, thresholds)
        motion_ranges = {
            key: _range(rows, key)
            for key in (
                "base_root_y_table_m",
                "knee_joint",
                "hip_pitch_joint",
                "hip_yaw_joint",
            )
        }
        whole_body_metrics = {
            side: {
                "isotropy_ratio": _optional_range(
                    rows, f"{side}_whole_body_isotropy_ratio"),
                "condition_number": _optional_range(
                    rows, f"{side}_whole_body_condition_number"),
            }
            for side in ("left", "right")
        }
        summary = {
            "format_version": 1,
            "comparison_conclusion": result["conclusion"],
            "dataset": dataset.report(),
            "episodes": list(episodes),
            "robot_model": model.report(),
            "coordinate_registration": scene.report(),
            "thresholds": thresholds.report(),
            "strategies": {"adaptive_whole_body": result},
            "mobility": {
                "classification": "kinematic_upper_bound",
                "base_motion": "longitudinal table-y position selected per frame",
                "base_y_search_range_m": [args.base_y_min, args.base_y_max],
                "shared_active_joints": [
                    "knee_joint", "hip_pitch_joint", "hip_yaw_joint"],
                "motion_ranges": motion_ranges,
                "whole_body_jacobian_metrics": whole_body_metrics,
                "selection_metric": (
                    "pose tolerance first; arm-only isotropy used to choose among "
                    "reachable local solutions"
                ),
            },
            "scope": {
                "self_collision_checked": False,
                "environment_collision_checked": False,
                "velocity_acceleration_checked": False,
                "base_and_torso_continuity_seeded": True,
                "base_and_torso_velocity_bounded": False,
                "unreachable_manipulability": "undefined and stored as null",
            },
        }
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with (output / "adaptive_whole_body_frames.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_quality_dashboard(
            output / "quality_dashboard.html",
            summary,
            {"adaptive_whole_body": rows},
        )
        report = [
            "# 人形底盘/腰部自适应轨迹质量报告", "",
            f"结论：**{result['conclusion']}**", "",
            f"双臂同时可达：{result['both_reachable_frames']:,} / "
            f"{result['frames']:,} = {result['both_reachable_rate']:.4%}。",
        ]
        for side, label in (("left", "左臂"), ("right", "右臂")):
            arm = result["arms"][side]
            report.append(
                f"{label}可达率 {arm['reachable_rate']:.4%}；可达帧中低 "
                f"isotropy 比例 "
                f"{arm['low_manipulability_rate_among_reachable']:.4%}。"
            )
        report.extend([
            "", "这是未检查碰撞、速度、加速度和力矩的运动学理论上限；",
            "底盘和腰部可以逐帧变化，不能直接作为真机安全结论。", "",
        ])
        (output / "report.md").write_text("\n".join(report), encoding="utf-8")
        print(
            f"HUMANOID_MOBILITY_{result['conclusion']} "
            f"both={result['both_reachable_rate']:.6f} output={output}",
            flush=True,
        )
        return 0 if result["conclusion"] == "PASS" else 2
    except (OSError, ValueError, ReplayError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
