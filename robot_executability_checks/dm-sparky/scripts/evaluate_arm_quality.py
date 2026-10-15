#!/usr/bin/env python3
"""Evaluate LeRobot v3 dual-TCP trajectories on Tianji or mobile humanoid."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.arm_quality import (  # noqa: E402
    HumanoidUrdfKinematics,
    TianjiUrdfKinematics,
    Thresholds,
    build_summary,
    evaluate_dataset,
    write_outputs,
)
from tianji_lerobot_replay.errors import ReplayError  # noqa: E402
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.lerobot_v3 import open_dataset  # noqa: E402


HUMANOID_URDF = (
    PROJECT_ROOT / "models" / "ros_packages" /
    "dm25_pr02_hr_pd04_eg_20260810" / "urdf" /
    "dm25_pr02_hr_pd04_eg_20260810.urdf"
)
HUMANOID_JOINT_PARAMS = PROJECT_ROOT / "config" / "dm25_joints_params.xml"
HUMANOID_SCENE = PROJECT_ROOT / "config" / "mobile_humanoid_scene.json"
TIANJI_URDF = (
    PROJECT_ROOT / "models" / "ros_packages" /
    "daimon_tj_fx_description" / "urdf" /
    "daimon_tj_fx_dual_arm_with_gripper.urdf"
)
TIANJI_JOINT_PARAMS = PROJECT_ROOT / "config" / "tianji_joints_params.xml"
TIANJI_SCENE = PROJECT_ROOT / "config" / "default_scene.json"


ROBOT_DEFAULTS = {
    "tianji": (TIANJI_URDF, TIANJI_JOINT_PARAMS, TIANJI_SCENE),
    "mobile-humanoid": (
        HUMANOID_URDF, HUMANOID_JOINT_PARAMS, HUMANOID_SCENE),
}


def _evaluate_partition(payload: dict) -> tuple[tuple[int, ...], dict]:
    """Worker entry point; all arguments are plain pickle-safe values."""
    dataset = open_dataset(
        payload["dataset"],
        pose_feature=payload["pose_feature"],
        pose_layout_path=payload["pose_layout"],
    )
    config = load_scene_config(payload["scene_config"])
    model = (
        TianjiUrdfKinematics(payload["urdf"], payload["joint_params"])
        if payload["robot"] == "tianji" else
        HumanoidUrdfKinematics(payload["urdf"], payload["joint_params"])
    )
    thresholds = Thresholds(**payload["thresholds"])
    _, rows = evaluate_dataset(
        dataset,
        payload["episodes"],
        config,
        model,
        thresholds,
        fixed_joints=payload["fixed_joints"],
    )
    return tuple(payload["episodes"]), rows


def _partitions(dataset, episode_ids: tuple[int, ...], jobs: int) -> list[tuple[int, ...]]:
    """Greedily balance workers using episode lengths from LeRobot metadata."""
    groups: list[list[int]] = [[] for _ in range(min(jobs, len(episode_ids)))]
    loads = [0 for _ in groups]
    for episode in sorted(
        episode_ids, key=lambda value: dataset.episodes[value].length, reverse=True
    ):
        target = min(range(len(groups)), key=loads.__getitem__)
        groups[target].append(episode)
        loads[target] += dataset.episodes[episode].length
    return [tuple(sorted(group)) for group in groups if group]


def _episode_ids(text: str, available: tuple[int, ...]) -> tuple[int, ...]:
    if text == "all":
        return available
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(item) for item in token.split("-", 1))
            selected.update(range(first, last + 1))
        else:
            selected.add(int(token))
    missing = sorted(selected.difference(available))
    if missing:
        raise ReplayError(f"数据集没有这些 episode：{missing}")
    if not selected:
        raise ReplayError("--episodes 没有选中任何 episode。")
    return tuple(sorted(selected))


def _fixed_joints(assignments: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ReplayError(f"固定关节必须写成 NAME=RAD：{assignment!r}")
        name, value = assignment.split("=", 1)
        result[name] = float(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot v3 数据集根目录。")
    parser.add_argument(
        "--robot", choices=tuple(ROBOT_DEFAULTS), default="tianji",
        help="评估机器人（默认 tianji）。",
    )
    parser.add_argument(
        "--episodes", default="all",
        help="all、单个编号或逗号/范围，例如 0,2-4（默认 all）。",
    )
    parser.add_argument("--pose-feature", default="observation.state")
    parser.add_argument("--pose-layout", default=None)
    parser.add_argument(
        "--scene-config",
        default=None,
        help="UMI 标定板、桌子和机器人 base_link 的场景配置。",
    )
    parser.add_argument("--urdf", default=None, help="覆盖所选机器人的默认 URDF。")
    parser.add_argument(
        "--joint-params", default=None,
        help="覆盖所选机器人的默认运行时关节限位 XML。",
    )
    parser.add_argument(
        "--fixed-joint", action="append", default=[], metavar="NAME=RAD",
        help="固定非双臂关节角，可重复指定；未指定时为零。",
    )
    parser.add_argument(
        "--output", default=None,
        help="输出目录（默认按机器人分开写入 outputs）。",
    )
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--orientation-weight", type=float, default=0.30)
    parser.add_argument("--isotropy-threshold", type=float, default=0.04)
    parser.add_argument("--minimum-reachable-rate", type=float, default=0.99)
    parser.add_argument("--maximum-low-manipulability-rate", type=float, default=0.05)
    parser.add_argument("--ik-max-nfev", type=int, default=120)
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="并行评估进程数（默认 1；全量数据可设为 CPU 核数的一半）。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        default_urdf, default_joint_params, default_scene = ROBOT_DEFAULTS[args.robot]
        urdf = Path(args.urdf) if args.urdf else default_urdf
        joint_params = Path(args.joint_params) if args.joint_params else default_joint_params
        scene_config = Path(args.scene_config) if args.scene_config else default_scene
        output = (
            Path(args.output) if args.output else
            PROJECT_ROOT / "outputs" / f"{args.robot.replace('-', '_')}_arm_quality"
        )
        dataset = open_dataset(
            args.dataset,
            pose_feature=args.pose_feature,
            pose_layout_path=args.pose_layout,
        )
        selected = _episode_ids(args.episodes, dataset.episode_ids)
        config = load_scene_config(scene_config)
        model = (
            TianjiUrdfKinematics(urdf, joint_params)
            if args.robot == "tianji" else
            HumanoidUrdfKinematics(urdf, joint_params)
        )
        thresholds = Thresholds(
            position_tolerance_m=args.position_tolerance_mm / 1000.0,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
            orientation_weight_m_per_rad=args.orientation_weight,
            isotropy_ratio_low=args.isotropy_threshold,
            minimum_both_reachable_rate=args.minimum_reachable_rate,
            maximum_low_manipulability_rate=args.maximum_low_manipulability_rate,
            ik_max_nfev=args.ik_max_nfev,
        )
        if args.jobs <= 0:
            raise ReplayError("--jobs 必须大于 0。")
        fixed_joints = _fixed_joints(args.fixed_joint)
        last_percent: dict[int, int] = {}

        def progress(episode: int, done: int, total: int) -> None:
            percent = int(done * 100 / total)
            if percent != last_percent.get(episode) and (
                    percent % 10 == 0 or done == total):
                print(f"episode {episode}: {percent}% ({done}/{total})", flush=True)
                last_percent[episode] = percent

        if args.jobs == 1 or len(selected) == 1:
            summary, rows = evaluate_dataset(
                dataset,
                selected,
                config,
                model,
                thresholds,
                fixed_joints=fixed_joints,
                progress=progress,
            )
        else:
            partitions = _partitions(dataset, selected, args.jobs)
            print(
                f"并行评估：{len(selected)} episodes / "
                f"{sum(dataset.episodes[e].length for e in selected)} frames / "
                f"{len(partitions)} workers",
                flush=True,
            )
            payload_base = {
                "dataset": str(dataset.root),
                "pose_feature": args.pose_feature,
                "pose_layout": args.pose_layout,
                "scene_config": str(scene_config),
                "urdf": str(urdf),
                "joint_params": str(joint_params),
                "robot": args.robot,
                "thresholds": thresholds.report(),
                "fixed_joints": fixed_joints,
            }
            rows = {"independent": [], "continuous_midpoint": []}
            with ProcessPoolExecutor(max_workers=len(partitions)) as executor:
                futures = []
                for episodes in partitions:
                    payload = dict(payload_base)
                    payload["episodes"] = episodes
                    futures.append(executor.submit(_evaluate_partition, payload))
                completed = 0
                for future in as_completed(futures):
                    episodes, partial = future.result()
                    completed += len(episodes)
                    for strategy in rows:
                        rows[strategy].extend(partial[strategy])
                    print(
                        f"已完成 {completed}/{len(selected)} episodes "
                        f"(本分片 {episodes[0]}..{episodes[-1]})",
                        flush=True,
                    )
            for strategy in rows:
                rows[strategy].sort(
                    key=lambda row: (row["episode_index"], row["frame_index"])
                )
            summary = build_summary(
                dataset, selected, config, model, thresholds, fixed_joints, rows
            )
        write_outputs(output, summary, rows)
        print(
            "ARM_QUALITY_{} robot={} independent={} continuous_midpoint={} output={} dashboard={}".format(
                summary["comparison_conclusion"],
                args.robot,
                summary["strategies"]["independent"]["conclusion"],
                summary["strategies"]["continuous_midpoint"]["conclusion"],
                output.expanduser().resolve(),
                (output / "quality_dashboard.html").expanduser().resolve(),
            )
        )
        return 0 if summary["comparison_conclusion"] == "PASS_BOTH" else 2
    except (ReplayError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
