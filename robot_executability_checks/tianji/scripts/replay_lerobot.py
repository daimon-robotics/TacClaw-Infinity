#!/usr/bin/env python3
"""Replay one dual-TCP LeRobot v3 dataset with the Tianji MuJoCo model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay import ReplayError  # noqa: E402
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.lerobot_v3 import open_dataset  # noqa: E402
from tianji_lerobot_replay.replay import create_session  # noqa: E402


DEFAULT_URDF = (
    PROJECT_ROOT
    / "models"
    / "ros_packages"
    / "daimon_tj_fx_description"
    / "urdf"
    / "daimon_tj_fx_dual_arm_with_gripper.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "在 MuJoCo 中将 LeRobot v3 的双 UMI TCP 轨迹以 Aik7 解析运动学回放到天机双臂。"
            "本工具不执行可达性、碰撞、速度或真机执行判定。"
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="LeRobot v3 数据集根目录（含 meta/info.json 与 data/）。",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="初始 episode_index（默认 0；不是第 N 条的隐式索引）。",
    )
    parser.add_argument(
        "--pose-feature",
        default="observation.state",
        help=(
            "存放双 TCP pose 的 LeRobot 向量 feature（默认 observation.state；"
            "用 auto 可按字段名选择）。"
        ),
    )
    parser.add_argument(
        "--pose-layout",
        default=None,
        help="其他 LeRobot v3 schema 的左右 TCP 字段布局 JSON；缺失时严格检查标准 UMI 字段名。",
    )
    parser.add_argument(
        "--scene-config",
        default=str(PROJECT_ROOT / "config" / "default_scene.json"),
        help="桌子、实测标定板和天机根坐标的 JSON 配置。",
    )
    parser.add_argument(
        "--urdf",
        default=str(DEFAULT_URDF),
        help="天机双臂 URDF 路径。",
    )
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        help=(
            "ROS package 根目录，可重复指定。内置 models/ros_packages 会自动加入；"
            "自定义 URDF 有 package:// asset 时按需添加。"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "replay"),
        help="生成 scene.xml、准备后的 URDF、virtual_scene_registration.json 和 manifest 的目录。",
    )
    parser.add_argument(
        "--play-speed",
        type=float,
        default=1.0,
        help="按记录时间戳播放的倍率（默认 1.0）。",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="窗口打开时暂停在第 1 帧。",
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="只列出数据集 episode 与已解析 TCP 布局，不建立场景或打开窗口。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="建立场景并拟合指定 episode 后退出，不打开 OpenGL 查看器。",
    )
    parser.add_argument(
        "--workspace-cache",
        default=None,
        help=(
            "可选 workspace.npz；启用 W/O 可切换的碰撞过滤可达空间包络与 "
            "manipulability 热图。"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.play_speed <= 0.0:
            raise ReplayError("--play-speed 必须大于 0。")
        dataset = open_dataset(
            args.dataset,
            pose_feature=args.pose_feature,
            pose_layout_path=args.pose_layout,
        )
        if args.list_episodes:
            print(f"LeRobot {dataset.info.get('codebase_version')}：{dataset.root}")
            print(f"fps: {dataset.fps:g}")
            print("episode_index: " + ", ".join(str(value) for value in dataset.episode_ids))
            print("TCP layout: " + str(dataset.layout.report()))
            return 0

        config = load_scene_config(args.scene_config)
        package_roots = [PROJECT_ROOT / "models" / "ros_packages"]
        package_roots.extend(Path(value).expanduser() for value in args.package_root)
        session = create_session(
            dataset=dataset,
            urdf=args.urdf,
            output_dir=args.output,
            config=config,
            package_roots=package_roots,
            scene_config_path=args.scene_config,
        )
        if args.dry_run:
            session.load_viewer_track(args.episode, verbose=True)
            print("dry-run 完成：场景与该 episode 的纯运动学回放已建立；未打开查看器。")
            return 0
        session.run(
            args.episode,
            play_speed=args.play_speed,
            paused=args.paused,
            workspace_cache=args.workspace_cache,
        )
        return 0
    except (ReplayError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
