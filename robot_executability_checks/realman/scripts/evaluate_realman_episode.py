#!/usr/bin/env python3
"""Evaluate a dual RM75-B visualization model on selected LeRobot episodes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.arm_quality import (  # noqa: E402
    DualArmUrdfKinematics,
    Thresholds,
    evaluate_dataset,
    write_outputs,
)
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.lerobot_v3 import open_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = open_dataset(args.dataset)
    scene = load_scene_config(args.scene_config)
    joints = {
        side: tuple(f"{side}_joint_{index}" for index in range(1, 8))
        for side in ("left", "right")
    }
    model = DualArmUrdfKinematics(
        args.urdf,
        None,
        robot_name="RealMan RM75-B dual-arm visualization",
        arm_joints=joints,
        target_links={"left": "left_tcp", "right": "right_tcp"},
        root_link="dual_base",
    )
    thresholds = Thresholds()
    summary, rows = evaluate_dataset(
        dataset, (args.episode,), scene, model, thresholds)
    write_outputs(args.output, summary, rows)
    result = summary["strategies"]["continuous_midpoint"]
    print(
        "REALMAN_EPISODE_EVALUATED "
        f"both={result['both_reachable_rate']:.6f} "
        f"output={Path(args.output).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
