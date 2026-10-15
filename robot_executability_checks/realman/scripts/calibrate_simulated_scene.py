#!/usr/bin/env python3
"""Export the current MuJoCo scene's virtual base_link ← UMI registration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay import ReplayError  # noqa: E402
from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.sim_registration import (  # noqa: E402
    calibrate_current_simulation,
    write_virtual_scene_registration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "根据当前 MuJoCo 场景配置导出 base_link←UMI world 的虚拟场景配准。"
            "它不是物理相机手眼标定，也不读取旧 YAML。"
        )
    )
    parser.add_argument(
        "--scene-config",
        default=str(PROJECT_ROOT / "config" / "default_scene.json"),
        help="包含实测标定板尺寸和当前仿真桌/机器人摆位的 JSON。",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "replay" / "virtual_scene_registration.json"),
        help="导出的虚拟场景配准 JSON。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_scene_config(args.scene_config)
        registration = calibrate_current_simulation(config)
        output = write_virtual_scene_registration(
            args.output,
            registration,
            config,
            scene_config_path=args.scene_config,
        )
        report = registration.report()
        print(f"已写入虚拟场景配准：{output}")
        print("parent=base_link, child=umi_world")
        print("translation_base_from_umi_m=" + str(report["translation_base_from_umi_m"]))
        print("rotation_quaternion_xyzw=" + str(report["rotation_quaternion_xyzw"]))
        print(
            "virtual fit rms/max="
            f"{report['fit_rms_m']:.3e}/{report['fit_max_m']:.3e} m；"
            "physical_handeye=false"
        )
        return 0
    except (ReplayError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
