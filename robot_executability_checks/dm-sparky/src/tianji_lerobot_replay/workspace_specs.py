"""Robot-specific names for collision-free workspace computation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .arm_quality import HUMANOID_ARM_JOINTS, TIANJI_ARM_JOINTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WorkspaceRobotSpec:
    key: str
    title: str
    urdf: Path
    scene_config: Path
    joint_names: dict[str, tuple[str, ...]]
    ee_bodies: dict[str, str]
    ee_sites: dict[str, str]
    reference_body: str
    quality_csv: Path


SPECS = {
    "tianji": WorkspaceRobotSpec(
        key="tianji",
        title="Tianji",
        urdf=PROJECT_ROOT / "models/ros_packages/daimon_tj_fx_description/urdf/"
        "daimon_tj_fx_dual_arm_with_gripper.urdf",
        scene_config=PROJECT_ROOT / "config/base_up_60mm_farther_100mm_scene.json",
        joint_names=TIANJI_ARM_JOINTS,
        ee_bodies={
            "left": "gripper_base_link_L",
            "right": "gripper_base_link_R",
        },
        ee_sites={
            "left": "workspace_left_tcp",
            "right": "workspace_right_tcp",
        },
        reference_body="base_link",
        quality_csv=PROJECT_ROOT / "outputs/full_base_up_60mm_farther_100mm_arm_quality/"
        "continuous_midpoint_frames.csv",
    ),
    "humanoid": WorkspaceRobotSpec(
        key="humanoid",
        title="DM-SPARKY (full-body)",
        urdf=PROJECT_ROOT / "models/ros_packages/dm25_pr02_hr_pd04_eg_20260810/urdf/"
        "dm25_pr02_hr_pd04_eg_20260810.urdf",
        scene_config=PROJECT_ROOT / "config/mobile_humanoid_scene.json",
        joint_names=HUMANOID_ARM_JOINTS,
        ee_bodies={
            "left": "left_gripper_base_link",
            "right": "right_gripper_base_link",
        },
        ee_sites={
            "left": "workspace_left_tcp",
            "right": "workspace_right_tcp",
        },
        # Using the last shared torso body makes the arm-only workspace move
        # with the full-body posture without mixing base/knee/waist ranges into
        # the required seven-dimensional per-arm workspace definition.
        reference_body="hip_yaw_link",
        quality_csv=PROJECT_ROOT / "outputs/full_cup_humanoid_adaptive_mobility/"
        "adaptive_whole_body_frames.csv",
    ),
}


def get_workspace_spec(robot: str) -> WorkspaceRobotSpec:
    try:
        return SPECS[robot]
    except KeyError as exc:
        raise ValueError(f"unsupported robot {robot!r}") from exc
