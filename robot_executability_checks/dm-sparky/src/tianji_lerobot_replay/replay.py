"""Replay orchestration: LeRobot v3 TCP samples → configured Tianji viewer."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import ReplayError
from .frames import SceneConfig, quaternion_xyzw_to_matrix
from .kinematics import TianjiKinematicReplayer
from .lerobot_v3 import LeRobotV3Dataset, TcpEpisode, load_episode
from .scene import ReplayScene, build_scene
from .sim_registration import (
    VirtualSceneRegistration,
    calibrate_current_simulation,
    write_virtual_scene_registration,
)
from .viewer import ViewerTrack, run_viewer


@dataclass
class ReplaySession:
    """A prepared visual scene and a lazy LeRobot episode-to-viewer adapter."""

    dataset: LeRobotV3Dataset
    config: SceneConfig
    scene: ReplayScene

    def __post_init__(self) -> None:
        self._solver = TianjiKinematicReplayer(self.scene.model)

    def load_viewer_track(self, episode_index: int, *, verbose: bool = True) -> ViewerTrack:
        source = load_episode(self.dataset, episode_index)
        if verbose:
            print(f"正在拟合 episode {episode_index}（{source.frame_count} 帧）…")
        return self._fit_episode(source, verbose=verbose)

    def _fit_episode(self, source: TcpEpisode, *, verbose: bool) -> ViewerTrack:
        rotations_umi = quaternion_xyzw_to_matrix(source.quaternions_world_xyzw)
        target_positions = self.config.robot_from_umi_positions(source.positions_world_m)
        target_rotations = self.config.robot_from_umi_rotations(rotations_umi)

        last_progress = -1

        def progress(done: int, total: int) -> None:
            nonlocal last_progress
            if not verbose:
                return
            percent = int(done * 100 / total)
            if percent != last_progress and (percent % 10 == 0 or done == total):
                print(f"  IK {percent}% ({done}/{total})")
                last_progress = percent

        trajectory = self._solver.solve_episode(
            target_positions,
            target_rotations,
            progress=progress,
        )
        track = ViewerTrack(
            episode_index=source.episode_index,
            timestamps_s=source.timestamps_s,
            target_positions_robot_m=target_positions,
            target_rotations_robot=target_rotations,
            kinematics=trajectory,
            fps=self.dataset.fps,
        )
        if verbose:
            report = trajectory.report()
            print(
                "  Aik7 运动学回放残差（仅供画面核对，不是可达性判定）："
                f"平均 {report['mean_position_residual_mm']:.2f} mm / "
                f"{report['mean_orientation_residual_deg']:.2f}°，"
                f"最大 {report['max_position_residual_mm']:.2f} mm / "
                f"{report['max_orientation_residual_deg']:.2f}°。"
            )
            missing = int(report["frames_without_analytic_branch"])
            if missing:
                print(
                    f"  提示：{missing}/{report['total_arm_frames']} 个臂帧没有返回 Aik7 解析分支，"
                    "这些帧保留前一帧关节姿态以便连续查看。"
                )
        return track

    def run(
        self,
        initial_episode: int,
        *,
        play_speed: float,
        paused: bool,
        workspace_cache: str | Path | None = None,
    ) -> None:
        scene = self.scene
        if workspace_cache is not None:
            scene = _workspace_scene(workspace_cache)
        run_viewer(
            scene,
            episode_ids=self.dataset.episode_ids,
            initial_episode=initial_episode,
            load_track=self.load_viewer_track,
            play_speed=play_speed,
            paused=paused,
            workspace_cache=workspace_cache,
        )


def _workspace_scene(workspace_cache: str | Path) -> ReplayScene:
    """Load the cached Tianji envelope model without changing the replay model."""
    import mujoco

    cache = Path(workspace_cache).expanduser().resolve()
    xml_path = cache.parent / "envelope" / "workspace_envelope.xml"
    if not cache.is_file() or not xml_path.is_file():
        raise ReplayError(
            "workspace 缓存不完整；请先运行 scripts/compute_workspace.py："
            f"{cache}"
        )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    target_mocap_ids = {}
    for side in ("left", "right"):
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"umi_target_{side}")
        if body_id < 0 or model.body_mocapid[body_id] < 0:
            raise ReplayError(f"workspace 场景缺少 {side} TCP target mocap body。")
        target_mocap_ids[side] = int(model.body_mocapid[body_id])
    return ReplayScene(xml_path=xml_path, model=model, target_mocap_ids=target_mocap_ids)


def create_session(
    *,
    dataset: LeRobotV3Dataset,
    urdf: str | Path,
    output_dir: str | Path,
    config: SceneConfig,
    package_roots: Iterable[str | Path] = (),
    scene_config_path: str | Path | None = None,
) -> ReplaySession:
    """Prepare the scene and write its explicit virtual scene registration."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = build_scene(urdf, output / "scene", config, package_roots)
    registration = calibrate_current_simulation(config)
    registration_path = write_virtual_scene_registration(
        output / "virtual_scene_registration.json",
        registration,
        config,
        scene_config_path=scene_config_path,
    )
    session = ReplaySession(dataset, config, scene)
    _write_manifest(
        output / "replay_manifest.json",
        dataset,
        config,
        urdf,
        scene.xml_path,
        registration,
        registration_path,
        session,
    )
    return session


def _write_manifest(
    path: Path,
    dataset: LeRobotV3Dataset,
    config: SceneConfig,
    urdf: str | Path,
    scene_xml: Path,
    registration: VirtualSceneRegistration,
    registration_path: Path,
    session: ReplaySession,
) -> None:
    payload: dict[str, Any] = {
        "purpose": "visual_kinematic_replay_only",
        "does_not_include": [
            "reachability_sampling",
            "collision_checking",
            "velocity_or_acceleration_analysis",
            "joint_range_or_hardware_replay_decisions",
        ],
        "ik": session._solver.backend_report(),
        "dataset": dataset.report(),
        "scene": config.report(),
        "urdf": str(Path(urdf).expanduser().resolve()),
        "scene_xml": str(scene_xml),
        "virtual_scene_registration": registration.report(),
        "virtual_scene_registration_json": str(registration_path),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
