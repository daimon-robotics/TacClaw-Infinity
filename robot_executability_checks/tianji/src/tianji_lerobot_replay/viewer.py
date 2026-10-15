"""Small interactive MuJoCo viewer for a kinematic replay track.

The viewer deliberately has no safety, collision, reachability, or dynamics
logic.  It only shows the recorded TCP target markers together with the joint
sequence fitted by :mod:`tianji_lerobot_replay.kinematics`.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import time
from typing import Callable, Iterable
from pathlib import Path

import numpy as np

from .errors import ReplayError
from .frames import matrix_to_quaternion_xyzw
from .kinematics import GRIPPER_CAD_QPOS, KinematicTrajectory
from .scene import ReplayScene
from .workspace_visualization import WorkspaceOverlay


@dataclass(frozen=True)
class ViewerTrack:
    """A fitted episode plus its original TCP targets in robot coordinates."""

    episode_index: int
    timestamps_s: np.ndarray
    target_positions_robot_m: np.ndarray  # (frames, 2, 3), left/right
    target_rotations_robot: np.ndarray  # (frames, 2, 3, 3), left/right
    kinematics: KinematicTrajectory
    fps: float

    def __post_init__(self) -> None:
        frames = int(self.timestamps_s.size)
        expected_positions = (frames, 2, 3)
        expected_rotations = (frames, 2, 3, 3)
        if frames <= 0:
            raise ReplayError("回放轨迹不能为空。")
        if self.target_positions_robot_m.shape != expected_positions:
            raise ReplayError(
                "查看器 target position shape 无效："
                f"期望 {expected_positions}，实际 {self.target_positions_robot_m.shape}。"
            )
        if self.target_rotations_robot.shape != expected_rotations:
            raise ReplayError(
                "查看器 target rotation shape 无效："
                f"期望 {expected_rotations}，实际 {self.target_rotations_robot.shape}。"
            )
        if self.kinematics.qpos.shape[0] != frames:
            raise ReplayError("查看器 qpos 帧数与 TCP 轨迹帧数不一致。")
        expected_arm_values = (frames, 2)
        if self.kinematics.analytic_branch_found.shape != expected_arm_values:
            raise ReplayError("查看器 Aik7 分支标记的帧数与 TCP 轨迹不一致。")
        if self.kinematics.analytic_branch_count.shape != expected_arm_values:
            raise ReplayError("查看器 Aik7 分支数量的帧数与 TCP 轨迹不一致。")
        if self.kinematics.analytic_elbow_sign.shape != expected_arm_values:
            raise ReplayError("查看器 Aik7 肘部分支标记的帧数与 TCP 轨迹不一致。")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ReplayError("查看器 fps 必须为正的有限数值。")

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_s.size)


TrackLoader = Callable[[int], ViewerTrack]


def run_viewer(
    scene: ReplayScene,
    *,
    episode_ids: Iterable[int],
    initial_episode: int,
    load_track: TrackLoader,
    play_speed: float = 1.0,
    paused: bool = False,
    workspace_cache: str | Path | None = None,
) -> None:
    """Open a passive MuJoCo viewer and replay selected LeRobot episodes.

    Episode loading and IK happen lazily when the user switches episodes.  The
    GLFW callback only queues actions; all MuJoCo state changes remain in the
    main thread under the passive-viewer lock.
    """
    if not np.isfinite(play_speed) or play_speed <= 0.0:
        raise ReplayError("播放倍率必须是正的有限数值。")
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ReplayError("缺少 mujoco；请先运行 ./setup.sh。") from exc

    episodes = tuple(sorted(set(int(value) for value in episode_ids)))
    if not episodes:
        raise ReplayError("没有可供查看的 episode。")
    try:
        episode_cursor = episodes.index(initial_episode)
    except ValueError as exc:
        preview = ", ".join(str(value) for value in episodes[:20])
        raise ReplayError(
            f"找不到 episode {initial_episode}；可用 episode 示例：{preview}。"
        ) from exc

    model = scene.model
    data = mujoco.MjData(model)
    workspace = (
        WorkspaceOverlay(model, workspace_cache, initial_mode=1)
        if workspace_cache is not None else None
    )
    gripper_addresses = _gripper_qpos_addresses(mujoco, model)
    track = load_track(episodes[episode_cursor])
    frame = 0
    playing = not paused
    status = "已加载；这是纯运动学可视化，不执行可达性或碰撞判定。"
    accumulated = 0.0
    actions: queue.SimpleQueue[str] = queue.SimpleQueue()

    def key_callback(key: int) -> None:
        action = _key_action(key)
        if action is not None:
            actions.put(action)

    def apply_frame(viewer: object) -> None:
        """Write one kinematic frame and the two transparent TCP targets."""
        with viewer.lock():
            data.qpos[:] = track.kinematics.qpos[frame]
            data.qpos[gripper_addresses] = GRIPPER_CAD_QPOS
            for side_index, side in enumerate(("left", "right")):
                mocap_id = scene.target_mocap_ids[side]
                data.mocap_pos[mocap_id] = track.target_positions_robot_m[frame, side_index]
                quat_xyzw = matrix_to_quaternion_xyzw(
                    track.target_rotations_robot[frame, side_index]
                )
                data.mocap_quat[mocap_id] = np.array(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float
                )
            data.time = float(track.timestamps_s[frame])
            mujoco.mj_forward(model, data)

    def switch_episode(new_cursor: int) -> bool:
        nonlocal episode_cursor, track, frame, playing, accumulated, status
        next_episode = episodes[new_cursor]
        status = f"正在为 episode {next_episode} 拟合纯运动学回放…"
        try:
            next_track = load_track(next_episode)
        except ReplayError as exc:
            status = f"无法加载 episode {next_episode}：{exc}"
            print(f"提示：{status}")
            return False
        episode_cursor = new_cursor
        track = next_track
        frame = 0
        playing = False
        accumulated = 0.0
        status = "已切换；按 Space 播放。"
        print(f"已切换到 episode {next_episode}（{track.frame_count} 帧）。")
        return True

    print("MuJoCo Tianji/LeRobot 回放窗口已打开；关闭窗口即可退出。")
    print("窗口内：Space 播放/暂停；R 回到首帧；[ 或 , 上一段；] 或 . 下一段。")
    print("红色/蓝色半透明球：记录的左/右 TCP 目标；夹爪固定在厂商 CAD q=0 参考姿态。")

    # MuJoCo invokes key_callback from its GUI thread.  Queueing actions here
    # avoids unsafe simultaneous access to qpos/mocap state.
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=True, show_right_ui=True
    ) as viewer:
        if workspace is not None:
            viewer.opt.geomgroup[3] = 1
        previous_clock = time.perf_counter()
        while viewer.is_running():
            current_clock = time.perf_counter()
            wall_elapsed = current_clock - previous_clock
            previous_clock = current_clock

            while not actions.empty():
                action = actions.get()
                if action == "toggle":
                    playing = not playing
                    status = "正在播放。" if playing else "已暂停。"
                elif action == "restart":
                    frame = 0
                    accumulated = 0.0
                    playing = False
                    status = "已回到第 1 帧；按 Space 播放。"
                elif action == "previous_episode":
                    switch_episode((episode_cursor - 1) % len(episodes))
                elif action == "next_episode":
                    switch_episode((episode_cursor + 1) % len(episodes))
                elif action == "workspace_mode" and workspace is not None:
                    workspace.cycle_mode()
                elif action == "workspace_scope" and workspace is not None:
                    workspace.toggle_scope()

            if playing and frame < track.frame_count - 1:
                accumulated += wall_elapsed * play_speed
                while frame < track.frame_count - 1:
                    interval = _playback_interval(track.timestamps_s, frame, track.fps)
                    if accumulated < interval:
                        break
                    accumulated -= interval
                    frame += 1
                if frame == track.frame_count - 1:
                    playing = False
                    accumulated = 0.0
                    status = "已到最后一帧；按 R 回放。"

            apply_frame(viewer)
            draw = {"requested": 0, "displayed": 0}
            if workspace is not None:
                with viewer.lock():
                    draw = workspace.update_user_scene(viewer.user_scn, data)
            viewer.set_texts(_overlay(
                track, frame, playing, status, workspace=workspace, draw=draw))
            viewer.sync()
            time.sleep(0.001)


def _gripper_qpos_addresses(mujoco: object, model: object) -> np.ndarray:
    names = (
        "left_endeffector_joint",
        "left_endeffector_mimic_joint",
        "right_endeffector_joint",
        "right_endeffector_mimic_joint",
    )
    addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ReplayError(f"场景缺少夹爪关节 {name!r}。")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def _key_action(key: int) -> str | None:
    return {
        32: "toggle",  # GLFW_KEY_SPACE
        ord("r"): "restart",
        ord("R"): "restart",
        ord("["): "previous_episode",
        ord(","): "previous_episode",
        ord("]"): "next_episode",
        ord("."): "next_episode",
        ord("w"): "workspace_mode",
        ord("W"): "workspace_mode",
        ord("o"): "workspace_scope",
        ord("O"): "workspace_scope",
    }.get(key)


def _playback_interval(timestamps_s: np.ndarray, frame: int, fps: float) -> float:
    if frame >= timestamps_s.size - 1:
        return 1.0 / fps
    interval = float(timestamps_s[frame + 1] - timestamps_s[frame])
    return interval if np.isfinite(interval) and interval > 1e-7 else 1.0 / fps


def _overlay(
    track: ViewerTrack,
    frame: int,
    playing: bool,
    status: str,
    *,
    workspace: WorkspaceOverlay | None = None,
    draw: dict[str, int] | None = None,
) -> tuple[None, None, str, str]:
    elapsed = float(track.timestamps_s[frame])
    duration = float(track.timestamps_s[-1])
    state = "播放中" if playing else "暂停"
    position_mm = track.kinematics.position_residual_m[frame] * 1000.0
    orientation_deg = np.degrees(track.kinematics.orientation_residual_rad[frame])
    branch_found = track.kinematics.analytic_branch_found[frame]
    branch_count = track.kinematics.analytic_branch_count[frame]
    elbow_sign = track.kinematics.analytic_elbow_sign[frame]
    default_sign = (-1, 1)
    branch_text = " | ".join(
        (
            _branch_label("L", branch_found[0], branch_count[0], elbow_sign[0], default_sign[0]),
            _branch_label("R", branch_found[1], branch_count[1], elbow_sign[1], default_sign[1]),
        )
    )
    left = (
        "TIANJI LEROBOT REPLAY\n"
        f"episode {track.episode_index} | frame {frame + 1}/{track.frame_count}\n"
        f"{elapsed:.2f}/{duration:.2f} s | {state}\n"
        f"本帧 IK 拟合残差：L {position_mm[0]:.2f} mm / {orientation_deg[0]:.2f}°，"
        f"R {position_mm[1]:.2f} mm / {orientation_deg[1]:.2f}°\n"
        f"Aik7：{branch_text}\n"
        "说明：仅显示运动学拟合；没有可达性、碰撞或真机执行结论。\n"
        f"{status}"
    )
    if workspace is not None:
        draw = draw or {"requested": 0, "displayed": 0}
        left += (
            f"\nWorkspace: {workspace.mode_name} | {workspace.scope_name} | "
            f"shell {draw['displayed']}/{draw['requested']}"
        )
    right = (
        "Space: play / pause\n"
        "R: first frame\n"
        "[ or , : previous episode\n"
        "] or . : next episode\n"
        "red / blue marker: recorded left / right TCP"
    )
    if workspace is not None:
        right += "\nW: workspace mode\nO: all / overlap only"
    return None, None, left, right


def _branch_label(
    side: str, found: bool, count: int, elbow_sign: int, default_sign: int
) -> str:
    if not found:
        return f"{side} 无解析分支，保持前帧"
    sign = "+" if elbow_sign > 0 else "-"
    suffix = "（备用肘部分支）" if elbow_sign != default_sign else ""
    return f"{side} 分支 {count} / 肘部 {sign}{suffix}"
