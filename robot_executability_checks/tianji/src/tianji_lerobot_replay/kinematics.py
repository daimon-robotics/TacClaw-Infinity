"""Analytical Aik7 replay kinematics for Tianji TCP trajectories.

This module deliberately uses the second IK implementation from the Tianji
ROS workspace (``Aik7_ow``, used by ``/start_wbc``), not the Pinocchio/HQP
teleoperation controller.  The recorded pose is a gripper ``base_link`` TCP,
whereas Aik7 solves for ``ee_link``; the fixed TCP-to-ee transform is obtained
from the actual MuJoCo model at startup instead of being hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from .aik7 import Aik7AnalyticSolver, rotation_distance_rad
from .errors import ReplayError


GRIPPER_JOINT_NAMES = (
    "left_endeffector_joint",
    "left_endeffector_mimic_joint",
    "right_endeffector_joint",
    "right_endeffector_mimic_joint",
)
GRIPPER_CAD_QPOS = 0.0
_ELBOW_SIGN = {"left": -1, "right": 1}
_Q7_GRID_STEP_RAD = math.radians(1.0)

# These are model-consistency guards, not reachability thresholds.  Aik7's
# exact-DH equations and the URDF's decimal ``1.5708`` literals differ by only
# a few micrometres / tens of microradians.  Any accepted candidate should be
# far inside these generous bounds after MuJoCo FK.
_CANDIDATE_FK_POSITION_TOL_M = 2e-4
_CANDIDATE_FK_ROTATION_TOL_RAD = math.radians(0.1)
_MODEL_ALIGNMENT_POSITION_TOL_M = 5e-5
_MODEL_ALIGNMENT_ROTATION_TOL_RAD = 1e-3
_FIXED_TCP_CHAIN_POSITION_TOL_M = 1e-8
_FIXED_TCP_CHAIN_ROTATION_TOL_RAD = 1e-7


@dataclass(frozen=True)
class ArmKinematics:
    side: str
    qpos_addresses: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    midpoint: np.ndarray
    tcp_body_id: int
    link7_body_id: int
    ee_body_id: int
    elbow_sign: int


@dataclass(frozen=True)
class KinematicTrajectory:
    """Joint sequence plus model-fit diagnostics for a visual replay."""

    qpos: np.ndarray
    position_residual_m: np.ndarray  # frames, left/right
    orientation_residual_rad: np.ndarray  # frames, left/right
    analytic_branch_found: np.ndarray  # frames, left/right
    analytic_branch_count: np.ndarray  # frames, left/right
    analytic_elbow_sign: np.ndarray  # frames, left/right; 0 when no branch

    def report(self) -> dict[str, float | int]:
        found = np.asarray(self.analytic_branch_found, dtype=bool)
        return {
            "mean_position_residual_mm": float(np.mean(self.position_residual_m) * 1000.0),
            "max_position_residual_mm": float(np.max(self.position_residual_m) * 1000.0),
            "mean_orientation_residual_deg": float(
                np.mean(self.orientation_residual_rad) * 180.0 / math.pi
            ),
            "max_orientation_residual_deg": float(
                np.max(self.orientation_residual_rad) * 180.0 / math.pi
            ),
            "analytic_branch_frames": int(np.count_nonzero(found)),
            "total_arm_frames": int(found.size),
            "frames_without_analytic_branch": int(found.size - np.count_nonzero(found)),
            "alternate_elbow_arm_frames": int(
                np.count_nonzero(
                    (self.analytic_elbow_sign != 0)
                    & (self.analytic_elbow_sign != np.array([-1, 1], dtype=int))
                )
            ),
        }


class TianjiKinematicReplayer:
    """Replay with the ROS WBC path's Aik7 closed-form IK core.

    The solver preserves the preceding frame's joint branch.  It first tries
    the preceding seventh-axis value, then uses the original WBC-style 1° q7
    scan only when that branch has no analytical solution.  This avoids the
    discontinuous seed-restart behaviour of the former DLS implementation.

    This remains visualization-only code: it does not inspect contacts,
    calculate a workspace, classify safety, or issue a real-robot command.
    """

    def __init__(self, model: Any) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover
            raise ReplayError("缺少 mujoco；请先运行 ./setup.sh。") from exc
        self.mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_body_id < 0:
            raise ReplayError("MuJoCo 模型缺少 base_link，无法建立 Aik7 的基座坐标系。")
        self.arms = {side: self._make_arm(side) for side in ("left", "right")}
        self.gripper_qpos_addresses = self._gripper_qpos_addresses()
        self.aik7 = Aik7AnalyticSolver()
        self.reset_pose()
        self.link7_from_tcp, self.link7_from_ee = self._capture_tcp_flange_chain()
        self._validate_fixed_tcp_chain()
        self._validate_analytic_model_alignment()
        self.reset_pose()

    def _make_arm(self, side: str) -> ArmKinematics:
        suffix = "L" if side == "left" else "R"
        joint_ids: list[int] = []
        for index in range(1, 8):
            name = f"Joint{index}_{suffix}"
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ReplayError(f"MuJoCo 模型缺少 {name!r}，不是预期的天机双臂模型。")
            joint_ids.append(joint_id)
        ids = np.asarray(joint_ids, dtype=int)
        if not np.all(self.model.jnt_limited[ids]):
            raise ReplayError(f"{side} 天机臂存在无关节限位的轴，无法运行 Aik7。")
        tcp_body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"gripper_base_link_{suffix}"
        )
        link7_body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"Link7_{suffix}"
        )
        ee_body_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, f"ee_link_{suffix}"
        )
        if tcp_body_id < 0 or link7_body_id < 0 or ee_body_id < 0:
            raise ReplayError(
                f"MuJoCo 模型缺少 {side} 的 Link7、ee_link 或 gripper_base_link；"
                "无法执行 TCP → 法兰 → ee_link 转换。"
            )
        lower = self.model.jnt_range[ids, 0].copy()
        upper = self.model.jnt_range[ids, 1].copy()
        return ArmKinematics(
            side=side,
            qpos_addresses=self.model.jnt_qposadr[ids].astype(int),
            lower=lower,
            upper=upper,
            midpoint=(lower + upper) / 2.0,
            tcp_body_id=int(tcp_body_id),
            link7_body_id=int(link7_body_id),
            ee_body_id=int(ee_body_id),
            elbow_sign=_ELBOW_SIGN[side],
        )

    def _gripper_qpos_addresses(self) -> np.ndarray:
        addresses: list[int] = []
        for name in GRIPPER_JOINT_NAMES:
            joint_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ReplayError(f"MuJoCo 模型缺少夹爪关节 {name!r}。")
            lower, upper = self.model.jnt_range[joint_id]
            if not lower <= GRIPPER_CAD_QPOS <= upper:
                raise ReplayError(
                    f"夹爪 CAD 参考位 0.0 超出 {name} 的范围 [{lower}, {upper}]。"
                )
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=int)

    def reset_pose(self) -> None:
        """Set arm joints to midpoint and freeze both CAD grippers at q=0."""
        self.mujoco.mj_resetData(self.model, self.data)
        for arm in self.arms.values():
            self.data.qpos[arm.qpos_addresses] = arm.midpoint
        self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
        self.mujoco.mj_forward(self.model, self.data)

    def backend_report(self) -> dict[str, object]:
        """Return the transform provenance written to replay manifests."""
        return {
            "backend": "Aik7_ow analytical IK from ROS /start_wbc path",
            "not_used": "Pinocchio/HQP teleoperation IK",
            "target_chain": "gripper_base_link TCP -> Link7 flange -> ee_link",
            "branch_policy": (
                "first frame uses WBC preferred elbow root (left -, right +); subsequent "
                "frames first retain the previous elbow root; previous q7 first; 1-degree q7 "
                "scan only when needed; alternate elbow root only if the preferred root has "
                "no branch; nearest prior q branch"
            ),
            "tcp_to_link7": {
                side: self.link7_from_tcp[side].tolist() for side in ("left", "right")
            },
            "link7_to_ee": {
                side: self.link7_from_ee[side].tolist() for side in ("left", "right")
            },
        }

    def _body_transform(self, body_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body_id]
        return transform

    def _capture_tcp_flange_chain(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Read the fixed ``TCP → Link7 → ee`` chain from the MuJoCo URDF."""
        link7_from_tcp: dict[str, np.ndarray] = {}
        link7_from_ee: dict[str, np.ndarray] = {}
        for side, arm in self.arms.items():
            world_from_link7 = self._body_transform(arm.link7_body_id)
            link7_from_tcp[side] = np.linalg.inv(world_from_link7) @ self._body_transform(
                arm.tcp_body_id
            )
            link7_from_ee[side] = np.linalg.inv(world_from_link7) @ self._body_transform(
                arm.ee_body_id
            )
        return link7_from_tcp, link7_from_ee

    def _validate_fixed_tcp_chain(self) -> None:
        """Ensure ee→TCP is really fixed for the supplied combined URDF."""
        perturbations = {
            "left": np.array([0.17, -0.21, 0.13, -0.19, 0.11, 0.08, -0.09]),
            "right": np.array([-0.16, 0.20, -0.12, 0.18, -0.10, -0.07, 0.08]),
        }
        for side, arm in self.arms.items():
            self.reset_pose()
            self.data.qpos[arm.qpos_addresses] = np.clip(
                arm.midpoint + perturbations[side], arm.lower, arm.upper
            )
            self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
            self.mujoco.mj_forward(self.model, self.data)
            world_from_link7 = self._body_transform(arm.link7_body_id)
            observed_transforms = {
                "Link7 → TCP": np.linalg.inv(world_from_link7)
                @ self._body_transform(arm.tcp_body_id),
                "Link7 → ee": np.linalg.inv(world_from_link7)
                @ self._body_transform(arm.ee_body_id),
            }
            references = {
                "Link7 → TCP": self.link7_from_tcp[side],
                "Link7 → ee": self.link7_from_ee[side],
            }
            for name, observed in observed_transforms.items():
                reference = references[name]
                position_error = float(np.linalg.norm(observed[:3, 3] - reference[:3, 3]))
                rotation_error = rotation_distance_rad(observed[:3, :3], reference[:3, :3])
                if (
                    position_error > _FIXED_TCP_CHAIN_POSITION_TOL_M
                    or rotation_error > _FIXED_TCP_CHAIN_ROTATION_TOL_RAD
                ):
                    raise ReplayError(
                        f"{side} {name} 不是固定变换；不能将 TCP 目标交给 Aik7。"
                    )

    def _validate_analytic_model_alignment(self) -> None:
        """Refuse a URDF whose 7-axis FK no longer matches the Aik7 backend."""
        samples = {
            "left": np.array([0.25, -0.55, 0.48, -1.00, 0.35, 0.15, -0.40]),
            "right": np.array([0.25, 0.55, -0.48, 0.50, -0.35, -0.15, 0.40]),
        }
        for side, arm in self.arms.items():
            sample = samples[side]
            if np.any(sample < arm.lower) or np.any(sample > arm.upper):
                raise ReplayError(f"内置的 {side} Aik7 FK 校验姿态超出当前 URDF 的关节范围。")
            self.reset_pose()
            self.data.qpos[arm.qpos_addresses] = sample
            self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
            self.mujoco.mj_forward(self.model, self.data)
            mujoco_transform = self._body_transform(arm.ee_body_id)
            analytic_transform = self.aik7.forward_base_ee(side, sample)
            position_error = float(
                np.linalg.norm(mujoco_transform[:3, 3] - analytic_transform[:3, 3])
            )
            rotation_error = rotation_distance_rad(
                mujoco_transform[:3, :3], analytic_transform[:3, :3]
            )
            if (
                position_error > _MODEL_ALIGNMENT_POSITION_TOL_M
                or rotation_error > _MODEL_ALIGNMENT_ROTATION_TOL_RAD
            ):
                raise ReplayError(
                    f"当前 URDF 与 Aik7 解析模型不一致（{side}: "
                    f"{position_error * 1000.0:.3f} mm / "
                    f"{math.degrees(rotation_error):.4f}°）；"
                    "请使用本项目随附的完整天机 URDF，或重新生成 Aik7 模型。"
                )

    def solve_episode(
        self,
        target_positions: np.ndarray,
        target_rotations: np.ndarray,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> KinematicTrajectory:
        """Solve a continuous dual-arm sequence for absolute TCP targets."""
        positions = np.asarray(target_positions, dtype=float)
        rotations = np.asarray(target_rotations, dtype=float)
        if positions.ndim != 3 or positions.shape[1:] != (2, 3):
            raise ReplayError(f"TCP 目标位置应为 (frames,2,3)，实际为 {positions.shape}。")
        if rotations.shape != positions.shape[:2] + (3, 3):
            raise ReplayError(f"TCP 目标姿态应为 (frames,2,3,3)，实际为 {rotations.shape}。")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
            raise ReplayError("TCP 目标含有 NaN 或 Inf。")

        self.reset_pose()
        frames = positions.shape[0]
        qpos = np.empty((frames, self.model.nq), dtype=float)
        position_residual = np.empty((frames, 2), dtype=float)
        orientation_residual = np.empty((frames, 2), dtype=float)
        branch_found = np.zeros((frames, 2), dtype=bool)
        branch_count = np.zeros((frames, 2), dtype=int)
        elbow_sign = np.zeros((frames, 2), dtype=int)
        previous_elbow_sign = {side: arm.elbow_sign for side, arm in self.arms.items()}
        for frame in range(frames):
            for side_index, side in enumerate(("left", "right")):
                position_error, orientation_error, found, count, selected_elbow = self._solve_arm(
                    side,
                    positions[frame, side_index],
                    rotations[frame, side_index],
                    preferred_elbow_sign=previous_elbow_sign[side],
                )
                position_residual[frame, side_index] = position_error
                orientation_residual[frame, side_index] = orientation_error
                branch_found[frame, side_index] = found
                branch_count[frame, side_index] = count
                elbow_sign[frame, side_index] = selected_elbow
                if found:
                    previous_elbow_sign[side] = selected_elbow
            self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
            self.mujoco.mj_forward(self.model, self.data)
            qpos[frame] = self.data.qpos
            if progress is not None:
                progress(frame + 1, frames)
        return KinematicTrajectory(
            qpos=qpos,
            position_residual_m=position_residual,
            orientation_residual_rad=orientation_residual,
            analytic_branch_found=branch_found,
            analytic_branch_count=branch_count,
            analytic_elbow_sign=elbow_sign,
        )

    def _solve_arm(
        self,
        side: str,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        preferred_elbow_sign: int,
    ) -> tuple[float, float, bool, int, int]:
        arm = self.arms[side]
        target_position = np.asarray(target_position, dtype=float)
        target_rotation = np.asarray(target_rotation, dtype=float)
        if target_position.shape != (3,) or target_rotation.shape != (3, 3):
            raise ReplayError(f"{side} TCP target 的位置或姿态维度无效。")
        target_tcp_world = np.eye(4, dtype=float)
        target_tcp_world[:3, :3] = target_rotation
        target_tcp_world[:3, 3] = target_position
        # The data TCP is ``gripper_base_link``.  Aik7's second WBC backend
        # instead consumes ``ee_link``.  Keep the full chain explicit:
        # TCP -> Link7 (flange) -> ee_link (Aik7 endpoint).
        base_from_world = np.linalg.inv(self._body_transform(self.base_body_id))
        target_tcp_base = base_from_world @ target_tcp_world
        target_link7_base = target_tcp_base @ np.linalg.inv(self.link7_from_tcp[side])
        target_ee = target_link7_base @ self.link7_from_ee[side]
        start_qpos = self.data.qpos.copy()
        previous_q = start_qpos[arm.qpos_addresses].copy()

        # Try the preceding redundant coordinate first.  In ordinary playback
        # this succeeds, so it gives a continuous solution without a grid
        # restart.  Only then fall back to the WBC's original 1° q7 scan.
        q7_values = self._ordered_q7_values(arm, float(previous_q[6]))
        if preferred_elbow_sign not in (-1, 1):
            raise ReplayError("Aik7 肘部分支必须是 -1 或 +1。")
        # Start with the prior frame's chosen root.  The first frame receives
        # the ROS WBC convention (left negative, right positive); after an
        # explicitly marked alternate root is necessary, retaining it avoids
        # jumping back to a newly available but remote default branch.
        for selected_elbow in (preferred_elbow_sign, -preferred_elbow_sign):
            candidates: list[tuple[np.ndarray, float, float]] = []
            for attempt, q7 in enumerate(q7_values):
                branches = self.aik7.solve_base_ee(
                    side, target_ee, q7, elbow_sign=selected_elbow
                )
                candidates.extend(
                    self._mujoco_validated_candidates(
                        arm, branches, start_qpos, target_tcp_world
                    )
                )
                if attempt == 0 and candidates:
                    # q7=current has a valid branch: no global redundancy
                    # scan is necessary, avoiding avoidable branch jumps.
                    break
            if candidates:
                def score(candidate: tuple[np.ndarray, float, float]) -> tuple[float, float, float]:
                    q_candidate, position_error, orientation_error = candidate
                    distance = float(np.linalg.norm(q_candidate - previous_q))
                    return distance, position_error, orientation_error

                selected_q, position_error, orientation_error = min(candidates, key=score)
                self.data.qpos[:] = start_qpos
                self.data.qpos[arm.qpos_addresses] = selected_q
                self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
                self.mujoco.mj_forward(self.model, self.data)
                return position_error, orientation_error, True, len(candidates), selected_elbow

        # Keep the last continuous state rather than replacing it with an
        # arbitrary numerical seed.  The viewer reports the raw residual and
        # says that no Aik7 analytical branch was returned for this frame.
        self.data.qpos[:] = start_qpos
        self.mujoco.mj_forward(self.model, self.data)
        position_error, orientation_error = self._tcp_residual(arm, target_tcp_world)
        return position_error, orientation_error, False, 0, 0

    def _ordered_q7_values(self, arm: ArmKinematics, current_q7: float) -> tuple[float, ...]:
        clipped_current = float(np.clip(current_q7, arm.lower[6], arm.upper[6]))
        low_index = math.ceil(arm.lower[6] / _Q7_GRID_STEP_RAD)
        high_index = math.floor(arm.upper[6] / _Q7_GRID_STEP_RAD)
        grid = [index * _Q7_GRID_STEP_RAD for index in range(low_index, high_index + 1)]
        grid.extend((float(arm.lower[6]), float(arm.upper[6])))
        ordered = [clipped_current]
        for value in sorted(grid, key=lambda candidate: abs(candidate - clipped_current)):
            if not any(abs(value - existing) < 1e-10 for existing in ordered):
                ordered.append(float(value))
        return tuple(ordered)

    def _mujoco_validated_candidates(
        self,
        arm: ArmKinematics,
        branches: np.ndarray,
        start_qpos: np.ndarray,
        target_tcp: np.ndarray,
    ) -> list[tuple[np.ndarray, float, float]]:
        accepted: list[tuple[np.ndarray, float, float]] = []
        for q_candidate in np.asarray(branches, dtype=float):
            if q_candidate.shape != (7,) or not np.all(np.isfinite(q_candidate)):
                continue
            if np.any(q_candidate < arm.lower - 1e-8) or np.any(q_candidate > arm.upper + 1e-8):
                continue
            self.data.qpos[:] = start_qpos
            self.data.qpos[arm.qpos_addresses] = q_candidate
            self.data.qpos[self.gripper_qpos_addresses] = GRIPPER_CAD_QPOS
            self.mujoco.mj_forward(self.model, self.data)
            position_error, orientation_error = self._tcp_residual(arm, target_tcp)
            if (
                position_error <= _CANDIDATE_FK_POSITION_TOL_M
                and orientation_error <= _CANDIDATE_FK_ROTATION_TOL_RAD
            ):
                accepted.append((q_candidate.copy(), position_error, orientation_error))
        self.data.qpos[:] = start_qpos
        self.mujoco.mj_forward(self.model, self.data)
        return accepted

    def _tcp_residual(self, arm: ArmKinematics, target_tcp: np.ndarray) -> tuple[float, float]:
        current_tcp = self._body_transform(arm.tcp_body_id)
        position_error = float(np.linalg.norm(current_tcp[:3, 3] - target_tcp[:3, 3]))
        orientation_error = rotation_distance_rad(current_tcp[:3, :3], target_tcp[:3, :3])
        return position_error, orientation_error
