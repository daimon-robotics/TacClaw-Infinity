"""Whole-body kinematic upper-bound evaluation for the wheeled humanoid.

The fixed-base evaluator intentionally freezes knee/waist joints.  This module
adds one bounded longitudinal base coordinate plus the three shared torso
joints to a coupled bimanual least-squares problem.  It does not model
collision, dynamics, or base/torso velocity, so its result is an explicitly
labelled kinematic upper bound rather than a replay safety decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .arm_quality import (
    ArmQualityEvaluator,
    HumanoidUrdfKinematics,
    SIDES,
    Thresholds,
    _log_derivative,
    _pose_error,
    _rotation_class,
    _target_matrix,
    _transform,
)
from .frames import SceneConfig, quaternion_xyzw_to_matrix
from .lerobot_v3 import LeRobotV3Dataset, load_episode


TORSO_JOINTS = ("knee_joint", "hip_pitch_joint", "hip_yaw_joint")
DEFAULT_POSTURES = (
    (-0.30, 0.00, 0.30, 0.45),
    (-0.35, 0.00, 0.35, 0.55),
    (-0.20, 0.00, 0.00, 0.00),
    (-0.35, 0.00, 0.40, 0.45),
)


@dataclass(frozen=True)
class MobilityBounds:
    """Search limits in the table frame and in radians."""

    base_y_min_m: float = -0.50
    base_y_max_m: float = -0.05

    def arrays(self, model: HumanoidUrdfKinematics) -> tuple[np.ndarray, np.ndarray]:
        if not self.base_y_min_m < self.base_y_max_m:
            raise ValueError("base_y_min_m must be smaller than base_y_max_m")
        torso = []
        for name in TORSO_JOINTS:
            if name not in model.all_limits:
                raise ValueError(f"missing effective limit for {name}")
            torso.append(model.all_limits[name])
        left = list(zip(*model.bounds("left")))
        right = list(zip(*model.bounds("right")))
        limits = [(self.base_y_min_m, self.base_y_max_m), *torso, *left, *right]
        return (
            np.asarray([item[0] for item in limits], dtype=float),
            np.asarray([item[1] for item in limits], dtype=float),
        )


@dataclass(frozen=True)
class WholeBodyIkResult:
    x: np.ndarray
    left_position_error_m: float
    left_orientation_error_rad: float
    right_position_error_m: float
    right_orientation_error_rad: float
    left_reachable: bool
    right_reachable: bool
    seed_name: str
    evaluations: int

    @property
    def both_reachable(self) -> bool:
        return self.left_reachable and self.right_reachable


class HumanoidMobilityEvaluator:
    """Coupled bimanual IK with a longitudinal base and shared torso joints."""

    def __init__(
        self,
        model: HumanoidUrdfKinematics,
        thresholds: Thresholds,
        scene: SceneConfig,
        bounds: MobilityBounds = MobilityBounds(),
    ) -> None:
        thresholds.validate()
        self.model = model
        self.thresholds = thresholds
        self.scene = scene
        self.bounds_spec = bounds
        self.lower, self.upper = bounds.arrays(model)
        self.scale = self.upper - self.lower
        self.reference_base_y_m = float(scene.robot_root_in_table_m[1])

    @staticmethod
    def _arm_slice(side: str) -> slice:
        return slice(4, 11) if side == "left" else slice(11, 18)

    def adjusted_target(self, target: np.ndarray, base_y_m: float) -> np.ndarray:
        adjusted = target.copy()
        # With the configured robot/table rotation, moving the root toward
        # table +Y shifts every target toward robot -X by the same amount.
        adjusted[0, 3] += self.reference_base_y_m - float(base_y_m)
        return adjusted

    def _forward_extended(
        self, side: str, arm_q: np.ndarray, torso_q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        names = (*TORSO_JOINTS, *self.model.arm_joints[side])
        values = dict(zip(TORSO_JOINTS, np.asarray(torso_q, dtype=float)))
        values.update(zip(self.model.arm_joints[side], np.asarray(arm_q, dtype=float)))
        active_names = set(names)
        active: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
        current = np.eye(4, dtype=float)
        for joint in self.model.chains[side]:
            current = current @ joint["origin"]
            name = joint["name"]
            joint_type = joint["type"]
            if joint_type in ("revolute", "continuous"):
                if name in active_names:
                    active[name] = (
                        current[:3, 3].copy(),
                        current[:3, :3] @ joint["axis"],
                        joint_type,
                    )
                current = current @ _transform(
                    rotation=_rotation_class().from_rotvec(
                        joint["axis"] * float(values.get(name, 0.0))))
            elif joint_type == "prismatic":
                if name in active_names:
                    active[name] = (
                        current[:3, 3].copy(),
                        current[:3, :3] @ joint["axis"],
                        joint_type,
                    )
                current = current @ _transform(
                    joint["axis"] * float(values.get(name, 0.0)))
        endpoint = current[:3, 3]
        jacobian = np.zeros((6, len(names)), dtype=float)
        for index, name in enumerate(names):
            point, axis, joint_type = active[name]
            if joint_type == "prismatic":
                jacobian[:3, index] = axis
            else:
                jacobian[:3, index] = np.cross(axis, endpoint - point)
                jacobian[3:, index] = axis
        return current, jacobian

    def _solve_once(
        self,
        targets: dict[str, np.ndarray],
        seed: np.ndarray,
        seed_name: str,
        regularization: float,
    ) -> WholeBodyIkResult:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("scipy.optimize is required") from exc
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)

        def residual(x: np.ndarray) -> np.ndarray:
            torso = x[1:4]
            values = []
            for side in SIDES:
                current, _ = self._forward_extended(
                    side, x[self._arm_slice(side)], torso)
                target = self.adjusted_target(targets[side], x[0])
                _, _, rotation = _pose_error(current, target)
                values.extend((
                    current[:3, 3] - target[:3, 3],
                    self.thresholds.orientation_weight_m_per_rad * rotation,
                ))
            values.append(regularization * (x - seed) / self.scale)
            return np.concatenate(values)

        def jacobian(x: np.ndarray) -> np.ndarray:
            torso = x[1:4]
            blocks = []
            for side in SIDES:
                current, geometric = self._forward_extended(
                    side, x[self._arm_slice(side)], torso)
                target = self.adjusted_target(targets[side], x[0])
                _, _, rotation = _pose_error(current, target)
                full = np.zeros((6, 18), dtype=float)
                full[0, 0] = 1.0
                full[:3, 1:4] = geometric[:3, :3]
                full[:3, self._arm_slice(side)] = geometric[:3, 3:]
                angular_target = target[:3, :3].T @ geometric[3:, :]
                orientation = _log_derivative(rotation) @ angular_target
                full[3:, 1:4] = (
                    self.thresholds.orientation_weight_m_per_rad
                    * orientation[:, :3]
                )
                full[3:, self._arm_slice(side)] = (
                    self.thresholds.orientation_weight_m_per_rad
                    * orientation[:, 3:]
                )
                blocks.append(full)
            blocks.append(np.diag(np.full(18, regularization) / self.scale))
            return np.vstack(blocks)

        result = least_squares(
            residual,
            seed,
            jac=jacobian,
            bounds=(self.lower, self.upper),
            max_nfev=self.thresholds.ik_max_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        errors: dict[str, tuple[float, float]] = {}
        for side in SIDES:
            current, _ = self._forward_extended(
                side, result.x[self._arm_slice(side)], result.x[1:4])
            errors[side] = _pose_error(
                current, self.adjusted_target(targets[side], result.x[0]))[:2]
        reachable = {
            side: bool(
                errors[side][0] <= self.thresholds.position_tolerance_m
                and errors[side][1] <= self.thresholds.orientation_tolerance_rad
            )
            for side in SIDES
        }
        return WholeBodyIkResult(
            x=result.x.copy(),
            left_position_error_m=errors["left"][0],
            left_orientation_error_rad=errors["left"][1],
            right_position_error_m=errors["right"][0],
            right_orientation_error_rad=errors["right"][1],
            left_reachable=reachable["left"],
            right_reachable=reachable["right"],
            seed_name=seed_name,
            evaluations=int(result.nfev),
        )

    def seed_from_posture(
        self, targets: dict[str, np.ndarray], posture: tuple[float, float, float, float]
    ) -> np.ndarray:
        base_y, knee, pitch, yaw = posture
        torso = np.asarray((knee, pitch, yaw), dtype=float)
        fixed = dict(zip(TORSO_JOINTS, torso))
        arm_solver = ArmQualityEvaluator(self.model, self.thresholds, fixed)
        values = [base_y, *torso]
        for side in SIDES:
            target = self.adjusted_target(targets[side], base_y)
            result = arm_solver.solve_independent(side, target)
            values.extend(result.q)
        return np.clip(np.asarray(values, dtype=float), self.lower, self.upper)

    def solve(
        self, targets: dict[str, np.ndarray], previous: np.ndarray | None
    ) -> WholeBodyIkResult:
        candidates: list[WholeBodyIkResult] = []
        if previous is not None:
            first = self._solve_once(
                targets, previous, "continuous_whole_body", regularization=7e-4)
            candidates.append(first)
            if first.both_reachable and self.minimum_arm_isotropy(first.x) >= 0.04:
                return first
        for index, posture in enumerate(DEFAULT_POSTURES):
            seed = self.seed_from_posture(targets, posture)
            candidate = self._solve_once(
                targets, seed, f"posture_{index}", regularization=2e-4)
            candidates.append(candidate)
            if candidate.both_reachable and self.minimum_arm_isotropy(candidate.x) >= 0.04:
                return candidate
        reachable = [item for item in candidates if item.both_reachable]
        if reachable:
            return max(reachable, key=lambda item: self.minimum_arm_isotropy(item.x))
        return min(
            candidates,
            key=lambda item: (
                item.left_position_error_m + item.right_position_error_m
                + self.thresholds.orientation_weight_m_per_rad
                * (item.left_orientation_error_rad + item.right_orientation_error_rad)
            ),
        )

    @staticmethod
    def _isotropy(
        jacobian: np.ndarray, orientation_weight: float
    ) -> tuple[float, float, float, float]:
        weighted = jacobian.copy()
        weighted[3:, :] *= orientation_weight
        singular = np.linalg.svd(weighted, compute_uv=False)
        sigma_max = float(singular[0])
        sigma_min = float(singular[-1])
        ratio = sigma_min / sigma_max if sigma_max > 1e-12 else 0.0
        condition = sigma_max / sigma_min if sigma_min > 1e-12 else math.inf
        return sigma_min, sigma_max, ratio, condition

    def arm_metrics(self, side: str, x: np.ndarray) -> dict[str, float]:
        _, extended = self._forward_extended(
            side, x[self._arm_slice(side)], x[1:4])
        arm_only = extended[:, 3:]
        base_column = np.zeros((6, 1), dtype=float)
        base_column[0, 0] = 1.0
        whole_body = np.hstack((base_column, extended))
        sigma_min, sigma_max, ratio, condition = self._isotropy(
            arm_only, self.thresholds.orientation_weight_m_per_rad)
        _, _, whole_ratio, whole_condition = self._isotropy(
            whole_body, self.thresholds.orientation_weight_m_per_rad)
        lower, upper = self.model.bounds(side)
        arm_q = x[self._arm_slice(side)]
        margin = float(np.min(np.minimum(
            (arm_q - lower) / (upper - lower),
            (upper - arm_q) / (upper - lower),
        )))
        return {
            "joint_margin_fraction": margin,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "isotropy_ratio": ratio,
            "condition_number": condition,
            "whole_body_isotropy_ratio": whole_ratio,
            "whole_body_condition_number": whole_condition,
        }

    def minimum_arm_isotropy(self, x: np.ndarray) -> float:
        return min(self.arm_metrics(side, x)["isotropy_ratio"] for side in SIDES)

    def row(
        self,
        episode_index: int,
        frame_index: int,
        timestamp_s: float,
        targets: dict[str, np.ndarray],
        result: WholeBodyIkResult,
    ) -> dict[str, Any]:
        x = result.x
        row: dict[str, Any] = {
            "strategy": "adaptive_whole_body",
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "base_root_y_table_m": float(x[0]),
            "knee_joint": float(x[1]),
            "hip_pitch_joint": float(x[2]),
            "hip_yaw_joint": float(x[3]),
            "seed": result.seed_name,
            "function_evaluations": result.evaluations,
        }
        for side_index, side in enumerate(SIDES):
            reachable = result.left_reachable if side == "left" else result.right_reachable
            position_error = (
                result.left_position_error_m if side == "left"
                else result.right_position_error_m
            )
            orientation_error = (
                result.left_orientation_error_rad if side == "left"
                else result.right_orientation_error_rad
            )
            target = self.adjusted_target(targets[side], x[0])
            row.update({
                f"{side}_target_x_m": float(target[0, 3]),
                f"{side}_target_y_m": float(target[1, 3]),
                f"{side}_target_z_m": float(target[2, 3]),
                f"{side}_reachable": reachable,
                f"{side}_position_error_m": position_error,
                f"{side}_orientation_error_deg": math.degrees(orientation_error),
            })
            metrics = self.arm_metrics(side, x)
            for key, value in metrics.items():
                row[f"{side}_{key}"] = value if reachable else None
            for index, value in enumerate(x[self._arm_slice(side)], 1):
                row[f"{side}_q{index}"] = float(value)
        row["both_reachable"] = result.both_reachable
        return row


def evaluate_mobility_dataset(
    dataset: LeRobotV3Dataset,
    episode_ids: Iterable[int],
    scene: SceneConfig,
    model: HumanoidUrdfKinematics,
    thresholds: Thresholds,
    bounds: MobilityBounds = MobilityBounds(),
    progress: Any = None,
) -> list[dict[str, Any]]:
    evaluator = HumanoidMobilityEvaluator(model, thresholds, scene, bounds)
    rows: list[dict[str, Any]] = []
    for episode_index in episode_ids:
        source = load_episode(dataset, episode_index)
        positions = scene.robot_from_umi_positions(source.positions_world_m)
        rotations = scene.robot_from_umi_rotations(
            quaternion_xyzw_to_matrix(source.quaternions_world_xyzw))
        previous: np.ndarray | None = None
        for frame in range(source.frame_count):
            targets = {
                side: _target_matrix(positions[frame, side_index], rotations[frame, side_index])
                for side_index, side in enumerate(SIDES)
            }
            result = evaluator.solve(targets, previous)
            previous = result.x.copy()
            rows.append(evaluator.row(
                episode_index,
                frame,
                float(source.timestamps_s[frame]),
                targets,
                result,
            ))
            if progress:
                progress(episode_index, frame + 1, source.frame_count)
    return rows
