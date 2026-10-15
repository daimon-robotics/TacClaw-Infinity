"""Fixed-body dual-arm reachability and Jacobian manipulability evaluation.

The numerical evaluator is shared by the fixed-base Tianji and the wheeled
mobile humanoid.  Robot-specific joint/link names live in the thin model
classes below; the IK, Jacobian metrics and reporting code remain identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np

from .errors import ReplayError
from .frames import SceneConfig, quaternion_xyzw_to_matrix
from .lerobot_v3 import LeRobotV3Dataset, load_episode


SIDES = ("left", "right")
HUMANOID_ARM_JOINTS = {
    side: tuple(f"arm_{side}_{index}_joint" for index in range(1, 8))
    for side in SIDES
}
HUMANOID_TARGET_LINKS = {
    "left": "left_gripper_base_link",
    "right": "right_gripper_base_link",
}
TIANJI_ARM_JOINTS = {
    "left": tuple(f"Joint{index}_L" for index in range(1, 8)),
    "right": tuple(f"Joint{index}_R" for index in range(1, 8)),
}
TIANJI_TARGET_LINKS = {
    "left": "gripper_base_link_L",
    "right": "gripper_base_link_R",
}


def _rotation_class():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ReplayError("缺少 scipy；请先运行 ./setup.sh。") from exc
    return Rotation


def _rotation_matrix(rotation: Any) -> np.ndarray:
    return rotation.as_matrix() if hasattr(rotation, "as_matrix") else rotation.as_dcm()


def _rotation_from_matrix(matrix: np.ndarray) -> Any:
    rotation = _rotation_class()
    return (rotation.from_matrix(matrix) if hasattr(rotation, "from_matrix")
            else rotation.from_dcm(matrix))


def _transform(position: Iterable[float] = (0.0, 0.0, 0.0), rotation: Any = None) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = np.asarray(position, dtype=float)
    if rotation is not None:
        result[:3, :3] = _rotation_matrix(rotation)
    return result


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    values = default if text is None else tuple(float(item) for item in text.split())
    if len(values) != 3:
        raise ReplayError(f"URDF xyz/rpy/axis 必须有 3 个值：{text!r}")
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class Thresholds:
    position_tolerance_m: float = 0.005
    orientation_tolerance_deg: float = 3.0
    orientation_weight_m_per_rad: float = 0.30
    isotropy_ratio_low: float = 0.04
    minimum_both_reachable_rate: float = 0.99
    maximum_low_manipulability_rate: float = 0.05
    ik_max_nfev: int = 120

    def validate(self) -> None:
        positive = (
            self.position_tolerance_m,
            self.orientation_tolerance_deg,
            self.orientation_weight_m_per_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ReplayError("位置/姿态容差和姿态权重必须为正有限数。")
        for value in (
            self.isotropy_ratio_low,
            self.minimum_both_reachable_rate,
            self.maximum_low_manipulability_rate,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReplayError("比例阈值必须位于 [0,1]。")
        if self.ik_max_nfev <= 0:
            raise ReplayError("IK 最大迭代次数必须大于 0。")

    @property
    def orientation_tolerance_rad(self) -> float:
        return math.radians(self.orientation_tolerance_deg)

    def report(self) -> dict[str, Any]:
        return {
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_deg": self.orientation_tolerance_deg,
            "orientation_weight_m_per_rad": self.orientation_weight_m_per_rad,
            "isotropy_ratio_low": self.isotropy_ratio_low,
            "minimum_both_reachable_rate": self.minimum_both_reachable_rate,
            "maximum_low_manipulability_rate": self.maximum_low_manipulability_rate,
            "ik_max_nfev": self.ik_max_nfev,
        }


class DualArmUrdfKinematics:
    """Read a full URDF and expose exactly two configured 7-DoF arm chains."""

    def __init__(
        self,
        urdf: str | Path,
        joint_params: str | Path | None,
        *,
        robot_name: str,
        arm_joints: dict[str, tuple[str, ...]],
        target_links: dict[str, str],
        root_link: str = "base_link",
    ) -> None:
        if set(arm_joints) != set(SIDES) or set(target_links) != set(SIDES):
            raise ReplayError("双臂模型必须同时定义 left/right 关节和 TCP link。")
        if any(len(arm_joints[side]) != 7 for side in SIDES):
            raise ReplayError("双臂模型的每只手必须正好定义 7 个关节。")
        if len({name for side in SIDES for name in arm_joints[side]}) != 14:
            raise ReplayError("左右 14 个手臂关节名必须互不重复。")
        self.robot_name = str(robot_name)
        self.root_link = str(root_link)
        self.arm_joints = {
            side: tuple(str(name) for name in arm_joints[side]) for side in SIDES
        }
        self.target_links = {
            side: str(target_links[side]) for side in SIDES
        }
        self.urdf_path = Path(urdf).expanduser().resolve()
        self.joint_params_path = (
            Path(joint_params).expanduser().resolve() if joint_params else None
        )
        try:
            root = ET.parse(self.urdf_path).getroot()
        except (OSError, ET.ParseError) as exc:
            raise ReplayError(f"无法读取整机 URDF {self.urdf_path}: {exc}") from exc
        self.by_child: dict[str, dict[str, Any]] = {}
        urdf_limits: dict[str, tuple[float, float]] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis_element = element.find("axis")
            axis = _vector(
                axis_element.get("xyz") if axis_element is not None else None,
                (1.0, 0.0, 0.0),
            )
            norm = float(np.linalg.norm(axis))
            if norm > 1e-12:
                axis /= norm
            joint = {
                "name": element.get("name"),
                "type": element.get("type"),
                "parent": element.find("parent").get("link"),
                "child": element.find("child").get("link"),
                "origin": _transform(
                    _vector(origin.get("xyz") if origin is not None else None,
                            (0.0, 0.0, 0.0)),
                    _rotation_class().from_euler(
                        "xyz", _vector(
                            origin.get("rpy") if origin is not None else None,
                            (0.0, 0.0, 0.0))),
                ),
                "axis": axis,
            }
            self.by_child[joint["child"]] = joint
            limit = element.find("limit")
            if limit is not None and limit.get("lower") is not None:
                urdf_limits[joint["name"]] = (
                    float(limit.get("lower")), float(limit.get("upper")))

        runtime_limits: dict[str, tuple[float, float]] = {}
        if self.joint_params_path:
            try:
                params = ET.parse(self.joint_params_path).getroot()
            except (OSError, ET.ParseError) as exc:
                raise ReplayError(
                    f"无法读取运行时关节限位 {self.joint_params_path}: {exc}") from exc
            for child in params:
                position = child.find("limits/position")
                if position is not None:
                    runtime_limits[child.tag] = (
                        float(position.get("low")), float(position.get("high")))

        self.all_limits: dict[str, tuple[float, float]] = {}
        for name, (low, high) in urdf_limits.items():
            if name in runtime_limits:
                low = max(low, runtime_limits[name][0])
                high = min(high, runtime_limits[name][1])
            if low < high:
                self.all_limits[name] = (low, high)

        self.limits: dict[str, tuple[float, float]] = {}
        self.chains: dict[str, list[dict[str, Any]]] = {}
        for side in SIDES:
            self.chains[side] = self._chain(self.target_links[side])
            chain_names = {joint["name"] for joint in self.chains[side]}
            missing = [name for name in self.arm_joints[side] if name not in chain_names]
            if missing:
                raise ReplayError(
                    f"整机 URDF 的 {side} TCP 链缺少：{', '.join(missing)}")
            for name in self.arm_joints[side]:
                if name not in urdf_limits:
                    raise ReplayError(f"整机 URDF 缺少 {name} 的位置限位。")
                if name not in self.all_limits:
                    raise ReplayError(f"{name} 的 URDF/运行时限位交集为空。")
                self.limits[name] = self.all_limits[name]

    def _chain(self, target: str) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        link = target
        # A URDF may contain a fixed synthetic parent above base_link (Tianji
        # uses base_root -> base_link).  Stop at the requested reporting frame
        # instead of blindly walking to the XML tree root.
        while link != self.root_link and link in self.by_child:
            joint = self.by_child[link]
            chain.append(joint)
            link = joint["parent"]
        if link != self.root_link:
            raise ReplayError(f"{target} 不是 {self.root_link} 的后代。")
        chain.reverse()
        return chain

    def bounds(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        values = [self.limits[name] for name in self.arm_joints[side]]
        return (np.asarray([item[0] for item in values]),
                np.asarray([item[1] for item in values]))

    def forward_jacobian(
        self, side: str, q: np.ndarray, fixed_joints: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        values = dict(fixed_joints)
        values.update(zip(self.arm_joints[side], np.asarray(q, dtype=float)))
        current = np.eye(4, dtype=float)
        active: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        arm_names = set(self.arm_joints[side])
        for joint in self.chains[side]:
            current = current @ joint["origin"]
            if joint["type"] in ("revolute", "continuous"):
                if joint["name"] in arm_names:
                    active[joint["name"]] = (
                        current[:3, 3].copy(),
                        current[:3, :3] @ joint["axis"],
                    )
                current = current @ _transform(
                    rotation=_rotation_class().from_rotvec(
                        joint["axis"] * float(values.get(joint["name"], 0.0))))
            elif joint["type"] == "prismatic":
                current = current @ _transform(
                    joint["axis"] * float(values.get(joint["name"], 0.0)))
        jacobian = np.zeros((6, 7), dtype=float)
        endpoint = current[:3, 3]
        for index, name in enumerate(self.arm_joints[side]):
            point, axis = active[name]
            jacobian[:3, index] = np.cross(axis, endpoint - point)
            jacobian[3:, index] = axis
        return current, jacobian

    def report(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "urdf": str(self.urdf_path),
            "urdf_sha256": hashlib.sha256(self.urdf_path.read_bytes()).hexdigest(),
            "joint_params": str(self.joint_params_path) if self.joint_params_path else None,
            "joint_params_sha256": (
                hashlib.sha256(self.joint_params_path.read_bytes()).hexdigest()
                if self.joint_params_path else None),
            "root_frame": self.root_link,
            "target_links": self.target_links,
            "moving_joint_order": {
                side: list(self.arm_joints[side]) for side in SIDES
            },
            "effective_limits": {
                name: list(self.limits[name])
                for side in SIDES for name in self.arm_joints[side]
            },
        }

    def effective_fixed_joints(
        self, configured: dict[str, float] | None = None
    ) -> dict[str, float]:
        configured = dict(configured or {})
        names = {
            joint["name"]
            for side in SIDES
            for joint in self.chains[side]
            if joint["type"] in ("revolute", "continuous", "prismatic")
            and joint["name"] not in self.arm_joints["left"]
            and joint["name"] not in self.arm_joints["right"]
        }
        return {name: float(configured.get(name, 0.0)) for name in sorted(names)}


class HumanoidUrdfKinematics(DualArmUrdfKinematics):
    """Mobile-humanoid compatibility wrapper retained for existing callers."""

    def __init__(self, urdf: str | Path, joint_params: str | Path | None = None) -> None:
        super().__init__(
            urdf,
            joint_params,
            robot_name="wheeled mobile-base humanoid",
            arm_joints=HUMANOID_ARM_JOINTS,
            target_links=HUMANOID_TARGET_LINKS,
        )


class TianjiUrdfKinematics(DualArmUrdfKinematics):
    """Fixed-base Tianji dual-arm model used by the MuJoCo replay."""

    def __init__(self, urdf: str | Path, joint_params: str | Path | None = None) -> None:
        super().__init__(
            urdf,
            joint_params,
            robot_name="Tianji fixed-base dual arm",
            arm_joints=TIANJI_ARM_JOINTS,
            target_links=TIANJI_TARGET_LINKS,
        )


def _pose_error(current: np.ndarray, target: np.ndarray) -> tuple[float, float, np.ndarray]:
    position = float(np.linalg.norm(current[:3, 3] - target[:3, 3]))
    rotation_vector = _rotation_from_matrix(
        target[:3, :3].T @ current[:3, :3]).as_rotvec()
    return position, float(np.linalg.norm(rotation_vector)), rotation_vector


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _log_derivative(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    cross = _skew(rotation_vector)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * cross + cross @ cross / 12.0
    coefficient = 1.0 / angle**2 - (1.0 + math.cos(angle)) / (
        2.0 * angle * math.sin(angle))
    return np.eye(3) - 0.5 * cross + coefficient * cross @ cross


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    reachable: bool
    seed_name: str
    evaluations: int


class ArmQualityEvaluator:
    def __init__(
        self,
        model: DualArmUrdfKinematics,
        thresholds: Thresholds,
        fixed_joints: dict[str, float] | None = None,
    ) -> None:
        thresholds.validate()
        self.model = model
        self.thresholds = thresholds
        self.fixed_joints = dict(fixed_joints or {})

    def _solve(
        self, side: str, target: np.ndarray, seed: np.ndarray, seed_name: str,
        regularization: float,
    ) -> IkResult:
        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise ReplayError("缺少 scipy.optimize；请先运行 ./setup.sh。") from exc
        lower, upper = self.model.bounds(side)
        scale = upper - lower
        seed = np.clip(np.asarray(seed, dtype=float), lower, upper)

        def residual(q: np.ndarray) -> np.ndarray:
            current, _ = self.model.forward_jacobian(side, q, self.fixed_joints)
            _, _, rotation = _pose_error(current, target)
            return np.concatenate((
                current[:3, 3] - target[:3, 3],
                self.thresholds.orientation_weight_m_per_rad * rotation,
                regularization * (q - seed) / scale,
            ))

        def jacobian(q: np.ndarray) -> np.ndarray:
            current, geometric = self.model.forward_jacobian(
                side, q, self.fixed_joints)
            _, _, rotation = _pose_error(current, target)
            angular_target = target[:3, :3].T @ geometric[3:, :]
            orientation = _log_derivative(rotation) @ angular_target
            return np.vstack((
                geometric[:3, :],
                self.thresholds.orientation_weight_m_per_rad * orientation,
                np.diag(np.full(7, regularization) / scale),
            ))

        result = least_squares(
            residual,
            seed,
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=self.thresholds.ik_max_nfev,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        current, _ = self.model.forward_jacobian(side, result.x, self.fixed_joints)
        position, orientation, _ = _pose_error(current, target)
        reachable = bool(
            position <= self.thresholds.position_tolerance_m
            and orientation <= self.thresholds.orientation_tolerance_rad
            and np.all(result.x >= lower - 1e-9)
            and np.all(result.x <= upper + 1e-9)
        )
        return IkResult(
            result.x.copy(), position, orientation, reachable, seed_name, int(result.nfev))

    def solve_independent(self, side: str, target: np.ndarray) -> IkResult:
        """Per-frame theoretical solve: no state inherited from adjacent frames."""
        lower, upper = self.model.bounds(side)
        midpoint = 0.5 * (lower + upper)
        span = upper - lower
        seeds = (
            ("midpoint", midpoint),
            ("midpoint_positive_pattern", np.clip(
                midpoint + 0.18 * span * np.array([1, -1, 1, -1, 1, -1, 0]),
                lower, upper)),
            ("midpoint_negative_pattern", np.clip(
                midpoint - 0.18 * span * np.array([1, -1, 1, -1, 1, -1, 0]),
                lower, upper)),
        )
        best: IkResult | None = None
        for name, seed in seeds:
            candidate = self._solve(side, target, seed, name, regularization=3e-4)
            if (best is None or
                    candidate.position_error_m + 0.30 * candidate.orientation_error_rad
                    < best.position_error_m + 0.30 * best.orientation_error_rad):
                best = candidate
            if candidate.reachable:
                break
        assert best is not None
        return best

    def solve_continuous_midpoint(
        self, side: str, target: np.ndarray, previous_q: np.ndarray
    ) -> IkResult:
        """Use previous-frame continuity seed, then retry from joint midpoint."""
        first = self._solve(
            side, target, previous_q, "continuous", regularization=1e-3)
        if first.reachable:
            return first
        lower, upper = self.model.bounds(side)
        midpoint = 0.5 * (lower + upper)
        second = self._solve(
            side, target, midpoint, "continuous_then_midpoint", regularization=3e-4)
        if second.reachable:
            return second
        first_score = first.position_error_m + 0.30 * first.orientation_error_rad
        second_score = second.position_error_m + 0.30 * second.orientation_error_rad
        return second if second_score < first_score else first

    def metrics(self, side: str, result: IkResult) -> dict[str, Any]:
        lower, upper = self.model.bounds(side)
        span = upper - lower
        margin = float(np.min(np.minimum(
            (result.q - lower) / span, (upper - result.q) / span)))
        values: dict[str, Any] = {
            "reachable": result.reachable,
            "position_error_m": result.position_error_m,
            "orientation_error_deg": math.degrees(result.orientation_error_rad),
            "seed": result.seed_name,
            "function_evaluations": result.evaluations,
            "joint_margin_fraction": margin,
        }
        for index, q in enumerate(result.q, 1):
            values[f"q{index}"] = float(q)
        if not result.reachable:
            for key in (
                "sigma_min", "sigma_max", "isotropy_ratio", "condition_number",
                "weighted_manipulability", "translational_manipulability",
            ):
                values[key] = None
            return values
        _, jacobian = self.model.forward_jacobian(
            side, result.q, self.fixed_joints)
        weighted = jacobian.copy()
        weighted[3:, :] *= self.thresholds.orientation_weight_m_per_rad
        singular = np.linalg.svd(weighted, compute_uv=False)
        translation = np.linalg.svd(jacobian[:3, :], compute_uv=False)
        sigma_max = float(singular[0])
        sigma_min = float(singular[-1])
        values.update({
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "isotropy_ratio": sigma_min / sigma_max if sigma_max > 1e-12 else 0.0,
            "condition_number": sigma_max / sigma_min if sigma_min > 1e-12 else None,
            "weighted_manipulability": float(np.prod(singular)),
            "translational_manipulability": float(np.prod(translation)),
        })
        return values


def _target_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    target = np.eye(4, dtype=float)
    target[:3, 3] = position
    target[:3, :3] = rotation
    return target


def evaluate_dataset(
    dataset: LeRobotV3Dataset,
    episode_ids: Iterable[int],
    scene_config: SceneConfig,
    model: DualArmUrdfKinematics,
    thresholds: Thresholds,
    fixed_joints: dict[str, float] | None = None,
    progress: Any = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    evaluator = ArmQualityEvaluator(model, thresholds, fixed_joints)
    rows = {"independent": [], "continuous_midpoint": []}
    episode_ids = tuple(episode_ids)
    for episode_index in episode_ids:
        source = load_episode(dataset, episode_index)
        positions = scene_config.robot_from_umi_positions(source.positions_world_m)
        rotations = scene_config.robot_from_umi_rotations(
            quaternion_xyzw_to_matrix(source.quaternions_world_xyzw))
        previous = {
            side: 0.5 * sum(model.bounds(side)) for side in SIDES
        }
        for frame in range(source.frame_count):
            targets = {
                side: _target_matrix(positions[frame, side_index],
                                     rotations[frame, side_index])
                for side_index, side in enumerate(SIDES)
            }
            for strategy in rows:
                row: dict[str, Any] = {
                    "strategy": strategy,
                    "episode_index": episode_index,
                    "frame_index": frame,
                    "timestamp_s": float(source.timestamps_s[frame]),
                }
                for side in SIDES:
                    side_position = targets[side][:3, 3]
                    row.update({
                        f"{side}_target_x_m": float(side_position[0]),
                        f"{side}_target_y_m": float(side_position[1]),
                        f"{side}_target_z_m": float(side_position[2]),
                    })
                    if strategy == "independent":
                        result = evaluator.solve_independent(side, targets[side])
                    else:
                        result = evaluator.solve_continuous_midpoint(
                            side, targets[side], previous[side])
                        previous[side] = result.q.copy()
                    metrics = evaluator.metrics(side, result)
                    row.update({f"{side}_{key}": value
                                for key, value in metrics.items()})
                row["both_reachable"] = bool(
                    row["left_reachable"] and row["right_reachable"])
                rows[strategy].append(row)
            if progress:
                progress(episode_index, frame + 1, source.frame_count)
    return build_summary(
        dataset, episode_ids, scene_config, model, thresholds, fixed_joints, rows
    ), rows


def build_summary(
    dataset: LeRobotV3Dataset,
    episode_ids: Iterable[int],
    scene_config: SceneConfig,
    model: DualArmUrdfKinematics,
    thresholds: Thresholds,
    fixed_joints: dict[str, float] | None,
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build one aggregate report from already evaluated per-frame rows."""
    episode_ids = tuple(episode_ids)
    summaries = {
        strategy: summarize(strategy, strategy_rows, thresholds)
        for strategy, strategy_rows in rows.items()
    }
    independent_pass = summaries["independent"]["conclusion"] == "PASS"
    continuous_pass = summaries["continuous_midpoint"]["conclusion"] == "PASS"
    if independent_pass and continuous_pass:
        comparison = "PASS_BOTH"
    elif independent_pass:
        comparison = "THEORETICAL_PASS_CONTINUOUS_FAIL"
    elif continuous_pass:
        comparison = "THEORETICAL_FAIL_CONTINUOUS_PASS_REVIEW_REQUIRED"
    else:
        comparison = "FAIL_BOTH"
    summary = {
        "format_version": 2,
        "comparison_conclusion": comparison,
        "dataset": dataset.report(),
        "episodes": list(episode_ids),
        "robot_model": model.report(),
        "coordinate_registration": scene_config.report(),
        "fixed_joints": model.effective_fixed_joints(fixed_joints),
        "thresholds": thresholds.report(),
        "scope": {
            "robot": model.robot_name,
            "moving_joints": "left/right 7-DoF arms only",
            "all_other_joints": "fixed at configured values; unspecified joints are zero",
            "self_collision_checked": False,
            "environment_collision_checked": False,
            "velocity_limits_checked": False,
            "jacobian": "URDF geometric TCP Jacobian, 6x7 per arm",
            "jacobian_rows": "linear xyz then angular xyz in base_link",
            "unreachable_manipulability": "undefined and stored as null",
        },
        "strategies": summaries,
    }
    summary["strategies"]["independent"]["method"] = (
        "Each frame and arm solved independently from midpoint; two deterministic "
        "supplementary seeds are tried only when midpoint is unreachable."
    )
    summary["strategies"]["continuous_midpoint"]["method"] = (
        "Previous-frame joint solution first; midpoint retry only when continuity "
        "seed does not meet Cartesian tolerances."
    )
    return summary


def _finite(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values
            if value is not None and math.isfinite(float(value))]


def summarize(strategy: str, rows: list[dict[str, Any]], thresholds: Thresholds) -> dict[str, Any]:
    if not rows:
        raise ReplayError(f"{strategy} 没有评估帧。")
    count = len(rows)
    arms: dict[str, Any] = {}
    for side in SIDES:
        reachable = [row for row in rows if row[f"{side}_reachable"]]
        ratios = _finite(row[f"{side}_isotropy_ratio"] for row in reachable)
        low = sum(value < thresholds.isotropy_ratio_low for value in ratios)
        arms[side] = {
            "frames": count,
            "reachable_frames": len(reachable),
            "reachable_rate": len(reachable) / count,
            "max_position_error_m": max(float(row[f"{side}_position_error_m"])
                                          for row in rows),
            "max_orientation_error_deg": max(float(row[f"{side}_orientation_error_deg"])
                                               for row in rows),
            "min_joint_margin_fraction_reachable": min(
                (float(row[f"{side}_joint_margin_fraction"])
                 for row in reachable), default=None),
            "min_isotropy_ratio": min(ratios, default=None),
            "p05_isotropy_ratio": float(np.percentile(ratios, 5)) if ratios else None,
            "p95_condition_number": (
                float(np.percentile(_finite(
                    row[f"{side}_condition_number"] for row in reachable), 95))
                if reachable else None),
            "low_manipulability_frames": low,
            "low_manipulability_rate_among_reachable": (
                low / len(reachable) if reachable else None),
        }
    both_rate = sum(bool(row["both_reachable"]) for row in rows) / count
    reachability_pass = both_rate >= thresholds.minimum_both_reachable_rate
    manipulability_pass = all(
        arm["low_manipulability_rate_among_reachable"] is not None
        and arm["low_manipulability_rate_among_reachable"]
        <= thresholds.maximum_low_manipulability_rate
        for arm in arms.values()
    )
    return {
        "strategy": strategy,
        "conclusion": "PASS" if reachability_pass and manipulability_pass else "FAIL",
        "reachability_conclusion": "PASS" if reachability_pass else "FAIL",
        "manipulability_conclusion": "PASS" if manipulability_pass else "FAIL",
        "frames": count,
        "both_reachable_frames": sum(bool(row["both_reachable"]) for row in rows),
        "both_reachable_rate": both_rate,
        "arms": arms,
    }


def write_outputs(
    output: str | Path,
    summary: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    from .quality_visualization import write_quality_dashboard

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for strategy, strategy_rows in rows.items():
        with (destination / f"{strategy}_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(strategy_rows[0]))
            writer.writeheader()
            writer.writerows(strategy_rows)
    robot_name = str(summary["robot_model"]["robot_name"])
    moving_joints = summary["robot_model"]["moving_joint_order"]
    target_links = summary["robot_model"]["target_links"]
    fixed_joints = summary["fixed_joints"]
    lines = [
        "# 双臂轨迹可操纵性评估报告",
        "",
        f"机器人：`{robot_name}`",
        "",
        f"并列结论：**{summary['comparison_conclusion']}**",
        "",
    ]
    labels = {
        "independent": "方案一：纯理论独立单帧求解",
        "continuous_midpoint": "方案二：连续性种子 + 中点复求",
    }
    for strategy, label in labels.items():
        result = summary["strategies"][strategy]
        lines.extend([
            f"## {label}", "",
            f"- 结论：**{result['conclusion']}**",
            f"- 双臂同时可达率：{result['both_reachable_rate']:.3%}",
        ])
        for side, side_label in (("left", "左臂"), ("right", "右臂")):
            arm = result["arms"][side]
            ratio = arm["min_isotropy_ratio"]
            lines.append(
                f"- {side_label}：可达率 {arm['reachable_rate']:.3%}，"
                f"最小 isotropy {('N/A' if ratio is None else f'{ratio:.6f}')}"
            )
        lines.append("")
    lines.extend([
        "## 评估口径", "",
        f"左臂运动关节：`{', '.join(moving_joints['left'])}`。",
        f"右臂运动关节：`{', '.join(moving_joints['right'])}`。",
        (
            "上游非手臂关节固定为："
            + ", ".join(f"`{name}={value:.6f}`" for name, value in fixed_joints.items())
            + "。"
            if fixed_joints else
            "TCP 的上游链中没有额外非手臂活动关节。"
        ),
        f"左/右目标 link 为 `{target_links['left']}` / "
        f"`{target_links['right']}`（TCP），不是法兰。",
        "不检查自碰撞、环境碰撞或速度超限；不可达帧的可操作性为 N/A。", "",
    ])
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_quality_dashboard(destination / "quality_dashboard.html", summary, rows)
