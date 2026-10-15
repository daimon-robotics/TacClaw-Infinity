"""Derive and audit the current MuJoCo scene's virtual board-to-base transform.

There is no camera observation in a purely MuJoCo scene.  This module therefore
does **not** claim to estimate a physical hand-eye calibration.  It generates
non-coplanar virtual board fiducials from the configured scene, fits their
known positions in ``base_link``, and writes the resulting reproducible scene
registration.  It is useful for removing hidden, duplicate transforms from a
simulation replay pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ReplayError
from .frames import SceneConfig, matrix_to_quaternion_xyzw


@dataclass(frozen=True)
class VirtualSceneRegistration:
    """``base_link <- UMI world`` obtained from current simulation geometry."""

    rotation_base_from_umi: np.ndarray
    translation_base_from_umi_m: np.ndarray
    fiducial_names: tuple[str, ...]
    fiducials_umi_m: np.ndarray
    fiducials_base_m: np.ndarray
    fit_rms_m: float
    fit_max_m: float
    closure_translation_error_m: float
    closure_rotation_error_deg: float

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_base_from_umi, dtype=float)
        translation = np.asarray(self.translation_base_from_umi_m, dtype=float)
        source = np.asarray(self.fiducials_umi_m, dtype=float)
        target = np.asarray(self.fiducials_base_m, dtype=float)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ReplayError("虚拟场景配准旋转不是合法 SO(3)。")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ReplayError("虚拟场景配准旋转的 determinant 不是 +1。")
        if translation.shape != (3,) or source.ndim != 2 or source.shape[1:] != (3,):
            raise ReplayError("虚拟场景配准数组维度无效。")
        if target.shape != source.shape or len(self.fiducial_names) != source.shape[0]:
            raise ReplayError("虚拟场景配准 fiducial 数量不一致。")
        if not all(
            math.isfinite(float(value))
            for value in (self.fit_rms_m, self.fit_max_m, self.closure_translation_error_m, self.closure_rotation_error_deg)
        ):
            raise ReplayError("虚拟场景配准含有非有限误差。")
        object.__setattr__(self, "rotation_base_from_umi", rotation.copy())
        object.__setattr__(self, "translation_base_from_umi_m", translation.copy())
        object.__setattr__(self, "fiducials_umi_m", source.copy())
        object.__setattr__(self, "fiducials_base_m", target.copy())

    def apply_positions(self, positions_umi_m: np.ndarray) -> np.ndarray:
        values = np.asarray(positions_umi_m, dtype=float)
        if values.shape[-1:] != (3,) or not np.all(np.isfinite(values)):
            raise ReplayError("虚拟场景配准输入位置必须是最后一维为 3 的有限数组。")
        return (
            np.einsum("ij,...j->...i", self.rotation_base_from_umi, values)
            + self.translation_base_from_umi_m
        )

    def apply_rotations(self, rotations_umi: np.ndarray) -> np.ndarray:
        values = np.asarray(rotations_umi, dtype=float)
        if values.shape[-2:] != (3, 3) or not np.all(np.isfinite(values)):
            raise ReplayError("虚拟场景配准输入姿态必须是最后两维为 3×3 的有限数组。")
        return np.einsum("ij,...jk->...ik", self.rotation_base_from_umi, values)

    def report(self) -> dict[str, Any]:
        return {
            "kind": "virtual_scene_registration",
            "physical_handeye": False,
            "source": "scene_config_model_derived_virtual_fiducials",
            "parent_frame": "base_link",
            "child_frame": "umi_world",
            "transform_convention": "p_base = R_base_from_umi @ p_umi + t_base_from_umi",
            "rotation_base_from_umi": self.rotation_base_from_umi.tolist(),
            "translation_base_from_umi_m": self.translation_base_from_umi_m.tolist(),
            "rotation_quaternion_xyzw": matrix_to_quaternion_xyzw(
                self.rotation_base_from_umi
            ).tolist(),
            "virtual_fiducials": [
                {
                    "name": name,
                    "position_umi_m": source.tolist(),
                    "position_base_m": target.tolist(),
                }
                for name, source, target in zip(
                    self.fiducial_names,
                    self.fiducials_umi_m,
                    self.fiducials_base_m,
                    strict=True,
                )
            ],
            "fit_rms_m": self.fit_rms_m,
            "fit_max_m": self.fit_max_m,
            "closure_translation_error_m": self.closure_translation_error_m,
            "closure_rotation_error_deg": self.closure_rotation_error_deg,
            "limitations": (
                "This is an exact registration of the configured MuJoCo scene, not a "
                "physical camera/robot hand-eye estimate. It cannot validate or correct "
                "a real robot/table placement by itself."
            ),
        }


def calibrate_current_simulation(config: SceneConfig) -> VirtualSceneRegistration:
    """Fit current virtual board fiducials and return ``base_link <- UMI``.

    The source points contain the UMI origin, board directions and an inward
    normal reference, so the fitted transform is full 6-DoF rather than a
    coplanar-only board fit.
    """
    board = config.board
    names = (
        "umi_world_origin",
        "board_right_edge_center",
        "board_top_edge_center",
        "board_bottom_edge_center",
        "board_inward_normal_reference",
    )
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [board.panel_width_m / 2.0, 0.0, 0.0],
            board.top_edge_center_in_umi_m,
            board.bottom_edge_center_in_umi_m,
            [0.0, 0.0, max(0.02, board.panel_thickness_m * 4.0)],
        ],
        dtype=float,
    )
    target = config.robot_from_umi_positions(source)
    rotation, translation = _fit_rigid_transform(source, target)
    predicted = np.einsum("ij,nj->ni", rotation, source) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    expected_rotation = config.rotation_robot_from_umi
    expected_translation = config.translation_robot_from_umi_m
    if np.allclose(rotation, expected_rotation, atol=1e-12):
        closure_rotation_error_deg = 0.0
    else:
        rotation_delta = rotation @ expected_rotation.T
        trace = float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        closure_rotation_error_deg = float(math.degrees(math.acos(trace)))
    return VirtualSceneRegistration(
        rotation_base_from_umi=rotation,
        translation_base_from_umi_m=translation,
        fiducial_names=names,
        fiducials_umi_m=source,
        fiducials_base_m=target,
        fit_rms_m=float(np.sqrt(np.mean(residuals**2))),
        fit_max_m=float(np.max(residuals)),
        closure_translation_error_m=float(np.linalg.norm(translation - expected_translation)),
        closure_rotation_error_deg=closure_rotation_error_deg,
    )


def write_virtual_scene_registration(
    path: str | Path,
    registration: VirtualSceneRegistration,
    config: SceneConfig,
    *,
    scene_config_path: str | Path | None = None,
) -> Path:
    """Write a standalone, explicitly non-physical virtual calibration JSON."""
    destination = Path(path).expanduser().resolve()
    payload = registration.report()
    payload["scene"] = config.report()
    if scene_config_path is not None:
        source = Path(scene_config_path).expanduser().resolve()
        try:
            payload["scene_config_path"] = str(source)
            payload["scene_config_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError as exc:
            raise ReplayError(f"无法计算场景配置哈希 {source}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def _fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit target = R @ source + t with a proper-rotation Kabsch solution."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ReplayError("虚拟 hand-eye 拟合需要至少三个对应的三维点。")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation
