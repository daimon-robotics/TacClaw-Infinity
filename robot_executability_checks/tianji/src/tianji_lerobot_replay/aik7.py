"""Thin Python binding for the standalone TJ_FX Aik7 analytic IK backend.

The C++ core is the analytic solver used by the ROS ``/start_wbc`` path.  It
is intentionally separate from the real-time HQP teleoperation controller.
Only a small C ABI crosses into Python so the replay package does not depend
on ROS at runtime.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from .errors import ReplayError


_SIDE_INDEX = {"left": 0, "right": 1}
_DEFAULT_ELBOW_SIGN = {"left": -1, "right": 1}
_MAX_BRANCHES = 16


def rotation_distance_rad(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Return the geodesic SO(3) distance between two proper rotations."""
    relative = np.asarray(rotation_a, dtype=float).T @ np.asarray(rotation_b, dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _project_transform(value: np.ndarray) -> np.ndarray:
    """Validate a homogeneous transform and remove harmless float round-off."""
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ReplayError("Aik7 目标位姿必须是一个 4×4 的有限齐次变换矩阵。")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ReplayError("Aik7 目标位姿最后一行必须是 [0, 0, 0, 1]。")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise ReplayError("Aik7 目标姿态必须是合法的右手 SO(3) 旋转。")
    # Quaternion and MuJoCo matrices are already orthogonal to numerical
    # precision.  The projection only makes the C ABI's strict SO(3) check
    # deterministic after several matrix multiplications.
    left, _, right_t = np.linalg.svd(rotation)
    projected = left @ right_t
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_t
    result = transform.copy()
    result[:3, :3] = projected
    return np.ascontiguousarray(result, dtype=np.float64)


class Aik7AnalyticSolver:
    """Call the vendored second (analytical WBC) IK implementation.

    Transform convention is ``base_from_ee``: points expressed in ``ee_link``
    map to ``base_link`` as ``p_base = R @ p_ee + t``.  TCP conversion belongs
    in :mod:`kinematics`, because this backend deliberately knows only the
    seven-joint arm and ``ee_link``.
    """

    def __init__(self, library_path: str | Path | None = None) -> None:
        self.library_path = self._resolve_library(library_path)
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as exc:
            raise ReplayError(
                f"无法加载 Aik7 解析 IK 库：{self.library_path}。"
                "请在项目根目录重新运行 ./setup.sh。"
            ) from exc

        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
        self._library.tianji_aik7_api_version.argtypes = []
        self._library.tianji_aik7_api_version.restype = ctypes.c_int
        self._library.tianji_aik7_last_error.argtypes = []
        self._library.tianji_aik7_last_error.restype = ctypes.c_char_p
        self._library.tianji_aik7_forward.argtypes = [ctypes.c_int, vector, vector]
        self._library.tianji_aik7_forward.restype = ctypes.c_int
        self._library.tianji_aik7_solve.argtypes = [
            ctypes.c_int,
            vector,
            ctypes.c_double,
            ctypes.c_int,
            vector,
            ctypes.c_int,
        ]
        self._library.tianji_aik7_solve.restype = ctypes.c_int
        if self._library.tianji_aik7_api_version() != 1:
            raise ReplayError("Aik7 解析 IK 库 API 版本不兼容；请重新运行 ./setup.sh。")

    @staticmethod
    def _resolve_library(library_path: str | Path | None) -> Path:
        if library_path is not None:
            candidate = Path(library_path).expanduser().resolve()
            if candidate.is_file():
                return candidate
            raise ReplayError(f"指定的 Aik7 解析 IK 库不存在：{candidate}")
        configured = os.environ.get("TIANJI_AIK7_LIBRARY")
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured).expanduser())
        project_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            project_root / "native" / "build" / filename
            for filename in ("libtianji_aik7.so", "libtianji_aik7.dylib", "tianji_aik7.dll")
        )
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
        raise ReplayError(
            "找不到 Aik7 解析 IK 库。请在项目根目录运行 ./setup.sh；"
            "也可用环境变量 TIANJI_AIK7_LIBRARY 指向已编译的库。"
        )

    def _last_error(self) -> str:
        value = self._library.tianji_aik7_last_error()
        return value.decode("utf-8", errors="replace") if value else "未知 C++ 后端错误"

    @staticmethod
    def _side_index(side: str) -> int:
        try:
            return _SIDE_INDEX[side]
        except KeyError as exc:
            raise ReplayError("Aik7 side 必须是 left 或 right。") from exc

    def forward_base_ee(self, side: str, q: np.ndarray) -> np.ndarray:
        """Evaluate the analytic model's ``base_link ← ee_link`` FK."""
        joint_position = np.ascontiguousarray(np.asarray(q, dtype=np.float64).reshape(-1))
        if joint_position.shape != (7,) or not np.all(np.isfinite(joint_position)):
            raise ReplayError("Aik7 正解需要七个有限关节角（rad）。")
        output = np.empty(16, dtype=np.float64)
        status = self._library.tianji_aik7_forward(
            self._side_index(side), joint_position, output
        )
        if status != 0:
            raise ReplayError(f"Aik7 正解失败：{self._last_error()}")
        return output.reshape(4, 4).copy()

    def solve_base_ee(
        self,
        side: str,
        base_from_ee: np.ndarray,
        q7_rad: float,
        *,
        elbow_sign: int | None = None,
    ) -> np.ndarray:
        """Return all valid analytical branches, shaped ``(branches, 7)``."""
        if not np.isfinite(q7_rad):
            raise ReplayError("Aik7 的第 7 轴冗余角必须是有限数值。")
        side_index = self._side_index(side)
        target = _project_transform(base_from_ee).reshape(-1)
        output = np.empty(_MAX_BRANCHES * 7, dtype=np.float64)
        selected_elbow = _DEFAULT_ELBOW_SIGN[side] if elbow_sign is None else int(elbow_sign)
        status = self._library.tianji_aik7_solve(
            side_index,
            target,
            float(q7_rad),
            selected_elbow,
            output,
            _MAX_BRANCHES,
        )
        if status < 0:
            raise ReplayError(f"Aik7 解析逆解调用失败：{self._last_error()}")
        return output[: status * 7].reshape(status, 7).copy()
