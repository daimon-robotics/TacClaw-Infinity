"""Lazy, schema-checked loading of dual TCP poses from LeRobot v3 datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import ReplayError


CANONICAL_LEFT_FIELDS = (
    "left_x", "left_y", "left_z", "left_qx", "left_qy", "left_qz", "left_qw"
)
CANONICAL_RIGHT_FIELDS = (
    "right_x", "right_y", "right_z", "right_qx", "right_qy", "right_qz", "right_qw"
)


@dataclass(frozen=True)
class PoseLayout:
    """How a vector feature stores left/right ``xyz + xyzw`` TCP poses."""

    feature: str
    left_indices: tuple[int, int, int, int, int, int, int]
    right_indices: tuple[int, int, int, int, int, int, int]
    source: str
    left_fields: tuple[str | int, ...]
    right_fields: tuple[str | int, ...]

    def report(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "quaternion_order": "xyzw",
            "left_indices": list(self.left_indices),
            "right_indices": list(self.right_indices),
            "left_fields": list(self.left_fields),
            "right_fields": list(self.right_fields),
            "source": self.source,
        }


@dataclass(frozen=True)
class EpisodeLocator:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    data_path: Path


@dataclass(frozen=True)
class LeRobotV3Dataset:
    root: Path
    info: dict[str, Any]
    info_sha256: str
    fps: float
    layout: PoseLayout
    episodes: dict[int, EpisodeLocator]

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.episodes))

    def report(self) -> dict[str, Any]:
        return {
            "format": str(self.info.get("codebase_version")),
            "dataset": str(self.root),
            "info_sha256": self.info_sha256,
            "fps": self.fps,
            "episode_ids": list(self.episode_ids),
            "pose_layout": self.layout.report(),
        }


@dataclass(frozen=True)
class TcpEpisode:
    """One source episode in the recorded UMI world frame."""

    episode_index: int
    timestamps_s: np.ndarray  # relative, shape (frames,)
    positions_world_m: np.ndarray  # (frames, 2, 3)
    quaternions_world_xyzw: np.ndarray  # (frames, 2, 4)

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_s.size)


def open_dataset(
    dataset_dir: str | Path,
    *,
    pose_feature: str = "observation.state",
    pose_layout_path: str | Path | None = None,
) -> LeRobotV3Dataset:
    """Open LeRobot v3 metadata; episode payloads are read lazily later."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - CLI dependency error
        raise ReplayError("缺少 pyarrow；请先运行 ./setup.sh。") from exc

    root = Path(dataset_dir).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ReplayError(f"不是 LeRobot v3 数据集：缺少 {info_path}。")
    try:
        info_bytes = info_path.read_bytes()
        info = json.loads(info_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"无法读取 LeRobot 元数据 {info_path}: {exc}") from exc
    if not isinstance(info, dict):
        raise ReplayError(f"LeRobot 元数据 {info_path} 根节点必须是 JSON 对象。")
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ReplayError(
            f"仅支持 LeRobot v3 数据集，{info_path} 的 codebase_version 为 {version!r}。"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise ReplayError(f"{info_path} 缺少 features 对象。")
    layout = _resolve_pose_layout(features, pose_feature, pose_layout_path)
    fps = float(info.get("fps", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ReplayError(f"{info_path} 的 fps 必须是正的有限数值。")
    episodes = _load_episode_locators(root, info, pq)
    if not episodes:
        raise ReplayError(f"{root} 没有可用的 LeRobot episode 元数据。")
    return LeRobotV3Dataset(
        root=root,
        info=info,
        info_sha256=hashlib.sha256(info_bytes).hexdigest(),
        fps=fps,
        layout=layout,
        episodes=episodes,
    )


def load_episode(dataset: LeRobotV3Dataset, episode_index: int) -> TcpEpisode:
    """Read exactly one episode's TCP pose vector from its data parquet file."""
    if episode_index not in dataset.episodes:
        preview = ", ".join(str(value) for value in dataset.episode_ids[:20])
        raise ReplayError(
            f"数据集没有 episode {episode_index}；可用 episode 示例：{preview or '无'}。"
        )
    locator = dataset.episodes[episode_index]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ReplayError("缺少 pyarrow；请先运行 ./setup.sh。") from exc
    if not locator.data_path.is_file():
        raise ReplayError(f"episode {episode_index} 的数据文件不存在：{locator.data_path}")
    parquet = pq.ParquetFile(locator.data_path)
    required = (dataset.layout.feature, "timestamp", "episode_index", "index")
    missing = [name for name in required if name not in parquet.schema_arrow.names]
    if missing:
        raise ReplayError(
            f"{locator.data_path} 缺少 LeRobot v3 回放所需列：{', '.join(missing)}。"
        )
    feature_rows: list[Any] = []
    timestamp_rows: list[float] = []
    index_rows: list[int] = []
    for batch in parquet.iter_batches(columns=list(required), batch_size=4096):
        values = batch.to_pydict()
        local_episode_ids = np.asarray(values["episode_index"], dtype=int)
        selected = np.flatnonzero(local_episode_ids == episode_index)
        if selected.size == 0:
            continue
        feature_rows.extend(values[dataset.layout.feature][int(index)] for index in selected)
        timestamp_rows.extend(float(values["timestamp"][int(index)]) for index in selected)
        index_rows.extend(int(values["index"][int(index)]) for index in selected)
    if not feature_rows:
        raise ReplayError(
            f"metadata 将 episode {episode_index} 指向 {locator.data_path}，但文件中找不到该 episode。"
        )
    order = np.argsort(np.asarray(index_rows, dtype=int), kind="stable")
    vectors = np.asarray([feature_rows[int(index)] for index in order], dtype=float)
    timestamps = np.asarray([timestamp_rows[int(index)] for index in order], dtype=float)
    global_indices = np.asarray([index_rows[int(index)] for index in order], dtype=int)
    _validate_episode_arrays(locator, vectors, timestamps, global_indices)
    maximum = max(*dataset.layout.left_indices, *dataset.layout.right_indices)
    if vectors.ndim != 2 or vectors.shape[1] <= maximum:
        raise ReplayError(
            f"episode {episode_index} 的 {dataset.layout.feature} shape 为 {vectors.shape}，"
            f"不足以按布局读取最大索引 {maximum}。"
        )
    left = vectors[:, dataset.layout.left_indices]
    right = vectors[:, dataset.layout.right_indices]
    poses = np.stack((left, right), axis=1)
    if not np.all(np.isfinite(poses)):
        raise ReplayError(f"episode {episode_index} 的 TCP pose 含 NaN/Inf。")
    quaternion_norm = np.linalg.norm(poses[:, :, 3:7], axis=-1)
    if np.any(quaternion_norm < 1e-10):
        raise ReplayError(f"episode {episode_index} 的 TCP pose 含零长度四元数。")
    return TcpEpisode(
        episode_index=episode_index,
        timestamps_s=timestamps - timestamps[0],
        positions_world_m=poses[:, :, 0:3].copy(),
        quaternions_world_xyzw=poses[:, :, 3:7].copy(),
    )


def _resolve_pose_layout(
    features: dict[str, Any], pose_feature: str, pose_layout_path: str | Path | None
) -> PoseLayout:
    if pose_layout_path is not None:
        return _load_pose_layout_file(features, pose_layout_path, pose_feature)
    feature = _resolve_default_feature(features, pose_feature)
    definition = features.get(feature)
    if not isinstance(definition, dict):
        raise ReplayError(f"features.{feature} 不是对象。")
    names = definition.get("names")
    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ReplayError(
            f"LeRobot feature {feature!r} 没有字段 names，无法安全猜测左右 TCP。"
            "请提供 --pose-layout。"
        )
    return _layout_from_items(
        feature,
        names,
        CANONICAL_LEFT_FIELDS,
        CANONICAL_RIGHT_FIELDS,
        source="canonical_leftright_names",
    )


def _resolve_default_feature(features: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        if requested not in features:
            numeric = []
            for name, definition in features.items():
                if isinstance(definition, dict) and isinstance(definition.get("shape"), list):
                    numeric.append(f"{name} shape={definition['shape']}")
            raise ReplayError(
                f"LeRobot 数据集没有 pose feature {requested!r}。候选向量字段："
                + "; ".join(numeric[:16])
            )
        return requested
    candidates: list[str] = []
    for name, definition in features.items():
        if not isinstance(definition, dict) or not isinstance(definition.get("names"), list):
            continue
        names = set(definition["names"])
        if set(CANONICAL_LEFT_FIELDS).issubset(names) and set(CANONICAL_RIGHT_FIELDS).issubset(names):
            candidates.append(name)
    if "observation.state" in candidates:
        return "observation.state"
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ReplayError("自动选择 pose feature 失败：没有含完整左右 TCP 字段名的 feature；请用 --pose-layout。")
    raise ReplayError(
        "自动选择 pose feature 不唯一：" + ", ".join(candidates) + "；请用 --pose-feature。"
    )


def _load_pose_layout_file(
    features: dict[str, Any], path: str | Path, requested_feature: str
) -> PoseLayout:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"无法读取 TCP 布局文件 {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayError("TCP 布局文件根节点必须为 JSON 对象。")
    feature = payload.get("feature")
    if not isinstance(feature, str):
        raise ReplayError("TCP 布局文件缺少字符串 feature。")
    if requested_feature not in ("auto", feature):
        raise ReplayError(
            f"--pose-feature={requested_feature!r} 与布局文件 feature={feature!r} 不一致。"
        )
    if payload.get("quaternion_order") != "xyzw":
        raise ReplayError("TCP 布局文件必须显式声明 quaternion_order 为 'xyzw'。")
    definition = features.get(feature)
    if not isinstance(definition, dict):
        raise ReplayError(f"布局文件指定的 feature {feature!r} 不在数据集中。")
    names = definition.get("names")
    if names is not None and (not isinstance(names, list) or not all(isinstance(item, str) for item in names)):
        raise ReplayError(f"features.{feature}.names 格式无效。")
    left = payload.get("left")
    right = payload.get("right")
    if not isinstance(left, list) or not isinstance(right, list):
        raise ReplayError("TCP 布局文件必须包含 left/right 两个长度为 7 的数组。")
    return _layout_from_items(
        feature,
        names if isinstance(names, list) else None,
        left,
        right,
        source=f"layout_file:{source}",
    )


def _layout_from_items(
    feature: str,
    names: list[str] | None,
    left_items: Iterable[str | int],
    right_items: Iterable[str | int],
    *,
    source: str,
) -> PoseLayout:
    def resolve(side: str, values: Iterable[str | int]) -> tuple[tuple[int, ...], tuple[str | int, ...]]:
        items = tuple(values)
        if len(items) != 7:
            raise ReplayError(f"{side} TCP 布局必须正好有 xyz+xyzw 共 7 项。")
        indices: list[int] = []
        for item in items:
            if isinstance(item, int) and not isinstance(item, bool):
                if item < 0:
                    raise ReplayError(f"{side} TCP 布局索引不能为负：{item}。")
                indices.append(item)
            elif isinstance(item, str):
                if names is None:
                    raise ReplayError(
                        f"{side} TCP 布局使用字段名 {item!r}，但 feature 没有 names；请改用整数索引。"
                    )
                try:
                    indices.append(names.index(item))
                except ValueError as exc:
                    raise ReplayError(f"{side} TCP 布局字段 {item!r} 不在 feature names 中。") from exc
            else:
                raise ReplayError(f"{side} TCP 布局项必须是字段名或整数索引。")
        if len(set(indices)) != 7:
            raise ReplayError(f"{side} TCP 布局不能重复使用同一个字段。")
        return tuple(indices), items

    left_indices, left_fields = resolve("left", left_items)
    right_indices, right_fields = resolve("right", right_items)
    return PoseLayout(
        feature=feature,
        left_indices=left_indices,  # type: ignore[arg-type]
        right_indices=right_indices,  # type: ignore[arg-type]
        source=source,
        left_fields=left_fields,
        right_fields=right_fields,
    )


def _load_episode_locators(root: Path, info: dict[str, Any], pq: Any) -> dict[int, EpisodeLocator]:
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ReplayError("LeRobot v3 info.json 缺少 data_path 模板。")
    metadata_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not metadata_files:
        raise ReplayError(f"找不到 episode 元数据：{root / 'meta' / 'episodes'}。")
    required = (
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    )
    result: dict[int, EpisodeLocator] = {}
    for metadata_path in metadata_files:
        parquet = pq.ParquetFile(metadata_path)
        missing = [name for name in required if name not in parquet.schema_arrow.names]
        if missing:
            raise ReplayError(f"{metadata_path} 缺少 LeRobot v3 episode 字段：{', '.join(missing)}。")
        values = parquet.read(columns=list(required)).to_pydict()
        for row in zip(*(values[name] for name in required), strict=True):
            episode, length, chunk, file_index, start, end = (int(value) for value in row)
            if episode in result:
                raise ReplayError(f"episode 元数据重复定义 episode {episode}。")
            if length <= 0 or end - start != length:
                raise ReplayError(f"episode {episode} 的 length/global index 范围无效。")
            try:
                relative_path = template.format(chunk_index=chunk, file_index=file_index)
            except (KeyError, ValueError) as exc:
                raise ReplayError(f"LeRobot data_path 模板无法填充 chunk/file index：{template!r}") from exc
            result[episode] = EpisodeLocator(
                episode_index=episode,
                length=length,
                dataset_from_index=start,
                dataset_to_index=end,
                data_path=(root / relative_path).resolve(),
            )
    return result


def _validate_episode_arrays(
    locator: EpisodeLocator, vectors: np.ndarray, timestamps: np.ndarray, indices: np.ndarray
) -> None:
    if vectors.shape[0] != locator.length:
        raise ReplayError(
            f"episode {locator.episode_index} 读取到 {vectors.shape[0]} 帧，"
            f"metadata 声明 {locator.length} 帧。"
        )
    expected = np.arange(locator.dataset_from_index, locator.dataset_to_index, dtype=int)
    if not np.array_equal(indices, expected):
        raise ReplayError(
            f"episode {locator.episode_index} 的 global index 不连续或与 metadata 不一致。"
        )
    if timestamps.ndim != 1 or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ReplayError(f"episode {locator.episode_index} 的 timestamp 必须严格递增且有限。")
