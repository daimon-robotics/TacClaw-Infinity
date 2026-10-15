#!/usr/bin/env python3
"""Align a raw SXR Ego stereo recording to one LeRobot episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--search-step", type=float, default=0.01)
    parser.add_argument(
        "--max-visual-alignment-mse",
        type=float,
        default=0.20,
        help=(
            "maximum normalized multi-frame RGB matching error; larger values "
            "usually mean the raw recording and LeRobot episode are not the same capture"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _dataset_head_trajectory(
    dataset: Path, episode: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    position_indices = [names.index(f"head_{axis}") for axis in "xyz"]
    quaternion_indices = [names.index(f"head_q{axis}") for axis in "xyzw"]
    rows: list[dict[str, object]] = []
    for path in sorted((dataset / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "timestamp", "observation.state"],
        )
        rows.extend(
            row for row in table.to_pylist() if int(row["episode_index"]) == episode
        )
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError(f"dataset has no frames for episode {episode}: {dataset}")
    expected = list(range(len(rows)))
    actual = [int(row["frame_index"]) for row in rows]
    if actual != expected:
        raise ValueError("episode frame_index must be contiguous and start at zero")
    times = np.asarray([float(row["timestamp"]) for row in rows], dtype=float)
    times -= times[0]
    positions = np.asarray(
        [
            [float(row["observation.state"][index]) for index in position_indices]
            for row in rows
        ],
        dtype=float,
    )
    quaternions = np.asarray(
        [
            [float(row["observation.state"][index]) for index in quaternion_indices]
            for row in rows
        ],
        dtype=float,
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return times, positions, quaternions, len(rows)


def _episode_video_path(dataset: Path, episode: int, side: str = "left") -> Path:
    key = f"videos/observation.images.ego_{side}"
    for path in sorted((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                f"{key}/chunk_index",
                f"{key}/file_index",
            ],
        )
        for row in table.to_pylist():
            if int(row["episode_index"]) != episode:
                continue
            chunk = int(row[f"{key}/chunk_index"])
            index = int(row[f"{key}/file_index"])
            matches = sorted(
                (dataset / key / f"chunk-{chunk:03d}").glob(f"file-{index:03d}.*")
            )
            if len(matches) != 1:
                raise FileNotFoundError(f"cannot resolve {key} video for episode {episode}")
            return matches[0]
    raise ValueError(f"episode {episode} was not found in dataset metadata")


def _image_feature(frame: np.ndarray, *, split_left: bool) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment guidance
        raise RuntimeError("visual Ego alignment requires opencv-python-headless") from exc
    if split_left:
        frame = frame[:, : frame.shape[1] // 2]
    image = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image = cv2.GaussianBlur(image, (5, 5), 0.0)
    return (image - image.mean()) / (image.std() + 1e-6)


def estimate_visual_rgb_offset(
    raw_video: Path,
    reference_video: Path,
    frame_count: int,
    raw_fps: float,
    reference_fps: float,
) -> dict[str, object]:
    """Match the LeRobot Ego sequence directly against the raw left camera."""
    import cv2

    reference_indices = np.unique(
        np.linspace(0, frame_count - 1, min(13, frame_count), dtype=int)
    )
    reference_features: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(reference_video))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in reference_indices:
            reference_features[index] = _image_feature(frame, split_left=False)
        index += 1
    capture.release()
    if len(reference_features) != len(reference_indices):
        raise ValueError(f"could not decode reference frames from {reference_video}")

    raw_features: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(raw_video))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        raw_features.append(_image_feature(frame, split_left=True))
    capture.release()
    if not raw_features:
        raise ValueError(f"could not decode raw Ego frames from {raw_video}")
    raw_array = np.asarray(raw_features, dtype=np.float32)
    first_error = np.mean(
        np.square(raw_array - reference_features[0][None, :, :]), axis=(1, 2)
    )
    candidates = np.argsort(first_error)[: min(80, len(first_error))]
    ratio = raw_fps / reference_fps
    scored: list[tuple[float, int]] = []
    for candidate in candidates:
        errors = []
        for reference_index, feature in reference_features.items():
            raw_index = int(round(int(candidate) + reference_index * ratio))
            if raw_index >= len(raw_array):
                errors = []
                break
            errors.append(float(np.mean(np.square(raw_array[raw_index] - feature))))
        if errors:
            scored.append((float(np.mean(errors)), int(candidate)))
    if not scored:
        raise ValueError("raw Ego video does not contain the complete LeRobot episode")
    sequence_score, raw_frame = min(scored)
    return {
        "method": "direct_left_rgb_sequence_match",
        "raw_frame_index": raw_frame,
        "rgb_offset_s": raw_frame / raw_fps,
        "first_frame_normalized_mse": float(first_error[raw_frame]),
        "sequence_normalized_mse": sequence_score,
        "reference_frame_indices": reference_indices.tolist(),
        "raw_fps": float(raw_fps),
        "reference_fps": float(reference_fps),
        "reference_video": str(reference_video),
    }


def _raw_head_trajectory(
    sensor_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from mcap.reader import NonSeekingReader
    except ImportError as exc:  # pragma: no cover - environment-specific guidance
        raise RuntimeError("raw Ego alignment requires the Python package 'mcap'") from exc
    timestamps: list[int] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    with sensor_path.open("rb") as stream:
        # Some source MCAP files contain unusable chunk time indices. Sequential
        # reading intentionally bypasses those indices.
        for _schema, channel, message in NonSeekingReader(stream).iter_messages(
            log_time_order=False
        ):
            if channel.topic != "head_pose" or len(message.data) != 36:
                continue
            timestamp, x, y, z, qx, qy, qz, qw = struct.unpack(
                "<q7f", message.data
            )
            timestamps.append(timestamp)
            positions.append((x, y, z))
            quaternions.append((qx, qy, qz, qw))
    if len(timestamps) < 2:
        raise ValueError(f"no head_pose samples found in {sensor_path}")
    order = np.argsort(timestamps)
    raw_times = np.asarray(timestamps, dtype=np.float64)[order]
    raw_times = (raw_times - raw_times[0]) * 1e-9
    raw_positions = np.asarray(positions, dtype=float)[order]
    raw_quaternions = np.asarray(quaternions, dtype=float)[order]
    raw_quaternions /= np.linalg.norm(raw_quaternions, axis=1, keepdims=True)
    unique = np.concatenate(([True], np.diff(raw_times) > 0.0))
    return raw_times[unique], raw_positions[unique], raw_quaternions[unique]


def _interpolate_positions(
    raw_times: np.ndarray, raw_positions: np.ndarray, query_times: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_times, raw_times, raw_positions[:, axis]) for axis in range(3)]
    )


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _singular, vt = np.linalg.svd(np.asarray(rotations).sum(axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _pose_alignment_error(
    source_positions: np.ndarray,
    source_rotations: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    # If both streams describe the same physical IMU frame, every orientation
    # pair yields the same dataset-world-from-raw-world rotation.
    relative_rotations = target_rotations @ np.transpose(
        source_rotations, (0, 2, 1)
    )
    rotation = _mean_rotation(relative_rotations)
    translation = target_positions.mean(axis=0) - (
        source_positions @ rotation.T
    ).mean(axis=0)
    position_distances = np.linalg.norm(
        source_positions @ rotation.T + translation - target_positions, axis=1
    )
    rotation_residual = Rotation.from_matrix(
        relative_rotations @ rotation.T
    ).magnitude()
    return (
        float(np.sqrt(np.mean(np.square(position_distances)))),
        float(np.median(position_distances)),
        float(np.sqrt(np.mean(np.square(rotation_residual)))),
        float(np.median(rotation_residual)),
        rotation,
        translation,
    )


def estimate_time_offset(
    raw_times: np.ndarray,
    raw_positions: np.ndarray,
    raw_quaternions: np.ndarray,
    episode_times: np.ndarray,
    episode_positions: np.ndarray,
    episode_quaternions: np.ndarray,
    search_step: float = 0.01,
) -> dict[str, object]:
    """Estimate time and a single raw-tracking-world to dataset-world transform."""
    if search_step <= 0.0:
        raise ValueError("search_step must be positive")
    maximum = float(raw_times[-1] - episode_times[-1])
    if maximum < 0.0:
        raise ValueError("raw head trajectory is shorter than the LeRobot episode")

    raw_slerp = Slerp(raw_times, Rotation.from_quat(raw_quaternions))
    target_rotations = Rotation.from_quat(episode_quaternions).as_matrix()

    def evaluate(offset: float) -> tuple[float, ...]:
        query_times = episode_times + float(offset)
        source_positions = _interpolate_positions(
            raw_times, raw_positions, episode_times + float(offset)
        )
        source_rotations = raw_slerp(query_times).as_matrix()
        return _pose_alignment_error(
            source_positions,
            source_rotations,
            episode_positions,
            target_rotations,
        )

    def score(result: tuple[float, ...]) -> float:
        # One radian of orientation inconsistency receives the same cost as
        # 0.10 m of positional inconsistency. This uses both measured pose
        # channels without allowing either one to dominate the time search.
        return float(result[0] + 0.10 * result[2])

    coarse = np.arange(0.0, maximum + search_step * 0.5, search_step)
    errors = np.asarray([score(evaluate(offset)) for offset in coarse])
    best_coarse = float(coarse[int(np.argmin(errors))])
    fine_step = search_step / 10.0
    fine = np.arange(
        max(0.0, best_coarse - search_step),
        min(maximum, best_coarse + search_step) + fine_step * 0.5,
        fine_step,
    )
    fine_results = [evaluate(float(offset)) for offset in fine]
    best_index = int(np.argmin([score(result) for result in fine_results]))
    (
        rms,
        median,
        rotation_rms,
        rotation_median,
        rotation,
        translation,
    ) = fine_results[best_index]
    return {
        "raw_head_offset_s": float(fine[best_index]),
        "position_rms_m": rms,
        "position_median_m": median,
        "orientation_rms_deg": float(np.rad2deg(rotation_rms)),
        "orientation_median_deg": float(np.rad2deg(rotation_median)),
        "raw_head_to_dataset_rotation": rotation.tolist(),
        "raw_head_to_dataset_translation_m": translation.tolist(),
        "coarse_search_step_s": float(search_step),
        "fine_search_step_s": float(fine_step),
        "search_interval_s": [0.0, maximum],
    }


def _metadata_start_us(metadata: dict[str, object], group: str, name: str) -> int:
    values = metadata[group]
    matches = [row for row in values if row["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} entry in {group}")
    return int(matches[0]["start_offset_us"])


def _pose_alignment_at_offset(
    raw_times: np.ndarray,
    raw_positions: np.ndarray,
    raw_quaternions: np.ndarray,
    episode_times: np.ndarray,
    episode_positions: np.ndarray,
    episode_quaternions: np.ndarray,
    offset: float,
) -> dict[str, object]:
    query_times = episode_times + float(offset)
    source_positions = _interpolate_positions(raw_times, raw_positions, query_times)
    source_rotations = Slerp(
        raw_times, Rotation.from_quat(raw_quaternions)
    )(query_times).as_matrix()
    target_rotations = Rotation.from_quat(episode_quaternions).as_matrix()
    (
        rms,
        median,
        rotation_rms,
        rotation_median,
        rotation,
        translation,
    ) = _pose_alignment_error(
        source_positions, source_rotations, episode_positions, target_rotations
    )
    return {
        "raw_head_offset_s": float(offset),
        "position_rms_m": rms,
        "position_median_m": median,
        "orientation_rms_deg": float(np.rad2deg(rotation_rms)),
        "orientation_median_deg": float(np.rad2deg(rotation_median)),
        "raw_head_to_dataset_rotation": rotation.tolist(),
        "raw_head_to_dataset_translation_m": translation.tolist(),
    }


def main() -> int:
    args = parse_args()
    raw_episode = args.raw_episode.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_ego = raw_episode / "ego"
    raw_video = raw_ego / "rgb.mp4"
    raw_calibration = raw_ego / "calibration.json"
    raw_sensor = raw_ego / "sensor.mcap"
    for path in (raw_video, raw_calibration, raw_sensor, raw_ego / "metadata.json"):
        if not path.is_file():
            raise FileNotFoundError(path)

    (
        episode_times,
        episode_positions,
        episode_quaternions,
        frame_count,
    ) = _dataset_head_trajectory(dataset, args.episode)
    raw_times, raw_positions, raw_quaternions = _raw_head_trajectory(raw_sensor)
    metadata = json.loads((raw_ego / "metadata.json").read_text(encoding="utf-8"))
    rgb_start_us = _metadata_start_us(metadata, "video_details", "rgb.mp4")
    head_start_us = _metadata_start_us(metadata, "head_pose_details", "head_pose")
    rgb_details = next(
        row for row in metadata["video_details"] if row["name"] == "rgb.mp4"
    )
    dataset_info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    reference_video = _episode_video_path(dataset, args.episode, "left")
    visual_alignment = estimate_visual_rgb_offset(
        raw_video,
        reference_video,
        frame_count,
        float(rgb_details["fps"]),
        float(dataset_info["fps"]),
    )
    if float(visual_alignment["sequence_normalized_mse"]) > float(
        args.max_visual_alignment_mse
    ):
        raise ValueError(
            "raw Ego recording failed the same-capture check: normalized multi-frame "
            f"MSE={visual_alignment['sequence_normalized_mse']:.4f} exceeds "
            f"{args.max_visual_alignment_mse:.4f}. Do not combine this point cloud "
            "with the requested LeRobot trajectory; provide the raw recording that "
            "was used to build that episode."
        )
    rgb_offset_s = float(visual_alignment["rgb_offset_s"])
    if rgb_offset_s < 0.0:
        raise ValueError(f"computed negative RGB offset: {rgb_offset_s}")
    raw_head_offset_s = rgb_offset_s + (rgb_start_us - head_start_us) * 1e-6
    alignment = _pose_alignment_at_offset(
        raw_times,
        raw_positions,
        raw_quaternions,
        episode_times,
        episode_positions,
        episode_quaternions,
        raw_head_offset_s,
    )
    alignment["visual_alignment"] = visual_alignment

    # The visual sequence match makes each extracted RGB frame correspond to a
    # LeRobot frame.  Save the head pose from that same frame so RGB, TCP and
    # camera pose share one timeline; raw MCAP head poses remain a diagnostic.
    pose_times = episode_times.copy()
    world_from_head = np.repeat(np.eye(4)[None, :, :], frame_count, axis=0)
    world_from_head[:, :3, :3] = Rotation.from_quat(
        episode_quaternions
    ).as_matrix()
    world_from_head[:, :3, 3] = episode_positions
    pose_path = output / "world_from_head.npz"
    np.savez_compressed(
        pose_path,
        frame_index=np.arange(frame_count, dtype=np.int32),
        lerobot_time_s=pose_times,
        world_from_head=world_from_head,
    )

    calibration_path = output / "calibration.json"
    shutil.copy2(raw_calibration, calibration_path)
    stereo_path = output / "rgb.mp4"
    if args.overwrite or not stereo_path.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{rgb_offset_s:.6f}", "-i", str(raw_video),
                "-vf", f"fps={args.fps:g}:round=near",
                "-frames:v", str(frame_count), "-an", "-c:v", "libx264",
                "-preset", "medium", "-crf", "10", "-pix_fmt", "yuv420p",
                str(stereo_path),
            ],
            check=True,
        )

    alignment.update(
        {
            "rgb_offset_s": rgb_offset_s,
            "rgb_minus_head_start_s": (rgb_start_us - head_start_us) * 1e-6,
            "pose_source": "synchronized_lerobot_head_pose",
            "dataset_frame_count": frame_count,
            "output_fps": float(args.fps),
        }
    )
    manifest = {
        "source_kind": "raw_sxr_ego",
        "raw_episode": str(raw_episode),
        "raw_video": str(raw_video),
        "raw_calibration": str(raw_calibration),
        "dataset": str(dataset),
        "episode": int(args.episode),
        "stereo_video": str(stereo_path),
        "calibration": str(calibration_path),
        "world_from_head": str(pose_path),
        "alignment": alignment,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "RAW_EGO_STEREO_WRITTEN "
        f"output={output} rgb_offset_s={rgb_offset_s:.3f} "
        f"alignment_rms_mm={float(alignment['position_rms_m']) * 1000.0:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
