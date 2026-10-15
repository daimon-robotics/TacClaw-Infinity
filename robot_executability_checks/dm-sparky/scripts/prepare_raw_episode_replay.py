#!/usr/bin/env python3
"""Build a LeRobot-v3-compatible replay bundle from one raw UMI episode.

The raw gripper VINS poses, Ego head pose, and Ego stereo video are sampled on
one hardware-timestamp time axis.  No spatial alignment is estimated here: the
input poses are required to already use the shared calibration-board world
frame used during collection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_raw_ego_stereo import _raw_head_trajectory  # noqa: E402


POSE_NAMES = tuple(
    f"{side}_{field}"
    for side in ("left", "right", "head")
    for field in ("x", "y", "z", "qx", "qy", "qz", "qw")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument(
        "--pose-dir",
        type=Path,
        help="directory containing pose_data_left.csv and pose_data_right.csv",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_start_us(metadata: dict, group: str, name: str) -> int:
    rows = [row for row in metadata[group] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one {name!r} entry in {group}")
    return int(rows[0]["start_offset_us"])


def _metadata_duration_s(metadata: dict, group: str, name: str) -> float:
    rows = [row for row in metadata[group] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"expected one {name!r} entry in {group}")
    return float(rows[0]["duration_s"])


def _read_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[int] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = (
            "Timestamp_us", "X", "Y", "Z",
            "Quat_X", "Quat_Y", "Quat_Z", "Quat_W",
        )
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        for row in reader:
            timestamps.append(int(row["Timestamp_us"]))
            positions.append(tuple(float(row[name]) for name in ("X", "Y", "Z")))
            quaternions.append(
                tuple(float(row[name]) for name in (
                    "Quat_X", "Quat_Y", "Quat_Z", "Quat_W"
                ))
            )
    times = np.asarray(timestamps, dtype=np.float64) * 1e-6
    position = np.asarray(positions, dtype=float)
    quaternion = np.asarray(quaternions, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} must contain strictly increasing pose samples")
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return times, position, quaternion


def _interpolate_pose(
    times: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.column_stack(
        [np.interp(query, times, positions[:, axis]) for axis in range(3)]
    )
    quaternion = Slerp(times, Rotation.from_quat(quaternions))(query).as_quat()
    return position, quaternion


def _write_dataset(
    root: Path,
    fps: float,
    timestamps_s: np.ndarray,
    states: np.ndarray,
    camera_from_imu: list[list[float]],
    source_episode: Path,
) -> None:
    data_dir = root / "data" / "chunk-000"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    count = int(len(timestamps_s))
    state_type = pa.list_(pa.float32(), len(POSE_NAMES))
    table = pa.table(
        {
            "episode_index": pa.array(np.zeros(count, dtype=np.int64)),
            "frame_index": pa.array(np.arange(count, dtype=np.int64)),
            "timestamp": pa.array(timestamps_s, type=pa.float64()),
            "index": pa.array(np.arange(count, dtype=np.int64)),
            "observation.state": pa.array(states.astype(np.float32).tolist(), type=state_type),
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")
    episode_table = pa.table(
        {
            "episode_index": pa.array([0], type=pa.int64()),
            "length": pa.array([count], type=pa.int64()),
            "data/chunk_index": pa.array([0], type=pa.int64()),
            "data/file_index": pa.array([0], type=pa.int64()),
            "dataset_from_index": pa.array([0], type=pa.int64()),
            "dataset_to_index": pa.array([count], type=pa.int64()),
        }
    )
    pq.write_table(episode_table, episodes_dir / "file-000.parquet")
    info = {
        "codebase_version": "v3.0",
        "robot_type": "dual_ugripper_v2_5_raw_adapter",
        "fps": float(fps),
        "total_episodes": 1,
        "total_frames": count,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(POSE_NAMES)],
                "names": list(POSE_NAMES),
            },
            "observation.images.ego_left": {
                "dtype": "video",
                "shape": [3, 1748, 2328],
                "imu_to_rgb_camera": camera_from_imu,
            },
        },
        "raw_source_episode": str(source_episode),
        "pose_frame": "shared_calibration_board_world",
        "quaternion_order": "xyzw",
    }
    info_dir = root / "meta"
    info_dir.mkdir(parents=True, exist_ok=True)
    (info_dir / "info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    raw = args.raw_episode.expanduser().resolve()
    pose_dir = (
        args.pose_dir.expanduser().resolve()
        if args.pose_dir is not None else raw / "pose_data"
    )
    output = args.output_dir.expanduser().resolve()
    dataset = output / "dataset"
    stereo = output / "stereo_input"
    stereo.mkdir(parents=True, exist_ok=True)

    episode_metadata = json.loads((raw / "metadata.json").read_text(encoding="utf-8"))
    ego = raw / "ego"
    ego_metadata = json.loads((ego / "metadata.json").read_text(encoding="utf-8"))
    ego_calibration = json.loads((ego / "calibration.json").read_text(encoding="utf-8"))
    left_t, left_p, left_q = _read_pose_csv(pose_dir / "pose_data_left.csv")
    right_t, right_p, right_q = _read_pose_csv(pose_dir / "pose_data_right.csv")
    head_rel_t, head_p, head_q = _raw_head_trajectory(ego / "sensor.mcap")

    head_start_s = _metadata_start_us(
        ego_metadata, "head_pose_details", "head_pose"
    ) * 1e-6
    head_t = head_start_s + head_rel_t
    rgb_start_s = _metadata_start_us(ego_metadata, "video_details", "rgb.mp4") * 1e-6
    rgb_end_s = rgb_start_s + _metadata_duration_s(
        ego_metadata, "video_details", "rgb.mp4"
    )
    task_start_s = float(episode_metadata["task_start_s"])
    task_stop_s = float(episode_metadata["task_stop_s"])
    common_start = max(
        task_start_s, left_t[0], right_t[0], head_t[0], rgb_start_s
    )
    common_stop = min(
        task_stop_s, left_t[-1], right_t[-1], head_t[-1], rgb_end_s
    )
    if common_stop <= common_start:
        raise ValueError("raw pose/video streams do not overlap the task window")
    frame_count = int(np.floor((common_stop - common_start) * args.fps))
    if frame_count < 2:
        raise ValueError("common task window contains fewer than two output frames")
    absolute_times = common_start + np.arange(frame_count, dtype=float) / args.fps
    relative_times = absolute_times - common_start

    left_position, left_quaternion = _interpolate_pose(
        left_t, left_p, left_q, absolute_times
    )
    right_position, right_quaternion = _interpolate_pose(
        right_t, right_p, right_q, absolute_times
    )
    head_position, head_quaternion = _interpolate_pose(
        head_t, head_p, head_q, absolute_times
    )
    states = np.concatenate(
        (
            left_position, left_quaternion,
            right_position, right_quaternion,
            head_position, head_quaternion,
        ),
        axis=1,
    )
    camera_from_imu = ego_calibration["observation"]["images"]["rgb"][
        "extrinsics"
    ]["T_ic_imu0_cam0"]
    _write_dataset(
        dataset, args.fps, relative_times, states, camera_from_imu, raw
    )

    world_from_head = np.repeat(np.eye(4)[None, :, :], frame_count, axis=0)
    world_from_head[:, :3, :3] = Rotation.from_quat(head_quaternion).as_matrix()
    world_from_head[:, :3, 3] = head_position
    np.savez_compressed(
        stereo / "world_from_head.npz",
        frame_index=np.arange(frame_count, dtype=np.int32),
        timestamp_s=relative_times,
        world_from_head=world_from_head,
    )
    shutil.copy2(ego / "calibration.json", stereo / "calibration.json")
    stereo_video = stereo / "rgb.mp4"
    if args.overwrite or not stereo_video.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{common_start - rgb_start_s:.9f}",
                "-i", str(ego / "rgb.mp4"),
                "-vf", f"fps={args.fps:g}:round=near",
                "-frames:v", str(frame_count), "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "10",
                "-pix_fmt", "yuv420p", str(stereo_video),
            ],
            check=True,
        )
    manifest = {
        "source_kind": "same_raw_umi_episode",
        "raw_episode": str(raw),
        "pose_dir": str(pose_dir),
        "pose_frame_assumption": "shared_calibration_board_world",
        "task_window_s": [task_start_s, task_stop_s],
        "common_window_s": [common_start, common_stop],
        "rgb_seek_s": common_start - rgb_start_s,
        "fps": float(args.fps),
        "frame_count": frame_count,
        "dataset": str(dataset),
        "stereo_input": str(stereo),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "RAW_EPISODE_REPLAY_WRITTEN "
        f"frames={frame_count} duration_s={relative_times[-1]:.3f} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
