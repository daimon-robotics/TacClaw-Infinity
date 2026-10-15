#!/usr/bin/env python3
"""Recover the raw-VINS-to-dataset-world transform for one raw UMI episode.

The raw ``pose_data_{side}.csv`` files are written in each side's own VIO
initialization frame.  The shipped LeRobot dataset is expressed in a different
world frame, and the offline conversion step that produced it is not available
on this machine.  This utility therefore treats the mapping as an unknown rigid
transform and solves for it directly from the two trajectories:

1. Resample the raw IMU-rate trajectory onto the dataset's frame rate.
2. For every dataset episode, slide the episode window along the raw recording
   and score each offset with the speed profile, which no rigid transform can
   change.
3. Refine the best offsets with a Kabsch fit and keep the residual.

A small residual over a long window means the two recordings are the same take
and the fitted transform is the missing registration.  A large residual means
they are not, and no transform is reported.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.lerobot_v3 import load_episode, open_dataset  # noqa: E402


SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--episodes",
        default="all",
        help="all, a single index, or a comma/range list such as 0,2-4",
    )
    parser.add_argument(
        "--coarse-stride",
        type=int,
        default=3,
        help="frame stride used while scanning candidate offsets",
    )
    parser.add_argument(
        "--max-accept-rmse-m",
        type=float,
        default=0.02,
        help="Kabsch residual below which a dataset episode is accepted as the same take",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_episode_list(text: str, total: int) -> list[int]:
    if text.strip() == "all":
        return list(range(total))
    chosen: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            chosen.extend(range(int(start), int(end) + 1))
        else:
            chosen.append(int(token))
    return sorted(set(chosen))


def load_raw_positions(pose_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Return hardware timestamps in seconds and XYZ from one VINS pose CSV."""
    times: list[float] = []
    xyz: list[list[float]] = []
    with (pose_dir / f"pose_data_{side}.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            times.append(int(row["Timestamp_us"]) / 1e6)
            xyz.append([float(row["X"]), float(row["Y"]), float(row["Z"])])
    return np.asarray(times, dtype=float), np.asarray(xyz, dtype=float)


def resample_to_grid(
    times_s: np.ndarray, values: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample an irregular trajectory onto a fixed-rate grid."""
    if times_s.ndim != 1 or values.shape[0] != times_s.shape[0]:
        raise ValueError("timestamps and values must share their first axis")
    grid = np.arange(times_s[0], times_s[-1], 1.0 / fps)
    resampled = np.stack(
        [np.interp(grid, times_s, values[:, axis]) for axis in range(values.shape[1])],
        axis=-1,
    )
    return grid, resampled


def speed_profile(positions: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame speed, which is invariant to any rigid transform."""
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) * fps


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit target ~= rotation @ source + translation and return the RMSE."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must be matching Nx3 arrays")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ correction @ vt
    translation = target_mean - rotation @ source_mean
    residual = target - (source @ rotation.T + translation)
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return rotation, translation, rmse


def best_offset_by_speed(
    raw_speed: np.ndarray, episode_speed: np.ndarray, stride: int
) -> list[int]:
    """Rank window offsets by the correlation of their speed profiles."""
    window = episode_speed.shape[0]
    limit = raw_speed.shape[0] - window
    if limit < 0:
        return []
    reference = episode_speed - episode_speed.mean()
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return []
    scores = []
    for offset in range(0, limit + 1, stride):
        segment = raw_speed[offset : offset + window]
        centered = segment - segment.mean()
        norm = float(np.linalg.norm(centered))
        if norm == 0.0:
            continue
        scores.append((float(centered @ reference / (norm * reference_norm)), offset))
    scores.sort(reverse=True)
    return [offset for _, offset in scores[:8]]


def refine_offset(
    raw_positions: np.ndarray,
    episode_positions: np.ndarray,
    seed_offsets: list[int],
    stride: int,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Search near each seed offset for the smallest Kabsch residual."""
    window = episode_positions.shape[0]
    best = (float("inf"), -1, np.eye(3), np.zeros(3))
    limit = raw_positions.shape[0] - window
    candidates = set()
    for seed in seed_offsets:
        candidates.update(
            offset
            for offset in range(seed - stride, seed + stride + 1)
            if 0 <= offset <= limit
        )
    for offset in sorted(candidates):
        source = raw_positions[offset : offset + window]
        rotation, translation, rmse = kabsch(source, episode_positions)
        if rmse < best[0]:
            best = (rmse, offset, rotation, translation)
    rmse, offset, rotation, translation = best
    return offset, rotation, translation, rmse


def main() -> int:
    args = parse_args()
    dataset = open_dataset(args.dataset.expanduser().resolve())
    total_episodes = int(dataset.info["total_episodes"])
    episodes = parse_episode_list(args.episodes, total_episodes)

    raw_grid: dict[str, np.ndarray] = {}
    raw_positions: dict[str, np.ndarray] = {}
    raw_speed: dict[str, np.ndarray] = {}
    for side in SIDES:
        times_s, xyz = load_raw_positions(args.pose_dir.expanduser().resolve(), side)
        grid, resampled = resample_to_grid(times_s, xyz, args.fps)
        raw_grid[side] = grid
        raw_positions[side] = resampled
        raw_speed[side] = speed_profile(resampled, args.fps)

    results = []
    for episode_index in episodes:
        episode = load_episode(dataset, episode_index)
        entry: dict[str, object] = {
            "episode_index": episode_index,
            "frame_count": int(episode.frame_count),
            "sides": {},
        }
        worst_rmse = 0.0
        for side_index, side in enumerate(SIDES):
            episode_positions = np.asarray(
                episode.positions_world_m[:, side_index], dtype=float
            )
            episode_speed = speed_profile(episode_positions, args.fps)
            seeds = best_offset_by_speed(
                raw_speed[side], episode_speed, args.coarse_stride
            )
            if not seeds:
                entry["sides"][side] = {"status": "episode_longer_than_raw_recording"}
                worst_rmse = float("inf")
                continue
            offset, rotation, translation, rmse = refine_offset(
                raw_positions[side], episode_positions, seeds, args.coarse_stride
            )
            entry["sides"][side] = {
                "offset_frames": int(offset),
                "offset_s": float(offset / args.fps),
                "raw_window_start_unix_s": float(raw_grid[side][offset]),
                "rmse_m": rmse,
                "rotation_dataset_from_raw": rotation.tolist(),
                "translation_dataset_from_raw_m": translation.tolist(),
            }
            worst_rmse = max(worst_rmse, rmse)
        entry["worst_rmse_m"] = worst_rmse
        entry["same_take"] = bool(worst_rmse <= args.max_accept_rmse_m)
        results.append(entry)
        print(
            f"episode {episode_index:3d} frames={entry['frame_count']:4d} "
            f"worst_rmse={worst_rmse:.4f} m"
            f"{'  <-- candidate' if entry['same_take'] else ''}",
            flush=True,
        )

    results.sort(key=lambda item: item["worst_rmse_m"])
    report = {
        "pose_dir": str(args.pose_dir.expanduser().resolve()),
        "dataset": str(args.dataset.expanduser().resolve()),
        "fps": args.fps,
        "max_accept_rmse_m": args.max_accept_rmse_m,
        "accepted": [item for item in results if item["same_take"]],
        "ranked": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {args.output}")
    best = results[0]
    print(
        f"best match: episode {best['episode_index']} "
        f"worst_rmse={best['worst_rmse_m']:.4f} m same_take={best['same_take']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
