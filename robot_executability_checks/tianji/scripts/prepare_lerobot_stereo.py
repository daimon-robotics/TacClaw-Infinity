#!/usr/bin/env python3
"""Prepare one LeRobot episode for the sibling stereo_depth pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_row(dataset: Path, episode: int) -> dict[str, object]:
    files = sorted((dataset / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing episode metadata under {dataset}")
    for path in files:
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "videos/observation.images.ego_left/chunk_index",
                "videos/observation.images.ego_left/file_index",
                "videos/observation.images.ego_right/chunk_index",
                "videos/observation.images.ego_right/file_index",
            ],
        )
        for row in table.to_pylist():
            if int(row["episode_index"]) == episode:
                return row
    raise ValueError(f"episode {episode} was not found in {dataset}")


def _video_path(dataset: Path, row: dict[str, object], side: str) -> Path:
    prefix = f"videos/observation.images.ego_{side}"
    chunk = int(row[f"{prefix}/chunk_index"])
    index = int(row[f"{prefix}/file_index"])
    directory = dataset / prefix / f"chunk-{chunk:03d}"
    matches = sorted(directory.glob(f"file-{index:03d}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one video for episode mapping {prefix} chunk={chunk} file={index}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _camera(info: dict[str, object], side: str) -> dict[str, object]:
    feature = info["features"][f"observation.images.ego_{side}"]
    height, width = map(int, feature["shape"][-2:])
    resolution = f"{width}x{height}"
    return {
        "intrinsics": {resolution: feature["intrinsics"][resolution]},
        "distortion_coeffs": feature["distortion_coeffs"],
    }


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    row = _episode_row(dataset, args.episode)
    left = _video_path(dataset, row, "left")
    right = _video_path(dataset, row, "right")

    calibration = {
        "calibration_info": {
            "calibration_status": "lerobot_info_json",
            "format_version": "1.0",
            "source": str(dataset / "meta" / "info.json"),
        },
        "observation": {
            "images": {
                "rgb": {
                    "cam0": _camera(info, "left"),
                    "cam1": _camera(info, "right"),
                    "camera_model": "pinhole",
                    "distortion_model": "equidistant",
                    "extrinsics": {
                        "T_ic_imu0_cam0": info["features"][
                            "observation.images.ego_left"
                        ]["imu_to_rgb_camera"],
                        "T_ic_imu0_cam1": info["features"][
                            "observation.images.ego_right"
                        ]["imu_to_rgb_camera"],
                    },
                }
            }
        },
    }
    calibration_path = output / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    stereo_path = output / "rgb.mp4"
    if args.overwrite or not stereo_path.exists():
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(left), "-i", str(right),
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
                "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "10", "-pix_fmt", "yuv420p", str(stereo_path),
            ],
            check=True,
        )

    manifest = {
        "dataset": str(dataset),
        "episode": int(args.episode),
        "left_video": str(left),
        "right_video": str(right),
        "stereo_video": str(stereo_path),
        "calibration": str(calibration_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"LEROBOT_STEREO_WRITTEN output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
