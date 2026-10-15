#!/usr/bin/env python3
"""Build a synchronized 16:9 camera + three-embodiment replay video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq


CAMERAS = (
    ("observation.images.cam_left_wrist", "left_umi_wrist", "UMI left wrist"),
    ("observation.images.ego_left", "ego_left", "Ego left"),
    ("observation.images.cam_right_wrist", "right_umi_wrist", "UMI right wrist"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--humanoid-video", required=True)
    parser.add_argument("--tianji-video", required=True)
    parser.add_argument("--realman-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="episode_000_camera_robot_replay_16x9.mp4")
    parser.add_argument(
        "--balance",
        type=float,
        default=0.4,
        help="OpenCV fisheye crop/FOV balance in [0, 1]",
    )
    parser.add_argument("--reuse-cameras", action="store_true")
    return parser.parse_args()


def _episode_row(dataset: Path, episode: int) -> dict:
    files = sorted((dataset / "meta" / "episodes").glob("**/*.parquet"))
    for path in files:
        for row in pq.read_table(path).to_pylist():
            if int(row["episode_index"]) == episode:
                return row
    raise ValueError(f"dataset has no episode {episode}")


def _matrix_from_intrinsics(feature: dict) -> tuple[np.ndarray, tuple[int, int]]:
    intrinsics = feature.get("intrinsics") or {}
    if len(intrinsics) != 1:
        raise ValueError("camera feature must contain exactly one calibrated resolution")
    resolution, values = next(iter(intrinsics.items()))
    width, height = (int(value) for value in resolution.lower().split("x", 1))
    matrix = np.asarray(
        [
            [values["fx"], 0.0, values["ppx"]],
            [0.0, values["fy"], values["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix, (width, height)


def _camera_extrinsic(feature: dict) -> tuple[str, np.ndarray]:
    for name in ("base_link_to_main_camera", "imu_to_rgb_camera"):
        if name in feature:
            matrix = np.asarray(feature[name], dtype=float)
            if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError(f"invalid {name} matrix")
            return name, matrix
    raise ValueError("camera feature is missing its declared extrinsic transform")


def _video_source(dataset: Path, info: dict, row: dict, key: str) -> Path:
    prefix = f"videos/{key}"
    template = info["video_path"]
    path = dataset / template.format(
        video_key=key,
        chunk_index=int(row[f"{prefix}/chunk_index"]),
        file_index=int(row[f"{prefix}/file_index"]),
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _undistort_episode(
    source: Path,
    output: Path,
    feature: dict,
    start_timestamp: float,
    frame_count: int,
    fps: float,
    balance: float,
) -> None:
    matrix, calibrated_size = _matrix_from_intrinsics(feature)
    distortion = np.asarray(feature["distortion_coeffs"], dtype=np.float64).reshape(4, 1)
    if feature.get("distortion_model") != "equidistant":
        raise ValueError(f"unsupported distortion model: {feature.get('distortion_model')}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    if size != calibrated_size:
        scale_x = width / calibrated_size[0]
        scale_y = height / calibrated_size[1]
        matrix[0, :] *= scale_x
        matrix[1, :] *= scale_y

    new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        matrix, distortion, size, np.eye(3), balance=balance
    )
    focal_ratio = new_matrix[0, 0] / matrix[0, 0]
    principal_point_valid = (
        0.0 <= new_matrix[0, 2] < width
        and 0.0 <= new_matrix[1, 2] < height
    )
    # OpenCV can return a finite but unusable projection for very wide lenses.
    # In particular, the right-wrist calibration in the September dataset
    # produces a negative output principal point at balance=0.4, which crops
    # away the gripper and stretches the scene.  Preserve the calibrated K in
    # that case instead of accepting the degenerate auto-estimate.
    if not (0.25 <= focal_ratio <= 2.0 and principal_point_valid):
        new_matrix = matrix.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        matrix, distortion, np.eye(3), new_matrix, size, cv2.CV_16SC2
    )

    start_frame = int(round(start_timestamp * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    written = 0
    while written < frame_count:
        ok, frame = capture.read()
        if not ok:
            break
        corrected = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        encoder.stdin.write(corrected.tobytes())
        written += 1
    capture.release()
    encoder.stdin.close()
    code = encoder.wait()
    if code:
        raise RuntimeError(f"camera encoder failed with code {code}")
    if written != frame_count:
        raise RuntimeError(
            f"camera segment produced {written} frames, expected {frame_count}"
        )


def _compose(
    cameras: list[Path], robots: list[Path], output: Path,
    frame_count: int, fps: float,
) -> None:
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    labels = ("UMI Left Wrist", "Ego Left", "UMI Right Wrist")
    filters: list[str] = []
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]trim=end_frame={frame_count},setpts=N/({fps:g}*TB),"
            "scale=640:360:force_original_aspect_ratio=increase,crop=640:360,"
            f"drawtext=fontfile={font}:text='{label}':x=18:y=14:fontsize=28:"
            "fontcolor=white:borderw=2:bordercolor=black@0.8[c" + str(index) + "]"
        )
    for index in range(3):
        input_index = index + 3
        filters.append(
            f"[{input_index}:v]trim=end_frame={frame_count},"
            f"setpts=N/({fps:g}*TB),scale=640:720[r{index}]"
        )
    filters.extend(
        [
            "[c0][c1][c2]hstack=inputs=3[top]",
            "[r0][r1][r2]hstack=inputs=3[bottom]",
            "[top][bottom]vstack=inputs=2,"
            "drawbox=x=0:y=358:w=1920:h=4:color=black:t=fill,"
            "drawtext=fontfile=" + font + ":text='Blue\\: left-arm workspace':"
            "x=330:y=1040:fontsize=28:fontcolor=0x309AFF:borderw=2:bordercolor=black,"
            "drawtext=fontfile=" + font + ":text='Orange\\: right-arm workspace':"
            "x=760:y=1040:fontsize=28:fontcolor=0xFF772D:borderw=2:bordercolor=black,"
            "drawtext=fontfile=" + font + ":text='Purple\\: spatial overlap':"
            "x=1280:y=1040:fontsize=28:fontcolor=0xB860F0:borderw=2:bordercolor=black[out]",
        ]
    )
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path in cameras + robots:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
            "-frames:v", str(frame_count), "-r", f"{fps:g}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
    )
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    row = _episode_row(dataset, args.episode)
    fps = float(info["fps"])
    frame_count = int(row["length"])

    camera_paths: list[Path] = []
    calibration_report: dict[str, dict] = {}
    for key, stem, _ in CAMERAS:
        feature = info["features"][key]
        extrinsic_name, extrinsic = _camera_extrinsic(feature)
        source = _video_source(dataset, info, row, key)
        prefix = f"videos/{key}"
        output = output_dir / "cameras" / f"{stem}_undistorted.mp4"
        if not (args.reuse_cameras and output.is_file()):
            _undistort_episode(
                source,
                output,
                feature,
                float(row[f"{prefix}/from_timestamp"]),
                frame_count,
                fps,
                args.balance,
            )
        camera_paths.append(output)
        calibration_report[key] = {
            "source": str(source),
            "intrinsics": feature["intrinsics"],
            "distortion_model": feature["distortion_model"],
            "distortion_coeffs": feature["distortion_coeffs"],
            "extrinsic_name": extrinsic_name,
            "extrinsic": extrinsic.tolist(),
            "extrinsic_usage": (
                "validated for camera/frame identity; per-camera pixel undistortion "
                "uses intrinsics and distortion only"
            ),
        }

    robots = [
        Path(args.humanoid_video).expanduser().resolve(),
        Path(args.tianji_video).expanduser().resolve(),
        Path(args.realman_video).expanduser().resolve(),
    ]
    for path in robots:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = output_dir / args.output_name
    _compose(camera_paths, robots, output, frame_count, fps)

    manifest = {
        "dataset": str(dataset),
        "episode": args.episode,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / fps,
        "layout": {
            "canvas": [1920, 1080],
            "top": ["UMI left wrist", "Ego left", "UMI right wrist"],
            "bottom": ["DM-SPARKY (full-body)", "Tianji", "RealMan RM75-B"],
        },
        "calibration": calibration_report,
        "robot_videos": [str(path) for path in robots],
        "output": str(output),
    }
    (output_dir / "episode_000_composition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COMBINED_REPLAY_WRITTEN output={output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
