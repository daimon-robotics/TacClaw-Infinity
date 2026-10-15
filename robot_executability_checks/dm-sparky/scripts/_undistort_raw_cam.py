#!/usr/bin/env python3
"""Undistort a fisheye (equidistant) recorded video using stored intrinsics."""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--ppx", type=float, required=True)
    parser.add_argument("--ppy", type=float, required=True)
    parser.add_argument("--dist", type=float, nargs=4, required=True)
    parser.add_argument("--balance", type=float, default=0.4)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    K = np.array(
        [[args.fx, 0.0, args.ppx], [0.0, args.fy, args.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.array(args.dist, dtype=np.float64).reshape(4, 1)

    capture = cv2.VideoCapture(args.input)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    if args.balance < 0:
        new_K = K.copy()
    else:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, size, np.eye(3), balance=args.balance
        )
        focal_ratio = new_K[0, 0] / K[0, 0]
        principal_point_valid = (
            0.0 <= new_K[0, 2] < width
            and 0.0 <= new_K[1, 2] < height
        )
        if not (0.25 <= focal_ratio <= 2.0 and principal_point_valid):
            # estimateNewCameraMatrixForUndistortRectify can degenerate for
            # wide lenses even when its focal length remains finite.  Reject
            # estimates whose principal point lies outside the output image.
            new_K = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, size, cv2.CV_16SC2
    )

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{args.fps:g}",
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", args.output,
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        undistorted = cv2.remap(
            frame, map1, map2, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        encoder.stdin.write(undistorted.tobytes())
        frame_count += 1
    capture.release()
    encoder.stdin.close()
    code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg encoder failed with code {code}")
    print(f"UNDISTORTED frames={frame_count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
