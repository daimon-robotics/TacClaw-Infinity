#!/usr/bin/env python3
"""Compute and cache collision-free dual-arm TCP workspaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.workspace import (  # noqa: E402
    build_workspace_envelopes,
    compute_workspace,
)
from tianji_lerobot_replay.workspace_specs import get_workspace_spec  # noqa: E402
from prepare_workspace_model import prepare as prepare_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("tianji", "humanoid", "realman"))
    parser.add_argument("--samples", type=int, default=2**15)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-envelope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0 or args.samples & (args.samples - 1):
        raise ValueError("--samples must be a positive power of two")
    if not math.isfinite(args.voxel_size) or args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    cache = PROJECT_ROOT / ".workspace_cache" / args.robot
    model_manifest = prepare_model(args.robot, cache)
    output = Path(args.output).expanduser() if args.output else cache / "workspace.npz"

    def progress(side: str, done: int, total: int, elapsed: float) -> None:
        fraction = done / total
        eta = elapsed * (1.0 / fraction - 1.0) if fraction else float("inf")
        print(
            f"WORKSPACE_PROGRESS robot={args.robot} side={side} "
            f"tasks={done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )

    metadata = compute_workspace(
        model_path=model_manifest["derived_xml"],
        robot=args.robot,
        sample_count=args.samples,
        voxel_size_m=args.voxel_size,
        seed=args.seed,
        jobs=args.jobs,
        output=output,
        progress=progress,
    )
    envelopes = {}
    if not args.no_envelope:
        spec = get_workspace_spec(args.robot)
        envelopes = build_workspace_envelopes(
            output,
            model_manifest["derived_xml"],
            spec.reference_body,
            cache / "envelope",
        )
    summary = {
        "robot": args.robot,
        "workspace": str(output.resolve()),
        "sampling": metadata["sampling"],
        "occupancy": metadata["occupancy"],
        "envelopes": {key: str(value) for key, value in envelopes.items()},
    }
    print("WORKSPACE_COMPLETE " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
