#!/usr/bin/env python3
"""Validate cached workspaces against solved replay trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_dilation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.workspace_specs import get_workspace_spec  # noqa: E402
from tianji_lerobot_replay.workspace_visualization import WorkspaceOverlay  # noqa: E402
from render_workspace_visualization import PoseWriter  # noqa: E402


ROBOTS = ("tianji", "humanoid", "realman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(
        PROJECT_ROOT / "outputs/collision_free_workspace/validation_report.md"))
    parser.add_argument("--benchmark-frames", type=int, default=1000)
    return parser.parse_args()


def _rows(path: Path, episode: int = 0) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        result = [
            row for row in csv.DictReader(stream)
            if int(row["episode_index"]) == episode
        ]
    result.sort(key=lambda row: int(row["frame_index"]))
    if not result:
        raise RuntimeError(f"no episode {episode} rows in {path}")
    return result


def _local_position(data: object, reference_body_id: int, world: np.ndarray) -> np.ndarray:
    rotation = data.xmat[reference_body_id].reshape(3, 3)
    return rotation.T @ (world - data.xpos[reference_body_id])


def _coverage(
    positions: np.ndarray,
    occupancy: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> dict[str, float | int]:
    indices = np.floor((positions - origin) / voxel_size + 1e-9).astype(int)
    valid = np.all((indices >= 0) & (indices < np.asarray(occupancy.shape)), axis=1)
    raw = np.zeros(len(indices), dtype=bool)
    tolerant = np.zeros(len(indices), dtype=bool)
    if np.any(valid):
        key = tuple(indices[valid].T)
        raw[valid] = occupancy[key]
        expanded = binary_dilation(
            occupancy, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
        tolerant[valid] = expanded[key]
    return {
        "frames": int(len(positions)),
        "raw_inside": int(np.count_nonzero(raw)),
        "raw_ratio": float(np.mean(raw)),
        "tolerance_inside": int(np.count_nonzero(tolerant)),
        "tolerance_ratio": float(np.mean(tolerant)),
    }


def _cache_metrics(
    path: Path,
    positions: dict[str, np.ndarray],
    shoulder_origins: dict[str, np.ndarray],
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"]))
        origin = np.asarray(cache["grid_origin_m"], dtype=float)
        voxel = float(cache["voxel_size_m"])
        coverage = {
            side: _coverage(
                positions[side], np.asarray(cache[f"occ_{side}_raw"], dtype=bool),
                origin, voxel)
            for side in ("left", "right")
        }
        maximum_radius = {}
        for side in ("left", "right"):
            centers = origin + np.argwhere(cache[f"occ_{side}_raw"]) * voxel
            maximum_radius[side] = {
                "torso_origin_max_m": float(np.max(np.linalg.norm(centers, axis=1))),
                "shoulder_origin_max_m": float(np.max(
                    np.linalg.norm(centers - shoulder_origins[side], axis=1))),
                "shoulder_origin_local_m": shoulder_origins[side].tolist(),
            }
    return {
        "path": str(path),
        "samples_per_arm": metadata["sampling"]["samples_per_arm"],
        "voxel_size_m": voxel,
        "occupancy": metadata["occupancy"],
        "eta_statistics": metadata["eta_statistics"],
        "coverage": coverage,
        "maximum_radius": maximum_radius,
    }


def _percentiles_ms(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float) * 1000.0
    return {
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def validate_robot(robot: str, benchmark_frames: int) -> dict[str, object]:
    import mujoco

    spec = get_workspace_spec(robot)
    root = PROJECT_ROOT / ".workspace_cache" / robot
    main_cache = root / "workspace.npz"
    model = mujoco.MjModel.from_xml_path(str(root / "envelope/workspace_envelope.xml"))
    data = mujoco.MjData(model)
    config = load_scene_config(spec.scene_config)
    writer = PoseWriter(mujoco, model, robot, config)
    rows = _rows(spec.quality_csv)
    reference_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.reference_body)
    site_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_sites[side])
        for side in ("left", "right")
    }
    positions = {side: [] for side in ("left", "right")}
    for row in rows:
        writer.apply(data, row)
        for side in ("left", "right"):
            positions[side].append(_local_position(
                data, reference_id, data.site_xpos[site_ids[side]]))
    position_arrays = {side: np.asarray(values) for side, values in positions.items()}

    writer.apply(data, rows[len(rows) // 2])
    reference_rotation = data.xmat[reference_id].reshape(3, 3)
    reference_position = data.xpos[reference_id]
    shoulder_origins = {}
    for side in ("left", "right"):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_names[side][0])
        shoulder_origins[side] = reference_rotation.T @ (
            data.xanchor[joint_id] - reference_position)

    runs = {
        "A": _cache_metrics(main_cache, position_arrays, shoulder_origins),
        "B": _cache_metrics(
            root / "sensitivity_n262144_v004.npz", position_arrays, shoulder_origins),
        "C": _cache_metrics(
            root / "sensitivity_n1048576_v005.npz", position_arrays, shoulder_origins),
    }

    overlay = WorkspaceOverlay(model, main_cache, initial_mode=0)
    benchmark = {}
    for mode, label in ((0, "off"), (1, "envelope"), (2, "heatmap")):
        overlay.mode = mode
        overlay.apply_envelope_visibility()
        scene = mujoco.MjvScene(model, maxgeom=6000)
        frame_times: list[float] = []
        overlay_times: list[float] = []
        started = time.perf_counter()
        for index in range(benchmark_frames):
            row = rows[index % len(rows)]
            frame_start = time.perf_counter()
            writer.apply(data, row)
            overlay_start = time.perf_counter()
            overlay.update_user_scene(scene, data)
            overlay_times.append(time.perf_counter() - overlay_start)
            frame_times.append(time.perf_counter() - frame_start)
        wall = time.perf_counter() - started
        benchmark[label] = {
            "frames": benchmark_frames,
            "headless_pipeline_fps": benchmark_frames / wall,
            "frame": _percentiles_ms(frame_times),
            "overlay_update": _percentiles_ms(overlay_times),
            "viewer_sync": None,
        }
    off_fps = benchmark["off"]["headless_pipeline_fps"]
    for label in ("envelope", "heatmap"):
        benchmark[label]["degradation_fraction"] = (
            1.0 - benchmark[label]["headless_pipeline_fps"] / off_fps)

    metadata = json.loads(main_cache.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "robot": robot,
        "title": spec.title,
        "trajectory_frames": len(rows),
        "reference_body": spec.reference_body,
        "sites": spec.ee_sites,
        "runs": runs,
        "maximum_radius": runs["A"]["maximum_radius"],
        "sampling": metadata["sampling"],
        "collision_filtering": metadata["collision_filtering"],
        "grid": metadata["grid"],
        "model_audit": metadata["model_audit"],
        "performance": benchmark,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _report(results: list[dict[str, object]]) -> str:
    lines = [
        "# Collision-free dual-arm workspace validation",
        "",
        "MuJoCo version: **3.10.0**. The sampler uses `mj_fwdPosition` for the "
        "position/collision pipeline and `mj_jacSite` for the TCP Jacobian. Each arm "
        "uses seven independently limited hinge joints sampled by scrambled Sobol "
        "sequences with fixed seeds.",
        "",
        "The reported overlap is the spatial intersection of the two independently "
        "sampled collision-free arm workspaces; it is not a simultaneous 14-DoF "
        "bimanual collision-free workspace.",
        "",
    ]
    for result in results:
        run_a = result["runs"]["A"]
        occupancy = run_a["occupancy"]
        lines.extend([
            f"## {result['title']}",
            "",
            f"Reference body: `{result['reference_body']}`; TCP sites: "
            f"`{result['sites']['left']}` and `{result['sites']['right']}`; "
            f"validation trajectory: {result['trajectory_frames']} frames.",
            f"Run A uses {result['sampling']['samples_per_arm']} samples per arm, "
            f"seed L/R = {result['sampling']['seed_left']} / "
            f"{result['sampling']['seed_right']}, and a 0.04-m shared voxel grid. "
            f"The qpos0 audit found "
            f"{result['model_audit']['baseline_contact_pair_count']} baseline contact pairs.",
            "",
            "### Joint mapping",
            "",
            "| Arm | Joint | Joint ID | qpos address | DoF address | Range (rad) |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for side in ("left", "right"):
            for joint in result["model_audit"]["joints"][side]:
                lines.append(
                    f"| {side.title()} | `{joint['name']}` | {joint['joint_id']} | "
                    f"{joint['qpos_address']} | {joint['dof_address']} | "
                    f"[{joint['range'][0]:.5f}, {joint['range'][1]:.5f}] |"
                )
        lines.extend([
            "",
            "### Run A quantitative results",
            "",
            "| Metric | Left | Right | Overlap |",
            "|---|---:|---:|---:|",
            f"| Occupied volume (m³) | {occupancy['left_volume_m3']:.6f} | "
            f"{occupancy['right_volume_m3']:.6f} | {occupancy['overlap_volume_m3']:.6f} |",
            f"| Overlap / arm | {_pct(occupancy['overlap_over_left'])} | "
            f"{_pct(occupancy['overlap_over_right'])} | — |",
            f"| Raw replay coverage | {_pct(run_a['coverage']['left']['raw_ratio'])} | "
            f"{_pct(run_a['coverage']['right']['raw_ratio'])} | — |",
            f"| ≤1-voxel coverage | {_pct(run_a['coverage']['left']['tolerance_ratio'])} | "
            f"{_pct(run_a['coverage']['right']['tolerance_ratio'])} | — |",
            "",
            "The raw occupancy is used for all quantitative values. One-voxel binary "
            f"closing is visualization-only; changes were "
            f"L +{result['grid']['closing']['left']['added_voxels']} / "
            f"−{result['grid']['closing']['left']['deleted_voxels']} and "
            f"R +{result['grid']['closing']['right']['added_voxels']} / "
            f"−{result['grid']['closing']['right']['deleted_voxels']} voxels.",
            "",
            "### Manipulability and radius",
            "",
            "| Arm | η min | η p1 | η median | η p95 | η p99 | η max | "
            "Max radius from torso (m) | Max radius from shoulder (m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for side in ("left", "right"):
            eta = run_a["eta_statistics"][side]
            radius = result["maximum_radius"][side]
            lines.append(
                f"| {side.title()} | {eta['min']:.6f} | {eta['p1']:.6f} | "
                f"{eta['median']:.6f} | {eta['p95']:.6f} | {eta['p99']:.6f} | "
                f"{eta['max']:.6f} | {radius['torso_origin_max_m']:.4f} | "
                f"{radius['shoulder_origin_max_m']:.4f} |"
            )
        lines.extend([
            "",
            "### Sensitivity",
            "",
            "| Run | Samples/arm | Voxel (m) | Left vol. | Right vol. | Overlap vol. | "
            "Raw coverage L/R | Max torso radius L/R | η median L/R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("A", "B", "C"):
            run = result["runs"][name]
            occ = run["occupancy"]
            cov = run["coverage"]
            radius = run["maximum_radius"]
            eta = run["eta_statistics"]
            lines.append(
                f"| {name} | {run['samples_per_arm']} | {run['voxel_size_m']:.2f} | "
                f"{occ['left_volume_m3']:.4f} | {occ['right_volume_m3']:.4f} | "
                f"{occ['overlap_volume_m3']:.4f} | "
                f"{_pct(cov['left']['raw_ratio'])} / {_pct(cov['right']['raw_ratio'])} | "
                f"{radius['left']['torso_origin_max_m']:.4f} / "
                f"{radius['right']['torso_origin_max_m']:.4f} | "
                f"{eta['left']['median']:.4f} / {eta['right']['median']:.4f} |"
            )
        lines.extend([
            "",
            "### Collision filtering",
            "",
            "A sampled state is rejected for sampled-arm self-collision, collision "
            "with torso/base, or collision with the static table/environment. "
            "Opposite-arm contacts are counted but do not reject the individual-arm "
            "workspace. Baseline qpos0 contacts are accepted only while they do not "
            "become more than 0.5 mm deeper.",
            "",
            "| Arm | Accepted | Rejected any | Self | Torso/base | Environment | Opposite arm (count only) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for side in ("left", "right"):
            stats = result["sampling"]["stats"][side]
            lines.append(
                f"| {side.title()} | {stats['accepted']} | {stats['rejected_any']} | "
                f"{stats['rejected_self']} | {stats['rejected_torso_base']} | "
                f"{stats['rejected_environment']} | {stats['opposite_arm_collision']} |"
            )
        lines.extend([
            "",
            "### Headless 1000-frame pipeline benchmark",
            "",
            "| Mode | FPS | Mean frame (ms) | p50 | p95 | Overlay update mean (ms) | Headless pipeline Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label in ("off", "envelope", "heatmap"):
            item = result["performance"][label]
            degradation = item.get("degradation_fraction")
            lines.append(
                f"| {label} | {item['headless_pipeline_fps']:.1f} | "
                f"{item['frame']['mean_ms']:.3f} | {item['frame']['p50_ms']:.3f} | "
                f"{item['frame']['p95_ms']:.3f} | "
                f"{item['overlay_update']['mean_ms']:.3f} | "
                f"{'—' if degradation is None else _pct(degradation)} |"
            )
        lines.extend([
            "",
            "`viewer.sync()` timing is not reported because this machine cannot create "
            "a GLFW/GLX interactive context. The table therefore measures the identical "
            "1000-frame kinematic and overlay-update path headlessly and must not be "
            "interpreted as display FPS. Consequently, the ≤30% interactive display "
            "degradation target remains unverified on this headless host and must be "
            "checked on a desktop with a working GLX context.",
            "",
        ])
    lines.extend([
        "## Known limitations",
        "",
        "The workspace is position-only after sampling full TCP orientations through "
        "the arm joint space. It does not enforce a particular task orientation, "
        "simultaneous two-arm collision freedom, velocity/acceleration limits, payload, "
        "or hardware safety. Mode-B shell rendering may be explicitly spatially "
        "subsampled when the MuJoCo scene geometry capacity is exceeded; raw occupancy "
        "and all metrics remain unchanged.",
        "",
        "## API references",
        "",
        "- MuJoCo API functions: https://mujoco.readthedocs.io/en/3.10.0/APIreference/APIfunctions.html",
        "- Python passive viewer: https://mujoco.readthedocs.io/en/3.1.2/python.html",
        "- MJCF collision filtering: https://mujoco.readthedocs.io/en/3.2.2/XMLreference.html",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    results = [validate_robot(robot, args.benchmark_frames) for robot in ROBOTS]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_report(results), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WORKSPACE_VALIDATION_WRITTEN output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
