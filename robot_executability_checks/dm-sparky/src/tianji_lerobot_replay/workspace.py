"""Collision-free per-arm workspace sampling, voxelization, and caching."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
from typing import Any
import warnings
import xml.etree.ElementTree as ET

import numpy as np

from .workspace_specs import get_workspace_spec


SIDES = ("left", "right")
ORIENTATION_WEIGHT_M_PER_RAD = 0.30
BASELINE_PENETRATION_TOLERANCE_M = 0.0005

_WORKER: dict[str, Any] = {}


def _name(mujoco: Any, model: Any, obj: Any, index: int) -> str:
    return mujoco.mj_id2name(model, obj, int(index)) or f"unnamed_{index}"


def _descendant_bodies(model: Any, root_body: int) -> set[int]:
    result = {int(root_body)}
    changed = True
    while changed:
        changed = False
        for body_id in range(1, model.nbody):
            if body_id not in result and int(model.body_parentid[body_id]) in result:
                result.add(body_id)
                changed = True
    return result


def _model_mapping(model_path: str, robot: str) -> dict[str, Any]:
    import mujoco

    spec = get_workspace_spec(robot)
    model = mujoco.MjModel.from_xml_path(model_path)
    joint_ids: dict[str, np.ndarray] = {}
    qpos_addresses: dict[str, np.ndarray] = {}
    dof_addresses: dict[str, np.ndarray] = {}
    limits: dict[str, np.ndarray] = {}
    arm_bodies: dict[str, set[int]] = {}
    arm_geoms: dict[str, set[int]] = {}
    site_ids: dict[str, int] = {}
    for side in SIDES:
        ids = np.asarray([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in spec.joint_names[side]
        ], dtype=int)
        if np.any(ids < 0):
            raise RuntimeError(f"{robot} {side} arm joint mapping is incomplete")
        if np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"{robot} {side} arm contains a non-hinge joint")
        if not np.all(model.jnt_limited[ids]):
            raise RuntimeError(f"{robot} {side} arm contains an unlimited joint")
        joint_ids[side] = ids
        qpos_addresses[side] = model.jnt_qposadr[ids].astype(int)
        dof_addresses[side] = model.jnt_dofadr[ids].astype(int)
        limits[side] = model.jnt_range[ids].copy()
        first_body = int(model.jnt_bodyid[ids[0]])
        arm_bodies[side] = _descendant_bodies(model, first_body)
        arm_geoms[side] = {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in arm_bodies[side]
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        }
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_sites[side])
        if site_id < 0:
            raise RuntimeError(f"{robot} is missing EE site {spec.ee_sites[side]!r}")
        site_ids[side] = int(site_id)

    reference_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.reference_body)
    if reference_body_id < 0:
        raise RuntimeError(f"{robot} is missing reference body {spec.reference_body!r}")
    environment_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if (_name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            .endswith("_workspace_collision"))
    }
    collidable = {
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_contype[geom_id]) != 0
        and int(model.geom_conaffinity[geom_id]) != 0
    }
    base_geoms = collidable - environment_geoms - arm_geoms["left"] - arm_geoms["right"]
    if not arm_geoms["left"] or not arm_geoms["right"]:
        raise RuntimeError(f"{robot} does not have collision geoms on both arms")
    if not environment_geoms:
        raise RuntimeError(f"{robot} does not have workspace environment collisions")
    return {
        "model": model,
        "joint_ids": joint_ids,
        "qpos_addresses": qpos_addresses,
        "dof_addresses": dof_addresses,
        "limits": limits,
        "arm_geoms": arm_geoms,
        "environment_geoms": environment_geoms,
        "base_geoms": base_geoms,
        "site_ids": site_ids,
        "reference_body_id": int(reference_body_id),
    }


def _baseline_contacts(mujoco: Any, model: Any) -> tuple[dict[tuple[int, int], float], list[dict[str, Any]]]:
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_fwdPosition(model, data)
    depths: dict[tuple[int, int], float] = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        depths[pair] = min(depths.get(pair, math.inf), float(contact.dist))
    report = []
    for (geom1, geom2), distance in sorted(depths.items()):
        report.append({
            "geom1_id": geom1,
            "geom1_name": _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
            "geom1_body": _name(
                mujoco, model, mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom1])),
            "geom2_id": geom2,
            "geom2_name": _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
            "geom2_body": _name(
                mujoco, model, mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom2])),
            "distance_m": distance,
        })
    return depths, report


def inspect_workspace_model(model_path: str | Path, robot: str) -> dict[str, Any]:
    import mujoco

    model_path = str(Path(model_path).expanduser().resolve())
    mapping = _model_mapping(model_path, robot)
    model = mapping["model"]
    baseline, baseline_report = _baseline_contacts(mujoco, model)
    spec = get_workspace_spec(robot)
    joints = {}
    for side in SIDES:
        joints[side] = []
        for name, joint_id in zip(spec.joint_names[side], mapping["joint_ids"][side]):
            joints[side].append({
                "name": name,
                "joint_id": int(joint_id),
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "dof_address": int(model.jnt_dofadr[joint_id]),
                "range": model.jnt_range[joint_id].tolist(),
            })
    return {
        "model_path": model_path,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "robot": robot,
        "reference_body": spec.reference_body,
        "reference_body_id": mapping["reference_body_id"],
        "ee_sites": spec.ee_sites,
        "site_ids": mapping["site_ids"],
        "joints": joints,
        "geom_ids": {
            "left_arm": sorted(mapping["arm_geoms"]["left"]),
            "right_arm": sorted(mapping["arm_geoms"]["right"]),
            "torso_base": sorted(mapping["base_geoms"]),
            "environment": sorted(mapping["environment_geoms"]),
        },
        "neq": model.neq,
        "ntendon": model.ntendon,
        "baseline_contact_pair_count": len(baseline),
        "baseline_contacts": baseline_report,
    }


def _init_worker(model_path: str, robot: str) -> None:
    import mujoco

    mapping = _model_mapping(model_path, robot)
    baseline, _ = _baseline_contacts(mujoco, mapping["model"])
    mapping["baseline"] = baseline
    mapping["data"] = mujoco.MjData(mapping["model"])
    mapping["mujoco"] = mujoco
    _WORKER.clear()
    _WORKER.update(mapping)


def _contact_category(side: str, geom1: int, geom2: int) -> str | None:
    arm = _WORKER["arm_geoms"][side]
    other = _WORKER["arm_geoms"]["right" if side == "left" else "left"]
    environment = _WORKER["environment_geoms"]
    base = _WORKER["base_geoms"]
    if geom1 not in arm and geom2 not in arm:
        return None
    partner = geom2 if geom1 in arm else geom1
    if partner in arm:
        return "self"
    if partner in environment:
        return "environment"
    if partner in other:
        return "opposite_arm"
    if partner in base:
        return "torso_base"
    return "torso_base"


def _sample_range(task: tuple[str, int, int, int]) -> dict[str, Any]:
    from scipy.stats import qmc

    side, start, count, seed = task
    model = _WORKER["model"]
    data = _WORKER["data"]
    mujoco = _WORKER["mujoco"]
    low = _WORKER["limits"][side][:, 0]
    high = _WORKER["limits"][side][:, 1]
    sampler = qmc.Sobol(7, scramble=True, seed=seed)
    if start:
        sampler.fast_forward(start)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        unit = sampler.random(count)
    q_values = low + unit * (high - low)
    accepted_positions = []
    accepted_eta = []
    stats = {
        "samples": count,
        "accepted": 0,
        "rejected_any": 0,
        "rejected_self": 0,
        "rejected_torso_base": 0,
        "rejected_environment": 0,
        "opposite_arm_collision": 0,
    }
    qpos_addresses = _WORKER["qpos_addresses"][side]
    dof_addresses = _WORKER["dof_addresses"][side]
    site_id = _WORKER["site_ids"][side]
    reference_body_id = _WORKER["reference_body_id"]
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    baseline = _WORKER["baseline"]

    for q in q_values:
        data.qpos[:] = model.qpos0
        data.qpos[qpos_addresses] = q
        mujoco.mj_fwdPosition(model, data)
        categories: set[str] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            category = _contact_category(side, geom1, geom2)
            if category is None:
                continue
            pair = tuple(sorted((geom1, geom2)))
            baseline_distance = baseline.get(pair)
            limit = (
                baseline_distance - BASELINE_PENETRATION_TOLERANCE_M
                if baseline_distance is not None else -1e-7
            )
            if float(contact.dist) < limit:
                categories.add(category)
        if "opposite_arm" in categories:
            stats["opposite_arm_collision"] += 1
        rejecting = categories.intersection(("self", "torso_base", "environment"))
        for category in rejecting:
            stats[f"rejected_{category}"] += 1
        if rejecting:
            stats["rejected_any"] += 1
            continue

        reference_rotation = data.xmat[reference_body_id].reshape(3, 3)
        position_local = reference_rotation.T @ (
            data.site_xpos[site_id] - data.xpos[reference_body_id])
        jacp.fill(0.0)
        jacr.fill(0.0)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jacobian = np.vstack((jacp[:, dof_addresses], jacr[:, dof_addresses]))
        jacobian[3:, :] *= ORIENTATION_WEIGHT_M_PER_RAD
        singular = np.linalg.svd(jacobian, compute_uv=False)
        eta = float(singular[-1] / singular[0]) if singular[0] > 1e-12 else 0.0
        accepted_positions.append(position_local)
        accepted_eta.append(eta)
        stats["accepted"] += 1
    return {
        "start": start,
        "positions": np.asarray(accepted_positions, dtype=np.float32).reshape(-1, 3),
        "eta": np.asarray(accepted_eta, dtype=np.float32),
        "stats": stats,
    }


def _ranges(sample_count: int, task_count: int) -> list[tuple[int, int]]:
    task_count = max(1, min(task_count, sample_count))
    boundaries = np.linspace(0, sample_count, task_count + 1, dtype=int)
    return [
        (int(boundaries[index]), int(boundaries[index + 1] - boundaries[index]))
        for index in range(task_count)
        if boundaries[index + 1] > boundaries[index]
    ]


def sample_arm(
    model_path: str | Path,
    robot: str,
    side: str,
    sample_count: int,
    seed: int,
    jobs: int,
    progress: Any = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    model_path = str(Path(model_path).expanduser().resolve())
    jobs = max(1, int(jobs))
    task_ranges = _ranges(sample_count, max(jobs * 4, 1))
    tasks = [(side, start, count, seed) for start, count in task_ranges]
    results = []
    start_clock = time.perf_counter()
    if jobs == 1:
        _init_worker(model_path, robot)
        for task_index, task in enumerate(tasks, 1):
            results.append(_sample_range(task))
            if progress:
                progress(task_index, len(tasks), time.perf_counter() - start_clock)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=_init_worker,
            initargs=(model_path, robot),
        ) as executor:
            futures = [executor.submit(_sample_range, task) for task in tasks]
            for task_index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if progress:
                    progress(task_index, len(tasks), time.perf_counter() - start_clock)
    results.sort(key=lambda item: item["start"])
    positions = np.concatenate([item["positions"] for item in results], axis=0)
    eta = np.concatenate([item["eta"] for item in results], axis=0)
    stats = {key: 0 for key in results[0]["stats"]}
    for item in results:
        for key, value in item["stats"].items():
            stats[key] += int(value)
    stats["elapsed_s"] = time.perf_counter() - start_clock
    return positions, eta, stats


def _voxel_grid(
    points: dict[str, np.ndarray],
    eta: dict[str, np.ndarray],
    voxel_size_m: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from scipy.ndimage import binary_closing

    combined = np.concatenate([points[side] for side in SIDES], axis=0)
    if not len(combined):
        raise RuntimeError("workspace contains no collision-free samples")
    origin = np.floor(np.min(combined, axis=0) / voxel_size_m) * voxel_size_m
    maximum = np.ceil(np.max(combined, axis=0) / voxel_size_m) * voxel_size_m
    shape = np.ceil((maximum - origin) / voxel_size_m).astype(int) + 1
    if np.prod(shape) > 50_000_000:
        raise RuntimeError(f"voxel grid is unexpectedly large: {shape.tolist()}")
    arrays: dict[str, np.ndarray] = {}
    closing_report = {}
    eta_values = []
    for side in SIDES:
        indices = np.floor((points[side] - origin) / voxel_size_m + 1e-9).astype(int)
        indices = np.clip(indices, 0, shape - 1)
        key = tuple(indices.T)
        count = np.zeros(shape, dtype=np.uint32)
        eta_max = np.full(shape, np.nan, dtype=np.float32)
        np.add.at(count, key, 1)
        eta_accumulator = np.full(shape, -np.inf, dtype=np.float32)
        np.maximum.at(eta_accumulator, key, eta[side])
        occupied = count > 0
        eta_max[occupied] = eta_accumulator[occupied]
        visual = binary_closing(occupied, structure=np.ones((3, 3, 3), dtype=bool))
        closing_report[side] = {
            "added_voxels": int(np.count_nonzero(visual & ~occupied)),
            "deleted_voxels": int(np.count_nonzero(occupied & ~visual)),
        }
        arrays[f"count_{side}"] = count
        arrays[f"eta_{side}_max"] = eta_max
        arrays[f"occ_{side}_raw"] = occupied
        arrays[f"occ_{side}_visual"] = visual
        eta_values.append(eta[side])
    arrays["occ_overlap_raw"] = arrays["occ_left_raw"] & arrays["occ_right_raw"]
    arrays["occ_overlap_visual"] = (
        arrays["occ_left_visual"] & arrays["occ_right_visual"])
    finite_eta = np.concatenate(eta_values)
    robust_low, robust_high = np.percentile(finite_eta, (1, 99))
    denominator = max(float(robust_high - robust_low), 1e-12)
    normalized = {}
    for side in SIDES:
        values = arrays[f"eta_{side}_max"]
        normalized[side] = np.clip((values - robust_low) / denominator, 0.0, 1.0)
    overlap_score = np.full(shape, np.nan, dtype=np.float32)
    overlap = arrays["occ_overlap_raw"]
    overlap_score[overlap] = np.minimum(
        normalized["left"][overlap], normalized["right"][overlap])
    arrays["eta_overlap_dexterity"] = overlap_score
    grid = {
        "origin_m": origin,
        "maximum_m": maximum,
        "shape": shape,
        "voxel_size_m": float(voxel_size_m),
        "closing": closing_report,
        "eta_robust_range": [float(robust_low), float(robust_high)],
    }
    return arrays, grid


def _eta_statistics(values: np.ndarray) -> dict[str, float]:
    keys = ("min", "p1", "median", "p95", "p99", "max")
    data = np.asarray([
        np.min(values),
        *np.percentile(values, (1, 50, 95, 99)),
        np.max(values),
    ], dtype=float)
    return dict(zip(keys, data.tolist()))


def compute_workspace(
    *,
    model_path: str | Path,
    robot: str,
    sample_count: int,
    voxel_size_m: float,
    seed: int,
    jobs: int,
    output: str | Path,
    progress: Any = None,
) -> dict[str, Any]:
    import mujoco

    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    inspection = inspect_workspace_model(model_path, robot)
    points: dict[str, np.ndarray] = {}
    eta: dict[str, np.ndarray] = {}
    sampling_stats = {}
    for side_index, side in enumerate(SIDES):
        callback = (
            (lambda done, total, elapsed, current_side=side:
             progress(current_side, done, total, elapsed))
            if progress else None
        )
        points[side], eta[side], sampling_stats[side] = sample_arm(
            model_path, robot, side, sample_count,
            seed + side_index, jobs, callback)
    arrays, grid = _voxel_grid(points, eta, voxel_size_m)
    voxel_volume = voxel_size_m ** 3
    occupancy_stats = {
        "left_volume_m3": float(np.count_nonzero(arrays["occ_left_raw"]) * voxel_volume),
        "right_volume_m3": float(np.count_nonzero(arrays["occ_right_raw"]) * voxel_volume),
        "overlap_volume_m3": float(np.count_nonzero(arrays["occ_overlap_raw"]) * voxel_volume),
    }
    occupancy_stats["overlap_over_left"] = (
        occupancy_stats["overlap_volume_m3"] / occupancy_stats["left_volume_m3"])
    occupancy_stats["overlap_over_right"] = (
        occupancy_stats["overlap_volume_m3"] / occupancy_stats["right_volume_m3"])
    metadata = {
        "format_version": 1,
        "robot": robot,
        "mujoco_version": mujoco.__version__,
        "workspace_definition": (
            "Per-arm seven-joint collision-free TCP position workspace. The opposite "
            "arm remains at qpos0; opposite-arm contacts are counted but do not reject "
            "basic-workspace samples."
        ),
        "overlap_definition": (
            "Spatial intersection of the left and right individual collision-free "
            "workspaces; not a simultaneous 14-DoF bimanual workspace."
        ),
        "manipulability": {
            "name": "weighted Jacobian isotropy ratio",
            "formula": "sigma_min([Jp; 0.30*Jr]) / sigma_max([Jp; 0.30*Jr])",
            "orientation_weight_m_per_rad": ORIENTATION_WEIGHT_M_PER_RAD,
        },
        "sampling": {
            "method": "scipy.stats.qmc.Sobol",
            "scramble": True,
            "seed_left": seed,
            "seed_right": seed + 1,
            "samples_per_arm": int(sample_count),
            "jobs": int(jobs),
            "stats": sampling_stats,
        },
        "collision_filtering": {
            "pipeline": "mujoco.mj_fwdPosition",
            "baseline_penetration_tolerance_m": BASELINE_PENETRATION_TOLERANCE_M,
            "rejection_classes": ["sampled-arm self", "torso/base", "static environment"],
            "opposite_arm_rejects_basic_workspace": False,
        },
        "grid": {
            "origin_m": grid["origin_m"].tolist(),
            "maximum_m": grid["maximum_m"].tolist(),
            "shape": grid["shape"].tolist(),
            "voxel_size_m": grid["voxel_size_m"],
            "closing": grid["closing"],
            "eta_robust_range": grid["eta_robust_range"],
        },
        "occupancy": occupancy_stats,
        "eta_statistics": {side: _eta_statistics(eta[side]) for side in SIDES},
        "model_audit": inspection,
    }
    np.savez_compressed(
        output,
        grid_origin_m=grid["origin_m"],
        grid_maximum_m=grid["maximum_m"],
        voxel_size_m=np.asarray(voxel_size_m),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        **arrays,
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# collision-free workspace envelope\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces + 1:
            stream.write(f"f {face[0]} {face[1]} {face[2]}\n")


def _taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 8,
    forward: float = 0.48,
    backward: float = -0.50,
) -> np.ndarray:
    """Smooth a marching-cubes surface without the shrinkage of plain Laplacian smoothing."""
    from scipy.sparse import coo_matrix

    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1.0)
    result = np.asarray(vertices, dtype=np.float64).copy()
    for _ in range(iterations):
        for factor in (forward, backward):
            neighbor_mean = adjacency @ result / degree[:, None]
            result += factor * (neighbor_mean - result)
    return result


def build_workspace_envelopes(
    workspace_npz: str | Path,
    base_xml: str | Path,
    reference_body: str,
    output_dir: str | Path,
) -> dict[str, Path]:
    import mujoco
    from skimage.measure import marching_cubes

    workspace_npz = Path(workspace_npz).expanduser().resolve()
    base_xml = Path(base_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(workspace_npz, allow_pickle=False) as cache:
        origin = cache["grid_origin_m"]
        voxel_size = float(cache["voxel_size_m"])
        occupancies = {
            "left": cache["occ_left_visual"],
            "right": cache["occ_right_visual"],
            "overlap": cache["occ_overlap_visual"],
        }
    meshes: dict[str, Path] = {}
    for name, occupancy in occupancies.items():
        if np.count_nonzero(occupancy) < 4:
            continue
        padded = np.pad(occupancy.astype(np.float32), 1)
        vertices, faces, _, _ = marching_cubes(
            padded, level=0.5, spacing=(voxel_size,) * 3)
        vertices += origin - voxel_size
        vertices = _taubin_smooth(vertices, faces.astype(np.int32))
        path = output_dir / f"workspace_{name}.obj"
        _write_obj(path, vertices, faces.astype(np.int32))
        meshes[name] = path

    tree = ET.parse(base_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody = root.find("worldbody")
        root.insert(list(root).index(worldbody), asset)
    body = next(
        (item for item in root.findall(".//body") if item.get("name") == reference_body),
        None,
    )
    if body is None:
        raise RuntimeError(f"workspace reference body {reference_body!r} is missing")
    colors = {
        "left": "0.02 0.42 1.00 0.15",
        "right": "1.00 0.35 0.02 0.15",
        "overlap": "0.64 0.12 0.95 0.23",
    }
    for name, path in meshes.items():
        mesh_name = f"workspace_{name}_mesh"
        ET.SubElement(asset, "mesh", name=mesh_name, file=str(path))
        ET.SubElement(
            body,
            "geom",
            name=f"workspace_{name}_envelope",
            type="mesh",
            mesh=mesh_name,
            rgba=colors[name],
            contype="0",
            conaffinity="0",
            group="3",
        )
    envelope_xml = output_dir / "workspace_envelope.xml"
    ET.indent(tree, space="  ")
    tree.write(envelope_xml, encoding="utf-8", xml_declaration=True)
    mujoco.MjModel.from_xml_path(str(envelope_xml))
    meshes["xml"] = envelope_xml
    return meshes
