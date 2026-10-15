"""Reject replay CSVs whose source episode or scene differs from the input.

This checks numerical consistency, not physical world-frame registration.
Passing does not establish that the source world is an AprilTag frame.
"""

from __future__ import annotations

import numpy as np

from .errors import ReplayError


def validate_render_rows(rows, episode, config, *, dataset=None, summary=None):
    if len(rows) != episode.frame_count:
        raise ReplayError(
            f"Replay CSV has {len(rows)} rows, episode has {episode.frame_count} frames."
        )
    indices = np.asarray([int(row["frame_index"]) for row in rows])
    if not np.array_equal(indices, np.arange(episode.frame_count)):
        raise ReplayError("Replay CSV frame indices are missing, duplicated or reordered.")
    if any(int(row["episode_index"]) != episode.episode_index for row in rows):
        raise ReplayError("Replay CSV belongs to another episode.")
    times = np.asarray([float(row["timestamp_s"]) for row in rows])
    if not np.allclose(times, episode.timestamps_s, atol=1e-5, rtol=0):
        raise ReplayError("Replay CSV timestamps differ from the source episode.")
    expected = config.robot_from_umi_positions(episode.positions_world_m)
    columns = [f"{side}_target_{axis}_m" for side in ("left", "right") for axis in "xyz"]
    try:
        actual = np.asarray([[float(row[key]) for key in columns] for row in rows])
    except (KeyError, ValueError) as exc:
        raise ReplayError("Replay CSV lacks finite left/right TCP targets.") from exc
    actual = actual.reshape(-1, 2, 3)
    # Full-body evaluation stores targets relative to the selected moving
    # base, whereas video targets stay in the fixed scene/world frame.
    # Undo that declared longitudinal translation before comparing.
    if all(row.get("strategy") == "adaptive_whole_body" for row in rows):
        try:
            base = np.asarray([float(row["base_root_y_table_m"]) for row in rows])
        except (KeyError, ValueError) as exc:
            raise ReplayError("Full-body CSV lacks a finite base position.") from exc
        actual = actual + (base - config.robot_root_in_table_m[1])[:, None, None] * config.rotation_robot_from_table[:, 1]
    error = np.linalg.norm(actual - expected, axis=-1)
    if not np.all(np.isfinite(error)) or np.max(error) > 1e-5:
        raise ReplayError(
            "Replay CSV target positions differ from the dataset/scene "
            f"(maximum difference {float(np.max(error)):.6g} m). Re-evaluate first."
        )
    if dataset is not None and summary is not None:
        recorded_hash = summary.get("dataset", {}).get("info_sha256")
        if recorded_hash is not None and recorded_hash != dataset.info_sha256:
            raise ReplayError("Replay summary metadata hash differs from the dataset.")
    return {"frames_checked": len(rows), "max_target_difference_m": float(error.max()),
            "physical_registration_verified": False}
