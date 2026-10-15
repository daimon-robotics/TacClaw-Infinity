"""MuJoCo overlays for cached collision-free dual-arm workspaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import binary_erosion


SIDES = ("left", "right", "overlap")


def _viridis(values: np.ndarray) -> np.ndarray:
    """Small dependency-free interpolation of the perceptually ordered viridis map."""
    anchors = np.asarray(
        [
            (0.267004, 0.004874, 0.329415),
            (0.229739, 0.322361, 0.545706),
            (0.127568, 0.566949, 0.550556),
            (0.369214, 0.788888, 0.382914),
            (0.993248, 0.906157, 0.143936),
        ],
        dtype=np.float32,
    )
    scaled = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[:, None]
    return anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction


@dataclass(frozen=True)
class WorkspaceShell:
    centers_local_m: np.ndarray
    rgba: np.ndarray


class WorkspaceOverlay:
    """Toggleable Mode-A meshes and Mode-B shell voxels for a passive viewer."""

    MODE_NAMES = ("OFF", "Envelope", "Dexterity heatmap")

    def __init__(
        self,
        model: object,
        cache_path: str | Path,
        *,
        initial_mode: int = 1,
        overlap_only: bool = False,
    ) -> None:
        import mujoco

        self.model = model
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.mode = int(initial_mode) % 3
        self.overlap_only = bool(overlap_only)
        self._warning_keys: set[tuple[int, bool, int]] = set()
        self._scene_states: dict[int, dict[str, object]] = {}
        with np.load(self.cache_path, allow_pickle=False) as cache:
            self.metadata = json.loads(str(cache["metadata_json"]))
            self.voxel_size_m = float(cache["voxel_size_m"])
            origin = np.asarray(cache["grid_origin_m"], dtype=float)
            robust_low, robust_high = self.metadata["grid"]["eta_robust_range"]
            denominator = max(float(robust_high - robust_low), 1e-12)
            shells: dict[str, WorkspaceShell] = {}
            for side in SIDES:
                occupancy = np.asarray(cache[f"occ_{side}_visual"], dtype=bool)
                shell = occupancy & ~binary_erosion(
                    occupancy,
                    structure=np.ones((3, 3, 3), dtype=bool),
                    border_value=0,
                )
                indices = np.argwhere(shell)
                centers = origin + indices * self.voxel_size_m
                if side == "overlap":
                    metric = np.asarray(cache["eta_overlap_dexterity"], dtype=float)[shell]
                    # Visual closing can introduce shell voxels absent from the raw
                    # overlap score.  Use the minimum adjacent-arm normalized score
                    # where available and zero for visualization-only additions.
                    normalized = np.nan_to_num(metric, nan=0.0)
                else:
                    metric = np.asarray(cache[f"eta_{side}_max"], dtype=float)[shell]
                    normalized = np.clip((metric - robust_low) / denominator, 0.0, 1.0)
                    normalized = np.nan_to_num(normalized, nan=0.0)
                rgb = _viridis(normalized)
                alpha = np.full((len(rgb), 1), 0.30, dtype=np.float32)
                shells[side] = WorkspaceShell(centers, np.concatenate((rgb, alpha), axis=1))
        self.shells = shells

        reference_name = self.metadata["model_audit"]["reference_body"]
        self.reference_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, reference_name)
        if self.reference_body_id < 0:
            raise RuntimeError(f"workspace reference body {reference_name!r} is missing")

        self._envelope_geoms: dict[str, int] = {}
        self._envelope_rgba: dict[str, np.ndarray] = {}
        self._envelope_alpha_scale = 1.0
        for side in SIDES:
            name = f"workspace_{side}_envelope"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                self._envelope_geoms[side] = int(geom_id)
                self._envelope_rgba[side] = model.geom_rgba[geom_id].copy()
        self.apply_envelope_visibility()

    @property
    def mode_name(self) -> str:
        return self.MODE_NAMES[self.mode]

    @property
    def scope_name(self) -> str:
        return "overlap only" if self.overlap_only else "left + right + overlap"

    def cycle_mode(self) -> None:
        self.mode = (self.mode + 1) % len(self.MODE_NAMES)
        self.apply_envelope_visibility()

    def toggle_scope(self) -> None:
        self.overlap_only = not self.overlap_only
        self.apply_envelope_visibility()

    def apply_envelope_visibility(self) -> None:
        for side, geom_id in self._envelope_geoms.items():
            visible = self.mode == 1 and (not self.overlap_only or side == "overlap")
            self.model.geom_rgba[geom_id] = self._envelope_rgba[side]
            self.model.geom_rgba[geom_id, 3] *= self._envelope_alpha_scale
            if not visible:
                self.model.geom_rgba[geom_id, 3] = 0.0

    def set_envelope_alpha_scale(self, scale: float) -> None:
        """Scale Mode-A opacity without changing the cached source colors."""
        if not 0.0 <= scale <= 1.0:
            raise ValueError("envelope alpha scale must be in [0, 1]")
        self._envelope_alpha_scale = float(scale)
        self.apply_envelope_visibility()

    def update_user_scene(
        self, scene: object, data: object, *, reset: bool = True
    ) -> dict[str, int]:
        """Draw the Mode-B shell into a user scene or append it to a renderer scene."""
        import mujoco

        if reset:
            scene.ngeom = 0
        start_geom = int(scene.ngeom)
        if self.mode != 2:
            return {"requested": 0, "displayed": 0}
        names = ("overlap",) if self.overlap_only else SIDES
        requested = sum(len(self.shells[name].centers_local_m) for name in names)
        capacity = max(0, int(scene.maxgeom) - int(scene.ngeom))
        counts = self._display_counts(names, capacity)
        rotation = data.xmat[self.reference_body_id].reshape(3, 3)
        translation = data.xpos[self.reference_body_id]
        radius = 0.32 * self.voxel_size_m
        state_key = (names, capacity, start_geom)
        state = self._scene_states.get(id(scene))
        if state is None or state["key"] != state_key:
            centers_parts = []
            rgba_parts = []
            for name in names:
                shell = self.shells[name]
                count = counts[name]
                indices = _spatial_subsample(len(shell.centers_local_m), count)
                centers_parts.append(shell.centers_local_m[indices])
                rgba_parts.append(shell.rgba[indices])
            centers_local = (
                np.concatenate(centers_parts, axis=0)
                if centers_parts else np.empty((0, 3), dtype=float)
            )
            rgba = (
                np.concatenate(rgba_parts, axis=0)
                if rgba_parts else np.empty((0, 4), dtype=float)
            )
            state = {
                "key": state_key,
                "centers_local": centers_local,
                "rgba": rgba,
                "last_rotation": None,
                "last_translation": None,
            }
            self._scene_states[id(scene)] = state
        centers_local = state["centers_local"]
        rgba = state["rgba"]
        displayed = len(centers_local)
        scene.ngeom = start_geom + displayed
        unchanged = (
            state["last_rotation"] is not None
            and np.array_equal(rotation, state["last_rotation"])
            and np.array_equal(translation, state["last_translation"])
        )
        if not unchanged:
            centers_world = centers_local @ rotation.T + translation
            if state["last_rotation"] is None:
                for index, (position, color) in enumerate(zip(centers_world, rgba)):
                    geom = scene.geoms[start_geom + index]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, radius, dtype=float),
                        position,
                        np.eye(3).reshape(-1),
                        color,
                    )
            else:
                # Geometry type, size, material and color are immutable after
                # construction.  A moving reference body only requires the
                # three position floats to be refreshed.
                for index, position in enumerate(centers_world):
                    scene.geoms[start_geom + index].pos[:] = position
            state["last_rotation"] = rotation.copy()
            state["last_translation"] = translation.copy()
        if displayed < requested:
            key = (self.mode, self.overlap_only, capacity)
            if key not in self._warning_keys:
                warnings.warn(
                    "workspace shell exceeds the visualization scene maxgeom; "
                    f"spatially subsampled {requested} -> {displayed} geoms at "
                    f"the original {self.voxel_size_m:.3f}-m voxel size",
                    RuntimeWarning,
                )
                self._warning_keys.add(key)
        return {"requested": requested, "displayed": displayed}

    def _display_counts(self, names: tuple[str, ...], capacity: int) -> dict[str, int]:
        lengths = {name: len(self.shells[name].centers_local_m) for name in names}
        requested = sum(lengths.values())
        if requested <= capacity:
            return lengths
        # Preserve the overlap shell first, then divide the remaining budget
        # proportionally between the individual-arm shells.
        result = {name: 0 for name in names}
        remaining = capacity
        if "overlap" in names:
            result["overlap"] = min(lengths["overlap"], remaining)
            remaining -= result["overlap"]
        peers = [name for name in names if name != "overlap"]
        peer_total = sum(lengths[name] for name in peers)
        for index, name in enumerate(peers):
            count = (
                remaining
                if index == len(peers) - 1
                else int(remaining * lengths[name] / max(peer_total, 1))
            )
            result[name] = min(lengths[name], count)
            remaining -= result[name]
            peer_total -= lengths[name]
        return result


def _spatial_subsample(total: int, count: int) -> np.ndarray:
    """Deterministic even-index subsampling that retains the full shell extent."""
    if count >= total:
        return np.arange(total, dtype=int)
    if count <= 0:
        return np.empty(0, dtype=int)
    return np.linspace(0, total - 1, count, dtype=int)
