#!/usr/bin/env python3
"""Create collision-capable derived MuJoCo models without changing source URDFs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tianji_lerobot_replay.frames import load_scene_config  # noqa: E402
from tianji_lerobot_replay.scene import build_scene  # noqa: E402
from tianji_lerobot_replay.workspace_specs import get_workspace_spec  # noqa: E402
from render_action_space_video import _generic_scene  # noqa: E402


ENVIRONMENT_PREFIXES = (
    "table_top", "table_leg", "floor", "umi_calibration_board", "board_support",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("tianji",))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_scene(robot: str, cache: Path) -> tuple[Path, Path]:
    spec = get_workspace_spec(robot)
    config = load_scene_config(spec.scene_config)
    if robot == "tianji":
        scene = build_scene(
            spec.urdf,
            cache / "base_scene",
            config,
            [PROJECT_ROOT / "models/ros_packages"],
        )
        return scene.xml_path, spec.urdf
    urdf = spec.urdf
    _generic_scene(urdf, cache / "base_scene", config)
    return cache / "base_scene/scene.xml", urdf


def _body(root: ET.Element, name: str) -> ET.Element:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    raise RuntimeError(f"derived scene is missing body {name!r}")


def _inject_sites_and_environment_collisions(
    source_xml: Path,
    output_xml: Path,
    robot: str,
) -> None:
    spec = get_workspace_spec(robot)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("derived scene has no worldbody")

    for side in ("left", "right"):
        body = _body(root, spec.ee_bodies[side])
        ET.SubElement(
            body,
            "site",
            name=spec.ee_sites[side],
            pos="0 0 0",
            size="0.003",
            rgba="0 0 0 0",
            group="4",
        )

    added = []
    for geom in list(worldbody.findall("geom")):
        name = geom.get("name") or ""
        if not name.startswith(ENVIRONMENT_PREFIXES):
            continue
        collision = copy.deepcopy(geom)
        collision_name = f"{name}_workspace_collision"
        collision.set("name", collision_name)
        collision.set("contype", "1")
        collision.set("conaffinity", "1")
        collision.set("rgba", "0 0 0 0")
        collision.set("group", "4")
        worldbody.append(collision)
        added.append(collision_name)
    if not added:
        raise RuntimeError("no static environment geoms were duplicated for collision")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def prepare(robot: str, output: Path) -> dict[str, object]:
    import mujoco

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_xml, source_urdf = _base_scene(robot, output)
    derived_xml = output / "workspace_model.xml"
    _inject_sites_and_environment_collisions(base_xml, derived_xml, robot)
    model = mujoco.MjModel.from_xml_path(str(derived_xml))
    spec = get_workspace_spec(robot)
    for site in spec.ee_sites.values():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) < 0:
            raise RuntimeError(f"compiled workspace model is missing site {site!r}")
    manifest = {
        "robot": robot,
        "derived_xml": str(derived_xml),
        "derived_xml_sha256": _sha256(derived_xml),
        "source_urdf": str(source_urdf),
        "source_urdf_sha256": _sha256(source_urdf),
        "original_spec_urdf": str(spec.urdf),
        "original_spec_urdf_sha256": _sha256(spec.urdf),
        "model_counts": {
            "nq": model.nq,
            "nv": model.nv,
            "nbody": model.nbody,
            "ngeom": model.ngeom,
            "nsite": model.nsite,
            "neq": model.neq,
            "ntendon": model.ntendon,
        },
        "ee_sites": spec.ee_sites,
        "reference_body": spec.reference_body,
    }
    (output / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = prepare(args.robot, Path(args.output))
    print(
        f"WORKSPACE_MODEL_READY robot={args.robot} "
        f"xml={manifest['derived_xml']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
