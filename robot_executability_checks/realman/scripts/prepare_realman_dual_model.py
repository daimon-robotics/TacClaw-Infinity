#!/usr/bin/env python3
"""Build a dual RM75-B URDF with the complete in-house UMI grippers."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET


def _mesh_visual(
    filename: str,
    origin: str,
    rpy: str,
    color: str,
    scale: str = "0.001 0.001 0.001",
) -> ET.Element:
    visual = ET.Element("visual")
    ET.SubElement(visual, "origin", xyz=origin, rpy=rpy)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=filename, scale=scale)
    material = ET.SubElement(visual, "material", name="")
    ET.SubElement(material, "color", rgba=color)
    return visual


def _mesh_collision(
    filename: str,
    origin: str,
    rpy: str,
    scale: str = "0.001 0.001 0.001",
) -> ET.Element:
    collision = ET.Element("collision")
    ET.SubElement(collision, "origin", xyz=origin, rpy=rpy)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "mesh", filename=filename, scale=scale)
    return collision


def _stand_mesh_uri(
    stand_mesh_dir: Path,
    stem: str,
    asset_uri_prefix: str | None,
) -> str:
    if asset_uri_prefix:
        return f"{asset_uri_prefix}/stand/{stem}.stl"
    return str((stand_mesh_dir / f"{stem}.stl").resolve())


def _prefix_arm(
    source_root: ET.Element,
    side: str,
    package_dir: Path,
    asset_uri_prefix: str | None,
    preserve_collision: bool,
) -> tuple[list[ET.Element], str]:
    prefix = f"{side}_"
    links = [copy.deepcopy(item) for item in source_root.findall("link")]
    joints = [copy.deepcopy(item) for item in source_root.findall("joint")]
    for link in links:
        link.set("name", prefix + str(link.get("name")))
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename") or ""
            marker = "package://RM75-B/"
            if filename.startswith(marker):
                source = package_dir / filename[len(marker):]
                if asset_uri_prefix:
                    mesh.set("filename", f"{asset_uri_prefix}/rm75_b/{source.name}")
                else:
                    mesh.set("filename", str(source.resolve()))
        if not preserve_collision:
            # Collision geometry is irrelevant for the lightweight replay
            # model. Workspace analysis explicitly opts into the vendor
            # collision meshes with --preserve-collision.
            for collision in list(link.findall("collision")):
                link.remove(collision)
        rgba = "0.88 0.90 0.96 1" if side == "left" else "0.78 0.84 0.96 1"
        for material in link.findall("./visual/material"):
            color = material.find("color")
            if color is None:
                color = ET.SubElement(material, "color")
            color.set("rgba", rgba)
    for joint in joints:
        joint.set("name", prefix + str(joint.get("name")))
        joint.find("parent").set("link", prefix + joint.find("parent").get("link"))
        joint.find("child").set("link", prefix + joint.find("child").get("link"))
    return [*links, *joints], prefix + "base_link"


def _copy_gripper(
    source_root: ET.Element,
    side: str,
    package_dir: Path,
    mount_plate_mesh: Path,
    asset_uri_prefix: str | None,
    preserve_collision: bool,
) -> tuple[list[ET.Element], str]:
    """Copy the complete DM25PD03-DA URDF and retain base_link as the TCP."""
    source_links = {
        link.get("name"): link
        for link in source_root.findall("link")
    }
    expected = {"base_link", "main_camera", "binocular_camera"}
    if set(source_links) != expected:
        raise RuntimeError(
            "expected UMI gripper links " + ", ".join(sorted(expected)))

    link_names = {
        "base_link": f"{side}_tcp",
        "main_camera": f"{side}_gripper_main_camera",
        "binocular_camera": f"{side}_gripper_binocular_camera",
    }

    links: list[ET.Element] = []
    for old_name, source_link in source_links.items():
        link = copy.deepcopy(source_link)
        link.set("name", link_names[old_name])
        for mesh in link.findall(".//mesh"):
            filename = mesh.get("filename") or ""
            marker = "package://ugripper_urdf/meshes/"
            if not filename.startswith(marker):
                raise RuntimeError(f"unexpected gripper mesh URI: {filename}")
            basename = Path(filename[len(marker):]).name
            # MuJoCo identifies imported meshes by basename. RM75-B also owns
            # a base_link.STL, so give the UMI mesh a stable unique basename.
            if basename.casefold() == "base_link.stl":
                basename = "umi_gripper_base_link.STL"
            if asset_uri_prefix:
                mesh.set("filename", f"{asset_uri_prefix}/umi_gripper/{basename}")
            else:
                mesh.set("filename", str((package_dir / basename).resolve()))
        if not preserve_collision:
            for collision in list(link.findall("collision")):
                link.remove(collision)
        links.append(link)

    joints: list[ET.Element] = []
    for source_joint in source_root.findall("joint"):
        parent = source_joint.find("parent")
        child = source_joint.find("child")
        if parent is None or child is None:
            continue
        old_parent = parent.get("link") or ""
        old_child = child.get("link") or ""
        if old_parent not in link_names or old_child not in link_names:
            continue
        joint = copy.deepcopy(source_joint)
        joint.set("name", f"{side}_gripper_{joint.get('name')}")
        joint.find("parent").set("link", link_names[old_parent])
        joint.find("child").set("link", link_names[old_child])
        joints.append(joint)

    mount = ET.Element("link", name=f"{side}_gripper_mount_plate")
    if asset_uri_prefix:
        mount_uri = f"{asset_uri_prefix}/dm25pd04_eg/Link2.STL"
    else:
        mount_uri = str(mount_plate_mesh.resolve())
    mount.append(_mesh_visual(
        mount_uri, "0 0 0", "0 0 0", "0.72 0.74 0.78 1", "1 1 1"))
    if preserve_collision:
        mount.append(_mesh_collision(
            mount_uri, "0 0 0", "0 0 0", "1 1 1"))
    adapter = ET.Element(
        "joint", name=f"{side}_gripper_adapter_joint", type="fixed")
    ET.SubElement(adapter, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(adapter, "parent", link=f"{side}_link_7")
    ET.SubElement(adapter, "child", link=f"{side}_gripper_mount_plate")

    # The DM25 mount plate-to-TCP transform is the inverse of the vendor
    # base_link -> Link2 fixed joint. UMI base_link and robot gripper base_link
    # use the same TCP-axis convention confirmed for the dataset.
    tcp_joint = ET.Element(
        "joint", name=f"{side}_gripper_tcp_joint", type="fixed")
    ET.SubElement(tcp_joint, "origin", xyz="0.010346 0 0.074711", rpy="0 0 0")
    ET.SubElement(tcp_joint, "parent", link=f"{side}_gripper_mount_plate")
    ET.SubElement(tcp_joint, "child", link=f"{side}_tcp")
    return [mount, *links, *joints, adapter, tcp_joint], f"{side}_tcp"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-urdf", required=True)
    parser.add_argument("--gripper-urdf", required=True)
    parser.add_argument("--gripper-package-dir", required=True)
    parser.add_argument("--gripper-mount-plate-mesh", required=True)
    parser.add_argument("--stand-mesh-dir", required=True)
    parser.add_argument(
        "--preserve-collision",
        action="store_true",
        help="retain vendor collision meshes and add stand/mount collisions",
    )
    parser.add_argument(
        "--asset-uri-prefix",
        help="optional package URI prefix, e.g. package://dual_rm75_b_description/meshes",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    official = Path(args.official_urdf).expanduser().resolve()
    source_root = ET.parse(official).getroot()
    package_dir = official.parents[1]
    gripper_root = ET.parse(Path(args.gripper_urdf).expanduser().resolve()).getroot()
    gripper_package_dir = Path(args.gripper_package_dir).expanduser().resolve()
    gripper_mount_plate_mesh = Path(
        args.gripper_mount_plate_mesh).expanduser().resolve()
    stand_mesh_dir = Path(args.stand_mesh_dir).expanduser().resolve()
    robot = ET.Element("robot", name="dual_rm75_b_visualization")
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco, "compiler", balanceinertia="true", discardvisual="false",
        fusestatic="false", strippath="false")

    stand = ET.SubElement(robot, "link", name="dual_base")
    # DM-TR01 stand assembled around the existing, evaluated arm mounting
    # frame. STEP coordinates are millimetres. The long 950-mm member is the
    # vertical mast. The original 450-mm cross beam is shortened to 100 mm by
    # scaling only its local length axis; its cross-section remains unchanged.
    stand_parts = (
        ("DM-TR01-01-01A", "-0.06 0 -1.2235", "0 0 0", "0.24 0.27 0.31 1", "0.001 0.001 0.001"),
        ("DM-TR01-01-02A", "-0.06 0 -1.2135", "0 -1.57079632679 0", "0.52 0.55 0.59 1", "0.001 0.001 0.001"),
        ("DM-TR01-01-04A", "-0.06 0 -0.2535", "0 -1.57079632679 0", "0.34 0.37 0.41 1", "0.001 0.001 0.001"),
        ("DM-TR01-01-03A", "-0.06 -0.050 -0.0895", "-1.57079632679 0 0", "0.58 0.61 0.65 1", "0.001 0.001 0.000222222222"),
    )
    for stem, origin, rpy, color, scale in stand_parts:
        mesh_uri = _stand_mesh_uri(
            stand_mesh_dir, stem, args.asset_uri_prefix)
        stand.append(_mesh_visual(
            mesh_uri,
            origin,
            rpy,
            color,
            scale,
        ))
        if args.preserve_collision:
            stand.append(_mesh_collision(mesh_uri, origin, rpy, scale))

    # Mount both base_link interface planes directly onto the two ends of the
    # 100-mm cross beam. The mirrored +/-90-degree rolls point each arm outward
    # while keeping the base mounting faces flush with the beam end planes.
    for side, x, y, z, rpy in (
        ("left", -0.06, 0.050, -0.0895, "-1.57079632679 0 0"),
        ("right", -0.06, -0.050, -0.0895, "-1.57079632679 0 3.14159265359"),
    ):
        nodes, base_link = _prefix_arm(
            source_root, side, package_dir, args.asset_uri_prefix,
            args.preserve_collision)
        robot.extend(nodes)
        mount = ET.SubElement(robot, "joint", name=f"{side}_mount_joint", type="fixed")
        ET.SubElement(
            mount, "origin", xyz=f"{x:.6f} {y:.6f} {z:.6f}",
            rpy=rpy)
        ET.SubElement(mount, "parent", link="dual_base")
        ET.SubElement(mount, "child", link=base_link)

        gripper_nodes, _ = _copy_gripper(
            gripper_root,
            side,
            gripper_package_dir,
            gripper_mount_plate_mesh,
            args.asset_uri_prefix,
            args.preserve_collision,
        )
        robot.extend(gripper_nodes)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
