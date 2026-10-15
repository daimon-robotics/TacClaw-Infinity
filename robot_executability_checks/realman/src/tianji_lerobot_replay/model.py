"""Prepare the supplied Tianji URDF for MuJoCo without mutating vendor files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Iterable
import xml.etree.ElementTree as ET

from .errors import ReplayError


def prepare_urdf(
    source: str | Path,
    destination: str | Path,
    package_roots: Iterable[str | Path] = (),
) -> Path:
    """Write a MuJoCo-compatible copy of *source* and resolve its mesh assets.

    The generated copy restores the vendor's fixed q=0 gripper CAD reference
    and renames only duplicated mesh basenames.  Both changes are essential
    for the current combined Tianji URDF, but never alter the original files.
    """
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise ReplayError(f"找不到天机 URDF：{source_path}")
    if source_path.suffix.lower() == ".xacro":
        raise ReplayError("MuJoCo 不能直接读取 xacro；请先展开成 URDF。")
    try:
        tree = ET.parse(source_path)
    except (ET.ParseError, OSError) as exc:
        raise ReplayError(f"URDF XML 解析失败：{source_path}: {exc}") from exc
    root = tree.getroot()
    if root.tag.rsplit("}", 1)[-1] != "robot":
        raise ReplayError(f"URDF 顶层必须是 <robot>，实际为 <{root.tag}>。")

    _restore_vendor_gripper_cad_reference(root)
    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        mujoco = root.find("mujoco")
        if mujoco is None:
            mujoco = ET.Element("mujoco")
            root.insert(0, mujoco)
        compiler = ET.SubElement(mujoco, "compiler")
    compiler.set("balanceinertia", "true")
    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")

    roots = [Path(item).expanduser().resolve() for item in package_roots]
    roots.extend([source_path.parent, *source_path.parents])
    resolved_meshes: list[tuple[ET.Element, Path]] = []
    unresolved: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename") or mesh.get("url")
        if not filename:
            continue
        resolved = _resolve_asset(filename, source_path.parent, roots)
        if resolved is None:
            unresolved.append(filename)
        else:
            resolved_meshes.append((mesh, resolved))
    if unresolved:
        refs = "\n  - ".join(sorted(set(unresolved)))
        raise ReplayError(
            "以下 URDF mesh 路径无法解析：\n  - " + refs
            + "\n请用 --package-root 指向含有对应 package.xml 的目录。"
        )

    # MuJoCo identifies imported meshes by basename.  The stand and gripper
    # both contain base_link.STL; give only colliding basenames unique copied
    # filenames or the viewer can show the stand mesh as a gripper.
    paths_by_basename: dict[str, set[Path]] = {}
    for _, path in resolved_meshes:
        paths_by_basename.setdefault(path.name.casefold(), set()).add(path)
    duplicate_paths = {
        path
        for paths in paths_by_basename.values()
        if len(paths) > 1
        for path in paths
    }
    copied: dict[Path, Path] = {}
    if duplicate_paths:
        asset_dir = destination_path.parent / f"{destination_path.stem}_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(duplicate_paths):
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
            target = asset_dir / f"{path.stem}__{digest}{path.suffix}"
            shutil.copy2(path, target)
            copied[path] = target.resolve()
    for mesh, path in resolved_meshes:
        mesh.set("filename", str(copied.get(path, path)))
        mesh.attrib.pop("url", None)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination_path, encoding="utf-8", xml_declaration=True)
    return destination_path


def _restore_vendor_gripper_cad_reference(root: ET.Element) -> None:
    """Fix the merged URDF's synthetic finger transforms in the output copy."""
    names = (
        "left_endeffector_joint",
        "left_endeffector_mimic_joint",
        "right_endeffector_joint",
        "right_endeffector_mimic_joint",
    )
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    found = [name for name in names if name in by_name]
    if not found:
        return
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ReplayError("夹爪关节定义不完整：" + ", ".join(missing))
    for name in names:
        joint = by_name[name]
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if origin is None or axis is None or limit is None:
            raise ReplayError(f"夹爪关节 {name!r} 缺少 origin、axis 或 limit。")
        origin.set("rpy", "0 0 0")
        axis.set("xyz", "-1 0 0")
        # MuJoCo treats [0,0] from URDF as unlimited.  Keep a true but tiny
        # bounded interval so the GUI cannot accidentally drive the fingers.
        limit.set("lower", "0")
        limit.set("upper", "1e-9")
        mimic = joint.find("mimic")
        if mimic is not None:
            joint.remove(mimic)


def _resolve_asset(filename: str, urdf_dir: Path, roots: list[Path]) -> Path | None:
    value = filename[7:] if filename.startswith("file://") else filename
    if value.startswith("package://"):
        relative = Path(value[len("package://") :])
        if not relative.parts:
            return None
        package_name = relative.parts[0]
        for root in roots:
            candidates = [root / relative]
            if root.name == package_name:
                candidates.append(root.joinpath(*relative.parts[1:]))
            package_directory = _find_ros_package(root, package_name)
            if package_directory is not None:
                candidates.append(package_directory.joinpath(*relative.parts[1:]))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = urdf_dir / path
    return path.resolve() if path.is_file() else None


def _find_ros_package(root: Path, package_name: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [root / "package.xml", *root.glob("*/package.xml")]
    for package_xml in candidates:
        if not package_xml.is_file():
            continue
        try:
            package_root = ET.parse(package_xml).getroot()
            name = package_root.findtext("name", default="").strip()
        except (ET.ParseError, OSError):
            continue
        if name == package_name:
            return package_xml.parent.resolve()
    return None
