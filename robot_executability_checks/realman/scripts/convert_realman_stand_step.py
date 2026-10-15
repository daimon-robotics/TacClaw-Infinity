#!/usr/bin/env python3
"""Tessellate the supplied RealMan stand STEP parts into MuJoCo-ready STL."""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mesh-size-mm", type=float, default=8.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(input_dir.glob("*.STEP"))
    if not sources:
        raise SystemExit(f"no STEP files found in {input_dir}")

    for source in sources:
        destination = output_dir / f"{source.stem}.stl"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.Binary", 1)
            gmsh.option.setNumber("Mesh.MeshSizeMin", args.mesh_size_mm / 4.0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", args.mesh_size_mm)
            gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.1)
            gmsh.model.add(source.stem)
            gmsh.model.occ.importShapes(str(source), highestDimOnly=True)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            gmsh.write(str(destination))
        finally:
            gmsh.finalize()
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
