#!/usr/bin/env python3
"""Colour a room mesh from the scan's posed RGB frames, and show it in Open3D.

The scan app ships its own texture atlas, but a video-only capture will not --
it will have geometry plus frames with known poses, and nothing else. This
script covers that case: it projects every frame onto the mesh and averages
what each vertex was seen as, weighting views that saw a surface head-on and
up close. Pass `--source texture` to display the app's atlas instead, as a
reference for how close the projected version gets.

Examples:
    python3 scripts/colorize_room_mesh.py data/lidar-data/2026_09_27_18_46_10
    python3 scripts/colorize_room_mesh.py data/lidar-data/2026_09_27_18_46_10 --every 4 --save-mesh out/room.ply
    python3 scripts/colorize_room_mesh.py data/lidar-data/2026_09_27_18_46_10 --source texture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d as o3d  # noqa: E402

from roommodel.lidar.scanner_app import ScannerAppScan  # noqa: E402
from roommodel.core.vertex_coloring import colorize_mesh_from_frames  # noqa: E402
from roommodel.core import o3d_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scan_dir", help="A 3D Scanner App export directory")
    parser.add_argument(
        "--source",
        choices=("frames", "texture"),
        default="frames",
        help="Colour by projecting the captured frames (default), or just load "
        "the app's own texture atlas",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Use every Nth frame -- higher is faster, with more gaps (default: 1)",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=0.5,
        help="Downscale frames before sampling colour (default: 0.5)",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Leave vertices no frame saw grey instead of filling from neighbours",
    )
    parser.add_argument(
        "--show-box", action="store_true", help="Draw the scan's room bounding box"
    )
    parser.add_argument("--save-mesh", type=Path, help="Write the coloured mesh here")
    parser.add_argument("--no-show", action="store_true", help="Skip the Open3D window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan = ScannerAppScan(args.scan_dir)
    print(f"Scan: {scan.dir}  ({len(scan)} frames)")

    if args.source == "texture":
        print("Loading the app's textured mesh ...")
        mesh = scan.load_mesh(textured=True)
        print(f"  {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    else:
        mesh = scan.load_mesh()
        frames = scan.frames[:: max(1, args.every)]
        print(f"Projecting {len(frames)} frames onto "
              f"{len(mesh.vertices):,} vertices ...")
        mesh, stats = colorize_mesh_from_frames(
            mesh,
            frames,
            image_scale=args.image_scale,
            fill_iterations=0 if args.no_fill else 4,
        )
        print(f"  coloured {stats.num_colored:,}/{stats.num_vertices:,} "
              f"({stats.coverage:.1%}), {stats.num_filled:,} filled from neighbours")
        print(f"  {stats.mean_observations:.1f} views per vertex on average")

    if args.save_mesh:
        args.save_mesh.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(args.save_mesh), mesh)
        print(f"Wrote {args.save_mesh}")

    if args.no_show:
        return

    geometries = [mesh]
    if args.show_box and scan.room_box_points is not None:
        geometries.append(
            o3d_scene.box_wireframe(scan.room_box_points, scan.room_box_edges())
        )
    # Shading is left off (see o3d_scene.show): these colours already carry the
    # room's real lighting, and shading them again blows the bright ones out.
    o3d_scene.show(geometries, title=f"{scan.dir.name} - coloured room")


if __name__ == "__main__":
    main()
