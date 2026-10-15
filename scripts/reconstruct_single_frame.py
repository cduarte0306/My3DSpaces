#!/usr/bin/env python3
"""Reconstruct one frame of a 3D Scanner App room scan and show it in Open3D.

This is the single-frame base case of the room pipeline. For one captured
frame it:

  1. reads the frame's ARKit intrinsics and camera pose,
  2. predicts relative (affine-invariant) depth from the RGB image,
  3. solves that prediction's unknown scale and shift against metric
     reference depth, so it comes out in real metres,
  4. back-projects it into the scan's world frame as a coloured point cloud,
  5. draws it with the camera, the room box and the world axes.

Metric reference depth is obtained by ray-casting the scan's own LiDAR mesh
from the same camera pose. That is a stand-in for what a phone without LiDAR
will use later (sparse SfM points, or user-entered wall measurements), and it
doubles as ground truth: the reported error is how far the monocular guess is
from the LiDAR surface, in metres.

Examples:
    python3 scripts/reconstruct_single_frame.py data/lidar-data/2026_09_27_18_46_10
    python3 scripts/reconstruct_single_frame.py data/lidar-data/2026_09_27_18_46_10 --frame 60 --show-mesh
    python3 scripts/reconstruct_single_frame.py data/lidar-data/2026_09_27_18_46_10 --source lidar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d as o3d  # noqa: E402

from roommodel.lidar.scanner_app import ScannerAppScan  # noqa: E402
from roommodel.core.backprojection import depth_to_point_cloud  # noqa: E402
from roommodel.core.raycast import MeshDepthRenderer  # noqa: E402
from roommodel.monocular.depth_anything import estimate_depth  # noqa: E402
from roommodel.monocular.metric_alignment import (  # noqa: E402
    depth_error_stats,
    fit_affine_to_metric,
)
from roommodel.core.vertex_coloring import colorize_mesh_from_frames  # noqa: E402
from roommodel.core import o3d_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scan_dir", help="A 3D Scanner App export directory")
    parser.add_argument(
        "--frame", type=int, default=0, help="Index into the scan's frames (default: 0)"
    )
    parser.add_argument(
        "--source",
        choices=("mono", "lidar"),
        default="mono",
        help="Depth to reconstruct from: the monocular prediction (default), or "
        "the LiDAR mesh directly, which is useful as a reference view",
    )
    parser.add_argument(
        "--encoder", default="vits", help="Depth Anything V2 encoder (default: vits)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Keep every Nth pixel when building the cloud (default: 3)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Discard points beyond this many metres (default: 10)",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Overlay the app's LiDAR mesh to compare against",
    )
    parser.add_argument(
        "--mesh-color",
        choices=("gray", "frames", "texture"),
        default="gray",
        help="How to colour that overlaid mesh: flat grey (default), projected "
        "from the captured frames, or the app's own texture atlas",
    )
    parser.add_argument("--save-ply", type=Path, help="Write the point cloud here")
    parser.add_argument("--no-show", action="store_true", help="Skip the Open3D window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scan = ScannerAppScan(args.scan_dir)
    if not -len(scan) <= args.frame < len(scan):
        raise SystemExit(f"--frame {args.frame} out of range: scan has {len(scan)} frames")

    frame = scan[args.frame]
    image = frame.load_image()
    height, width = image.shape[:2]
    intrinsics = frame.intrinsics_for(width, height)

    print(f"Scan       : {scan.dir}  ({len(scan)} frames)")
    print(f"Frame      : {frame.index}  {frame.image_path.name}  {width}x{height}")
    print(f"Camera at  : {np.round(frame.position, 3)} m (ARKit world, +y up)")
    dimensions = scan.room_dimensions()
    if dimensions is not None:
        print("Room box   : %.2f x %.2f m floor, %.2f m ceiling" % (
            dimensions[0], dimensions[2], dimensions[1]))

    # Reference depth from the scan's own LiDAR mesh, rendered from this exact
    # camera. Needed to scale the monocular prediction, and worth having in
    # either mode so the two can be compared.
    print("\nRay-casting the LiDAR mesh for reference depth ...")
    mesh = scan.load_mesh()
    reference_depth = MeshDepthRenderer(mesh).render(
        intrinsics, frame.cam_to_world, width, height
    )
    hit_rate = float(np.isfinite(reference_depth).mean())
    print(f"  mesh covers {hit_rate:.1%} of the view, "
          f"median depth {np.nanmedian(reference_depth):.2f} m")

    if args.source == "lidar":
        depth = reference_depth
    else:
        print(f"\nPredicting monocular depth (Depth Anything V2 '{args.encoder}') ...")
        disparity = estimate_depth(image, encoder=args.encoder)

        fit = fit_affine_to_metric(disparity, reference_depth)
        print(f"  disparity -> metres: scale {fit.scale:.4f}, shift {fit.shift:+.4f} "
              f"(fit on {fit.num_points:,} pixels)")

        depth = fit.to_metric_depth(disparity)
        stats = depth_error_stats(depth, reference_depth)
        print("  vs LiDAR: median abs error %.3f m, RMSE %.3f m, mean rel error %.1f%%" % (
            stats["median_abs_error_m"], stats["rmse_m"], stats["abs_rel"] * 100))

    cloud = depth_to_point_cloud(
        depth,
        intrinsics,
        frame.cam_to_world,
        color_bgr=image,
        stride=args.stride,
        max_depth=args.max_depth,
    )
    print(f"\nPoint cloud: {len(cloud.points):,} points "
          f"(stride {args.stride}, capped at {args.max_depth} m)")

    if args.save_ply:
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(args.save_ply), cloud)
        print(f"Wrote {args.save_ply}")

    if args.no_show:
        return

    geometries = [
        cloud,
        o3d_scene.camera_frustum(intrinsics, frame.cam_to_world, width, height),
        o3d_scene.trajectory_line(scan.trajectory()),
        o3d_scene.world_axes(),
    ]
    corners = scan.room_box_points
    if corners is not None:
        geometries.append(o3d_scene.box_wireframe(corners, scan.room_box_edges()))
    if args.show_mesh:
        geometries.append(_colored_mesh(scan, mesh, args.mesh_color))

    o3d_scene.show(geometries, title=f"{scan.dir.name} - frame {frame.index}")


def _colored_mesh(scan: ScannerAppScan, mesh, mode: str):
    """The overlay mesh, coloured however `--mesh-color` asked for."""
    if mode == "gray":
        return mesh.paint_uniform_color([0.7, 0.7, 0.7])
    if mode == "texture":
        print("\nLoading the app's textured mesh ...")
        return scan.load_mesh(textured=True)

    print(f"\nProjecting {len(scan)} frames onto the mesh for colour ...")
    mesh, stats = colorize_mesh_from_frames(mesh, scan.frames)
    print(f"  coloured {stats.coverage:.1%} of vertices")
    return mesh


if __name__ == "__main__":
    main()
