#!/usr/bin/env python3
"""Build a 3D map of a room from RGB frames alone -- no LiDAR, no depth sensor.

This is the non-LiDAR counterpart to `reconstruct_single_frame.py`. Where that
script scaled one frame's monocular depth against the LiDAR mesh, this one
reconstructs the whole room using only images and camera poses:

  1. Depth Anything V2 predicts relative depth for every frame.
  2. Metric scale is recovered by triangulating SIFT matches between frames
     that are far enough apart to give real parallax -- the poses are metric,
     so the triangulated points are too. Frames with too few anchors inherit
     their neighbours' scale instead of trusting a bad fit.
  3. All frames are fused into one coloured mesh with a TSDF volume, which
     averages away the disagreement between per-frame depth maps.

Poses come from the scan's ARKit metadata, which is *not* LiDAR -- ARKit tracks
visual-inertially and reports poses on any modern iPhone. Swapping in COLMAP
poses (roommodel/monocular/colmap_pipeline.py) is the plain-video path, and
needs only a loader exposing the same `PosedFrame` interface.

Pass --compare-lidar to score the result against the scan's LiDAR mesh, which
is the only reason this example scan is useful: it tells you what the
sensor-free path actually costs.

Examples:
    python3 scripts/map_room_monocular.py data/lidar-data/2026_09_27_18_46_10
    python3 scripts/map_room_monocular.py data/lidar-data/2026_09_27_18_46_10 --every 2 --compare-lidar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import open3d as o3d  # noqa: E402

from roommodel.core import o3d_scene  # noqa: E402
from roommodel.lidar.scanner_app import ScannerAppScan  # noqa: E402
from roommodel.monocular.colmap_frames import ColmapScan  # noqa: E402
from roommodel.monocular.depth_anything import estimate_depth  # noqa: E402
from roommodel.monocular.fusion import TSDFFuser  # noqa: E402
from roommodel.monocular.gravity_scale import (  # noqa: E402
    DEFAULT_CAMERA_HEIGHT_M,
    scale_from_camera_height,
)
from roommodel.monocular.metric_alignment import (  # noqa: E402
    fill_unreliable_fits,
    fit_affine_to_sparse,
    reject_inconsistent_fits,
    sample_anchor_disparity,
    solve_affine,
)
from roommodel.monocular.sparse_anchors import AnchorTriangulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("scan_dir", nargs="?",
                        help="A 3D Scanner App export directory (poses already metric)")
    parser.add_argument("--colmap", type=Path, metavar="SPARSE_DIR",
                        help="Use poses solved by COLMAP instead of a scan app. "
                             "Requires --images. These poses are only correct up "
                             "to scale, so --scale-from floor is implied")
    parser.add_argument("--images", type=Path,
                        help="Directory of the images --colmap was run on")
    parser.add_argument("--every", type=int, default=None,
                        help="Use every Nth frame. Defaults to 3 for a scan-app "
                             "capture (hundreds of frames, each an extra chance "
                             "at a bad scale) and 1 for COLMAP, where only the "
                             "registered frames survive and dropping more leaves "
                             "the volume too sparse to fuse")
    parser.add_argument("--image-scale", type=float, default=0.4,
                        help="Depth/fusion resolution as a fraction of native (default: 0.4)")
    parser.add_argument("--anchor-scale", type=float, default=0.6,
                        help="Feature-matching resolution; higher finds more "
                             "keypoints on bare walls (default: 0.6)")
    parser.add_argument("--encoder", default="vits",
                        help="Depth Anything V2 encoder (default: vits)")
    parser.add_argument("--voxel-size", type=float, default=0.04,
                        help="TSDF voxel size in metres (default: 0.04)")
    parser.add_argument("--depth-trunc", type=float, default=6.0,
                        help="Ignore depth beyond this many metres (default: 6)")
    parser.add_argument("--scale-from", choices=("poses", "floor"), default="poses",
                        help="Where absolute size comes from: metric camera poses "
                             "(default, e.g. ARKit), or the floor plane plus an "
                             "assumed camera height -- the option for poses that "
                             "are only correct up to scale, like COLMAP's")
    parser.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA_HEIGHT_M,
                        help=f"Assumed height of the camera above the floor in "
                             f"metres, used by --scale-from floor "
                             f"(default: {DEFAULT_CAMERA_HEIGHT_M})")
    parser.add_argument("--scale-mode", choices=("per-frame", "global"), default="per-frame",
                        help="Solve one scale for the whole capture (default, more "
                             "robust) or one per frame")
    parser.add_argument("--min-anchors", type=int, default=12,
                        help="Fewest triangulated points to trust a frame's scale")
    parser.add_argument("--max-scale-ratio", type=float, default=1.5,
                        help="Reject a frame's scale if it differs from the "
                             "capture median by more than this factor (default: 1.5)")
    parser.add_argument("--max-reproj-px", type=float, default=2.0,
                        help="Reprojection tolerance when triangulating anchors "
                             "(default: 2). Loosening it admits more frames but "
                             "measurably worsens the result -- see notes")
    parser.add_argument("--compare-lidar", action="store_true",
                        help="Score the result against the scan's LiDAR mesh")
    parser.add_argument("--save-mesh", type=Path, help="Write the fused mesh here")
    parser.add_argument("--no-show", action="store_true", help="Skip the Open3D window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.colmap is not None:
        if args.images is None:
            raise SystemExit("--colmap also needs --images")
        scan = ColmapScan(args.colmap, args.images)
        scan_name = Path(args.colmap).name
        print(f"Scan  : {args.colmap} (COLMAP)")
        print(f"        {len(scan)} registered of {scan.num_input_images} images "
              f"({scan.registration_rate:.0%})")
        # Nothing in a set of photographs says how big the room is, so COLMAP's
        # units are arbitrary. Fix that here, before anything downstream applies
        # a metric threshold: the sparse cloud is already a reconstruction, so
        # the floor can be found in it and the whole model rescaled up front.
        # Rescaling afterwards would be too late -- fusion truncates depth in
        # metres and would discard every frame.
        floor_scale = scale_from_camera_height(
            scan.points3D, scan.frames, assumed_height=args.camera_height
        )
        print(f"        up direction {np.round(floor_scale.up, 3)}")
        print(f"        {floor_scale.describe()}")
        scan.rescale(floor_scale.factor)
        args.scale_from = "poses"  # metric now
        if args.every is None:
            args.every = 1
    elif args.scan_dir is not None:
        scan = ScannerAppScan(args.scan_dir)
        scan_name = scan.dir.name
        print(f"Scan  : {scan.dir}")
    else:
        raise SystemExit("Give either a scan directory or --colmap with --images")

    if args.every is None:
        args.every = 3
    frames = scan.frames[:: max(1, args.every)]
    print(f"Frames: {len(frames)} of {len(scan)} (every {args.every}), "
          f"working at {args.image_scale:g}x native\n")

    triangulator = AnchorTriangulator(
        image_scale=args.anchor_scale,
        max_reprojection_px=args.max_reproj_px,
        min_depth=0.2,
        max_depth=args.depth_trunc * 2,
    )

    # Pass 1: relative depth, plus the metric anchors that give it a size.
    print("Pass 1/2  predicting depth and triangulating metric anchors ...")
    started = time.time()
    disparities: list[np.ndarray] = []
    fits: list = []
    anchor_counts: list[int] = []
    pooled_pred: list[np.ndarray] = []
    pooled_target: list[np.ndarray] = []

    for position, frame in enumerate(frames):
        image = _working_image(frame, args.image_scale)
        disparity = estimate_depth(image, encoder=args.encoder)
        anchors = triangulator.for_frame(frames, position)

        disparities.append(disparity.astype(np.float16))
        anchor_counts.append(len(anchors))
        fits.append(fit_affine_to_sparse(disparity, anchors, min_anchors=args.min_anchors))

        pred, target = sample_anchor_disparity(disparity, anchors)
        pooled_pred.append(pred)
        pooled_target.append(target)

        if position % 10 == 0 or position == len(frames) - 1:
            print(f"  {position + 1}/{len(frames)}  "
                  f"{sum(f is not None for f in fits)} frames scaled", end="\r")
    print()

    solved = sum(f is not None for f in fits)
    print(f"  {solved}/{len(frames)} frames solvable individually "
          f"({np.median(anchor_counts):.0f} anchors/frame median)")

    if args.scale_mode == "global":
        # One camera, one room, one depth model: the relationship between
        # predicted disparity and metres barely changes over a capture. Pooling
        # every anchor into a single solve trades a little per-frame accuracy
        # for consistency between frames -- and consistency is what TSDF fusion
        # needs, since frames that disagree blur each other out.
        combined = solve_affine(
            np.concatenate(pooled_pred), np.concatenate(pooled_target),
            min_points=args.min_anchors,
        )
        if combined is None:
            raise SystemExit(
                "Not enough anchors across the whole capture to recover scale."
            )
        print(f"  global fit on {sum(len(p) for p in pooled_pred):,} anchors: "
              f"scale {combined.scale:.4f}, shift {combined.shift:+.4f}")
        fits = [combined] * len(frames)
    else:
        fits = reject_inconsistent_fits(fits, max_scale_ratio=args.max_scale_ratio)
        consistent = sum(f is not None for f in fits)
        print(f"  {solved - consistent} rejected as inconsistent; "
              f"{len(frames) - consistent} filled from neighbours")
        fits = fill_unreliable_fits(fits)
        scales = np.array([f.scale for f in fits])
        print(f"  scale across capture: median {np.median(scales):.4f}, "
              f"spread {scales.min():.4f}-{scales.max():.4f}")
    print(f"  [{time.time() - started:.0f}s]\n")

    # Pass 2: turn each relative map into metres and fuse.
    print(f"Pass 2/2  fusing at {args.voxel_size * 100:.0f}cm voxels ...")
    started = time.time()
    fuser = TSDFFuser(voxel_size=args.voxel_size, depth_trunc=args.depth_trunc)
    for position, frame in enumerate(frames):
        image = _working_image(frame, args.image_scale)
        depth = fits[position].to_metric_depth(disparities[position].astype(np.float32))
        fuser.integrate(frame, depth, image)
        if position % 10 == 0 or position == len(frames) - 1:
            print(f"  {position + 1}/{len(frames)} integrated", end="\r")
    print()

    mesh = fuser.extract_mesh()
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles "
          f"[{time.time() - started:.0f}s]\n")

    if args.scale_from == "floor":
        print("Recovering absolute scale from the floor plane ...")
        floor_scale = scale_from_camera_height(
            np.asarray(mesh.vertices), frames, assumed_height=args.camera_height
        )
        print(f"  up direction {np.round(floor_scale.up, 3)}")
        print(f"  {floor_scale.describe()}")
        # A similarity transform of the whole world, so it scales about the origin
        # that the poses are expressed in.
        mesh.scale(floor_scale.factor, center=np.zeros(3))
        print()

    if args.compare_lidar:
        if isinstance(scan, ColmapScan):
            print("Skipping --compare-lidar: a COLMAP scan has no LiDAR mesh.\n")
        else:
            _compare_to_lidar(mesh, scan)

    if args.save_mesh:
        args.save_mesh.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(args.save_mesh), mesh)
        print(f"Wrote {args.save_mesh}")

    if args.no_show:
        return

    geometries = [mesh]
    if scan.room_box_points is not None:
        geometries.append(
            o3d_scene.box_wireframe(scan.room_box_points, scan.room_box_edges())
        )
    o3d_scene.show(geometries, title=f"{scan_name} - monocular map")


def _working_image(frame, scale: float) -> np.ndarray:
    image = frame.load_image()
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _compare_to_lidar(mesh, scan: ScannerAppScan) -> None:
    """How far is the RGB-only reconstruction from the LiDAR ground truth?"""
    print("Comparing against the LiDAR mesh ...")
    reference = o3d.t.geometry.RaycastingScene()
    reference.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scan.load_mesh()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    distance = reference.compute_distance(o3d.core.Tensor(vertices)).numpy()
    print(f"  distance to LiDAR surface: median {np.median(distance) * 100:.1f} cm, "
          f"p90 {np.percentile(distance, 90) * 100:.1f} cm")
    within = [(np.mean(distance < t), t) for t in (0.05, 0.10, 0.20)]
    print("  " + ",  ".join(f"{frac:.0%} within {t * 100:.0f}cm" for frac, t in within))

    dimensions = scan.room_dimensions()
    if dimensions is not None:
        extent = mesh.get_axis_aligned_bounding_box().get_extent()
        print(f"  room box (LiDAR) {dimensions[0]:.2f} x {dimensions[2]:.2f} x "
              f"{dimensions[1]:.2f} m  vs  fused extent "
              f"{extent[0]:.2f} x {extent[2]:.2f} x {extent[1]:.2f} m")
    print()


if __name__ == "__main__":
    main()
