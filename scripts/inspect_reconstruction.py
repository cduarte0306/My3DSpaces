#!/usr/bin/env python3
"""Milestone 2: visualize a COLMAP reconstruction's camera trajectory + sparse
points in Rerun, to sanity-check pose recovery (no teleporting cameras, no
collapsed/degenerate reconstruction)."""

import argparse

import numpy as np
import pycolmap
import rerun as rr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse_dir", nargs="?", default="workspace/sparse")
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Write a .rrd recording instead of spawning a viewer here "
        "(use this on WSL2 if the native viewer fails to launch -- transfer "
        "the file to Windows and open it with the Rerun Viewer app instead)",
    )
    args = parser.parse_args()

    recon = pycolmap.Reconstruction(args.sparse_dir)

    rr.init("colmap_reconstruction", spawn=not args.save)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points = np.array([p.xyz for p in recon.points3D.values()])
    colors = np.array([p.color for p in recon.points3D.values()])
    rr.log("world/points", rr.Points3D(points, colors=colors, radii=0.01))

    # Sort by name so the trajectory line follows video/frame order, not
    # registration order (COLMAP registers images in whatever order the
    # incremental pipeline found best, not filename order).
    images = sorted(recon.images.values(), key=lambda img: img.name)
    centers = np.array([img.projection_center() for img in images])
    rr.log("world/trajectory", rr.LineStrips3D([centers]))

    for img in images:
        cam = img.camera
        rr.log(
            f"world/cameras/{img.name}",
            rr.Transform3D(
                translation=img.projection_center(),
                mat3x3=img.cam_from_world().rotation.matrix().T,
            ),
        )
        rr.log(
            f"world/cameras/{img.name}",
            rr.Pinhole(
                focal_length=(cam.focal_length_x, cam.focal_length_y),
                width=cam.width,
                height=cam.height,
            ),
        )

    if args.save:
        rr.save(args.save)
        print(f"Saved recording to {args.save}")

    print(f"{len(images)} registered images, {len(points)} sparse points")
    print("Missing/unregistered frames won't appear here at all --")
    print("compare this count against the total frames in data/frames/.")


if __name__ == "__main__":
    main()
