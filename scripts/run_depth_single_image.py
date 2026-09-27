#!/usr/bin/env python3
"""Milestone 1: run Depth Anything V2 on a single room photo."""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
from open3d import visualization as vis


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roommodel.depth_estimation.depth_anything import estimate_depth

def to3DSurfaceMap(depth, stride: int = 4, relief: float = 0.3):
    # Height-map mesh: pixel (x, y) as ground-plane position, depth as height.
    # This is a stylized relief, not a perspective-correct reconstruction --
    # we don't have camera intrinsics for this photo, so there's no real
    # back-projection yet (see notes.md). Triangles are built directly from
    # the known grid connectivity rather than via Poisson reconstruction,
    # which is for unorganized point clouds and is both slower and only
    # approximate on data that's already a regular grid.
    depth = depth[::stride, ::stride]
    h, w = depth.shape

    # Raw depth (~0-8) is tiny next to the plane's pixel extent (thousands),
    # so the relief is invisible unless rescaled -- normalize to [0, 1] then
    # scale by a fraction of the plane size ("vertical exaggeration").
    normalized = (depth - depth.min()) / (depth.max() - depth.min())
    normalized = 1 - normalized
    zz = normalized * relief * max(h, w)

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    verts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3).astype(np.float64)

    idx = np.arange(h * w).reshape(h, w)
    top_left, top_right = idx[:-1, :-1], idx[:-1, 1:]
    bottom_left, bottom_right = idx[1:, :-1], idx[1:, 1:]
    tris_a = np.stack((top_left, bottom_left, top_right), axis=-1).reshape(-1, 3)
    tris_b = np.stack((top_right, bottom_left, bottom_right), axis=-1).reshape(-1, 3)
    tris = np.concatenate((tris_a, tris_b), axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()
    return mesh

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a room photo")
    args = parser.parse_args()

    depth = estimate_depth(args.image)
    # print(depth)
    # plt.imshow(depth, cmap="viridis")   # or "Spectral_r", "inferno", "magma"...
    # plt.colorbar(label="inverse depth")
    # plt.axis("off")
    # plt.savefig("depth_heatmap.png", bbox_inches="tight")
    # plt.show()  # if you have a display; on WSL2 this may need an X server
    cloud = to3DSurfaceMap(depth, stride=2)
    print(cloud)
    vis.draw_geometries([cloud])

if __name__ == "__main__":
    main()
