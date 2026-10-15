# Step 3 (first half): lift a depth map into a world-space coloured point cloud.

from __future__ import annotations

import numpy as np
import open3d as o3d


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
    color_bgr: np.ndarray | None = None,
    stride: int = 1,
    min_depth: float = 0.05,
    max_depth: float = 20.0,
) -> o3d.geometry.PointCloud:
    """Back-project a metric depth map through the camera into world space.

    `depth` is distance along the optical axis in metres (NaN/non-positive
    entries are dropped), `intrinsics` matches its resolution, and
    `cam_to_world` uses the OpenCV camera basis (+x right, +y down, +z
    forward). `color_bgr` is an optional matching BGR image.
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    if color_bgr is not None and color_bgr.shape[:2] != depth.shape:
        raise ValueError(
            f"Colour image {color_bgr.shape[:2]} does not match depth {depth.shape}"
        )

    # Subsampling is on pixels, so the intrinsics must be shifted/scaled to
    # match the decimated grid rather than the original one.
    if stride > 1:
        depth = depth[::stride, ::stride]
        if color_bgr is not None:
            color_bgr = color_bgr[::stride, ::stride]
        intrinsics = intrinsics.copy()
        intrinsics[:2] /= stride

    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))

    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    if not valid.any():
        return o3d.geometry.PointCloud()

    z = depth[valid]
    x = (us[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (vs[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1]

    points_cam = np.stack([x, y, z], axis=1)
    points_world = points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_world)
    if color_bgr is not None:
        cloud.colors = o3d.utility.Vector3dVector(color_bgr[valid][:, ::-1] / 255.0)
    return cloud
