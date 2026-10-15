# Render a metric depth map by ray-casting a known mesh from a known camera.
#
# For a LiDAR scan this gives reference depth in real metres, which is what
# lets an affine-invariant monocular prediction be pinned to metric scale
# (see depth_estimation.metric_alignment) and scored against ground truth.

from __future__ import annotations

import numpy as np
import open3d as o3d


class MeshDepthRenderer:
    """Casts camera rays against a mesh to produce metric z-depth maps.

    Building the ray-tracing acceleration structure costs a few seconds on a
    million-triangle room mesh, so hold on to one instance across frames.
    """

    def __init__(self, mesh: o3d.geometry.TriangleMesh):
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(
        self,
        intrinsics: np.ndarray,
        cam_to_world: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Depth along the optical axis, in metres. NaN where a ray misses.

        `cam_to_world` uses the OpenCV camera basis (+z forward).
        """
        us, vs = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=0)

        # Ray directions in camera space, scaled so their z component is 1.
        dirs_cam = np.linalg.inv(intrinsics) @ pixels
        dirs_world = cam_to_world[:3, :3] @ dirs_cam

        # Open3D expects (and may internally normalise) unit directions, so
        # normalise here and keep the factor: for a direction whose camera-space
        # z is 1, its length is exactly the ratio of ray distance to z-depth.
        lengths = np.linalg.norm(dirs_world, axis=0)
        origin = np.tile(cam_to_world[:3, 3][:, None], (1, dirs_world.shape[1]))
        rays = np.concatenate([origin, dirs_world / lengths], axis=0).T

        hits = self._scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
        distance = hits["t_hit"].numpy()

        depth = distance / lengths
        depth[~np.isfinite(depth)] = np.nan  # misses come back as +inf
        return depth.reshape(height, width)
