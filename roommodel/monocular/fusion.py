# Step 3 (second half): fuse many posed depth maps into one room mesh.
#
# A single frame gives a wedge of surface. Overlapping frames disagree with
# each other -- especially monocular ones, whose per-frame scale is only
# approximately recovered -- so they cannot simply be stacked as point clouds
# without producing a thick, doubled shell.
#
# A truncated signed distance field solves that by averaging. Each frame votes
# on how far every nearby voxel is from a surface, and the surface is
# extracted where those votes cross zero. Disagreement between frames blurs
# the result rather than duplicating it, which is exactly the failure mode
# monocular depth needs.
#
# This is built on Open3D's tensor API (`VoxelBlockGrid`), not the older
# `pipelines.integration.ScalableTSDFVolume`. That legacy class allocates only
# a single volume unit per integration in Open3D 0.20 -- feeding it perfect
# LiDAR depth for a whole room yields a 77-vertex scrap whose size tracks
# `voxel_length` instead of the scene. `VoxelBlockGrid` is also the API that
# runs on the GPU, which matters once there are hundreds of frames.

from __future__ import annotations

import numpy as np
import open3d as o3d
import open3d.core as o3c

# Depth is handed to Open3D as millimetres in a uint16 image, its native
# convention; `depth_scale` converts back to metres inside the integrator.
_DEPTH_SCALE = 1000.0
_MAX_UINT16_METRES = 65.535


class TSDFFuser:
    """Accumulates posed RGB-D frames into a single coloured mesh."""

    def __init__(
        self,
        voxel_size: float = 0.04,
        depth_trunc: float = 6.0,
        block_resolution: int = 16,
        block_count: int = 60_000,
        device: str | None = None,
    ):
        if depth_trunc > _MAX_UINT16_METRES:
            raise ValueError(
                f"depth_trunc {depth_trunc} m exceeds what millimetre uint16 depth "
                f"can carry ({_MAX_UINT16_METRES} m)"
            )

        if device is None:
            device = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
        self.device = o3c.Device(device)
        self.depth_trunc = depth_trunc

        self._grid = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=block_resolution,
            block_count=block_count,
            device=self.device,
        )

    def integrate(self, frame, depth: np.ndarray, color_bgr: np.ndarray) -> None:
        """Add one frame. `depth` is metric, in `color_bgr`'s resolution."""
        if depth.shape != color_bgr.shape[:2]:
            raise ValueError(
                f"Depth {depth.shape} does not match colour {color_bgr.shape[:2]}"
            )

        height, width = depth.shape
        intrinsics = frame.intrinsics_for(width, height)

        # 0 means "no measurement" to Open3D, so NaNs and negatives become 0.
        clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        clean = np.clip(clean, 0.0, self.depth_trunc)
        depth_mm = (clean * _DEPTH_SCALE).astype(np.uint16)

        depth_image = o3d.t.geometry.Image(o3c.Tensor(depth_mm)).to(self.device)
        color_image = o3d.t.geometry.Image(
            o3c.Tensor(np.ascontiguousarray(color_bgr[:, :, ::-1]))
        ).to(self.device)

        intrinsic = o3c.Tensor(intrinsics, o3c.float64)
        extrinsic = o3c.Tensor(frame.world_to_cam, o3c.float64)

        blocks = self._grid.compute_unique_block_coordinates(
            depth_image, intrinsic, extrinsic, _DEPTH_SCALE, self.depth_trunc
        )
        self._grid.integrate(
            blocks, depth_image, color_image,
            intrinsic, intrinsic, extrinsic,
            _DEPTH_SCALE, self.depth_trunc,
        )

    def extract_mesh(self, largest_component_only: bool = True) -> o3d.geometry.TriangleMesh:
        """The fused surface. Colour comes along from the integrated frames."""
        mesh = self._grid.extract_triangle_mesh().to_legacy()
        mesh.compute_vertex_normals()

        # Monocular depth blunders leave small blobs floating off the room;
        # keeping only the largest connected piece removes most of them.
        if largest_component_only and len(mesh.triangles):
            labels, counts, _ = mesh.cluster_connected_triangles()
            labels, counts = np.asarray(labels), np.asarray(counts)
            if len(counts) > 1:
                mesh.remove_triangles_by_mask(labels != int(counts.argmax()))
                mesh.remove_unreferenced_vertices()
        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        return self._grid.extract_point_cloud().to_legacy()
