# Colour a mesh by projecting posed RGB frames onto it.
#
# The scan app ships a texture atlas, but a video-only capture will not: all
# it has is geometry plus frames with known poses. This module covers that
# case -- it is the same job a texture-baking step would do, resolved per
# vertex instead of per texel, which keeps it dependency-free and good enough
# to look at while the geometry is what is being debugged.
#
# For each frame, every vertex is projected into the image and kept only if it
# is genuinely seen by that camera:
#
#   * in front of the camera and inside the image bounds,
#   * facing it -- a vertex whose normal points away is seen edge-on at best,
#   * not occluded -- checked against a ray-cast depth map of the mesh, so a
#     vertex behind the far wall does not get painted with the wall's colour.
#
# Surviving observations are averaged with a weight of cos(view angle) / z^2,
# which favours frames that saw the surface head-on and up close.

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix

from roommodel.core.raycast import MeshDepthRenderer


@dataclass(frozen=True)
class ColorizeStats:
    num_vertices: int
    num_colored: int
    num_filled: int  # gained from neighbour fill rather than a direct view
    num_frames: int
    mean_observations: float

    @property
    def coverage(self) -> float:
        return self.num_colored / max(1, self.num_vertices)


def colorize_mesh_from_frames(
    mesh: o3d.geometry.TriangleMesh,
    frames,
    image_scale: float = 0.5,
    depth_scale: float = 0.25,
    occlusion_tolerance: float = 0.05,
    min_view_cos: float = 0.15,
    min_depth: float = 0.1,
    max_depth: float = 12.0,
    fill_iterations: int = 4,
    fallback_color: tuple[float, float, float] = (0.6, 0.6, 0.6),
    progress: bool = True,
) -> tuple[o3d.geometry.TriangleMesh, ColorizeStats]:
    """Paint per-vertex colours onto `mesh` from an iterable of posed `Frame`s.

    `image_scale` shrinks each frame before sampling and `depth_scale` sets the
    resolution of the occlusion depth map -- both trade sharpness for speed,
    and the defaults keep a 144-frame room scan to well under a minute.

    Vertices no frame saw are then filled from their coloured neighbours over
    `fill_iterations` passes; without this they show as bright speckle against
    the photographic colours around them. Anything still unseen after that
    keeps `fallback_color`.

    Returns the mesh (coloured in place) and coverage statistics.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("No frames given to colorize from")

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    accumulated = np.zeros((len(vertices), 3))
    total_weight = np.zeros(len(vertices))
    observations = np.zeros(len(vertices), dtype=np.int32)

    renderer = MeshDepthRenderer(mesh)

    for position, frame in enumerate(frames):
        image = frame.load_image()
        if image_scale != 1.0:
            image = cv2.resize(image, None, fx=image_scale, fy=image_scale,
                               interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        intrinsics = frame.intrinsics_for(width, height)

        world_to_cam = frame.world_to_cam
        points_cam = vertices @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        z = points_cam[:, 2]

        candidate = (z > min_depth) & (z < max_depth)
        if not candidate.any():
            continue

        # Project, then keep what lands inside the image.
        us = np.full(len(vertices), -1.0)
        vs = np.full(len(vertices), -1.0)
        us[candidate] = (
            intrinsics[0, 0] * points_cam[candidate, 0] / z[candidate] + intrinsics[0, 2]
        )
        vs[candidate] = (
            intrinsics[1, 1] * points_cam[candidate, 1] / z[candidate] + intrinsics[1, 2]
        )
        candidate &= (us >= 0) & (us < width - 1) & (vs >= 0) & (vs < height - 1)
        if not candidate.any():
            continue

        # Drop back-facing surfaces: the normal has to point back at the camera.
        to_camera = frame.position - vertices
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-9)
        view_cos = np.einsum("ij,ij->i", normals, to_camera)
        candidate &= view_cos > min_view_cos
        if not candidate.any():
            continue

        # Occlusion: compare each vertex's depth against the first surface the
        # camera actually sees along that pixel. A coarse depth map is enough,
        # and testing `z < depth + tol` (rather than a symmetric band) avoids
        # rejecting valid vertices where a low-res sample straddles an edge.
        depth_width = max(1, int(round(width * depth_scale)))
        depth_height = max(1, int(round(height * depth_scale)))
        depth_map = renderer.render(
            frame.intrinsics_for(depth_width, depth_height),
            frame.cam_to_world,
            depth_width,
            depth_height,
        )
        du = np.clip((us[candidate] * depth_scale).astype(int), 0, depth_width - 1)
        dv = np.clip((vs[candidate] * depth_scale).astype(int), 0, depth_height - 1)
        surface = depth_map[dv, du]
        # A ray that hit nothing (NaN) means no occluder was found.
        unoccluded = ~np.isfinite(surface) | (z[candidate] < surface + occlusion_tolerance)

        index = np.flatnonzero(candidate)[unoccluded]
        if index.size == 0:
            continue

        colors = image[vs[index].astype(int), us[index].astype(int)][:, ::-1] / 255.0
        weight = view_cos[index] / np.square(z[index])

        accumulated[index] += colors * weight[:, None]
        total_weight[index] += weight
        observations[index] += 1

        if progress and (position % 10 == 0 or position == len(frames) - 1):
            seen = int((total_weight > 0).sum())
            print(f"  frame {position + 1}/{len(frames)}: "
                  f"{seen / len(vertices):.1%} of vertices coloured", end="\r")

    if progress:
        print()

    colored = total_weight > 0
    vertex_colors = np.tile(fallback_color, (len(vertices), 1)).astype(float)
    vertex_colors[colored] = accumulated[colored] / total_weight[colored, None]

    num_directly_colored = int(colored.sum())
    if fill_iterations > 0 and not colored.all():
        vertex_colors, colored = _fill_from_neighbours(
            np.asarray(mesh.triangles), vertex_colors, colored, fill_iterations
        )

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vertex_colors, 0.0, 1.0))

    stats = ColorizeStats(
        num_vertices=len(vertices),
        num_colored=int(colored.sum()),
        num_filled=int(colored.sum()) - num_directly_colored,
        num_frames=len(frames),
        mean_observations=float(observations[observations > 0].mean())
        if (observations > 0).any()
        else 0.0,
    )
    return mesh, stats


def _fill_from_neighbours(
    triangles: np.ndarray,
    colors: np.ndarray,
    colored: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Spread colour into unseen vertices from their coloured neighbours.

    One pass reaches vertices one edge away from a coloured one, so a few
    passes close the small holes left by grazing views and occlusion, without
    inventing colour for whole unobserved regions (those simply stay put).
    """
    num_vertices = len(colors)
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])  # adjacency must be symmetric
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(num_vertices, num_vertices),
    ).tocsr()

    for _ in range(iterations):
        missing = ~colored
        if not missing.any():
            break
        neighbour_count = adjacency @ colored.astype(float)
        neighbour_sum = adjacency @ (colors * colored[:, None])

        fillable = missing & (neighbour_count > 0)
        if not fillable.any():
            break
        colors[fillable] = neighbour_sum[fillable] / neighbour_count[fillable, None]
        colored = colored | fillable

    return colors, colored
