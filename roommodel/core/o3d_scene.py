# Open3D display helpers: camera frusta, the room box, and world axes.
#
# Most reconstruction bugs are coordinate-frame mistakes, so the point of
# these is to draw the camera and the room box alongside the geometry -- a
# cloud that floats outside the box, or sits behind the camera, says the pose
# convention is wrong far faster than any numeric check.

from __future__ import annotations

import numpy as np
import open3d as o3d

_CORNER_PIXELS = [(0, 0), (1, 0), (1, 1), (0, 1)]  # as fractions of (w, h)


def camera_frustum(
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
    width: int,
    height: int,
    scale: float = 0.4,
    color: tuple[float, float, float] = (1.0, 0.4, 0.0),
) -> o3d.geometry.LineSet:
    """A wireframe pyramid showing where a camera is and which way it looks.

    `scale` is how far down the optical axis, in metres, to place the image
    plane -- purely cosmetic.
    """
    inverse = np.linalg.inv(intrinsics)
    corners_cam = [
        (inverse @ np.array([u * width, v * height, 1.0])) * scale
        for u, v in _CORNER_PIXELS
    ]
    points_cam = np.vstack([np.zeros(3), *corners_cam])
    points_world = points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]

    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return _line_set(points_world, edges, color)


def box_wireframe(
    corners: np.ndarray,
    edges: list[tuple[int, int]],
    color: tuple[float, float, float] = (0.2, 0.7, 1.0),
) -> o3d.geometry.LineSet:
    """Wireframe for an explicitly-cornered box, e.g. the scan's `userOBB`."""
    return _line_set(np.asarray(corners, dtype=float), edges, color)


def trajectory_line(
    positions: np.ndarray,
    color: tuple[float, float, float] = (0.6, 0.6, 0.6),
) -> o3d.geometry.LineSet:
    """The path the camera walked, in capture order."""
    positions = np.asarray(positions, dtype=float)
    edges = [(i, i + 1) for i in range(len(positions) - 1)]
    return _line_set(positions, edges, color)


def world_axes(size: float = 0.5) -> o3d.geometry.TriangleMesh:
    """Origin marker: red = +x, green = +y (up in ARKit), blue = +z."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def show(
    geometries: list,
    title: str = "Room reconstruction",
    lit: bool | None = None,
) -> None:
    """Open a viewer window.

    `lit` controls Open3D's shading. Leave it as None to decide automatically:
    geometry carrying photographic colour is shown unlit, because the default
    shading multiplies its own highlights on top of colour that already has
    the room's real lighting baked in -- bright surfaces blow out to white and
    the result looks far worse than the data actually is. Untextured geometry
    keeps shading, which is the only thing giving it shape.
    """
    geometries = [g for g in geometries if g is not None]
    if lit is None:
        lit = not any(_has_baked_color(g) for g in geometries)

    if lit:
        o3d.visualization.draw_geometries(geometries, window_name=title)
        return

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name=title)
    for geometry in geometries:
        viewer.add_geometry(geometry)
    options = viewer.get_render_option()
    options.light_on = False
    options.mesh_show_back_face = True  # rooms are viewed from inside
    viewer.run()
    viewer.destroy_window()


def _has_baked_color(geometry) -> bool:
    """Does this geometry carry real per-surface colour, as opposed to a tint?

    `paint_uniform_color` also sets vertex colours, so the presence of colours
    is not enough -- a flat-painted mesh still needs shading to read as a
    shape, and would otherwise render as a silhouette. Varying colour is the
    thing that signals photographic content.
    """
    if not isinstance(geometry, o3d.geometry.TriangleMesh):
        return False
    if geometry.has_textures():
        return True
    if not geometry.has_vertex_colors():
        return False

    colors = np.asarray(geometry.vertex_colors)
    sample = colors[:: max(1, len(colors) // 10_000)]
    return float(np.ptp(sample, axis=0).max()) > 1e-6


def _line_set(points, edges, color) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return line_set
