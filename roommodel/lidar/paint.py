"""Click-to-paint on a segmented room mesh.

A Painter turns a screen pixel into a ray, finds which segmented plane it hits,
and recolours that whole plane. It keeps the scan's own light/shadow variation
(each vertex's brightness relative to its plane's average) so a painted wall
still looks like a lit wall instead of a flat cartoon. No viewer code in here.
"""
import colorsys
import json

import numpy as np
import open3d as o3d

# name -> RGB 0..1; keys 1..8 in the viewer
PALETTE = [
    ("white",       (0.96, 0.96, 0.94)),
    ("warm white",  (0.93, 0.89, 0.80)),
    ("sage",        (0.60, 0.69, 0.57)),
    ("blue-grey",   (0.50, 0.60, 0.70)),
    ("terracotta",  (0.76, 0.42, 0.31)),
    ("navy",        (0.15, 0.22, 0.38)),
    ("charcoal",    (0.22, 0.23, 0.25)),
    ("butter",      (0.95, 0.85, 0.50)),
]


def adjust_color(rgb, dh=0.0, ds=0.0, dv=0.0):
    """Shift hue (wraps), saturation and brightness of an RGB 0..1 colour."""
    h, sat, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb((h + dh) % 1.0, min(max(sat + ds, 0.0), 1.0), min(max(v + dv, 0.0), 1.0))


def to_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)


def pixel_ray(cam, x, y):
    """World-space ray (origin, unit direction) through pixel (x, y).

    `cam` is the PinholeCameraParameters from ViewControl.convert_to_pinhole_camera_parameters():
    extrinsic is world->camera, OpenCV-style (camera looks down +z).
    """
    K = np.asarray(cam.intrinsic.intrinsic_matrix)
    E = np.asarray(cam.extrinsic)
    R, t = E[:3, :3], E[:3, 3]
    d_cam = np.array([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], 1.0])
    d = R.T @ d_cam
    return -R.T @ t, d / np.linalg.norm(d)


class Painter:
    def __init__(self, mesh, planes, labels):
        self.mesh, self.planes, self.labels = mesh, planes, np.asarray(labels)
        n = len(self.labels)
        base = np.asarray(mesh.vertex_colors)
        self.base = base.copy() if len(base) == n else np.full((n, 3), 0.7)
        if len(base) != n:                                # mesh had no colours: give it some to edit
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.base)
        self.lum = self.base @ np.array([0.299, 0.587, 0.114])
        self.paint = {}                                   # plane id -> (r, g, b)
        self.history = []                                 # (plane id, previous colour or None)
        self._tris = np.asarray(mesh.triangles)
        self._verts = np.asarray(mesh.vertices)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def pick(self, origin, direction):
        """Plane id under the ray, or None (sky, or geometry that is not on a plane)."""
        ray = o3d.core.Tensor([[*origin, *direction]], dtype=o3d.core.Dtype.Float32)
        hit = self._scene.cast_rays(ray)
        t = float(hit["t_hit"].numpy()[0])
        if not np.isfinite(t):
            return None
        tri = self._tris[int(hit["primitive_ids"].numpy()[0])]
        p = np.asarray(origin) + t * np.asarray(direction)
        # nearest of the triangle's corners that belongs to a plane
        for v in sorted(tri, key=lambda v: np.linalg.norm(self._verts[v] - p)):
            if self.labels[v] >= 0:
                return int(self.labels[v])
        return None

    def _write(self, plane_id):
        idx = np.flatnonzero(self.labels == plane_id)
        colors = np.asarray(self.mesh.vertex_colors)       # view: edits go straight to the mesh
        if plane_id in self.paint:
            rgb = np.array(self.paint[plane_id])
            factor = self.lum[idx] / max(self.lum[idx].mean(), 1e-3)
            colors[idx] = np.clip(rgb[None, :] * np.clip(factor, 0.6, 1.4)[:, None], 0, 1)
        else:
            colors[idx] = self.base[idx]

    def apply(self, plane_id, rgb):
        self.history.append((plane_id, self.paint.get(plane_id)))
        self.paint[plane_id] = tuple(rgb)
        self._write(plane_id)

    def recolor(self, plane_id, rgb):
        """Change an already-painted plane's colour in place (no new undo step)."""
        if plane_id in self.paint:
            self.paint[plane_id] = tuple(rgb)
            self._write(plane_id)

    def undo(self):
        if not self.history:
            return None
        plane_id, prev = self.history.pop()
        if prev is None:
            self.paint.pop(plane_id, None)
        else:
            self.paint[plane_id] = prev
        self._write(plane_id)
        return plane_id

    def reset(self):
        for pid in list(self.paint):
            self.paint.pop(pid)
            self._write(pid)
        self.history.clear()

    def save(self, prefix):
        o3d.io.write_triangle_mesh(prefix + "room_painted.ply", self.mesh)
        with open(prefix + "paint.json", "w") as f:
            json.dump({str(k): [round(float(c), 4) for c in v] for k, v in self.paint.items()}, f, indent=2)
