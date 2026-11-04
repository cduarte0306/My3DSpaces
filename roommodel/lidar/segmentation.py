"""Plane segmentation of a room mesh: floor, ceiling, walls, other horizontals.

Works on a gravity-aligned mesh (ARKit / Record3D worlds have +y up), so the
label rules use the y axis as "up". Original vertex colours are never touched;
segment colours go on a copy so the scan can be toggled back.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
import open3d as o3d

UP = np.array([0.0, 1.0, 0.0])

FLOOR, CEILING, WALL, HORIZONTAL, OTHER = "floor", "ceiling", "wall", "horizontal", "other"

LABEL_COLORS = {
    FLOOR:      (0.55, 0.38, 0.22),
    CEILING:    (0.90, 0.90, 0.85),
    HORIZONTAL: (0.95, 0.75, 0.20),   # tables, beds, shelves
    OTHER:      (0.60, 0.40, 0.70),   # slanted planes
}
WALL_COLORS = [(0.25, 0.55, 0.85), (0.30, 0.75, 0.55), (0.85, 0.45, 0.45),
               (0.40, 0.80, 0.85), (0.75, 0.65, 0.35), (0.55, 0.55, 0.90)]
UNLABELLED = (0.35, 0.35, 0.35)


@dataclass
class Plane:
    id: int
    kind: str
    normal: list          # unit, pointing out of the surface (towards the camera side)
    offset: float         # n . x + offset = 0
    n_vertices: int
    area_m2: float        # bounding area of the inlier points on the plane
    height: float         # mean y of the inliers (metres, world frame)


def _plane_area(points, normal):
    """Area of the inliers' bounding rectangle in the plane's own 2D basis."""
    a = np.cross(normal, UP if abs(normal @ UP) < 0.9 else [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    uv = np.stack([points @ a, points @ b], axis=1)
    ext = uv.max(0) - uv.min(0)
    return float(ext[0] * ext[1])


def _find_planes(pts, nrm, pool, accept, dist, min_n, max_planes, normal_cos,
                 ransac_iters, sample, rng, tries=6):
    """Greedy RANSAC over the vertices in `pool` (index array).

    `accept(normal)` rejects planes of the wrong orientation (RANSAC happily
    returns diagonal slices through a room), and a rejected attempt does not end
    the search -- it just draws a new sample, up to `tries` times in a row.
    Returns a list of (normal, offset, inlier_indices).
    """
    found, free, misses = [], pool, 0
    while len(found) < max_planes and len(free) >= min_n and misses < tries:
        pick = free if len(free) <= sample else rng.choice(free, sample, replace=False)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[pick]))
        (a, b, c, d), _ = pcd.segment_plane(dist, 3, ransac_iters)
        n = np.array([a, b, c]); k = np.linalg.norm(n); n /= k; d /= k
        if not accept(n):
            misses += 1
            continue
        idx = free[(np.abs(pts[free] @ n + d) < dist) & (np.abs(nrm[free] @ n) > normal_cos)]
        if len(idx) < min_n:
            misses += 1
            continue

        # refit on every inlier for a tighter plane, then recompute membership
        centroid = pts[idx].mean(0)
        n = np.linalg.svd(pts[idx] - centroid, full_matrices=False)[2][-1]
        d = -float(n @ centroid)
        idx = free[(np.abs(pts[free] @ n + d) < dist) & (np.abs(nrm[free] @ n) > normal_cos)]
        if len(idx) < min_n or not accept(n):
            misses += 1
            continue

        found.append((n, d, idx))
        misses = 0
        free = np.setdiff1d(free, idx, assume_unique=True)
    return found


def segment_planes(mesh, dist=0.03, min_frac=0.01, max_planes=10,
                   normal_cos=0.85, ransac_iters=2000, sample=300_000, seed=0):
    """Find the dominant planes of `mesh` and label every vertex.

    Horizontals (floor, ceiling, table tops) and verticals (walls) are searched
    separately using the vertex normals against gravity, because a single pass
    keeps latching onto diagonal planes that cut through the room.

    dist       RANSAC inlier distance in metres (3 cm).
    min_frac   a plane must hold at least this fraction of all vertices.
    normal_cos an inlier's own normal must agree with the plane's (cos of angle),
               so a table top is not glued onto the wall it touches.
    sample     RANSAC runs on at most this many points (speed); the plane is then
               applied to ALL candidate vertices.
    Returns (planes, labels) where labels[i] is a plane id or -1.
    """
    rng = np.random.default_rng(seed)
    mesh.compute_vertex_normals()
    pts = np.asarray(mesh.vertices)
    nrm = np.asarray(mesh.vertex_normals)
    labels = np.full(len(pts), -1, dtype=np.int32)
    min_n = int(min_frac * len(pts))
    ny = np.abs(nrm @ UP)

    horiz = _find_planes(pts, nrm, np.flatnonzero(ny > 0.9), lambda n: abs(n @ UP) > 0.95,
                         dist, min_n, max_planes, normal_cos, ransac_iters, sample, rng)
    vert = _find_planes(pts, nrm, np.flatnonzero(ny < 0.15), lambda n: abs(n @ UP) < 0.1,
                        dist, min_n, max_planes, normal_cos, ransac_iters, sample, rng)
    raw = horiz + vert

    # floor = lowest big horizontal, ceiling = highest; the rest stay "horizontal"
    heights = [pts[idx, 1].mean() for _, _, idx in raw]
    hi = list(range(len(horiz)))
    floor_i = min(hi, key=lambda i: heights[i]) if hi else None
    ceil_i = max(hi, key=lambda i: heights[i]) if len(hi) > 1 else None

    planes = []
    centre = pts.mean(0)
    for i, (n, d, idx) in enumerate(raw):
        if n @ centre + d < 0:                    # orient the normal towards the room's interior
            n, d = -n, -d
        labels[idx] = i
        if i < len(horiz):
            kind = FLOOR if i == floor_i else CEILING if i == ceil_i else HORIZONTAL
        else:
            kind = WALL
        planes.append(Plane(i, kind, [round(float(x), 5) for x in n], round(float(d), 5),
                            int(len(idx)), round(_plane_area(pts[idx], n), 2),
                            round(float(heights[i]), 3)))
    return planes, labels


def color_by_label(mesh, planes, labels):
    """Return a COPY of `mesh` painted by segment; the original is untouched."""
    out = o3d.geometry.TriangleMesh(mesh)
    cols = np.tile(np.array(UNLABELLED), (len(labels), 1))
    wall_n = 0
    for p in planes:
        if p.kind == WALL:
            col = WALL_COLORS[wall_n % len(WALL_COLORS)]
            wall_n += 1
        else:
            col = LABEL_COLORS[p.kind]
        cols[labels == p.id] = col
    out.vertex_colors = o3d.utility.Vector3dVector(cols)
    return out


def save_segmentation(path_prefix, planes, labels):
    np.save(path_prefix + "labels.npy", labels)
    with open(path_prefix + "planes.json", "w") as f:
        json.dump([asdict(p) for p in planes], f, indent=2)


def summarize(planes, labels):
    total = len(labels)
    lines = [f"{p.kind:<10} #{p.id:<2} {p.n_vertices / total:6.1%} of vertices  "
             f"area {p.area_m2:6.2f} m2  height {p.height:5.2f}" for p in planes]
    lines.append(f"unlabelled {np.mean(labels < 0):6.1%}")
    return "\n".join(lines)
