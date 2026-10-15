# Step 4 (first half): find the dominant planes in a reconstruction.
#
# The floor is the useful one to start with. It is the largest flat thing in
# almost any room, it is the reference every other structure is measured
# against, and -- because people hold phones at a fairly predictable height --
# knowing where it is turns out to be enough to recover real-world scale
# without measuring anything (see monocular.gravity_scale).
#
# "Floor" only means something once you know which way is up. With ARKit that
# comes for free (+y). With poses from structure-from-motion there is no
# gravity at all, so it has to be estimated from how the camera was held.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class Plane:
    """A plane as `normal . x + offset = 0`, normal oriented away from it."""

    normal: np.ndarray
    offset: float
    num_inliers: int

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from each point to the plane, positive on the normal's side."""
        return np.asarray(points) @ self.normal + self.offset

    def tilt_degrees(self, up: np.ndarray) -> float:
        """How far the plane's normal leans away from `up`."""
        cosine = float(np.clip(np.dot(self.normal, up), -1.0, 1.0))
        return float(np.degrees(np.arccos(abs(cosine))))


def estimate_up_direction(
    frames,
    points: np.ndarray | None = None,
    planarity_ratio: float = 0.25,
) -> np.ndarray:
    """Which way is up, inferred from the camera trajectory.

    Someone filming a room walks around on a level floor, so the path the
    camera traced is close to planar and horizontal. The direction of least
    variance in that path is therefore gravity, recovered here to within 1
    degree of ARKit's own answer on the sample scan.

    Do not be tempted to use the camera's own up axis instead. That depends on
    how the phone was held *and* on which orientation the sensor wrote: this
    iPhone records landscape while held portrait, which puts world-up along
    the camera's -x axis, and reading it off -y lands 109 degrees out.

    The sign is fixed by the observation that a room extends further below the
    camera than above it -- roughly 1.5m of floor clearance against 1m of
    ceiling. That needs `points` from the reconstruction; without them the
    camera's average tilt is used instead, which is weaker, since it only
    holds while the operator looks slightly downward.

    Raises if the path is too straight to define a plane -- a camera pushed
    along a line, or spun in place, leaves gravity ambiguous.
    """
    positions = np.array([frame.position for frame in frames], dtype=float)
    if len(positions) < 3:
        raise ValueError(f"Need at least 3 camera positions, got {len(positions)}")

    centred = positions - positions.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)

    if eigenvalues[1] < 1e-12 or eigenvalues[0] / eigenvalues[1] > planarity_ratio:
        raise ValueError(
            "The camera path is too close to a straight line to tell which way "
            "is up. Capture a path with some spread, or pass `up` explicitly."
        )
    axis = eigenvectors[:, 0]
    axis /= np.linalg.norm(axis)

    if points is not None and len(points):
        heights = np.asarray(points, dtype=float) @ axis
        camera_height = float(np.median(positions @ axis))
        below = camera_height - float(np.percentile(heights, 1))
        above = float(np.percentile(heights, 99)) - camera_height
        return axis if below > above else -axis

    # Fallback: room scanning tends to tilt the camera slightly downward.
    optical = np.array([frame.cam_to_world[:3, 2] for frame in frames]).mean(axis=0)
    return axis if float(optical @ axis) < 0 else -axis


def segment_planes(
    points: np.ndarray,
    max_planes: int = 6,
    distance_threshold: float = 0.04,
    min_inliers: int = 500,
    num_iterations: int = 1000,
) -> list[Plane]:
    """Peel off dominant planes one at a time with RANSAC, largest first."""
    remaining = o3d.geometry.PointCloud()
    remaining.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))

    planes: list[Plane] = []
    for _ in range(max_planes):
        if len(remaining.points) < max(min_inliers, 3):
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_inliers:
            break

        a, b, c, d = model
        normal = np.array([a, b, c], dtype=float)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            break
        planes.append(Plane(normal / length, float(d) / length, len(inliers)))

        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def find_floor_plane(
    points: np.ndarray,
    up: np.ndarray,
    reference_points: np.ndarray | None = None,
    max_tilt_degrees: float = 15.0,
    **segment_kwargs,
) -> Plane:
    """The lowest near-horizontal plane -- the floor.

    `reference_points` (the camera positions, normally) orients the result so
    that signed distances come out positive on the side the cameras are on,
    and is what distinguishes the floor from the ceiling: both are horizontal,
    but the floor is the one below everything.
    """
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    points = np.asarray(points, dtype=float)
    candidates = [
        plane for plane in segment_planes(points, **segment_kwargs)
        if plane.tilt_degrees(up) <= max_tilt_degrees
    ]
    if not candidates:
        raise ValueError(
            "No near-horizontal plane found, so the floor cannot be located. "
            "Either the reconstruction is too noisy, or the estimated up "
            "direction is wrong."
        )

    # Orient every candidate to face along `up`, then keep the lowest.
    oriented = [
        p if np.dot(p.normal, up) > 0 else Plane(-p.normal, -p.offset, p.num_inliers)
        for p in candidates
    ]
    # With the normal pointing up, a plane's height along `up` is -offset.
    floor = min(oriented, key=lambda p: -p.offset)

    if reference_points is not None and len(reference_points):
        if np.median(floor.signed_distance(reference_points)) < 0:
            floor = Plane(-floor.normal, -floor.offset, floor.num_inliers)
    return floor


def fit_planes(point_cloud: "o3d.geometry.PointCloud") -> list[Plane]:
    """Convenience wrapper: dominant planes of an Open3D point cloud."""
    return segment_planes(np.asarray(point_cloud.points))
