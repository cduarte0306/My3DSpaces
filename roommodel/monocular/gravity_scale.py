# Recover real-world scale with no measurement, using the floor as a ruler.
#
# Structure-from-motion solves a room's shape but not its size: COLMAP's world
# is correct up to an unknown factor. Something has to supply one real length.
# A user-entered wall measurement is the accurate answer (see scale_solver),
# but it asks the user to go and find a tape measure.
#
# A free alternative: people film rooms while standing, holding a phone at a
# fairly predictable height. Locate the floor, measure how far the camera path
# floats above it in whatever units the reconstruction happens to use, and
# declare that distance to be ~1.5 m.
#
# The floor is found as a low percentile of point height rather than by
# fitting a plane to it. That sounds cruder and is markedly more reliable
# here: a monocular reconstruction's floor is warped, and on the sample scan
# RANSAC measured it 11-16 degrees off level and then refused to call it
# horizontal at all. Only the floor's *height* is needed, and the up direction
# already comes from the camera path to within about a degree, so the plane's
# orientation was never the useful part. (`core.plane_fitting.find_floor_plane`
# still exists for clean geometry, where the plane itself is wanted.)
#
# Accuracy, measured against the sample scan's true 1.494 m camera height:
# 0.2% on the LiDAR mesh, -3.5% on the monocular one. Triangulating sparse
# features, by comparison, lands at 12-20%. Do not read too much into those
# figures -- camera height varies with the person and how they hold the phone,
# and even within this one capture it ranged 1.36-1.59 m. Expect a few percent.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roommodel.core.plane_fitting import estimate_up_direction

# A phone held at chest-to-eye level by a standing adult.
DEFAULT_CAMERA_HEIGHT_M = 1.5

# Low enough to sit on the floor rather than on the furniture, high enough to
# ignore the stray geometry monocular depth leaves below it -- through a
# doorway, or out of a window.
DEFAULT_FLOOR_PERCENTILE = 2.0


@dataclass(frozen=True)
class FloorScale:
    """The factor converting reconstruction units into metres."""

    factor: float
    observed_height: float  # camera above the floor, in reconstruction units
    assumed_height: float  # what we declared that distance to be, in metres
    floor_level: float  # floor height along `up`, in reconstruction units
    up: np.ndarray

    def describe(self) -> str:
        return (
            f"camera sits {self.observed_height:.3f} units above the floor, "
            f"called {self.assumed_height:.2f} m -> scale x{self.factor:.4f}"
        )


def scale_from_camera_height(
    points: np.ndarray,
    frames,
    assumed_height: float = DEFAULT_CAMERA_HEIGHT_M,
    up: np.ndarray | None = None,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    min_points: int = 500,
) -> FloorScale:
    """Solve the reconstruction's unknown scale from the height of the camera.

    `points` is the reconstruction (fused mesh vertices or a dense cloud) and
    `frames` the cameras that built it, both in the same arbitrary units. Pass
    `up` when the capture already knows gravity, as ARKit does; otherwise it is
    inferred from the camera path.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < min_points:
        raise ValueError(
            f"Only {len(points)} reconstruction points -- too few to locate a floor."
        )

    if up is None:
        up = estimate_up_direction(frames, points=points)
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)

    positions = np.array([frame.position for frame in frames], dtype=float)

    floor_level = float(np.percentile(points @ up, floor_percentile))
    observed = float(np.median(positions @ up)) - floor_level
    if observed <= 1e-6:
        raise ValueError(
            "The camera path came out at or below the floor, so the estimated up "
            "direction is probably inverted."
        )

    return FloorScale(
        factor=float(assumed_height / observed),
        observed_height=observed,
        assumed_height=float(assumed_height),
        floor_level=floor_level,
        up=up,
    )
