# The interface every posed frame must satisfy, whatever produced it.
#
# This is the seam between capture and reconstruction. The LiDAR loader
# (roommodel.lidar.scanner_app) satisfies it from ARKit metadata; a COLMAP
# loader will satisfy it from a solved sparse model. Everything downstream --
# back-projection, ray-casting, colouring, fusion -- is written against this
# protocol and works unchanged for either source.
#
# Conventions the implementation owes its callers:
#   * `cam_to_world` uses the OpenCV camera basis (+x right, +y down, +z
#     forward) and maps camera coordinates into a metric world frame.
#   * `intrinsics_for` returns intrinsics valid at the requested resolution,
#     so callers may freely resize images.
#   * `load_image` returns BGR uint8 at the frame's native resolution.

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PosedFrame(Protocol):
    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates, shape (3,)."""

    @property
    def cam_to_world(self) -> np.ndarray:
        """4x4, OpenCV camera basis."""

    @property
    def world_to_cam(self) -> np.ndarray:
        """4x4, the inverse of `cam_to_world`."""

    def intrinsics_for(self, width: int, height: int) -> np.ndarray:
        """3x3 pinhole intrinsics at the given resolution."""

    def load_image(self) -> np.ndarray:
        """BGR uint8."""
