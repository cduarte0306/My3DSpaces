# Loader for "3D Scanner App" (iOS) exports.
#
# An export directory holds, per captured frame:
#   frame_NNNNN.jpg   RGB, stored in the native landscape sensor orientation
#   frame_NNNNN.json  that frame's ARKit intrinsics + camera pose
# plus scan-wide files:
#   export.obj            LiDAR mesh, untextured (geometry only)
#   textured_output.obj   same mesh with a texture atlas (+ .mtl / .jpg)
#   info.json             metadata, including `userOBB` (room box, metres)
#
# Conventions worth spelling out, since getting any of them wrong yields a
# plausible-looking but wrong reconstruction:
#
# * `cameraPoseARFrame` is 16 floats in ROW-MAJOR order -- reshaping to 4x4
#   puts translation in the last column and [0,0,0,1] in the last row, i.e.
#   it is a camera-to-world matrix.
# * It uses the ARKit/OpenGL camera basis: +x right, +y up, +z pointing back
#   out of the screen (the camera looks down -z). Computer-vision code here
#   wants +x right, +y down, +z forward, so we post-multiply by
#   diag(1, -1, -1, 1).
# * ARKit world axes are gravity-aligned with +y up; the origin is wherever
#   the session started. The mesh and `userOBB` live in this same frame, so
#   poses, mesh and box can all be drawn together without extra alignment.
# * `intrinsics` describes the full-resolution landscape buffer, which is why
#   frames must be read with the EXIF orientation ignored (see image_io).

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import open3d as o3d

from roommodel.core.image_io import load_image_bgr, stored_size

# Post-multiply an ARKit camera-to-world matrix by this to get the same pose
# expressed with an OpenCV camera basis (+x right, +y down, +z forward).
ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])

# Vertex pairs forming the 12 edges of `userOBB`, whose 8 points are stored as
# the bottom face (0-3) followed by the matching top face (4-7).
_OBB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


@dataclass(frozen=True)
class Frame:
    """One captured frame: its image plus the camera that took it."""

    index: int
    image_path: Path
    pose_path: Path
    intrinsics: np.ndarray  # 3x3, for the stored (landscape) resolution
    cam_to_world: np.ndarray  # 4x4, OpenCV camera basis, ARKit world frame

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) the intrinsics above are expressed in."""
        return stored_size(self.image_path)

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return self.cam_to_world[:3, 3]

    @property
    def world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam_to_world)

    def load_image(self) -> np.ndarray:
        """BGR uint8 at the resolution the intrinsics describe."""
        return load_image_bgr(self.image_path)

    def intrinsics_for(self, width: int, height: int) -> np.ndarray:
        """Intrinsics rescaled for a resized copy of this frame."""
        native_width, native_height = self.size
        scale = np.array([width / native_width, height / native_height, 1.0])
        return self.intrinsics * scale[:, None]


class ScannerAppScan:
    """A single 3D Scanner App export directory."""

    def __init__(self, scan_dir: str | Path):
        self.dir = Path(scan_dir)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"No such scan directory: {self.dir}")

        # Pair up frames by index; a frame needs both its image and its pose,
        # and exports occasionally end with an image whose JSON never landed.
        self.frames: list[Frame] = []
        for image_path in sorted(self.dir.glob("frame_*.jpg")):
            pose_path = image_path.with_suffix(".json")
            if pose_path.is_file():
                self.frames.append(_load_frame(image_path, pose_path))

        if not self.frames:
            raise FileNotFoundError(
                f"No frame_*.jpg / frame_*.json pairs found in {self.dir}"
            )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Frame:
        return self.frames[index]

    @cached_property
    def info(self) -> dict:
        info_path = self.dir / "info.json"
        if not info_path.is_file():
            return {}
        return json.loads(info_path.read_text())

    @property
    def mesh_path(self) -> Path:
        """The app's LiDAR mesh (untextured -- geometry only, loads fast)."""
        return self.dir / "export.obj"

    @property
    def textured_mesh_path(self) -> Path:
        return self.dir / "textured_output.obj"

    def load_mesh(self, textured: bool = False) -> o3d.geometry.TriangleMesh:
        """Load the app's own LiDAR reconstruction, in the ARKit world frame.

        Useful as a reference: it is what we are trying to reproduce from RGB,
        and it supplies metric depth for scaling monocular predictions.
        """
        path = self.textured_mesh_path if textured else self.mesh_path
        if not path.is_file():
            raise FileNotFoundError(f"No mesh at {path}")

        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=textured)
        if mesh.is_empty():
            raise ValueError(f"Mesh at {path} loaded empty")
        mesh.compute_vertex_normals()
        return mesh

    @property
    def room_box_points(self) -> np.ndarray | None:
        """The user-drawn room bounding box: 8 world-space corners, in metres."""
        points = self.info.get("userOBB", {}).get("points")
        return None if points is None else np.asarray(points, dtype=float)

    def room_dimensions(self) -> tuple[float, float, float] | None:
        """(width, height, depth) of `userOBB` in metres, or None if absent.

        Height is the vertical (world +y) extent; the other two are the
        in-plane edge lengths of the box's bottom face.
        """
        points = self.room_box_points
        if points is None:
            return None
        width = float(np.linalg.norm(points[1] - points[0]))
        depth = float(np.linalg.norm(points[2] - points[1]))
        height = float(np.linalg.norm(points[4] - points[0]))
        return width, height, depth

    def room_box_edges(self) -> list[tuple[int, int]]:
        return list(_OBB_EDGES)

    def trajectory(self) -> np.ndarray:
        """(N, 3) camera centres, in capture order."""
        return np.array([frame.position for frame in self.frames])


def _load_frame(image_path: Path, pose_path: Path) -> Frame:
    data = json.loads(pose_path.read_text())

    intrinsics = np.asarray(data["intrinsics"], dtype=float).reshape(3, 3)
    arkit_pose = np.asarray(data["cameraPoseARFrame"], dtype=float).reshape(4, 4)

    return Frame(
        index=int(data.get("frame_index", int(image_path.stem.split("_")[-1]))),
        image_path=image_path,
        pose_path=pose_path,
        intrinsics=intrinsics,
        cam_to_world=arkit_pose @ ARKIT_TO_OPENCV,
    )
