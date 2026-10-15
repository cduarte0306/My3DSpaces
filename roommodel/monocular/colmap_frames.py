# Present a COLMAP reconstruction as posed frames.
#
# This is the adapter that lets the monocular pipeline run on a plain video.
# `scanner_app.ScannerAppScan` reads poses a capture app recorded; this reads
# poses structure-from-motion solved. Both satisfy `core.frames.PosedFrame`, so
# everything downstream -- depth, anchors, fusion, colouring -- is unchanged.
#
# One difference matters and cannot be papered over: **COLMAP's world is only
# correct up to scale.** It recovers the room's shape, never its size, because
# no image tells you how big anything is. Reconstructions built from these
# frames therefore need an external anchor before any measurement means
# anything -- `monocular.gravity_scale` (the floor and an assumed camera
# height) or a user-entered wall length (`monocular.scale_solver`).
#
# Conventions line up conveniently: COLMAP already uses the OpenCV camera basis
# (+x right, +y down, +z forward) and stores `cam_from_world`, so the only work
# is inverting it.
#
# Caveat: `calibration_matrix()` is the pinhole part of the camera only. COLMAP
# defaults to SIMPLE_RADIAL and does estimate a distortion coefficient, which
# everything downstream ignores because it assumes a pinhole projection. For a
# phone's main lens that is a small error, but it is not zero -- undistorting
# the frames up front is the fix if straight edges come out bowed.

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pycolmap

from roommodel.core.image_io import load_image_bgr


@dataclass(frozen=True)
class ColmapFrame:
    """One registered image, with the pose and intrinsics COLMAP solved."""

    index: int
    name: str
    image_path: Path
    intrinsics: np.ndarray  # 3x3, at `size`
    size: tuple[int, int]  # (width, height) the intrinsics refer to
    cam_to_world: np.ndarray  # 4x4, OpenCV basis, arbitrary scale

    @property
    def position(self) -> np.ndarray:
        return self.cam_to_world[:3, 3]

    @property
    def world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam_to_world)

    def intrinsics_for(self, width: int, height: int) -> np.ndarray:
        native_width, native_height = self.size
        scale = np.array([width / native_width, height / native_height, 1.0])
        return self.intrinsics * scale[:, None]

    def load_image(self) -> np.ndarray:
        return load_image_bgr(self.image_path)


class ColmapScan:
    """The registered images of a COLMAP reconstruction, as posed frames."""

    def __init__(self, sparse_dir: str | Path, image_dir: str | Path):
        self.sparse_dir = _resolve_model_dir(Path(sparse_dir))
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"No such image directory: {self.image_dir}")

        self.reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))

        frames: list[ColmapFrame] = []
        for image in self.reconstruction.images.values():
            if not image.has_pose:
                continue  # never registered; it has no pose to offer
            camera = self.reconstruction.cameras[image.camera_id]

            # `cam_from_world` is a method on pycolmap's Image, not a property.
            cam_from_world = np.eye(4)
            cam_from_world[:3, :4] = np.asarray(image.cam_from_world().matrix())

            frames.append(
                ColmapFrame(
                    index=int(image.image_id),
                    name=image.name,
                    image_path=self.image_dir / image.name,
                    intrinsics=np.asarray(camera.calibration_matrix(), dtype=float),
                    size=(int(camera.width), int(camera.height)),
                    cam_to_world=np.linalg.inv(cam_from_world),
                )
            )

        # Capture order, so that "neighbouring frames" means what it should for
        # baseline selection and for interpolating scale along the trajectory.
        self.frames = sorted(frames, key=lambda f: f.name)
        if not self.frames:
            raise ValueError(
                f"No registered images in {self.sparse_dir}. COLMAP solved nothing "
                "usable -- usually too little overlap, texture or parallax."
            )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> ColmapFrame:
        return self.frames[index]

    @cached_property
    def num_input_images(self) -> int:
        """How many images COLMAP was given, counted from the image directory.

        The reconstruction itself only carries images it managed to register,
        so it cannot answer this -- asking it always returns 100%.
        """
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sum(
            1 for path in self.image_dir.iterdir()
            if path.suffix.lower() in suffixes
        )

    @cached_property
    def registration_rate(self) -> float:
        """Registered images as a fraction of those COLMAP was given.

        A low rate means frames failed to link up -- too little overlap,
        texture or parallax -- and the reconstruction covers only part of the
        room even if what it does cover looks fine.
        """
        total = self.num_input_images
        return len(self.frames) / total if total else 0.0

    def trajectory(self) -> np.ndarray:
        return np.array([frame.position for frame in self.frames])

    @cached_property
    def points3D(self) -> np.ndarray:
        """The sparse triangulated cloud, (N, 3) in reconstruction units.

        This is what makes scale solvable before anything is fused: it is a
        reconstruction of the room already, just a thin one, so the floor can
        be located in it and the whole model rescaled up front.
        """
        return np.array(
            [point.xyz for point in self.reconstruction.points3D.values()],
            dtype=float,
        )

    def rescale(self, factor: float) -> None:
        """Multiply the whole reconstruction by `factor`, in place.

        A similarity transform about the origin: camera centres move, camera
        orientations do not, and the sparse cloud follows. Used to convert
        COLMAP's arbitrary units into metres once a real length is known.
        """
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"Scale factor must be positive and finite, got {factor}")

        scaled = []
        for frame in self.frames:
            cam_to_world = frame.cam_to_world.copy()
            cam_to_world[:3, 3] *= factor
            scaled.append(replace(frame, cam_to_world=cam_to_world))
        self.frames = scaled

        if "points3D" in self.__dict__:
            self.__dict__["points3D"] = self.points3D * factor

    # Kept so the display code can treat either scan type alike; COLMAP has no
    # equivalent of the scan app's user-drawn room box.
    @property
    def room_box_points(self) -> None:
        return None

    def room_dimensions(self) -> None:
        return None

    def room_box_edges(self) -> list:
        return []


def _resolve_model_dir(path: Path) -> Path:
    """Accept either a model directory or a parent holding numbered models."""
    if (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file():
        return path

    candidates = sorted(
        child for child in path.glob("*")
        if child.is_dir() and (child / "cameras.bin").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No COLMAP model (cameras.bin) under {path}")

    # Several models means the capture broke into disconnected pieces; the
    # biggest is the best of a bad situation.
    return max(candidates, key=lambda d: (d / "images.bin").stat().st_size)
