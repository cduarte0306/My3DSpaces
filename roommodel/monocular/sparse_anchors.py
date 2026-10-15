# Metric depth anchors from feature matches between posed frames.
#
# A monocular depth model returns affine-invariant disparity: the shape of the
# scene without its size. Something metric has to pin it down. With LiDAR that
# is a mesh; without it, the poses themselves are the metric reference -- if
# two cameras a known distance apart both see the same feature, triangulating
# it gives that point's depth in metres.
#
# The output is deliberately sparse and a bit noisy (tens of points per frame,
# ~10-20% depth error each). That is fine, because it is only ever used to fit
# two numbers -- a scale and a shift -- so the errors average down. It is not
# fine as geometry in its own right.
#
# The parallax filter is the part that matters most: two frames close together
# see a point from almost the same angle, and triangulating that is
# numerically hopeless. Hence matching across several baselines and keeping
# only well-separated observations.
#
# Detection settings matter more than they usually do. A lived-in room is dim,
# a hand-held phone motion-blurs, and painted walls carry almost no texture:
# with stock SIFT settings this scan yields ~120 keypoints per frame and ~25
# matches between adjacent views, which is not enough to scale anything.
# Equalising local contrast (CLAHE) and dropping the contrast threshold takes
# the same pair to ~460 matches. Those weaker features are individually less
# reliable, which is fine -- the parallax and reprojection filters below throw
# out the bad ones, and only two numbers are being fitted downstream.

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SparseAnchors:
    """Pixels in a frame whose metric depth is known by triangulation."""

    pixels: np.ndarray  # (N, 2) float, in `image_size` coordinates
    depths: np.ndarray  # (N,) metres along the optical axis
    parallax_deg: np.ndarray  # (N,) angle between the two views of each point
    image_size: tuple[int, int]  # (width, height) the pixels refer to

    def __len__(self) -> int:
        return len(self.depths)

    @staticmethod
    def concatenate(parts: list["SparseAnchors"]) -> "SparseAnchors":
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return SparseAnchors(np.zeros((0, 2)), np.zeros(0), np.zeros(0), (0, 0))
        sizes = {p.image_size for p in parts}
        if len(sizes) != 1:
            raise ValueError(f"Cannot pool anchors found at different sizes: {sizes}")
        return SparseAnchors(
            np.vstack([p.pixels for p in parts]),
            np.concatenate([p.depths for p in parts]),
            np.concatenate([p.parallax_deg for p in parts]),
            parts[0].image_size,
        )


class AnchorTriangulator:
    """Finds metric depth anchors by matching and triangulating posed frames.

    Feature detection is cached per frame, since each frame is matched against
    several partners and detection dominates the cost.
    """

    def __init__(
        self,
        image_scale: float = 0.6,
        num_features: int = 8000,
        ratio_test: float = 0.85,
        contrast_threshold: float = 0.01,
        equalize: bool = True,
        min_parallax_deg: float = 1.5,
        max_reprojection_px: float = 2.0,
        min_depth: float = 0.2,
        max_depth: float = 12.0,
    ):
        self.image_scale = image_scale
        self.ratio_test = ratio_test
        self.min_parallax_deg = min_parallax_deg
        self.max_reprojection_px = max_reprojection_px
        self.min_depth = min_depth
        self.max_depth = max_depth

        self._sift = cv2.SIFT_create(
            nfeatures=num_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=12,
        )
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) if equalize else None
        self._matcher = cv2.BFMatcher()
        self._cache: dict = {}

    def _features(self, frame):
        key = str(getattr(frame, "image_path", id(frame)))
        if key in self._cache:
            return self._cache[key]

        image = frame.load_image()
        if self.image_scale != 1.0:
            image = cv2.resize(image, None, fx=self.image_scale, fy=self.image_scale,
                               interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        points = (
            np.float64([k.pt for k in keypoints]) if keypoints else np.zeros((0, 2))
        )
        entry = (points, descriptors, frame.intrinsics_for(width, height), (width, height))
        self._cache[key] = entry
        return entry

    def between(self, frame_a, frame_b) -> SparseAnchors | None:
        """Anchors expressed in `frame_a`'s image, or None if the pair is unusable."""
        points_a, desc_a, intrinsics_a, _ = self._features(frame_a)
        points_b, desc_b, intrinsics_b, _ = self._features(frame_b)
        if desc_a is None or desc_b is None or len(points_a) < 8 or len(points_b) < 8:
            return None

        matches = self._matcher.knnMatch(desc_a, desc_b, k=2)
        good = [m for m, n in matches if m.distance < self.ratio_test * n.distance]
        if len(good) < 8:
            return None

        pixels_a = points_a[[m.queryIdx for m in good]].T  # (2, N)
        pixels_b = points_b[[m.trainIdx for m in good]].T

        world_a, world_b = frame_a.world_to_cam, frame_b.world_to_cam
        points_world = cv2.triangulatePoints(
            intrinsics_a @ world_a[:3], intrinsics_b @ world_b[:3], pixels_a, pixels_b
        )
        # Points at infinity come back with w ~ 0; they are not usable anchors.
        w = points_world[3]
        finite = np.abs(w) > 1e-9
        if not finite.any():
            return None
        points_world = (points_world[:3, finite] / w[finite]).T
        pixels_a = pixels_a[:, finite]

        in_a = points_world @ world_a[:3, :3].T + world_a[:3, 3]
        in_b = points_world @ world_b[:3, :3].T + world_b[:3, 3]
        depth_a, depth_b = in_a[:, 2], in_b[:, 2]

        projected = (intrinsics_a @ in_a.T)
        projected = projected[:2] / np.where(np.abs(projected[2]) < 1e-9, 1e-9, projected[2])
        reprojection_error = np.linalg.norm(projected - pixels_a, axis=0)

        to_a = points_world - frame_a.position
        to_b = points_world - frame_b.position
        cosine = np.einsum("ij,ij->i", to_a, to_b) / (
            np.linalg.norm(to_a, axis=1) * np.linalg.norm(to_b, axis=1) + 1e-12
        )
        parallax = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        keep = (
            (depth_a > self.min_depth)
            & (depth_b > self.min_depth)
            & (depth_a < self.max_depth)
            & (reprojection_error < self.max_reprojection_px)
            & (parallax > self.min_parallax_deg)
        )
        if not keep.any():
            return None
        _, _, _, size_a = self._features(frame_a)
        return SparseAnchors(
            pixels_a[:, keep].T, depth_a[keep], parallax[keep], size_a
        )

    def for_frame(
        self,
        frames,
        index: int,
        baselines_m=(0.2, 0.4, 0.7, 1.0),
        temporal_window: int = 40,
    ) -> SparseAnchors:
        """Pool anchors for one frame against several differently-spaced partners.

        Short baselines give many matches but shaky depth; long ones give solid
        depth but fewer matches. Pooling both is what keeps a frame from being
        left with a handful of anchors.

        Partners are chosen by how far the camera actually travelled, in metres,
        rather than by how many frames apart they are -- otherwise subsampling
        the capture silently changes every baseline, and with it the quality of
        every triangulation. The search stays inside a temporal window because
        two frames can be a metre apart while pointing at different walls.
        """
        centre = frames[index].position
        low = max(0, index - temporal_window)
        high = min(len(frames), index + temporal_window + 1)
        candidates = [j for j in range(low, high) if j != index]
        if not candidates:
            return SparseAnchors.concatenate([])

        distances = np.array(
            [np.linalg.norm(frames[j].position - centre) for j in candidates]
        )

        chosen: list[int] = []
        for target in baselines_m:
            nearest = candidates[int(np.argmin(np.abs(distances - target)))]
            if nearest not in chosen:
                chosen.append(nearest)

        return SparseAnchors.concatenate(
            [self.between(frames[index], frames[j]) for j in chosen]
        )
