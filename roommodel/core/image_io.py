# Shared image loading.
#
# iPhone JPEGs carry an EXIF orientation tag (the scan frames use tag 6 =
# "rotate 90 CW to display"). OpenCV applies that rotation by default, so
# `cv2.imread` hands back a 1440x1920 portrait array while the pixels the
# ARKit intrinsics/pose describe are the un-rotated 1920x1440 landscape
# sensor buffer. Back-projecting the rotated array against those intrinsics
# silently produces a garbage point cloud, so everything in this project
# loads frames with the orientation tag ignored.

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_bgr(path: str | Path, ignore_exif_orientation: bool = True) -> np.ndarray:
    """Read an image as BGR uint8, by default in its stored (un-rotated) orientation."""
    flags = cv2.IMREAD_COLOR
    if ignore_exif_orientation:
        flags |= cv2.IMREAD_IGNORE_ORIENTATION

    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def stored_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of the stored pixel buffer, read from the header only.

    Reported without applying EXIF orientation, so it matches `load_image_bgr`
    and therefore the resolution the camera intrinsics were recorded at.
    """
    with Image.open(str(path)) as image:
        return image.size
