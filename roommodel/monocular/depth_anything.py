# Step 3: per-frame depth (Depth Anything V2 / Depth Pro), aligned to sparse SfM points.

import sys
from pathlib import Path

import numpy as np
import torch

from roommodel.core.image_io import load_image_bgr

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEPTH_ANYTHING_V2_DIR = _REPO_ROOT / "third_party" / "depth-anything-v2"
if str(_DEPTH_ANYTHING_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPTH_ANYTHING_V2_DIR))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

_DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "data" / "output" / "checkpoint"

_model_cache: dict[tuple[str, str], DepthAnythingV2] = {}


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(encoder: str, checkpoint_path: str) -> DepthAnythingV2:
    cache_key = (encoder, checkpoint_path)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if encoder not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder '{encoder}', expected one of {list(_MODEL_CONFIGS)}")

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Download one from the "
            "Depth-Anything-V2 README and place it there."
        )

    model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(_get_device()).eval()

    _model_cache[cache_key] = model
    return model


def estimate_depth(
    image: str | Path | np.ndarray,
    encoder: str = "vits",
    checkpoint_path: str | None = None,
    input_size: int = 518,
) -> np.ndarray:
    """Relative inverse depth (disparity) for one image, as float32 (H, W).

    Accepts a path or an already-loaded BGR array. Values are affine-invariant
    -- larger means closer, but there is no unit and no fixed zero, so they
    only compare within a single image. Use
    `depth_estimation.metric_alignment` to convert them to metres.

    Paths are read with the EXIF orientation ignored, so the result lines up
    with intrinsics recorded against the native sensor buffer.
    """
    if checkpoint_path is None:
        checkpoint_path = str(_DEFAULT_CHECKPOINT_DIR / f"depth_anything_v2_{encoder}.pth")

    model = _load_model(encoder, checkpoint_path)

    raw_image = image if isinstance(image, np.ndarray) else load_image_bgr(image)

    with torch.no_grad():
        depth = model.infer_image(raw_image, input_size)

    return depth
