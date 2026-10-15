# Pin an affine-invariant monocular depth prediction to metric scale.
#
# Depth Anything V2's relative checkpoints output inverse depth (disparity)
# that is only defined up to an unknown scale and shift -- see the notes on
# `estimate_depth`. Recovering metres therefore means solving for that pair:
#
#     disparity_metric  ~=  scale * disparity_pred + shift
#     depth_metric       =  1 / disparity_metric
#
# The fit is done in disparity space (not depth space) because that is the
# space the ambiguity actually lives in, and because it keeps distant pixels,
# whose depth is wildly uncertain, from dominating the least-squares problem.
#
# The reference disparity comes from whatever metric depth is available. Here
# that is the scan's LiDAR mesh; with no LiDAR (the non-Pro iPhone case) the
# same solve works against sparse SfM points or user-measured distances.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineDepthFit:
    scale: float
    shift: float
    num_points: int  # pixels that survived masking and trimming

    def to_metric_depth(
        self,
        predicted_disparity: np.ndarray,
        min_disparity: float = 1e-3,
    ) -> np.ndarray:
        """Convert a relative prediction to metres; NaN where it is degenerate."""
        disparity = self.scale * predicted_disparity + self.shift
        # Disparity at or below zero means "at or beyond infinity" -- an
        # extrapolation artefact of the fit, not a real surface.
        depth = np.full(disparity.shape, np.nan, dtype=float)
        usable = disparity > min_disparity
        depth[usable] = 1.0 / disparity[usable]
        return depth


def fit_affine_to_metric(
    predicted_disparity: np.ndarray,
    reference_depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 15.0,
    trim_quantile: float = 0.8,
    iterations: int = 3,
) -> AffineDepthFit:
    """Least-squares fit of predicted disparity onto reference metric depth.

    `reference_depth` is metric depth in metres, with NaN where unknown. It is
    refit `iterations` times, each time keeping only the best-fitting
    `trim_quantile` fraction of pixels, so that holes in the reference mesh
    and genuine monocular blunders do not drag the solution.
    """
    if predicted_disparity.shape != reference_depth.shape:
        raise ValueError(
            f"Shape mismatch: prediction {predicted_disparity.shape} vs "
            f"reference {reference_depth.shape}"
        )

    valid = (
        np.isfinite(predicted_disparity)
        & np.isfinite(reference_depth)
        & (reference_depth > min_depth)
        & (reference_depth < max_depth)
    )
    if valid.sum() < 100:
        raise ValueError(
            f"Only {int(valid.sum())} pixels have both a prediction and reference "
            "depth -- too few to solve for scale. Check the camera pose "
            "convention and that the mesh covers this view."
        )

    pred = predicted_disparity[valid].astype(float)
    target = 1.0 / reference_depth[valid].astype(float)

    keep = np.ones(pred.size, dtype=bool)
    scale, shift = 1.0, 0.0
    for iteration in range(max(1, iterations)):
        design = np.stack([pred[keep], np.ones(keep.sum())], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, target[keep], rcond=None)

        if iteration == iterations - 1:
            break
        residuals = np.abs(scale * pred + shift - target)
        keep = residuals <= np.quantile(residuals, trim_quantile)

    return AffineDepthFit(float(scale), float(shift), int(keep.sum()))


def depth_error_stats(
    depth: np.ndarray,
    reference_depth: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 15.0,
) -> dict[str, float]:
    """Compare a metric depth map against a metric reference, in metres."""
    valid = (
        np.isfinite(depth)
        & np.isfinite(reference_depth)
        & (reference_depth > min_depth)
        & (reference_depth < max_depth)
    )
    if not valid.any():
        return {"coverage": 0.0}

    error = depth[valid] - reference_depth[valid]
    return {
        "coverage": float(valid.mean()),
        "mean_abs_error_m": float(np.mean(np.abs(error))),
        "median_abs_error_m": float(np.median(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(np.abs(error) / reference_depth[valid])),
    }


def sample_anchor_disparity(predicted_disparity: np.ndarray, anchors):
    """Line up anchors with the disparity map: returns (predicted, target) pairs.

    Anchor pixels are rescaled from `anchors.image_size` onto the disparity
    map's grid, so the depth model and the feature matcher are free to run at
    whatever resolution each works best at.
    """
    height, width = predicted_disparity.shape
    anchor_width, anchor_height = anchors.image_size
    if not anchor_width or not anchor_height or not len(anchors):
        return np.zeros(0), np.zeros(0)

    pixels = anchors.pixels * np.array([width / anchor_width, height / anchor_height])
    us = np.clip(pixels[:, 0].astype(int), 0, width - 1)
    vs = np.clip(pixels[:, 1].astype(int), 0, height - 1)

    pred = predicted_disparity[vs, us].astype(float)
    target = 1.0 / anchors.depths.astype(float)
    valid = np.isfinite(pred) & np.isfinite(target)
    return pred[valid], target[valid]


def solve_affine(
    pred: np.ndarray,
    target: np.ndarray,
    min_points: int = 12,
    trim_quantile: float = 0.75,
    iterations: int = 3,
) -> AffineDepthFit | None:
    """Trimmed least squares for the scale and shift relating the two."""
    if len(pred) < min_points:
        return None

    keep = np.ones(len(pred), dtype=bool)
    scale, shift = 1.0, 0.0
    for iteration in range(max(1, iterations)):
        design = np.stack([pred[keep], np.ones(keep.sum())], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, target[keep], rcond=None)

        if iteration == iterations - 1:
            break
        residuals = np.abs(scale * pred + shift - target)
        trimmed = residuals <= np.quantile(residuals, trim_quantile)
        # Never trim below the minimum -- an empty keep set has no solution.
        if trimmed.sum() < min_points:
            break
        keep = trimmed

    if not np.isfinite(scale) or not np.isfinite(shift) or scale <= 0:
        return None
    return AffineDepthFit(float(scale), float(shift), int(keep.sum()))


def fit_affine_to_sparse(
    predicted_disparity: np.ndarray,
    anchors,
    min_anchors: int = 12,
    trim_quantile: float = 0.75,
    iterations: int = 3,
) -> AffineDepthFit | None:
    """Same solve as `fit_affine_to_metric`, against sparse triangulated points.

    This is the no-LiDAR path: instead of a dense reference depth map, all that
    is available is a few dozen pixels whose depth is known from multi-view
    triangulation (see `sparse_anchors`).

    Anchor pixels are rescaled from `anchors.image_size` onto the disparity
    map's grid, so the depth model and the feature matcher are free to run at
    whatever resolution each works best at.

    Returns None when there are too few anchors to trust -- with only a handful
    of noisy points a two-parameter fit will happily produce a confident,
    wrong answer. Callers should fill those frames in from their neighbours
    (`fill_unreliable_fits`) rather than using a bad fit.
    """
    if anchors is None or len(anchors) < min_anchors:
        return None
    pred, target = sample_anchor_disparity(predicted_disparity, anchors)
    return solve_affine(pred, target, min_anchors, trim_quantile, iterations)


def fill_unreliable_fits(fits: list) -> list:
    """Replace missing per-frame fits by interpolating from neighbouring frames.

    The camera moves continuously and the depth model behaves consistently
    along the way, so scale and shift vary smoothly over a capture. A frame
    that could not be solved -- a blank wall with nothing to triangulate, or a
    turn with no parallax -- is therefore much better served by its
    neighbours' solution than by its own handful of noisy anchors.
    """
    solved = [i for i, fit in enumerate(fits) if fit is not None]
    if not solved:
        raise ValueError(
            "No frame produced a usable scale fit. Without any metric anchor the "
            "reconstruction has no real-world size -- check that the poses are "
            "metric and that the capture has enough parallax to triangulate."
        )

    indices = np.array(solved, dtype=float)
    scales = np.array([fits[i].scale for i in solved])
    shifts = np.array([fits[i].shift for i in solved])

    filled = list(fits)
    for i, fit in enumerate(fits):
        if fit is not None:
            continue
        # np.interp clamps at the ends, which is what we want: frames before the
        # first solve or after the last inherit the nearest known fit.
        filled[i] = AffineDepthFit(
            float(np.interp(i, indices, scales)),
            float(np.interp(i, indices, shifts)),
            0,
        )
    return filled


def reject_inconsistent_fits(fits: list, max_scale_ratio: float = 1.5) -> list:
    """Discard per-frame fits whose scale disagrees with the capture as a whole.

    One camera walking through one room, with one depth model, should not need
    wildly different scales from frame to frame. When it does, the outliers are
    bad fits -- a few mismatched features triangulated badly -- not real
    changes, and a two-parameter least squares will report them with total
    confidence.

    This matters more than it sounds. Admitting more frames by loosening the
    triangulation filters measurably *worsens* the reconstruction (8.4cm median
    error against LiDAR, versus 5.9cm when only a third of frames are solved
    directly), because the extra frames come in with bad scales. Frames dropped
    here are not lost -- `fill_unreliable_fits` interpolates them from
    neighbours, which is strictly better than a confident wrong answer.
    """
    scales = np.array(
        [fit.scale for fit in fits if fit is not None and fit.scale > 0]
    )
    if len(scales) < 3:
        return list(fits)

    reference = float(np.median(scales))
    low, high = reference / max_scale_ratio, reference * max_scale_ratio

    return [
        fit if (fit is not None and low <= fit.scale <= high) else None
        for fit in fits
    ]
