"""ROI masks on the Stokes (half-resolution) grid, plus the saturation check."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .stokes import Channels


def percentile_mask(s0: np.ndarray, percentile: float = 99.5) -> Tuple[np.ndarray, float]:
    """Pixels with S0 >= the given percentile of the S0 image. Returns (mask, threshold)."""
    thr = float(np.percentile(s0, percentile))
    return s0 >= thr, thr


def rect_mask(shape: Tuple[int, int], x: int, y: int, w: int, h: int) -> np.ndarray:
    """Rectangle mask; x, y is the top-left corner. Clipped to the image."""
    m = np.zeros(shape, dtype=bool)
    m[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)] = True
    return m


def circle_mask(shape: Tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    """Circle mask centred at (cx, cy) with radius r pixels."""
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def mask_from_config(s0: np.ndarray, cfg: dict) -> Tuple[np.ndarray, Optional[float]]:
    """Build the ROI mask from the processing.roi config block.

    Returns (mask, threshold); threshold is only set in auto mode.
    """
    mode = cfg.get("mode", "auto")
    if mode == "auto":
        return percentile_mask(s0, float(cfg.get("percentile", 99.5)))
    if mode == "rect":
        x, y, w, h = cfg["rect"]
        return rect_mask(s0.shape, int(x), int(y), int(w), int(h)), None
    if mode == "circle":
        cx, cy, r = cfg["circle"]
        return circle_mask(s0.shape, cx, cy, r), None
    raise ValueError(f"unknown ROI mode {mode!r}")


def saturation_check(
    raw_channels: Channels, mask: np.ndarray, full_scale: float, fraction: float = 0.98
) -> Tuple[bool, float]:
    """Check raw (uncorrected) channel pixels inside the mask against full scale.

    Returns (saturated, max_fraction) where max_fraction is the largest raw value in the
    ROI divided by full_scale. Use it to suggest an exposure change.
    """
    if not mask.any():
        return False, 0.0
    peak = max(float(ch[mask].max()) for ch in raw_channels.values())
    frac = peak / full_scale
    return frac >= fraction, frac


def suggested_exposure_scale(max_fraction: float, target: float = 0.8) -> float:
    """Factor to multiply exposure time by to bring the ROI peak to `target` of full scale."""
    return target / max_fraction if max_fraction > 0 else 1.0
