"""Synthetic polarized-mosaic frames for tests and --simulate."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .stokes import DEFAULT_LAYOUT, validate_layout


def make_mosaic(
    shape=(2048, 2448),
    dolp: float = 0.9,
    aolp_deg: float = 30.0,
    peak_s0: float = 1500.0,
    sigma_px: float = 60.0,
    centre: Optional[Sequence[float]] = None,
    layout=DEFAULT_LAYOUT,
    noise: float = 0.0,
    background: float = 0.0,
    full_scale: float = 4095.0,
    rng: Optional[np.random.Generator] = None,
    dtype=np.uint16,
) -> np.ndarray:
    """Gaussian spot with uniform polarization state, as a raw 2x2 polarizer mosaic.

    Channel intensity: I(theta) = 0.5 * S0 * (1 + dolp * cos(2*(theta - aolp))),
    which gives S0=(sum I)/2, S1=I0-I90, S2=I45-I135 back exactly. If noise == 0 and
    dtype is a float type the round trip is exact; integer dtypes quantize.
    """
    layout = validate_layout(layout)
    h, w = shape
    cy, cx = (h / 2.0, w / 2.0) if centre is None else (centre[1], centre[0])
    yy, xx = np.mgrid[0:h, 0:w]
    s0 = peak_s0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px**2))
    theta = np.empty((h, w))
    for r in range(2):
        for c in range(2):
            theta[r::2, c::2] = layout[r][c]
    img = 0.5 * s0 * (1.0 + dolp * np.cos(np.radians(2.0 * (theta - aolp_deg)))) + background
    if noise > 0:
        rng = rng or np.random.default_rng()
        img = img + rng.normal(0.0, noise, img.shape)
    if np.issubdtype(np.dtype(dtype), np.integer):
        img = np.clip(np.rint(img), 0, full_scale)
    return img.astype(dtype)
