"""Mosaic splitting, Stokes images, DoLP/AoLP and circular statistics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

ANGLES = (0, 45, 90, 135)
DEFAULT_LAYOUT = ((90, 45), (135, 0))  # rows top to bottom (IMX250MZR, to be verified)

Channels = Dict[int, np.ndarray]


def validate_layout(layout: Sequence[Sequence[int]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the layout as a tuple of tuples; raise if it is not a permutation of 0/45/90/135."""
    flat = [int(a) for row in layout for a in row]
    if len(layout) != 2 or any(len(r) != 2 for r in layout) or sorted(flat) != sorted(ANGLES):
        raise ValueError(f"layout must be 2x2 with each of {ANGLES} once, got {layout!r}")
    return ((int(layout[0][0]), int(layout[0][1])), (int(layout[1][0]), int(layout[1][1])))


def split_mosaic(mosaic: np.ndarray, layout: Sequence[Sequence[int]] = DEFAULT_LAYOUT) -> Channels:
    """Split a polarized mosaic into four half-resolution channel images keyed by angle (deg).

    The output arrays are float32 views converted from the raw integer data.
    """
    if mosaic.ndim != 2:
        raise ValueError("mosaic must be a 2-D array")
    layout = validate_layout(layout)
    h, w = mosaic.shape
    mosaic = mosaic[: h - h % 2, : w - w % 2]
    out: Channels = {}
    for r in range(2):
        for c in range(2):
            out[layout[r][c]] = mosaic[r::2, c::2].astype(np.float32)
    return out


def correct_channels(
    channels: Channels,
    dark: Optional[Channels] = None,
    gain: Optional[Dict[int, float]] = None,
    offset: Optional[Dict[int, float]] = None,
) -> Channels:
    """Subtract dark channels, then apply per-channel (I - offset) * gain. No clipping."""
    out: Channels = {}
    for a in ANGLES:
        img = channels[a]
        if dark is not None:
            img = img - dark[a]
        if offset is not None:
            img = img - offset.get(a, 0.0)
        if gain is not None:
            img = img * gain.get(a, 1.0)
        out[a] = img
    return out


def stokes_images(ch: Channels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S0, S1, S2) images: S0=(I0+I45+I90+I135)/2, S1=I0-I90, S2=I45-I135."""
    s0 = (ch[0] + ch[45] + ch[90] + ch[135]) * 0.5
    s1 = ch[0] - ch[90]
    s2 = ch[45] - ch[135]
    return s0, s1, s2


def dolp_aolp(s0: float, s1: float, s2: float, aolp_offset_deg: float = 0.0) -> Tuple[float, float]:
    """DoLP (fraction, unclipped) and AoLP (deg, wrapped to [0,180)) from Stokes values.

    Returns (nan, nan) when S0 <= 0.
    """
    if not s0 > 0:
        return float("nan"), float("nan")
    dolp = float(np.hypot(s1, s2) / s0)
    aolp = wrap_aolp(0.5 * np.degrees(np.arctan2(s2, s1)) - aolp_offset_deg)
    return dolp, float(aolp)


def wrap_aolp(angle_deg):
    """Wrap angle(s) in degrees to [0, 180)."""
    return np.mod(angle_deg, 180.0)


def roi_stokes(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, mask: np.ndarray
) -> Tuple[float, float, float, int]:
    """Sum S0, S1, S2 over the mask. Returns (sum S0, sum S1, sum S2, pixel count)."""
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        float(s0[mask].sum(dtype=np.float64)),
        float(s1[mask].sum(dtype=np.float64)),
        float(s2[mask].sum(dtype=np.float64)),
        n,
    )


def circular_mean_aolp(aolp_deg: Sequence[float]) -> float:
    """Circular mean of AoLP (period 180 deg), in [0, 180). NaNs are ignored."""
    a = np.asarray(aolp_deg, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    z = np.exp(2j * np.radians(a)).mean()
    return float(wrap_aolp(0.5 * np.degrees(np.angle(z))))


def circular_std_aolp(aolp_deg: Sequence[float]) -> float:
    """Circular standard deviation of AoLP in degrees (period 180 deg). NaNs are ignored."""
    a = np.asarray(aolp_deg, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    r = np.abs(np.exp(2j * np.radians(a)).mean())
    r = min(max(r, 1e-12), 1.0)
    return float(0.5 * np.degrees(np.sqrt(-2.0 * np.log(r))))


def break_wraps(t: Sequence[float], aolp_deg: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Like unwrap_aolp_for_plot, but returns (t, aolp) with a NaN inserted at every wrap."""
    t = np.asarray(t, dtype=np.float64)
    a = np.asarray(aolp_deg, dtype=np.float64)
    if a.size < 2:
        return t, a
    j = np.where(np.abs(np.diff(a)) > 90.0)[0]
    return np.insert(t, j + 1, t[j]), np.insert(a, j + 1, np.nan)


def unwrap_aolp_for_plot(aolp_deg: Sequence[float]) -> np.ndarray:
    """Insert NaN where AoLP wraps (jump > 90 deg) so a line plot has no wrap-around spike."""
    a = np.asarray(aolp_deg, dtype=np.float64).copy()
    if a.size < 2:
        return a
    jumps = np.where(np.abs(np.diff(a)) > 90.0)[0]
    out = np.insert(a, jumps + 1, np.nan)
    return out
