"""Per-frame processing: mosaic -> Stokes -> ROI -> DoLP/AoLP -> saturation check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import roi as roi_mod
from . import stokes
from .config import load_calibration


@dataclass
class Result:
    frame_index: int
    t: float
    s0: np.ndarray            # half-resolution S0 image (corrected)
    mask: np.ndarray          # ROI mask on the same grid
    threshold: Optional[float]  # auto-ROI S0 threshold (counts), else None
    roi_mode: str
    percentile: float
    s0_mean: float            # ROI mean per Stokes pixel
    s1_mean: float
    s2_mean: float
    n_px: int
    dolp: float               # fraction, not clipped
    aolp: float               # degrees in [0, 180)
    saturated: bool
    max_frac: float           # largest raw ROI pixel / full scale
    exposure_us: float
    gain_db: float

    @property
    def dolp_over_100(self) -> bool:
        return self.dolp > 1.0

    def scalars(self) -> dict:
        """JSON-friendly summary (no images)."""
        return {
            "frame_index": self.frame_index, "t": self.t, "roi_mode": self.roi_mode,
            "threshold": self.threshold, "s0_mean": self.s0_mean, "s1_mean": self.s1_mean,
            "s2_mean": self.s2_mean, "roi_px": self.n_px, "dolp": self.dolp,
            "dolp_percent": self.dolp * 100.0, "aolp_deg": self.aolp,
            "saturated": self.saturated, "max_fraction_of_full_scale": self.max_frac,
            "dolp_over_100": self.dolp_over_100,
            "exposure_us": self.exposure_us, "gain_db": self.gain_db,
        }


class Processor:
    """Holds processing settings, dark frame and calibration; turns mosaics into Results."""

    def __init__(self, proc_cfg: dict, full_scale: float):
        self.cfg = proc_cfg
        self.full_scale = float(full_scale)
        cal = load_calibration(proc_cfg.get("calibration_file"))
        self.layout = stokes.validate_layout(cal["layout"] or proc_cfg["layout"])
        self.gain = cal["gain"]
        self.offset = cal["offset"]
        self.roi_cfg = dict(proc_cfg["roi"])
        self.dark: Optional[stokes.Channels] = None
        self.dark_shape = None

    def set_dark(self, mosaic: Optional[np.ndarray]) -> None:
        """Set (or clear, with None) the dark frame, given as a raw mosaic."""
        if mosaic is None:
            self.dark, self.dark_shape = None, None
        else:
            self.dark = stokes.split_mosaic(mosaic, self.layout)
            self.dark_shape = mosaic.shape

    def set_roi(self, roi_cfg: dict) -> None:
        self.roi_cfg = dict(roi_cfg)  # atomic swap; read once per frame

    def process(self, mosaic: np.ndarray, frame_index: int, t: float,
                exposure_us: float = float("nan"), gain_db: float = float("nan")) -> Result:
        raw = stokes.split_mosaic(mosaic, self.layout)
        dark = self.dark
        if dark is not None and dark[0].shape != raw[0].shape:
            dark = None  # frame size changed (sensor ROI); ignore stale dark
        ch = stokes.correct_channels(raw, dark, self.gain, self.offset)
        s0, s1, s2 = stokes.stokes_images(ch)
        roi_cfg = self.roi_cfg
        mask, thr = roi_mod.mask_from_config(s0, roi_cfg)
        a, b, c, n = stokes.roi_stokes(s0, s1, s2, mask)
        dolp, aolp = stokes.dolp_aolp(a, b, c, float(self.cfg.get("aolp_offset_deg", 0.0)))
        sat, frac = roi_mod.saturation_check(
            raw, mask, self.full_scale, float(self.cfg.get("saturation_fraction", 0.98)))
        inv = 1.0 / n if n else float("nan")
        return Result(
            frame_index=frame_index, t=t, s0=s0, mask=mask, threshold=thr,
            roi_mode=roi_cfg.get("mode", "auto"), percentile=float(roi_cfg.get("percentile", 99.5)),
            s0_mean=a * inv, s1_mean=b * inv, s2_mean=c * inv, n_px=n, dolp=dolp, aolp=aolp,
            saturated=sat, max_frac=frac, exposure_us=exposure_us, gain_db=gain_db,
        )
