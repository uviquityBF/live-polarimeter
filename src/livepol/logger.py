"""CSV logging with a settings JSON written next to every log."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import __version__
from .pipeline import Result

COLUMNS = ["timestamp", "t_unix", "s0_mean", "s1_mean", "s2_mean", "dolp", "aolp_deg",
           "roi_px", "saturated", "exposure_us", "gain_db"]


class CSVLogger:
    """Rate-limited CSV logger. S0/S1/S2 are ROI means per Stokes pixel (sum = mean * roi_px)."""

    def __init__(self, path, note: str = "", settings: Optional[dict] = None, rate_hz: float = 5.0):
        self.path = Path(path)
        self.note = note
        self.settings = settings or {}
        self.min_dt = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._last = -1e18
        self._f = None
        self.rows = 0

    @property
    def settings_path(self) -> Path:
        return self.path.with_suffix(".settings.json")

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now().isoformat(timespec="seconds")
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._f.write(f"# livepol {__version__} log, started {started}\n")
        for line in (self.note or "").splitlines() or [""]:
            self._f.write(f"# note: {line}\n")
        self._f.write("# s0/s1/s2 are ROI means per Stokes pixel; dolp is a fraction (unclipped)\n")
        self._f.write(",".join(COLUMNS) + "\n")
        self._f.flush()
        meta = {"livepol_version": __version__, "started": started, "note": self.note,
                "log_file": self.path.name, **self.settings}
        self.settings_path.write_text(json.dumps(meta, indent=2, default=str))

    def write(self, r: Result) -> bool:
        """Write a row if the rate limit allows it. Returns True if a row was written."""
        if self._f is None or r.t - self._last < self.min_dt:
            return False
        self._last = r.t
        ts = datetime.fromtimestamp(r.t).isoformat(timespec="milliseconds")
        self._f.write(
            f"{ts},{r.t:.3f},{r.s0_mean:.6g},{r.s1_mean:.6g},{r.s2_mean:.6g},{r.dolp:.6g},"
            f"{r.aolp:.4f},{r.n_px},{int(r.saturated)},{r.exposure_us:.6g},{r.gain_db:.6g}\n")
        self._f.flush()
        self.rows += 1
        return True

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
