"""Acquisition and processing threads. The UI only reads results; it never touches the camera.

Acquirer thread: grabs frames (optionally averaging N) and keeps only the newest.
Engine thread:   processes the newest frame, keeps history, feeds the logger.
Frames that arrive while processing is busy replace the pending one (counted as dropped).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .camera import Camera, full_scale_from_format
from .logger import CSVLogger
from .pipeline import Processor, Result


class Acquirer(threading.Thread):
    def __init__(self, camera: Camera, timeout: float = 1.0, n_avg: int = 1):
        super().__init__(daemon=True, name="livepol-acquire")
        self.camera = camera
        self.timeout = timeout
        self.n_avg = max(1, int(n_avg))
        self.stop_event = threading.Event()
        self.new = threading.Event()
        self.lock = threading.Lock()
        self._latest = None
        self._index = 0
        self._collect = None
        self.n_grabbed = 0
        self.n_dropped = 0
        self.error: Optional[str] = None

    def run(self) -> None:
        acc, cnt = None, 0
        while not self.stop_event.is_set():
            try:
                frame = self.camera.grab(self.timeout)
            except Exception as e:  # surface camera failures to the UI
                self.error = f"camera error: {e}"
                break
            if frame is None:
                continue
            self.n_grabbed += 1
            self._feed_collect(frame)
            if self.n_avg > 1:
                acc = frame.astype(np.float32) if acc is None else acc + frame
                cnt += 1
                if cnt < self.n_avg:
                    continue
                frame, acc, cnt = acc / cnt, None, 0
            with self.lock:
                if self._latest is not None:
                    self.n_dropped += 1
                self._latest = (frame, time.time(), self._index)
                self._index += 1
            self.new.set()

    def take(self):
        with self.lock:
            item, self._latest = self._latest, None
            self.new.clear()
        return item

    def _feed_collect(self, frame: np.ndarray) -> None:
        c = self._collect
        if c is None:
            return
        c["sum"] = frame.astype(np.float64) if c["sum"] is None else c["sum"] + frame
        c["count"] += 1
        if c["count"] >= c["n"]:
            c["done"].set()
            self._collect = None

    def collect(self, n: int, timeout: float = 30.0) -> np.ndarray:
        """Average the next n raw frames (blocking). Used for dark frames."""
        c = {"n": int(n), "count": 0, "sum": None, "done": threading.Event()}
        self._collect = c
        if not c["done"].wait(timeout):
            self._collect = None
            raise TimeoutError("timed out collecting frames")
        return (c["sum"] / c["count"]).astype(np.float32)


class Engine:
    HISTORY_ROWS = 30000

    def __init__(self, camera: Camera, cfg: dict, out_dir: Optional[str] = None):
        self.camera = camera
        self.cfg = cfg
        self.out_dir = Path(out_dir or cfg["output"]["dir"])
        fs = cfg["camera"].get("full_scale", "auto")
        self.full_scale = (full_scale_from_format(cfg["camera"]["pixel_format"])
                           if fs == "auto" else float(fs))
        if hasattr(camera, "full_scale"):
            self.full_scale = float(camera.full_scale)
        self.processor = Processor(cfg["processing"], self.full_scale)
        self.acq = Acquirer(camera, n_avg=cfg["camera"].get("average_frames", 1))
        self.paused = False
        self.error: Optional[str] = None
        self.message = ""
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[Result] = None
        self._raw: Optional[np.ndarray] = None
        self._hist = deque(maxlen=self.HISTORY_ROWS)
        self._logger: Optional[CSVLogger] = None
        self._fps = 0.0
        self._last_t = None
        self.n_processed = 0
        self.dark_file: Optional[str] = None
        dark = cfg["processing"].get("dark_file")
        if dark:
            self.processor.set_dark(np.load(dark))
            self.dark_file = str(dark)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.camera.open()
        self.acq.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="livepol-engine")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.acq.stop_event.set()
        for t in (self._thread, self.acq):
            if t is not None and t.is_alive():
                t.join(timeout=5)
        self.stop_log()
        try:
            self.camera.close()
        except Exception as e:
            self.message = f"error closing camera: {e}"

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.acq.new.wait(0.2):
                if self.acq.error:
                    self.error = self.acq.error
                    return
                continue
            item = self.acq.take()
            if item is None or self.paused:
                continue
            mosaic, t, idx = item
            try:
                res = self.processor.process(
                    mosaic, idx, t, self.camera.exposure_us, self.camera.gain_db)
            except Exception as e:
                self.error = f"processing error: {e}"
                return
            if self._last_t is not None and t > self._last_t:
                inst = 1.0 / (t - self._last_t)
                self._fps = inst if self._fps == 0 else 0.8 * self._fps + 0.2 * inst
            self._last_t = t
            with self._lock:
                self._latest, self._raw = res, mosaic
                self._hist.append((t, res.s0_mean, res.s1_mean, res.s2_mean,
                                   res.dolp, res.aolp, res.n_px, res.saturated))
                self.n_processed += 1
                lg = self._logger
            if lg is not None:
                lg.write(res)

    # -- reading state ---------------------------------------------------
    def latest(self) -> Optional[Result]:
        with self._lock:
            return self._latest

    def history(self, seconds: float) -> dict:
        """Arrays (t, s0, s1, s2, dolp, aolp, n, sat) for the last `seconds`."""
        with self._lock:
            rows = list(self._hist)
        if not rows:
            return {k: np.empty(0) for k in ("t", "s0", "s1", "s2", "dolp", "aolp", "n", "sat")}
        a = np.array(rows, dtype=float)
        a = a[a[:, 0] >= a[-1, 0] - seconds]
        return dict(zip(("t", "s0", "s1", "s2", "dolp", "aolp", "n", "sat"), a.T))

    def stats(self) -> dict:
        return {"fps": self._fps, "grabbed": self.acq.n_grabbed, "dropped": self.acq.n_dropped,
                "processed": self.n_processed}

    # -- controls --------------------------------------------------------
    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def set_roi(self, roi_cfg: dict) -> None:
        self.processor.set_roi(roi_cfg)

    def set_exposure_us(self, v: float) -> None:
        self.camera.set_exposure_us(v)

    def set_gain_db(self, v: float) -> None:
        self.camera.set_gain_db(v)

    def set_average_frames(self, n: int) -> None:
        self.acq.n_avg = max(1, int(n))

    def capture_dark(self, n_frames: int = 20, on_done=None) -> None:
        """Average the next n raw frames (beam blocked) in the background and use as dark."""
        def job():
            try:
                dark = self.acq.collect(n_frames)
                self.processor.set_dark(dark)
                self.out_dir.mkdir(parents=True, exist_ok=True)
                path = self.out_dir / f"dark_{datetime.now():%Y%m%d_%H%M%S}.npy"
                np.save(path, dark)
                self.dark_file = str(path)
                self.message = f"dark frame saved: {path}"
            except Exception as e:
                self.message = f"dark capture failed: {e}"
            if on_done:
                on_done(self.message)
        self.message = f"capturing dark ({n_frames} frames)..."
        threading.Thread(target=job, daemon=True).start()

    def clear_dark(self) -> None:
        self.processor.set_dark(None)
        self.dark_file = None

    # -- saving ----------------------------------------------------------
    def settings_dict(self, note: str = "") -> dict:
        return {"note": note, "config": self.cfg, "camera": self.camera.info(),
                "dark_file": self.dark_file, "layout": self.processor.layout,
                "full_scale": self.full_scale, "livepol_version": __version__}

    def save_raw(self, path=None, note: str = "") -> Path:
        """Save the latest raw frame as .npy with a settings JSON next to it."""
        with self._lock:
            raw = None if self._raw is None else self._raw.copy()
        if raw is None:
            raise RuntimeError("no frame available yet")
        path = Path(path) if path else self.out_dir / f"frame_{datetime.now():%Y%m%d_%H%M%S}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, raw)
        path.with_suffix(".settings.json").write_text(
            json.dumps(self.settings_dict(note), indent=2, default=str))
        return path

    def start_log(self, path=None, note: str = "") -> Path:
        self.stop_log()
        path = Path(path) if path else self.out_dir / f"log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        lg = CSVLogger(path, note, self.settings_dict(note), self.cfg["output"]["log_rate_hz"])
        lg.open()
        with self._lock:
            self._logger = lg
        return path

    def stop_log(self) -> None:
        with self._lock:
            lg, self._logger = self._logger, None
        if lg is not None:
            lg.close()

    @property
    def logging_path(self) -> Optional[Path]:
        lg = self._logger
        return lg.path if lg else None

    @property
    def logging_rows(self) -> int:
        lg = self._logger
        return lg.rows if lg else 0
