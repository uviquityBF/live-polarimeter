"""Camera sources: simulated, replay and the LUCID Arena SDK wrapper.

All access to `arena_api` is confined to ArenaCamera in this file, and the SDK is imported
lazily so the rest of the program (and the tests) run without it.

ArenaCamera is UNTESTED ON HARDWARE. It was written from memory of the arena_api Python
API; run `livepol-probe` on the camera machine first and fix any node/attribute names.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

from .stokes import DEFAULT_LAYOUT, validate_layout


def full_scale_from_format(pixel_format: str, default: float = 4095.0) -> float:
    """Full-scale value implied by the trailing bit depth in a pixel format name."""
    m = re.findall(r"(\d+)", pixel_format or "")
    return float(2 ** int(m[-1]) - 1) if m else float(default)


def unpack_12p(raw: np.ndarray, n_pixels: int) -> np.ndarray:
    """Unpack 12-bit packed data (two pixels in three bytes, low nibble first) to uint16."""
    b = raw[: n_pixels * 3 // 2].reshape(-1, 3).astype(np.uint16)
    out = np.empty(n_pixels, dtype=np.uint16)
    out[0::2] = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    out[1::2] = (b[:, 1] >> 4) | (b[:, 2] << 4)
    return out


def decode_buffer(raw: np.ndarray, width: int, height: int, pixel_format: str = "") -> np.ndarray:
    """Turn a raw byte buffer into a (height, width) mosaic, guided by the buffer size.

    8-bit, unpacked 16-bit (including 10/12 bit in a 16-bit container) and 12-bit packed
    ("...12p") layouts are supported; GigE Vision "Packed" variants are not.
    """
    n = width * height
    raw = np.asarray(raw, dtype=np.uint8)
    if raw.size >= 2 * n:
        img = np.frombuffer(raw[: 2 * n].tobytes(), dtype="<u2")
    elif pixel_format.endswith("p") and raw.size >= 3 * n // 2:
        img = unpack_12p(raw, n)
    elif raw.size >= n and "Packed" not in pixel_format:
        img = raw[:n]
    else:
        raise ValueError(
            f"cannot decode {raw.size} bytes as {width}x{height} {pixel_format!r}; "
            "use an unpacked PolarizeMono format"
        )
    return img.reshape(height, width).copy()


class Camera(ABC):
    """Frame source. grab() returns a 2-D raw mosaic or None on timeout."""

    full_scale: float = 4095.0
    exposure_us: float = float("nan")
    gain_db: float = float("nan")

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]: ...

    @abstractmethod
    def close(self) -> None: ...

    def set_exposure_us(self, value: float) -> None:
        self.exposure_us = float(value)

    def set_gain_db(self, value: float) -> None:
        self.gain_db = float(value)

    def info(self) -> dict:
        return {"type": type(self).__name__, "exposure_us": self.exposure_us,
                "gain_db": self.gain_db, "full_scale": self.full_scale}


class SimCamera(Camera):
    """Gaussian spot with configurable DoLP/AoLP, noise and drift; honours exposure and gain."""

    def __init__(self, cfg: dict, cam_cfg: dict, layout=DEFAULT_LAYOUT, seed: Optional[int] = None):
        self.c = cfg
        self.layout = validate_layout(layout)
        self.fps = float(cfg["fps"])
        self.exposure_us = float(cam_cfg["exposure_us"])
        self.gain_db = float(cam_cfg["gain_db"])
        fs = cam_cfg.get("full_scale", "auto")
        self.full_scale = full_scale_from_format(cam_cfg["pixel_format"]) if fs == "auto" else float(fs)
        self.rng = np.random.default_rng(seed)
        self._t0 = 0.0
        self._next = 0.0

    def open(self) -> None:
        h, w = self.c["shape"]
        yy, xx = np.mgrid[0:h, 0:w]
        s = float(self.c["sigma_px"])
        s0 = np.exp(-(((xx - w / 2.0) ** 2) + ((yy - h / 2.0) ** 2)) / (2.0 * s * s)).astype(np.float32)
        theta = np.empty((h, w), dtype=np.float32)
        for r in range(2):
            for c in range(2):
                theta[r::2, c::2] = self.layout[r][c]
        th2 = np.radians(2.0 * theta)
        self._env = s0
        self._cos = s0 * np.cos(th2)
        self._sin = s0 * np.sin(th2)
        noise = float(self.c.get("noise", 0.0))
        self._noise = (self.rng.standard_normal(h * w + 65536).astype(np.float32) * noise
                       if noise > 0 else None)
        self._t0 = time.time()
        self._next = time.time()

    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + 1.0 / self.fps, time.time() - 1.0 / self.fps)
        t = time.time() - self._t0
        c = self.c
        aolp = (c["aolp_deg"] + c.get("drift_deg_per_s", 0.0) * t
                + c.get("wobble_deg", 0.0) * np.sin(2 * np.pi * t / 7.0))
        a2 = np.radians(2.0 * aolp)
        scale = (float(c["peak_s0"]) * self.exposure_us / float(c["reference_exposure_us"])
                 * 10 ** (self.gain_db / 20.0))
        d = float(c["dolp"])
        img = 0.5 * scale * (self._env + d * (np.cos(a2) * self._cos + np.sin(a2) * self._sin))
        if self._noise is not None:
            h, w = img.shape
            off = int(self.rng.integers(0, 65536))
            img += self._noise[off: off + h * w].reshape(h, w)
        np.clip(img, 0, self.full_scale, out=img)
        return np.rint(img).astype(np.uint16)

    def close(self) -> None:
        pass


class ReplayCamera(Camera):
    """Plays back saved .npy mosaics (a 2-D frame, a 3-D stack, or a directory of .npy files)."""

    def __init__(self, path: str, fps: float, cam_cfg: dict):
        self.path = Path(path)
        self.fps = float(fps)
        fs = cam_cfg.get("full_scale", "auto")
        self.full_scale = full_scale_from_format(cam_cfg["pixel_format"]) if fs == "auto" else float(fs)
        self._frames: List[np.ndarray] = []
        self._i = 0
        self._next = 0.0

    def open(self) -> None:
        files = sorted(self.path.glob("*.npy")) if self.path.is_dir() else [self.path]
        if not files or not all(f.exists() for f in files):
            raise FileNotFoundError(f"no .npy frames found at {self.path}")
        for f in files:
            a = np.load(f)
            if a.ndim == 2:
                self._frames.append(a)
            elif a.ndim == 3:
                self._frames.extend(list(a))
            else:
                raise ValueError(f"{f}: expected a 2-D mosaic or 3-D stack, got shape {a.shape}")
        self._next = time.time()

    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + 1.0 / self.fps, time.time() - 1.0 / self.fps)
        f = self._frames[self._i % len(self._frames)]
        self._i += 1
        return f

    def close(self) -> None:
        self._frames = []


class ArenaCamera(Camera):
    """LUCID Triton via the Arena SDK. UNTESTED ON HARDWARE."""

    def __init__(self, cam_cfg: dict, ip: Optional[str] = None, serial: Optional[str] = None):
        self.cfg = cam_cfg
        self.ip = ip or cam_cfg.get("ip")
        self.serial = serial or cam_cfg.get("serial")
        self.pixel_format = cam_cfg["pixel_format"]
        fs = cam_cfg.get("full_scale", "auto")
        self.full_scale = full_scale_from_format(self.pixel_format) if fs == "auto" else float(fs)
        self.device = None
        self.system = None
        self._streaming = False
        self.exposure_us = float(cam_cfg["exposure_us"])
        self.gain_db = float(cam_cfg["gain_db"])
        self.timeout_ms = int(cam_cfg.get("grab_timeout_ms", 2000))

    # -- helpers ---------------------------------------------------------
    def _node(self, name: str):
        return self.device.nodemap[name]

    def _try_set(self, name: str, value) -> bool:
        try:
            self._node(name).value = value
            return True
        except Exception as e:  # node missing or read-only on this model/firmware
            print(f"[camera] could not set {name}={value!r}: {e}")
            return False

    # -- lifecycle -------------------------------------------------------
    def open(self) -> None:
        try:
            from arena_api.system import system
        except ImportError as e:
            raise RuntimeError(
                "arena_api is not installed. Install the Arena SDK and its Python wheel, or run "
                "with --simulate / --replay."
            ) from e
        self.system = system
        infos = system.device_infos
        if not infos:
            raise RuntimeError(
                "No camera found. Check power, the cable, the NIC's IP/subnet and jumbo frames, "
                "and that ArenaView or another program is not holding the camera.")
        pick = infos
        if self.ip:
            pick = [i for i in infos if i.get("ip") == self.ip]
        elif self.serial:
            pick = [i for i in infos if str(i.get("serial")) == str(self.serial)]
        if not pick:
            found = ", ".join(f"{i.get('model')} {i.get('serial')} @ {i.get('ip')}" for i in infos)
            raise RuntimeError(f"Requested camera not found. Visible devices: {found}")
        self.device = system.create_device(device_infos=[pick[0]])[0]
        self._configure()
        self.device.start_stream()
        self._streaming = True

    def _configure(self) -> None:
        d = self.device
        tl = d.tl_stream_nodemap
        for name, val in (("StreamAutoNegotiatePacketSize", True),
                          ("StreamPacketResendEnable", True),
                          ("StreamBufferHandlingMode", "NewestOnly")):
            try:
                tl[name].value = val
            except Exception as e:
                print(f"[camera] stream node {name}: {e}")
        self._try_set("AcquisitionMode", "Continuous")
        self._try_set("TriggerMode", "Off")
        self._try_set("PixelFormat", self.pixel_format)
        b = int(self.cfg.get("binning", 1))
        if b > 1:
            self._try_set("BinningHorizontal", b)
            self._try_set("BinningVertical", b)
        roi = self.cfg.get("roi")
        if roi:
            x, y, w, h = [int(v) for v in roi]
            if x % 2 or y % 2:
                raise ValueError("sensor ROI x and y must be even to keep the 2x2 mosaic aligned")
            self._try_set("OffsetX", 0)
            self._try_set("OffsetY", 0)
            self._try_set("Width", w)
            self._try_set("Height", h)
            self._try_set("OffsetX", x)
            self._try_set("OffsetY", y)
        self._try_set("ExposureAuto", "Off")
        self.set_exposure_us(self.exposure_us)
        self._try_set("GainAuto", "Off")
        self.set_gain_db(self.gain_db)
        if self.cfg.get("black_level") is not None:
            self._try_set("BlackLevel", float(self.cfg["black_level"]))
        try:
            self.pixel_format = str(self._node("PixelFormat").value)
        except Exception:
            pass

    def set_exposure_us(self, value: float) -> None:
        self.exposure_us = float(value)
        if self.device is not None:
            self._try_set("ExposureTime", float(value))

    def set_gain_db(self, value: float) -> None:
        self.gain_db = float(value)
        if self.device is not None:
            self._try_set("Gain", float(value))

    def grab(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            buf = self.device.get_buffer(timeout=self.timeout_ms)
        except Exception as e:
            if "imeout" in str(e):
                return None
            raise
        try:
            if getattr(buf, "is_incomplete", False):
                return None
            n = int(getattr(buf, "size_filled", 0) or buf.buffer_size)
            try:
                raw = np.ctypeslib.as_array(buf.pdata, shape=(n,))
            except Exception:
                raw = np.asarray(buf.data, dtype=np.uint8)
            return decode_buffer(raw, buf.width, buf.height, self.pixel_format)
        finally:
            self.device.requeue_buffer(buf)

    def close(self) -> None:
        try:
            if self.device is not None and self._streaming:
                self.device.stop_stream()
        finally:
            self._streaming = False
            if self.system is not None:
                self.system.destroy_device()
            self.device = None

    def info(self) -> dict:
        d = super().info()
        d.update({"pixel_format": self.pixel_format, "ip": self.ip, "serial": self.serial})
        if self.device is not None:
            for key in ("DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion",
                        "Width", "Height"):
                try:
                    d[key] = self._node(key).value
                except Exception:
                    pass
        return d


def make_camera(cfg: dict, simulate: bool = False, replay: Optional[str] = None,
                ip: Optional[str] = None, serial: Optional[str] = None) -> Camera:
    """Build the camera source from the config and command-line choices."""
    layout = cfg["processing"]["layout"]
    if simulate:
        return SimCamera(cfg["simulate"], cfg["camera"], layout)
    if replay:
        return ReplayCamera(replay, cfg["replay"]["fps"], cfg["camera"])
    return ArenaCamera(cfg["camera"], ip=ip, serial=serial)
