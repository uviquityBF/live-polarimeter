"""Camera probe: run this FIRST on the camera machine (UNTESTED ON HARDWARE).

Lists cameras, prints the settings that matter (pixel formats, size, exposure/gain ranges,
IP, firmware), grabs a few frames and saves them under reference/ as .npy so they can be
used for development with --replay. Paste the printed report back to the developer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .camera import ArenaCamera, decode_buffer  # noqa: F401  (decode_buffer used by ArenaCamera)
from .config import load_config

NODES = ["DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion", "DeviceVendorName",
         "GevCurrentIPAddress", "PixelFormat", "Width", "Height", "WidthMax", "HeightMax",
         "OffsetX", "OffsetY", "BinningHorizontal", "BinningVertical", "ExposureAuto",
         "ExposureTime", "GainAuto", "Gain", "BlackLevel", "AcquisitionFrameRate",
         "DeviceLinkThroughputLimit", "GevSCPSPacketSize", "PayloadSize"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="livepol-probe")
    p.add_argument("--ip")
    p.add_argument("--serial")
    p.add_argument("--frames", type=int, default=5, help="frames to save (default 5)")
    p.add_argument("--out", default="reference", help="directory for the saved frames")
    p.add_argument("--pixel-format", help="try this pixel format (default from config)")
    args = p.parse_args(argv)

    try:
        from arena_api.system import system
    except ImportError:
        print("arena_api is not installed (install the wheel from the Arena SDK).", file=sys.stderr)
        return 1

    infos = system.device_infos
    print(f"{len(infos)} device(s) found:")
    for i in infos:
        print("  ", {k: i.get(k) for k in ("model", "vendor", "serial", "ip", "subnetmask", "mac")})
    if not infos:
        return 1

    cfg = load_config()
    if args.pixel_format:
        cfg["camera"]["pixel_format"] = args.pixel_format
    cam = ArenaCamera(cfg["camera"], ip=args.ip, serial=args.serial)
    try:
        cam.open()
    except Exception as e:
        print(f"open failed: {e}", file=sys.stderr)
        return 1
    try:
        print("\nNode report:")
        for name in NODES:
            try:
                n = cam._node(name)
                extra = ""
                for attr in ("min", "max"):
                    try:
                        extra += f" {attr}={getattr(n, attr)}"
                    except Exception:
                        pass
                print(f"  {name:28s} = {n.value}{extra}")
            except Exception as e:
                print(f"  {name:28s} unavailable ({type(e).__name__})")
        try:
            print("\nPixelFormat options:", list(cam._node("PixelFormat").enumentry_names))
        except Exception as e:
            print("\nPixelFormat options unavailable:", e)
        try:
            tl = cam.device.tl_stream_nodemap
            for name in ("StreamAutoNegotiatePacketSize", "StreamPacketResendEnable",
                         "StreamBufferHandlingMode"):
                print(f"  stream {name} = {tl[name].value}")
        except Exception as e:
            print("stream nodes unavailable:", e)

        frames = []
        for _ in range(args.frames):
            f = cam.grab()
            if f is not None:
                frames.append(f)
        if not frames:
            print("no frames received", file=sys.stderr)
            return 1
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stack = np.stack(frames)
        np.save(out / "probe_frames.npy", stack)
        f = frames[-1]
        print(f"\nsaved {len(frames)} frame(s) to {out / 'probe_frames.npy'}: "
              f"shape {f.shape}, dtype {f.dtype}, min {f.min()}, max {f.max()}, mean {f.mean():.1f}")
        print("Max value hints at the bit depth/alignment: ~4095 => 12-bit right-aligned, "
              "multiples of 16 => 12-bit left-aligned in 16 bits.")
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
