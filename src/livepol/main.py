"""Command-line entry point: livepol [--simulate | --replay FILE] [--no-gui] ..."""

from __future__ import annotations

import argparse
import sys
import time

from .camera import make_camera
from .config import load_config
from .engine import Engine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="livepol", description="Live linear polarimeter (Triton polarization camera)")
    p.add_argument("--config", help="YAML config merged over config/default.yaml")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--simulate", action="store_true", help="synthetic camera, no hardware needed")
    src.add_argument("--replay", metavar="FILE", help="play back saved .npy mosaic(s) (file or directory)")
    p.add_argument("--ip", help="camera IP address")
    p.add_argument("--serial", help="camera serial number")
    p.add_argument("--exposure", type=float, help="exposure time in us")
    p.add_argument("--gain", type=float, help="gain in dB")
    p.add_argument("--no-gui", action="store_true", help="headless: print readings (use with --log)")
    p.add_argument("--duration", type=float, default=0, help="headless run time in s (0 = until Ctrl+C)")
    p.add_argument("--log", metavar="CSV", help="start logging to this CSV file immediately")
    p.add_argument("--note", default="", help="free-text note written into the log header")
    p.add_argument("--out", help="output directory (default from config)")
    p.add_argument("--sim-dolp", type=float)
    p.add_argument("--sim-aolp", type=float)
    p.add_argument("--sim-noise", type=float)
    p.add_argument("--sim-shape", type=int, nargs=2, metavar=("ROWS", "COLS"))
    return p


def overrides_from_args(a: argparse.Namespace) -> dict:
    o: dict = {"camera": {}, "simulate": {}, "output": {}}
    if a.exposure is not None:
        o["camera"]["exposure_us"] = a.exposure
    if a.gain is not None:
        o["camera"]["gain_db"] = a.gain
    if a.sim_dolp is not None:
        o["simulate"]["dolp"] = a.sim_dolp
    if a.sim_aolp is not None:
        o["simulate"]["aolp_deg"] = a.sim_aolp
    if a.sim_noise is not None:
        o["simulate"]["noise"] = a.sim_noise
    if a.sim_shape:
        o["simulate"]["shape"] = list(a.sim_shape)
    if a.out:
        o["output"]["dir"] = a.out
    return o


def run_headless(engine: Engine, duration: float) -> None:
    t0 = time.time()
    try:
        while not engine.error and (duration <= 0 or time.time() - t0 < duration):
            time.sleep(1.0)
            r = engine.latest()
            if r is not None:
                print(f"DoLP {r.dolp * 100:6.2f} %  AoLP {r.aolp:6.2f} deg  S0 {r.s0_mean:8.1f}  "
                      f"ROI {r.n_px} px  sat={int(r.saturated)}  {engine.stats()['fps']:.1f} fps", flush=True)
    except KeyboardInterrupt:
        pass


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    try:
        camera = make_camera(cfg, simulate=args.simulate, replay=args.replay,
                             ip=args.ip, serial=args.serial)
        engine = Engine(camera, cfg)
        engine.start()
    except Exception as e:  # clear message instead of a traceback for the common failures
        print(f"livepol: {e}", file=sys.stderr)
        return 1
    if args.log:
        engine.start_log(args.log, args.note)
    try:
        if args.no_gui:
            run_headless(engine, args.duration)
            return 1 if engine.error else 0
        from .gui import run_gui
        return run_gui(engine, cfg)
    finally:
        engine.stop()
        if engine.error:
            print(f"livepol: {engine.error}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
