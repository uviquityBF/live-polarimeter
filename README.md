# live-polarimeter

Live linear polarimeter for a LUCID Triton TRI050S1-P (Sony IMX250MZR) polarization camera.
Measures S0, S1, S2, DoLP and AoLP of a laser or fiber-output spot and shows them live.
S3 (circular polarization) is not measurable with this sensor.

## Status

| Step | Content | State |
|---|---|---|
| 0-1 | Scaffold, `stokes.py`, `roi.py`, `overlap.py`, `simulate.py`, tests | done |
| 2 | Simulated camera, acquisition/processing threads, live display | done |
| 3 | CSV logging, settings JSON, snapshots, raw frames, `--replay` | done |
| 4 | Reference states A/B, delta, eta, overlap loss in the UI | done |
| 5 | Arena camera wrapper (`camera.py`) and `livepol-probe` | written, **untested on hardware** |
| 6 | Calibration utility | todo |
| 7 | Validation against the MATLAB script | todo (needs files in `reference/`) |

## Setup, development machine (no camera)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[gui,dev]
pytest
livepol --simulate
```

## Setup, camera machine

1. Install the LUCID Arena SDK, then its Python package `arena_api` (a wheel shipped with the
   SDK download, not on PyPI). Use a Python version the SDK supports (3.9 or newer).
2. Network: give the camera NIC a static IP on the camera's subnet, enable jumbo frames
   (about 9000 bytes) and close ArenaView or anything else holding the camera.
3. Clone and install into the same venv as `arena_api`:
   ```powershell
   git clone https://github.com/uviquityBF/live-polarimeter.git
   cd live-polarimeter
   python -m venv .venv ; .\.venv\Scripts\activate
   pip install <path-to-arena_api-wheel>
   pip install -e .[gui,dev]
   pytest                      # should pass without the camera
   ```
4. **Run the probe first** and keep its output:
   ```powershell
   livepol-probe               # add --ip 192.168.x.y if several cameras
   ```
   It lists the cameras, prints pixel-format names, sizes, exposure/gain ranges, IP and
   firmware, and saves a few raw frames to `reference/probe_frames.npy`.
5. Run the live view: `livepol` (or `livepol --ip 192.168.x.y --exposure 3000`).

## Running

```
livepol                         real camera, GUI
livepol --simulate              synthetic spot (try --sim-dolp 0.9 --sim-aolp 30 --sim-noise 3)
livepol --replay frames.npy     play back saved mosaics (file with 2-D/3-D array, or a folder)
livepol --no-gui --duration 60 --log run1.csv --note "delay arm"     headless logging
livepol --config my.yaml        override anything in config/default.yaml
```

GUI: S0 image with ROI outline, DoLP / AoLP / S0 readouts, polarization diagram, rolling
DoLP / AoLP / S0 plots with min/max/std (circular for AoLP). Buttons: pause, snapshot (PNG +
JSON), save raw frame (.npy + settings JSON), capture dark (beam blocked), start/stop CSV
log with a note, exposure/gain/averaging, ROI mode (auto percentile, manual rectangle,
manual circle; drag the yellow shape), and Store A / Store B which average the last N seconds
into a reference state and show delta, eta = 0.5 (1 + dA dB cos 2 delta) and loss in dB.

CSV columns: `timestamp, t_unix, s0_mean, s1_mean, s2_mean, dolp, aolp_deg, roi_px,
saturated, exposure_us, gain_db`. S0/S1/S2 are ROI means per Stokes pixel; DoLP and AoLP are
computed from the ROI sums, DoLP is a fraction and is never clipped. A `*.settings.json` is
written next to every log.

Dark frames depend on exposure, gain and sensor ROI: capture a new one after changing them.

## Performance

Processing a full 2448 x 2048 frame takes about 27 ms on the development laptop (split,
Stokes, percentile ROI, saturation check), i.e. up to about 35 fps. The camera delivers about
24 fps at full frame over GigE, and the display refreshes at `display.update_hz` (10 Hz).
The camera is the limit at full frame; use a sensor ROI or binning (`camera.roi`,
`camera.binning`) for higher rates. Acquisition and processing run in separate threads from
the UI, and frames that arrive while processing is busy are dropped (counted in the status bar).

## Untested on hardware

Everything that touches the real camera is untested until run on the camera machine:

- `ArenaCamera` in `camera.py` was written from memory of the `arena_api` Python API.
  Node names (`PixelFormat`, `ExposureTime`, `Gain`, `BinningHorizontal`, ...), the buffer
  attributes (`pdata`, `size_filled`, `width`, `height`) and the stream settings
  (`StreamAutoNegotiatePacketSize`, `StreamPacketResendEnable`, `StreamBufferHandlingMode`)
  need checking against the shipped examples. Failed node writes print `[camera] could not set ...`.
- Pixel format names (the spec says PolarizeMono12 / PolarizeMono16; the camera may call
  them something else, e.g. a packed `...12p`). Unpacked 8/16-bit and 12-bit `...12p` are
  decoded; GigE Vision "Packed" variants are not.
- Whether 12-bit data is right- or left-aligned in a 16-bit container. If it is left-aligned
  set `camera.full_scale: 65535` or shift the data; the probe prints the max value as a hint.
- The 2x2 layout (`processing.layout`, default 90/45 over 135/0). Check it by rotating a
  polarizer to a known angle; the calibration utility will automate this.
- Polarization diagram convention (E_x horizontal, E_y vertical, red line at AoLP with
  half-length sqrt(DoLP * S0)) and the MATLAB comparison: not yet checked against the script.
- GigE stream behaviour (packet size negotiation, resend, dropped frames) and real frame rate.
