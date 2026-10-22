# live-polarimeter

Live linear polarimeter for a LUCID Triton TRI050S1-P (Sony IMX250MZR) polarization camera.
Measures S0, S1, S2, DoLP and AoLP of a laser or fiber-output spot. S3 (circular
polarization) is not measurable with this sensor.

## Status

| Step | Content | State |
|---|---|---|
| 0 | Scaffold | done |
| 1 | `stokes.py`, `roi.py`, `overlap.py`, `simulate.py` + tests | done |
| 2 | Simulated camera + live display | todo |
| 3 | Logging, snapshots, replay | todo |
| 4 | Reference states A/B, overlap in the UI | todo |
| 5 | Real camera wrapper (`camera.py`) | todo |
| 6 | Calibration utility | todo |
| 7 | Validation against the MATLAB script | todo (needs files in `reference/`) |

## Setup (development machine, no camera)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[gui,dev]
pytest
```

## Setup (camera machine)

1. Install the LUCID Arena SDK and its Python package (`arena_api` wheel from the SDK
   download; it is not on PyPI). Check the supported Python versions for your SDK release.
2. `pip install -e .[gui,dev]` and install the `arena_api` wheel into the same venv.
3. Pull this repo, run the tests, then run with the camera.

## Untested on hardware

Everything that touches the real camera (`camera.py`, once written) is untested on
hardware until it has been run on the camera machine. Items to verify there:

- Pixel format names (PolarizeMono12 / PolarizeMono16) and the 2x2 layout
  (config `processing.layout`; the default is a recollection of the IMX250MZR layout).
- Stream/packet-size negotiation and buffer handling.
- Frame rate at full frame.
