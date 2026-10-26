# live-polarimeter: notes for Claude

Read `README.md` for what the project is and how it is laid out. This file records the state of
the camera PC (UVLab3) and what was established on it, so a new session can continue.

## Environment on this PC

- Repo: `C:\Users\UVLab3\Documents\GitHub\live-polarimeter` (cloned 2026-10-26, moved here from
  `C:\Users\UVLab3\live-polarimeter`; `.venv` rebuilt after the move).
- Python 3.11.0. Always use the project venv: `.venv\Scripts\python`, `.venv\Scripts\livepol`,
  `.venv\Scripts\livepol-probe`. PowerShell blocks `Activate.ps1` (execution policy), so either
  call the venv executables directly or run `Set-ExecutionPolicy -Scope Process Bypass` first.
- A stale editable install of `livepol` exists in the system-wide Python and points at the old,
  deleted path. Do not use bare `python`/`livepol` for this project; uninstalling it
  (`python -m pip uninstall livepol`) is pending the user's OK.
- Arena SDK v1.0.80.56 is installed in `C:\Program Files\Lucid Vision Labs\Arena SDK`. It was first
  installed without the C library; it was reinstalled with all components so
  `x64Release\ArenaC_v140.dll` exists (`arena_api` needs it).
- `arena_api` 2.9.5 is installed in `.venv` from the company shared drive:
  `G:\Shared drives\Corp Main\Engineering\Software Installers & Lab Computers (Lenovo IdeaCentres)\Installers for Apps\Lucid\arena_api-2.9.5-py3-none-any\arena_api-2.9.5-py3-none-any.whl`.
  LUCID's Python example scripts (`examples\py_*.py`) and docs are in that same folder. Use them as
  the reference for the `arena_api` calls in `camera.py`.
- To rebuild the venv:
  `python -m venv .venv` then `.venv\Scripts\python -m pip install <arena_api wheel> -e .[gui,dev]`.

## Camera (confirmed with `livepol-probe` on 2026-10-26)

- LUCID TRI050S1-P, serial 260302461, firmware 1.141.0.0, IP 169.254.65.130/16 (link-local, DHCP),
  directly connected.
- `ArenaCamera.open()` and every node write in `camera.py` succeeded; stream nodes
  `StreamAutoNegotiatePacketSize`, `StreamPacketResendEnable`, `StreamBufferHandlingMode` exist.
- Pixel format `PolarizeMono12` works. Frames are uint16, 2448 x 2048, max 4094, so the 12-bit data
  is **right-aligned** and `full_scale: auto` (4095) is correct.
- Other formats offered: Mono8/10/10p/10Packed/12/12p/12Packed/16, PolarizeMono8, PolarizeMono12p,
  PolarizeMono12Packed, PolarizeMono16, plus on-camera computed formats (PolarizedAngles_*,
  PolarizedStokes_*, PolarizedDolpAolp_*, PolarizedDolp_*, PolarizedAolp_*, PolarizedDolpAngle_*).
- Ranges: ExposureTime 33.5 to 94556 us, Gain 0 to 48 dB, BlackLevel 0 to 12.5, binning 1 to 8.
- **Frame rate is about 10.6 fps** (AcquisitionFrameRate max 10.56). GevSCPSPacketSize is 1500,
  so jumbo frames are not enabled on the NIC, and 10 MB/frame at 12-bit unpacked hits the 1 GbE
  limit. Options: enable jumbo frames (~9000) on the NIC, use `PolarizeMono12p`, or use a sensor
  ROI/binning. The README's "about 24 fps" claim is not what this setup achieves yet.
- Probe frames are in `reference\probe_frames.npy` (gitignored) and can be used with `--replay`.
- `livepol` with the real camera runs and the user reports it "looks pretty good". The
  polarizer-rotation check of the 2x2 layout (`processing.layout`, `[[90, 45], [135, 0]]`) has
  **not** been done yet.

## Launching and output

- `launch-livepol.sh` (repo root) runs the venv's `livepol` from the repo folder and passes extra
  arguments through. The desktop shortcut target is
  `"C:\Program Files\Git\git-bash.exe" "<repo>\launch-livepol.sh"`; pointing a shortcut at the
  `.sh` directly makes Windows ask which app to use.
- Saves (raw frames, snapshots, darks, CSV logs) go to `data\` in the repo (`output.dir`), which is
  gitignored. `persist.json` is written by the Arena SDK at runtime and is also gitignored.

## Gotchas found

- Only one process can control the camera. ArenaView, or any leftover Python process, causes
  `GC_ERR_ACCESS_DENIED` / `GC_ERR_RESOURCE_IN_USE`. After killing a holder, the camera can keep the
  lock for a few seconds (heartbeat timeout) before it is free again.
- `tests/test_app.py::test_cli_no_camera_message` assumes `arena_api` is not installed. On this PC it
  opens the real camera via `livepol --no-gui` and never returns, which also locks the camera.
  Run the suite with `--deselect tests/test_app.py::test_cli_no_camera_message` until it is fixed.
  The rest (69 tests) pass.
- On Windows a folder cannot be moved or deleted while a shell's working directory is inside it,
  including Claude's own Bash tool shell.

## Next steps (in rough order)

1. Fix `test_cli_no_camera_message` so it never touches real hardware (e.g. make the
   `arena_api` import fail or mock the camera), then run the full suite.
2. Verify the 2x2 polarizer layout and AoLP direction by rotating a linear polarizer in the beam
   (DoLP should stay near 1, AoLP should track the rotation).
3. Raise the frame rate: jumbo frames on the camera NIC, and/or `PolarizeMono12p`, sensor ROI.
4. Update the README "Untested on hardware" section and status table with the confirmed facts above.
5. Calibration utility (README step 6).
6. MATLAB comparison (README step 7); needs the MATLAB script and data in `reference/`.
