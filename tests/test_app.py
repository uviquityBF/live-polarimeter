import json
import os
import time

import numpy as np
import pytest

from livepol import camera as cam_mod
from livepol import overlap
from livepol.config import load_config
from livepol.engine import Engine
from livepol.logger import COLUMNS
from livepol.main import main as cli_main
from livepol.stokes import break_wraps

SMALL = {"simulate": {"shape": [128, 160], "sigma_px": 12, "noise": 1.0, "fps": 60.0,
                      "wobble_deg": 0.0, "dolp": 0.74, "aolp_deg": 174.3}}


def wait_for(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def make_engine(tmp_path, extra=None):
    cfg = load_config(overrides={**SMALL, "output": {"dir": str(tmp_path)}, **(extra or {})})
    eng = Engine(cam_mod.make_camera(cfg, simulate=True), cfg)
    eng.start()
    return eng


def test_decode_12p_roundtrip():
    vals = np.array([0, 4095, 1234, 77, 2048, 5], dtype=np.uint16)
    packed = np.zeros(len(vals) * 3 // 2, dtype=np.uint8)
    for k in range(0, len(vals), 2):
        a, b = int(vals[k]), int(vals[k + 1])
        packed[3 * k // 2: 3 * k // 2 + 3] = [a & 0xFF, ((a >> 8) & 0xF) | ((b & 0xF) << 4), b >> 4]
    out = cam_mod.decode_buffer(packed, 3, 2, "PolarizeMono12p")
    assert out.ravel().tolist() == vals.tolist()


def test_decode_16bit_and_8bit():
    a16 = np.arange(12, dtype="<u2").reshape(3, 4)
    assert np.array_equal(cam_mod.decode_buffer(np.frombuffer(a16.tobytes(), np.uint8), 4, 3), a16)
    a8 = np.arange(12, dtype=np.uint8)
    assert cam_mod.decode_buffer(a8, 4, 3).shape == (3, 4)
    with pytest.raises(ValueError):
        cam_mod.decode_buffer(a8[:5], 4, 3)


def test_full_scale_from_format():
    assert cam_mod.full_scale_from_format("PolarizeMono12") == 4095
    assert cam_mod.full_scale_from_format("PolarizeMono16") == 65535


def test_break_wraps():
    t, a = break_wraps([0, 1, 2, 3], [178, 179, 1, 2])
    assert len(t) == len(a) == 5 and np.isnan(a).sum() == 1 and np.isfinite(t).all()


def test_average_state_circular():
    s = overlap.average_state([0.9, 0.9], [179.0, 1.0], [10, 10])
    assert s["n"] == 2 and min(s["aolp_deg"], 180 - s["aolp_deg"]) < 1e-6
    assert overlap.average_state([], [])["n"] == 0


def test_simulated_engine_end_to_end(tmp_path):
    eng = make_engine(tmp_path)
    try:
        assert wait_for(lambda: eng.stats()["processed"] >= 5), eng.error
        r = eng.latest()
        assert r.dolp == pytest.approx(0.74, abs=0.02)
        d = (r.aolp - 174.3 + 90) % 180 - 90
        assert abs(d) < 1.0
        assert not r.saturated
        h = eng.history(10)
        assert h["t"].size >= 5
        log = eng.start_log(note="bench A\nsecond line")
        assert wait_for(lambda: eng.logging_rows >= 3)
        eng.stop_log()
        text = log.read_text().splitlines()
        assert text[0].startswith("# livepol") and any("note: bench A" in ln for ln in text)
        assert ",".join(COLUMNS) in text
        assert json.loads(log.with_suffix(".settings.json").read_text())["note"].startswith("bench A")
        raw = eng.save_raw()
        assert np.load(raw).shape == (128, 160) and raw.with_suffix(".settings.json").exists()
    finally:
        eng.stop()


def test_saturation_via_exposure_and_dark(tmp_path):
    eng = make_engine(tmp_path)
    try:
        assert wait_for(lambda: eng.latest() is not None), eng.error
        eng.set_exposure_us(40000)
        assert wait_for(lambda: eng.latest().saturated)
        eng.set_exposure_us(5000)
        done = []
        eng.capture_dark(5, on_done=done.append)
        assert wait_for(lambda: bool(done)), eng.message
        assert eng.processor.dark is not None and eng.dark_file
    finally:
        eng.stop()


def test_pause_and_manual_roi(tmp_path):
    eng = make_engine(tmp_path)
    try:
        assert wait_for(lambda: eng.latest() is not None)
        eng.set_roi({"mode": "rect", "rect": [30, 20, 10, 10], "circle": [0, 0, 5], "percentile": 99.5})
        assert wait_for(lambda: eng.latest().roi_mode == "rect")
        assert eng.latest().n_px == 100
        eng.set_paused(True)
        time.sleep(0.2)
        n = eng.stats()["processed"]
        time.sleep(0.3)
        assert eng.stats()["processed"] == n
    finally:
        eng.stop()


def test_replay_camera(tmp_path):
    from livepol.simulate import make_mosaic
    f = tmp_path / "f.npy"
    np.save(f, make_mosaic((64, 64), 0.5, 40.0, sigma_px=8))
    cfg = load_config(overrides={"output": {"dir": str(tmp_path)}, "replay": {"fps": 100.0}})
    eng = Engine(cam_mod.make_camera(cfg, replay=str(f)), cfg)
    eng.start()
    try:
        assert wait_for(lambda: eng.latest() is not None), eng.error
        assert eng.latest().dolp == pytest.approx(0.5, abs=0.02)
    finally:
        eng.stop()


def test_cli_headless_simulate(tmp_path):
    csv = tmp_path / "run.csv"
    rc = cli_main(["--simulate", "--no-gui", "--duration", "1.5", "--log", str(csv),
                   "--note", "cli test", "--sim-shape", "128", "160", "--sim-noise", "1",
                   "--out", str(tmp_path)])
    assert rc == 0
    rows = [ln for ln in csv.read_text().splitlines() if not ln.startswith("#")]
    assert len(rows) >= 3 and rows[0].startswith("timestamp")


def test_cli_no_camera_message(capsys):
    rc = cli_main(["--no-gui"])  # arena_api is not installed on the dev machine
    err = capsys.readouterr().err
    assert rc == 1 and "arena_api" in err or "camera" in err.lower()


def test_gui_offscreen(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets

    from livepol.gui import MainWindow, title_text
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    eng = make_engine(tmp_path)
    try:
        win = MainWindow(eng, eng.cfg)
        win.show()
        assert wait_for(lambda: eng.stats()["processed"] >= 5), eng.error
        for _ in range(5):
            app.processEvents()
            win._tick()
            time.sleep(0.05)
        assert "DoLP" in win.title.text() and "S3 not measurable" in win.title.text()
        win._store_ref("A")
        win._store_ref("B")
        assert win.last_overlap and win.last_overlap["eta"] == pytest.approx(1 - 0.0, abs=0.35)
        win.roi_mode.setCurrentText("rect")
        win.btn_log.setChecked(True)
        assert wait_for(lambda: eng.logging_rows >= 1)
        win.btn_log.setChecked(False)
        stem = win.snapshot()
        assert stem.with_suffix(".png").stat().st_size > 1000 and stem.with_suffix(".json").exists()
        assert "ROI" in title_text(eng.latest())
        win.timer.stop()
    finally:
        eng.stop()
