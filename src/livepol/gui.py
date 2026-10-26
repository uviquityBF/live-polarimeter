"""Live display (pyqtgraph + Qt). Reads results from the Engine; never touches the camera."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import overlap
from .engine import Engine
from .pipeline import Result
from .stokes import break_wraps, circular_std_aolp

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

BIG = "font-size: 26px; font-weight: bold;"


def title_text(r: Result) -> str:
    if r.roi_mode == "auto" and r.threshold is not None:
        roi = f"auto ROI (S0 >= {r.percentile:g}th pct = {r.threshold:.1f} counts, {r.n_px} px)"
    else:
        roi = f"{r.roi_mode} ROI ({r.n_px} px)"
    return (f"Representative state -- {roi}  DoLP = {r.dolp * 100:.1f} %  AoLP = {r.aolp:.1f} deg  "
            "(S3 not measurable with this sensor)")


def mask_edge_rgba(mask: np.ndarray) -> np.ndarray:
    """Red RGBA outline of a boolean mask (mask pixels with a non-mask 4-neighbour)."""
    m = mask
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    edge = m & ~inner
    rgba = np.zeros(m.shape + (4,), dtype=np.uint8)
    rgba[edge] = (255, 40, 40, 255)
    return rgba


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: Engine, cfg: dict):
        super().__init__()
        self.engine = engine
        self.cfg = cfg
        self.setWindowTitle("livepol - live polarimeter")
        self.refs = {"A": None, "B": None}
        self._last_index = -1
        self._last_roi_mode = None
        self.last_overlap = None

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)

        self.title = QtWidgets.QLabel("waiting for frames...")
        self.title.setWordWrap(True)
        v.addWidget(self.title)

        self.readout = QtWidgets.QLabel()
        self.readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.readout.setStyleSheet(BIG)
        v.addWidget(self.readout)
        self.warning = QtWidgets.QLabel()
        self.warning.setStyleSheet("color: #d22; font-weight: bold;")
        self.warning.setWordWrap(True)
        v.addWidget(self.warning)

        self._build_top(v)
        self._build_timeseries(v)
        self._build_controls(v)
        self.status = self.statusBar()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(int(1000 / max(1.0, float(cfg["display"]["update_hz"]))))
        self.resize(1400, 950)

    # -- layout ------------------------------------------------------------
    def _build_top(self, v):
        glw = pg.GraphicsLayoutWidget()
        self.vb = glw.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("viridis"))
        self.edge = pg.ImageItem()
        self.vb.addItem(self.img)
        self.vb.addItem(self.edge)
        roi_cfg = self.engine.processor.roi_cfg
        x, y, w, h = roi_cfg["rect"]
        cx, cy, r = roi_cfg["circle"]
        self.rect_roi = pg.RectROI([x, y], [w, h], pen=pg.mkPen("y", width=2))
        self.circ_roi = pg.CircleROI([cx - r, cy - r], [2 * r, 2 * r], pen=pg.mkPen("y", width=2))
        for item in (self.rect_roi, self.circ_roi):
            item.setVisible(False)
            item.sigRegionChangeFinished.connect(self._roi_moved)
            self.vb.addItem(item)

        self.dia = glw.addPlot(row=0, col=1, title="Polarization (relative amplitude)")
        self.dia.setAspectLocked(True)
        self.dia.showGrid(x=True, y=True, alpha=0.3)
        self.dia.setLabel("bottom", "E_x")
        self.dia.setLabel("left", "E_y")
        self.dia_circle = self.dia.plot(pen=pg.mkPen("w", width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.dia_line = self.dia.plot(pen=pg.mkPen("r", width=3))
        glw.ci.layout.setColumnStretchFactor(0, 3)
        glw.ci.layout.setColumnStretchFactor(1, 2)
        glw.setMinimumHeight(380)
        v.addWidget(glw, stretch=3)

    def _build_timeseries(self, v):
        glw = pg.GraphicsLayoutWidget()
        self.ts = {}
        for i, (key, label) in enumerate((("dolp", "DoLP (%)"), ("aolp", "AoLP (deg)"),
                                          ("s0", "S0 mean (counts)"))):
            p = glw.addPlot(row=0, col=i)
            p.setLabel("left", label)
            p.setLabel("bottom", "time (s)")
            p.showGrid(x=True, y=True, alpha=0.3)
            c = p.plot(pen=pg.mkPen("c", width=1.5))
            self.ts[key] = (p, c, label)
        self.ts["aolp"][0].setYRange(0, 180)
        glw.setMinimumHeight(200)
        v.addWidget(glw, stretch=2)

    def _build_controls(self, v):
        B = QtWidgets.QPushButton
        row1 = QtWidgets.QHBoxLayout()
        self.btn_pause = B("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.toggled.connect(self._pause)
        btn_snap, btn_raw, btn_dark = B("Snapshot"), B("Save raw frame"), B("Capture dark")
        btn_snap.clicked.connect(self.snapshot)
        btn_raw.clicked.connect(self._save_raw)
        btn_dark.clicked.connect(lambda: self.engine.capture_dark(20))
        self.btn_log = B("Start log")
        self.btn_log.setCheckable(True)
        self.btn_log.toggled.connect(self._toggle_log)
        self.note = QtWidgets.QLineEdit()
        self.note.setPlaceholderText("note for the log header (bench setup, fiber, ...)")
        for w in (self.btn_pause, btn_snap, btn_raw, btn_dark, self.btn_log):
            row1.addWidget(w)
        row1.addWidget(self.note, stretch=1)
        v.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        cam = self.cfg["camera"]
        self.exp = QtWidgets.QDoubleSpinBox()
        self.exp.setRange(1, 1e7)
        self.exp.setDecimals(0)
        self.exp.setSuffix(" us")
        self.exp.setValue(float(cam["exposure_us"]))
        self.exp.editingFinished.connect(lambda: self.engine.set_exposure_us(self.exp.value()))
        self.gain = QtWidgets.QDoubleSpinBox()
        self.gain.setRange(0, 48)
        self.gain.setSuffix(" dB")
        self.gain.setValue(float(cam["gain_db"]))
        self.gain.editingFinished.connect(lambda: self.engine.set_gain_db(self.gain.value()))
        self.navg = QtWidgets.QSpinBox()
        self.navg.setRange(1, 200)
        self.navg.setValue(int(cam.get("average_frames", 1)))
        self.navg.editingFinished.connect(lambda: self.engine.set_average_frames(self.navg.value()))
        self.roi_mode = QtWidgets.QComboBox()
        self.roi_mode.addItems(["auto", "rect", "circle"])
        self.roi_mode.setCurrentText(self.engine.processor.roi_cfg.get("mode", "auto"))
        self.roi_mode.currentTextChanged.connect(self._roi_mode_changed)
        self.pct = QtWidgets.QDoubleSpinBox()
        self.pct.setRange(0, 100)
        self.pct.setDecimals(2)
        self.pct.setValue(float(self.engine.processor.roi_cfg.get("percentile", 99.5)))
        self.pct.editingFinished.connect(self._roi_moved)
        self.hist = QtWidgets.QSpinBox()
        self.hist.setRange(5, 1800)
        self.hist.setSuffix(" s")
        self.hist.setValue(int(self.cfg["display"]["history_s"]))
        for lab, w in (("Exposure", self.exp), ("Gain", self.gain), ("Avg N", self.navg),
                       ("ROI", self.roi_mode), ("pct", self.pct), ("History", self.hist)):
            row2.addWidget(QtWidgets.QLabel(lab))
            row2.addWidget(w)
        row2.addStretch(1)
        v.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        self.ref_s = QtWidgets.QDoubleSpinBox()
        self.ref_s.setRange(0.5, 600)
        self.ref_s.setValue(5.0)
        self.ref_s.setSuffix(" s")
        btn_a, btn_b, btn_c = B("Store A"), B("Store B"), B("Clear A/B")
        btn_a.clicked.connect(lambda: self._store_ref("A"))
        btn_b.clicked.connect(lambda: self._store_ref("B"))
        btn_c.clicked.connect(self._clear_refs)
        row3.addWidget(QtWidgets.QLabel("Reference = mean of last"))
        row3.addWidget(self.ref_s)
        for w in (btn_a, btn_b, btn_c):
            row3.addWidget(w)
        self.ref_label = QtWidgets.QLabel("A: -    B: -")
        row3.addWidget(self.ref_label, stretch=1)
        v.addLayout(row3)

    # -- control callbacks -------------------------------------------------
    def _pause(self, on: bool) -> None:
        self.engine.set_paused(on)
        self.btn_pause.setText("Resume" if on else "Pause")

    def _toggle_log(self, on: bool) -> None:
        if on:
            try:
                path = self.engine.start_log(note=self.note.text())
            except Exception as e:
                self.btn_log.setChecked(False)
                self.status.showMessage(f"cannot start log: {e}", 8000)
                return
            self.btn_log.setText("Stop log")
            self.status.showMessage(f"logging to {path}", 8000)
        else:
            self.engine.stop_log()
            self.btn_log.setText("Start log")

    def _save_raw(self) -> None:
        try:
            self.status.showMessage(f"saved {self.engine.save_raw(note=self.note.text())}", 8000)
        except Exception as e:
            self.status.showMessage(f"cannot save raw frame: {e}", 8000)

    def _roi_mode_changed(self, mode: str) -> None:
        self.rect_roi.setVisible(mode == "rect")
        self.circ_roi.setVisible(mode == "circle")
        self._roi_moved()

    def _roi_moved(self, *_args) -> None:
        mode = self.roi_mode.currentText()
        cfg = dict(self.engine.processor.roi_cfg)
        cfg["mode"] = mode
        cfg["percentile"] = self.pct.value()
        p, s = self.rect_roi.pos(), self.rect_roi.size()
        cfg["rect"] = [int(p.x()), int(p.y()), int(s.x()), int(s.y())]
        p, s = self.circ_roi.pos(), self.circ_roi.size()
        cfg["circle"] = [p.x() + s.x() / 2, p.y() + s.y() / 2, s.x() / 2]
        self.engine.set_roi(cfg)

    def _store_ref(self, name: str) -> None:
        h = self.engine.history(self.ref_s.value())
        st = overlap.average_state(h["dolp"], h["aolp"], h["s0"])
        if st["n"] == 0:
            self.status.showMessage("no data to store as reference yet", 5000)
            return
        self.refs[name] = st
        self._update_refs()

    def _clear_refs(self) -> None:
        self.refs = {"A": None, "B": None}
        self.last_overlap = None
        self._update_refs()

    def _update_refs(self) -> None:
        fmt = lambda s: "-" if s is None else f"{s['dolp'] * 100:.1f}% / {s['aolp_deg']:.2f} deg"  # noqa: E731
        txt = f"A: {fmt(self.refs['A'])}    B: {fmt(self.refs['B'])}"
        a, b = self.refs["A"], self.refs["B"]
        self.last_overlap = None
        if a and b:
            o = overlap.compare_states(a["aolp_deg"], a["dolp"], b["aolp_deg"], b["dolp"])
            self.last_overlap = o
            txt += (f"    delta = {o['delta_deg']:+.2f} deg    eta = {o['eta']:.4f}    "
                    f"overlap loss = {o['loss_db']:.3f} dB")
        self.ref_label.setText(txt)

    # -- snapshot ----------------------------------------------------------
    def snapshot(self):
        out = self.engine.out_dir
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}"
        self.grab().save(str(stem.with_suffix(".png")))
        r = self.engine.latest()
        state = {"note": self.note.text(), "result": r.scalars() if r else None,
                 "references": self.refs, "overlap": self.last_overlap,
                 "settings": self.engine.settings_dict(self.note.text())}
        stem.with_suffix(".json").write_text(json.dumps(state, indent=2, default=str))
        self.status.showMessage(f"snapshot saved: {stem}.png / .json", 8000)
        return stem

    # -- periodic update ---------------------------------------------------
    def _tick(self) -> None:
        eng = self.engine
        if eng.error:
            self.status.showMessage(eng.error)
            self.warning.setText(eng.error)
        r = eng.latest()
        if r is None or r.frame_index == self._last_index:
            self._update_status()
            return
        self._last_index = r.frame_index
        self._draw(r)
        self._update_status()

    def _update_status(self) -> None:
        st = self.engine.stats()
        msg = (f"{st['fps']:.1f} fps processed | grabbed {st['grabbed']} | "
               f"dropped {st['dropped']} | exposure {self.engine.camera.exposure_us:.0f} us")
        if self.engine.logging_path:
            msg += f" | logging {self.engine.logging_rows} rows"
        if self.engine.message:
            msg += f" | {self.engine.message}"
        if not self.engine.error:
            self.status.showMessage(msg)

    def _draw(self, r: Result) -> None:
        self.title.setText(title_text(r))
        dtxt = "n/a" if math.isnan(r.dolp) else f"{r.dolp * 100:.1f} %"
        atxt = "n/a" if math.isnan(r.aolp) else f"{r.aolp:.1f}&deg;"
        self.readout.setText(
            f"DoLP {dtxt} &nbsp;&nbsp; AoLP {atxt} &nbsp;&nbsp; S0 {r.s0_mean:.1f} counts "
            f"&nbsp;&nbsp; ROI {r.n_px} px &nbsp;&nbsp; "
            + ("<span style='color:#d22'>SATURATED</span>" if r.saturated else "not saturated"))
        warn = []
        if r.saturated:
            scale = 0.8 / r.max_frac if r.max_frac > 0 else 1
            warn.append(f"Saturation: ROI peak is {r.max_frac * 100:.0f}% of full scale. "
                        f"Try exposure x{scale:.2f} ({self.engine.camera.exposure_us * scale:.0f} us).")
        if r.dolp_over_100:
            warn.append("DoLP above 100%: noise or channel imbalance (value shown unclipped).")
        if self.engine.processor.dark is None:
            warn.append("No dark frame loaded.")
        self.warning.setText("  ".join(warn))

        lvl = float(max(np.percentile(r.s0, 99.9), 1.0))
        self.img.setImage(r.s0, autoLevels=False, levels=(0, lvl))
        self.edge.setImage(mask_edge_rgba(r.mask))
        if r.roi_mode != self._last_roi_mode:
            self._last_roi_mode = r.roi_mode

        # polarization diagram: dashed circle = total amplitude, red line = linear component
        amp = math.sqrt(max(r.s0_mean, 0.0))
        if amp > 0 and not math.isnan(r.dolp):
            th = np.linspace(0, 2 * np.pi, 120)
            self.dia_circle.setData(amp * np.cos(th), amp * np.sin(th))
            half = math.sqrt(max(r.dolp, 0.0) * max(r.s0_mean, 0.0))
            a = math.radians(r.aolp)
            self.dia_line.setData([-half * math.cos(a), half * math.cos(a)],
                                  [-half * math.sin(a), half * math.sin(a)])
            self.dia.setRange(xRange=(-1.15 * amp, 1.15 * amp), yRange=(-1.15 * amp, 1.15 * amp), padding=0)

        h = self.engine.history(float(self.hist.value()))
        if h["t"].size:
            x = h["t"] - h["t"][-1]
            self._plot("dolp", x, h["dolp"] * 100.0)
            xa, ya = break_wraps(x, h["aolp"])
            self._plot("aolp", xa, ya, stats_from=h["aolp"], circular=True)
            self._plot("s0", x, h["s0"])
            for p, _, _ in self.ts.values():
                p.setXRange(-float(self.hist.value()), 0, padding=0.01)

    def _plot(self, key, x, y, stats_from=None, circular=False) -> None:
        p, curve, label = self.ts[key]
        curve.setData(x, y, connect="finite")
        v = np.asarray(y if stats_from is None else stats_from, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            sd = circular_std_aolp(v) if circular else float(v.std())
            p.setTitle(f"{label}: min {v.min():.2f}  max {v.max():.2f}  std {sd:.3f}"
                       + (" (circ.)" if circular else ""))

    def closeEvent(self, e) -> None:
        self.timer.stop()
        self.engine.stop()
        super().closeEvent(e)


def run_gui(engine: Engine, cfg: dict) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow(engine, cfg)
    win.show()
    return app.exec()
