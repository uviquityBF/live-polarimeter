import math

import numpy as np
import pytest

from livepol import overlap, roi, stokes
from livepol.simulate import make_mosaic

SHAPE = (256, 256)


def measure(mosaic, layout=stokes.DEFAULT_LAYOUT, pct=99.5):
    ch = stokes.split_mosaic(mosaic, layout)
    s0, s1, s2 = stokes.stokes_images(ch)
    mask, _ = roi.percentile_mask(s0, pct)
    a, b, c, n = stokes.roi_stokes(s0, s1, s2, mask)
    return stokes.dolp_aolp(a, b, c), n


@pytest.mark.parametrize("dolp", [0.0, 0.3, 0.738, 0.96, 1.0])
@pytest.mark.parametrize("aolp", [0.0, 0.1, 30.0, 89.9, 90.0, 135.0, 174.3, 179.9])
def test_known_states(dolp, aolp):
    m = make_mosaic(SHAPE, dolp, aolp, sigma_px=20, dtype=np.float64)
    (d, a), n = measure(m)
    assert n > 0
    assert d == pytest.approx(dolp, abs=0.002)
    if dolp > 0.01:  # AoLP undefined for unpolarized light
        diff = (a - aolp + 90) % 180 - 90  # wrap-aware comparison
        assert abs(diff) < 0.2


def test_unpolarized():
    m = make_mosaic(SHAPE, 0.0, 0.0, sigma_px=20, dtype=np.float64)
    (d, _), _ = measure(m)
    # Residual comes from the four polarizer pixels sampling the spot gradient at
    # slightly different positions (instantaneous-FOV effect), not from the maths.
    assert d < 0.002


def test_integer_quantization_ok():
    m = make_mosaic(SHAPE, 0.738, 174.3, sigma_px=20, dtype=np.uint16)
    (d, a), _ = measure(m)
    assert d == pytest.approx(0.738, abs=0.002)
    assert abs((a - 174.3 + 90) % 180 - 90) < 0.2


@pytest.mark.parametrize(
    "layout", [((90, 45), (135, 0)), ((0, 45), (135, 90)), ((45, 0), (90, 135))]
)
def test_layout_config(layout):
    m = make_mosaic(SHAPE, 0.8, 50.0, sigma_px=20, layout=layout, dtype=np.float64)
    (d, a), _ = measure(m, layout)
    assert d == pytest.approx(0.8, abs=0.002)
    assert a == pytest.approx(50.0, abs=0.2)


def test_wrong_layout_gives_wrong_answer():
    m = make_mosaic(SHAPE, 0.8, 50.0, sigma_px=20, layout=((90, 45), (135, 0)), dtype=np.float64)
    (_, a), _ = measure(m, ((0, 45), (135, 90)))
    assert abs(a - 50.0) > 1.0


def test_bad_layout_rejected():
    with pytest.raises(ValueError):
        stokes.validate_layout([[0, 0], [45, 90]])


def test_dark_and_gain_correction():
    m = make_mosaic(SHAPE, 0.9, 20.0, sigma_px=20, dtype=np.float64, background=50.0)
    ch = stokes.split_mosaic(m)
    dark = {a: np.full_like(ch[a], 50.0) for a in stokes.ANGLES}
    s0, s1, s2 = stokes.stokes_images(stokes.correct_channels(ch, dark=dark))
    mask, _ = roi.percentile_mask(s0)
    d, a = stokes.dolp_aolp(*stokes.roi_stokes(s0, s1, s2, mask)[:3])
    assert d == pytest.approx(0.9, abs=0.002)
    assert a == pytest.approx(20.0, abs=0.2)


def test_empty_roi():
    z = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    s0, s1, s2, n = stokes.roi_stokes(z, z, z, mask)
    assert n == 0
    d, a = stokes.dolp_aolp(s0, s1, s2)
    assert math.isnan(d) and math.isnan(a)


def test_dolp_over_100_not_clipped():
    d, _ = stokes.dolp_aolp(10.0, 10.5, 0.0)
    assert d == pytest.approx(1.05)


def test_aolp_offset():
    _, a = stokes.dolp_aolp(1.0, 1.0, 0.0, aolp_offset_deg=10.0)
    assert a == pytest.approx(170.0)


def test_circular_mean():
    assert stokes.circular_mean_aolp([179, 1]) == pytest.approx(0.0, abs=1e-9) or \
        stokes.circular_mean_aolp([179, 1]) == pytest.approx(180.0, abs=1e-9)
    assert stokes.circular_mean_aolp([170, 10]) == pytest.approx(0.0, abs=1e-9) or \
        stokes.circular_mean_aolp([170, 10]) == pytest.approx(180.0, abs=1e-9)
    assert stokes.circular_mean_aolp([10, 20]) == pytest.approx(15.0)
    assert math.isnan(stokes.circular_mean_aolp([float("nan")]))


def test_circular_std():
    assert stokes.circular_std_aolp([30, 30, 30]) == pytest.approx(0.0, abs=1e-3)
    wrapped = stokes.circular_std_aolp([179, 1, 179, 1])
    plain = stokes.circular_std_aolp([89, 91, 89, 91])
    assert wrapped == pytest.approx(plain, rel=1e-6)  # rotation invariance


def test_unwrap_for_plot_breaks_line():
    out = stokes.unwrap_aolp_for_plot([178, 179, 1, 2])
    assert np.isnan(out).sum() == 1 and len(out) == 5


def test_saturation_and_exposure_hint():
    m = make_mosaic(SHAPE, 0.9, 20.0, peak_s0=8000, sigma_px=20, dtype=np.uint16)
    ch = stokes.split_mosaic(m)
    s0, _, _ = stokes.stokes_images(ch)
    mask, _ = roi.percentile_mask(s0)
    sat, frac = roi.saturation_check(ch, mask, 4095)
    assert sat and frac >= 0.98
    assert roi.suggested_exposure_scale(frac) < 1.0
    m2 = make_mosaic(SHAPE, 0.9, 20.0, peak_s0=1000, sigma_px=20, dtype=np.uint16)
    ch2 = stokes.split_mosaic(m2)
    s0b, _, _ = stokes.stokes_images(ch2)
    mask2, _ = roi.percentile_mask(s0b)
    assert not roi.saturation_check(ch2, mask2, 4095)[0]


def test_roi_shapes():
    assert roi.rect_mask((10, 10), 2, 3, 4, 2).sum() == 8
    assert roi.circle_mask((21, 21), 10, 10, 3).sum() > 20
    s0 = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask, thr = roi.mask_from_config(s0, {"mode": "auto", "percentile": 90})
    assert mask.sum() == 10 and thr == pytest.approx(np.percentile(s0, 90))


def test_overlap_example():
    r = overlap.compare_states(30.0, 0.96, 22.0, 0.96)
    assert r["delta_deg"] == pytest.approx(8.0)
    assert r["eta"] == pytest.approx(0.943, abs=0.001)
    assert r["loss_db"] == pytest.approx(-10 * math.log10(r["eta"]))


def test_angle_difference_wrap():
    assert overlap.angle_difference(179, 1) == pytest.approx(-2.0)
    assert overlap.angle_difference(1, 179) == pytest.approx(2.0)
    assert overlap.angle_difference(90, 0) == pytest.approx(-90.0)
