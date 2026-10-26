"""Overlap of two linear polarization states (interference visibility)."""

from __future__ import annotations

import math


def angle_difference(aolp_a_deg: float, aolp_b_deg: float) -> float:
    """AoLP_A - AoLP_B wrapped to [-90, +90) degrees (AoLP has period 180)."""
    return (aolp_a_deg - aolp_b_deg + 90.0) % 180.0 - 90.0


def overlap_eta(dolp_a: float, dolp_b: float, delta_deg: float) -> float:
    """eta = 0.5 * (1 + d_A * d_B * cos(2*delta)); DoLP values as fractions."""
    return 0.5 * (1.0 + dolp_a * dolp_b * math.cos(math.radians(2.0 * delta_deg)))


def overlap_loss_db(eta: float) -> float:
    """Overlap loss in dB, -10 log10(eta). Returns inf if eta <= 0."""
    return -10.0 * math.log10(eta) if eta > 0 else math.inf


def average_state(dolp, aolp_deg, s0=None) -> dict:
    """Average a run of samples into one reference state (arithmetic DoLP/S0, circular AoLP)."""
    import numpy as np

    from .stokes import circular_mean_aolp, circular_std_aolp

    d = np.asarray(dolp, dtype=float)
    a = np.asarray(aolp_deg, dtype=float)
    ok = np.isfinite(d) & np.isfinite(a)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "dolp": float("nan"), "aolp_deg": float("nan"),
                "dolp_std": float("nan"), "aolp_std_deg": float("nan"), "s0_mean": float("nan")}
    s = float(np.mean(np.asarray(s0, dtype=float)[ok])) if s0 is not None else float("nan")
    return {"n": n, "dolp": float(d[ok].mean()), "aolp_deg": circular_mean_aolp(a[ok]),
            "dolp_std": float(d[ok].std()), "aolp_std_deg": circular_std_aolp(a[ok]), "s0_mean": s}


def compare_states(aolp_a: float, dolp_a: float, aolp_b: float, dolp_b: float) -> dict:
    """delta (deg), eta and loss (dB) for a reference pair."""
    delta = angle_difference(aolp_a, aolp_b)
    eta = overlap_eta(dolp_a, dolp_b, delta)
    return {"delta_deg": delta, "eta": eta, "loss_db": overlap_loss_db(eta)}
