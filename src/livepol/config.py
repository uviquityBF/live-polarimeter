"""Configuration loading: defaults from config/default.yaml, user file, then overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def deep_merge(base: dict, new: dict) -> dict:
    """Return base updated recursively with new (neither argument is modified)."""
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Dict[str, Any]:
    """Load defaults, merge a user YAML file, then merge `overrides` (nested dict)."""
    if not DEFAULT_PATH.exists():
        raise FileNotFoundError(
            f"default config not found at {DEFAULT_PATH}; install with 'pip install -e .' "
            "from a checkout of the repository"
        )
    cfg = yaml.safe_load(DEFAULT_PATH.read_text()) or {}
    if path:
        user = yaml.safe_load(Path(path).read_text()) or {}
        cfg = deep_merge(cfg, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def load_calibration(path: Optional[str]) -> dict:
    """Load a calibration JSON ({"gain": {"0": g, ...}, "offset": {...}, "layout": [[..],[..]]}).

    Returns {"gain": {int: float}, "offset": {int: float}, "layout": list or None}.
    """
    if not path:
        return {"gain": None, "offset": None, "layout": None}
    data = json.loads(Path(path).read_text())
    conv = lambda d: None if d is None else {int(k): float(v) for k, v in d.items()}  # noqa: E731
    return {"gain": conv(data.get("gain")), "offset": conv(data.get("offset")), "layout": data.get("layout")}
