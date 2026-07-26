"""
writer.py

Writes a list of Stamps to userLayers.json, matching PGA's .course
schema. This is the only module that produces JSON -- optimization and
layout modules work only with Stamp objects (see "Important Design
Rules": JSON is written only by the writer module, never the optimizer).

Field mapping from Stamp -> userLayers.json entry:
    tool     -- fixed at 0 (flatten). This compiler only targets
                flatten, not raise (tool 1) -- see terrain_model.py.
    position -- {x: stamp.x, y: "-Infinity", z: stamp.z}. y is always
                the literal string "-Infinity" for landscape-mode
                stamps -- height comes from `value`, not position.y.
    rotation -- {x: 0, y: 0, z: 0}. Unused for terrain stamps; PGA
                tracks yaw separately via orientation/_orientation.
    orientation / _orientation -- both set to stamp.rotation (degrees).
                Duplicated because the schema carries the same value
                under two different keys.
    scale    -- {x: stamp.radius, y: 1.0, z: stamp.radius}. Scale, not
                the separate "radius" field, is what actually sizes a
                landscape stamp; radius stays 0.0 (see below).
    type     -- stamp.brush (the brush id, e.g. 8 / 9 / 10 / 54)
    value    -- stamp.value, the pull-toward-value target height
    holeId   -- fixed at -1 (these stamps aren't tied to a specific hole)
    radius   -- fixed at 0.0. Present in the schema but unused for
                landscape terrain stamps; real sizing is via `scale`.

All numeric output is rounded to 3 decimal places.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from terrain.stamp import Stamp

TOOL_FLATTEN = 0
HOLE_ID_NONE = -1
UNUSED_RADIUS_FIELD = 0.0  # see module docstring: sizing is via `scale`
POSITION_Y = "-Infinity"

_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def stamp_to_entry(stamp: Stamp) -> dict:
    """Convert a single Stamp into one userLayers.json entry."""
    orientation = _round(stamp.rotation)
    radius = _round(stamp.radius)
    return {
        "tool": TOOL_FLATTEN,
        "position": {
            "x": _round(stamp.x),
            "y": POSITION_Y,
            "z": _round(stamp.z),
        },
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "_orientation": orientation,
        "scale": {
            "x": radius,
            "y": 1.0,
            "z": radius,
        },
        "type": stamp.brush,
        "value": _round(stamp.value),
        "holeId": HOLE_ID_NONE,
        "radius": UNUSED_RADIUS_FIELD,
        "orientation": orientation,
    }


def write_user_layers(stamps: Sequence[Stamp], path: Path) -> None:
    """Write `stamps` to `path` as a userLayers.json entry list."""
    entries = [stamp_to_entry(s) for s in stamps]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
