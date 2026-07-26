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

Height normalization (shifting the minimum value to 0, required by
PGA's height field convention) is NOT automatic here -- see
normalize_stamp_heights() below. write_user_layers() writes exactly
the values it's given; callers must normalize first if needed, so the
shift amount stays visible rather than happening silently inside a
"just write the file" call.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from terrain.stamp import Stamp

TOOL_FLATTEN = 0
HOLE_ID_NONE = -1
UNUSED_RADIUS_FIELD = 0.0  # see module docstring: sizing is via `scale`
POSITION_Y = "-Infinity"

# Observed in-game ceiling for terrain height. PGA's height field is
# 0-based -- negative or excessively large values aren't representable,
# so every stamp's value must be shifted so the minimum lands at exactly
# 0 before writing, and the resulting span must fit under this ceiling.
MAX_INGAME_HEIGHT_M = 275.0

_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def normalize_stamp_heights(stamps: Sequence[Stamp]) -> list[Stamp]:
    """
    Shift every stamp's value so the minimum becomes exactly 0, matching
    PGA's 0-based height field convention.

    Only `value` is touched -- positions, radii, and brush types are
    untouched. Raises rather than silently clipping if the resulting
    span would exceed MAX_INGAME_HEIGHT_M, since clipping would corrupt
    real terrain shape rather than just failing loudly.
    """
    if not stamps:
        return list(stamps)

    min_value = min(s.value for s in stamps)
    max_value = max(s.value for s in stamps)
    span = max_value - min_value

    if span > MAX_INGAME_HEIGHT_M:
        raise ValueError(
            f"Terrain relief ({span:.1f} m, from {min_value:.1f} to {max_value:.1f} m) "
            f"exceeds the known in-game height ceiling of {MAX_INGAME_HEIGHT_M} m even "
            "after shifting the minimum to 0. Writing this would require clipping real "
            "terrain shape -- consider a smaller/different crop rather than exporting "
            "as-is."
        )

    return [replace(s, value=s.value - min_value) for s in stamps]


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
