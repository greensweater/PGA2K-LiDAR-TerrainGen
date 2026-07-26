"""
writer.py

Writes a list of Stamps into userLayers.json, matching PGA's .course
schema. This is the only module that produces JSON -- optimization and
layout modules work only with Stamp objects (see "Important Design
Rules": JSON is written only by the writer module, never the optimizer).

userLayers.json is an OBJECT with many sibling keys (deletedHazards,
newHazards, objects, clearTrees, addTrees, treeDensity, hazards,
terrainHeight, height, trees, green, surfaces, outOfBounds,
crowdLocations, water) -- not a bare array. This compiler only
produces landscape-mode ("flatten", tool 0) stamps, which belong under
the "height" key; "terrainHeight" is where sculpt-mode stamps would go
(tied to a zero-height baseline), which we deliberately don't use (see
terrain_model.py). write_user_layers() therefore reads whatever's
already at the target path, replaces only "height", and leaves every
other key exactly as it found it -- falling back to an all-empty
schema if no file exists there yet. As more of the pipeline gets built
(trees, water, surfaces, ...), more keys will start getting generated
here instead of just passed through.

Field mapping from Stamp -> one entry in the "height" array:
    tool     -- fixed at 0 (flatten). This compiler only targets
                flatten, not raise (tool 1) -- see terrain_model.py.
    position -- {x, y: "-Infinity", z}. y is always the literal string
                "-Infinity" for landscape-mode stamps -- height comes
                from `value`, not position.y. x/z are shifted by
                GRID_ORIGIN_OFFSET: this compiler works in a local
                [0, COURSE_SIZE_M] frame, but PGA's own grid is centered
                on the origin ([-1000, 1000] for a 2000 m course), so
                x/z each have GRID_ORIGIN_OFFSET subtracted at write
                time -- nowhere else in the pipeline needs to know
                about PGA's centered frame.
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

NOTE on position sign convention: x/z are shifted by a plain
subtraction here, matching -X=west/+X=east/-Z=south/+Z=north with no
axis flip. This hasn't been confirmed against the in-game editor yet
-- if a stamp's compass direction in-game doesn't match its x/z sign
once viewed there, the fix is a reflection (e.g. z_out = -(stamp.z -
GRID_ORIGIN_OFFSET) instead of a plain subtraction), not a change to
GRID_ORIGIN_OFFSET itself.
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

# PGA's grid is centered on the origin; this compiler works in a local
# [0, COURSE_SIZE_M] frame. Subtracting this at write time converts
# between the two -- see module docstring's NOTE on sign convention.
GRID_ORIGIN_OFFSET = 1000.0

# Observed in-game ceiling for terrain height. PGA's height field is
# 0-based -- negative or excessively large values aren't representable,
# so every stamp's value must be shifted so the minimum lands at exactly
# 0 before writing, and the resulting span must fit under this ceiling.
MAX_INGAME_HEIGHT_M = 275.0

# Every top-level key userLayers.json is expected to have. Used as a
# fallback schema when writing to a path that doesn't exist yet -- in
# the normal course of things, the target file already exists (from an
# extracted blank .course) and its other keys are preserved as-is.
_BLANK_USER_LAYERS_SCHEMA = {
    "deletedHazards": [],
    "newHazards": [],
    "objects": [],
    "clearTrees": [],
    "addTrees": [],
    "treeDensity": [],
    "hazards": [],
    "terrainHeight": [],
    "height": [],
    "trees": [],
    "green": [],
    "surfaces": [],
    "outOfBounds": [],
    "crowdLocations": [],
    "water": [],
}

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
    """Convert a single Stamp into one "height" array entry."""
    orientation = _round(stamp.rotation)
    radius = _round(stamp.radius)
    return {
        "tool": TOOL_FLATTEN,
        "position": {
            "x": _round(stamp.x - GRID_ORIGIN_OFFSET),
            "y": POSITION_Y,
            "z": _round(stamp.z - GRID_ORIGIN_OFFSET),
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
    """
    Write `stamps` into the "height" key of the userLayers.json at
    `path`, preserving every other key already there.

    If `path` doesn't exist yet, falls back to an all-empty schema
    (_BLANK_USER_LAYERS_SCHEMA) rather than writing a bare array --
    userLayers.json is always the full object, never just our stamps.
    """
    path = Path(path)

    if path.exists():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict(_BLANK_USER_LAYERS_SCHEMA)

    data["height"] = [stamp_to_entry(s) for s in stamps]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
