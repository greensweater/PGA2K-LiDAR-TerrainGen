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
    tool     -- stamp.tool (0=flatten, 1=raise). Both live in the same
                "height" (Landscape mode) array -- tool is a per-stamp
                property, not a different output destination; see
                terrain/stamp.py and terrain_model.py.
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

import numpy as np

from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, TOOL_RAISE, Stamp
from terrain.terrain_model import TerrainModel

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


REGISTRATION_MARK_INSET_M = 5.0  # distance from each course edge to a corner mark's center
REGISTRATION_MARK_STAMP_RADIUS_M = 2.0
REGISTRATION_MARK_STAMP_HEIGHT_M = 0.3  # subtle -- just enough to spot deliberately, not a real bump
REGISTRATION_MARK_TYPE_73_CIRCLE = 73
REGISTRATION_MARK_CIRCLE_RADIUS_M = 5.0  # radius of the matching spline circle (see splines.py)


def registration_mark_corners(
    course_size_m: float, inset_m: float = REGISTRATION_MARK_INSET_M,
) -> list[tuple[float, float]]:
    """
    4 corner positions, each inset by inset_m from both edges, in the
    course-local [0, course_size_m] frame -- shared by
    build_registration_mark_stamps here and splines.py's
    build_registration_mark_splines, so the terrain bump and the
    spline circle at each corner land at the exact same point.
    """
    return [
        (inset_m, inset_m),
        (course_size_m - inset_m, inset_m),
        (inset_m, course_size_m - inset_m),
        (course_size_m - inset_m, course_size_m - inset_m),
    ]


def build_registration_mark_stamps(course_size_m: float) -> list[Stamp]:
    """
    4 small, subtle type-73 (circle) RAISE stamps, one at each course
    corner -- opt-in (see Write Terrain's registration_marks flag), for
    visually confirming in-game that terrain and splines land exactly
    where expected, and that the game isn't scaling/repositioning
    either one unexpectedly. Paired with a same-position circle spline
    (see splines.py's build_registration_mark_splines) using cart path
    surface, so each corner shows both a small raised bump and a
    visible ring around it.

    Type 73 (not 8) specifically requested: circular falloff reads as
    a clean, symmetric bump, not 8's flat-topped plateau silhouette --
    matters here since the whole point is an unambiguous, deliberately
    placed landmark, not something that could be mistaken for a
    refinement artifact.
    """
    return [
        Stamp(
            x=x, z=z, radius=REGISTRATION_MARK_STAMP_RADIUS_M,
            value=REGISTRATION_MARK_STAMP_HEIGHT_M, brush=REGISTRATION_MARK_TYPE_73_CIRCLE,
            tool=TOOL_RAISE,
        )
        for x, z in registration_mark_corners(course_size_m)
    ]


def build_course_wide_stamp(bounds: BoundingBox, value: float, tool: int) -> Stamp:
    """
    A single type-72 (hard square) stamp sized to cover the whole
    course at full (1.0) weight everywhere -- stamps are fundamentally
    square bitmaps (circular brushes are just a circle inscribed in
    that square), so a square stamp's scale is literally its
    half-width: scale = course half-width gives a stamp exactly as
    wide as the course. The margin beyond that half-width needs to
    clear this brush's bevel (see terrain/brush_profiles.py's
    _hard_edge_profile -- an ESTIMATE, no real measurements exist for
    this brush yet), not just be "a bit more than zero": at the
    estimated 3% bevel, a small ~20 m margin still leaves the course
    corners inside the bevel getting partial weight, not the full 1.0
    the whole point of this stamp depends on. 100 m comfortably clears
    that estimate with room to spare in case the real bevel turns out
    wider.

    Shared by both the zero-height shift shim (raise, applied last)
    and the baseline-flatten stamp (flatten, applied first) -- same
    brush, same margin reasoning, only the tool/value/place-in-sequence differ.
    """
    half_width = max(bounds.max_x - bounds.min_x, bounds.max_z - bounds.min_z) / 2.0
    margin = 100.0
    return Stamp(
        x=(bounds.min_x + bounds.max_x) / 2.0,
        z=(bounds.min_z + bounds.max_z) / 2.0,
        radius=half_width + margin,
        value=value,
        brush=72,
        tool=tool,
    )


def build_baseline_flatten_stamp(bounds: BoundingBox, mean_elevation: float) -> Stamp:
    """
    A course-wide type-72 flatten stamp, meant to be placed FIRST in
    the stamp list (applied before every real, detailed stamp) --
    makes the model's starting point for its sequential fold explicit
    (the mean bare-earth elevation) rather than relying on the
    implicit "start folding from height=0 everywhere" TerrainModel
    otherwise begins from. Real terrain-shaping stamps still pull
    every point to wherever the actual LIDAR-fit elevation is
    regardless of this starting point -- this only matters where stamp
    coverage isn't complete (rare gaps in the hex grid/refinement
    passes), where the final blended height partly reflects this
    baseline. Mean elevation keeps those rare gaps close to correct
    terrain instead of collapsing toward an arbitrary constant like 0
    (up to ~200-400 m away from real course elevation, depending on
    the site).
    """
    return build_course_wide_stamp(bounds, value=mean_elevation, tool=TOOL_FLATTEN)


def normalize_stamp_heights(
    stamps: Sequence[Stamp],
    bounds: BoundingBox,
    resolution: int = 200,
) -> list[Stamp]:
    """
    Shift the terrain so its minimum height is exactly 0, matching
    PGA's 0-based height field convention.

    This evaluates the actual resolved terrain (TerrainModel.render()
    over `bounds`) to find the true min/max, rather than scanning raw
    stamp.value fields directly -- naive min/max over .value is wrong
    once raise-tool stamps exist: a raise stamp's value is a relative
    delta (e.g. -15 to nudge an area down a little), not an absolute
    height, so mixing deltas with flatten's absolute values produces a
    meaningless range (a real course hit exactly this: reported span
    408 m from a delta of -15.7 and an absolute value of 392.3, when
    the actual resolved terrain only spanned about 120 m).

    The shift is applied as one additional raise-tool stamp, appended
    after every existing stamp, with a radius many times the course
    size -- large enough that its weight is (exactly, for every brush
    profile's flat plateau) 1.0 across the whole course. Because it's
    applied last, "old_height + shift * 1.0" shifts whatever the
    already-fully-resolved height is at every point by exactly `shift`,
    regardless of the mix of flatten/raise stamps and partial blend
    weights that produced that height -- directly rescaling individual
    stamps' own values can't be done reliably once raise stamps and
    partial (non-1.0) weights are involved (a uniform shift to every
    flatten target does *not* generally produce a uniformly shifted
    result wherever a point isn't fully committed to some stamp's
    center), so this appended-stamp approach sidesteps that entirely.

    Raises rather than silently clipping if the resulting span would
    exceed MAX_INGAME_HEIGHT_M, since clipping would corrupt real
    terrain shape rather than just failing loudly.
    """
    if not stamps:
        return list(stamps)

    model = TerrainModel(stamps)
    heights = model.render(resolution=resolution, bounds=bounds)
    true_min = float(np.min(heights))
    true_max = float(np.max(heights))
    span = true_max - true_min

    if span > MAX_INGAME_HEIGHT_M:
        raise ValueError(
            f"Terrain relief ({span:.1f} m, from {true_min:.1f} to {true_max:.1f} m) "
            f"exceeds the known in-game height ceiling of {MAX_INGAME_HEIGHT_M} m even "
            "after shifting the minimum to 0. Writing this would require clipping real "
            "terrain shape -- consider a smaller/different crop rather than exporting "
            "as-is."
        )

    shift = -true_min
    if abs(shift) < 1e-9:
        return list(stamps)

    # A hard square (type 72) is a better fit here than the previous
    # huge-radius circular workaround: stamps are fundamentally square
    # bitmaps (circular brushes are just a circle inscribed in that
    # square), so a square stamp's scale is literally its half-width --
    # scale = course half-width gives a stamp exactly as wide as the
    # course. The margin beyond that half-width needs to clear this
    # brush's bevel (see terrain/brush_profiles.py's _hard_edge_profile
    # -- an ESTIMATE, no real measurements exist for this brush yet),
    # not just be "a bit more than zero": at the estimated 3% bevel, a
    # small ~20 m margin still leaves the course corners inside the
    # bevel getting partial weight, not the full 1.0 the whole point of
    # this stamp depends on. 100 m comfortably clears that estimate
    # with room to spare in case the real bevel turns out wider.
    shim = build_course_wide_stamp(bounds, value=shift, tool=TOOL_RAISE)
    return list(stamps) + [shim]


def stamp_to_entry(stamp: Stamp) -> dict:
    """Convert a single Stamp into one "height" array entry."""
    orientation = _round(stamp.rotation)
    radius = _round(stamp.radius)
    return {
        "tool": stamp.tool,
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
