"""
terrain/stamp.py

The Stamp primitive: a single PGA terrain stamp placement.

`tool` selects the update rule (see terrain_model.py):
    0 = flatten -- pulls nearby terrain toward `value` (an absolute
        height), weighted by the brush's falloff. `value` is the
        absolute target the editor would have computed at placement
        time (cursor height + typed delta), not a relative offset.
    1 = raise -- adds `value` (a delta, not an absolute height) scaled
        by the brush's falloff, preserving whatever relief already
        exists in the affected area rather than overriding it with one
        flat target. Better suited to correcting a uniform bias over a
        region that's already the right shape but offset high or low
        (see terrain/adaptive_refine.py).

Both tools write into the same "height" node (Landscape mode) --
tool is a per-stamp property, not a different output destination.
Sculpt-mode stamps (a separate "terrainHeight" node) aren't used by
this compiler at all.

Optimization operates only on Stamp objects (see terrain_model.py /
optimizer.py). JSON is produced only by writer.py -- never here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TOOL_FLATTEN = 0
TOOL_RAISE = 1


@dataclass(slots=True)
class Stamp:
    """
    A single PGA terrain stamp in world coordinates.

    scale_x/scale_z: independent per-axis center-to-edge half-extent
    (world meters), replacing a single `radius`. For a circular brush
    the two must be equal (there's no other sensible reading of a
    single-radius falloff curve); for a square/rectangular brush
    (SHAPE_SQUARE -- currently just type 72) they may differ, giving an
    axis-aligned rectangle rather than a square. See terrain_model.py
    for how each shape interprets (scale_x, scale_z).
    """

    x: float
    z: float

    scale_x: float
    scale_z: float
    value: float

    brush: int

    rotation: float = 0.0
    tool: int = TOOL_FLATTEN


def local_square_offsets(stamp: Stamp, dx, dz):
    """
    Rotate a world-space (dx, dz) offset from `stamp`'s center into the
    stamp's own local frame -- `across` along the scale_x axis, `along`
    along the scale_z axis -- so a SHAPE_SQUARE stamp's per-axis
    Chebyshev reach test still applies correctly when rotation != 0.
    Identity (returns dx, dz unchanged) when rotation == 0, so every
    existing axis-aligned call site is unaffected.

    dx/dz may be Python floats or numpy arrays (evaluate() vs.
    evaluate_many()/render()'s vectorized callers).

    Convention matches fallline_fill_viz.py's build_interior_fill_stamps/
    _stamp_corners: rotation = degrees(atan2(perp_x, perp_z)), where perp
    (the scale_z "along" direction) is (sin theta, cos theta) and across
    (the scale_x direction) is (cos theta, -sin theta). Kept as the one
    shared implementation of this formula -- fallline_fill_viz.py's own
    docstring records a real bug from two independently-reasoned-but-
    unverified copies of this exact rotation convention disagreeing.
    """
    if stamp.rotation == 0.0:
        return dx, dz
    theta = math.radians(stamp.rotation)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    across = dx * cos_t - dz * sin_t
    along = dx * sin_t + dz * cos_t
    return across, along
