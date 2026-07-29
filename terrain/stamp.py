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

from dataclasses import dataclass

TOOL_FLATTEN = 0
TOOL_RAISE = 1


@dataclass(slots=True)
class Stamp:
    """A single PGA terrain stamp in world coordinates."""

    x: float
    z: float

    radius: float
    value: float

    brush: int

    rotation: float = 0.0
    tool: int = TOOL_FLATTEN
