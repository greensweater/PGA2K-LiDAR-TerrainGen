"""
terrain/stamp.py

The Stamp primitive: a single PGA terrain stamp placement.

Under the "flatten" tool (see terrain_model.py), a stamp doesn't add
height -- it pulls nearby terrain toward `value`, weighted by the
brush's falloff. `value` is the absolute target height the editor
would have computed at placement time (cursor height + typed delta),
not a relative offset.

Optimization operates only on Stamp objects (see terrain_model.py /
optimizer.py). JSON is produced only by writer.py -- never here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Stamp:
    """A single PGA terrain stamp in world coordinates."""

    x: float
    z: float

    radius: float
    value: float

    brush: int

    rotation: float = 0.0
