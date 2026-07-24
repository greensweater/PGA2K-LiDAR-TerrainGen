"""
terrain/stamp.py

The Stamp primitive: a single PGA terrain brush placement.

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
    amplitude: float

    brush: int

    rotation: float = 0.0
