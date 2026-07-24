"""
terrain/bounding_box.py

A simple horizontal-plane bounding box, in local meters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BoundingBox:
    min_x: float
    min_z: float
    max_x: float
    max_z: float
