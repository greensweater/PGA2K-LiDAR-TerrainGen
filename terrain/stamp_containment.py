"""
terrain/stamp_containment.py

Exact (non-discretized) geometric containment tests for a Stamp's own
footprint against an arbitrary shapely region -- used by PGA2k_gen.py's
_flag_previous_layer_blocked to decide whether a stamp from the layer
directly beneath a new masked generate/refine pass sits entirely inside
that pass's coverage area (a caller-supplied inward safety margin is
expected to already be baked into `region`, e.g.
mask_geometry.buffer(-margin_m) -- this module does no margin math of
its own).

Deliberately NOT a discretized/rasterized test. The removed
terrain/stamp_pruning.py (see CLAUDE.md's "Non-obvious patterns" for
the postmortem) proved "fully overwritten" via a rasterized brush-
weight sweep, whose grid choice depends on the whole current stamp
population -- an incremental/cached variant of that could disagree
with a fresh sweep on dense stamp packings (a real grid-phase aliasing
bug, not a hypothetical one). Plain shapely polygon containment is
exact algebra with no such resolution/phase dependency.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import shapely.vectorized
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from terrain.brush_profiles import BRUSH_PROFILES, SHAPE_SQUARE
from terrain.stamp import Stamp


def _rotated_rect_polygon(stamp: Stamp) -> Polygon:
    """
    Exact rotated-rectangle footprint for a SHAPE_SQUARE stamp -- the
    same perp/across construction as viz/visualize.py's
    _make_stamp_patch. Kept identical to that copy deliberately: see
    terrain/stamp.py's local_square_offsets docstring for a real past
    bug from two independently-reasoned copies of this exact rotation
    convention disagreeing.
    """
    theta = math.radians(stamp.rotation)
    perp = (math.sin(theta), math.cos(theta))     # "length" (scale_z) direction
    across = (math.cos(theta), -math.sin(theta))  # "width" (scale_x) direction
    half_len = stamp.scale_z
    half_wid = stamp.scale_x
    near = (stamp.x - perp[0] * half_len, stamp.z - perp[1] * half_len)
    far = (stamp.x + perp[0] * half_len, stamp.z + perp[1] * half_len)
    corners = [
        (near[0] - across[0] * half_wid, near[1] - across[1] * half_wid),
        (near[0] + across[0] * half_wid, near[1] + across[1] * half_wid),
        (far[0] + across[0] * half_wid, far[1] + across[1] * half_wid),
        (far[0] - across[0] * half_wid, far[1] - across[1] * half_wid),
    ]
    return Polygon(corners)


def stamps_fully_within(stamps: Sequence[Stamp], region: BaseGeometry | None) -> list[bool]:
    """
    One bool per stamp (same order as `stamps`): True iff that stamp's
    ENTIRE footprint -- accounting for brush shape and rotation -- lies
    within `region`.

    Circular brushes (the common case): grouped by reach (scale_x,
    rounded to avoid a fresh erosion per float-noise-distinct value),
    tested via the erode-then-test-center idiom
    course_output/object_clusters.py's circle-packing already uses for
    exact "whole circle fits inside this geometry" containment --
    region.buffer(-r) once per distinct r, then a single vectorized
    shapely.vectorized.contains(...) call per group (same vectorized
    point-in-polygon idiom terrain/hexgrid.py and terrain/rastergrid.py
    already use for mask containment).

    Square/rotated brushes (types 72/74): no cheap radius-erosion
    equivalent exists for a rectangle, so each gets an exact per-stamp
    rotated-rectangle Polygon tested via region.contains(rect) -- one
    shapely call per square stamp.
    """
    n = len(stamps)
    if region is None or region.is_empty or n == 0:
        return [False] * n

    result = [False] * n
    circular_idx: list[int] = []
    square_idx: list[int] = []
    for i, s in enumerate(stamps):
        profile = BRUSH_PROFILES.get(s.brush)
        if profile is not None and profile.shape == SHAPE_SQUARE:
            square_idx.append(i)
        else:
            circular_idx.append(i)

    if circular_idx:
        radii = np.array([round(stamps[i].scale_x, 6) for i in circular_idx])
        for r in np.unique(radii):
            group = [i for i, rr in zip(circular_idx, radii) if rr == r]
            eroded = region.buffer(-float(r))
            if eroded.is_empty:
                continue
            xs = np.array([stamps[i].x for i in group])
            zs = np.array([stamps[i].z for i in group])
            inside = shapely.vectorized.contains(eroded, xs, zs)
            for i, ok in zip(group, inside):
                result[i] = bool(ok)

    for i in square_idx:
        rect = _rotated_rect_polygon(stamps[i])
        result[i] = region.contains(rect)

    return result
