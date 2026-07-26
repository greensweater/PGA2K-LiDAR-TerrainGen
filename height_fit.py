"""
terrain/height_fit.py

Milestone 3's "initial height fitting": sets each Stamp's value from
the average nearby LIDAR elevation, in place of hexgrid.py's
placeholder value=0.0.

This is deliberately naive -- each stamp's target is a simple, unweighted
average of nearby bare-earth points. The architecture doc calls out
weighted least-squares as the "eventually" version of this step; this
is the "for now" one.

Naive per-stamp averaging is enough on its own for the initial hex
grid specifically, because of a property of that layout: stamp radius
(200 m) equals the lattice's nearest-neighbor spacing exactly, so every
stamp's own center sits exactly at r=1.0 relative to its neighbors --
the brush profile anchor, weight 0 by construction (see
brush_profiles.py). No other stamp reaches a given stamp's own center,
so there's nothing to account for there.

That won't generally be true once adaptive refinement adds smaller,
more tightly-packed stamps later, where a stamp's own center *can* already
be partway pulled by an earlier, larger stamp. fit_stamp_heights()
handles that case correctly too: it reads whatever height prior stamps
in the list already produced at each stamp's position (current), and
solves the value that pulls the remaining distance to the target,
given that brush's own center weight:

    target = current + (value - current) * weight_at_center
    value  = current + (target - current) / weight_at_center

which reduces to value = target exactly whenever current = 0 and
weight_at_center = 1 -- true for every stamp in the initial hex grid,
so this general form costs nothing there and is correct everywhere
else too.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ingest.laz_reader import PointCloud
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.stamp import Stamp
from terrain.terrain_kernel import TerrainKernel
from terrain.terrain_model import TerrainModel


def fit_stamp_heights(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bare_earth_only: bool = True,
    min_points: int = 3,
) -> list[Stamp]:
    """
    Return a new list of Stamps with value replaced by a per-stamp
    best-fit height, estimated from nearby LIDAR points.

    Stamps are processed in list order, and each one's fit accounts for
    whatever height stamps earlier in the list already left at its
    position -- see module docstring. Processing order should match
    whatever order the stamps will ultimately be applied in (placement
    order), same as terrain_model.py's evaluate().

    Stamps with fewer than `min_points` nearby points are left
    unchanged (their placeholder value untouched) rather than guessing
    from sparse or absent data -- left for a later pass (denser
    sampling, interpolation, or adaptive refinement) to handle.
    """
    kernels: dict[int, TerrainKernel] = {}
    fitted: list[Stamp] = []

    for stamp in stamps:
        if stamp.brush not in kernels:
            if stamp.brush not in BRUSH_PROFILES:
                raise ValueError(f"No BrushProfile registered for brush type {stamp.brush}")
            kernels[stamp.brush] = TerrainKernel(BRUSH_PROFILES[stamp.brush])

        idx = cloud.query_radius(stamp.x, stamp.z, stamp.radius)
        if bare_earth_only and idx.size > 0:
            idx = idx[cloud.bare_earth_mask()[idx]]

        if idx.size < min_points:
            fitted.append(stamp)
            continue

        target = float(np.mean(cloud.elevation[idx]))

        weight_at_center = kernels[stamp.brush].sample(0.0)
        if weight_at_center <= 0.0:
            # No registered brush should hit this (every profile has a
            # nonzero center weight), but guard against div-by-zero
            # rather than producing an inf/nan value.
            fitted.append(stamp)
            continue

        current = TerrainModel(fitted).evaluate(stamp.x, stamp.z)
        value = current + (target - current) / weight_at_center
        fitted.append(replace(stamp, value=value))

    return fitted
