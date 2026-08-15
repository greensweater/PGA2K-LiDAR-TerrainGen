"""
terrain/height_fit.py

Milestone 3's "initial height fitting": sets each Stamp's value from
the average nearby ground elevation, in place of hexgrid.py's
placeholder value=0.0.

This is deliberately naive -- each stamp's target is a simple, unweighted
average of nearby heightmap cells (bare-earth points, rasterized once
at ingest time -- see ingest/heightmap.py). The architecture doc calls
out weighted least-squares as the "eventually" version of this step;
this is the "for now" one.

Sampling the pre-rasterized heightmap instead of querying the raw
point cloud's KD-tree directly is the same "no tree traversal needed
on a regular grid" idea as the render() optimization, applied to the
ground-truth side of the fit rather than the predicted side: the
heightmap is built once (ingest-laz), and every stamp's footprint
becomes a direct bounding-box array slice instead of a per-stamp
spatial query.

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
solves the value that reaches the target, given that brush's own
center weight and the stamp's own tool (see terrain/stamp.py):

    flatten (tool=0): target = current + (value - current) * weight_at_center
                       value  = current + (target - current) / weight_at_center
    raise   (tool=1): target = current + value * weight_at_center
                       value  = (target - current) / weight_at_center

Flatten's form reduces to value = target exactly whenever current = 0
and weight_at_center = 1 -- true for every stamp in the initial hex
grid, so this general form costs nothing there and is correct
everywhere else too.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ingest.heightmap import sample_heightmap_mean
from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.stamp import TOOL_RAISE, Stamp
from terrain.terrain_kernel import TerrainKernel
from terrain.terrain_model import TerrainModel


def fit_stamp_heights(
    stamps: Sequence[Stamp],
    heights: np.ndarray,
    bounds: BoundingBox,
    min_valid_cells: int = 3,
    existing_stamps: Sequence[Stamp] = (),
) -> list[Stamp]:
    """
    Return a new list of Stamps (same order, same length as `stamps`)
    with value replaced by a per-stamp best-fit height (flatten) or
    delta (raise), estimated from nearby heightmap cells -- see module
    docstring for the two formulas.

    `heights`/`bounds` are the rasterized ground heightmap and the
    bounds it was built over (see ingest/heightmap.py's
    rasterize_ground_heightmap/load_heightmap) -- bare-earth filtering
    already happened once, at rasterization time, not per stamp here.

    `stamps` are processed in list order, and each one's fit accounts
    for whatever height stamps earlier in the list -- or in
    `existing_stamps`, which is seeded in first but not re-fit or
    included in the return value -- already left at its position. This
    is what makes it correct to fit new adaptive-refinement detail
    stamps against an already-fitted coarse terrain: pass the coarse
    stamps as `existing_stamps` so new stamps' fits account for the
    baseline that's already there, instead of fitting against a false
    zero baseline.

    Stamps with fewer than `min_valid_cells` nearby heightmap cells
    (e.g. sitting entirely over a lake or building, where bare-earth
    coverage doesn't exist) are left unchanged (their placeholder
    value untouched) rather than guessing from sparse or absent data
    -- left for a later pass to handle.
    """
    kernels: dict[int, TerrainKernel] = {}
    n_seed = len(existing_stamps)

    # Stamp positions are fixed for the whole call -- only their
    # values get filled in as we go -- so the KD-tree only needs
    # building once, not once per stamp. Each stamp's fitted value is
    # patched into model.stamps in place as soon as it's known;
    # max_stamp_index on evaluate() keeps a stamp's own "current"
    # height from folding in later stamps that haven't been fitted
    # yet (still holding their placeholder value).
    model = TerrainModel(list(existing_stamps) + list(stamps))

    for offset, stamp in enumerate(stamps):
        index = n_seed + offset
        if stamp.brush not in kernels:
            if stamp.brush not in BRUSH_PROFILES:
                raise ValueError(f"No BrushProfile registered for brush type {stamp.brush}")
            kernels[stamp.brush] = TerrainKernel(BRUSH_PROFILES[stamp.brush])

        # sample_heightmap_mean does a single-radius circular average --
        # only ever called against hex-grid stamps here (see this
        # module's docstring), which are always isotropic
        # (scale_x == scale_z), so max() is exact, not just a bound.
        target = sample_heightmap_mean(
            heights, bounds, stamp.x, stamp.z, max(stamp.scale_x, stamp.scale_z),
            min_valid_cells=min_valid_cells,
        )
        if target is None:
            continue

        weight_at_center = kernels[stamp.brush].sample(0.0)
        if weight_at_center <= 0.0:
            # No registered brush should hit this (every profile has a
            # nonzero center weight), but guard against div-by-zero
            # rather than producing an inf/nan value.
            continue

        current = model.evaluate(stamp.x, stamp.z, max_stamp_index=index)
        if stamp.tool == TOOL_RAISE:
            value = (target - current) / weight_at_center
        else:
            value = current + (target - current) / weight_at_center
        model.stamps[index] = replace(stamp, value=value)

    return model.stamps[n_seed:]
