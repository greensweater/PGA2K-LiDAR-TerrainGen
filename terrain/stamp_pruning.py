"""
terrain/stamp_pruning.py

Detects and drops stamps whose entire effect is masked out by later
stamps -- e.g. a coarse SHAPE_SQUARE "raster fill" tile (see
rastergrid.py) that a subsequent higher-resolution flatten pass has
completely paved over. Folded automatically into step_write_terrain/
step_write_water (see PGA2k_gen.py's _load_normalized_stamps) right
after the stamp list is loaded -- opt-out via --no-prune-overlapped-
stamps / the GUI's "Delete overlapped stamps" checkbox, on by default.

Why a later stamp can make an earlier one *provably* irrelevant:
TerrainModel folds every stamp as an affine map of the incoming height
(see terrain_model.py's docstring for the flatten/raise formulas) --

    flatten: h -> h*(1-w) + v*w      (slope 1-w)
    raise:   h -> h + v*w            (slope 1)

so the height any point ends up at, as a function of what it was
*before* stamp i ran, is itself affine after any run of later stamps:
h -> a*h + b, where `a` is the product of (1-w_j) over every later
FLATTEN stamp reaching that point (raise stamps only shift the
intercept `b`, never the slope `a`). If a == 0 at a point -- which
real square/plateau brushes reach over most of their own footprint,
see brush_profiles.py -- that point's final height no longer depends
on what it was before stamp i ran, i.e. stamp i's own contribution
there is provably erased regardless of its own tool or value. A stamp
is safe to drop once that holds (within `tol`) everywhere its own
weight is non-negligible.

Implementation: a single shared "not yet overwritten" grid (`a`
above), rasterized once over the whole stamp-covered area and updated
in place while sweeping stamps newest-to-oldest -- the same per-stamp
bounding-box index arithmetic TerrainModel.render() uses, adapted to
accumulate this survival factor instead of height. Each stamp touches
the grid exactly once (a read to decide keep/drop, and -- only if
kept and tool==flatten -- one multiply to fold its own shadow in for
everything still to come), so total cost is O(sum of stamp footprint
areas), not O(stamps x candidates); an earlier per-stamp
candidate-search design (spatial hash + rebuild-per-candidate weight
grids) was replaced after it went quadratic-ish in densely stamped
regions -- exactly the "many higher-res stamps packed over one masked
area" case this feature targets.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from terrain.brush_profiles import BRUSH_PROFILES, SHAPE_SQUARE
from terrain.stamp import TOOL_FLATTEN, Stamp, local_square_offsets
from terrain.terrain_kernel import TerrainKernel

WEIGHT_EPS = 1e-3  # a stamp's own weight below this is treated as "doesn't matter here"
DEFAULT_TOL = 1e-3  # max residual "not yet overwritten" factor (see module docstring's `a`) to still drop a stamp

# Grid resolution is sized off a small percentile of stamps' own reach
# (see _grid_cell_size) rather than the smallest reach outright, so one
# stray tiny stamp can't force the whole grid arbitrarily fine. That
# resolution choice is then hard-capped by total cell count (below) as
# a second line of defense against runaway memory/time on unusual
# inputs -- e.g. a huge course spanned by only a handful of very small
# stamps, where even a robust percentile would still pick a tiny cell.
RESOLUTION_PERCENTILE = 5
SAMPLES_ACROSS_SMALLEST = 20  # ~ how many grid cells span a small stamp's own diameter
MAX_GRID_CELLS = 64_000_000  # ~256 MB at float32


def _reach(stamp: Stamp) -> tuple[float, float]:
    """(reach_x, reach_z): axis-aligned half-extent of this stamp's own footprint."""
    profile = BRUSH_PROFILES.get(stamp.brush)
    is_square = profile is not None and profile.shape == SHAPE_SQUARE
    if is_square and stamp.rotation != 0.0:
        # Loose but safe -- same corner-reach fallback TerrainModel/render()
        # use for a rotated rectangle (see terrain_model.py).
        r = math.hypot(stamp.scale_x, stamp.scale_z)
        return r, r
    if is_square:
        return stamp.scale_x, stamp.scale_z
    return stamp.scale_x, stamp.scale_x


def _weight_grid(stamp: Stamp, kernel: TerrainKernel, xx: np.ndarray, zz: np.ndarray) -> np.ndarray:
    """Sample `stamp`'s brush weight at every (xx, zz) grid point (0 outside its reach)."""
    profile = BRUSH_PROFILES.get(stamp.brush)
    dx = xx - stamp.x
    dz = zz - stamp.z
    if profile is not None and profile.shape == SHAPE_SQUARE:
        ax, az = local_square_offsets(stamp, dx, dz)
        r = np.maximum(np.abs(ax) / stamp.scale_x, np.abs(az) / stamp.scale_z)
    else:
        r = np.hypot(dx, dz) / stamp.scale_x

    weight = np.zeros_like(r)
    in_reach = r <= 1.0
    if np.any(in_reach):
        weight[in_reach] = kernel.sample_many(r[in_reach])
    return weight


def _grid_cell_size(reaches: list[tuple[float, float]]) -> float:
    """
    Pick a world-meters grid cell size fine enough to resolve a "small"
    stamp's footprint (SAMPLES_ACROSS_SMALLEST cells across its own
    diameter), where "small" is RESOLUTION_PERCENTILE across all
    stamps' tighter axis -- robust to a single outlier tiny stamp,
    unlike using the true minimum.
    """
    tight = np.array([min(rx, rz) for rx, rz in reaches], dtype=np.float64)
    tight = tight[tight > 0]
    if tight.size == 0:
        return 1.0
    small_reach = float(np.percentile(tight, RESOLUTION_PERCENTILE))
    return max(small_reach * 2.0 / SAMPLES_ACROSS_SMALLEST, 1e-6)


def prune_overwritten_stamps(
    stamps: Sequence[Stamp], tol: float = DEFAULT_TOL,
) -> tuple[list[Stamp], int]:
    """
    Drop every stamp whose effect is fully masked out by later stamps
    -- e.g. a coarse raster-fill tile (rastergrid.py) completely paved
    over by a later, higher-resolution flatten pass -- without changing
    the resolved terrain beyond `tol` (see module docstring for why
    this is a provable test, not a heuristic).

    Processes stamps newest-to-oldest: the shared coverage grid only
    ever reflects stamps already confirmed to survive into the result
    (folded in immediately after each keep decision), so "am I fully
    overwritten?" is always answered against what the output will
    actually still contain, never against a stamp that's itself about
    to be dropped.

    Returns (surviving_stamps, number_dropped); surviving_stamps keeps
    the original relative order.
    """
    stamps = list(stamps)
    if not stamps:
        return stamps, 0

    kernels: dict[int, TerrainKernel] = {}
    for s in stamps:
        if s.brush not in kernels:
            if s.brush not in BRUSH_PROFILES:
                raise ValueError(f"No BrushProfile registered for brush type {s.brush}")
            kernels[s.brush] = TerrainKernel(BRUSH_PROFILES[s.brush])

    reaches = [_reach(s) for s in stamps]
    min_x = min(s.x - rx for s, (rx, rz) in zip(stamps, reaches))
    max_x = max(s.x + rx for s, (rx, rz) in zip(stamps, reaches))
    min_z = min(s.z - rz for s, (rx, rz) in zip(stamps, reaches))
    max_z = max(s.z + rz for s, (rx, rz) in zip(stamps, reaches))

    cell_size = _grid_cell_size(reaches)
    nx = max(1, math.ceil((max_x - min_x) / cell_size))
    nz = max(1, math.ceil((max_z - min_z) / cell_size))
    if nx * nz > MAX_GRID_CELLS:
        scale = math.sqrt((nx * nz) / MAX_GRID_CELLS)
        cell_size *= scale
        nx = max(1, math.ceil((max_x - min_x) / cell_size))
        nz = max(1, math.ceil((max_z - min_z) / cell_size))
        print(f"  NOTE: stamp_pruning coarsened its coverage grid to {cell_size:.3f} m/cell "
              f"({nx}x{nz}) to stay under {MAX_GRID_CELLS:,} cells -- very small stamps may be "
              "under-resolved and go unpruned as a result (never incorrectly pruned).")

    edges_x = min_x + np.arange(nx + 1) * cell_size
    edges_z = min_z + np.arange(nz + 1) * cell_size
    centers_x = (edges_x[:-1] + edges_x[1:]) / 2.0
    centers_z = (edges_z[:-1] + edges_z[1:]) / 2.0

    # a_grid[row, col] == the product of (1-weight) of every later
    # FLATTEN stamp already confirmed kept, at that cell's center --
    # i.e. exactly `a` from the module docstring.
    a_grid = np.ones((nz, nx), dtype=np.float32)

    keep_flags = [True] * len(stamps)

    for i in range(len(stamps) - 1, -1, -1):
        stamp = stamps[i]
        reach_x, reach_z = reaches[i]

        col_min = max(0, int((stamp.x - reach_x - min_x) / cell_size))
        col_max = min(nx, int((stamp.x + reach_x - min_x) / cell_size) + 1)
        row_min = max(0, int((stamp.z - reach_z - min_z) / cell_size))
        row_max = min(nz, int((stamp.z + reach_z - min_z) / cell_size) + 1)
        if col_min >= col_max or row_min >= row_max:
            continue  # footprint too small to register on the grid at all -- keep, can't verify

        sub_x = centers_x[col_min:col_max]
        sub_z = centers_z[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)

        own_weight = _weight_grid(stamp, kernels[stamp.brush], xx, zz)
        relevant = own_weight > WEIGHT_EPS

        sub_a = a_grid[row_min:row_max, col_min:col_max]
        if np.any(relevant) and np.all(sub_a[relevant] <= tol):
            keep_flags[i] = False
        elif stamp.tool == TOOL_FLATTEN:
            sub_a *= (1.0 - own_weight)

    surviving = [s for s, keep in zip(stamps, keep_flags) if keep]
    return surviving, len(stamps) - len(surviving)
