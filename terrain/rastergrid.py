"""
terrain/rastergrid.py

Generates a flat, axis-aligned raster (regular, non-offset square-grid)
lattice of type-72 ("hard square") stamp positions covering the
playable course -- an alternative initial coverage mode to hexgrid.py's
flat-top hex lattice.

Unlike hex mode (whose stamps start at a placeholder value and need a
separate least-squares fit against the heightmap, see height_fit.py)
or contour mode (whose stamps are valued at the local heightmap MEAN
within each one's own footprint, see contour_layers.py), a raster
stamp's value comes directly from a single nearest-cell heightmap
lookup (ingest.heightmap.sample_heightmap_nearest) at its own center.
There's nothing to fit or average here: the whole point of a regular
raster grid is spacing tight enough that "nearest sample" is already a
faithful reconstruction on its own, so this mode is much cheaper per
stamp than either alternative.

RASTER_SIZES = (256, 64, 32, 16, 8, 4, 2) is the fixed menu of valid
center-to-center grid spacings ("size") this mode supports -- pick ONE
per call (generate_raster_grid's `size` argument, same role as hex
mode's `pitch`), and layer coarse-to-fine yourself by re-running
generate-terrain at each size in turn, same next_n auto-increment
layering every other method already uses (see PGA2k_gen.py's
step_generate_terrain docstring). The jump from 256 to 64 (not 128) is
intentional, not a typo -- 256 is a fast, very coarse first pass;
every step from 64 down to 2 then simply halves.

STAMP SIZING: type 72's real measured brush profile (see
terrain/contour_layers.py's RECT FILL MODE docstring section, where
this exact formula is first derived) is a flat plateau out to
r=250/256 of its own nominal radius, then an EXACTLY instant drop to
0.0 the very next sample -- not a gradual falloff. TYPE72_PLATEAU_
FRACTION = 500/512 (contour_layers.py's own constant, imported here
rather than duplicated) is the full-width equivalent of that same
measurement. For neighboring stamps' plateaus to tile edge-to-edge at
center spacing `size` -- touching, no gap, no overlap:

    pitch (full nominal footprint) = size / TYPE72_PLATEAU_FRACTION
    radius (scale_x == scale_z)    = pitch / 2
                                    = size / (2 * TYPE72_PLATEAU_FRACTION)

SPREAD: `spread_ratio` (default DEFAULT_RASTER_SPREAD_RATIO = 1.0,
same name/role as hexgrid.py's HEX_DEFAULT_SPREAD_RATIO) scales that
same radius independently of `size` -- lattice centers stay exactly
where the size/pitch math above puts them; only the per-stamp
scale_x/scale_z grow or shrink. 1.0 reproduces the exact edge-to-edge
tiling derived above; >1 grows each stamp past its own cell so
neighbors overlap instead of just touching (softer blending at the
seams, at the cost of no longer being an exact per-cell nearest-sample
reconstruction); <1 leaves a gap between cells (only meaningful
alongside another pass that fills it). Exists mainly as an A/B knob
against the exact-tiling default -- see course_output/userLayers.py's
normalize_stamp_heights_by_value_shift for the related seam-artifact
investigation this is meant to help narrow down.

CENTER BIAS: TerrainModel folds overlapping stamps in strict list
order (see terrain_model.py's own docstring) -- intentional, since it
must match the in-game renderer's own replay order. This lattice's
list order always advances +x fastest within a row, then +z across
rows (see the meshgrid/ravel call below), so wherever two stamps'
footprints overlap, the one further +x/+z always wins the contested
area: a stamp's own uncontested "surviving" region is skewed toward
its -x/-z (bottom-left) corner rather than centered on the cell it's
meant to represent -- visible as a resolution-dependent "drop shadow"
biased left/down, especially where a masked refinement pass at a
finer `size` stops and whatever skew the previous, coarser pass left
behind is exposed again just outside it.

`center_bias_ratio_x`/`center_bias_ratio_z` (default
DEFAULT_RASTER_CENTER_BIAS_RATIO = 0.0 each, off) compensate by
shifting each stamp's *placement* -- its Stamp.x/Stamp.z, which drive
geometry and overwrite ordering -- by `size * center_bias_ratio_x`
along x and `size * center_bias_ratio_z` along z, independently, so
the surviving remnant re-centers over the intended cell. Kept as two
separate knobs rather than one diagonal ratio since the actual bias
axis isn't necessarily a clean 45 degrees -- e.g. brush/heightmap
anisotropy or a non-square course crop could make the x and z
components genuinely different; dial each in independently. Positive
values push +x/+z (the "later in list order" direction, see above);
negative values push -x/-z. Deliberately decoupled from value
sampling -- the heightmap lookup below always uses the true, unbiased
lattice position, so the height reconstructed for a cell still reflects
that cell's real reading regardless of this knob. There's no analytic
answer for the right ratio, so like `spread_ratio` these are opt-in
A/B knobs to dial in empirically, not derived constants.

COVERAGE AT THE EDGES: columns/rows are anchored to bounds.min_x/
min_z (not centered, same corner-anchoring idea as hexgrid.py's column
placement) and walked outward in steps of `size` for as long as a
center's own plateau (half-width = radius * TYPE72_PLATEAU_FRACTION,
which is size/2 exactly at the default spread_ratio=1, and scales
with spread_ratio otherwise -- see SPREAD above) still reaches past
the opposite bound -- i.e. while `center - plateau_half_width <
bound`. This is a direct per-axis coverage condition, not a separate
bleed constant the way hexgrid.py's HEX_BLEED_M is (that one exists
specifically to compensate for hex's offset-row asymmetry, which a
plain non-offset raster grid doesn't have) -- but it plays the same
role: guaranteeing the true course boundary sits inside some stamp's
plateau rather than past every center's reach. This walk is extended
further still (in whole cells, same lo-anchored phase) whenever
CENTER BIAS is non-zero, below -- otherwise a non-trivial bias could
shift every stamp's placement enough to open a real gap at whichever
edge it points away from, since the natural margin above is generous
at `lo` but only about one cell wide at `hi`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import shapely.vectorized
from shapely.geometry.base import BaseGeometry

from ingest.heightmap import sample_heightmap_nearest
from terrain.bounding_box import BoundingBox
from terrain.contour_layers import TYPE72_PLATEAU_FRACTION
from terrain.stamp import TOOL_FLATTEN, Stamp

RASTER_SIZES = (256, 64, 32, 16, 8, 4, 2)
RASTER_BRUSH = 72  # SHAPE_SQUARE -- see terrain/brush_profiles.py; only type 72 has the measured
                     # hard-plateau/instant-edge profile this module's sizing math assumes
DEFAULT_RASTER_SIZE = RASTER_SIZES[0]
DEFAULT_RASTER_SPREAD_RATIO = 1.0  # scales stamp radius independently of size; 1 = exact edge-to-edge tiling (see module docstring)
DEFAULT_RASTER_CENTER_BIAS_RATIO = 0.0  # per-axis placement bias, as a fraction of size; 0 = off (see module docstring's CENTER BIAS)

_EPS = 1e-6


def generate_raster_grid(
    bounds: BoundingBox,
    heightmap: np.ndarray,
    heightmap_bounds: BoundingBox,
    size: float = DEFAULT_RASTER_SIZE,
    spread_ratio: float = DEFAULT_RASTER_SPREAD_RATIO,
    center_bias_ratio_x: float = DEFAULT_RASTER_CENTER_BIAS_RATIO,
    center_bias_ratio_z: float = DEFAULT_RASTER_CENTER_BIAS_RATIO,
    brush: int = RASTER_BRUSH,
    tool: int = TOOL_FLATTEN,
    mask_geometry: Optional[BaseGeometry] = None,
) -> tuple[list[Stamp], int]:
    """
    Generate a regular (non-offset) square-grid lattice of Stamps
    covering `bounds`, each valued at the nearest heightmap cell to its
    own center (see module docstring -- no fitting, no averaging).

    `size` must be one of RASTER_SIZES -- the fixed menu this mode
    supports, not a free spacing (see module docstring for why).

    `spread_ratio` scales each stamp's radius independently of `size`
    (see module docstring's SPREAD section) -- lattice centers are
    still derived from `size` alone; only scale_x/scale_z change.
    Default 1.0 reproduces the exact edge-to-edge tiling radius.

    `center_bias_ratio_x`/`center_bias_ratio_z` shift each stamp's
    placement (not its sampled value) by `size * center_bias_ratio_x`
    along x and `size * center_bias_ratio_z` along z, independently --
    see module docstring's CENTER BIAS section. Default 0.0 (each) is
    off (identical behavior to before this knob existed).

    `heightmap`/`heightmap_bounds` are the rasterized ground heightmap
    and the bounds it was built over (ingest.heightmap.load_heightmap).
    Kept as separate arguments from `bounds` (rather than assumed
    identical) since a caller restricting `bounds` to something smaller
    than the full course -- e.g. a targeted masked pass -- should still
    sample against the FULL heightmap, same reasoning as hex mode's
    fit_stamp_heights taking the full heightmap regardless of how the
    lattice itself is restricted.

    mask_geometry, if given, restricts which grid points actually
    become Stamps to those whose CENTER falls inside it -- checked
    here, during generation, same mask_geometry contract as hexgrid.py's
    generate_hex_grid (grid phase is derived from the full `bounds`
    regardless of the mask, so a masked pass never re-derives a
    shifted/re-phased grid).

    Returns (stamps, n_skipped): n_skipped counts grid points dropped
    because their nearest heightmap cell was NaN (no bare-earth LIDAR
    coverage there, e.g. a lake or building) -- unlike hex mode, there's
    no placeholder-value fallback here, since a raster stamp's entire
    value comes from that one lookup and there's nothing sensible to
    fall back to.
    """
    if size not in RASTER_SIZES:
        raise ValueError(f"size must be one of {RASTER_SIZES}, got {size!r}")

    width = bounds.max_x - bounds.min_x
    height = bounds.max_z - bounds.min_z
    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate bounds for raster grid: {bounds}")

    radius = (size / (2.0 * TYPE72_PLATEAU_FRACTION)) * spread_ratio
    # Actual half-width of the flat plateau (see module docstring's
    # SPREAD section) -- scales with spread_ratio just like `radius`
    # itself, so the edge-coverage walk below still guarantees the
    # true boundary sits inside some stamp's plateau even when
    # spread_ratio != 1, not just at the default exact-tiling radius.
    plateau_half_width = radius * TYPE72_PLATEAU_FRACTION

    # Placement-only per-axis offset -- see module docstring's CENTER
    # BIAS section. Applied to the Stamp's own x/z below, never to
    # grid_x/grid_z (mask containment and the heightmap sample below
    # both stay at the true, unbiased lattice position).
    bias_x = size * center_bias_ratio_x
    bias_z = size * center_bias_ratio_z

    def _axis_centers(lo: float, hi: float, bias: float) -> np.ndarray:
        # Coverage margin is generous at `lo` (the first center is
        # always placed exactly on it, so its plateau already overhangs
        # past the boundary by a full plateau_half_width) but only
        # about one cell wide at `hi` (the walk stops as soon as the
        # boundary is just barely cleared) -- so shifting every
        # stamp's PLACEMENT by `bias` (below) can open a real gap at
        # whichever end `bias` points away from, regardless of
        # spread_ratio. Compensate by extending the walk, in whole
        # cells, on that side -- backward past `lo` if bias is
        # positive (stamps pushed toward `hi`, away from `lo`),
        # forward past `hi` if bias is negative -- so the true lo/hi
        # bounds still land inside some (still-unbiased) center's
        # reach once every generated center is shifted by `bias`.
        # Extending in whole cells keeps the same lo-anchored phase,
        # so every already-covered sample position is unaffected.
        extra_lo_cells = math.ceil(max(bias, 0.0) / size)
        extra_hi_cells = math.ceil(max(-bias, 0.0) / size)
        walk_lo = lo - extra_lo_cells * size
        walk_hi = hi + extra_hi_cells * size
        n = int(math.floor((walk_hi - walk_lo + plateau_half_width - _EPS) / size)) + 1
        return walk_lo + np.arange(n) * size

    xs = _axis_centers(bounds.min_x, bounds.max_x, bias_x)
    zs = _axis_centers(bounds.min_z, bounds.max_z, bias_z)
    grid_x, grid_z = np.meshgrid(xs, zs)
    grid_x = grid_x.ravel()
    grid_z = grid_z.ravel()

    if mask_geometry is not None:
        inside = shapely.vectorized.contains(mask_geometry, grid_x, grid_z)
        grid_x = grid_x[inside]
        grid_z = grid_z[inside]

    stamps: list[Stamp] = []
    n_skipped = 0
    for px, pz in zip(grid_x, grid_z):
        value = sample_heightmap_nearest(heightmap, heightmap_bounds, float(px), float(pz))
        if value is None:
            n_skipped += 1
            continue
        stamps.append(
            Stamp(x=float(px + bias_x), z=float(pz + bias_z), scale_x=radius, scale_z=radius,
                  value=value, brush=brush, tool=tool)
        )

    return stamps, n_skipped
