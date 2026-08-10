"""
terrain/contour_layers.py

Alternative initial base-layer generator to hexgrid.py's flat hex
lattice: instead of placing stamps only ON traced contour lines (an
earlier version of this module), this treats each pair of consecutive
elevation levels (A, A+) as a CHANNEL that needs to be filled, and
sizes every stamp from the REAL terrain's own distance-to-boundary,
not from ring curvature. This is what makes it self-adjusting across
wildly different terrain (craggy valleys vs. smooth fairways) without
any manual tuning: narrow channels (steep, complex terrain) naturally
get small, tightly-packed stamps; wide channels (gentle, simple
terrain) naturally get few, large ones.

THE CHANNEL MODEL, per level pair (A, A+) with A- the level below A:

  Rule 3 -- on ring A, valued at A, radius = local distance DOWN to
  A- (capped). Finishes the (A-, A) band from its own top edge with
  A's own correct value.

  Rule 4 -- on ring A+, valued at A (the LOWER of the two levels, not
  A+), radius = local distance down to A (capped). A rough, too-low
  advance pass into the (A, A+) band, placed BEFORE that band's own
  Rule 3 pass (which runs when A+ becomes "current").

Both rules for level A therefore need exactly one shared quantity:
dist_field(A) = distance_transform_edt(heights >= A) -- for every
pixel at or above A, its true Euclidean distance to the nearest pixel
below A. Sampled along ring A+ this is Rule 4's radius; sampled along
ring A next iteration, after re-deriving from the PREVIOUS level, it's
Rule 3's. So dist_field(A) is computed once per level and consumed by
both rings that touch it (ring A itself, and ring A+ from below).

Ring A+ therefore gets touched twice, in this order: Rule 4 (rough,
value A) first, Rule 3 (precise, value A+) second -- same ring points,
same radii (same dist_field), different values. Under this project's
sequential pull-toward-value compositing, that produces a genuine
smooth GRADIENT across the channel from purely flat-value stamps: near
ring A+ the later, same-weight precise pass dominates (-> A+), near
ring A (weight ~0 for both passes here) whatever Rule 3 already set at
ring A from the level below shows through (-> A) -- not a flat patch
with a hard edge, an actual blend between the two real elevations.

CAPPED REGIONS (hilltops and pits): a peak that never reaches the next
level, or a pond that never reaches the previous one, has no second
ring to pair with -- there's no "A+" or "A-" to run Rule 4/3 against.
These get filled directly and separately: connected components of
(heights >= A) whose own max height never reaches the next level are
capped hilltops; connected components of (heights < A) whose own min
height never reaches the previous level are capped pits (a pond that
doesn't bottom out below the previous ring, or literally the course's
lowest terrain, below the lowest traced level, always counts). Both
get filled via greedy largest-first placement (type 8 by default --
NOT 54, which measures at only ~62% vertical amplitude of type 8 per
2k25_terrain_arch.txt's own brush data, too soft/short-reaching for a
fill whose entire job is guaranteeing coverage) with the distance
transform recomputed LIVE against the shrinking remaining area each
iteration, cropped to the region's own bounding box -- see
_fill_region_greedy's docstring for why a single precomputed field
(an earlier version's approach) leaves real gaps near a region's true
edge once its middle gets claimed. Interiors don't need the boundary
passes' fine, curvature-matched footprint, they need coverage with as
few stamps as reasonably possible.

Caveat worth being upfront about: this component check operates on the
WHOLE connected landmass, so it correctly catches genuinely isolated
peaks and ponds, but NOT a locally-flat sub-region (a green on the
side of a larger hill, say) that's part of one big connected landmass
which does continue upward/downward elsewhere. That case falls through
to the final residual pass below instead -- still correctly filled
with large, overlapping stamps, just without the hilltop/pit
brush/sizing choice specifically.

RESIDUAL SAFETY NET: after every ring, hilltop, and pit pass, one final
distance-transform gap-fill (same style as an earlier version of this
module) rasterizes actual coverage and fills whatever's still
untouched -- a very wide flat channel whose two boundary passes can't
reach the true middle even at max radius, or the locally-flat-within-
a-larger-landmass case above. This is a genuine saturation guarantee
(runs until nothing uncovered remains), independent of every other
pass's correctness.

OVERLAP: every pass here uses the same claim-a-smaller-inner-fraction
idea (place at full radius, claim/space less) an earlier version of
this module already established, for the same reason -- a cosine-
falloff kernel's weight is ~0 right at a tangent point between two
touching stamps, so tangent-only placement reads as visible seams, not
a smooth result.

BRUSH SOFTNESS: geometric radius alone isn't a fair coverage proxy
across brush types. terrain/adaptive_refine.py already documents this
via BRUSH_RANK={8:0, 9:1, 10:2, 54:3} -- higher-rank brushes have
smoother falloff and genuinely weaker real influence relative to their
OWN nominal radius than type 8 does (measured directly for one case:
type 54 sits at only ~62% of type 8's amplitude at the same radius/
height, see 2k25_terrain_arch.txt). Ring/rough default to type 10 and
9 (ranks 2 and 1) -- both softer than type 8 -- so treating "geometric
distance reached" as "meaningfully covered" with no brush-dependent
correction systematically overstates real coverage for the defaults
this module actually ships with, leaving thin real gaps between
neighbors that geometrically touch or even slightly overlap. The fix
here is spacing, not radius: widening the PLACEMENT radius itself
risks a ring-A stamp overshooting into the A- band's own territory,
reintroducing the bleed this whole design exists to avoid. Instead,
edge_softness_ratio (default 1.0, a no-op, same "expose the knob and
tune against real results" philosophy as adaptive_refine.py's own
brush_radius_spread_ratio rather than a derived table) packs a brush's
stamps proportionally CLOSER together the higher its rank -- spacing
and claim fractions get multiplied by edge_softness_ratio **
BRUSH_RANK[brush], so a ratio below 1.0 tightens softer brushes while
leaving type 8 (rank 0, softness_ratio**0 == 1) unaffected regardless
of the value chosen.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from skimage import measure

from ingest.heightmap import downsample_heightmap
from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

DEFAULT_BAND_SPACING_M = 5.0  # Delta -- GUI/CLI tweakable, deliberately fixed rather than adaptive

# Ring (channel-boundary) passes -- Rule 3 (precise, own ring) / Rule 4 (rough, advance into next band)
DEFAULT_RING_BRUSH = 10          # Rule 3: smooth falloff, no flat plateau -- blends into the gradient
DEFAULT_ROUGH_BRUSH = 9          # Rule 4: slightly narrower plateau, corrected on top next iteration
DEFAULT_MIN_RING_RADIUS_M = 5.0
DEFAULT_MAX_RING_RADIUS_M = 50.0
DEFAULT_RING_SPACING_FRACTION = 0.5  # spacing = radius * this; matches Chad's radius=2x-spacing rule

# Mirrors terrain/adaptive_refine.py's BRUSH_RANK -- duplicated here rather
# than imported, since this module deliberately doesn't depend on
# adaptive_refine.py's much larger surface for one small shared constant.
# Genuinely belongs in terrain/brush_profiles.py as the real shared source
# of truth if the two ever drift; keep in sync manually until then.
BRUSH_RANK = {8: 0, 9: 1, 10: 2, 54: 3}

DEFAULT_EDGE_SOFTNESS_RATIO = 1.0  # 1.0 = no-op (old behavior). <1.0 packs higher-BRUSH_RANK (softer-
                                    # falloff) brushes proportionally tighter -- see module docstring.

# Capped hilltop/pit interior fills -- large, hard stamps; few needed by design
DEFAULT_INTERIOR_BRUSH = 8  # wide flat plateau, sharp edge -- full-strength pull across most of its
                            # radius. NOT type 54: measured at only ~62% vertical amplitude of type 8
                            # (see 2k25_terrain_arch.txt's brush measurements) -- too soft/short-reaching
                            # for a fill whose whole job is guaranteeing coverage.
DEFAULT_MIN_INTERIOR_RADIUS_M = 10.0
DEFAULT_MAX_INTERIOR_RADIUS_M = 150.0
DEFAULT_INTERIOR_CLAIM_RADIUS_FRACTION = 0.5

# Final residual safety net -- genuinely rare/small by design; hardest brush, real saturation guarantee
DEFAULT_RESIDUAL_BRUSH = 8
DEFAULT_MIN_RESIDUAL_RADIUS_M = 10.0
DEFAULT_MAX_RESIDUAL_RADIUS_M = 150.0
DEFAULT_RESIDUAL_CLAIM_RADIUS_FRACTION = 0.5
DEFAULT_COVERAGE_RESOLUTION = 400  # coverage-mask grid for the residual pass only; independent of heights' own res
DEFAULT_MAX_RESIDUAL_ITERATIONS = 20000


def _contour_levels(heights: np.ndarray, spacing: float) -> np.ndarray:
    """
    Elevation levels to trace, spaced `spacing` m apart from just above
    the heightmap's own minimum to just below its max -- skimage's
    find_contours requires a level strictly inside the data's range to
    return anything, so the two endpoints (where a "ring" would just be
    the single lowest/highest point, not a real feature) are skipped.
    """
    lo = float(np.nanmin(heights))
    hi = float(np.nanmax(heights))
    if hi - lo < spacing:
        return np.array([])
    return np.arange(lo + spacing, hi, spacing)


def _pixel_to_world(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], bounds: BoundingBox,
) -> tuple[np.ndarray, np.ndarray]:
    """Map heightmap index-space (row, col), possibly subpixel, to world (x, z)."""
    n_rows, n_cols = shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    x = bounds.min_x + cols * cell_x
    z = bounds.min_z + rows * cell_z
    return x, z


def _place_stamps_along_ring(
    x: np.ndarray, z: np.ndarray, radii: np.ndarray, value: float, brush: int,
    min_radius: float, max_radius: float, spacing_fraction: float, edge_softness_ratio: float,
) -> list[Stamp]:
    """
    Walk one ring (world coords) placing a flatten stamp every time
    accumulated arc length since the last placement reaches
    `spacing_fraction * edge_softness_ratio**BRUSH_RANK[brush] * radius`
    at the CURRENT point -- deliberately less than the full radius
    (default spacing_fraction=0.5, edge_softness_ratio=1.0 a no-op) so
    consecutive stamps' full-radius footprints overlap rather than
    merely touch (see module docstring's OVERLAP note), tightened
    further for higher-BRUSH_RANK (softer-falloff) brushes when
    edge_softness_ratio < 1.0 (see module docstring's BRUSH SOFTNESS
    note). `radii` is precomputed per-vertex (sampled from a distance
    field by the caller, not derived from curvature here).
    """
    n = len(x)
    if n < 2:
        return []

    seg_len = np.hypot(np.diff(x), np.diff(z))
    clipped = np.clip(radii, min_radius, max_radius)
    effective_fraction = spacing_fraction * edge_softness_ratio ** BRUSH_RANK.get(brush, 0)

    stamps: list[Stamp] = [
        Stamp(x=float(x[0]), z=float(z[0]), radius=float(clipped[0]),
              value=value, brush=brush, tool=TOOL_FLATTEN)
    ]
    accumulated = 0.0
    next_target = clipped[0] * effective_fraction

    for i in range(1, n):
        accumulated += seg_len[i - 1]
        if accumulated >= next_target:
            stamps.append(Stamp(x=float(x[i]), z=float(z[i]), radius=float(clipped[i]),
                                 value=value, brush=brush, tool=TOOL_FLATTEN))
            accumulated = 0.0
            next_target = clipped[i] * effective_fraction

    return stamps


def _capped_hilltop_mask(heights: np.ndarray, level: float, next_level: Optional[float]) -> np.ndarray:
    """
    True over any connected component of (heights >= level) whose own
    maximum height never reaches `next_level` -- a peak that caps out
    within this band, with no ring above to pair it with (see module
    docstring). next_level=None (the topmost traced level) means every
    component here is capped by definition, nothing above was traced.

    Whole-component check, not a per-pixel one -- see module docstring
    caveat on what this does and doesn't catch.
    """
    base = heights >= level
    if not base.any():
        return np.zeros_like(base)
    if next_level is None:
        return base
    labeled, num = ndimage.label(base)
    capped = np.zeros_like(base)
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        if float(np.max(heights[comp])) < next_level:
            capped |= comp
    return capped


def _capped_pit_mask(heights: np.ndarray, level: float, prev_level: Optional[float]) -> np.ndarray:
    """
    Mirror of _capped_hilltop_mask: True over any connected component
    of (heights < level) whose own minimum height never reaches
    `prev_level` -- a pond/depression that doesn't bottom out below the
    previous traced level. prev_level=None (the lowest traced level)
    means every component here is capped -- this is the course's
    actual lowest terrain, below anything traced.
    """
    base = heights < level
    if not base.any():
        return np.zeros_like(base)
    if prev_level is None:
        return base
    labeled, num = ndimage.label(base)
    capped = np.zeros_like(base)
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        if float(np.min(heights[comp])) > prev_level:
            capped |= comp
    return capped


def _fill_region_greedy(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, min_radius: float, max_radius: float, claim_radius_fraction: float,
    edge_softness_ratio: float,
) -> list[Stamp]:
    """
    Greedy largest-first fill of `mask`, recomputing the distance
    transform LIVE against the shrinking remaining area each iteration
    -- an earlier version used a single precomputed field for the
    whole loop, which has a real flaw: EDT boundary pixels of the
    ORIGINAL mask always sit at distance ~0 by definition, so once a
    big stamp claims a region's middle, whatever's left near the true
    edge can look like "no room" to a STALE field even when there's
    genuinely still uncovered area there relative to what's actually
    been placed. Recomputing each iteration (same approach the
    residual pass already uses) fixes that directly, at the cost of
    more EDT calls -- kept cheap by cropping to `mask`'s own bounding
    box (+ a max_radius margin) rather than paying for the full
    heightmap's extent on every single placement, since capped
    hilltop/pit regions are local by construction.

    Claims only claim_radius_fraction of each placed radius (see
    module docstring's OVERLAP note), so consecutive interior stamps'
    full footprints genuinely overlap. Value is the real local
    heightmap mean within the placed radius -- these fills aren't tied
    to one exact contour level the way ring stamps are.
    """
    rows_nz, cols_nz = np.nonzero(mask)
    if rows_nz.size == 0:
        return []

    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    margin_x = int(np.ceil(max_radius / cell_x)) + 2
    margin_z = int(np.ceil(max_radius / cell_z)) + 2

    row_min = max(0, int(rows_nz.min()) - margin_z)
    row_max = min(n_rows, int(rows_nz.max()) + margin_z + 1)
    col_min = max(0, int(cols_nz.min()) - margin_x)
    col_max = min(n_cols, int(cols_nz.max()) + margin_x + 1)

    remaining = mask[row_min:row_max, col_min:col_max].copy()
    sub_heights = heights[row_min:row_max, col_min:col_max]
    x_centers = bounds.min_x + (np.arange(col_min, col_max) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(row_min, row_max) + 0.5) * cell_z
    xx_full, zz_full = np.meshgrid(x_centers, z_centers)
    sampling = (cell_z, cell_x)
    effective_claim_fraction = claim_radius_fraction * edge_softness_ratio ** BRUSH_RANK.get(brush, 0)

    stamps: list[Stamp] = []
    while remaining.any():
        dist = ndimage.distance_transform_edt(remaining, sampling=sampling)
        peak = float(dist.max())
        if peak <= 0.0:
            break
        row, col = np.unravel_index(np.argmax(dist), dist.shape)
        radius = float(np.clip(peak, min_radius, max_radius))
        claim_radius = radius * effective_claim_fraction
        cx, cz = float(x_centers[col]), float(z_centers[row])

        dist_from_center = np.hypot(xx_full - cx, zz_full - cz)
        within_radius = dist_from_center <= radius
        within_claim = dist_from_center <= claim_radius

        sub_valid = np.isfinite(sub_heights) & within_radius
        if not sub_valid.any():
            remaining[row, col] = False
            continue
        value = float(np.mean(sub_heights[sub_valid]))

        stamps.append(Stamp(x=cx, z=cz, radius=radius, value=value, brush=brush, tool=TOOL_FLATTEN))
        remaining[within_claim] = False

    return stamps


def _rasterize_coverage(
    stamps: list[Stamp], bounds: BoundingBox, resolution: int, covered: np.ndarray,
) -> None:
    """Mark `covered` True wherever a placed stamp's own radius reaches (in-place)."""
    cell_x = (bounds.max_x - bounds.min_x) / resolution
    cell_z = (bounds.max_z - bounds.min_z) / resolution
    x_centers = bounds.min_x + (np.arange(resolution) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(resolution) + 0.5) * cell_z

    for s in stamps:
        col_min = max(0, int((s.x - s.radius - bounds.min_x) / cell_x))
        col_max = min(resolution, int((s.x + s.radius - bounds.min_x) / cell_x) + 1)
        row_min = max(0, int((s.z - s.radius - bounds.min_z) / cell_z))
        row_max = min(resolution, int((s.z + s.radius - bounds.min_z) / cell_z) + 1)
        if col_min >= col_max or row_min >= row_max:
            continue
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        within = np.hypot(xx - s.x, zz - s.z) <= s.radius
        covered[row_min:row_max, col_min:col_max] |= within


def _residual_fill_pass(
    heights: np.ndarray, bounds: BoundingBox, resolution: int, covered: np.ndarray,
    brush: int, min_radius: float, max_radius: float, claim_radius_fraction: float,
    edge_softness_ratio: float,
    max_iterations: int = DEFAULT_MAX_RESIDUAL_ITERATIONS,
) -> list[Stamp]:
    """
    Final safety net: distance-transform "largest inscribed circle"
    loop (recomputed each iteration, since -- unlike the interior
    fills above -- this is expected to run rarely/briefly, so a fresh
    EDT per placement is cheap in aggregate) over whatever the coverage
    mask still shows as uncovered after every other pass. Runs to
    genuine saturation, not a failure-count heuristic.
    """
    actual = downsample_heightmap(heights, bounds, resolution)
    valid = np.isfinite(actual)

    cell_x = (bounds.max_x - bounds.min_x) / resolution
    cell_z = (bounds.max_z - bounds.min_z) / resolution
    x_centers = bounds.min_x + (np.arange(resolution) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(resolution) + 0.5) * cell_z
    sampling = (cell_z, cell_x)
    effective_claim_fraction = claim_radius_fraction * edge_softness_ratio ** BRUSH_RANK.get(brush, 0)

    stamps: list[Stamp] = []
    for _ in range(max_iterations):
        uncovered = valid & ~covered
        if not uncovered.any():
            break
        dist = ndimage.distance_transform_edt(uncovered, sampling=sampling)
        peak = float(dist.max(initial=0.0))
        if peak <= 0.0:
            break
        row, col = np.unravel_index(np.argmax(dist), dist.shape)
        radius = float(np.clip(peak, min_radius, max_radius))
        claim_radius = radius * effective_claim_fraction
        cx, cz = float(x_centers[col]), float(z_centers[row])

        row_min = max(0, int((cz - radius - bounds.min_z) / cell_z))
        row_max = min(resolution, int((cz + radius - bounds.min_z) / cell_z) + 1)
        col_min = max(0, int((cx - radius - bounds.min_x) / cell_x))
        col_max = min(resolution, int((cx + radius - bounds.min_x) / cell_x) + 1)
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        dist_from_center = np.hypot(xx - cx, zz - cz)
        within_radius = dist_from_center <= radius
        within_claim = dist_from_center <= claim_radius

        sub_actual = actual[row_min:row_max, col_min:col_max]
        sub_valid = valid[row_min:row_max, col_min:col_max] & within_radius
        if not sub_valid.any():
            covered[row, col] = True
            continue
        value = float(np.mean(sub_actual[sub_valid]))

        stamps.append(Stamp(x=cx, z=cz, radius=radius, value=value, brush=brush, tool=TOOL_FLATTEN))
        covered[row_min:row_max, col_min:col_max][within_claim] = True

    return stamps


def generate_contour_layers(
    heights: np.ndarray,
    bounds: BoundingBox,
    band_spacing_m: float = DEFAULT_BAND_SPACING_M,
    ring_brush: int = DEFAULT_RING_BRUSH,
    rough_brush: int = DEFAULT_ROUGH_BRUSH,
    min_ring_radius: float = DEFAULT_MIN_RING_RADIUS_M,
    max_ring_radius: float = DEFAULT_MAX_RING_RADIUS_M,
    ring_spacing_fraction: float = DEFAULT_RING_SPACING_FRACTION,
    interior_brush: int = DEFAULT_INTERIOR_BRUSH,
    min_interior_radius: float = DEFAULT_MIN_INTERIOR_RADIUS_M,
    max_interior_radius: float = DEFAULT_MAX_INTERIOR_RADIUS_M,
    interior_claim_radius_fraction: float = DEFAULT_INTERIOR_CLAIM_RADIUS_FRACTION,
    residual_brush: int = DEFAULT_RESIDUAL_BRUSH,
    min_residual_radius: float = DEFAULT_MIN_RESIDUAL_RADIUS_M,
    max_residual_radius: float = DEFAULT_MAX_RESIDUAL_RADIUS_M,
    residual_claim_radius_fraction: float = DEFAULT_RESIDUAL_CLAIM_RADIUS_FRACTION,
    coverage_resolution: int = DEFAULT_COVERAGE_RESOLUTION,
    edge_softness_ratio: float = DEFAULT_EDGE_SOFTNESS_RATIO,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> list[Stamp]:
    """
    Generate an organic base layer via the channel model (see module
    docstring): ring passes (Rule 3 precise + Rule 4 rough) fill every
    band between consecutive traced levels, capped-component passes
    fill isolated hilltops/pits those rings can't pair against, and a
    final residual pass mops up anything still uncovered (a genuinely
    rare case by design, not the bulk of the work).

    Returns stamps in placement order: ring passes ascending by level
    (each level's Rule 4 rough pass on the ring above it, immediately
    followed -- once that ring becomes "current" -- by its own Rule 3
    precise pass), then all capped-hilltop/pit fills, then the residual
    pass. Under this project's sequential pull-toward-value compositing,
    later stamps take precedence in any overlap -- see module docstring
    for why this specific order produces a real gradient across each
    channel rather than a flat patch with a hard edge.

    edge_softness_ratio (default 1.0, a no-op) tightens spacing/claim
    fractions for higher-BRUSH_RANK (softer-falloff) brushes -- see
    module docstring's BRUSH SOFTNESS note. Matters most with the
    default ring_brush=10/rough_brush=9 (ranks 2 and 1), since their
    real influence relative to their own nominal radius is genuinely
    weaker than type 8's; a value below 1.0 (start around 0.7-0.85 and
    tune against real results, same philosophy as adaptive_refine.py's
    brush_radius_spread_ratio) packs their stamps closer together to
    compensate without changing any stamp's own geometric radius.

    progress_callback, if given, is called periodically (time-
    throttled to ~10s, not every level) with (stamps_placed_so_far,
    fraction_complete). Progress is tracked as levels processed out of
    a fixed budget of 2*n_levels + 1 steps -- the ring-pass loop (n
    levels, ascending from lowest to highest elevation), the hilltop/
    pit-fill loop (another n levels), and the final residual pass (one
    step) -- a genuinely meaningful estimate since it's the same
    lowest-to-highest level order the function actually walks, not a
    raw iteration count against an unrelated bound.
    """
    levels = _contour_levels(heights, band_spacing_m)
    n = len(levels)
    stamps: list[Stamp] = []

    total_steps = max(1, 2 * n + 1)
    completed_steps = 0
    last_progress_time = time.time()
    progress_interval_s = 10.0

    def _maybe_report() -> None:
        nonlocal last_progress_time
        if progress_callback is not None and time.time() - last_progress_time >= progress_interval_s:
            progress_callback(len(stamps), completed_steps / total_steps)
            last_progress_time = time.time()

    if n > 0:
        rings_by_level = [measure.find_contours(heights, level=lvl) for lvl in levels]
        above_dist_by_level = [
            ndimage.distance_transform_edt(
                heights >= lvl,
                sampling=((bounds.max_z - bounds.min_z) / heights.shape[0],
                          (bounds.max_x - bounds.min_x) / heights.shape[1]),
            )
            for lvl in levels
        ]

        for i, level in enumerate(levels):
            rings = rings_by_level[i]
            if i > 0:
                df_prev = above_dist_by_level[i - 1]
                for ring in rings:
                    rows, cols = ring[:, 0], ring[:, 1]
                    radii = ndimage.map_coordinates(df_prev, [rows, cols], order=1, mode="nearest")
                    x, z = _pixel_to_world(rows, cols, heights.shape, bounds)
                    stamps.extend(_place_stamps_along_ring(
                        x, z, radii, level, ring_brush,
                        min_ring_radius, max_ring_radius, ring_spacing_fraction, edge_softness_ratio,
                    ))
            if i < n - 1:
                df_cur = above_dist_by_level[i]
                for ring in rings_by_level[i + 1]:
                    rows, cols = ring[:, 0], ring[:, 1]
                    radii = ndimage.map_coordinates(df_cur, [rows, cols], order=1, mode="nearest")
                    x, z = _pixel_to_world(rows, cols, heights.shape, bounds)
                    stamps.extend(_place_stamps_along_ring(
                        x, z, radii, level, rough_brush,
                        min_ring_radius, max_ring_radius, ring_spacing_fraction, edge_softness_ratio,
                    ))
            completed_steps += 1
            _maybe_report()

        for i, level in enumerate(levels):
            next_level = levels[i + 1] if i < n - 1 else None
            prev_level = levels[i - 1] if i > 0 else None

            capped_up = _capped_hilltop_mask(heights, level, next_level)
            if capped_up.any():
                stamps.extend(_fill_region_greedy(
                    capped_up, heights, bounds,
                    interior_brush, min_interior_radius, max_interior_radius,
                    interior_claim_radius_fraction, edge_softness_ratio,
                ))

            capped_down = _capped_pit_mask(heights, level, prev_level)
            if capped_down.any():
                stamps.extend(_fill_region_greedy(
                    capped_down, heights, bounds,
                    interior_brush, min_interior_radius, max_interior_radius,
                    interior_claim_radius_fraction, edge_softness_ratio,
                ))
            completed_steps += 1
            _maybe_report()

    covered = np.zeros((coverage_resolution, coverage_resolution), dtype=bool)
    _rasterize_coverage(stamps, bounds, coverage_resolution, covered)
    stamps.extend(_residual_fill_pass(
        heights, bounds, coverage_resolution, covered,
        residual_brush, min_residual_radius, max_residual_radius, residual_claim_radius_fraction,
        edge_softness_ratio,
    ))
    completed_steps += 1
    if progress_callback is not None:
        progress_callback(len(stamps), 1.0)

    return stamps
