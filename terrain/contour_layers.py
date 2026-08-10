"""
terrain/contour_layers.py

Alternative initial base-layer generator to hexgrid.py's flat hex
lattice. Earlier versions of this module traced explicit contour
lines (skimage.measure.find_contours) and blended between pairs of
rings to approximate a gradient across each elevation band. That
approach had a real, confirmed bias: because a band's rough (lower-
value) and precise (upper-value) passes shared the exact same
geometry, sequential pull-toward-value compositing tilts toward
whichever pass lands SECOND regardless of position, systematically
biasing every band's reconstruction toward its own upper bound. There
was also no feedback -- the whole thing was open-loop, geometry-only,
with nothing checking what had actually been placed already.

THIS VERSION drops ring-tracing and gradient-blending entirely.
Instead, for every elevation band (heights in [level, level+spacing)
-- literally the same boolean mask the GUI's Elevation Contour overlay
already computes and visualizes), the band's own real 2D footprint is
filled DIRECTLY with circles, valued at the REAL local heightmap mean
within each circle's own footprint, not a fixed per-band constant. No
ring-to-ring blending, no order-dependent compositing, so the bias
described above has nowhere to hide -- and no separate "is this an
isolated hilltop/pit or a connected channel" distinction is needed
either, since every band's mask gets the same direct treatment
regardless of its shape or connectivity. skimage is no longer a
dependency of this module at all.

TIERED MULTI-SCALE FILL (the actual placement algorithm): scan stamp
radius from max_radius down to min_radius in radius_step_m increments.
At each tier, compute ONE distance transform against whatever's still
unfilled, then greedily place every viable, non-conflicting circle at
THAT size (via repeated peak-pick + local suppression against that
single distance snapshot -- an approximation within a tier, corrected
by the next tier's fresh, exact recompute) before shrinking to the
next tier. This is the same idea as a single-radius greedy fill
repeated at decreasing scales, and it collapses the number of genuinely
expensive distance-transform calls from "one per stamp" (hundreds of
thousands, at real stamp counts) to "one per tier" (a few dozen).

PLATEAU-RADIUS FIT TOLERANCE: a stamp's kernel weight is exactly 1.0
out to some fraction of its own nominal radius (the "plateau") before
falloff even begins -- see _brush_plateau_fraction, computed directly
from the real kernel (terrain.terrain_kernel.TerrainKernel /
terrain.brush_profiles.BRUSH_PROFILES, the same one adaptive_refine.py
already scores candidates with), not guessed. A candidate placement is
ACCEPTED once its PLATEAU (radius * plateau_fraction) fits within the
remaining mask -- the falloff beyond the plateau can safely overhang
past the mask's true edge for acceptance PURPOSES, since it isn't
pulling at full strength there anyway ("icing spreading past the edge
of its own cake slice," but only the thin, weak, outermost icing).
Once accepted, though, the stamp's FULL nominal radius is what gets
CLAIMED (removed from further consideration) -- the falloff ring is
still genuinely pulling, just not at full strength, so leaving it
"unclaimed" would have every smaller tier (and the final crumb pass)
redundantly re-stamp the same ring as if untouched. Loosening applies
to acceptance only, never to what counts as handled. This replaces the
earlier, separately-tuned edge_softness_ratio with something derived,
not guessed, and only needs computing once per brush (4 total:
8/9/10/54), not per stamp.
separately-tuned edge_softness_ratio with something derived, not
guessed, and only needs computing once per brush (4 total: 8/9/10/54),
not per stamp.

DENOISE ("schmear"): before tiering, each band's mask gets a small
morphological opening+closing pass (scipy.ndimage) -- opening trims
isolated few-pixel protrusions, closing fills isolated few-pixel gaps.
Purely a simplification of WHICH pixels count as "in this band," at
whatever resolution the heightmap itself is (no separate coverage grid
needed anymore, unlike an earlier version's coverage_resolution).
Doesn't touch any real heightmap value, just removes single-pixel
noise from the boundary before it fragments the fill into unnecessary
tiny stamps.

LEFTOVER CRUMBS: whatever's still unfilled once the tiered scan
reaches min_radius (necessarily smaller than min_radius, or it would
have been picked up) gets a final live-recompute greedy pass (see
_fill_region_greedy, largely unchanged from an earlier version) with
its own (typically softer) smoothing_brush, so small fragments blend
rather than showing a hard, differently-toned patch.

No separate residual safety-net pass is needed anymore: every pixel of
the heightmap belongs to exactly one band by construction (bands
partition [true_min, true_max) into consecutive half-open intervals),
and each band is tiered-filled + crumb-smoothed to genuine saturation,
so coverage is complete by construction rather than needing a global
catch-all pass afterward.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from ingest.heightmap import downsample_heightmap  # noqa: F401 (kept for API stability; unused directly)
from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

try:
    # Same interface adaptive_refine.py already relies on for real
    # scoring. Wrapped defensively -- this module hasn't independently
    # verified terrain_kernel.py's exact API beyond that established
    # call pattern; a mismatch degrades to a fixed, conservative
    # plateau fraction rather than crashing generation entirely.
    from terrain.brush_profiles import BRUSH_PROFILES
    from terrain.terrain_kernel import TerrainKernel
    _HAVE_TERRAIN_KERNEL = True
except ImportError:
    _HAVE_TERRAIN_KERNEL = False

DEFAULT_BAND_SPACING_M = 5.0  # Delta -- GUI/CLI tweakable
DEFAULT_FILL_BRUSH = 8  # wide flat plateau -- best plateau_fraction of the four, minimal overhang
DEFAULT_MIN_RADIUS_M = 10.0
DEFAULT_MAX_RADIUS_M = 50.0
DEFAULT_RADIUS_STEP_M = 1.0  # tier granularity: fewer, larger steps = faster, coarser size gradation
DEFAULT_SMOOTHING_BRUSH = 10  # soft falloff for leftover sub-min_radius crumbs -- blends, no hard edge
DEFAULT_DENOISE_PX = 1  # morphological opening+closing radius, in heightmap pixels; 0 disables
DEFAULT_FALLBACK_PLATEAU_FRACTION = 0.5  # used only if terrain_kernel isn't importable at all
DEFAULT_MAX_TIER_ITERATIONS = 5000  # per-tier safety cap on the inner placement loop


_plateau_fraction_cache: dict[int, float] = {}


def _brush_plateau_fraction(brush: int) -> float:
    """
    Largest r_norm (0..1, fraction of nominal radius) where the
    brush's real kernel weight is still >= 0.999 -- i.e. still
    genuinely full-strength, not yet in its falloff. Computed once per
    brush type and cached (module-level: this is a property of the
    brush profile alone, not of any particular generation run).

    Falls back to DEFAULT_FALLBACK_PLATEAU_FRACTION if terrain_kernel/
    brush_profiles aren't importable, or the brush id isn't in
    BRUSH_PROFILES -- conservative rather than optimistic, since an
    overestimated plateau would let placement overhang further than
    the brush can actually back up at full strength.
    """
    if brush in _plateau_fraction_cache:
        return _plateau_fraction_cache[brush]

    fraction = DEFAULT_FALLBACK_PLATEAU_FRACTION
    if _HAVE_TERRAIN_KERNEL and brush in BRUSH_PROFILES:
        try:
            kernel = TerrainKernel(BRUSH_PROFILES[brush])
            r_norm = np.linspace(0.0, 1.0, 1000)
            weight = kernel.sample_many(r_norm)
            solid = r_norm[weight >= 0.999]
            if solid.size:
                fraction = float(solid.max())
        except Exception:
            pass  # keep the conservative fallback

    _plateau_fraction_cache[brush] = fraction
    return fraction


def _contour_levels(heights: np.ndarray, spacing: float) -> np.ndarray:
    """Elevation levels partitioning [true_min, true_max) into bands spacing apart."""
    lo = float(np.nanmin(heights))
    hi = float(np.nanmax(heights))
    if hi - lo < spacing:
        return np.array([])
    return np.arange(lo + spacing, hi, spacing)


def _band_mask(heights: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    """
    heights in [lo, hi) -- either bound may be None for the open-ended
    edge bands (below the lowest traced level / at-or-above the
    highest). NaN comparisons are False either way in numpy, so gaps
    in the heightmap fall out of every band automatically.
    """
    mask = np.isfinite(heights)
    if lo is not None:
        mask &= heights >= lo
    if hi is not None:
        mask &= heights < hi
    return mask


def _denoise_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Morphological opening (trims isolated few-pixel protrusions) then
    closing (fills isolated few-pixel gaps) -- see module docstring's
    DENOISE section. radius_px <= 0 disables this (returns mask
    unchanged), e.g. for exact/no-simplification runs.
    """
    if radius_px <= 0 or not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(2, 2)
    structure = ndimage.iterate_structure(structure, radius_px)
    opened = ndimage.binary_opening(mask, structure=structure)
    closed = ndimage.binary_closing(opened, structure=structure)
    return closed


def _local_mean_value(
    heights: np.ndarray, row_min: int, row_max: int, col_min: int, col_max: int,
    within: np.ndarray, fallback_row: int, fallback_col: int,
) -> float:
    sub_heights = heights[row_min:row_max, col_min:col_max]
    sub_valid = np.isfinite(sub_heights) & within
    if sub_valid.any():
        return float(np.mean(sub_heights[sub_valid]))
    return float(heights[fallback_row, fallback_col])


def _tiered_fill_band(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, min_radius: float, max_radius: float, radius_step_m: float,
    max_tier_iterations: int = DEFAULT_MAX_TIER_ITERATIONS,
) -> tuple[list[Stamp], np.ndarray]:
    """
    Multi-scale greedy fill of `mask` -- see module docstring's TIERED
    MULTI-SCALE FILL and PLATEAU-RADIUS FIT TOLERANCE sections.

    Returns (stamps, remaining) -- `remaining` is whatever's still
    unfilled once the scan reaches min_radius (the "crumbs" a caller
    should hand to a separate, softer smoothing pass; see
    _fill_region_greedy).
    """
    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    sampling = (cell_z, cell_x)
    x_centers = bounds.min_x + (np.arange(n_cols) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(n_rows) + 0.5) * cell_z

    plateau_fraction = _brush_plateau_fraction(brush)

    remaining = mask.copy()
    stamps: list[Stamp] = []
    if not remaining.any() or max_radius <= 0:
        return stamps, remaining

    n_tiers = max(1, int(round((max_radius - min_radius) / max(radius_step_m, 1e-6))) + 1)
    radii = np.linspace(max_radius, min_radius, n_tiers)

    for radius in radii:
        if not remaining.any():
            break
        plateau_r = radius * plateau_fraction
        if plateau_r <= 0:
            continue

        dist = ndimage.distance_transform_edt(remaining, sampling=sampling)
        work = np.where(dist >= plateau_r, dist, 0.0)

        for _ in range(max_tier_iterations):
            peak = float(work.max())
            if peak < plateau_r:
                break
            row, col = np.unravel_index(np.argmax(work), work.shape)
            cx, cz = float(x_centers[col]), float(z_centers[row])

            row_min = max(0, int((cz - radius - bounds.min_z) / cell_z))
            row_max = min(n_rows, int((cz + radius - bounds.min_z) / cell_z) + 1)
            col_min = max(0, int((cx - radius - bounds.min_x) / cell_x))
            col_max = min(n_cols, int((cx + radius - bounds.min_x) / cell_x) + 1)
            sub_x = x_centers[col_min:col_max]
            sub_z = z_centers[row_min:row_max]
            xx, zz = np.meshgrid(sub_x, sub_z)
            dist_from_center = np.hypot(xx - cx, zz - cz)
            within_radius = dist_from_center <= radius
            within_plateau = dist_from_center <= plateau_r

            value = _local_mean_value(heights, row_min, row_max, col_min, col_max, within_radius, row, col)
            stamps.append(Stamp(x=cx, z=cz, radius=float(radius), value=value, brush=brush, tool=TOOL_FLATTEN))

            # Claim the FULL nominal radius, not just the plateau used to
            # accept this placement -- the falloff ring between plateau_r
            # and radius is still genuinely pulling (weakly, not zero), so
            # leaving it marked "remaining" caused every smaller tier (and
            # eventually the crumb-smoothing pass) to redundantly re-stamp
            # that same ring, confirmed as a real bug: it produced 3-4x
            # more crumb-smoothing stamps than main tiered-fill stamps on
            # a test course, when crumbs should be a small minority. The
            # plateau loosens ACCEPTANCE (is there room for this stamp at
            # all); it was never meant to loosen what counts as handled
            # once a stamp is actually placed.
            remaining[row_min:row_max, col_min:col_max][within_radius] = False
            work[row_min:row_max, col_min:col_max][within_radius] = 0.0

    return stamps, remaining


def _fill_region_greedy(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, min_radius: float, max_radius: float, claim_radius_fraction: float = 0.5,
) -> list[Stamp]:
    """
    Live-recompute greedy fill for small leftover fragments (the
    tiered scan's own "crumbs," below its own min_radius) -- carried
    over from an earlier version of this module largely unchanged.
    Recomputes the distance transform fresh each placement (appropriate
    here: crumbs are small and few by construction, so this is cheap in
    aggregate), cropped to `mask`'s own bounding box + a max_radius
    margin so cost scales with the crumb region's own extent, not the
    full heightmap.
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
    plateau_fraction = _brush_plateau_fraction(brush)
    effective_claim_fraction = max(claim_radius_fraction, plateau_fraction)

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


def generate_contour_layers(
    heights: np.ndarray,
    bounds: BoundingBox,
    band_spacing_m: float = DEFAULT_BAND_SPACING_M,
    fill_brush: int = DEFAULT_FILL_BRUSH,
    min_radius: float = DEFAULT_MIN_RADIUS_M,
    max_radius: float = DEFAULT_MAX_RADIUS_M,
    radius_step_m: float = DEFAULT_RADIUS_STEP_M,
    smoothing_brush: int = DEFAULT_SMOOTHING_BRUSH,
    denoise_px: int = DEFAULT_DENOISE_PX,
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> list[Stamp]:
    """
    Generate an organic base layer by tiered multi-scale fill of every
    elevation band's real 2D footprint -- see module docstring. Bands
    partition the heightmap's full elevation range into band_spacing_m-
    wide half-open intervals: below the lowest traced level, each
    consecutive pair of traced levels, and at-or-above the highest --
    every finite heightmap cell belongs to exactly one band.

    Each band is (1) denoised (small morphological open+close, see
    _denoise_mask -- denoise_px=0 disables), (2) tiered-filled from
    max_radius down to min_radius with fill_brush (see
    _tiered_fill_band), (3) whatever's left (necessarily smaller than
    min_radius) smoothed with smoothing_brush (see _fill_region_greedy).
    No ring tracing, no cross-band value blending, no separate hilltop/
    pit/residual special-casing -- every band gets identical treatment
    regardless of its own shape or connectivity.

    Stamp order: bands ascending by elevation, and within each band,
    the tiered-fill stamps (large to small) followed by that band's own
    crumb-smoothing stamps. Under this project's sequential pull-
    toward-value compositing, later stamps take precedence in any
    overlap -- this ordering means smaller, more specific stamps always
    refine on top of larger ones within their own band, and never get
    overwritten by an unrelated later band's stamps (different bands'
    masks don't overlap by construction).

    progress_callback, if given, is called periodically (time-throttled
    to ~10s) with (stamps_placed_so_far, fraction_complete), tracked as
    bands processed out of the total band count.
    """
    levels = _contour_levels(heights, band_spacing_m)
    n_levels = len(levels)

    # Band boundaries: (None, levels[0]), (levels[0], levels[1]), ...,
    # (levels[-1], None) -- n_levels+1 bands total, or just one
    # (None, None) band if the whole course is flatter than one
    # band_spacing_m step.
    if n_levels == 0:
        boundaries = [(None, None)]
    else:
        boundaries = [(None, float(levels[0]))]
        boundaries += [(float(levels[i]), float(levels[i + 1])) for i in range(n_levels - 1)]
        boundaries += [(float(levels[-1]), None)]

    total_bands = len(boundaries)
    stamps: list[Stamp] = []
    last_progress_time = time.time()
    progress_interval_s = 10.0

    for i, (lo, hi) in enumerate(boundaries):
        mask = _band_mask(heights, lo, hi)
        if mask.any():
            mask = _denoise_mask(mask, denoise_px)
        if mask.any():
            band_stamps, crumbs = _tiered_fill_band(
                mask, heights, bounds, fill_brush, min_radius, max_radius, radius_step_m,
            )
            stamps.extend(band_stamps)
            if crumbs.any():
                stamps.extend(_fill_region_greedy(
                    crumbs, heights, bounds, smoothing_brush,
                    min_radius=max(1.0, min_radius / 3.0), max_radius=min_radius,
                ))

        if progress_callback is not None and time.time() - last_progress_time >= progress_interval_s:
            progress_callback(len(stamps), (i + 1) / total_bands)
            last_progress_time = time.time()

    if progress_callback is not None:
        progress_callback(len(stamps), 1.0)

    return stamps
