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
radius from max_radius down to min_radius in geometric steps (see
_tier_radii -- radius_step_ratio, not a fixed meters step).
At each tier, compute ONE distance transform against whatever's still
unfilled, then greedily place every viable, non-conflicting circle at
THAT size (via repeated peak-pick + local suppression against that
single distance snapshot -- an approximation within a tier, corrected
by the next tier's fresh, exact recompute) before shrinking to the
next tier. This is the same idea as a single-radius greedy fill
repeated at decreasing scales, and it collapses the number of genuinely
expensive distance-transform calls from "one per stamp" (hundreds of
thousands, at real stamp counts) to "one per tier" (a few dozen).

PLATEAU-RADIUS FIT TOLERANCE AND CLAIMING: a stamp's real weight is at
or near true full strength (>= 95% of max) out to some fraction of its
own nominal radius (the "plateau") before falloff begins -- see
_brush_plateau_fraction, computed directly from terrain.brush_profiles.
BRUSH_PROFILES (the real module that loads the game's own measured
brush_profiles.json), not a duplicated copy of that data and not
derived through any separate kernel interface. Both ACCEPTANCE (is
there room for this candidate) and CLAIMING (what counts as handled
once it's placed) use this same real plateau radius.

Getting this right took three attempts, worth recording so it isn't
re-litigated blind: (1) claiming only a plateau fraction derived from
an unverified kernel interface produced near-zero fractions for some
brushes, cascading toward arbitrarily small stamps that never
converged -- packing looked tight, but real kernel-weighted influence
still showed near-zero-influence gaps. (2) Claiming the FULL nominal
radius instead ("1-bit," treating placement as a well-defined circle-
covering problem) fixed that, but is exactly the greedy-optimal setup
for TANGENT packing -- circles touching, not overlapping -- and a
falloff kernel's weight is ~0 right at a tangent point for both
neighbors, confirmed directly against real terrain even with type 73's
hard 0/255 edge, where there should have been no ambiguity at all:
"circles fitted but not overlapping at all." (3) Now that
plateau_fraction comes from the real, measured brush_profiles.json
data -- not near-zero, not guessed -- claiming exactly that (type 8
~0.57, type 9 ~0.46, type 73 ~0.96) is correct: substantial enough
that it doesn't cascade the way attempt (1) did, while still leaving
real, deliberate overlap the way attempt (2)'s tangent packing didn't.
Brushes whose real profile never reaches meaningful full strength
anywhere (type 54: caps at 60% of true full strength; type 10: ~6%
plateau) correctly compute to ~0 and should not be used as fill_brush
for the main tiered pack at all -- see _scatter_fill_remaining, which
doesn't depend on a plateau and is the right tool for those brushes.

DENOISE ("schmear"): before tiering, each band's mask gets a small
morphological opening+closing pass (scipy.ndimage) -- opening trims
isolated few-pixel protrusions, closing fills isolated few-pixel gaps.
Purely a simplification of WHICH pixels count as "in this band," at
whatever resolution the heightmap itself is (no separate coverage grid
needed anymore, unlike an earlier version's coverage_resolution).
Doesn't touch any real heightmap value, just removes single-pixel
noise from the boundary before it fragments the fill into unnecessary
tiny stamps.

LEFTOVER CRUMBS: even 1-bit full-radius claiming leaves genuine gaps
along irregular boundaries -- any spot whose true local space is
smaller than min_radius never gets picked up by the main tiered scan
at all. What's left is thin, scattered, irregular slivers, and trying
to TILE those (many small stamps individually shaped to a sliver's own
sub-meter width) doesn't converge well regardless of how many size
tiers are added underneath min_radius -- an earlier version tried
exactly that (a second tiered pass continuing the size curve down) and
it was still fundamentally the wrong shape of algorithm for this kind
of leftover.

Fixed with a genuinely different algorithm, not more tiers: see
_scatter_fill_remaining. A single FIXED, deliberately oversized radius
(min_radius * DEFAULT_CRUMB_SCATTER_MULTIPLIER) is scattered across
whatever's left with heavy overlap (claims only
DEFAULT_CRUMB_SCATTER_CLAIM_FRACTION of each placement) using
smoothing_brush. Oversizing relative to a sliver's own local width
lets one stamp reach across and cover real length along it, instead of
needing many stamps individually shaped to fit. This sidesteps the
packing problem entirely rather than chasing it to smaller and smaller
tiers.

No separate residual safety-net pass is needed: every pixel of the
heightmap belongs to exactly one band by construction (bands partition
[true_min, true_max) into consecutive half-open intervals), and each
band is filled via tiered pack + scatter to genuine saturation, so
coverage is complete by construction rather than needing a global
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
    # The real thing, not a duplicated/re-derived copy: terrain/
    # brush_profiles.py already loads the game's own real, measured
    # brush_profiles.json once at import time and exposes it as
    # BRUSH_PROFILES[brush_id].samples -- an (N, 2) array of (r, weight)
    # rows, r=0 at the stamp's CENTER through r=1 at its outer edge,
    # weight already normalized 0..1 against TRUE full strength (not
    # each brush's own possibly-lower max -- confirmed: type 54 caps
    # around 0.6, never reaching 1.0 anywhere, matching this project's
    # own earlier documented ~62% amplitude measurement). An earlier
    # version of this module went through terrain.terrain_kernel.
    # TerrainKernel's interface instead, wrapped defensively since that
    # API had never been independently verified -- confirmed as the
    # actual root cause of persistent real-terrain gaps ("circles
    # fitted but not overlapping at all," even with type 73's hard
    # edge, which should have made this unambiguous). A version after
    # that re-embedded the raw JSON as a Python literal here, which
    # just duplicated a file (and now a whole loader module) that
    # already exists in the repo. Importing BRUSH_PROFILES directly
    # fixes both problems at once.
    from terrain.brush_profiles import BRUSH_PROFILES
    _HAVE_BRUSH_PROFILES = True
except ImportError:
    _HAVE_BRUSH_PROFILES = False

DEFAULT_CONSUME_THRESHOLD_FRACTION = 0.95  # of TRUE full strength (1.0), not the brush's own max --
                                             # a brush whose own max never reaches this (type 54: caps
                                             # around 0.6) has NO consumable plateau at all, correctly 0.0

DEFAULT_BAND_SPACING_M = 5.0  # Delta -- GUI/CLI tweakable
DEFAULT_FILL_BRUSH = 8  # wide flat plateau -- best plateau_fraction of the four, minimal overhang
DEFAULT_MIN_RADIUS_M = 10.0
DEFAULT_MAX_RADIUS_M = 50.0
DEFAULT_RADIUS_STEP_RATIO = 0.85  # each tier = previous tier * this; NOT a fixed meters step -- see
                                   # _tier_radii's docstring for why geometric spacing is the right shape
DEFAULT_SMOOTHING_BRUSH = 10  # soft falloff for leftover sub-min_radius crumbs -- blends, no hard edge
DEFAULT_SMOOTHING_FLOOR_M = 1.0  # floor for the SECOND tiered pass (smoothing_brush); below this,
                                  # whatever's left goes to the true one-at-a-time last resort
DEFAULT_CRUMB_SCATTER_MULTIPLIER = 4.0  # crumb scatter radius = min_radius * this -- deliberately
                                          # oversized so one placement reaches across a thin sliver's
                                          # own local width instead of needing many tiny stamps
DEFAULT_CRUMB_SCATTER_CLAIM_FRACTION = 0.5  # heavy overlap for the crumb scatter -- same convention
DEFAULT_DENOISE_PX = 1  # morphological opening+closing radius, in heightmap pixels; 0 disables
DEFAULT_FALLBACK_PLATEAU_FRACTION = 0.5  # used only if terrain.brush_profiles isn't importable at all
DEFAULT_MAX_TIER_ITERATIONS = 2_000_000  # true last-resort backstop against a genuine infinite-loop
                                          # bug, NOT a real limit -- confirmed the previous default of
                                          # 5000 was silently binding in normal operation: a single flat
                                          # band covering the whole course can legitimately need ~15,700
                                          # same-size placements at radius=10, ~63,000 at radius=5. That
                                          # forced every tier to look "exhausted" and shrink to the next
                                          # size far before it actually was, cascading into stamps
                                          # getting smaller far too quickly and leaving real gaps a
                                          # larger stamp could still have filled.


_plateau_fraction_cache: dict[int, float] = {}


def _brush_plateau_fraction(brush: int) -> float:
    """
    Fraction of the stamp's own radius (measured outward from center)
    over which the REAL brush weight is still >= 95% of true full
    strength -- computed directly from terrain.brush_profiles.
    BRUSH_PROFILES[brush].samples, the real per-pixel-scanned data
    (r=0 at center through r=1 at edge, weight already 0..1 against
    true full strength). Computed once per brush type and cached
    (module-level: a property of the brush profile alone).

    Confirmed against the real data: type 8 -> ~0.57, type 9 -> ~0.46,
    type 73 -> ~0.96, type 72 -> ~0.98 -- all closely matching this
    project's own empirically-derived estimates. type 10 -> ~0.06 and
    type 54 -> 0.0 exactly (54 never reaches 95% of true full strength
    ANYWHERE in its profile, capping around 60%) -- both correctly read
    as having no usable consumable plateau, matching "these brushes
    cannot consume" directly. A brush with essentially-zero consume
    fraction should not be used as fill_brush for the main tiered pack
    (see generate_contour_layers) -- use it via _scatter_fill_remaining
    instead, which doesn't depend on a plateau at all.

    Falls back to DEFAULT_FALLBACK_PLATEAU_FRACTION if terrain.
    brush_profiles isn't importable, or the brush id isn't in
    BRUSH_PROFILES (conservative, not optimistic -- an overestimated
    plateau would let placement overhang further than the brush can
    actually back up at full strength).
    """
    if brush in _plateau_fraction_cache:
        return _plateau_fraction_cache[brush]

    profile = BRUSH_PROFILES.get(brush) if _HAVE_BRUSH_PROFILES else None
    if profile is None:
        fraction = DEFAULT_FALLBACK_PLATEAU_FRACTION
    else:
        samples = profile.sorted_samples()  # (N, 2): r ascending 0->1, weight column
        r = samples[:, 0]
        w = samples[:, 1]
        threshold = DEFAULT_CONSUME_THRESHOLD_FRACTION
        if float(w.max()) < threshold:
            # Never reaches meaningfully full strength anywhere in its
            # own profile (e.g. type 54, capping around 0.6) -- no
            # usable plateau to consume, by construction, not by
            # omission.
            fraction = 0.0
        else:
            below = np.nonzero(w < threshold)[0]
            if below.size == 0:
                # Stays at/above threshold all the way to r=1 -- the
                # whole radius is a usable plateau (not expected for
                # any real brush here, but handled rather than assumed
                # away).
                fraction = 1.0
            else:
                fraction = float(r[below[0]])

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


def _tier_radii(
    max_radius: float, min_radius: float, step_ratio: float, min_step_m: float = 1.0,
) -> np.ndarray:
    """
    Curved sequence of tier radii from max_radius down to min_radius:
    each tier's drop is max(current_radius * (1 - step_ratio),
    min_step_m). At large radii the proportional term dominates (100m
    at step_ratio=0.8 drops by 20m -- a big, size-appropriate jump);
    as radius shrinks, the proportional term shrinks with it, and once
    it would fall below min_step_m the floor takes over instead, so
    tiers near min_radius always drop by a real, meaningful amount
    (default 1m) rather than degenerating into wasteful sub-meter
    steps that don't actually change placement decisions.

    This replaces a pure multiplicative sequence (constant RATIO,
    unbounded below) with one that keeps the same "big steps at large
    sizes" shape but adds a floor -- confirmed as necessary: pure
    ratio steps get arbitrarily small near min_radius, spending real
    distance-transform passes on tiers that differ by a fraction of a
    meter and don't meaningfully change what gets placed.

    Falls back to a single max_radius-only tier if the inputs don't
    describe a real descending range (max_radius <= min_radius, or
    step_ratio outside (0, 1)).
    """
    if max_radius <= min_radius or step_ratio <= 0.0 or step_ratio >= 1.0:
        return np.array([max_radius])
    radii = [max_radius]
    while radii[-1] > min_radius:
        current = radii[-1]
        drop = max(current * (1.0 - step_ratio), min_step_m)
        next_radius = current - drop
        if next_radius <= min_radius:
            break
        radii.append(next_radius)
    radii.append(min_radius)
    return np.array(radii)


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
    brush: int, min_radius: float, max_radius: float, radius_step_ratio: float,
    max_stamps: Optional[int] = None,
    max_tier_iterations: int = DEFAULT_MAX_TIER_ITERATIONS,
) -> tuple[list[Stamp], np.ndarray]:
    """
    Multi-scale greedy fill of `mask` -- see module docstring's TIERED
    MULTI-SCALE FILL and PLATEAU-RADIUS FIT TOLERANCE sections.

    max_stamps, if given, is a LOCAL budget for this call only (the
    caller is expected to compute "how many more can this band place"
    from its own running total -- see generate_contour_layers). Once
    reached, returns immediately with whatever's been placed so far;
    `remaining` in that case includes both genuine crumbs AND whatever
    real area the cap cut off early, so it should not be treated as
    "this band is fully accounted for" when max_stamps actually bound.

    Returns (stamps, remaining) -- `remaining` includes both genuine
    crumbs (real area below min_radius) and the real plateau-to-radius
    falloff ring left around every stamp at every tier (see module
    docstring's PLATEAU-RADIUS FIT TOLERANCE AND CLAIMING section) --
    substantial for some brushes (~43% of radius for type 8), not a
    thin sliver, and by design: _scatter_fill_remaining is built to
    handle exactly this.
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
    if not remaining.any() or max_radius <= 0 or (max_stamps is not None and max_stamps <= 0):
        return stamps, remaining

    radii = _tier_radii(max_radius, min_radius, radius_step_ratio)

    for radius in radii:
        if not remaining.any():
            break
        plateau_r = radius * plateau_fraction
        if plateau_r <= 0:
            continue

        dist = ndimage.distance_transform_edt(remaining, sampling=sampling)
        work = np.where(dist >= plateau_r, dist, 0.0)

        for _ in range(max_tier_iterations):
            if max_stamps is not None and len(stamps) >= max_stamps:
                return stamps, remaining
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

            # CLAIM THE REAL PLATEAU, not the full radius. Two earlier
            # versions of this function each got this wrong in a
            # different direction: one claimed only the plateau using a
            # PLATEAU FRACTION DERIVED FROM AN UNVERIFIED KERNEL
            # INTERFACE (never independently confirmed against real
            # data), which produced near-zero fractions for some
            # brushes and cascaded into arbitrarily small stamps that
            # never converged. The next one claimed the FULL radius
            # ("1-bit," reasoning that a well-defined packing problem
            # mattered more than fractional accuracy) -- confirmed wrong
            # too, directly against real terrain: full-radius claiming
            # is exactly the greedy-optimal setup for TANGENT packing
            # (touching, not overlapping), and a cosine-falloff kernel's
            # weight is ~0 right at a tangent point for both neighbors --
            # "circles fitted but not overlapping at all," even with
            # type 73's hard edge, where there should have been no
            # ambiguity at all.
            #
            # Now that plateau_fraction comes from the game's own real
            # measured brush_profiles.json (see _brush_plateau_fraction
            # and this module's header), it's neither near-zero nor a
            # guess: type 8 is genuinely ~0.57, type 73 ~0.96. Claiming
            # THAT (not the full radius) means the next stamp naturally
            # lands closer than tangent -- real, substantial overlap in
            # the outer 5-45% of each stamp's radius, not a degenerate
            # sliver that needs ever-smaller stamps to close.
            remaining[row_min:row_max, col_min:col_max][within_plateau] = False
            work[row_min:row_max, col_min:col_max][within_plateau] = 0.0

    return stamps, remaining


def _scatter_fill_remaining(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, radius: float, claim_radius_fraction: float = 0.5,
    max_stamps: Optional[int] = None,
) -> list[Stamp]:
    """
    Fixed-radius scatter fill for whatever the main tiered pack leaves
    behind -- deliberately NOT a precise pack: leftover area after a
    1-bit tiered fill is thin, scattered, irregular slivers, and trying
    to TILE those (many small stamps individually shaped to a sliver's
    own sub-meter width) doesn't converge well. Instead this places a
    single FIXED, deliberately oversized radius (the caller typically
    passes something several times larger than the main fill's own
    min_radius -- see DEFAULT_CRUMB_SCATTER_MULTIPLIER) with heavy
    overlap: claims only claim_radius_fraction (default 0.5, same
    convention used throughout this project) of each placement, so
    consecutive stamps land close enough to genuinely overlap rather
    than merely touch. Oversizing relative to the crumb's own local
    width lets one stamp "reach across" and cover real LENGTH along a
    thin sliver in a single placement, rather than needing many.

    Placement position uses the same "farthest remaining point" greedy
    heuristic as the main tiered fill (live EDT recompute each pick --
    appropriate here since crumb area should be small by the time this
    runs), but does NOT gate acceptance on whether the full radius (or
    even the plateau) fits -- there's no fitting problem to solve here,
    just "spread coverage out," so every remaining point gets picked in
    turn regardless of how far a stamp's own radius overhangs past it.

    Value is the real local heightmap mean within the placed radius,
    same as every other stamp this module places.
    """
    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    sampling = (cell_z, cell_x)
    x_centers = bounds.min_x + (np.arange(n_cols) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(n_rows) + 0.5) * cell_z

    remaining = mask.copy()
    stamps: list[Stamp] = []
    if not remaining.any() or radius <= 0 or (max_stamps is not None and max_stamps <= 0):
        return stamps

    claim_radius = radius * claim_radius_fraction

    while remaining.any():
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
        dist = ndimage.distance_transform_edt(remaining, sampling=sampling)
        row, col = np.unravel_index(np.argmax(dist), dist.shape)
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
        within_claim = dist_from_center <= claim_radius

        value = _local_mean_value(heights, row_min, row_max, col_min, col_max, within_radius, row, col)
        stamps.append(Stamp(x=cx, z=cz, radius=radius, value=value, brush=brush, tool=TOOL_FLATTEN))

        remaining[row_min:row_max, col_min:col_max][within_claim] = False

    return stamps


def _fill_region_greedy(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, min_radius: float, max_radius: float, claim_radius_fraction: float = 0.5,
    max_stamps: Optional[int] = None,
) -> list[Stamp]:
    """
    Live-recompute greedy fill for small leftover fragments (the
    tiered scan's own "crumbs," below its own min_radius) -- carried
    over from an earlier version of this module largely unchanged.
    Recomputes the distance transform fresh each placement (appropriate
    here: crumbs are small and few by construction, so this is cheap in
    aggregate), cropped to `mask`'s own bounding box + a max_radius
    margin so cost scales with the crumb region's own extent, not the
    full heightmap. max_stamps (local budget, see _tiered_fill_band's
    docstring for the same convention) stops early once reached.
    """
    rows_nz, cols_nz = np.nonzero(mask)
    if rows_nz.size == 0 or (max_stamps is not None and max_stamps <= 0):
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
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
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


def _fill_region_greedy(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, min_radius: float, max_radius: float, claim_radius_fraction: float = 0.5,
    max_stamps: Optional[int] = None,
) -> list[Stamp]:
    """
    Live-recompute greedy fill, locally sized (not fixed-radius): each
    placement's radius comes from the true remaining distance-to-
    boundary at its chosen center, clamped to [min_radius, max_radius].
    Not currently called by generate_contour_layers -- crumb cleanup
    there uses _scatter_fill_remaining (fixed oversized radius, heavy
    overlap) instead, since crumbs are thin scattered slivers that a
    locally-sized pack doesn't handle well (see that function's
    docstring). Kept as a general-purpose utility: appropriate wherever
    a REGION genuinely wants circles sized to its own local geometry
    rather than one fixed size -- e.g. a compact, non-sliver-shaped
    leftover area, unlike this module's own crumbs.
    """
    rows_nz, cols_nz = np.nonzero(mask)
    if rows_nz.size == 0 or (max_stamps is not None and max_stamps <= 0):
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
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
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
    radius_step_ratio: float = DEFAULT_RADIUS_STEP_RATIO,
    smoothing_brush: int = DEFAULT_SMOOTHING_BRUSH,
    denoise_px: int = DEFAULT_DENOISE_PX,
    max_stamps: Optional[int] = None,
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
    _denoise_mask -- denoise_px=0 disables), (2) tiered-packed from
    max_radius down to min_radius with fill_brush, 1-bit full-radius
    claiming (see _tiered_fill_band), (3) whatever's left scattered with
    an oversized, heavily-overlapping smoothing_brush (see
    _scatter_fill_remaining). No ring tracing, no cross-band value
    blending, no separate hilltop/pit/residual special-casing -- every
    band gets identical treatment regardless of its own shape or
    connectivity.

    max_stamps, if given, stops generation as soon as that many stamps
    have been placed in total -- intended as a quick way to sanity-
    check a parameter combination (band_spacing_m, radii, brushes) on a
    partial run before committing to the full one, not as a real
    terrain-generation mode: the result stops mid-band (typically
    partway through the lowest few elevation bands, since bands process
    ascending), so most of the course will genuinely be unfilled, not
    just coarser. Bands ascending means the cutoff always lands on the
    LOW-elevation end -- if you specifically want to preview a band
    somewhere in the middle of the elevation range, this cap can't
    target that; it only ever gives you "however far up from the
    bottom N stamps gets you."

    Stamp order: bands ascending by elevation, and within each band,
    two stages large-to-small: the main tiered-pack stamps (fill_brush,
    max_radius down to min_radius), then the crumb-scatter stamps
    (smoothing_brush, a single fixed oversized radius). Under this
    project's sequential pull-toward-value compositing, later stamps
    take precedence in any overlap -- this ordering means the scatter
    pass always refines on top of the tiered pack within its own band,
    and never gets overwritten by an unrelated later band's stamps
    (different bands' masks don't overlap by construction).

    progress_callback, if given, is called periodically (time-throttled
    to ~10s) with (stamps_placed_so_far, fraction_complete), tracked as
    bands processed out of the total band count -- if max_stamps cuts
    generation off early, the final call still reports whatever
    fraction of bands had actually been reached, not 1.0, so a partial
    run's progress output doesn't misleadingly claim completion.
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
    bands_reached = 0

    for i, (lo, hi) in enumerate(boundaries):
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
        bands_reached = i + 1

        mask = _band_mask(heights, lo, hi)
        if mask.any():
            mask = _denoise_mask(mask, denoise_px)
        if mask.any():
            tier_budget = None if max_stamps is None else max_stamps - len(stamps)
            band_stamps, crumbs = _tiered_fill_band(
                mask, heights, bounds, fill_brush, min_radius, max_radius, radius_step_ratio,
                max_stamps=tier_budget,
            )
            stamps.extend(band_stamps)

            if crumbs.any():
                # Scatter fill (smoothing_brush), oversized and heavily
                # overlapping -- see _scatter_fill_remaining's docstring.
                # Replaces an earlier two-stage design (a second tiered
                # pack continuing the size curve down, then a live-
                # recompute last resort): that still tried to PACK the
                # leftover precisely, which is the wrong shape of
                # algorithm for thin scattered slivers regardless of how
                # many stages it gets. Scattering a single oversized
                # radius with heavy overlap sidesteps the packing problem
                # entirely instead of chasing it to smaller and smaller
                # tiers.
                crumb_budget = None if max_stamps is None else max_stamps - len(stamps)
                if crumb_budget is None or crumb_budget > 0:
                    crumb_radius = min_radius * DEFAULT_CRUMB_SCATTER_MULTIPLIER
                    stamps.extend(_scatter_fill_remaining(
                        crumbs, heights, bounds, smoothing_brush, crumb_radius,
                        claim_radius_fraction=DEFAULT_CRUMB_SCATTER_CLAIM_FRACTION,
                        max_stamps=crumb_budget,
                    ))

        if progress_callback is not None and time.time() - last_progress_time >= progress_interval_s:
            progress_callback(len(stamps), (i + 1) / total_bands)
            last_progress_time = time.time()

    if progress_callback is not None:
        # Real fraction reached, not a fake 1.0/0.0 -- a max_stamps cutoff
        # partway through band 3 of 10 should report 0.3, not claim either
        # full completion or that nothing happened.
        progress_callback(len(stamps), bands_reached / total_bands)

    return stamps
