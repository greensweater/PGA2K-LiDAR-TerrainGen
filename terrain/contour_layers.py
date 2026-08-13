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
closing pass (fills isolated few-pixel gaps) followed by connected-
component AREA filtering (scipy.ndimage.label + a pixel-count
threshold, not morphological opening -- see _denoise_mask's docstring
for why: opening erases by WIDTH, which was confirmed to delete real,
thin, connected terrain features before any fill stage ever saw them,
reading as true zero influence rather than weak coverage). Purely a
simplification of WHICH pixels count as "in this band," at whatever
resolution the heightmap itself is (no separate coverage grid needed
anymore, unlike an earlier version's coverage_resolution). Doesn't
touch any real heightmap value, just removes small ISOLATED noise
specks (by area) from the boundary before they fragment the fill into
unnecessary tiny stamps -- long thin real features, however narrow,
are left alone.

LEFTOVER CRUMBS: even with the real, substantial plateau claimed at
every tier (see PLATEAU-RADIUS FIT TOLERANCE AND CLAIMING above),
genuine gaps remain along irregular boundaries -- any spot whose true
local space is smaller than min_radius never gets picked up by the
main tiered scan at all, and the plateau-to-radius falloff ring around
every stamp adds further leftover on top of that. What's left is thin,
scattered, irregular slivers, and trying to TILE those (many small
stamps individually shaped to a sliver's own sub-meter width) doesn't
converge well regardless of how many size tiers are added underneath
min_radius -- an earlier version tried exactly that (a second tiered
pass continuing the size curve down) and it was still fundamentally
the wrong shape of algorithm for this kind of leftover.

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

import os
import time
from concurrent.futures import ProcessPoolExecutor
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
DEFAULT_EDGE_DISTANCE_M = 0.0  # pass-1-only buffer past the true band boundary (see _poisson_pack_band);
                                 # 0 disables. The crumb pass (pass 2) always ignores this.
DEFAULT_SMOOTHING_BRUSH = 10  # soft falloff for whatever pass 1 leaves as genuine crumbs
DEFAULT_SMOOTHING_MIN_RADIUS_M = 4.0  # pass 2's OWN radius floor -- independent of the main pack's
                                        # min_radius; the crumb stage's scale should be tuned on its own
DEFAULT_SMOOTHING_FLOOR_M = 1.0  # floor for the SECOND tiered pass (smoothing_brush); below this,
                                  # whatever's left goes to the true one-at-a-time last resort
DEFAULT_CRUMB_SCATTER_MULTIPLIER = 4.0  # crumb scatter radius = smoothing_min_radius * this --
                                          # deliberately oversized so one placement reaches across a
                                          # thin sliver's own local width instead of needing many tiny
                                          # stamps; "spread," per the 2-4x range this was tuned against
DEFAULT_SMOOTH_CLAIM_FRACTION = 0.25  # "eat" -- how much of each crumb-scatter stamp's placed radius
                                        # gets claimed; deliberately much heavier overlap (claim less)
                                        # than the main pack's own claim (its real plateau fraction),
                                        # since the crumb pass's whole job is blanket-covering whatever
                                        # the main pack's large hard stamps couldn't reach
DEFAULT_CANDIDATES_PER_RADIUS = 2000  # starting point for auto-tuning (see _auto_tune_candidates), or
                                        # used directly if candidates_per_radius is set explicitly
DEFAULT_SWEET_SPOT_STAMP_RATIO = 0.10  # auto-tune target: keep doubling candidates_per_radius as
                                         # long as each doubling reduces a band's own uncovered-area
                                         # fraction by at least this much, RELATIVE to the previous
                                         # doubling's own uncovered fraction (0.10 = stop once another
                                         # doubling buys less than a 10% relative improvement) -- NOT
                                         # an absolute area target (see _auto_tune_candidates' docstring
                                         # for why: pass 1 has a genuine structural floor -- real area
                                         # smaller than min_radius -- that no candidate count can ever
                                         # close, so a fixed target can be unreachable regardless of
                                         # true coverage quality, exhausting the search uselessly)
DEFAULT_SWEET_SPOT_SAMPLE_BANDS = 3  # how many regularly-spaced bands to calibrate against, not every
                                       # band -- bands are similar enough in character that per-band
                                       # tuning would mostly repeat the same search for no benefit
DEFAULT_SWEET_SPOT_SEEDS = 2  # random seeds per sampled band, so one lucky/unlucky seed doesn't skew
                                # the calibration -- the MAX candidate count found across every
                                # band/seed combination is what actually gets used for the real run
DEFAULT_SWEET_SPOT_MAX_CANDIDATES = 50_000  # safety cap on the auto-tune search itself
DEFAULT_SWEET_SPOT_TIME_BUDGET_S = 60.0  # hard wall-clock ceiling on the whole auto-tune search --
                                           # once exceeded, returns whatever's best so far rather than
                                           # letting calibration dominate total run time unpredictably
DEFAULT_RANDOM_SEED = 1
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
    Closing (fills isolated few-pixel gaps) then AREA-based removal of
    small isolated components (pixel count, not width) -- see module
    docstring's DENOISE section. radius_px <= 0 disables this (returns
    mask unchanged), e.g. for exact/no-simplification runs.

    Deliberately NOT morphological opening (an earlier version used
    it): opening erodes-then-dilates, which erases anything NARROWER
    than its structuring element regardless of how long or real that
    feature is -- confirmed as a real bug against actual terrain: a
    genuinely thin but real, connected tendril of a band got erased
    entirely before any fill stage ever saw it, reading as true zero
    influence (not weak/under-filled -- the area was never a candidate
    at all). Filtering by connected-component AREA instead only
    removes small, ISOLATED specks (true single-pixel noise), leaving
    long thin real features alone no matter how narrow they are.
    """
    if radius_px <= 0 or not mask.any():
        return mask

    structure = ndimage.generate_binary_structure(2, 2)
    structure = ndimage.iterate_structure(structure, radius_px)
    closed = ndimage.binary_closing(mask, structure=structure)

    min_area = max(1, (2 * radius_px + 1) ** 2 // 2)  # same rough scale as the old structuring element
    labeled, n = ndimage.label(closed)
    if n == 0:
        return closed
    sizes = ndimage.sum(closed, labeled, index=np.arange(1, n + 1))
    keep_labels = np.nonzero(sizes >= min_area)[0] + 1
    return np.isin(labeled, keep_labels)


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


def _poisson_pack_band(
    mask: np.ndarray, heights: np.ndarray, bounds: BoundingBox,
    brush: int, max_radius: float, min_radius: float, radius_step_ratio: float,
    edge_distance_m: float, candidates_per_radius: int, rng: np.random.Generator,
    max_stamps: Optional[int] = None,
) -> tuple[list[Stamp], np.ndarray]:
    """
    Fast first pass ("chunky poisson fill"): a STATIC distance field
    (computed once for the whole band, not per tier), random-candidate
    sampling capped at candidates_per_radius per tier, and a spatial
    hash grid for O(1)-amortized overlap rejection -- architecturally
    different from _tiered_fill_band's exhaustive argmax-per-tier
    approach, trading a per-call completeness guarantee for real speed.
    Not the whole story on its own: generate_contour_layers always
    follows this with _scatter_fill_remaining over whatever this
    leaves uncovered, so the lack of a guarantee here doesn't cost
    overall coverage -- see that function and this module's LEFTOVER
    CRUMBS docstring section.

    ACCEPTANCE AND OVERLAP are both based on each stamp's real PLATEAU
    radius (radius * _brush_plateau_fraction(brush)), not its full
    nominal radius -- "let the stamp itself do the work of overlapping"
    via the real falloff geometry beyond the plateau, rather than a
    separate tuned overlap parameter on top of it. Two accepted
    stamps' PLATEAUS are never allowed to intersect; their full
    nominal radii (which extend further) naturally do, which is what
    produces real blending in the rendered terrain. Intended for hard,
    high-plateau brushes (type 8/73) -- a brush with no real plateau
    (_brush_plateau_fraction <= 0, e.g. type 10/54) can't do this
    pass's job at all and is refused outright (returns everything as
    crumbs for pass 2 instead of silently placing zero-claim stamps).

    edge_distance_m (this pass only -- _scatter_fill_remaining ignores
    it entirely) requires each candidate's plateau to additionally
    clear the true band boundary by this much, not just fit within it
    -- deliberately leaves a buffer strip along every band edge for
    the crumb pass to handle instead, so this pass's large, hard
    stamps never come close enough to a real boundary for fit-test
    tolerance to matter.

    Returns (stamps, crumbs) -- `crumbs` is computed from the REAL
    accepted stamps' own plateau footprints (not the static distance
    field, which never updates), so it reflects genuine leftover area,
    the same contract _tiered_fill_band's return value has.
    """
    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    sampling = (cell_z, cell_x)

    stamps: list[Stamp] = []
    if not mask.any() or max_radius <= 0 or (max_stamps is not None and max_stamps <= 0):
        return stamps, mask.copy()

    plateau_fraction = _brush_plateau_fraction(brush)
    if plateau_fraction <= 0:
        return stamps, mask.copy()

    # Crop to mask's own bounding box + a max_radius margin (same
    # convention as _tiered_fill_band/_scatter_fill_remaining) so the
    # one-time static EDT and every candidate lookup scale with the
    # band's own extent, not the whole heightmap.
    rows_nz, cols_nz = np.nonzero(mask)
    margin_x = int(np.ceil(max_radius / cell_x)) + 2
    margin_z = int(np.ceil(max_radius / cell_z)) + 2
    row0 = max(0, int(rows_nz.min()) - margin_z)
    row1 = min(n_rows, int(rows_nz.max()) + margin_z + 1)
    col0 = max(0, int(cols_nz.min()) - margin_x)
    col1 = min(n_cols, int(cols_nz.max()) + margin_x + 1)
    mask_crop = mask[row0:row1, col0:col1]
    crop_rows, crop_cols = mask_crop.shape

    crop_x_centers = bounds.min_x + (col0 + np.arange(crop_cols) + 0.5) * cell_x
    crop_z_centers = bounds.min_z + (row0 + np.arange(crop_rows) + 0.5) * cell_z

    # ONE static distance field for the whole pass -- not recomputed
    # per tier, unlike _tiered_fill_band. Every tier just filters this
    # by its own (plateau_r + edge_distance_m) threshold.
    dist_static = ndimage.distance_transform_edt(mask_crop, sampling=sampling)

    radii = _tier_radii(max_radius, min_radius, radius_step_ratio)

    # Spatial hash for overlap rejection -- cell size covers the
    # largest possible interaction distance (two max-radius plateaus),
    # same convention as the reference tool this pass is based on.
    cell_size = max(cell_x, cell_z) + 2 * max_radius * plateau_fraction
    grid: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    max_plateau_seen = max_radius * plateau_fraction  # tight upper bound: radii process largest-first

    def cell_key(x: float, z: float) -> tuple[int, int]:
        return (int(x // cell_size), int(z // cell_size))

    def conflicts(x: float, z: float, plateau_r: float) -> bool:
        search_r = plateau_r + max_plateau_seen
        min_gx = int((x - search_r) // cell_size)
        max_gx = int((x + search_r) // cell_size)
        min_gz = int((z - search_r) // cell_size)
        max_gz = int((z + search_r) // cell_size)
        for gz in range(min_gz, max_gz + 1):
            for gx in range(min_gx, max_gx + 1):
                for ox, oz, o_plateau_r in grid.get((gx, gz), ()):
                    min_dist = plateau_r + o_plateau_r
                    if (x - ox) ** 2 + (z - oz) ** 2 < min_dist * min_dist:
                        return True
        return False

    for radius in radii:
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
        plateau_r = radius * plateau_fraction
        if plateau_r <= 0:
            continue
        threshold = plateau_r + max(0.0, edge_distance_m)

        eligible_rows, eligible_cols = np.nonzero(dist_static >= threshold)
        if eligible_rows.size == 0:
            continue

        n_pick = min(candidates_per_radius, eligible_rows.size)
        pick_idx = rng.choice(eligible_rows.size, size=n_pick, replace=False)

        for i in pick_idx:
            if max_stamps is not None and len(stamps) >= max_stamps:
                break
            r_local, c_local = int(eligible_rows[i]), int(eligible_cols[i])
            cx = float(crop_x_centers[c_local])
            cz = float(crop_z_centers[r_local])

            if conflicts(cx, cz, plateau_r):
                continue

            row_full, col_full = r_local + row0, c_local + col0
            row_min = max(0, int((cz - radius - bounds.min_z) / cell_z))
            row_max = min(n_rows, int((cz + radius - bounds.min_z) / cell_z) + 1)
            col_min = max(0, int((cx - radius - bounds.min_x) / cell_x))
            col_max = min(n_cols, int((cx + radius - bounds.min_x) / cell_x) + 1)
            sub_x = bounds.min_x + (np.arange(col_min, col_max) + 0.5) * cell_x
            sub_z = bounds.min_z + (np.arange(row_min, row_max) + 0.5) * cell_z
            xx, zz = np.meshgrid(sub_x, sub_z)
            within_radius = np.hypot(xx - cx, zz - cz) <= radius
            value = _local_mean_value(
                heights, row_min, row_max, col_min, col_max, within_radius, row_full, col_full,
            )

            stamps.append(Stamp(x=cx, z=cz, radius=float(radius), value=value, brush=brush, tool=TOOL_FLATTEN))
            grid.setdefault(cell_key(cx, cz), []).append((cx, cz, plateau_r))

    # Rasterize what pass 1 actually claimed (real plateau footprints,
    # not the never-updated static field) to compute genuine crumbs.
    covered = np.zeros_like(mask)
    if stamps:
        full_x_centers = bounds.min_x + (np.arange(n_cols) + 0.5) * cell_x
        full_z_centers = bounds.min_z + (np.arange(n_rows) + 0.5) * cell_z
        for s in stamps:
            s_plateau_r = s.radius * plateau_fraction
            row_min = max(0, int((s.z - s_plateau_r - bounds.min_z) / cell_z))
            row_max = min(n_rows, int((s.z + s_plateau_r - bounds.min_z) / cell_z) + 1)
            col_min = max(0, int((s.x - s_plateau_r - bounds.min_x) / cell_x))
            col_max = min(n_cols, int((s.x + s_plateau_r - bounds.min_x) / cell_x) + 1)
            sub_x = full_x_centers[col_min:col_max]
            sub_z = full_z_centers[row_min:row_max]
            xx, zz = np.meshgrid(sub_x, sub_z)
            within = np.hypot(xx - s.x, zz - s.z) <= s_plateau_r
            covered[row_min:row_max, col_min:col_max] |= within

    crumbs = mask & ~covered
    return stamps, crumbs


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

    # Crop the working region to `mask`'s own bounding box + a max_radius
    # margin (same convention _fill_region_greedy already established) --
    # confirmed as the dominant real cost otherwise: the inner placement
    # loop's own work.max()/argmax were scanning the FULL heightmap-sized
    # array on every single stamp placement, not just once per tier,
    # regardless of how small the band actually is. Measured directly: a
    # single tier can legitimately need ~28,000 placements on a real
    # course, which at full 2000x2000 array size means ~28,000 x
    # 4,000,000 element scans for ONE tier alone. `remaining_crop` is a
    # VIEW into `remaining` (basic slicing, not fancy indexing), so
    # mutations made through either one stay in sync automatically --
    # no separate write-back step needed.
    rows_nz, cols_nz = np.nonzero(remaining)
    margin_x = int(np.ceil(max_radius / cell_x)) + 2
    margin_z = int(np.ceil(max_radius / cell_z)) + 2
    row0 = max(0, int(rows_nz.min()) - margin_z)
    row1 = min(n_rows, int(rows_nz.max()) + margin_z + 1)
    col0 = max(0, int(cols_nz.min()) - margin_x)
    col1 = min(n_cols, int(cols_nz.max()) + margin_x + 1)
    remaining_crop = remaining[row0:row1, col0:col1]

    radii = _tier_radii(max_radius, min_radius, radius_step_ratio)

    for radius in radii:
        if not remaining_crop.any():
            break
        plateau_r = radius * plateau_fraction
        if plateau_r <= 0:
            continue

        dist = ndimage.distance_transform_edt(remaining_crop, sampling=sampling)
        work = np.where(dist >= plateau_r, dist, 0.0)

        for _ in range(max_tier_iterations):
            if max_stamps is not None and len(stamps) >= max_stamps:
                return stamps, remaining
            peak = float(work.max())
            if peak < plateau_r:
                break
            row_c, col_c = np.unravel_index(np.argmax(work), work.shape)
            row, col = row_c + row0, col_c + col0  # crop-local -> full-array indices
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
            # `work` is crop-local; the claim window computed above is
            # in full-array coordinates and is guaranteed to fall
            # entirely within the crop (radius <= max_radius, and the
            # crop's own margin was sized to max_radius), so a plain
            # offset by (row0, col0) is enough -- no bounds clipping
            # needed.
            work[row_min - row0:row_max - row0, col_min - col0:col_max - col0][within_plateau] = 0.0

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

    # Crop to `mask`'s own bounding box + a radius margin -- see
    # _tiered_fill_band's matching comment for why this matters: an
    # earlier version of this function recomputed a FULL heightmap-sized
    # distance transform on EVERY single iteration, with no cropping at
    # all (worse than _tiered_fill_band's own pre-fix issue, since an
    # EDT is more expensive per call than a max/argmax, and this runs
    # once per band). `remaining_crop` is a VIEW into `remaining`, so
    # mutations through either stay in sync automatically.
    rows_nz, cols_nz = np.nonzero(remaining)
    margin_x = int(np.ceil(radius / cell_x)) + 2
    margin_z = int(np.ceil(radius / cell_z)) + 2
    row0 = max(0, int(rows_nz.min()) - margin_z)
    row1 = min(n_rows, int(rows_nz.max()) + margin_z + 1)
    col0 = max(0, int(cols_nz.min()) - margin_x)
    col1 = min(n_cols, int(cols_nz.max()) + margin_x + 1)
    remaining_crop = remaining[row0:row1, col0:col1]

    claim_radius = radius * claim_radius_fraction
    # Floor the claim radius at ~1.5 grid cells -- confirmed as a real,
    # reproducible pathology otherwise: if claim_radius shrinks below
    # one cell (e.g. a small claim_radius_fraction combined with a
    # coarse heightmap resolution, or just an aggressive "eat" tuning),
    # a stamp placed exactly at a cell center no longer reaches ANY
    # neighboring cell's own center, so the claim only ever removes the
    # single cell it was placed on -- forcing one stamp per remaining
    # cell instead of one stamp covering many. Measured directly: 1,875
    # stamps for 1,875 crumb cells (a 1:1 ratio) at a resolution where
    # claim_radius (4m) was smaller than a grid cell (6.67m), versus the
    # intended "one large stamp reaches across many cells" behavior.
    # This floor doesn't change claim_radius_fraction's real intent at
    # any resolution fine enough for it to matter -- it only prevents
    # the degenerate case where the tuned value can't make progress at
    # the grid's own resolution at all.
    min_cell = 1.5 * max(cell_x, cell_z)
    if claim_radius < min_cell:
        claim_radius = min_cell

    while remaining_crop.any():
        if max_stamps is not None and len(stamps) >= max_stamps:
            break
        dist = ndimage.distance_transform_edt(remaining_crop, sampling=sampling)
        row_c, col_c = np.unravel_index(np.argmax(dist), dist.shape)
        row, col = row_c + row0, col_c + col0  # crop-local -> full-array indices
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


def _auto_tune_candidates(
    heights: np.ndarray,
    bounds: BoundingBox,
    boundaries: list[tuple[Optional[float], Optional[float]]],
    fill_brush: int,
    max_radius: float,
    min_radius: float,
    radius_step_ratio: float,
    edge_distance_m: float,
    denoise_px: int,
    sweet_spot_ratio: float,
    sample_band_count: int,
    seeds_per_band: int,
    initial_candidates: int,
    max_candidates: int,
    time_budget_s: float = 60.0,
) -> int:
    """
    Calibrates a single candidates_per_radius value for the WHOLE
    course by searching for the point of diminishing returns on a
    handful of regularly-spaced sample bands, rather than tuning every
    band individually -- bands are similar enough in character that
    per-band tuning would mostly repeat the same search for no benefit.

    For each sampled band and each of seeds_per_band random seeds
    (so one lucky/unlucky seed doesn't skew the result), doubles
    candidates_per_radius starting from the search's own running best
    (see WARM-STARTING below) until another doubling's improvement in
    the band's own uncovered AREA fraction drops to sweet_spot_ratio or
    below (a RELATIVE, diminishing-returns criterion -- see DIMINISHING
    RETURNS below, not an absolute target). The MAX candidate count
    found across every sampled band/seed combination is returned -- a
    single global value, not tuned per band, since the real run needs
    one number that works safely everywhere, not just on the bands it
    happened to sample.

    AREA, not stamp count: an earlier version compared pass 2's own
    stamp count against pass 1's instead -- confirmed wrong, because
    that conflates two unrelated things. Stamp count ratio depends
    heavily on the RADIUS SCALE MISMATCH between the two passes (pass 1
    placing few large stamps, pass 2 needing many small ones for even a
    tiny leftover sliver), not on how much real area pass 1 actually
    left uncovered -- confirmed directly: auto-tune hit its max
    candidate cap without ever satisfying a stamp-count-ratio target,
    even where pass 1's own coverage was already good, simply because
    pass 1 and pass 2's radii were scaled very differently. Measuring
    the crumb mask's own area fraction instead is invariant to that
    mismatch, directly measures what actually matters (how much of the
    band pass 1 left behind), and is cheaper too -- pass 2 doesn't need
    to run at all during calibration, only pass 1's own crumb output.

    DIMINISHING RETURNS, not an absolute target: a version after that
    stopped once uncovered_fraction dropped below a FIXED threshold --
    also confirmed wrong, because pass 1 has a genuine STRUCTURAL floor
    (real area smaller than min_radius, which no candidate count can
    ever close) that can sit above any fixed target regardless of how
    well pass 1 is actually doing. Measured directly on one real band:
    uncovered area plateaued at ~15.67% starting around 8,000 candidates
    and never improved further even at 50,000 -- a fixed target simply
    unreachable there, so the search exhausted to the cap uselessly
    every single time. Comparing each doubling's improvement against
    the PREVIOUS doubling instead correctly recognizes "more candidates
    stopped helping," regardless of what the achievable floor actually
    is for that particular band's geometry.

    WARM-STARTING: each trial after the first starts its own doubling
    search from half the best candidate count found by any PRIOR trial
    (never below initial_candidates), not always from scratch -- bands
    are similar enough that a value which worked well for one band is a
    much better starting guess for the next than restarting at the
    floor every time. Confirmed necessary: naive from-scratch doubling
    across every sampled band/seed combination took ~113s on even a
    small test course, dominated by repeatedly re-discovering the same
    ballpark candidate count.

    time_budget_s is a hard wall-clock ceiling on the WHOLE search
    (default 60s) -- once exceeded, whatever the best candidate count
    found so far is gets returned immediately, rather than the
    calibration step being able to dominate total run time
    unpredictably on unusually large or complex terrain. Not a
    per-trial timeout; a running total across every sampled band/seed.
    """
    if not boundaries:
        return initial_candidates

    sample_positions = np.linspace(0, len(boundaries) - 1, min(sample_band_count, len(boundaries)))
    sample_indices = sorted(set(int(round(p)) for p in sample_positions))

    best_candidates = initial_candidates
    start_time = time.time()

    for bi in sample_indices:
        if time.time() - start_time > time_budget_s:
            break
        lo, hi = boundaries[bi]
        mask = _band_mask(heights, lo, hi)
        if denoise_px > 0:
            mask = _denoise_mask(mask, denoise_px)
        band_area = int(mask.sum())
        if band_area == 0:
            continue

        for seed_offset in range(seeds_per_band):
            if time.time() - start_time > time_budget_s:
                break
            seed = DEFAULT_RANDOM_SEED + seed_offset
            candidates = max(initial_candidates, best_candidates // 2)
            prev_uncovered: Optional[float] = None
            while candidates <= max_candidates:
                if time.time() - start_time > time_budget_s:
                    break
                rng = np.random.default_rng(seed)
                pass1_stamps, crumbs = _poisson_pack_band(
                    mask, heights, bounds, fill_brush, max_radius, min_radius, radius_step_ratio,
                    edge_distance_m, candidates, rng,
                )
                if not pass1_stamps:
                    break
                uncovered_fraction = float(crumbs.sum()) / band_area

                # DIMINISHING RETURNS, not an absolute target: an earlier
                # version stopped once uncovered_fraction dropped below a
                # fixed threshold -- confirmed wrong, because pass 1 has
                # a genuine STRUCTURAL floor (real area smaller than
                # min_radius, which no candidate count can ever reach)
                # that can sit above any fixed target regardless of how
                # well pass 1 is actually doing. Measured directly:
                # uncovered area plateaued at ~15.67% starting around
                # 8,000 candidates and never improved further even at
                # 50,000 -- a fixed 2% target made every search exhaust
                # to the cap uselessly. Comparing each doubling's
                # improvement against the PREVIOUS doubling instead
                # correctly recognizes "more candidates stopped helping"
                # regardless of what the achievable floor actually is.
                if prev_uncovered is not None and prev_uncovered > 0:
                    relative_improvement = (prev_uncovered - uncovered_fraction) / prev_uncovered
                    if relative_improvement <= sweet_spot_ratio:
                        break
                prev_uncovered = uncovered_fraction
                candidates *= 2
            best_candidates = max(best_candidates, min(candidates, max_candidates))

    return best_candidates


DEFAULT_N_WORKERS = None  # None = auto (os.cpu_count()); see generate_contour_layers' own docstring

# Per-worker-process globals, set once by _init_band_worker -- avoids
# re-pickling the (potentially large, e.g. 2000x2000 floats = 32MB)
# heights array and bounds for every single band task. Each worker
# process gets ONE copy at pool startup; every band task after that
# just references these directly.
_worker_heights: Optional[np.ndarray] = None
_worker_bounds: Optional[BoundingBox] = None


def _init_band_worker(heights: np.ndarray, bounds: BoundingBox) -> None:
    global _worker_heights, _worker_bounds
    _worker_heights = heights
    _worker_bounds = bounds


def _process_one_band(args: tuple) -> list[Stamp]:
    """
    Top-level (picklable) per-band worker: everything a single call to
    generate_contour_layers' own sequential loop body does for ONE
    band -- mask, denoise, pass 1, pass 2 -- packaged so it can run in
    a separate process. Reads heights/bounds from the per-process
    globals set by _init_band_worker, not from `args`, to avoid
    re-pickling them per task (see that function's own docstring).

    Bands never spatially overlap by construction (different bands'
    masks partition the heightmap into disjoint elevation ranges), so
    this is safe to run fully independently, with no shared state and
    no coordination needed between bands -- confirmed throughout this
    module's own design, not a new assumption introduced for
    parallelism specifically.
    """
    (band_index, lo, hi, fill_brush, min_radius, max_radius, radius_step_ratio, edge_distance_m,
     denoise_px, random_seed, candidates_per_radius, smoothing_brush, crumb_radius,
     smooth_claim_fraction) = args

    heights = _worker_heights
    bounds = _worker_bounds
    assert heights is not None and bounds is not None  # _init_band_worker must have run first

    mask = _band_mask(heights, lo, hi)
    if mask.any():
        mask = _denoise_mask(mask, denoise_px)
    if not mask.any():
        return []

    rng = np.random.default_rng(random_seed + band_index)
    pass1_stamps, crumbs = _poisson_pack_band(
        mask, heights, bounds, fill_brush, max_radius, min_radius, radius_step_ratio,
        edge_distance_m, candidates_per_radius, rng,
    )
    stamps = list(pass1_stamps)

    if crumbs.any():
        stamps.extend(_scatter_fill_remaining(
            crumbs, heights, bounds, smoothing_brush, crumb_radius,
            claim_radius_fraction=smooth_claim_fraction,
        ))

    return stamps


def generate_contour_layers(
    heights: np.ndarray,
    bounds: BoundingBox,
    band_spacing_m: float = DEFAULT_BAND_SPACING_M,
    fill_brush: int = DEFAULT_FILL_BRUSH,
    min_radius: float = DEFAULT_MIN_RADIUS_M,
    max_radius: float = DEFAULT_MAX_RADIUS_M,
    radius_step_ratio: float = DEFAULT_RADIUS_STEP_RATIO,
    edge_distance_m: float = DEFAULT_EDGE_DISTANCE_M,
    smoothing_brush: int = DEFAULT_SMOOTHING_BRUSH,
    smoothing_min_radius: float = DEFAULT_SMOOTHING_MIN_RADIUS_M,
    smooth_ratio: float = DEFAULT_CRUMB_SCATTER_MULTIPLIER,
    smooth_claim_fraction: float = DEFAULT_SMOOTH_CLAIM_FRACTION,
    candidates_per_radius: Optional[int] = None,
    sweet_spot_ratio: float = DEFAULT_SWEET_SPOT_STAMP_RATIO,
    sweet_spot_sample_bands: int = DEFAULT_SWEET_SPOT_SAMPLE_BANDS,
    sweet_spot_seeds: int = DEFAULT_SWEET_SPOT_SEEDS,
    sweet_spot_max_candidates: int = DEFAULT_SWEET_SPOT_MAX_CANDIDATES,
    sweet_spot_time_budget_s: float = DEFAULT_SWEET_SPOT_TIME_BUDGET_S,
    random_seed: int = DEFAULT_RANDOM_SEED,
    denoise_px: int = DEFAULT_DENOISE_PX,
    max_stamps: Optional[int] = None,
    n_workers: Optional[int] = DEFAULT_N_WORKERS,
    progress_callback: Optional[Callable[[int, float], None]] = None,
    on_candidates_tuned: Optional[Callable[[int], None]] = None,
) -> list[Stamp]:
    """
    Generate an organic base layer via two passes per elevation band --
    see module docstring. Bands partition the heightmap's full
    elevation range into band_spacing_m-wide half-open intervals: below
    the lowest traced level, each consecutive pair of traced levels,
    and at-or-above the highest -- every finite heightmap cell belongs
    to exactly one band.

    Each band is (1) denoised (see _denoise_mask -- denoise_px=0
    disables), (2) PASS 1: fast random-candidate poisson pack with
    fill_brush (see _poisson_pack_band) -- a hard, high-plateau brush
    (type 8/73) doing the bulk of the work quickly, trading a per-call
    completeness guarantee for speed, (3) PASS 2: whatever pass 1
    leaves as genuine crumbs gets an oversized, heavily-overlapping
    scatter fill with smoothing_brush (see _scatter_fill_remaining),
    which IS exhaustive -- so overall coverage is still complete by
    construction, just split across a fast bulk pass and a smaller,
    guaranteed-complete cleanup pass instead of one mechanism doing
    both jobs. No ring tracing, no cross-band value blending, no
    separate hilltop/pit/residual special-casing -- every band gets
    identical treatment regardless of its own shape or connectivity.

    edge_distance_m (pass 1 only -- pass 2 always ignores it) buffers
    every candidate's plateau an extra edge_distance_m past the band's
    true boundary, on top of just fitting within it. Leaves a strip
    along every band edge that pass 1's large hard stamps never get
    close to, for pass 2's finer, more precisely-targeted crumb fill to
    handle instead.

    smoothing_min_radius / smooth_ratio together set pass 2's fixed
    scatter radius (smoothing_min_radius * smooth_ratio -- default 4m *
    4.0 = 16m). Deliberately independent of pass 1's own min_radius:
    the crumb stage's scale is a property of how it does its OWN job
    (reaching across thin leftover slivers), not of how finely pass 1
    happened to be tiered. smooth_claim_fraction ("eat," default 0.25)
    is pass 2's own claim fraction -- deliberately much heavier overlap
    than pass 1's real-plateau-derived claim, since pass 2's whole job
    is blanket-covering whatever pass 1's large hard stamps couldn't
    reach, not precise packing.

    candidates_per_radius controls pass 1's random-candidate cap per
    tier -- left at None (default), it's auto-tuned once at the start
    of the run (see _auto_tune_candidates) by searching a handful of
    regularly-spaced sample bands (sweet_spot_sample_bands) across
    multiple random seeds (sweet_spot_seeds) for the point where pass
    2's own stamp count drops to sweet_spot_ratio of pass 1's -- past
    that point, more candidates buys diminishing pass-1 coverage while
    pass 2 does proportionally less mop-up work. The single highest
    candidate count found across every sampled band/seed combination is
    used for the WHOLE real run, not tuned per band -- bands are similar
    enough in character that per-band tuning would mostly repeat the
    same search for no benefit, and a single global value is what a
    real run actually needs to be safe everywhere. Set explicitly to
    skip auto-tuning and use one fixed value directly.

    on_candidates_tuned, if given, is called ONCE with the auto-tuned
    candidates_per_radius value, right after calibration finishes and
    before the real per-band generation starts -- lets a caller (see
    PGA2k_gen.py's step_generate_terrain) surface that number before
    committing to the full run, so it can be noted and passed back in
    explicitly next time to skip re-running the calibration search.
    Never called if candidates_per_radius was already given explicitly
    (nothing new to report in that case).

    random_seed seeds pass 1's own randomness -- each band gets
    random_seed + its own index, so the whole run is reproducible given
    the same inputs, while still varying naturally from band to band
    (not the identical random pattern repeated at every elevation).

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
    bottom N stamps gets you." FORCES n_workers=1 regardless of what's
    passed -- max_stamps needs a running total checked band-by-band to
    know when to stop, which is fundamentally sequential/streaming and
    doesn't parallelize; it's also explicitly a quick-preview tool
    where speed matters far less than it does for a real full run.

    n_workers parallelizes the main per-band loop across separate OS
    processes (concurrent.futures.ProcessPoolExecutor) -- bands never
    spatially overlap by construction (different bands' masks
    partition the heightmap into disjoint elevation ranges), so each
    band's own mask/denoise/pass-1/pass-2 work is fully independent and
    embarrassingly parallel, with no coordination needed between bands.
    None (default) auto-detects via os.cpu_count(); 1 forces the old
    single-process sequential behavior (e.g. for debugging, or when
    max_stamps is set, which forces this regardless -- see above). The
    heights array is sent to each worker process ONCE at pool startup
    (not re-pickled per band task), and results are reassembled in the
    SAME ascending-elevation order the sequential version always used,
    not whatever order workers happen to finish in -- output is
    identical to a sequential run at the same random_seed, this only
    changes how fast it gets there. progress_callback still fires the
    same way, just as each band's result comes back rather than as it
    completes in-process.

    Stamp order: bands ascending by elevation, and within each band,
    pass 1 (fill_brush) then pass 2 (smoothing_brush). Under this
    project's sequential pull-toward-value compositing, later stamps
    take precedence in any overlap -- this ordering means pass 2 always
    refines on top of pass 1 within its own band, and never gets
    overwritten by an unrelated later band's stamps (different bands'
    masks don't overlap by construction).

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

    crumb_radius = smoothing_min_radius * smooth_ratio

    if candidates_per_radius is None:
        candidates_per_radius = _auto_tune_candidates(
            heights, bounds, boundaries, fill_brush, max_radius, min_radius, radius_step_ratio,
            edge_distance_m, denoise_px,
            sweet_spot_ratio, sweet_spot_sample_bands, sweet_spot_seeds,
            initial_candidates=DEFAULT_CANDIDATES_PER_RADIUS, max_candidates=sweet_spot_max_candidates,
            time_budget_s=sweet_spot_time_budget_s,
        )
        if on_candidates_tuned is not None:
            on_candidates_tuned(candidates_per_radius)

    total_bands = len(boundaries)
    stamps: list[Stamp] = []
    last_progress_time = time.time()
    progress_interval_s = 10.0
    bands_reached = 0

    # max_stamps needs a running total checked band-by-band to know
    # when to stop -- fundamentally sequential/streaming, and it's
    # explicitly a quick-preview tool where speed matters far less
    # than for a real full run, so it isn't worth parallelizing around.
    effective_workers = 1 if max_stamps is not None else (n_workers or os.cpu_count() or 1)

    if effective_workers <= 1:
        for i, (lo, hi) in enumerate(boundaries):
            if max_stamps is not None and len(stamps) >= max_stamps:
                break
            bands_reached = i + 1

            mask = _band_mask(heights, lo, hi)
            if mask.any():
                mask = _denoise_mask(mask, denoise_px)
            if mask.any():
                rng = np.random.default_rng(random_seed + i)
                tier_budget = None if max_stamps is None else max_stamps - len(stamps)
                pass1_stamps, crumbs = _poisson_pack_band(
                    mask, heights, bounds, fill_brush, max_radius, min_radius, radius_step_ratio,
                    edge_distance_m, candidates_per_radius, rng, max_stamps=tier_budget,
                )
                stamps.extend(pass1_stamps)

                if crumbs.any():
                    crumb_budget = None if max_stamps is None else max_stamps - len(stamps)
                    if crumb_budget is None or crumb_budget > 0:
                        stamps.extend(_scatter_fill_remaining(
                            crumbs, heights, bounds, smoothing_brush, crumb_radius,
                            claim_radius_fraction=smooth_claim_fraction,
                            max_stamps=crumb_budget,
                        ))

            if progress_callback is not None and time.time() - last_progress_time >= progress_interval_s:
                progress_callback(len(stamps), (i + 1) / total_bands)
                last_progress_time = time.time()
    else:
        # max_stamps is None here (forced effective_workers=1 above if
        # it were set), so no per-band budget to thread through -- every
        # band runs to full completion, matching a real (non-preview) run.
        tasks = [
            (i, lo, hi, fill_brush, min_radius, max_radius, radius_step_ratio, edge_distance_m,
             denoise_px, random_seed, candidates_per_radius, smoothing_brush, crumb_radius,
             smooth_claim_fraction)
            for i, (lo, hi) in enumerate(boundaries)
        ]
        with ProcessPoolExecutor(
            max_workers=effective_workers, initializer=_init_band_worker, initargs=(heights, bounds),
        ) as executor:
            # executor.map preserves SUBMISSION order in its results,
            # regardless of which worker finishes which band first --
            # this is what keeps output identical to the sequential
            # version (ascending elevation order), not just "correct
            # but shuffled."
            for i, band_stamps in enumerate(executor.map(_process_one_band, tasks)):
                stamps.extend(band_stamps)
                bands_reached = i + 1
                if progress_callback is not None and time.time() - last_progress_time >= progress_interval_s:
                    progress_callback(len(stamps), (i + 1) / total_bands)
                    last_progress_time = time.time()

    if progress_callback is not None:
        # Real fraction reached, not a fake 1.0/0.0 -- a max_stamps cutoff
        # partway through band 3 of 10 should report 0.3, not claim either
        # full completion or that nothing happened.
        progress_callback(len(stamps), bands_reached / total_bands)

    return stamps
