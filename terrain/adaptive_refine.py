"""
terrain/adaptive_refine.py

Milestone 4's adaptive refinement pass: find where the terrain's
prediction is worst against real LIDAR, and add small detail stamps
exactly there -- rather than uniformly subdividing everywhere (see the
architecture doc's "Adaptive Refinement": "Only subdivide stamps whose
local error exceeds tolerance").

Targeting works directly off the same binned error grid preview_error.png
already visualizes, using a Euclidean distance transform to find the
best-centered stamp for each unclaimed flagged region:

  1. Build two boolean masks over the (unclaimed) error grid: cells
     overshooting (error > tolerance) and cells undershooting
     (error < -tolerance). Kept separate so a single stamp can never
     straddle both an overshoot and an undershoot region.
  2. For each mask, compute a distance transform: every flagged cell's
     value becomes its distance to the nearest *unflagged* cell (i.e.
     the region's boundary). The cell with the largest such value is
     the most-interior point of that region -- the center of the
     largest circle that fits entirely inside it without crossing the
     boundary -- and that distance *is* the natural radius for a stamp
     placed there. Take whichever mask (over or under) currently has
     the larger such value as this iteration's hotspot.
  3. Fit the region: try all four brushes x both tools (flatten pulls
     toward an absolute value, discarding whatever relief the coarse
     fit already got right; raise adds a delta on top of it, correcting
     a uniform bias while preserving existing shape -- see
     terrain/stamp.py) and keep whichever combination gives the lowest
     RMS over the region's actual LIDAR points.
  4. Claim only the cells actually covered by the placed (possibly
     radius-clamped) circle -- not the whole flagged region -- and
     repeat from the next-largest remaining distance-transform value.

Claiming only the placed circle, not the whole flagged region, matters
a lot for elongated features (a long, narrow river-valley error band,
for instance): the first iteration correctly centers a stamp on the
valley's centerline (the distance transform is unaffected by the
region's overall length, unlike a plain centroid-of-all-flagged-cells
approach, which gets pulled toward whichever end of an elongated
region happens to be flagged more thickly, effectively "targeting the
edges rather than the centers"), and leaves the rest of the valley's
length unclaimed for the next iteration to find -- producing a line of
separately-centered stamps tracing the valley, rather than one
mis-centered stamp (or, in an earlier version that thresholded and
grew without claiming per-circle, one giant stamp swallowing the
entire elongated region).

An even earlier version thresholded the whole grid at once and took
each resulting connected component as one hotspot, whatever its size --
with tolerance below the ambient curvature-driven error floor (see
below), a single contiguous blob could span a huge fraction of the
course. min_radius/max_radius clamps (tied to the coarse hex lattice's
own scale) still apply as a safety net.

Claiming the *entire* placement radius, though, means neighboring
stamps can touch at best, never overlap -- which produces a visibly
"cratered" look in practice (corrected divots surrounded by untouched
original terrain, since each stamp's influence tapers to ~0 at its own
edge and nothing beyond it gets pulled at all). claim_radius_fraction
(feature-flagged, see below) claims only an inner fraction of the
placed radius, leaving an outer band unclaimed so the next iteration's
stamp naturally lands closer and its influence overlaps this one's.

A prior version also capped radius by proximity to already-placed
hotspots in the same pass, specifically to *prevent* that overlap --
which is now the opposite of the goal, and turned out to have a real
bug besides: it could force radius down to half the distance between
two nearby centroids with no floor re-applied, producing degenerate
near-zero-radius stamps (confirmed: min_radius clamped to 25 m, but
stamps with radius 1 m or less were observed in practice whenever two
centroids landed close together). Removed outright rather than patched,
since deliberate overlap is now the point.

brush_radius_spread_ratio (feature-flagged) addresses a separate,
subtler issue: scoring all four brushes at the *same* radius
structurally favors whichever brush's falloff shape happens to match
that size, since RMS is measured over that one region only, blind to
how the edge blends into whatever's outside it. Type 8 (wide flat
plateau, sharp drop) tends to win this way even where a gentler brush
would blend better -- in practice, one real course's refinement runs
came out ~98% type 8. brush_radius_spread_ratio scales each brush's
candidate radius by spread_ratio ** BRUSH_RANK[brush] before scoring
(rank 0..3 for types 8/9/10/54, so spread_ratio=1 is a no-op and larger
values progressively favor giving smoother brushes more reach). This
was originally derived from the measured profiles (each brush's radius
scaled to reach the same half-weight distance as type 8's), but
checking Chad's TGC-Designer-Tools source directly showed that isn't
actually how his tool achieves smooth results -- he uses exactly one
brush type (10) everywhere, with a single fixed rule (stamp radius =
2x its placement grid spacing), not a per-brush-type multiplier table.
So rather than chase a "correct" derived table, this is exposed as a
plain tunable ratio instead: 1.0 disables it, 2.0 matches doubling per
rank step, whatever value works for real data is the real answer here,
not a formula.

Claiming uses the *pre-scaling* base radius, not whatever a winning
brush's scaled candidate_radius came out to -- otherwise
brush_radius_spread_ratio would also inflate how much area gets marked
done purely because a larger-radius brush happened to win a given
hotspot, making stamp density (and therefore overlap) depend on which
brush was picked rather than being controlled by claim_radius_fraction
alone. Keeping these independent means each knob controls exactly one
thing: claim_radius_fraction controls overlap/density, spread_ratio
controls which brush gets a fair comparison at a larger size, and
changing one doesn't secretly change the other's effect too.

Both new knobs default to their old (pre-existing) behavior when
disabled: claim_radius_fraction=1.0 claims the whole placement radius
exactly as before, brush_radius_spread_ratio=1.0 scores every brush at
the same radius exactly as before. PGA2k_gen.py persists whichever
settings were used to project.json, so they carry forward between
refine-terrain runs without needing to be retyped each time.

No masks exist yet (Milestone 5), so this uses a single global error
tolerance rather than the mask-driven per-region tolerances the doc
describes as the eventual design -- this is the "for now" version,
same spirit as height_fit.py's naive averaging.

Designed to run repeatedly: each call re-scores the *current* terrain
(coarse stamps plus any previously-added detail stamps) against the
same grid, so "largest error -> split -> repeat" falls out of just
calling refine_stamps() again on its own prior output. Tightening
tolerance between passes should be done carefully (check
preview_error.png between passes) -- refining below the ambient
curvature-driven error floor floods the grid with hotspots that are
mostly noise, not real features, and can rescore/overwrite points that
were already correctly fit by an earlier, more targeted pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from ingest.heightmap import downsample_heightmap, query_heightmap_cells
from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.hexgrid import HEX_STAMP_RADIUS_M
from terrain.stamp import TOOL_FLATTEN, TOOL_RAISE, Stamp
from terrain.terrain_kernel import TerrainKernel
from terrain.terrain_model import TerrainModel

DEFAULT_RESOLUTION = 200
DEFAULT_MIN_POINTS = 3
DEFAULT_MIN_HOTSPOT_RADIUS_CELLS = 1.0  # below this (pre-clamp), treat as noise, not a feature
DEFAULT_CLAIM_RADIUS_FRACTION = 1.0  # 1.0 = claim the whole radius (old behavior, no overlap)
DEFAULT_BRUSH_RADIUS_SPREAD_RATIO = 1.0  # 1.0 = every brush scored at the same radius (old behavior)
DEFAULT_RADIUS_DECAY_PER_PASS = 1.0  # 1.0 = every refine-terrain pass uses the same clamps (old behavior)

# Candidate brushes/tools tried per hotspot; whichever combination
# gives the lowest RMS over the region's actual LIDAR points wins.
# BRUSH_RANK orders them for brush_radius_spread_ratio (see
# _brush_radius_multiplier): candidate_radius = base_radius *
# spread_ratio ** BRUSH_RANK[brush], so spread_ratio=1 means every
# brush gets the same radius (multiplier 1 regardless of rank) and
# larger values progressively favor giving smoother-falloff brushes
# (higher rank) more reach. This is a deliberately simple, tunable
# parameterization rather than a fixed per-brush table -- see module
# docstring for why: it turned out not to match how Chad's tool
# actually achieves smooth results (a single always-radius-2x-spacing
# rule with one brush type, not a per-brush-type multiplier table), so
# a fixed "correct" table isn't something to chase -- better to expose
# the ratio directly and let it be tuned against real results.
CANDIDATE_BRUSHES = (8, 9, 10, 54)
CANDIDATE_TOOLS = (TOOL_FLATTEN, TOOL_RAISE)
BRUSH_RANK = {8: 0, 9: 1, 10: 2, 54: 3}

# Safety-net clamps on hotspot stamp radius, tied to the main lattice's
# own scale rather than an arbitrary number -- max is half the coarse
# stamp radius, min is half of that.
DEFAULT_MAX_HOTSPOT_RADIUS_M = HEX_STAMP_RADIUS_M / 2.0
DEFAULT_MIN_HOTSPOT_RADIUS_M = DEFAULT_MAX_HOTSPOT_RADIUS_M / 2.0


def _brush_radius_multiplier(brush: int, spread_ratio: float) -> float:
    return spread_ratio ** BRUSH_RANK[brush]


@dataclass(slots=True)
class ErrorHotspot:
    x: float
    z: float
    radius: float
    peak_error: float
    n_cells: int
    brush: int
    tool: int
    value: float
    fit_rms: float

    def to_stamp(self) -> Stamp:
        return Stamp(x=self.x, z=self.z, radius=self.radius, value=self.value,
                     brush=self.brush, tool=self.tool)


def _score_candidate(
    hx: float, hz: float, radius: float, brush: int, tool: int,
    px: np.ndarray, pz: np.ndarray, actual: np.ndarray,
    current_at_points: np.ndarray, current_at_center: float, target_mean: float,
) -> Optional[tuple[float, float]]:
    """
    Analytically fit `value` for one (brush, tool) candidate and score
    it against the region's actual LIDAR points, without rebuilding a
    TerrainModel (and its KD-tree) per candidate -- current_at_points /
    current_at_center are the *existing* terrain's prediction (computed
    once per hotspot, reused across all 8 candidates), so overlaying
    one new stamp's effect is just the flatten/raise formula applied
    directly (see terrain/terrain_model.py).

    Returns (value, rms), or None if this brush's center weight is
    degenerate (shouldn't happen for any registered profile, but
    guarded rather than dividing by zero).
    """
    kernel = TerrainKernel(BRUSH_PROFILES[brush])
    dist = np.sqrt((px - hx) ** 2 + (pz - hz) ** 2)
    r_norm = np.clip(dist / radius, 0.0, 1.0)
    weight = kernel.sample_many(r_norm)
    weight_center = kernel.sample(0.0)
    if weight_center <= 0.0:
        return None

    if tool == TOOL_RAISE:
        value = (target_mean - current_at_center) / weight_center
        predicted = current_at_points + value * weight
    else:
        value = current_at_center + (target_mean - current_at_center) / weight_center
        predicted = current_at_points + (value - current_at_points) * weight

    rms = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    return value, rms


def find_error_hotspots(
    stamps: Sequence[Stamp],
    heights: np.ndarray,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_hotspot_radius_cells: float = DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    claim_radius_fraction: float = DEFAULT_CLAIM_RADIUS_FRACTION,
    brush_radius_spread_ratio: float = DEFAULT_BRUSH_RADIUS_SPREAD_RATIO,
    min_valid_cells: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> list[ErrorHotspot]:
    """
    Find and fit error hotspots via distance-transform region centering
    (see module docstring). Each returned ErrorHotspot already carries
    its best-fit brush/tool/value -- no separate fit_stamp_heights()
    call needed afterward, since scoring candidates requires the
    region's actual ground heightmap cells anyway and that's already
    done here.

    heights/bounds are the rasterized ground heightmap (see
    ingest/heightmap.py's rasterize_ground_heightmap/load_heightmap) --
    bare-earth filtering already happened once, at rasterization time,
    not per hotspot here. Sampling this pre-rasterized grid instead of
    querying a PointCloud's KD-tree directly is the same "no tree
    traversal needed on a regular grid" idea as the render()
    optimization, applied to the ground-truth side of the fit.

    mask, if given, is a boolean grid the same (resolution, resolution)
    shape as the error grid -- True where refinement is allowed to
    place stamps, False where it isn't (e.g. outside fairway/green; see
    ingest/osm.py's build_height_mask/rasterize_mask). Cells outside the
    mask are treated exactly like invalid/no-data cells: excluded from
    consideration entirely, never becoming a peak or getting claimed,
    rather than being scored and then discarded.

    Returns hotspots in the order found (largest inscribed radius
    first, which tracks -- but isn't identical to -- worst peak error).
    """
    actual = downsample_heightmap(heights, bounds, resolution)
    model = TerrainModel(stamps)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual  # NaN where actual has no data

    valid = np.isfinite(error)
    if mask is not None:
        valid = valid & mask
    claimed = ~valid

    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution
    sampling = (cell_size_z, cell_size_x)  # (row, col) spacing for distance_transform_edt

    max_brush_scale = max(
        _brush_radius_multiplier(b, brush_radius_spread_ratio) for b in CANDIDATE_BRUSHES
    )

    hotspots: list[ErrorHotspot] = []

    # dist_over/dist_under are cached across iterations rather than
    # recomputed from scratch every time: a claim only ever touches
    # cells matching the peak's own sign (see the claim step below, and
    # its comment on why that restriction matters for correctness too,
    # not just this), so the *other* sign's mask is provably unchanged
    # by that claim and its distance transform is still valid --
    # recomputing it anyway was pure wasted work every single
    # iteration. Each is recomputed only the first time, and again only
    # after an iteration actually claims cells of its matching sign.
    dist_over: Optional[np.ndarray] = None
    dist_under: Optional[np.ndarray] = None

    for _ in range(resolution * resolution):
        if max_new_stamps is not None and len(hotspots) >= max_new_stamps:
            break

        if dist_over is None:
            over_mask = (error > tolerance) & ~claimed
            dist_over = ndimage.distance_transform_edt(over_mask, sampling=sampling)
        if dist_under is None:
            under_mask = (error < -tolerance) & ~claimed
            dist_under = ndimage.distance_transform_edt(under_mask, sampling=sampling)

        if dist_over.max(initial=0.0) >= dist_under.max(initial=0.0):
            dist_map = dist_over
            peak_is_over = True
        else:
            dist_map = dist_under
            peak_is_over = False

        peak_dist = dist_map.max(initial=0.0)
        if peak_dist <= 0.0:
            break  # nothing left flagged in either direction

        peak_row, peak_col = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        peak_sign = 1.0 if peak_is_over else -1.0

        min_dist_needed = min_hotspot_radius_cells * min(cell_size_x, cell_size_z)
        if peak_dist < min_dist_needed:
            # Too small to be a real feature rather than noise -- claim
            # just this one cell so it can't be picked again, but don't
            # place a stamp for it.
            claimed[peak_row, peak_col] = True
            if peak_is_over:
                dist_over = None
            else:
                dist_under = None
            continue

        centroid_x = float(x_centers[peak_col])
        centroid_z = float(z_centers[peak_row])
        base_radius = max(peak_dist, min_radius)
        base_radius = min(base_radius, max_radius)

        peak_error = float(abs(error[peak_row, peak_col]))

        # Query at the largest radius any candidate could use (base
        # radius scaled up for the most-generous brush, if brush radius
        # scaling is on) so every candidate scores against its own
        # correct point set with a single query -- points beyond a
        # given candidate's own radius simply get r_norm=1 (weight 0)
        # for that candidate, which is already correct/harmless.
        query_radius = min(max_radius, base_radius * max_brush_scale)
        px, pz, actual_pts = query_heightmap_cells(heights, bounds, centroid_x, centroid_z, query_radius)
        if px.size < min_valid_cells:
            claimed[peak_row, peak_col] = True
            if peak_is_over:
                dist_over = None
            else:
                dist_under = None
            continue

        current_at_points = model.evaluate_many(np.column_stack((px, pz)))
        current_at_center = model.evaluate(centroid_x, centroid_z)
        target_mean = float(np.mean(actual_pts))

        best: Optional[tuple[int, int, float, float, float]] = None  # + candidate_radius
        for brush in CANDIDATE_BRUSHES:
            candidate_radius = base_radius * _brush_radius_multiplier(brush, brush_radius_spread_ratio)
            candidate_radius = max(min_radius, min(candidate_radius, max_radius))

            for tool in CANDIDATE_TOOLS:
                result = _score_candidate(
                    centroid_x, centroid_z, candidate_radius, brush, tool,
                    px, pz, actual_pts, current_at_points, current_at_center, target_mean,
                )
                if result is None:
                    continue
                value, rms = result
                if best is None or rms < best[3]:
                    best = (brush, tool, value, rms, candidate_radius)

        if best is None:
            claimed[peak_row, peak_col] = True
            if peak_is_over:
                dist_over = None
            else:
                dist_under = None
            continue

        brush, tool, value, rms, final_radius = best

        # Claim only an inner fraction of the *base* (pre-brush-scaling)
        # radius -- not final_radius, which may be inflated by
        # brush_radius_spread_ratio. Using final_radius here would let
        # spread_ratio also control claimed area/density purely as a
        # side effect of which brush happened to win, entangling two
        # knobs that should stay independent (see module docstring).
        #
        # Restricted to cells matching the peak's own sign: a stamp
        # placed to correct an overshoot doesn't do anything to fix a
        # nearby undershoot, so marking that undershoot "claimed" just
        # because it happened to be geometrically close (which base_radius
        # inflated up to min_radius can cause) would wrongly prevent a
        # future pass from ever placing a stamp to correct it. This also
        # what makes the distance-transform caching above valid: without
        # it, a claim could silently modify the *other* sign's mask too.
        claim_radius = base_radius * claim_radius_fraction
        rows_idx, cols_idx = np.nonzero(valid & ~claimed & (np.sign(error) == peak_sign))
        if rows_idx.size:
            cell_dist = np.hypot(
                (z_centers[rows_idx] - centroid_z), (x_centers[cols_idx] - centroid_x)
            )
            claimed[rows_idx[cell_dist <= claim_radius], cols_idx[cell_dist <= claim_radius]] = True
        claimed[peak_row, peak_col] = True  # guaranteed claimed even if claim_radius rounds to ~0

        if peak_is_over:
            dist_over = None
        else:
            dist_under = None

        n_cells = int(np.sum(valid & (np.hypot(
            (z_centers[:, None] - centroid_z), (x_centers[None, :] - centroid_x)
        ) <= final_radius)))
        hotspots.append(ErrorHotspot(
            x=centroid_x, z=centroid_z, radius=final_radius, peak_error=peak_error,
            n_cells=n_cells, brush=brush, tool=tool, value=value, fit_rms=rms,
        ))

    return hotspots


def refine_stamps(
    stamps: Sequence[Stamp],
    heights: np.ndarray,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_hotspot_radius_cells: float = DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    claim_radius_fraction: float = DEFAULT_CLAIM_RADIUS_FRACTION,
    brush_radius_spread_ratio: float = DEFAULT_BRUSH_RADIUS_SPREAD_RATIO,
    min_valid_cells: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
) -> tuple[list[Stamp], list[ErrorHotspot]]:
    """
    One adaptive refinement pass: find and fit error hotspots (see
    find_error_hotspots), append their stamps to the existing
    (unchanged) list, and return the combined result.

    Existing stamps are never removed or re-fit -- new stamps are
    appended after them, so under pull-toward-value/raise semantics
    they refine on top of the existing baseline (see
    terrain_model.py). Safe to call repeatedly on its own output.
    """
    hotspots = find_error_hotspots(
        stamps, heights, bounds, tolerance,
        resolution=resolution, min_hotspot_radius_cells=min_hotspot_radius_cells,
        min_radius=min_radius, max_radius=max_radius,
        claim_radius_fraction=claim_radius_fraction,
        brush_radius_spread_ratio=brush_radius_spread_ratio,
        mask=mask,
        min_valid_cells=min_valid_cells,
        max_new_stamps=max_new_stamps,
    )

    new_stamps = [h.to_stamp() for h in hotspots]
    return list(stamps) + new_stamps, hotspots
