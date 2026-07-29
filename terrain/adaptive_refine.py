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
course, producing a stamp whose LIDAR-averaged target was a poor fit
almost everywhere within it. min_radius/max_radius clamps (tied to the
coarse hex lattice's own scale) still apply as a safety net, plus a
proximity cap against other hotspots found in the same pass (the
min-radius clamp alone could otherwise push two nearby small regions'
stamps into physical overlap even though their flagged cells never
touched).

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

from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.hexgrid import HEX_STAMP_RADIUS_M
from terrain.stamp import TOOL_FLATTEN, TOOL_RAISE, Stamp
from terrain.terrain_kernel import TerrainKernel
from terrain.terrain_model import TerrainModel

DEFAULT_RESOLUTION = 200
DEFAULT_MIN_POINTS = 3
DEFAULT_MIN_HOTSPOT_RADIUS_CELLS = 1.0  # below this (pre-clamp), treat as noise, not a feature

# Candidate brushes/tools tried per hotspot; whichever combination
# gives the lowest RMS over the region's actual LIDAR points wins.
CANDIDATE_BRUSHES = (8, 9, 10, 54)
CANDIDATE_TOOLS = (TOOL_FLATTEN, TOOL_RAISE)

# Safety-net clamps on hotspot stamp radius, tied to the main lattice's
# own scale rather than an arbitrary number -- max is half the coarse
# stamp radius, min is half of that.
DEFAULT_MAX_HOTSPOT_RADIUS_M = HEX_STAMP_RADIUS_M / 2.0
DEFAULT_MIN_HOTSPOT_RADIUS_M = DEFAULT_MAX_HOTSPOT_RADIUS_M / 2.0


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


def _bin_actual_elevation(
    cloud: PointCloud, bounds: BoundingBox, resolution: int, bare_earth_only: bool = True,
) -> np.ndarray:
    """
    Mean LIDAR elevation per grid cell over `bounds`, resolution x
    resolution, NaN where a cell has no points. Same binning convention
    as visualize.py's _bin_point_cloud (rows = z, columns = x) so this
    lines up cell-for-cell with TerrainModel.render()'s own grid.

    bare_earth_only matters a lot here: without it, building roofs and
    vegetation returns get compared directly against predicted ground
    height and show up as "error" the refinement pass then tries to
    correct.
    """
    if bare_earth_only:
        mask = cloud.bare_earth_mask()
        x, z, elevation = cloud.x[mask], cloud.z[mask], cloud.elevation[mask]
    else:
        x, z, elevation = cloud.x, cloud.z, cloud.elevation

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)

    sums, _, _ = np.histogram2d(z, x, bins=[z_edges, x_edges], weights=elevation)
    counts, _, _ = np.histogram2d(z, x, bins=[z_edges, x_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means


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
    cloud: PointCloud,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_hotspot_radius_cells: float = DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
) -> list[ErrorHotspot]:
    """
    Find and fit error hotspots via distance-transform region centering
    (see module docstring). Each returned ErrorHotspot already carries
    its best-fit brush/tool/value -- no separate fit_stamp_heights()
    call needed afterward, since scoring candidates requires the
    region's actual LIDAR points anyway and that's already done here.

    Returns hotspots in the order found (largest inscribed radius
    first, which tracks -- but isn't identical to -- worst peak error).
    """
    actual = _bin_actual_elevation(cloud, bounds, resolution, bare_earth_only=bare_earth_only)
    model = TerrainModel(stamps)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual  # NaN where actual has no data

    valid = np.isfinite(error)
    claimed = ~valid

    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution
    sampling = (cell_size_z, cell_size_x)  # (row, col) spacing for distance_transform_edt

    hotspots: list[ErrorHotspot] = []
    placed_centroids: list[tuple[float, float]] = []

    for _ in range(resolution * resolution):
        if max_new_stamps is not None and len(hotspots) >= max_new_stamps:
            break

        over_mask = (error > tolerance) & ~claimed
        under_mask = (error < -tolerance) & ~claimed

        dist_over = ndimage.distance_transform_edt(over_mask, sampling=sampling)
        dist_under = ndimage.distance_transform_edt(under_mask, sampling=sampling)

        if dist_over.max(initial=0.0) >= dist_under.max(initial=0.0):
            dist_map, sign_mask = dist_over, over_mask
        else:
            dist_map, sign_mask = dist_under, under_mask

        peak_dist = dist_map.max(initial=0.0)
        if peak_dist <= 0.0:
            break  # nothing left flagged in either direction

        peak_row, peak_col = np.unravel_index(np.argmax(dist_map), dist_map.shape)

        min_dist_needed = min_hotspot_radius_cells * min(cell_size_x, cell_size_z)
        if peak_dist < min_dist_needed:
            # Too small to be a real feature rather than noise -- claim
            # just this one cell so it can't be picked again, but don't
            # place a stamp for it.
            claimed[peak_row, peak_col] = True
            continue

        centroid_x = float(x_centers[peak_col])
        centroid_z = float(z_centers[peak_row])
        radius = max(peak_dist, min_radius)
        radius = min(radius, max_radius)

        # Proximity cap: the min-radius clamp above could otherwise push
        # this stamp into physical overlap with an already-placed one
        # from this same pass, even though their flagged cells never
        # touched. Half the distance to the nearest prior centroid keeps
        # them from overlapping regardless of what the clamp did.
        for px_c, pz_c in placed_centroids:
            d = np.hypot(centroid_x - px_c, centroid_z - pz_c)
            radius = min(radius, 0.5 * d)

        peak_error = float(abs(error[peak_row, peak_col]))

        # Claim every originally-valid cell within the placed (possibly
        # clamped) radius -- not the whole flagged region -- so an
        # elongated feature longer than this stamp's reach leaves the
        # rest of itself available for the next iteration to find (see
        # module docstring's river-valley example).
        rows_idx, cols_idx = np.nonzero(valid & ~claimed)
        if rows_idx.size:
            cell_dist = np.hypot(
                (z_centers[rows_idx] - centroid_z), (x_centers[cols_idx] - centroid_x)
            )
            claimed[rows_idx[cell_dist <= radius], cols_idx[cell_dist <= radius]] = True
        claimed[peak_row, peak_col] = True  # guaranteed claimed even if radius rounds to ~0

        idx = cloud.query_radius(centroid_x, centroid_z, radius)
        if bare_earth_only and idx.size > 0:
            idx = idx[cloud.bare_earth_mask()[idx]]
        if idx.size < min_points:
            continue

        px, pz = cloud.x[idx], cloud.z[idx]
        actual_pts = cloud.elevation[idx]
        current_at_points = model.evaluate_many(np.column_stack((px, pz)))
        current_at_center = model.evaluate(centroid_x, centroid_z)
        target_mean = float(np.mean(actual_pts))

        best: Optional[tuple[int, int, float, float]] = None
        for brush in CANDIDATE_BRUSHES:
            for tool in CANDIDATE_TOOLS:
                result = _score_candidate(
                    centroid_x, centroid_z, radius, brush, tool,
                    px, pz, actual_pts, current_at_points, current_at_center, target_mean,
                )
                if result is None:
                    continue
                value, rms = result
                if best is None or rms < best[3]:
                    best = (brush, tool, value, rms)

        if best is None:
            continue

        brush, tool, value, rms = best
        n_cells = int(np.sum((valid) & (np.hypot(
            (z_centers[:, None] - centroid_z), (x_centers[None, :] - centroid_x)
        ) <= radius)))
        hotspots.append(ErrorHotspot(
            x=centroid_x, z=centroid_z, radius=radius, peak_error=peak_error,
            n_cells=n_cells, brush=brush, tool=tool, value=value, fit_rms=rms,
        ))
        placed_centroids.append((centroid_x, centroid_z))

    return hotspots


def refine_stamps(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_hotspot_radius_cells: float = DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
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
        stamps, cloud, bounds, tolerance,
        resolution=resolution, min_hotspot_radius_cells=min_hotspot_radius_cells,
        min_radius=min_radius, max_radius=max_radius,
        bare_earth_only=bare_earth_only, min_points=min_points,
        max_new_stamps=max_new_stamps,
    )

    new_stamps = [h.to_stamp() for h in hotspots]
    return list(stamps) + new_stamps, hotspots
