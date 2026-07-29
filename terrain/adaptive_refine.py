"""
terrain/adaptive_refine.py

Milestone 4's adaptive refinement pass: find where the terrain's
prediction is worst against real LIDAR, and add small detail stamps
exactly there -- rather than uniformly subdividing everywhere (see the
architecture doc's "Adaptive Refinement": "Only subdivide stamps whose
local error exceeds tolerance").

Targeting works directly off the same binned error grid preview_error.png
already visualizes, using peak-seeking region growth rather than naive
thresholding:

  1. Find the single worst-error unclaimed cell (the peak).
  2. Flood-fill outward to same-signed, unclaimed neighbors, stopping
     once error drops below a fraction of the peak's own magnitude
     (zero_crossing_fraction) rather than at some fraction of
     tolerance -- sized relative to the peak, not an absolute floor,
     so a broad low-level same-signed background can't get swept in
     (see _grow_region). Requiring same sign is what makes this a real
     boundary rather than an arbitrary cutoff: a region can't straddle
     both an overshoot and an undershoot, since by definition error
     changes sign somewhere between them.
  3. Fit the region: try all four brushes x both tools (flatten pulls
     toward an absolute value, discarding whatever relief the coarse
     fit already got right; raise adds a delta on top of it, correcting
     a uniform bias while preserving existing shape -- see
     terrain/stamp.py) and keep whichever combination gives the lowest
     RMS over the region's actual LIDAR points.
  4. Claim the region's cells (excluded from future peaks in this
     pass) and repeat from the next-worst unclaimed peak.

An earlier version thresholded the whole grid at once and took each
resulting connected component as one hotspot, whatever its size --
with tolerance below the ambient curvature-driven error floor (see
below), a single contiguous blob could span a huge fraction of the
course, producing a stamp whose LIDAR-averaged target was a poor fit
almost everywhere within it, actively making things worse where it got
applied (order-dependent pull semantics mean the last-applied stamp at
a point dominates). Peak-seeking with a same-signed, zero-crossing
boundary bounds region size at the source rather than clamping after
the fact -- though min_radius/max_radius clamps (tied to the coarse
hex lattice's own scale) still apply as a second line of defense, plus
a proximity cap against other hotspots found in the same pass (the
min-radius clamp alone could otherwise push two nearby small regions'
stamps into physical overlap even though their error cells never
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
DEFAULT_MIN_REGION_CELLS = 2
DEFAULT_ZERO_CROSSING_FRACTION = 0.2  # stop growing once error drops below this fraction of the peak

# Candidate brushes/tools tried per hotspot; whichever combination
# gives the lowest RMS over the region's actual LIDAR points wins.
CANDIDATE_BRUSHES = (8, 9, 10, 54)
CANDIDATE_TOOLS = (TOOL_FLATTEN, TOOL_RAISE)

# Safety-net clamps on hotspot stamp radius, tied to the main lattice's
# own scale rather than an arbitrary number -- max is half the coarse
# stamp radius, min is half of that.
DEFAULT_MAX_HOTSPOT_RADIUS_M = HEX_STAMP_RADIUS_M / 2.0
DEFAULT_MIN_HOTSPOT_RADIUS_M = DEFAULT_MAX_HOTSPOT_RADIUS_M / 2.0

_NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 4-connectivity


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


def _grow_region(
    error: np.ndarray,
    claimed: np.ndarray,
    peak: tuple[int, int],
    fraction: float,
) -> list[tuple[int, int]]:
    """
    Flood-fill from `peak` to same-signed, unclaimed neighbors whose
    |error| is still at least `fraction` of the peak's own |error|,
    4-connected.

    The threshold is relative to the peak, not an absolute floor --
    using a fixed absolute cutoff (e.g. "keep growing until |error|
    drops below 0.1 m") doesn't actually bound growth to the local
    feature if the surrounding *background* error also happens to
    share the peak's sign over a wide area (a broad, low-level
    curvature-smoothing bias easily does this): growth would keep
    picking up same-signed background cells indefinitely, producing a
    region spanning most of the grid instead of just the anomaly. Sizing
    the cutoff to the peak's own magnitude means a genuine sharp local
    feature (which decays substantially away from its own peak) still
    gets isolated correctly, while a flat, low-magnitude background
    fails the relative test almost immediately.
    """
    peak_error = error[peak]
    peak_sign = np.sign(peak_error)
    threshold = fraction * abs(peak_error)
    rows, cols = error.shape

    region = [peak]
    visited = {peak}
    stack = [peak]

    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBOR_OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in visited or claimed[nr, nc]:
                continue
            e = error[nr, nc]
            if not np.isfinite(e) or np.sign(e) != peak_sign or abs(e) < threshold:
                continue
            visited.add((nr, nc))
            region.append((nr, nc))
            stack.append((nr, nc))

    return region


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
    min_region_cells: int = DEFAULT_MIN_REGION_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    zero_crossing_fraction: float = DEFAULT_ZERO_CROSSING_FRACTION,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
) -> list[ErrorHotspot]:
    """
    Find and fit error hotspots via peak-seeking region growth (see
    module docstring). Each returned ErrorHotspot already carries its
    best-fit brush/tool/value -- no separate fit_stamp_heights() call
    needed afterward, since scoring candidates requires the region's
    actual LIDAR points anyway and that's already done here.

    Returns hotspots sorted by peak |error|, worst first (which is
    also the order they were found in, since each pass claims the
    current worst peak before moving to the next).
    """
    actual = _bin_actual_elevation(cloud, bounds, resolution, bare_earth_only=bare_earth_only)
    model = TerrainModel(stamps)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual  # NaN where actual has no data

    claimed = ~np.isfinite(error)

    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution

    hotspots: list[ErrorHotspot] = []
    placed_centroids: list[tuple[float, float]] = []

    # Bounded iteration count as a hard safety valve -- each loop either
    # produces a hotspot or claims at least the peak cell, so this can't
    # spin forever, but an explicit cap is cheap insurance.
    for _ in range(resolution * resolution):
        if max_new_stamps is not None and len(hotspots) >= max_new_stamps:
            break

        masked = np.where(claimed, -np.inf, np.abs(error))
        peak = np.unravel_index(np.argmax(masked), masked.shape)
        peak_val = masked[peak]
        if not np.isfinite(peak_val) or peak_val <= tolerance:
            break

        region = _grow_region(error, claimed, peak, zero_crossing_fraction)
        for cell in region:
            claimed[cell] = True

        if len(region) < min_region_cells:
            continue

        rows = np.array([c[0] for c in region])
        cols = np.array([c[1] for c in region])
        cell_x = x_centers[cols]
        cell_z = z_centers[rows]
        centroid_x = float(np.mean(cell_x))
        centroid_z = float(np.mean(cell_z))

        dist_to_centroid = np.sqrt((cell_x - centroid_x) ** 2 + (cell_z - centroid_z) ** 2)
        radius = float(np.max(dist_to_centroid)) + 0.5 * max(cell_size_x, cell_size_z)
        radius = max(radius, min_radius)
        radius = min(radius, max_radius)

        # Proximity cap: the min-radius clamp above could otherwise push
        # this stamp into physical overlap with an already-placed one
        # from this same pass, even though their error regions never
        # touched. Half the distance to the nearest prior centroid keeps
        # them from overlapping regardless of what the clamp did.
        for px_c, pz_c in placed_centroids:
            d = np.hypot(centroid_x - px_c, centroid_z - pz_c)
            radius = min(radius, 0.5 * d)

        peak_error = float(np.max(np.abs(error[rows, cols])))

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
        hotspots.append(ErrorHotspot(
            x=centroid_x, z=centroid_z, radius=radius, peak_error=peak_error,
            n_cells=len(region), brush=brush, tool=tool, value=value, fit_rms=rms,
        ))
        placed_centroids.append((centroid_x, centroid_z))

    return hotspots


def refine_stamps(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_region_cells: int = DEFAULT_MIN_REGION_CELLS,
    min_radius: float = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: float = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    zero_crossing_fraction: float = DEFAULT_ZERO_CROSSING_FRACTION,
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
        resolution=resolution, min_region_cells=min_region_cells,
        min_radius=min_radius, max_radius=max_radius,
        zero_crossing_fraction=zero_crossing_fraction,
        bare_earth_only=bare_earth_only, min_points=min_points,
        max_new_stamps=max_new_stamps,
    )

    new_stamps = [h.to_stamp() for h in hotspots]
    return list(stamps) + new_stamps, hotspots
