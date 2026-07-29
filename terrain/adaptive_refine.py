"""
terrain/adaptive_refine.py

Milestone 4's adaptive refinement pass: find where the terrain's
prediction is worst against real LIDAR, and add small detail stamps
exactly there -- rather than uniformly subdividing everywhere (see the
architecture doc's "Adaptive Refinement": "Only subdivide stamps whose
local error exceeds tolerance").

Targeting works directly off the same binned error grid preview_error.png
already visualizes -- not per-stamp averages. An earlier version scored
each existing stamp's own full-radius disk as one RMS number, but that
dilutes a sharp localized error against everything else already fitted
well within the same disk: a stamp's 100 m-radius neighborhood can
easily average out to "fine" even with a severe, narrow miss somewhere
inside it, so a real course found "0 stamps over tolerance" against a
tolerance well below what the error heatmap visibly showed. Scoring a
fine grid directly (find_error_hotspots) and centering new stamps on
the actual flagged cells (not on whichever pre-existing lattice point
happened to contain them) fixes both the accuracy problem and, as a
side effect, is far faster: no more per-stamp query-and-evaluate loop
over every point in every stamp's disk.

No masks exist yet (Milestone 5), so this uses a single global error
tolerance rather than the mask-driven per-region tolerances the doc
describes as the eventual design -- this is the "for now" version,
same spirit as height_fit.py's naive averaging.

Designed to run repeatedly: each call re-scores the *current* terrain
(coarse stamps plus any previously-added detail stamps) against the
same grid, so "largest error -> split -> repeat" falls out of just
calling refine_stamps() again on its own prior output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import HEX_STAMP_RADIUS_M, PLACEHOLDER_VALUE
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel

DEFAULT_RESOLUTION = 200
DEFAULT_MIN_POINTS = 3
DEFAULT_MIN_REGION_CELLS = 2

# Hotspot patch brush is deliberately independent from hexgrid.DEFAULT_BRUSH
# (the main coarse lattice's brush) -- they can have different needs. Type
# 9 has a genuine flat plateau (unlike type 10's cosine falloff, which
# starts weakening immediately from center), so an isolated patch stamp
# gets uniform correction across a real area instead of a sharp point-fix
# surrounded by a barely-touched halo ("punching holes").
DEFAULT_HOTSPOT_BRUSH = 9

# Safety-net clamps on hotspot stamp radius, tied to the main lattice's
# own scale rather than an arbitrary number -- max is half the coarse
# stamp radius, min is half of that. Without a max clamp, a large
# contiguous over-threshold region (e.g. tolerance set below the ambient
# curvature-driven error floor) can otherwise produce a stamp sized to
# the whole blob -- hundreds of meters, averaging LIDAR across a huge,
# varied area into one poor-fit value that makes things worse where it
# gets applied (see module docstring).
DEFAULT_MAX_HOTSPOT_RADIUS_M = HEX_STAMP_RADIUS_M / 2.0
DEFAULT_MIN_HOTSPOT_RADIUS_M = DEFAULT_MAX_HOTSPOT_RADIUS_M / 2.0


@dataclass(slots=True)
class ErrorHotspot:
    x: float
    z: float
    radius: float
    peak_error: float
    n_cells: int


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
    correct -- fit_stamp_heights already filters to bare-earth for
    exactly this reason; this needs to match it.
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


def find_error_hotspots(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_region_cells: int = DEFAULT_MIN_REGION_CELLS,
    min_radius: Optional[float] = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: Optional[float] = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    bare_earth_only: bool = True,
) -> list[ErrorHotspot]:
    """
    Find contiguous regions of the error grid (predicted - actual)
    where |error| > tolerance, exactly like preview_error.png's
    heatmap. Each connected region becomes one hotspot: centered on
    the region's centroid, sized to its own extent (max distance from
    centroid to any flagged cell in it) -- "fit a circle to the area
    bounded by low error", since connected-component labeling only
    groups cells that exceed tolerance, so a region's boundary is
    exactly where the surrounding cells drop back under it.

    Regions smaller than `min_region_cells` are dropped -- a single
    flagged cell is as likely to be point-level noise as a real
    feature. min_radius/max_radius clamp the resulting stamp size
    (default: half the coarse hex radius for max, half of that for
    min -- see DEFAULT_MAX_HOTSPOT_RADIUS_M/DEFAULT_MIN_HOTSPOT_RADIUS_M),
    so a one-cell region doesn't produce an absurdly tiny stamp, and
    -- more importantly -- a large contiguous blob (e.g. tolerance set
    below the ambient curvature-driven error floor, see module
    docstring) doesn't produce an absurdly *large* one. This clamp is
    a safety net, not a full fix: it caps the damage a bad region can
    do, but doesn't stop unrelated distant cells from merging into one
    misshapen blob in the first place. Properly bounding region growth
    at the source -- e.g. flood-filling out from each error peak only
    while error stays meaningfully non-zero, rather than thresholding
    the whole grid and connected-component labeling everything at
    once -- is the real fix and hasn't been built yet.

    Returns hotspots sorted by peak |error|, worst first.
    """
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution
    if min_radius is None:
        min_radius = 1.5 * max(cell_size_x, cell_size_z)

    actual = _bin_actual_elevation(cloud, bounds, resolution, bare_earth_only=bare_earth_only)
    model = TerrainModel(stamps)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual  # NaN where actual has no data -- never flagged

    mask = np.abs(error) > tolerance
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return []

    x_centers = (np.linspace(bounds.min_x, bounds.max_x, resolution + 1)[:-1]
                 + np.linspace(bounds.min_x, bounds.max_x, resolution + 1)[1:]) / 2.0
    z_centers = (np.linspace(bounds.min_z, bounds.max_z, resolution + 1)[:-1]
                 + np.linspace(bounds.min_z, bounds.max_z, resolution + 1)[1:]) / 2.0

    hotspots: list[ErrorHotspot] = []
    for label_id in range(1, n_labels + 1):
        rows, cols = np.nonzero(labels == label_id)
        if rows.size < min_region_cells:
            continue

        cell_x = x_centers[cols]
        cell_z = z_centers[rows]
        centroid_x = float(np.mean(cell_x))
        centroid_z = float(np.mean(cell_z))

        dist = np.sqrt((cell_x - centroid_x) ** 2 + (cell_z - centroid_z) ** 2)
        radius = float(np.max(dist)) + 0.5 * max(cell_size_x, cell_size_z)
        radius = max(radius, min_radius)
        if max_radius is not None:
            radius = min(radius, max_radius)

        peak_error = float(np.max(np.abs(error[rows, cols])))
        hotspots.append(ErrorHotspot(
            x=centroid_x, z=centroid_z, radius=radius,
            peak_error=peak_error, n_cells=int(rows.size),
        ))

    hotspots.sort(key=lambda h: h.peak_error, reverse=True)
    return hotspots


def generate_hotspot_stamps(
    hotspots: Sequence[ErrorHotspot],
    brush: int = DEFAULT_HOTSPOT_BRUSH,
) -> list[Stamp]:
    """One placeholder-value Stamp per hotspot, centered and sized on it."""
    return [
        Stamp(x=h.x, z=h.z, radius=h.radius, value=PLACEHOLDER_VALUE, brush=brush)
        for h in hotspots
    ]


def refine_stamps(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bounds: BoundingBox,
    tolerance: float,
    resolution: int = DEFAULT_RESOLUTION,
    min_region_cells: int = DEFAULT_MIN_REGION_CELLS,
    min_radius: Optional[float] = DEFAULT_MIN_HOTSPOT_RADIUS_M,
    max_radius: Optional[float] = DEFAULT_MAX_HOTSPOT_RADIUS_M,
    brush: int = DEFAULT_HOTSPOT_BRUSH,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
    max_new_stamps: Optional[int] = None,
) -> tuple[list[Stamp], list[ErrorHotspot]]:
    """
    One adaptive refinement pass: find error hotspots directly from the
    binned error grid, add one stamp per hotspot (centered and sized on
    it, not on any pre-existing lattice cell), fit their heights against
    the existing (unchanged) coarse baseline, and return the combined
    stamp list.

    Existing stamps are never removed -- new stamps are appended after
    them, so under pull-toward-value semantics they refine on top of
    the existing baseline rather than replacing it (see
    terrain_model.py). Safe to call repeatedly on its own output.

    Returns (new_stamp_list, hotspots) -- hotspots is useful for
    reporting/diagnostics even though the stamps themselves are already
    folded into the returned list.
    """
    hotspots = find_error_hotspots(
        stamps, cloud, bounds, tolerance, resolution=resolution,
        min_region_cells=min_region_cells, min_radius=min_radius, max_radius=max_radius,
        bare_earth_only=bare_earth_only,
    )
    if max_new_stamps is not None:
        hotspots = hotspots[:max_new_stamps]

    if not hotspots:
        return list(stamps), hotspots

    new_positions = generate_hotspot_stamps(hotspots, brush=brush)
    fitted = fit_stamp_heights(
        new_positions, cloud,
        bare_earth_only=bare_earth_only, min_points=min_points,
        existing_stamps=stamps,
    )

    return list(stamps) + fitted, hotspots
