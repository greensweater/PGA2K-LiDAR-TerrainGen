"""
ingest/heightmap.py

Rasterizes bare-earth LIDAR points into a single, persistent height
grid -- computed and cached for height_fit.py's stamp fitting, 
adaptive_refine.py's error scoring, etc. vs. KD-Tree walking

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox

DEFAULT_HEIGHTMAP_RESOLUTION = 2000

DEFAULT_FILL_MAX_ITERATIONS_PER_LEVEL = 200
DEFAULT_FILL_TOLERANCE = 1e-4
DEFAULT_FILL_MIN_COARSE_RESOLUTION = 32


def rasterize_ground_heightmap(
    cloud: PointCloud,
    bounds: BoundingBox,
    resolution: int = DEFAULT_HEIGHTMAP_RESOLUTION,
    bare_earth_only: bool = True,
) -> np.ndarray:
    """
    Mean bare-earth elevation per cell (proper area-averaging over
    every LIDAR point landing in a cell, not nearest-neighbor
    sampling), resolution x resolution, NaN where a cell has no
    points at all. Same row=z/col=x, bin-center convention as
    adaptive_refine.py's _bin_actual_elevation, which this is lifted
    directly from -- the only real change is computing it once as a
    persisted artifact, not recomputing it from raw points on every
    find_error_hotspots call and every stamp fit.
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


def save_heightmap(heights: np.ndarray, bounds: BoundingBox, path: Path) -> None:
    """Persist the rasterized heightmap (float32, NaN gaps preserved) plus the bounds it was built over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        heights=heights.astype(np.float32),
        min_x=bounds.min_x, min_z=bounds.min_z, max_x=bounds.max_x, max_z=bounds.max_z,
    )


def load_heightmap(path: Path) -> tuple[np.ndarray, BoundingBox]:
    with np.load(path) as data:
        heights = data["heights"].astype(np.float64)
        bounds = BoundingBox(
            min_x=float(data["min_x"]), min_z=float(data["min_z"]),
            max_x=float(data["max_x"]), max_z=float(data["max_z"]),
        )
    return heights, bounds


def downsample_heightmap(
    heights: np.ndarray, bounds: BoundingBox, target_resolution: int,
) -> np.ndarray:
    """
    Re-bin a fine heightmap down to a coarser target_resolution x
    target_resolution grid, mean-per-cell (NaN where a coarse cell has
    no valid fine cells) -- the heightmap-sourced replacement for
    adaptive_refine.py's _bin_actual_elevation, which did the same
    mean-per-cell binning directly from raw LIDAR points. Works for
    any resolution ratio (not just exact integer multiples), by
    treating the fine heightmap's own cell centers as points and
    re-binning them, same approach as rasterize_ground_heightmap
    itself, just with the fine grid's cells as input instead of raw
    LIDAR points.
    """
    resolution_z, resolution_x = heights.shape
    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution_x + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution_z + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    xx, zz = np.meshgrid(x_centers, z_centers)

    valid = np.isfinite(heights)
    x_valid, z_valid, h_valid = xx[valid], zz[valid], heights[valid]

    target_x_edges = np.linspace(bounds.min_x, bounds.max_x, target_resolution + 1)
    target_z_edges = np.linspace(bounds.min_z, bounds.max_z, target_resolution + 1)

    sums, _, _ = np.histogram2d(z_valid, x_valid, bins=[target_z_edges, target_x_edges], weights=h_valid)
    counts, _, _ = np.histogram2d(z_valid, x_valid, bins=[target_z_edges, target_x_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means


def _jacobi_relax(
    seeded: np.ndarray, missing: np.ndarray, iterations: int, tol: float,
) -> np.ndarray:
    """
    In-place-equivalent harmonic relaxation: repeatedly replace every
    `missing` cell with the plain mean of its 4-connected neighbors
    (edge-replicated at the grid boundary), leaving every other cell
    fixed at whatever value `seeded` already gives it. This is Jacobi
    iteration for Laplace's equation with `~missing` cells as fixed
    (Dirichlet) boundary values -- the well-behaved, direction-blind
    version of "flood-fill with the average of adjacent points": a
    single raster-scan flood-fill pass would bias the result toward
    whichever direction got filled first, since later cells in the
    scan order see already-filled neighbors while earlier ones don't.
    Relaxing repeatedly (rather than filling once and stopping)
    removes that bias -- every missing cell converges toward a value
    consistent with the whole boundary around it, not just whichever
    neighbor happened to be filled first.

    `seeded` must have no NaNs left in it (see fill_heightmap_gaps'
    initial nearest-valid seeding and coarse-to-fine prolongation) --
    only `missing` says which cells are still free to move; the actual
    array values are used directly in the neighbor-mean arithmetic
    every iteration, so a stray NaN here would poison every cell it
    touches, not just the one it started in.

    Stops early once the largest single-cell change between
    iterations (over just the `missing` cells) drops below `tol`.
    """
    filled = seeded
    if not missing.any():
        return filled
    for _ in range(iterations):
        padded = np.pad(filled, 1, mode="edge")
        neighbor_avg = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +
            padded[1:-1, :-2] + padded[1:-1, 2:]
        ) / 4.0
        new_filled = filled.copy()
        new_filled[missing] = neighbor_avg[missing]
        delta = float(np.max(np.abs(new_filled[missing] - filled[missing])))
        filled = new_filled
        if delta < tol:
            break
    return filled


def fill_heightmap_gaps(
    heights: np.ndarray,
    bounds: BoundingBox,
    max_iterations_per_level: int = DEFAULT_FILL_MAX_ITERATIONS_PER_LEVEL,
    tol: float = DEFAULT_FILL_TOLERANCE,
    min_coarse_resolution: int = DEFAULT_FILL_MIN_COARSE_RESOLUTION,
) -> np.ndarray:
    """
    Harmonic (Laplace-equation) inpainting of every NaN cell in
    `heights` -- water, buildings, and any other no-ground-points gap
    -- via iterative neighbor-average relaxation (see _jacobi_relax),
    not a single-pass flood-fill. Every originally-valid cell is left
    completely untouched, exactly as measured; only NaN cells are ever
    written.

    This deliberately does NOT special-case water polygons to a flat
    constant height: the in-game water is a separately-placed, sized
    plane object that gets fit into whatever recess the terrain
    itself has, so what this needs to produce under a pond is a
    plausible *recessed basin* shape blending in from the shore all
    around -- not a flat disc -- and one fill rule for every kind of
    gap (water, buildings, anything else) is simpler to reason about
    than remembering which kind gets which treatment.

    Solved as a coarse-to-fine pyramid rather than plain relaxation at
    the full resolution directly: same-resolution Jacobi relaxation
    converges in roughly (hole diameter in cells)^2 iterations, which
    is fine for a building footprint (tens of cells across) but far
    too slow for a wide lake spanning hundreds of meters (hundreds of
    cells across, at this module's default 2000x2000 resolution).
    Downsampling first (via downsample_heightmap, which already means
    only over genuinely valid data) shrinks every hole by the same
    factor, so the coarse solve converges fast; upsampling that coarse
    solution back up seeds the next-finer level already close to
    right, needing only a handful of refining iterations rather than
    solving from scratch -- a standard multigrid trick, applied here
    to this specific fill instead of a general PDE solver.

    Raises ValueError if `heights` isn't square (the pyramid halves
    both axes together) or has no valid cells at all to fill from.
    """
    if heights.shape[0] != heights.shape[1]:
        raise ValueError(
            f"fill_heightmap_gaps expects a square heightmap, got shape {heights.shape}"
        )
    resolution = heights.shape[0]

    missing_full = ~np.isfinite(heights)
    if not missing_full.any():
        return heights.copy()
    if missing_full.all():
        raise ValueError("heightmap has no valid cells at all to fill from")

    # Pyramid of resolutions, coarsest first, halving down to
    # min_coarse_resolution (or landing above it if resolution isn't
    # a clean power-of-two multiple -- max() keeps every step a real
    # reduction without ever going below the floor).
    resolutions = [resolution]
    while resolutions[-1] > min_coarse_resolution:
        resolutions.append(max(min_coarse_resolution, resolutions[-1] // 2))
    resolutions = resolutions[::-1]

    # Coarsest level: downsample straight from the original data (or
    # use it directly if the heightmap is already <= the coarse
    # floor), seed any still-missing coarse cell with its nearest
    # valid coarse cell (a much better starting guess than a flat 0 --
    # converges faster and never introduces an artificial flat patch),
    # then relax to convergence -- cheap, since this grid is small.
    current = heights if resolutions[0] == resolution else downsample_heightmap(heights, bounds, resolutions[0])
    current_missing = ~np.isfinite(current)
    if current_missing.all():
        # Entire course has no bare-earth coverage at all at even the
        # coarsest level (shouldn't happen in practice, but a global
        # mean is a safe, inert fallback rather than raising here).
        current = np.full_like(current, float(np.nanmean(heights)), dtype=np.float64)
    else:
        nearest_idx = ndimage.distance_transform_edt(
            current_missing, return_distances=False, return_indices=True
        )
        seeded = np.where(current_missing, current[tuple(nearest_idx)], current)
        current = _jacobi_relax(seeded, current_missing, max_iterations_per_level * 4, tol)

    # Refine level by level: upsample the previous (coarser) solution
    # as the initial guess for every still-missing cell at this level,
    # re-impose this level's own actually-known cells exactly (from a
    # fresh downsample of the original data, not from the coarser
    # estimate), then relax a bit more to correct whatever the
    # upsample interpolation got wrong right at the real boundary.
    for target_res in resolutions[1:]:
        zoom_factor = target_res / current.shape[0]
        upsampled = ndimage.zoom(current, zoom_factor, order=1)
        if upsampled.shape != (target_res, target_res):
            # ndimage.zoom can land one cell off target_res due to
            # rounding -- force the exact shape needed to align with
            # this level's own known-data grid.
            upsampled = ndimage.zoom(
                current, (target_res / current.shape[0], target_res / current.shape[1]), order=1
            )
            upsampled = upsampled[:target_res, :target_res]

        this_level_actual = heights if target_res == resolution else downsample_heightmap(heights, bounds, target_res)
        this_level_missing = ~np.isfinite(this_level_actual)
        seeded = np.where(this_level_missing, upsampled, this_level_actual)
        current = _jacobi_relax(seeded, this_level_missing, max_iterations_per_level, tol)

    return current


def query_heightmap_cells(
    heights: np.ndarray, bounds: BoundingBox, x: float, z: float, radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (px, pz, values) for every valid (non-NaN) heightmap cell whose
    center falls within `radius` of (x, z) -- the heightmap-sourced
    replacement for querying/filtering raw LIDAR points via a
    PointCloud's KD-tree (see adaptive_refine.py's per-hotspot
    candidate scoring). Empty arrays if nothing valid is in range.
    """
    resolution_z, resolution_x = heights.shape
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution_x
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution_z

    col_min = max(0, int((x - radius - bounds.min_x) / cell_size_x))
    col_max = min(resolution_x, int((x + radius - bounds.min_x) / cell_size_x) + 1)
    row_min = max(0, int((z - radius - bounds.min_z) / cell_size_z))
    row_max = min(resolution_z, int((z + radius - bounds.min_z) / cell_size_z) + 1)
    if col_min >= col_max or row_min >= row_max:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution_x + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution_z + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    sub_x = x_centers[col_min:col_max]
    sub_z = z_centers[row_min:row_max]
    xx, zz = np.meshgrid(sub_x, sub_z)
    dist = np.hypot(xx - x, zz - z)

    sub_heights = heights[row_min:row_max, col_min:col_max]
    valid = (dist <= radius) & np.isfinite(sub_heights)
    return xx[valid], zz[valid], sub_heights[valid]


def sample_heightmap_mean(
    heights: np.ndarray, bounds: BoundingBox, x: float, z: float, radius: float,
    min_valid_cells: int = 1,
) -> Optional[float]:
    """
    Mean of every heightmap cell whose center falls within `radius` of
    (x, z) -- the direct, KD-tree-free replacement for querying and
    averaging raw LIDAR points within a stamp's footprint. Returns
    None if fewer than `min_valid_cells` valid (non-NaN, in-range)
    cells exist -- e.g. a stamp sitting entirely over a lake or
    building, where bare-earth points (and therefore heightmap
    coverage) don't exist.
    """
    resolution_z, resolution_x = heights.shape
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution_x
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution_z

    col_min = max(0, int((x - radius - bounds.min_x) / cell_size_x))
    col_max = min(resolution_x, int((x + radius - bounds.min_x) / cell_size_x) + 1)
    row_min = max(0, int((z - radius - bounds.min_z) / cell_size_z))
    row_max = min(resolution_z, int((z + radius - bounds.min_z) / cell_size_z) + 1)
    if col_min >= col_max or row_min >= row_max:
        return None

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution_x + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution_z + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    sub_x = x_centers[col_min:col_max]
    sub_z = z_centers[row_min:row_max]
    xx, zz = np.meshgrid(sub_x, sub_z)
    dist = np.hypot(xx - x, zz - z)

    sub_heights = heights[row_min:row_max, col_min:col_max]
    valid = (dist <= radius) & np.isfinite(sub_heights)
    if np.count_nonzero(valid) < min_valid_cells:
        return None
    return float(np.mean(sub_heights[valid]))
