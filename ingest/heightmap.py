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

from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox

DEFAULT_HEIGHTMAP_RESOLUTION = 2000


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
