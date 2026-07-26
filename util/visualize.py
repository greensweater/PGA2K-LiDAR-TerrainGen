"""
visualize.py

Diagnostic PNG previews for every major pipeline stage. These are
output-only: nothing here is ever read back in as terrain data (see
"Important Design Rules": never rasterize terrain internally, and
terrain_model.py's own render() docstring -- these images exist so the
compiler is never a black box, not to feed later stages).

Five previews, one per stage:
    preview_lidar.png    -- raw point cloud, colored by elevation
    preview_hex.png      -- stamp layout: center + radius per stamp
    preview_stamps.png   -- same layout, colored by fitted value instead
                            of brush type, so mis-fits are visible at a
                            glance
    preview_height.png   -- TerrainModel's predicted height field
    preview_error.png    -- predicted height vs. binned LIDAR elevation,
                            diverging colormap centered on zero error
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, just writes files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from constants import DEBUG_IMAGE_SIZE
from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel

_DPI = 100
_FIGSIZE = (DEBUG_IMAGE_SIZE / _DPI, DEBUG_IMAGE_SIZE / _DPI)

# Distinct colors per brush type, for preview_hex.png
_BRUSH_COLORS = {8: "#4C72B0", 9: "#55A868", 10: "#C44E52", 54: "#8172B2"}
_DEFAULT_BRUSH_COLOR = "#888888"


def _new_figure(bounds: BoundingBox):
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_z, bounds.max_z)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    return fig, ax


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def render_lidar_preview(cloud: PointCloud, path: Path, max_points: int = 200_000) -> None:
    """
    Scatter the point cloud in the x/z plane, colored by elevation.

    Subsamples to max_points for large clouds -- this is a diagnostic
    image, not a data product, so it doesn't need every point.
    """
    bounds = cloud.bounds
    fig, ax = _new_figure(bounds)

    n = cloud.count
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
    else:
        idx = slice(None)

    sc = ax.scatter(
        cloud.x[idx], cloud.z[idx], c=cloud.elevation[idx],
        s=1, cmap="terrain", linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="elevation (m)", fraction=0.046, pad=0.04)
    ax.set_title(f"LIDAR point cloud ({n:,} points)")
    _save(fig, path)


def _stamp_patches(stamps: Sequence[Stamp], colors: list[str]) -> PatchCollection:
    circles = [Circle((s.x, s.z), s.radius) for s in stamps]
    return PatchCollection(circles, facecolor=colors, edgecolor="black", linewidths=0.3, alpha=0.35)


def render_hex_preview(stamps: Sequence[Stamp], bounds: BoundingBox, path: Path) -> None:
    """Stamp layout: one circle per stamp (center + radius), colored by brush type."""
    fig, ax = _new_figure(bounds)

    colors = [_BRUSH_COLORS.get(s.brush, _DEFAULT_BRUSH_COLOR) for s in stamps]
    ax.add_collection(_stamp_patches(stamps, colors))
    ax.scatter([s.x for s in stamps], [s.z for s in stamps], c="black", s=2, zorder=3)

    used_brushes = sorted(set(s.brush for s in stamps))
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=_BRUSH_COLORS.get(b, _DEFAULT_BRUSH_COLOR),
                   markeredgecolor="black", label=f"type {b}")
        for b in used_brushes
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    ax.set_title(f"Stamp layout ({len(stamps)} stamps)")
    _save(fig, path)


def render_stamps_preview(stamps: Sequence[Stamp], bounds: BoundingBox, path: Path) -> None:
    """
    Same layout as preview_hex.png, but colored by fitted value instead
    of brush type -- makes unfit stamps (still at their placeholder,
    typically 0.0) and outlier fits visually obvious.
    """
    fig, ax = _new_figure(bounds)

    values = np.array([s.value for s in stamps])
    circles = [Circle((s.x, s.z), s.radius) for s in stamps]
    coll = PatchCollection(circles, edgecolor="black", linewidths=0.3, alpha=0.6)
    coll.set_array(values)
    coll.set_cmap("terrain")
    ax.add_collection(coll)
    ax.scatter([s.x for s in stamps], [s.z for s in stamps], c="black", s=2, zorder=3)

    fig.colorbar(coll, ax=ax, label="fitted value (m)", fraction=0.046, pad=0.04)
    ax.set_title(f"Stamp values ({len(stamps)} stamps)")
    _save(fig, path)


def render_height_preview(
    model: TerrainModel,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 400,
) -> None:
    """TerrainModel's predicted height field over `bounds`."""
    grid = model.render(resolution=resolution, bounds=bounds)

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    im = ax.imshow(
        grid, origin="lower", cmap="terrain",
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, label="predicted height (m)", fraction=0.046, pad=0.04)
    ax.set_title(f"Predicted terrain height ({resolution}x{resolution})")
    _save(fig, path)


def _bin_point_cloud(cloud: PointCloud, bounds: BoundingBox, resolution: int) -> np.ndarray:
    """
    Bin cloud.elevation into a resolution x resolution grid over bounds
    (mean elevation per cell, NaN where a cell has no points).
    """
    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)

    sums, _, _ = np.histogram2d(cloud.z, cloud.x, bins=[z_edges, x_edges], weights=cloud.elevation)
    counts, _, _ = np.histogram2d(cloud.z, cloud.x, bins=[z_edges, x_edges])

    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return means  # NaN where counts == 0


def render_error_preview(
    model: TerrainModel,
    cloud: PointCloud,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 200,
) -> None:
    """
    Predicted height vs. binned LIDAR elevation, as signed error
    (predicted - actual) on a diverging colormap centered at zero.
    Cells with no LIDAR points are left blank (NaN), not zero -- a
    missing measurement isn't the same as a confirmed-zero error.
    """
    actual = _bin_point_cloud(cloud, bounds, resolution)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual

    finite = error[np.isfinite(error)]
    if finite.size == 0:
        raise ValueError("No overlapping LIDAR coverage in bounds -- can't compute error preview.")
    vmax = np.percentile(np.abs(finite), 98)  # robust to a few outlier cells

    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=_DPI)
    im = ax.imshow(
        error, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.colorbar(im, ax=ax, label="predicted - actual (m)", fraction=0.046, pad=0.04)

    rms = float(np.sqrt(np.mean(np.square(finite))))
    ax.set_title(f"Height error, RMS={rms:.2f} m ({resolution}x{resolution})")
    _save(fig, path)
