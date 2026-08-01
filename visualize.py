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
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, just writes files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from constants import DEBUG_IMAGE_SIZE, PREVIEW_LIDAR_HEIGHTMAP
from ingest.laz_reader import PointCloud
from ingest.osm import Feature
from terrain.bounding_box import BoundingBox
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel
from matplotlib.lines import Line2D

_DPI = 100

# Margins defined in absolute inches, not figure fractions -- this is
# what actually eliminates wasted padding, not just relocates it. Course
# bounds are always square (2000x2000) and every plot uses
# ax.set_aspect("equal"), so if the allocated axes rect isn't ALSO
# square in absolute terms, matplotlib shrinks the plot to fit the
# smaller dimension and centers it, leaving the mismatch as blank
# padding inside the rect (measured directly: ~140px of exactly this,
# landing almost entirely at the top, before this fix). Forcing a
# square *figure* and picking fractions independently for width vs.
# height (the previous approach) can't avoid this -- the fractions
# have to correspond to equal absolute sizes, which means computing
# them from a common plot size in inches, and letting the figure itself
# be non-square rather than forcing it square and eating the mismatch.
_PLOT_SIZE_IN = DEBUG_IMAGE_SIZE / _DPI * 0.8   # square plot area
_LEFT_MARGIN_IN = 1.2     # y-axis label + tick labels
_RIGHT_MARGIN_IN = 2.4    # colorbar (or dummy) + its label + tick labels
_BOTTOM_MARGIN_IN = 1.0   # x-axis label + tick labels
_TOP_MARGIN_IN = 0.8      # title

_FIG_W_IN = _LEFT_MARGIN_IN + _PLOT_SIZE_IN + _RIGHT_MARGIN_IN
_FIG_H_IN = _BOTTOM_MARGIN_IN + _PLOT_SIZE_IN + _TOP_MARGIN_IN
_FIGSIZE = (_FIG_W_IN, _FIG_H_IN)

# Distinct colors per brush type, for preview_hex.png
_BRUSH_COLORS = {8: "#4C72B0", 9: "#55A868", 10: "#C44E52", 54: "#8172B2"}
_DEFAULT_BRUSH_COLOR = "#888888"


# Fixed plot-area and colorbar positions, in figure-fraction coordinates
# -- every preview uses the exact same rectangle (in absolute inches,
# per _FIG_W_IN/_FIG_H_IN above) for its main plot and its colorbar
# (real or dummy), so no preview's pixel dimensions or internal layout
# depend on its own data (e.g. a wider "-10.0" tick label vs "1.0" on a
# different preview). This is what actually fixes switching-between-
# previews reflow in the GUI, which bbox_inches="tight" (removed from
# _save below) could not: tight-bbox crops to the rendered content's
# own bounding box, which shifts with tick label width -- exactly the
# thing being fixed here.
_PLOT_RECT = (
    _LEFT_MARGIN_IN / _FIG_W_IN, _BOTTOM_MARGIN_IN / _FIG_H_IN,
    _PLOT_SIZE_IN / _FIG_W_IN, _PLOT_SIZE_IN / _FIG_H_IN,
)
_COLORBAR_RECT = (
    (_LEFT_MARGIN_IN + _PLOT_SIZE_IN + 0.3) / _FIG_W_IN, _BOTTOM_MARGIN_IN / _FIG_H_IN,
    0.35 / _FIG_W_IN, _PLOT_SIZE_IN / _FIG_H_IN,
)


def _new_figure(bounds: BoundingBox):
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_axes(_PLOT_RECT)
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_z, bounds.max_z)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    return fig, ax


def _add_colorbar(fig, mappable, label: str) -> None:
    cax = fig.add_axes(_COLORBAR_RECT)
    fig.colorbar(mappable, cax=cax, label=label)


def _add_dummy_scale(fig) -> None:
    """
    Blank placeholder occupying the same rectangle _add_colorbar would
    -- so preview_hex.png (brush-type legend, not a continuous
    colorbar) still reserves identical space, and its plot area ends
    up pixel-identical in size/position to every colorbar-having
    preview.
    """
    cax = fig.add_axes(_COLORBAR_RECT)
    cax.axis("off")


def _archive_existing(path: Path, max_history: int = 10) -> None:
    """
    If `path` already exists, shift it into a numbered history rather
    than overwrite it silently: path -> path_1, an existing path_1 ->
    path_2, and so on, so the previous run's preview stays around for
    comparison. Anything beyond max_history is dropped rather than
    kept forever.
    """
    if not path.exists():
        return

    stem, suffix, parent = path.stem, path.suffix, path.parent

    existing_n = 0
    while (parent / f"{stem}_{existing_n + 1}{suffix}").exists():
        existing_n += 1

    for n in range(existing_n, 0, -1):
        src = parent / f"{stem}_{n}{suffix}"
        if n + 1 > max_history:
            src.unlink()
        else:
            src.rename(parent / f"{stem}_{n + 1}{suffix}")

    path.rename(parent / f"{stem}_1{suffix}")


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _archive_existing(path)
    # No bbox_inches="tight": that crops to the rendered content's own
    # bounding box, which shifts with tick-label width -- exactly what
    # causes different previews to come out at different pixel sizes.
    # _PLOT_RECT/_COLORBAR_RECT (see _new_figure) already give a fixed,
    # tight-looking layout without being content-dependent.
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _save_transparent(fig, path: Path) -> None:
    """Like _save, but with a transparent background -- for the OSM overlay, meant to be composited over another preview, not viewed alone."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _archive_existing(path)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _new_overlay_figure(bounds: BoundingBox):
    """
    Same _PLOT_RECT position/size as _new_figure (so content lines up
    pixel-for-pixel with every other preview when composited), but with
    a transparent background and no axis chrome (ticks/labels/spines) --
    those would double up visually on top of whatever base preview this
    gets composited over, which already has its own.
    """
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes(_PLOT_RECT)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_z, bounds.max_z)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# Distinct colors per OSM feature kind (see ingest/osm.py's classify_way).
# Deliberately saturated/high-contrast, since this overlay is meant to
# be composited at partial opacity over another preview -- muted colors
# would wash out and become hard to distinguish once blended.
_OSM_FEATURE_COLORS = {
    "green": "#3CB371",
    "tee": "#2E8B57",
    "fairway": "#7CFC00",
    "rough": "#556B2F",
    "bunker": "#EDC9AF",
    "water": "#4682B4",
    "cartpath": "#8B7355",
    "path": "#A9A9A9",
    "building": "#B22222",
    "wood": "#228B22",
    "hole": "#FFD700",
}
_OSM_DEFAULT_COLOR = "#FF00FF"  # unclassified kind -- deliberately jarring so it's obvious


def _draw_osm_feature(ax, feature: Feature) -> None:
    color = _OSM_FEATURE_COLORS.get(feature.kind, _OSM_DEFAULT_COLOR)
    geom = feature.geometry
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for part in parts:
        if part.geom_type == "Polygon":
            xs, zs = part.exterior.xy
            ax.fill(xs, zs, facecolor=color, edgecolor=color, alpha=0.55, linewidth=1.2)
        elif part.geom_type == "LineString":
            xs, zs = part.xy
            ax.plot(xs, zs, color=color, linewidth=2.0, alpha=0.85, solid_capstyle="round")


def render_osm_features(features: Sequence[Feature], bounds: BoundingBox, path: Path) -> None:
    """
    Transparent PNG of every OSM Feature (see ingest/osm.py), colored by
    kind, at the exact same plot-area position/size every other preview
    uses -- meant to be alpha-composited over any of them (in the GUI,
    not baked into a new file per base preview), not viewed standalone.
    """
    fig, ax = _new_overlay_figure(bounds)

    kinds_present = set()
    for feature in features:
        _draw_osm_feature(ax, feature)
        kinds_present.add(feature.kind)

    if kinds_present:
        handles = [
            Line2D(
                [0], [0], marker="s", linestyle="", markersize=8,
                markerfacecolor=_OSM_FEATURE_COLORS.get(kind, _OSM_DEFAULT_COLOR),
                markeredgecolor="black", label=kind,
            )
            for kind in sorted(kinds_present)
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.7)

    _save_transparent(fig, path)


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
    _add_colorbar(fig, sc, "elevation (m)")
    ax.set_title(f"LIDAR point cloud ({n:,} points)")
    _save(fig, path)


def _add_compass_labels(ax, bounds: BoundingBox) -> None:
    """
    Burn N/S/E/W labels onto the plot edges, per this pipeline's
    coordinate convention: +z = north, +x = east (inherited directly
    from the source LAZ's projected northing/easting, never flipped in
    ingest.laz_reader or terrain_model.py). This lets a rendered image
    be checked against known site geography to confirm whether PGA's
    own grid actually matches this convention once exported (see
    writer.py's NOTE on sign convention) -- if compass directions don't
    match reality in-game, the mismatch is in the game's own grid
    mapping or the writer's position transform, not in this image.
    """
    cx = (bounds.min_x + bounds.max_x) / 2.0
    cz = (bounds.min_z + bounds.max_z) / 2.0
    style = dict(color="red", fontsize=14, fontweight="bold")
    ax.text(cx, bounds.max_z, "N", **style, ha="center", va="bottom")
    ax.text(cx, bounds.min_z, "S", **style, ha="center", va="top")
    ax.text(bounds.max_x, cz, "E", **style, ha="left", va="center")
    ax.text(bounds.min_x, cz, "W", **style, ha="right", va="center")


def render_lidar_heightmap(
    cloud: PointCloud,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 800,
) -> None:
    """
    Grayscale binned heightmap of the point cloud, with N/S/E/W labels
    burned in (see _add_compass_labels), for checking against known
    site geography whether PGA's in-game grid orientation actually
    matches this pipeline's -z=south/+z=north/-x=west/+x=east
    convention once exported and viewed in-editor.

    Distinct from render_lidar_preview(): that's a scatter plot colored
    by a perceptual colormap (good for spotting classification/coverage
    issues); this is a proper binned grid in true grayscale, which
    reads more like an actual heightmap/DEM and is easier to compare
    shape-for-shape against an in-game view.
    """
    heights = _bin_point_cloud(cloud, bounds, resolution)

    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        heights, origin="lower", cmap="gray",
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "elevation (m)")
    _add_compass_labels(ax, bounds)
    ax.set_title(f"LIDAR heightmap ({resolution}x{resolution})")
    _save(fig, path)


def _stamp_patches(stamps: Sequence[Stamp], colors: list[str]) -> PatchCollection:
    circles = [Circle((s.x, s.z), s.radius) for s in stamps]
    return PatchCollection(circles, facecolor=colors, edgecolor="black", linewidths=0.3, alpha=0.35)


def _set_title(ax, base_title: str, extra_label: Optional[str] = None) -> None:
    """
    Set the plot title, with an optional second line summarizing
    whatever parameters actually produced this preview (e.g. the
    refine-terrain settings used for the latest pass) -- so a preview
    is self-documenting about what generated it without needing to
    cross-reference a separate log.
    """
    ax.set_title(f"{base_title}\n{extra_label}" if extra_label else base_title, fontsize=10)


def render_hex_preview(
    stamps: Sequence[Stamp], bounds: BoundingBox, path: Path, extra_label: Optional[str] = None,
) -> None:
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
    _set_title(ax, f"Stamp layout ({len(stamps)} stamps)", extra_label)
    _add_dummy_scale(fig)
    _save(fig, path)


def render_stamps_preview(
    stamps: Sequence[Stamp], bounds: BoundingBox, path: Path, extra_label: Optional[str] = None,
) -> None:
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

    _add_colorbar(fig, coll, "fitted value (m)")
    _set_title(ax, f"Stamp values ({len(stamps)} stamps)", extra_label)
    _save(fig, path)


def render_height_preview(
    model: TerrainModel,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 400,
    extra_label: Optional[str] = None,
) -> None:
    """TerrainModel's predicted height field over `bounds`."""
    grid = model.render(resolution=resolution, bounds=bounds)

    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        grid, origin="lower", cmap="terrain",
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "predicted height (m)")
    _set_title(ax, f"Predicted terrain height ({resolution}x{resolution})", extra_label)
    _save(fig, path)


def _bin_point_cloud(
    cloud: PointCloud, bounds: BoundingBox, resolution: int, bare_earth_only: bool = False,
) -> np.ndarray:
    """
    Bin cloud.elevation into a resolution x resolution grid over bounds
    (mean elevation per cell, NaN where a cell has no points).

    bare_earth_only defaults to False here since render_lidar_heightmap
    (the general orientation/inspection view) benefits from showing
    buildings and vegetation, not hiding them. render_error_preview
    passes True explicitly -- comparing predicted terrain against
    building-roof or treetop elevation isn't a meaningful error signal
    (see terrain/adaptive_refine.py, which had this exact bug).
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
    return means  # NaN where counts == 0


def render_error_preview(
    model: TerrainModel,
    cloud: PointCloud,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 200,
    extra_label: Optional[str] = None,
) -> None:
    """
    Predicted height vs. binned LIDAR elevation, as signed error
    (predicted - actual) on a diverging colormap centered at zero.
    Cells with no LIDAR points are left blank (NaN), not zero -- a
    missing measurement isn't the same as a confirmed-zero error.
    """
    actual = _bin_point_cloud(cloud, bounds, resolution, bare_earth_only=True)
    predicted = model.render(resolution=resolution, bounds=bounds)
    error = predicted - actual

    finite = error[np.isfinite(error)]
    if finite.size == 0:
        raise ValueError("No overlapping LIDAR coverage in bounds -- can't compute error preview.")
    vmax = np.percentile(np.abs(finite), 98)  # robust to a few outlier cells

    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        error, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "predicted - actual (m)")

    rms = float(np.sqrt(np.mean(np.square(finite))))
    _set_title(ax, f"Height error, RMS={rms:.2f} m ({resolution}x{resolution})", extra_label)
    _save(fig, path)
