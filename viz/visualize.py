"""
viz/visualize.py

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

import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, just writes files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from terrain.brush_profiles import BRUSH_PROFILES, SHAPE_SQUARE
from matplotlib.collections import PatchCollection

from constants import DEBUG_IMAGE_SIZE, PREVIEW_LIDAR_HEIGHTMAP
from ingest.laz_reader import PointCloud
from ingest.osm import Feature
from shapely.geometry.base import BaseGeometry
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
_PLOT_SIZE_IN = DEBUG_IMAGE_SIZE / _DPI   # square plot area -- exactly 2000x2000 px at _DPI=100,
                                            # matching the course's native 2000x2000 m size 1:1 (a
                                            # previous *0.8 factor here left the actual plotted data
                                            # area at only 1600x1600 px despite the course itself
                                            # being 2000x2000 -- removed so stamp/heightmap previews
                                            # actually show native per-meter resolution, not a
                                            # downscaled version of it).
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


_VERSION_SUFFIX_RE = re.compile(r"^(.*)_(\d+)(\.[^.]+)$")


def strip_preview_version(filename: str) -> str:
    """
    'preview_lidar_5.png' -> 'preview_lidar.png' -- recovers the
    logical preview "kind" from a versioned filename, for callers that
    need to know which series a file belongs to (e.g. picking the
    right OSM overlay variant) without caring which version it is.
    """
    m = _VERSION_SUFFIX_RE.match(filename)
    if not m:
        return filename
    stem, _n, suffix = m.groups()
    return f"{stem}{suffix}"


def find_all_preview_versions(directory: Path, base_name: str) -> list:
    """
    Every existing version of `base_name`'s series (e.g.
    "preview_lidar.png" finds preview_lidar_0.png, _1.png, ...) in
    `directory`, sorted by version number descending -- index 0 is
    always the latest.
    """
    stem, suffix = Path(base_name).stem, Path(base_name).suffix
    if not directory.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    found = []
    for f in directory.iterdir():
        m = pattern.match(f.name)
        if m:
            found.append((int(m.group(1)), f))
    found.sort(key=lambda t: t[0], reverse=True)
    return [f for _n, f in found]


def find_latest_preview(directory: Path, base_name: str):
    """The highest-numbered existing version of `base_name`'s series, or None if none exists yet."""
    versions = find_all_preview_versions(directory, base_name)
    return versions[0] if versions else None


def _next_version_path(path: Path) -> Path:
    """
    Every preview is written as {stem}_{N}{suffix} (N starting at 0),
    never as a bare unsuffixed name -- "latest" is simply whichever N
    is highest on disk. This replaces an earlier scheme that kept the
    current file unsuffixed and shifted every archived version up by
    one on each write (path -> path_1, existing path_1 -> path_2, and
    so on): that cascaded through the *entire* history on every single
    write, and made "undo" a two-step delete-then-rename dance instead
    of just deleting the newest file. Append-only means there is
    nothing to rename, ever, in either direction -- writing a new
    version never touches an existing file, and undoing the latest
    version is exactly one delete, nothing else.
    """
    stem, suffix, parent = path.stem, path.suffix, path.parent
    existing = find_all_preview_versions(parent, path.name)
    next_n = 0
    if existing:
        m = _VERSION_SUFFIX_RE.match(existing[0].name)
        next_n = int(m.group(2)) + 1
    return parent / f"{stem}_{next_n}{suffix}"


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _next_version_path(path)
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
    path = _next_version_path(path)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def _new_overlay_figure(bounds: BoundingBox):
    """
    Same _PLOT_RECT position/size as _new_figure (so content lines up
    pixel-for-pixel with every other preview when composited), with a
    transparent background and no axis chrome (ticks/labels/spines) --
    those would double up visually on top of whatever base preview this
    gets composited over, which already has its own.

    Callers that need an opaque (not transparent) background -- e.g.
    render_mask_preview -- can override fig.patch/ax.patch after
    getting these back; NOT via ax.axis("off"), which also calls
    set_frame_on(False) and hides the axes' own background patch
    entirely (confirmed directly: ax.set_facecolor(...) has no visible
    effect at all with axis("off") on). Hiding just the ticks/spines
    keeps the patch intact either way, transparent or not.
    """
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes(_PLOT_RECT)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(bounds.min_x, bounds.max_x)
    ax.set_ylim(bounds.min_z, bounds.max_z)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


# Distinct fill/edge colors per OSM feature kind (see ingest/osm.py's
# classify_way) -- (fill_color, edge_color), fill_color=None meaning no
# fill at all (edge-only outline). Split into a pair (rather than one
# color used for both, as this used to be) specifically so a kind can
# have a transparent interior with a distinctly-colored border --
# heavyrough/wood are large, often course-spanning background areas
# where a solid fill would obscure everything else in this overlay,
# but their outline still needs to be visible and distinguishable from
# each other.
_OSM_FEATURE_STYLES: dict[str, tuple[Optional[str], str]] = {
    "green": ("#3CB371", "#3CB371"),
    "tee": ("#2E8B57", "#2E8B57"),
    "fairway": ("#7CFC00", "#7CFC00"),
    "rough": ("#556B2F", "#556B2F"),
    "heavyrough": (None, "#006400"),         # transparent, dark green border
    "bunker": ("#EDC9AF", "#EDC9AF"),
    "water": ("#4682B4", "#4682B4"),
    "cartpath": ("#FFFFFF", "#FFFFFF"),      # white
    "service_road": ("#808080", "#000000"),  # same as roadway
    "roadway": ("#808080", "#000000"),       # same fill as pavement, black border
    "driveway": ("#808080", "#FFA500"),      # same gray (surface3 family), orange border to stand apart
    "path": ("#A9A9A9", "#A9A9A9"),
    "pavement": ("#808080", "#FFFFFF"),      # gray, darker than path's; white border
    "building": ("#B22222", "#B22222"),
    "wood": (None, "#90EE90"),               # transparent, light green border
    "mulch": ("#C8A165", "#C8A165"),         # light brown, solid fill (like cartpath's old style)
    "hole": ("#FFD700", "#FFD700"),
}
_OSM_DEFAULT_COLOR = "#FF00FF"  # unclassified kind -- deliberately jarring so it's obvious


def _feature_style(kind: str) -> tuple[Optional[str], str]:
    """(fill_color_or_None, edge_color) for an OSM feature kind -- see _OSM_FEATURE_STYLES."""
    return _OSM_FEATURE_STYLES.get(kind, (_OSM_DEFAULT_COLOR, _OSM_DEFAULT_COLOR))


def _draw_osm_feature(ax, feature: Feature) -> None:
    fill_color, edge_color = _feature_style(feature.kind)
    geom = feature.geometry
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    for part in parts:
        if part.geom_type == "Polygon":
            xs, zs = part.exterior.xy
            if fill_color is None:
                ax.fill(xs, zs, facecolor="none", edgecolor=edge_color, alpha=0.9, linewidth=1.5)
            else:
                ax.fill(xs, zs, facecolor=fill_color, edgecolor=edge_color, alpha=0.55, linewidth=1.2)
        elif part.geom_type == "LineString":
            xs, zs = part.xy
            ax.plot(xs, zs, color=edge_color, linewidth=2.0, alpha=0.85, solid_capstyle="round")


def render_osm_features(
    features: Sequence[Feature], bounds: BoundingBox, path: Path,
    crop_box: Optional[BoundingBox] = None,
) -> None:
    """
    Transparent PNG of every OSM Feature (see ingest/osm.py), colored by
    kind, at the exact same plot-area position/size every other preview
    uses -- meant to be alpha-composited over any of them (in the GUI,
    not baked into a new file per base preview), not viewed standalone.

    crop_box, if given, draws a dashed rectangle outline (no fill) at
    that position/size, in the same coordinate frame as `bounds` --
    meant for showing where the [0, COURSE_SIZE_M] course crop
    currently sits within the full merged point cloud's own, larger
    frame (features here are typically uncropped in that case, so
    detail beyond the crop is visible too -- see step_ingest_osm).
    """
    fig, ax = _new_overlay_figure(bounds)

    kinds_present = set()
    for feature in features:
        _draw_osm_feature(ax, feature)
        kinds_present.add(feature.kind)

    if crop_box is not None:
        ax.add_patch(Rectangle(
            (crop_box.min_x, crop_box.min_z),
            crop_box.max_x - crop_box.min_x, crop_box.max_z - crop_box.min_z,
            fill=False, edgecolor="red", linewidth=2.0, linestyle="--", zorder=10,
        ))

    if kinds_present:
        handles = []
        for kind in sorted(kinds_present):
            fill_color, edge_color = _feature_style(kind)
            handles.append(Line2D(
                [0], [0], marker="s", linestyle="", markersize=8,
                markerfacecolor=fill_color if fill_color is not None else "none",
                markeredgecolor=edge_color, markeredgewidth=1.5, label=kind,
            ))
        if crop_box is not None:
            handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=2.0, label="course crop"))
        ax.legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.7)

    _save_transparent(fig, path)


def render_mask_preview(mask_geometry: Optional[BaseGeometry], bounds: BoundingBox, path: Path) -> None:
    """
    Plain black/white PNG for use as a PIL ImageChops.multiply mask --
    same "black conceals, white reveals" convention as a Photoshop layer
    mask: white wherever the buffered fairway/green outline (see
    ingest/osm.py's build_height_mask) covers, black everywhere else.
    Multiplying another image by this one crushes everything outside
    the mask to black while leaving everything inside it unchanged
    (white * x = x; black * x = 0).

    Rendered directly from the vector geometry, not the lower-resolution
    boolean grid find_error_hotspots actually rasterizes internally --
    the boundary stays crisp/exact here rather than blocky, since this
    is for looking at, not for the algorithm to consume.

    mask_geometry=None (no fairway/green features found) renders solid
    white -- "no mask" means "don't restrict/darken anything", matching
    ingest/osm.py's rasterize_mask same-situation convention.
    """
    fig, ax = _new_overlay_figure(bounds)
    # _new_overlay_figure defaults to transparent -- override to opaque
    # here: white margin (pass-through, so multiplying this against a
    # base image doesn't blacken its labels/border area), black data
    # area by default (masked-out), with the mask polygon(s) filled
    # white (masked-in) on top.
    fig.patch.set_alpha(1.0)
    fig.patch.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.set_facecolor("black")

    if mask_geometry is None:
        ax.axhspan(bounds.min_z, bounds.max_z, facecolor="white")
    else:
        parts = mask_geometry.geoms if hasattr(mask_geometry, "geoms") else [mask_geometry]
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            xs, zs = part.exterior.xy
            ax.fill(xs, zs, facecolor="white", edgecolor="none")
            for interior in part.interiors:
                ixs, izs = interior.xy
                ax.fill(ixs, izs, facecolor="black", edgecolor="none")

    _save(fig, path)


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
        s=3, cmap="terrain", linewidths=0,
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
    userLayers.py's NOTE on sign convention) -- if compass directions don't
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


def _make_stamp_patch(stamp: Stamp):
    """
    A Circle for round-brush stamps, or an axis-aligned Rectangle
    (side = 2*radius, centered on the stamp) for square-brush ones --
    e.g. type 72, used by the course-wide baseline-flatten stamp and
    zero-height shim, both of which are square, not circular, and were
    previously always drawn as a circle regardless of actual brush
    shape (a real, previously-latent bug: harmless-looking when the
    only square stamp was the shim, appended after preview_hex.png was
    already generated, but now visibly wrong now that the baseline
    stamp -- square, and huge, covering nearly the whole course -- is
    part of initial_stamps.json from the start).
    """
    profile = BRUSH_PROFILES.get(stamp.brush)
    if profile is not None and profile.shape == SHAPE_SQUARE:
        side = 2.0 * stamp.radius
        return Rectangle((stamp.x - stamp.radius, stamp.z - stamp.radius), side, side)
    return Circle((stamp.x, stamp.z), stamp.radius)


def _stamp_patches(stamps: Sequence[Stamp], colors: list[str]) -> PatchCollection:
    patches = [_make_stamp_patch(s) for s in stamps]
    return PatchCollection(patches, facecolor=colors, edgecolor="black", linewidths=0.3, alpha=0.35)


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
    # s is marker AREA in points^2 (matplotlib convention), not pixels or
    # diameter -- 0.4 is the exact area giving a 1px-diameter marker at
    # this module's _DPI=100 (s=2, the old value, rendered at ~2.2px).
    # linewidths=0 matters as much as s here: scatter's default stroke
    # (~1.5pt, ~2px at this DPI) is otherwise LARGER than the fill
    # itself, so without this the outline -- not the fill -- dominates
    # what actually renders, and a stroke that thin relative to its own
    # circle can rasterize as a cross/star rather than a clean ring.
    ax.scatter([s.x for s in stamps], [s.z for s in stamps], c="black", s=0.4,
               linewidths=0, zorder=3)

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
    circles = [_make_stamp_patch(s) for s in stamps]
    coll = PatchCollection(circles, edgecolor="black", linewidths=0.3, alpha=0.6)
    coll.set_array(values)
    coll.set_cmap("terrain")
    ax.add_collection(coll)
    # See render_hex_preview's comment: s=0.4 -> 1px-diameter marker at
    # _DPI=100, linewidths=0 -> no default stroke dominating that fill.
    ax.scatter([s.x for s in stamps], [s.z for s in stamps], c="black", s=0.4,
               linewidths=0, zorder=3)

    _add_colorbar(fig, coll, "fitted value (m)")
    _set_title(ax, f"Stamp values ({len(stamps)} stamps)", extra_label)
    _save(fig, path)


def render_height_preview(
    model: TerrainModel,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 2000,
    extra_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    TerrainModel's predicted height field over `bounds`.

    resolution defaults to 2000 (native, 1 px = 1 m), matching
    render_ground_lidar_preview/render_composite_preview -- previously
    400, traded off against model.render()'s real, non-trivial cost at
    high resolution with a large stamp count (unlike plain point
    binning, which is cheap regardless of resolution -- see
    render_ground_lidar_preview's own docstring). Measured directly at
    a realistic 22,573-stamp course: 884ms at 400 vs 3087ms at 2000 --
    a real ~2.2s added to every refine-terrain pass's auto-visualize
    call, not free, but small next to how long a refine-terrain pass
    itself takes, and worth it for comparing against
    preview_lidar_ground.png/preview_composite.png (both already
    native) at full, matching detail rather than a blurrier 400x400.

    vmin/vmax, if given, fix the color scale instead of the default
    auto-scale-to-this-image's-own-data-range -- pass the same values
    to render_ground_lidar_preview's own vmin/vmax so the two are
    color-comparable at a glance (a mismatch that looks dramatic in
    one could otherwise look subtle in the other, or vice versa,
    purely from each image picking its own independent range).
    """
    grid = model.render(resolution=resolution, bounds=bounds)

    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        grid, origin="lower", cmap="terrain", vmin=vmin, vmax=vmax,
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "predicted height (m)")
    _set_title(ax, f"Predicted terrain height ({resolution}x{resolution})", extra_label)
    _save(fig, path)


def render_ground_lidar_preview(
    heights: np.ndarray,
    bounds: BoundingBox,
    path: Path,
    extra_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Actual (not predicted) ground-only LIDAR height -- the saved,
    already gap-filled heightmap.npz array (see ingest/heightmap.py's
    fill_heightmap_gaps), at whatever resolution it was rasterized at
    during Ingest LAZ -- meant to be flipped back and forth against
    preview_height.png as a direct "ground truth vs. our fitted model"
    comparison.

    Deliberately switched from binning the raw point cloud directly
    (an earlier version did, via _bin_point_cloud) to using the SAME
    already-filled array everything downstream (refine-terrain,
    scatter, etc.) actually operates against -- confirmed as a real,
    reported mismatch: with the raw-point-cloud version, this preview
    kept showing "missing data" gaps under water/buildings even when
    Ingest LAZ's "Fill heightmap gaps" ran and successfully filled
    them, because this preview was silently re-deriving its own,
    never-filled view from scratch instead of showing what the
    pipeline is actually using. "Never behave as a black box" cuts
    both ways -- a diagnostic preview that shows something OTHER than
    what's really being used is its own kind of black box.

    One real tradeoff from this switch: the raw-point-cloud version
    could show genuine per-point noise a filled/gridded array can't
    (harmonic inpainting is smooth by construction) -- this preview no
    longer serves that specific purpose. If inspecting raw point noise
    directly becomes useful again later, that's a different, new
    preview to add, not a reason to revert this one.

    vmin/vmax: see render_height_preview's own docstring -- pass the
    same values to both for a directly color-comparable pair.
    """
    resolution = heights.shape[0]
    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        heights, origin="lower", cmap="terrain", vmin=vmin, vmax=vmax,
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "ground elevation (m)")
    _set_title(ax, f"Ground-only LIDAR height, actual ({resolution}x{resolution})", extra_label)
    _save(fig, path)


def render_composite_preview(
    stamps,
    bounds: BoundingBox,
    path: Path,
    resolution: int = 2000,
    brush_dir: Optional[Path] = None,
    extra_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Terrain height from composite_render.py's real-PNG-compositing
    renderer, not TerrainModel's kernel-based one -- an independent
    cross-check of "what we think the stamps do" (TerrainKernel's
    measured/interpolated 1D radial profiles) against real 2D brush
    image compositing, same stamp list, same tool semantics. Meant to
    be flipped against preview_height.png (and preview_lidar_ground.png)
    the same way those two are meant to be compared against each other.

    Manual/opt-in only (see PGA2k_gen.py's step_visualize) -- a per-
    stamp scipy.ndimage.map_coordinates sampling loop is meaningfully
    slower than TerrainModel's vectorized render(), fine for an
    explicit trigger, not for automatic regeneration after every
    refine pass.

    Requires the real brush PNG assets -- see composite_render.py's
    module docstring for where to place them. Raises a clear,
    actionable FileNotFoundError (from load_brush_image) if missing,
    rather than silently producing a blank or wrong preview.

    vmin/vmax: see render_height_preview's own docstring -- pass the
    same values here too for a 3-way color-comparable set.

    Normalizes stamps the same way the real export does (userLayers.py's
    normalize_stamp_heights: shift so the minimum resolved height lands
    at 0) before compositing -- without this, stamp.value is whatever
    raw, un-normalized height TerrainModel.render() itself produces
    (which is what preview_height.png shows, and can be well above
    275m or even negative; that shift only happens at actual export
    time), while composite_stamps_to_canvas's 16-bit conversion assumes
    values are already in the final [0, 275m] in-game range. Skipping
    this step was a real bug, confirmed directly: a real course's
    un-normalized model output reached 393.75m, and every stamp value
    above the 275m ceiling was getting silently clipped to it, pinning
    almost the entire canvas at the max -- exactly the "all blue" (or
    however it lands relative to the shared color scale) symptom.
    """
    from viz.composite_render import composite_stamps_to_canvas, canvas_to_meters, DEFAULT_BRUSH_DIR
    from course_output.userLayers import normalize_stamp_heights

    stamps = normalize_stamp_heights(stamps, bounds)
    grid = canvas_to_meters(
        composite_stamps_to_canvas(stamps, bounds, resolution, brush_dir or DEFAULT_BRUSH_DIR)
    )
    # Printed unconditionally (not just on some verbose flag) so a
    # wrong-looking preview is immediately diagnosable from the
    # console log alone: are the actual computed values wrong, or is
    # the shared vmin/vmax color scale (see render_height_preview's
    # docstring) just not matching this data's own range?
    print(f"  composite canvas: min={grid.min():.2f}m max={grid.max():.2f}m mean={grid.mean():.2f}m "
          f"(display range: vmin={vmin}, vmax={vmax})")

    fig, ax = _new_figure(bounds)
    im = ax.imshow(
        grid, origin="lower", cmap="terrain", vmin=vmin, vmax=vmax,
        extent=(bounds.min_x, bounds.max_x, bounds.min_z, bounds.max_z),
    )
    _add_colorbar(fig, im, "composited height (m)")
    _set_title(ax, f"Composited terrain height, real brush PNGs ({resolution}x{resolution})", extra_label)
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
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Predicted height vs. binned LIDAR elevation, as signed error
    (predicted - actual) on a diverging colormap centered at zero.
    Cells with no LIDAR points are left blank (NaN), not zero -- a
    missing measurement isn't the same as a confirmed-zero error.

    mask, if given (the same resolution x resolution boolean grid
    refine-terrain's --use-height-mask restricts hotspot placement
    to), reports RMS two ways: over the whole course, and over just
    the masked-in area -- without it, RMS was silently computed over
    the *entire* course regardless of whether refinement was actually
    restricted to a fraction of it, diluting the number with however
    much of the un-refined remainder still carried old error.

    Returns {"rms", "bias", "masked_rms", "masked_bias"} (the latter
    two None if no mask given or the mask is empty at this resolution)
    so callers (see PGA2k_gen.py's step_visualize) can print these
    numbers straight to the console instead of requiring the PNG to be
    reopened just to read its title.
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
    # Mean signed error, not just RMS -- RMS is magnitude-only (squares
    # away the sign), so it can't distinguish "large but balanced
    # positive/negative error" from "a systematic directional bias."
    # Confirmed real need: a genuine map-wide upward bias was only
    # visible by eyeballing color balance on the diverging colormap
    # before this existed, with no precise number to check it against.
    mean_bias = float(np.mean(finite))
    stats = {"rms": rms, "bias": mean_bias, "masked_rms": None, "masked_bias": None}
    if mask is not None:
        masked_finite = error[np.isfinite(error) & mask]
        if masked_finite.size > 0:
            masked_rms = float(np.sqrt(np.mean(np.square(masked_finite))))
            masked_mean_bias = float(np.mean(masked_finite))
            stats["masked_rms"] = masked_rms
            stats["masked_bias"] = masked_mean_bias
            title = (
                f"Height error, RMS={rms:.2f} m / bias={mean_bias:+.2f} m whole course, "
                f"RMS={masked_rms:.2f} m / bias={masked_mean_bias:+.2f} m masked area "
                f"({resolution}x{resolution})"
            )
        else:
            title = (
                f"Height error, RMS={rms:.2f} m / bias={mean_bias:+.2f} m whole course "
                "(mask empty at this resolution)"
            )
    else:
        title = f"Height error, RMS={rms:.2f} m / bias={mean_bias:+.2f} m ({resolution}x{resolution})"
    _set_title(ax, title, extra_label)
    _save(fig, path)
    return stats
