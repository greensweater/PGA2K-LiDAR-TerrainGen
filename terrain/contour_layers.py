"""
terrain/contour_layers.py

Alternative initial base-layer generator to hexgrid.py's flat hex
lattice: traces elevation-band contours of the real ground heightmap
(via skimage.measure.find_contours -- marching squares) and places a
chain of type-10/54 flatten stamps along each ring, spaced by the
ring's own local curvature (tight spacing on sharp bends, wide spacing
on straight runs) rather than a fixed pitch. Type 10/54's smooth,
no-flat-plateau falloff means adjacent bands blend into each other
instead of showing terrace edges where one band's stamp meets the
next.

Each ring stamp's target value is exact, not fitted: a contour at
level L is (by construction) where the heightmap equals L, so
value=L needs no separate height-fitting pass afterward, unlike
hexgrid.py's lattice (whose stamps get their value from a later
fit_stamp_heights() call against nearby points).

WHY EDGE-ONLY TRACING NEEDS A GAP-FILL PASS:

A contour ring is topologically blind to its own interior -- it's an
isoline, the boundary of a level set, by definition. On sloped ground
this doesn't matter: for a fixed elevation step, consecutive bands sit
close together in map-space, so their stamps' radii naturally overlap
and the slope gets filled as a side effect of tracing its bands. But a
genuinely flat area (a green, a flat fairway landing zone, a pond
surface) is a single ring no matter how large its interior is -- the
next contour for that region, if it exists at all, doesn't reappear
until you're off the flat area entirely. No amount of tuning the
elevation step fixes this; it's what contour lines are, not a bug in
how densely they're spaced.

_gap_fill_pass closes this the same way find_error_hotspots/
find_deep_holes already do elsewhere in this project: rasterize what's
actually been covered by a placed stamp, run a distance transform over
the uncovered area, and the point of maximum distance from any covered
cell is both the most-interior uncovered point AND the natural radius
for a stamp that fills it -- repeat, claiming only the placed circle,
until nothing uncovered remains. This runs regardless of how tightly
the elevation bands are spaced, so it's a real guarantee, not a
tuning-dependent hope.

COVERAGE GUARANTEE, more precisely: ring placement always uses
spacing == radius (each new stamp sits exactly one radius past the
last, reaching back to the previous stamp's own center -- the same
"reach to neighbor center" principle hexgrid.py's HEX_STAMP_RADIUS_M
uses, just walked along a curve instead of a lattice). Combined with
gap_fill_pass running to full saturation (loop until the distance
transform's max is <= 0, not a failure-count heuristic), every point
in the valid area ends up within SOME placed stamp's own radius of its
center -- if it weren't, gap_fill_pass would have placed a stamp there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import measure

from ingest.heightmap import downsample_heightmap
from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

DEFAULT_BAND_SPACING_M = 5.0  # Delta -- GUI/CLI tweakable, deliberately fixed rather than adaptive
DEFAULT_BRUSH = 10  # smooth cosine-like falloff, no flat plateau -- bands blend, no terrace edges
DEFAULT_MIN_RING_RADIUS_M = 5.0
DEFAULT_MAX_RING_RADIUS_M = 50.0
DEFAULT_CURVATURE_WINDOW_M = 10.0  # arc-length window used to estimate local curvature
DEFAULT_CURVATURE_CONTRAST_GAMMA = 2.0  # same role as adaptive_refine.py's variation_contrast_gamma
DEFAULT_GAP_FILL_BRUSH = 10
DEFAULT_MIN_GAP_RADIUS_M = 5.0
DEFAULT_GAP_FILL_RESOLUTION = 400  # coverage-mask grid for the gap-fill pass; see generate_contour_layers
DEFAULT_MAX_GAP_FILL_ITERATIONS = 20000  # safety cap, not expected to bind in practice


def _contour_levels(heights: np.ndarray, spacing: float) -> np.ndarray:
    """
    Elevation levels to trace, spaced `spacing` m apart from just above
    the heightmap's own minimum to just below its max -- skimage's
    find_contours requires a level strictly inside the data's range to
    return anything, so the two endpoints (where a "ring" would just be
    the single lowest/highest point, not a real feature) are skipped.
    """
    lo = float(np.nanmin(heights))
    hi = float(np.nanmax(heights))
    if hi - lo < spacing:
        return np.array([])
    return np.arange(lo + spacing, hi, spacing)


def _pixel_to_world(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], bounds: BoundingBox,
) -> tuple[np.ndarray, np.ndarray]:
    """
    skimage.measure.find_contours returns (row, col) in the heightmap's
    own index space (subpixel, since marching squares interpolates
    between cells) -- map those to world (x, z) the same way every
    other cell-center convention in this project does.
    """
    n_rows, n_cols = shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    x = bounds.min_x + cols * cell_x
    z = bounds.min_z + rows * cell_z
    return x, z


def _ring_curvature(x: np.ndarray, z: np.ndarray, window_pts: int) -> np.ndarray:
    """
    Discrete curvature estimate at each vertex of a polyline: the
    turning angle (radians) between the incoming and outgoing chord
    over a `window_pts`-vertex lookback/lookahead, divided by the
    chord's own arc length -- standard "angle change per unit length"
    curvature, not a true osculating-circle fit (unnecessary precision
    for a radius-SIZING signal; this never touches what gets written).

    window_pts is a point-count, not a distance -- the caller derives
    it from DEFAULT_CURVATURE_WINDOW_M using the ring's actual point
    spacing (skimage's contour output isn't evenly spaced by arc
    length in general, but is dense/near-uniform in practice for
    marching squares on a regular grid, so this is a fair approximation
    without needing to resample the ring first).
    """
    n = len(x)
    w = max(1, window_pts)
    curvature = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - w)
        i1 = min(n - 1, i + w)
        if i1 - i0 < 2:
            continue
        v1 = np.array([x[i] - x[i0], z[i] - z[i0]])
        v2 = np.array([x[i1] - x[i], z[i1] - z[i]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        arc = n1 + n2
        curvature[i] = angle / arc if arc > 1e-9 else 0.0
    return curvature


def _curvature_to_radius(
    curvature: np.ndarray, min_radius: float, max_radius: float, contrast_gamma: float,
) -> np.ndarray:
    """
    Map per-vertex curvature to a target stamp radius in [min_radius,
    max_radius] -- inverse: low curvature (straight run) -> large
    radius (few, wide stamps), high curvature (sharp bend) -> small
    radius (many, tight stamps). Same percentile-normalize + power-law
    contrast shape as adaptive_refine.py's _variation_to_radius_grid,
    for the same reason (a single noisy sharp corner shouldn't compress
    the whole ring's usable range onto one vertex).
    """
    if curvature.size == 0:
        return np.array([])
    lo, hi = np.percentile(curvature, [5.0, 95.0])
    if hi - lo < 1e-12:
        return np.full_like(curvature, max_radius)
    normalized = np.clip((curvature - lo) / (hi - lo), 0.0, 1.0)  # 0=straight, 1=sharp
    contrasted = normalized ** contrast_gamma
    return max_radius - contrasted * (max_radius - min_radius)


def _place_stamps_along_ring(
    x: np.ndarray, z: np.ndarray, level: float, brush: int,
    min_radius: float, max_radius: float, curvature_window_m: float, contrast_gamma: float,
) -> list[Stamp]:
    """
    Walk one contour ring (already in world coordinates) and place a
    flatten stamp every time accumulated arc length since the last
    placement reaches the CURRENT local target radius -- i.e. spacing
    == radius, reaching back exactly to the previous stamp's own
    center, mirroring hexgrid.py's own neighbor-spacing convention
    (see module docstring's coverage-guarantee note) just walked along
    a curve with a locally-varying radius instead of a fixed lattice.

    Ring is treated as closed if find_contours returned matching first
    and last points (the normal case for a ring that doesn't touch the
    heightmap's edge); open rings (which do touch the edge) are walked
    start-to-end only, no wraparound.
    """
    n = len(x)
    if n < 2:
        return []

    seg_dx = np.diff(x)
    seg_dz = np.diff(z)
    seg_len = np.hypot(seg_dx, seg_dz)
    mean_spacing = float(np.mean(seg_len)) if seg_len.size else 1.0
    window_pts = max(1, int(round(curvature_window_m / max(mean_spacing, 1e-6))))

    curvature = _ring_curvature(x, z, window_pts)
    radius_at_vertex = _curvature_to_radius(curvature, min_radius, max_radius, contrast_gamma)

    stamps: list[Stamp] = []
    accumulated = 0.0
    stamps.append(Stamp(x=float(x[0]), z=float(z[0]), radius=float(radius_at_vertex[0]),
                         value=level, brush=brush, tool=TOOL_FLATTEN))
    next_target = radius_at_vertex[0]

    for i in range(1, n):
        accumulated += seg_len[i - 1]
        if accumulated >= next_target:
            stamps.append(Stamp(x=float(x[i]), z=float(z[i]), radius=float(radius_at_vertex[i]),
                                 value=level, brush=brush, tool=TOOL_FLATTEN))
            accumulated = 0.0
            next_target = radius_at_vertex[i]

    return stamps


def _rasterize_coverage(
    stamps: list[Stamp], bounds: BoundingBox, resolution: int, covered: np.ndarray,
) -> None:
    """
    Mark `covered` (a boolean (resolution, resolution) grid, modified
    in place) True wherever a placed stamp's own radius reaches --
    each stamp only touches its own bounding-box window of the grid,
    same windowed approach as adaptive_refine.py's hotspot n_cells
    computation, rather than a dense per-stamp full-grid distance
    calculation.
    """
    cell_x = (bounds.max_x - bounds.min_x) / resolution
    cell_z = (bounds.max_z - bounds.min_z) / resolution
    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0

    for s in stamps:
        col_min = max(0, int((s.x - s.radius - bounds.min_x) / cell_x))
        col_max = min(resolution, int((s.x + s.radius - bounds.min_x) / cell_x) + 1)
        row_min = max(0, int((s.z - s.radius - bounds.min_z) / cell_z))
        row_max = min(resolution, int((s.z + s.radius - bounds.min_z) / cell_z) + 1)
        if col_min >= col_max or row_min >= row_max:
            continue
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        within = np.hypot(xx - s.x, zz - s.z) <= s.radius
        covered[row_min:row_max, col_min:col_max] |= within


def _gap_fill_pass(
    heights: np.ndarray, bounds: BoundingBox, resolution: int, covered: np.ndarray,
    brush: int, min_radius: float, max_radius: float,
    max_iterations: int = DEFAULT_MAX_GAP_FILL_ITERATIONS,
) -> list[Stamp]:
    """
    Close whatever ring tracing structurally can't reach (see module
    docstring) -- same distance-transform "largest inscribed circle"
    loop as adaptive_refine.py's find_error_hotspots/find_deep_holes,
    applied to a plain coverage mask instead of a signed-error mask
    (there's no over/under distinction here, just covered/uncovered).

    Runs to genuine saturation -- the loop only exits when the
    distance transform's peak is <= 0, i.e. nothing valid remains
    uncovered -- not a failure-count heuristic, so this is a real
    guarantee regardless of how the elevation bands above were spaced.
    """
    actual = downsample_heightmap(heights, bounds, resolution)
    valid = np.isfinite(actual)

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    cell_x = (bounds.max_x - bounds.min_x) / resolution
    cell_z = (bounds.max_z - bounds.min_z) / resolution
    sampling = (cell_z, cell_x)

    stamps: list[Stamp] = []
    for _ in range(max_iterations):
        uncovered = valid & ~covered
        if not uncovered.any():
            break
        dist = ndimage.distance_transform_edt(uncovered, sampling=sampling)
        peak = dist.max(initial=0.0)
        if peak <= 0.0:
            break
        row, col = np.unravel_index(np.argmax(dist), dist.shape)
        radius = float(np.clip(peak, min_radius, max_radius))
        cx, cz = float(x_centers[col]), float(z_centers[row])

        row_min = max(0, int((cz - radius - bounds.min_z) / cell_z))
        row_max = min(resolution, int((cz + radius - bounds.min_z) / cell_z) + 1)
        col_min = max(0, int((cx - radius - bounds.min_x) / cell_x))
        col_max = min(resolution, int((cx + radius - bounds.min_x) / cell_x) + 1)
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        within = np.hypot(xx - cx, zz - cz) <= radius
        sub_actual = actual[row_min:row_max, col_min:col_max]
        sub_valid = valid[row_min:row_max, col_min:col_max] & within
        if not sub_valid.any():
            covered[row, col] = True  # can't fit a value here; claim just this cell and move on
            continue
        target = float(np.mean(sub_actual[sub_valid]))

        stamps.append(Stamp(x=cx, z=cz, radius=radius, value=target, brush=brush, tool=TOOL_FLATTEN))
        covered[row_min:row_max, col_min:col_max][within] = True

    return stamps


def generate_contour_layers(
    heights: np.ndarray,
    bounds: BoundingBox,
    band_spacing_m: float = DEFAULT_BAND_SPACING_M,
    brush: int = DEFAULT_BRUSH,
    min_ring_radius: float = DEFAULT_MIN_RING_RADIUS_M,
    max_ring_radius: float = DEFAULT_MAX_RING_RADIUS_M,
    curvature_window_m: float = DEFAULT_CURVATURE_WINDOW_M,
    curvature_contrast_gamma: float = DEFAULT_CURVATURE_CONTRAST_GAMMA,
    gap_fill_brush: int = DEFAULT_GAP_FILL_BRUSH,
    min_gap_radius: float = DEFAULT_MIN_GAP_RADIUS_M,
    max_gap_radius: Optional[float] = None,
    gap_fill_resolution: int = DEFAULT_GAP_FILL_RESOLUTION,
) -> list[Stamp]:
    """
    Generate an organic base layer by tracing elevation-band contours
    of `heights` (see module docstring) and gap-filling whatever's left
    uncovered. Returns stamps ordered lowest-elevation ring first,
    highest last, with every gap-fill stamp appended at the very end --
    under this project's sequential/pull-toward-value compositing
    model (see terrain_model.py), later stamps take precedence in any
    overlap, so this ordering paints coarse low bands first and lets
    higher, more specific bands (and finally, gap-fills) refine on top
    -- the same intent as a stacked-plywood relief model, low layers
    first.

    Every ring stamp's value is the exact contour level it was traced
    from -- no separate height-fitting pass needed for those (unlike
    hexgrid.py's lattice). Only gap-fill stamps need a fitted value
    (the local heightmap mean over their own footprint, same as
    adaptive_refine.py's scatter_stamps), since they don't come from a
    contour level at all.

    max_gap_radius defaults to max_ring_radius when not given.
    gap_fill_resolution controls the coverage-mask grid the gap-fill
    pass runs against -- finer catches smaller gaps (a green missed
    between two widely-spaced bands) at higher cost; it does not need
    to match whatever resolution later refinement/scatter passes use.
    """
    if max_gap_radius is None:
        max_gap_radius = max_ring_radius

    levels = _contour_levels(heights, band_spacing_m)

    stamps: list[Stamp] = []
    for level in levels:
        rings = measure.find_contours(heights, level=level)
        for ring in rings:
            rows, cols = ring[:, 0], ring[:, 1]
            x, z = _pixel_to_world(rows, cols, heights.shape, bounds)
            stamps.extend(_place_stamps_along_ring(
                x, z, level, brush, min_ring_radius, max_ring_radius,
                curvature_window_m, curvature_contrast_gamma,
            ))

    covered = np.zeros((gap_fill_resolution, gap_fill_resolution), dtype=bool)
    _rasterize_coverage(stamps, bounds, gap_fill_resolution, covered)

    gap_stamps = _gap_fill_pass(
        heights, bounds, gap_fill_resolution, covered,
        gap_fill_brush, min_gap_radius, max_gap_radius,
    )
    stamps.extend(gap_stamps)

    return stamps
