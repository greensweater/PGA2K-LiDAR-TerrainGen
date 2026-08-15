"""
terrain/cart_paths.py

Cart path terrain-flattening stamps: for each cart path spline, builds
a "pearl necklace" of points along the path (the spline itself is
never modified), then places directionally-rotated type 15 ("blurred
square") Stamp objects connecting consecutive pearls, valued at the
average real ground height under each stamp's own footprint.

Mirrors the course's existing generate_streams.py pattern (same pearl-
necklace idea, same directional-rotation-from-consecutive-points idea)
but built around this pipeline's own primitives throughout: real
terrain.stamp.Stamp objects (not raw JSON dicts), Shapely geometry for
the spline/pearl work (not hand-rolled bezier+accumulator loops), and
the real heightmap array for height sampling (not a JSON lookup
against already-placed stamps one step removed from LIDAR).

############################################################
ASSUMPTIONS, not confirmed facts -- flagged here and at each usage
site below. Confirm/correct all three before trusting real output:

1. CART_PATH_SURFACE_VALUES -- the surface value(s) identifying a cart
   path spline. Unknown; must be set by the caller.

2. "not filled" is read as `isFilled == False` on the spline itself,
   by analogy with generate_streams.py's own isClosed/state pattern
   and the isFilled field its own polygon_to_object_spline() writes.

3. Stamp.scale_x/scale_z's real-world meaning for a SQUARE (non-
   circular) brush. Every circular-brush usage elsewhere in this
   pipeline treats Stamp.scale_x/scale_z as a true center-to-edge
   distance (checked directly: `dist_from_center <= radius`), and
   this module follows that same convention for type 15's plateau-
   scale calculation (scale_x == scale_z here, isotropic). If
   writer.py converts scale_x/scale_z to the game's own scale.x/
   scale.z JSON fields differently for square brushes specifically
   (e.g. a factor-of-2 diameter/radius mismatch), the resulting path
   will come out at 2x or 0.5x the intended width -- verify in-game
   and report back if so.
############################################################
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import LineString

from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

CART_PATH_STAMP_TYPE = 15
CART_PATH_WIDTH_M = 1.7

# Type 15 ("blurred square") real measured geometry, at its native
# 512px texture resolution -- given directly, not guessed:
#   - weight is 100% (full plateau) from 110px in from each border
#   - weight falls to 0 exactly 50px in from each border
# So the plateau (100% zone) is a (512 - 2*110) = 292px square, and the
# full nonzero ("active") footprint is (512 - 2*50) = 412px square.
TYPE15_TEXTURE_PX = 512
TYPE15_PLATEAU_MARGIN_PX = 110
TYPE15_FALLOFF_ZERO_PX = 50
TYPE15_PLATEAU_PX = TYPE15_TEXTURE_PX - 2 * TYPE15_PLATEAU_MARGIN_PX  # 292
TYPE15_ACTIVE_PX = TYPE15_TEXTURE_PX - 2 * TYPE15_FALLOFF_ZERO_PX  # 412

# Stamp radius such that the PLATEAU -- not the full footprint -- comes
# out to exactly CART_PATH_WIDTH_M. See assumption 3 above regarding
# whether this radius is a center-to-edge distance (this module's own
# assumption, consistent with every circular brush elsewhere in this
# pipeline) or something else for a square brush specifically.
#   radius = width_m * texture_px / plateau_px / 2
#          = 1.7 * 512 / 292 / 2 = 1.4904...m
# (the /2 converts the FULL plateau width into a center-to-edge
# distance, matching Stamp.radius's own convention elsewhere)
CART_PATH_STAMP_RADIUS = CART_PATH_WIDTH_M * TYPE15_TEXTURE_PX / TYPE15_PLATEAU_PX / 2.0

# Pearl spacing along the path: spaced CLOSER than the plateau's own
# along-path extent (CART_PATH_WIDTH_M, for this square brush) so
# consecutive stamps' plateaus genuinely OVERLAP lengthwise, not just
# meet edge-to-edge -- the same "let the stamp itself do the work of
# overlapping" reasoning used throughout the rest of this pipeline,
# applied to the along-path direction specifically. 0.85 is a starting
# point, not a measured constant -- tune if the path reads seamy or
# over-stamped once you see it in-game.
CART_PATH_OVERLAP_FRACTION = 0.85
CART_PATH_SPACING_M = CART_PATH_OVERLAP_FRACTION * CART_PATH_WIDTH_M

# Radius (m) to average real ground height over at each stamp -- half
# the stamp's own active (nonzero) footprint, i.e. the area this stamp
# actually influences, not an arbitrarily chosen window. Same texture-
# fraction logic as CART_PATH_STAMP_RADIUS above, using the ACTIVE
# (not plateau) pixel margin.
CART_PATH_HEIGHT_AVG_RADIUS_M = (TYPE15_ACTIVE_PX / TYPE15_TEXTURE_PX) * CART_PATH_STAMP_RADIUS


@dataclass
class CartPathSpline:
    """
    One cart path spline, already reduced to a Shapely LineString --
    built by the caller from whatever this project's own spline source
    format is (see cart_paths_from_bezier_waypoints below for the one
    concrete conversion this module provides, matching the bezier
    waypoint format generate_streams.py already works with).
    """
    line: LineString
    source_id: Optional[str] = None  # whatever identifies this spline in its source, for diagnostics only


def _cubic_bezier(p0, p1, p2, p3, t):
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    x = mt2 * mt * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t2 * t * p3[0]
    z = mt2 * mt * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t2 * t * p3[1]
    return (x, z)


def bezier_waypoints_to_linestring(waypoints: list[dict]) -> Optional[LineString]:
    """
    Converts this project's own bezier waypoint JSON structure (each
    waypoint has 'waypoint'/'pointOne'/'pointTwo' 2D control points,
    same shape generate_streams.py's sample_spline_dense already
    parses) into a densely-sampled Shapely LineString. Densification
    step count scales with each segment's own approximate control-
    polygon length (same heuristic as the reference script), not a
    fixed sample count, so long segments stay smooth and short ones
    aren't over-sampled for no benefit.

    Returns None if there are fewer than 2 waypoints, or every segment
    is missing a required control point.
    """
    if len(waypoints) < 2:
        return None

    def extract(p):
        if not isinstance(p, dict):
            return None
        try:
            return (float(p["x"]), float(p.get("z", p.get("y"))))
        except (KeyError, TypeError, ValueError):
            return None

    points: list[tuple[float, float]] = []

    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        p0 = extract(a.get("waypoint"))
        p1 = extract(a.get("pointTwo"))
        p2 = extract(b.get("pointOne"))
        p3 = extract(b.get("waypoint"))
        if not all((p0, p1, p2, p3)):
            continue

        approx_len = math.dist(p0, p1) + math.dist(p1, p2) + math.dist(p2, p3)
        steps = max(8, int(approx_len))
        for s in range(steps):
            t = s / float(steps)
            points.append(_cubic_bezier(p0, p1, p2, p3, t))
        if i == len(waypoints) - 2:
            points.append(p3)

    if len(points) < 2:
        return None
    return LineString(points)


def generate_cart_path_stamps(
    splines: list[CartPathSpline],
    heights: np.ndarray,
    bounds: BoundingBox,
    spacing_m: float = CART_PATH_SPACING_M,
    stamp_radius: float = CART_PATH_STAMP_RADIUS,
    height_avg_radius_m: float = CART_PATH_HEIGHT_AVG_RADIUS_M,
    brush: int = CART_PATH_STAMP_TYPE,
) -> list[Stamp]:
    """
    Builds directionally-rotated type 15 Stamp objects along every
    given spline's own pearl necklace (Shapely's own `line.interpolate`
    at fixed distance steps -- the spline geometry itself is never
    touched, this only ever reads it). One stamp per interval between
    consecutive pearls (n_pearls - 1 stamps per spline), positioned at
    the EARLIER pearl and rotated toward the next one, matching
    generate_streams.py's own convention for how consecutive stamps
    connect along a path's direction of travel.

    Each stamp's value is the mean of `heights` within
    height_avg_radius_m of its own position -- real LIDAR-derived
    ground height (this function has direct access to the actual
    heightmap array, not an approximation against already-placed
    stamps). Pearls whose footprint has no finite heightmap data at
    all (a genuine LIDAR gap) are skipped entirely -- no stamp placed,
    rather than guessing a value.
    """
    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    x_centers = bounds.min_x + (np.arange(n_cols) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(n_rows) + 0.5) * cell_z

    def average_height(x: float, z: float, radius_m: float) -> Optional[float]:
        col_min = max(0, int((x - radius_m - bounds.min_x) / cell_x))
        col_max = min(n_cols, int((x + radius_m - bounds.min_x) / cell_x) + 1)
        row_min = max(0, int((z - radius_m - bounds.min_z) / cell_z))
        row_max = min(n_rows, int((z + radius_m - bounds.min_z) / cell_z) + 1)
        if col_min >= col_max or row_min >= row_max:
            return None
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        within = np.hypot(xx - x, zz - z) <= radius_m
        sub_heights = heights[row_min:row_max, col_min:col_max]
        valid = within & np.isfinite(sub_heights)
        if not valid.any():
            return None
        return float(np.mean(sub_heights[valid]))

    stamps: list[Stamp] = []
    skipped_no_height = 0

    for spline in splines:
        length = spline.line.length
        if length <= 0:
            continue

        n_pearls = max(2, int(length / spacing_m) + 1)
        distances = np.linspace(0.0, length, n_pearls)
        pearls = [spline.line.interpolate(d) for d in distances]

        for i in range(len(pearls) - 1):
            cur, nxt = pearls[i], pearls[i + 1]
            dx, dz = nxt.x - cur.x, nxt.y - cur.y
            mag = math.hypot(dx, dz)
            if mag <= 1e-9:
                continue
            ndx, ndz = dx / mag, dz / mag
            rot_y = math.degrees(math.atan2(ndx, ndz))

            avg_h = average_height(cur.x, cur.y, height_avg_radius_m)
            if avg_h is None:
                skipped_no_height += 1
                continue

            stamps.append(Stamp(
                x=float(cur.x), z=float(cur.y), scale_x=float(stamp_radius), scale_z=float(stamp_radius),
                value=avg_h, brush=brush, rotation=float(rot_y), tool=TOOL_FLATTEN,
            ))

    if skipped_no_height:
        print(f"  {skipped_no_height} cart path pearls had no finite heightmap data "
              "nearby and were skipped entirely (no stamp placed) -- likely a real LIDAR gap")

    return stamps
