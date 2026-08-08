"""
water.py

Builds userLayers.json's "water" entries (see userLayers.py's
_BLANK_USER_LAYERS_SCHEMA) from OSM water Features -- one flat,
rotated, horizontal plane object per water body, sized/positioned to
best-fit its polygon, elevated to the low point of the terrain stamps
that already cover that area.

Schema confirmed directly (not derived/guessed) against a real water
entry from userLayers.json:
    surfaceCategory: 9        same surface id as splines.py's
                               FEATURES_TO_SURFACES["water"]
    position: {x, y, z}       y is the water LEVEL (elevation), not a
                               sentinel like userLayers.py's terrain
                               stamps' "-Infinity"
    rotation: {x:0, y, z:0}   y is the plane's yaw, degrees
    _orientation: 0.0         always 0.0 in the reference sample --
                               unlike userLayers.py's terrain stamps,
                               which set this to the actual rotation;
                               reproduced as given here, not assumed
    scale: {x, y:1.0, z}      x/z are the plane's full width/depth in
                               meters; y stays 1.0, matching terrain
                               stamps' own scale.y=1.0 convention
    type: 72                  fixed, confirmed from the reference sample
    value: same as position.y -- redundant, both carry the water
                               level; reproduced as given
    holeId: -1
    options: {flowOrientation: 0.0, flowSpeed: 1.0} -- reproduced as
                               given; no per-water-body customization
                               implemented yet (still/uniform flow)
    radius: 0.0
    orientation: 0.0          NOT the same value as rotation.y in the
                               reference sample (unlike terrain stamps'
                               orientation/_orientation, which
                               duplicate each other) -- reproduced as
                               given rather than assumed to mirror
                               rotation.y

Geometry: the base plane asset is assumed to be a flat 1x1 m unit quad
centered on its own origin (the standard Unity primitive-plane/quad
convention), so scale.x/scale.z directly ARE the real-world width/depth
in meters, not a half-extent or other multiplier. This is an
assumption, not something independently confirmed against the actual
mesh -- if a water plane renders at roughly half (or double) the real
pond size once viewed in-game, that's the tell this assumption is
wrong, and the fix is a single multiplier here, not a rethink of the
whole approach. Likewise the mapping from the fitted rectangle's two
edge lengths to scale.x vs. scale.z (and how that pairs with
rotation.y) is this module's own convention, not independently
verified against which local axis the plane mesh's width actually
runs along -- worth a visual check the first time a real water body
renders noticeably rotated from the true pond outline.

Fit shape: shapely's minimum_rotated_rectangle -- the smallest-area
rectangle (at any rotation, not just axis-aligned) that fully contains
the water polygon. Matches "a square plane, placed and scaled" -- one
rotated rectangle, not a closer per-vertex fit -- while wasting much
less area than a plain axis-aligned bounding box would on any non-
axis-aligned or elongated pond.

Water level: the minimum stamp.value among every already-placed
terrain stamp whose (x, z) center falls within the water polygon --
literally "the low point of stamps in the water area", not a query
against the raw LIDAR heightmap. This means water objects must be
built AFTER terrain generation/refinement has produced its stamp list
(see PGA2k_gen.py's step_output_terrain), not before. A water body
with no stamp centers inside it (rare, but possible for a very small
pond sitting between coarse hex-grid stamps) has no low point to
compute from and is skipped -- logged by the caller, not silently
dropped here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from shapely.geometry import Point

from ingest.osm import Feature
from course_output.userLayers import GRID_ORIGIN_OFFSET
from terrain.stamp import Stamp

WATER_SURFACE_CATEGORY = 9  # same id as splines.py's FEATURES_TO_SURFACES["water"]
WATER_TYPE = 72
_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def _low_point_among_stamps(polygon, stamps: Sequence[Stamp]) -> Optional[float]:
    """
    Minimum .value among stamps whose (x, z) center falls within
    `polygon` -- both must already be in the same local
    [0, COURSE_SIZE_M] frame (neither is GRID_ORIGIN_OFFSET-shifted
    yet). None if no stamp center falls inside at all.
    """
    values = [s.value for s in stamps if polygon.contains(Point(s.x, s.z))]
    return min(values) if values else None


def _fit_rectangle(polygon) -> Optional[tuple[float, float, float, float, float]]:
    """
    (center_x, center_z, width, depth, rotation_degrees) for the
    minimum-area rotated rectangle enclosing `polygon` -- see module
    docstring's "Fit shape". None if the geometry is too degenerate to
    fit one (e.g. collapses to a point or line).
    """
    rect = polygon.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return None
    coords = list(rect.exterior.coords[:-1])
    if len(coords) != 4:
        return None

    cx = sum(c[0] for c in coords) / 4.0
    cz = sum(c[1] for c in coords) / 4.0
    edge1 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    edge2 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    width = math.hypot(*edge1)
    depth = math.hypot(*edge2)
    rotation_deg = math.degrees(math.atan2(edge1[1], edge1[0])) % 360.0
    return cx, cz, width, depth, rotation_deg


def build_water_objects(
    water_features: Sequence[Feature], stamps: Sequence[Stamp], printf=print,
) -> list[dict]:
    """
    One water entry per "water" Feature with Polygon geometry (see
    module docstring) -- water_features must already be cropped to the
    course (see PGA2k_gen.py's _crop_features_to_course) and in the
    same local [0, COURSE_SIZE_M] frame as `stamps`; GRID_ORIGIN_OFFSET
    is applied here, at the point of writing, same as every other
    writer in this project.

    Skips, with a printed reason (not silently and not an error), any
    water feature that isn't a Polygon, whose minimum_rotated_rectangle
    can't be computed, or that contains no stamp centers to determine
    a water level from.
    """
    entries = []
    skipped = 0
    for f in water_features:
        if f.kind != "water":
            continue
        if f.geometry.geom_type != "Polygon":
            printf(f"  Skipping non-polygon water feature (geom_type={f.geometry.geom_type}) -- "
                   "only filled water bodies get a water object, not centerlines.")
            skipped += 1
            continue

        fit = _fit_rectangle(f.geometry)
        if fit is None:
            printf("  Skipping a water feature -- couldn't fit a rectangle to its geometry "
                   "(likely degenerate/too small).")
            skipped += 1
            continue
        cx, cz, width, depth, rotation_deg = fit

        level = _low_point_among_stamps(f.geometry, stamps)
        if level is None:
            printf(f"  Skipping a water feature near ({cx:.0f}, {cz:.0f}) -- no terrain stamp centers "
                   "fall inside it, so there's no low point to set its water level from.")
            skipped += 1
            continue

        entries.append({
            "surfaceCategory": WATER_SURFACE_CATEGORY,
            "position": {
                "x": _round(cx - GRID_ORIGIN_OFFSET),
                "y": _round(level),
                "z": _round(cz - GRID_ORIGIN_OFFSET),
            },
            "rotation": {"x": 0.0, "y": _round(rotation_deg), "z": 0.0},
            "_orientation": 0.0,
            "scale": {"x": _round(width), "y": 1.0, "z": _round(depth)},
            "type": WATER_TYPE,
            "value": _round(level),
            "holeId": -1,
            "options": {"flowOrientation": 0.0, "flowSpeed": 1.0},
            "radius": 0.0,
            "orientation": 0.0,
        })

    printf(f"  {len(entries)} water object(s) built" + (f", {skipped} skipped" if skipped else ""))
    return entries
