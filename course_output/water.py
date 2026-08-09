"""
water.py

Builds userLayers.json's "water" entries (see userLayers.py's
_BLANK_USER_LAYERS_SCHEMA) from OSM water Features -- one flat,
rotated, horizontal plane object per water body, sized/positioned to
best-fit its polygon, elevated to a robust high point of the terrain
that already covers that area.

Schema confirmed directly (not derived/guessed) against a real water
entry from userLayers.json:
    surfaceCategory: 9        same surface id as splines.py's
                               FEATURES_TO_SURFACES["water"]
    position: {x, y, z}       y is the water LEVEL (elevation) -- and,
                               critically, an EXPLICIT absolute number,
                               not a sentinel like userLayers.py's
                               terrain stamps' "-Infinity". Terrain
                               stamps can get away with "-Infinity"
                               because the game computes their actual
                               rendered height from `value` plus every
                               other stamp's blend at load time; a
                               water plane has no such computation --
                               position.y IS the elevation it renders
                               at, verbatim. That means it has to
                               already be fully, correctly shifted by
                               whatever normalize_stamp_heights applied
                               to the rest of the course -- see
                               "Water level" below for why this was
                               getting missed.
    rotation: {x:0, y, z:0}   y is the plane's yaw, degrees
    _orientation: 0.0         always 0.0 in the reference sample --
                               unlike userLayers.py's terrain stamps,
                               which set this to the actual rotation;
                               reproduced as given here, not assumed
    scale: {x, y:1.0, z}      see "Geometry" below -- NOT simply the
                               fitted rectangle's width/depth in meters
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

Geometry: an earlier version of this module assumed the base plane
asset was a flat 1x1 m unit quad, so scale.x/scale.z could just BE the
real-world width/depth in meters directly. Confirmed wrong in practice
(water planes rendering "way bigger" than their actual pond) -- the
far more likely culprit is Unity's own built-in Plane primitive, which
is NOT 1x1: it's a well-known 10x10 unit mesh by default. Dividing by
WATER_MESH_BASE_SIZE_M (10.0) below corrects for that. This is still
an assumption, not something independently re-confirmed against the
actual mesh -- if a water plane is now consistently off by some OTHER
constant factor, that factor is this one constant to adjust, not a
rethink of the whole approach. A small WATER_RECT_MARGIN is also
applied on top, so the plane errs toward "same size or slightly
larger" than the fitted rectangle rather than an exact, zero-tolerance
fit that could leave a sliver of real pond peeking out at the corners.

Likewise the mapping from the fitted rectangle's two edge lengths to
scale.x vs. scale.z (and how that pairs with rotation.y) is this
module's own convention, not independently verified against which
local axis the plane mesh's width actually runs along -- worth a
visual check the first time a real water body renders noticeably
rotated from the true pond outline.

Fit shape: shapely's minimum_rotated_rectangle -- the smallest-area
rectangle (at any rotation, not just axis-aligned) that fully contains
the water polygon. Matches "a square plane, placed and scaled" -- one
rotated rectangle, not a closer per-vertex fit -- while wasting much
less area than a plain axis-aligned bounding box would on any non-
axis-aligned or elongated pond.

Water level: a robust HIGH point of the terrain within the water
polygon -- specifically the 90th percentile (i.e. clipping off the top
10% as likely outliers, per direct instruction) of each terrain stamp
CENTER's fully-resolved elevation inside the polygon. Two things had
to change from an earlier version to get here:

  1. It read stamp.value directly. That's wrong: normalize_stamp_
     heights (see userLayers.py) does NOT rewrite existing stamps'
     .value fields to shift them -- it APPENDS one new course-wide
     raise-tool stamp whose effect only shows up once the WHOLE stamp
     list is evaluated together through a TerrainModel. Every other
     stamp's own .value is still its pre-shift number. Reading .value
     directly therefore silently used pre-shift elevations for nearly
     every stamp -- which is exactly why some water bodies lined up
     fine (courses needing little or no shift) while others were off
     by "hundreds of feet" (courses needing a large one, e.g. real-
     world elevations far from a local zero). Fixed by evaluating a
     TerrainModel built from the (already-normalized) `stamps` this
     module receives, at each candidate center, instead of trusting
     any individual stamp's own stored value.

  2. It took the single MINIMUM instead of a robust high point. Made
     sense when only hand-identified error hotspots got stamps; wrong
     now that "scatter" mode (see terrain/adaptive_refine.py) blankets
     the whole course, including areas in and around water bodies
     themselves, where real LIDAR returns are frequently noisy (partial
     penetration, surface bounce) -- the single lowest reading inside a
     water polygon is disproportionately likely to be a bad-data
     artifact, not the real basin floor. A pond's surrounding bank is
     generally a better-behaved signal than its noisy interior, so this
     now looks at the HIGH end instead -- with the top 10% clipped off
     to guard against the opposite failure mode (one stray too-tall
     artifact dragging the plane up).

This means water objects must be built AFTER terrain generation/
refinement AND normalize_stamp_heights have both already run (see
PGA2k_gen.py's step_output_terrain) -- not before, and not against the
pre-normalization stamp list. A water body with no stamp centers inside
it (rare, but possible for a very small pond sitting between coarse
hex-grid stamps) has no level to compute from and is skipped -- logged
by the caller, not silently dropped here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Point

from ingest.osm import Feature
from course_output.userLayers import GRID_ORIGIN_OFFSET
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel

WATER_SURFACE_CATEGORY = 9  # same id as splines.py's FEATURES_TO_SURFACES["water"]
WATER_TYPE = 72
_DECIMALS = 3

# Unity's built-in Plane primitive is a 10x10 unit mesh by default, not
# 1x1 -- see module docstring's "Geometry". scale.x/scale.z = desired
# real-world meters / this constant.
WATER_MESH_BASE_SIZE_M = 10.0

# Small safety margin on the fitted rectangle so the water plane errs
# toward "same size or slightly larger" than the real pond outline,
# not an exact zero-tolerance fit -- see module docstring.
WATER_RECT_MARGIN = 1.02

# Clip the top this-much of evaluated heights within a water polygon
# before taking the max -- see module docstring's "Water level" #2.
WATER_LEVEL_PERCENTILE = 90.0


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def _water_level_from_stamps(
    polygon, stamps: Sequence[Stamp], model: TerrainModel,
) -> Optional[float]:
    """
    WATER_LEVEL_PERCENTILE-th percentile of `model`'s fully-resolved
    elevation at every stamp center falling within `polygon` -- see
    module docstring's "Water level". `stamps`/`polygon` must already
    be in the same local [0, COURSE_SIZE_M] frame (neither is
    GRID_ORIGIN_OFFSET-shifted yet); `model` must be built from the
    ALREADY-height-normalized stamp list (see
    userLayers.py's normalize_stamp_heights) -- evaluating it, rather
    than reading any stamp's own .value, is what actually captures
    that normalization shift (see module docstring's "Water level" #1).
    None if no stamp center falls inside the polygon at all.
    """
    points = [(s.x, s.z) for s in stamps if polygon.contains(Point(s.x, s.z))]
    if not points:
        return None
    evaluated = model.evaluate_many(np.asarray(points, dtype=np.float64))
    return float(np.percentile(evaluated, WATER_LEVEL_PERCENTILE))


def _fit_rectangle(polygon) -> Optional[tuple[float, float, float, float, float]]:
    """
    (center_x, center_z, width, depth, rotation_degrees) for the
    minimum-area rotated rectangle enclosing `polygon`, expanded by
    WATER_RECT_MARGIN -- see module docstring's "Fit shape"/"Geometry".
    None if the geometry is too degenerate to fit one (e.g. collapses
    to a point or line). width/depth here are the real-world meters
    the plane should SPAN -- converting that into the mesh's own scale
    units (see WATER_MESH_BASE_SIZE_M) happens in build_water_objects,
    not here, so this function's output stays independently checkable
    against the source geometry in real-world units.
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
    width = math.hypot(*edge1) * WATER_RECT_MARGIN
    depth = math.hypot(*edge2) * WATER_RECT_MARGIN
    rotation_deg = math.degrees(math.atan2(edge1[1], edge1[0])) % 360.0
    return cx, cz, width, depth, rotation_deg


def build_water_objects(
    water_features: Sequence[Feature], stamps: Sequence[Stamp], printf=print,
) -> list[dict]:
    """
    One water entry per "water" Feature with Polygon geometry (see
    module docstring) -- water_features must already be cropped to the
    course (see PGA2k_gen.py's _crop_features_to_course); `stamps` must
    be the FULLY height-normalized list (see userLayers.py's
    normalize_stamp_heights -- this must run BEFORE calling this
    function, not after) and in the same local [0, COURSE_SIZE_M] frame
    as water_features' geometry. GRID_ORIGIN_OFFSET is applied here, at
    the point of writing, same as every other writer in this project.

    Skips, with a printed reason (not silently and not an error), any
    water feature that isn't a Polygon, whose minimum_rotated_rectangle
    can't be computed, or that contains no stamp centers to determine
    a water level from.
    """
    model = TerrainModel(stamps)
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
        cx, cz, width_m, depth_m, rotation_deg = fit

        level = _water_level_from_stamps(f.geometry, stamps, model)
        if level is None:
            printf(f"  Skipping a water feature near ({cx:.0f}, {cz:.0f}) -- no terrain stamp centers "
                   "fall inside it, so there's no level to set its water elevation from.")
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
            "scale": {
                "x": _round(width_m / WATER_MESH_BASE_SIZE_M),
                "y": 1.0,
                "z": _round(depth_m / WATER_MESH_BASE_SIZE_M),
            },
            "type": WATER_TYPE,
            "value": _round(level),
            "holeId": -1,
            "options": {"flowOrientation": 0.0, "flowSpeed": 1.0},
            "radius": 0.0,
            "orientation": 0.0,
        })

    printf(f"  {len(entries)} water object(s) built" + (f", {skipped} skipped" if skipped else ""))
    return entries
