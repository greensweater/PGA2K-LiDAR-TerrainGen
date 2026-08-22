"""
course_output/object_clusters.py

Fills user-selected spline areas (Feature geometry, features.geojson)
with placedObjects2 "clusters" -- density-fill scatter stamps, not
individually placed instances (see objects.py's module docstring) --
using real per-asset density data from asset_catalog.py.

PACKING: a 2-pass scheme per asset (see _pack_circles), same overall
"coarse first, final pass mops up what's left" idea
terrain/contour_layers.py's tiered band-fill already uses for terrain
stamps, adapted from a heightmap raster to a vector shapely polygon:

  Pass 1: circles at the category's own cluster_radius (the largest a
    stamp is ever allowed to be) -- dart-thrown (random candidate,
    reject on overlap) same rejection-sampling idea as
    terrain/adaptive_refine.py's scatter_stamps ("the poisson fill
    elsewhere in here"), reimplemented directly against a shapely
    polygon rather than called as-is, since that function is tightly
    coupled to heightmap rasters/Stamp objects that don't apply here.
  Pass 2: half that radius, dart-thrown the same way as pass 1, but
    only into whatever pass 1 left uncovered (see _pack_circles for why
    this is dart-thrown too, not a deterministic grid, now that
    CONTAINMENT below can leave that remainder an irregular sliver).

CONTAINMENT: every stamp's WHOLE circle must land inside the spline --
never just its center, and never allowed to spill past the spline's
own exterior boundary, in either pass. Enforced by testing candidate
centers against geometry eroded by that stamp's own radius
(geometry.buffer(-radius)) -- the standard "does a disc of this radius
centered here fit entirely inside geometry" test -- rather than the
raw geometry. An earlier version tested plain center-inside-geometry
(pass 1) or even geometry DILATED by radius (pass 2's coverage mop-up,
deliberately allowing edge overspill for fuller coverage) -- both
replaced after real fills showed stamps overlapping well past spline
edges, which looked wrong regardless of any coverage benefit. The
trade-off this accepts: a strip within one stamp's radius of the
boundary (and the entirety of any area narrower than 2x a tier's
radius) can never be covered at all, since no valid circle of that
size can be centered close enough to reach it without crossing the
edge -- pass 2's smaller radius covers some of what pass 1 couldn't
reach this way, but perfect edge-to-edge coverage is not the goal here
(see SUBDIVIDE_RATIO if a finer mop-up tier is ever wanted).

A cluster stamp's radius is always one of these 2 tier values -- never
independently tunable per stamp beyond that, since cluster_radius is an
engine property of the category, not a per-placement choice. `ratio`
(the GUI's "Raster ratio" knob) controls how much overlap sibling
circles tolerate between each other (never the boundary, which is
never crossed regardless of ratio): minimum required center-to-center
separation is `(r1 + r2) * ratio` -- 1.0 means circles may only just
touch, <1 lets them overlap more (denser fill), >1 spaces them out
more. count per stamp is asset_catalog.cluster_count(that stamp's own
radius, asset.spacing) -- computed per-tier, since a half-radius stamp
covers a quarter the area and should get proportionally fewer instances
at the same measured density.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from shapely.geometry import Point
from shapely.ops import unary_union

from course_output.asset_catalog import ASSET_CATEGORIES, ASSET_ENTRIES, AssetCategory, AssetEntry, cluster_count
from course_output.userLayers import GRID_ORIGIN_OFFSET
from ingest.osm import Feature

# Feature.tags key the GUI's "Fill with Clusters" action sets on a
# selected spline (see PGA2k_gen_gui.py) -- pga_-prefixed, same custom-
# tag convention as objects.py's TREE_TYPE_TAG. Value is a LIST of
# {"category": int, "type": int, "ratio": float} dicts, not a single
# scalar tag, specifically so multiple assets (e.g. grass understory +
# scattered rocks) can be layered onto the same spline -- each Fill
# action appends to it rather than overwriting it. Feature.tags is a
# plain dict JSON-dumped as-is by ingest.osm.save_features, so a list
# value round-trips fine (not restricted to OSM-style string tags).
PGA_CLUSTER_FILLS_TAG = "pga_cluster_fills"

DEFAULT_RASTER_RATIO = 1.0
SUBDIVIDE_RATIO = 0.5  # pass 2's radius relative to pass 1's (max_radius * this)
MAX_CONSECUTIVE_FAILURES = 30  # a dart-throw pass gives up once this many candidates in a row are rejected
MAX_PACK_ATTEMPTS = 500  # hard cap on candidate draws per dart-throw pass, safety net for pathological geometry
_MIN_USEFUL_RADIUS = 0.1  # below this a pass is pointless (every real category's radius is well above it)
_AREA_GEOM_TYPES = ("Polygon", "MultiPolygon")

_ENTRIES_BY_KEY = {(e.category, e.type): e for e in ASSET_ENTRIES}


def _resolve_spec(spec: dict) -> Optional[tuple[AssetCategory, AssetEntry]]:
    """(category, entry) for a {"category","type",...} fill spec, or
    None if it doesn't resolve to a real clusterable catalog entry --
    e.g. a stale tag left over after asset_catalog.json changed."""
    category = ASSET_CATEGORIES.get(spec.get("category"))
    entry = _ENTRIES_BY_KEY.get((spec.get("category"), spec.get("type")))
    if category is None or category.cluster_radius is None or entry is None or not entry.spacing:
        return None
    return category, entry


def _dart_throw_pass(
    geometry, radius: float, already_placed: list[tuple[float, float, float]], ratio: float, rng: random.Random,
) -> list[tuple[float, float, float]]:
    """
    One organic, non-overlapping-by-default packing pass at a fixed
    `radius`: repeatedly draws a random candidate center inside the
    bounding box of geometry ERODED by `radius` (see module docstring's
    CONTAINMENT), accepts it if the candidate's WHOLE circle therefore
    fits inside `geometry` and it's at least
    `(radius + other_radius) * ratio` away from every circle in
    `already_placed` (this pass's own accepted circles included, as
    they accumulate). Returns newly accepted (x, z, radius) triples;
    does not mutate `already_placed`. [] immediately if nothing this
    large fits inside geometry at all (the erosion comes back empty).

    Stops once MAX_CONSECUTIVE_FAILURES candidates in a row are
    rejected (this radius no longer fits anywhere) or
    MAX_PACK_ATTEMPTS total draws are used up, whichever comes first --
    same failure-count termination idea as
    terrain/adaptive_refine.py's scatter_stamps.
    """
    eroded = geometry.buffer(-radius)
    if eroded.is_empty:
        return []

    min_x, min_z, max_x, max_z = eroded.bounds
    all_circles = list(already_placed)
    accepted: list[tuple[float, float, float]] = []
    consecutive_failures = 0
    attempts = 0
    while consecutive_failures < MAX_CONSECUTIVE_FAILURES and attempts < MAX_PACK_ATTEMPTS:
        attempts += 1
        x = rng.uniform(min_x, max_x)
        z = rng.uniform(min_z, max_z)
        if not eroded.contains(Point(x, z)):
            consecutive_failures += 1
            continue
        if any(math.hypot(x - ox, z - oz) < (radius + orad) * ratio for ox, oz, orad in all_circles):
            consecutive_failures += 1
            continue
        triple = (x, z, radius)
        all_circles.append(triple)
        accepted.append(triple)
        consecutive_failures = 0
    return accepted


def _subtract_circles(geometry, circles: list[tuple[float, float, float]]):
    """geometry with every (x,z,radius) circle's disc cut out -- the
    "still uncovered" remainder the next pass targets. `geometry`
    itself if `circles` is empty (nothing to subtract)."""
    if not circles:
        return geometry
    covered = unary_union([Point(x, z).buffer(r) for x, z, r in circles])
    return geometry.difference(covered)


def _pack_circles(geometry, max_radius: float, ratio: float, rng: random.Random) -> list[tuple[float, float, float]]:
    """
    2-pass circle packing over `geometry`'s area -- see module
    docstring. Both passes are dart-throw (not a deterministic grid):
    once containment requires a stamp's WHOLE circle to fit inside
    geometry (see CONTAINMENT), the valid region left for pass 2 after
    pass 1 -- geometry eroded by radius2 -- can be an irregular, thin
    sliver (e.g. the narrow middle of a tight spline), which a fixed-
    phase grid can miss entirely regardless of resolution. Random
    sampling instead finds it with probability proportional to its
    actual area, same as pass 1.
    """
    placed: list[tuple[float, float, float]] = []

    radius1 = max_radius
    pass1 = _dart_throw_pass(geometry, radius1, placed, ratio, rng)
    placed += pass1
    remaining = _subtract_circles(geometry, pass1)

    radius2 = max_radius * SUBDIVIDE_RATIO
    if not remaining.is_empty and radius2 >= _MIN_USEFUL_RADIUS:
        pass2 = _dart_throw_pass(remaining, radius2, placed, ratio, rng)
        placed += pass2

    return placed


def _cluster_entry(x: float, z: float, radius: float, count: int, rng: random.Random) -> dict:
    """One Value.clusters entry -- schema confirmed directly against a real placedObjects2.json."""
    return {
        "position": {"x": x - GRID_ORIGIN_OFFSET, "y": "-Infinity", "z": z - GRID_ORIGIN_OFFSET},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "seed": rng.randrange(0, 2**31 - 1),
        "count": count,
        "radius": radius,
    }


def _clusters_for_spec(geometry, category: AssetCategory, entry: AssetEntry, ratio: float, rng: random.Random) -> list[dict]:
    circles = _pack_circles(geometry, category.cluster_radius, ratio, rng)
    return [_cluster_entry(x, z, radius, cluster_count(radius, entry.spacing), rng) for x, z, radius in circles]


def fill_feature_with_clusters(feature: Feature, rng: Optional[random.Random] = None) -> list[dict]:
    """
    Cluster-entry dicts covering `feature.geometry`'s area, one packing
    run per {"category","type","ratio"} spec in
    feature.tags[PGA_CLUSTER_FILLS_TAG]. [] if the feature isn't
    tagged, its geometry has no area (e.g. a bare line), or a spec
    doesn't resolve (skipped with a printed note rather than raising --
    a stale tag shouldn't break the whole write-objects step).
    """
    if rng is None:
        rng = random.Random()
    if feature.geometry.geom_type not in _AREA_GEOM_TYPES:
        return []

    specs = feature.tags.get(PGA_CLUSTER_FILLS_TAG)
    if not specs:
        return []

    clusters: list[dict] = []
    for spec in specs:
        resolved = _resolve_spec(spec)
        if resolved is None:
            print(f"  NOTE: skipping unresolvable cluster fill spec "
                  f"category={spec.get('category')}/type={spec.get('type')} "
                  "(stale tag or asset_catalog.json no longer has it)")
            continue
        category, entry = resolved
        ratio = spec.get("ratio", DEFAULT_RASTER_RATIO)
        clusters.extend(_clusters_for_spec(feature.geometry, category, entry, ratio, rng))

    return clusters


def build_cluster_objects_v2019(features: list[Feature], rng: Optional[random.Random] = None) -> list[dict]:
    """
    placedObjects2 groups for every Feature carrying PGA_CLUSTER_FILLS_TAG
    -- one {"Key": {"category","type","theme"}, "Value": {"items": [],
    "clusters": [...]}} group per distinct (category,type,theme) seen
    across every tagged feature, merging clusters from features that
    share an asset (same per-key grouping idea as objects.py's
    build_tree_objects_v2019).
    """
    if rng is None:
        rng = random.Random()

    groups: dict[tuple[int, int, bool], dict] = {}
    for feature in features:
        if feature.geometry.geom_type not in _AREA_GEOM_TYPES:
            continue
        specs = feature.tags.get(PGA_CLUSTER_FILLS_TAG)
        if not specs:
            continue
        for spec in specs:
            resolved = _resolve_spec(spec)
            if resolved is None:
                print(f"  NOTE: skipping unresolvable cluster fill spec "
                      f"category={spec.get('category')}/type={spec.get('type')} "
                      "(stale tag or asset_catalog.json no longer has it)")
                continue
            category, entry = resolved
            ratio = spec.get("ratio", DEFAULT_RASTER_RATIO)
            new_clusters = _clusters_for_spec(feature.geometry, category, entry, ratio, rng)
            if not new_clusters:
                continue
            key = (category.id, entry.type, entry.theme)
            group = groups.setdefault(key, {
                "Key": {"category": category.id, "type": entry.type, "theme": entry.theme},
                "Value": {"items": [], "clusters": []},
            })
            group["Value"]["clusters"].extend(new_clusters)

    return list(groups.values())
