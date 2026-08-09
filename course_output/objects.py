"""
objects.py

Placed objects -- trees to start, plus building corner stakes -- for
the game's placedObjects2.json (course/CourseDescription_nodes/).

VERSIONING: PGA 2K's .course schema (this file included) is NOT one
fixed thing -- it diverges across the game's own version history
(2019 -> 2021 -> 2023 -> 2025), and this project's whole export
pipeline (userLayers.py, splines.py, objects.py) will eventually need to
be version-aware for exactly that reason. objects.py is the first
place that divergence is confirmed concretely:

  - v2019 keys a placedObjects2 group by {"category", "type", "theme"}
    -- a numeric prefab-catalog triple. This is Chad Rockey's
    TGC-Designer-Tools (OSMTGC.py's newTree/get_trees,
    tgc_definitions.py's normal_trees/skinny_trees/THEMES tables)
    exactly, since v2019 is the version his tool targeted -- see
    build_tree_objects_v2019.

  - v2021+ keys a group by {"path": "Assets/..."} -- a Unity asset
    path string -- confirmed directly from this project's own
    sort_objects_v3.py/generate_rough_border_v2.py utilities, which
    read/write real v2021+(?) course files against exactly this
    shape. v2021 also introduces object splines (density-fill scatter
    regions, not individual placed instances -- see
    generate_rough_border_v2.py's NATURE-list pattern under
    Value.splines), which v2019 has no equivalent for at all. See
    build_tree_objects_v2021.

v2023 (spline fences) and v2025 (terrain painting, spline water) are
NOT implemented here yet -- their placedObjects2 schema hasn't been
confirmed against a real extracted .course file from those versions,
so IMPLEMENTED_GAME_VERSIONS deliberately excludes them rather than
guessing. GAME_VERSIONS lists all four for UI/CLI purposes (so a
version can be selected and stored even before it's implemented);
IMPLEMENTED_GAME_VERSIONS is the subset this module can actually
build output for right now.

The position/rotation/scale shape of one placed-object *item* --
{"position": {x, y: "-Infinity", z}, "rotation": {x,y,z}, "scale":
{x,y,z}} -- is assumed shared across every version (it's the
underlying engine primitive both known schemas above already agree
on, just grouped under a different Key), and every position has
GRID_ORIGIN_OFFSET subtracted, same as splines.py/holes.py -- the
game's grid is centered on the origin ([-1000, 1000] for a 2000 m
course), not this compiler's local [0, COURSE_SIZE_M] working frame.
sort_objects_v3.py's own in_bounds check (MAP_MIN=-1000, MAP_MAX=1000)
confirms placed objects need this same shift regardless of version.

The uniform x=y=z scale-from-height rule (see build_tree_objects_v2019
/ _v2021) is applied identically in both versions, and turns out to
already be established practice for v2021+ specifically --
sort_objects_v3.py's normalize_scale() does exactly this as a cleanup
pass over an existing file. Building it in at generation time means
that cleanup pass has nothing left to fix.

Trees are parsed here directly from OSM node data (natural=tree),
deliberately NOT through ingest/osm.py's Feature/parse_osm_features
pipeline -- see that module's docstring: tree nodes are a separate
concern from the way-based vector features (splines, masks,
water/building/wood polygons) osm.py handles. This module does its
own minimal, tree-only OSM XML parse instead, reusing osm.py's
latlon_to_local for the actual coordinate transform (the same tested,
LAZ-CRS-based local frame every other feature already lands in)
rather than duplicating that math. Custom tags on a tree node (e.g. a
tree-type tag picking a specific asset) are preserved through this
parse -- see TREE_TYPE_TAG/TREE_HEIGHT_TAG -- version-independent,
since which tags exist on an OSM node has nothing to do with which
.course schema they eventually feed.

Deliberately NOT built yet (see conversation, pending v2021+ schema
confirmation beyond what generate_rough_border_v2.py already reverse-
engineered): a generalized object-spline "scatter template" builder
(e.g. a named recipe like "grass1, bush2, tree3" at various
densities, applied to any OSM area instead of the hardcoded NATURE
list), and area-based tree-species "hints" (an OSM polygon tagged
conifer/deciduous/maple/etc., visible from satellite imagery, that an
untagged tree node falling inside it inherits for asset selection --
a natural companion to TREE_TYPE_TAG's per-node override, at the area
level instead). Both are v2021+-only concepts (object splines don't
exist in v2019 at all) and are next up once v2019 is solid.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import overpy
import pyproj
from shapely.geometry import Point

from ingest.osm import Feature, latlon_to_local
from terrain.bounding_box import BoundingBox
from course_output.userLayers import GRID_ORIGIN_OFFSET

# All versions this project knows *about*; only IMPLEMENTED_GAME_VERSIONS
# can actually be built for right now (see module docstring). Kept as
# strings (not ints) since "2019" etc. are display/config labels, not
# quantities -- nothing here does arithmetic on a version.
GAME_VERSIONS = ("2019", "2021", "2023", "2025")
IMPLEMENTED_GAME_VERSIONS = ("2019", "2021")
DEFAULT_GAME_VERSION = "2019"

# Generic tree size -- used only when a tree has no more specific size
# info (see TREE_HEIGHT_TAG). OSM tree nodes carry no real size data by
# default, so every plain OSM-sourced tree gets the same stand-in
# radius/height, same idea as Chad's OSMTGC.py newTree() used.
TREE_RADIUS_M = 7.0
TREE_HEIGHT_M = 10.0

# Custom OSM tag keys this module looks for on natural=tree nodes, on
# top of the standard natural=tree itself -- lets a hand-placed OSM
# node opt into a specific asset/type (rather than a random pick from
# the general pool) and/or a real height for the scale calculation.
# These are just this project's own convention (not an OSM standard
# tag), free to rename -- see parse_osm_trees. Version-independent:
# which tree_type value maps to which theme-id (v2019) or asset path
# (v2021+) is resolved separately per version, see below.
TREE_TYPE_TAG = "pga_tree_type"
TREE_HEIGHT_TAG = "pga_tree_height"
TREE_RADIUS_TAG = "pga_tree_radius"

MIN_HEIGHT_SCALE = 0.5
MAX_HEIGHT_SCALE = 1.2

# ---------------------------------------------------------------------------
# v2019 -- Chad Rockey's TGC-Designer-Tools category/type/theme scheme,
# ported from OSMTGC.py's newTree()/tgc_image_terrain.py's get_trees(),
# tables from tgc_definitions.py (confirmed via direct fetch).
# ---------------------------------------------------------------------------

THEMES_V2019 = {
    2: "desert", 5: "boreal", 6: "tropical", 7: "countryside", 8: "harvest",
    10: "delta", 11: "rustic", 12: "swiss", 13: "steppe", 14: "autumn", 15: "highlands",
}

NORMAL_TREES_V2019 = {
    2: [0, 1, 2, 3, 9],
    5: [0, 1],
    6: [0, 1, 2, 3, 4, 5, 6, 7],
    7: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
    8: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
    11: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    12: [0, 1, 3, 4, 7, 8, 10],
    13: [0, 1, 2, 3, 4, 5, 6, 7],
    14: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    15: [0, 1, 6, 7],
}

SKINNY_TREES_V2019 = {
    2: [10, 11, 12, 13, 14, 15, 16],
    5: [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16],
    6: [8, 9, 13, 14, 15, 16, 17, 18, 19],
    7: [13, 14],
    8: [0, 1, 2],
    10: [],
    11: [13, 14, 16],
    12: [2, 5, 9],
    13: [8, 9, 10, 11, 12, 13, 14, 15],
    14: [12, 15],
    15: [2, 3, 8, 9, 10, 22],
}

SKINNY_HEIGHT_TO_RADIUS_RATIO_V2019 = 2.5  # h/r >= this -> classified "skinny", per Chad's get_trees()


def parse_osm_trees(
    osm_xml_path: Path,
    crs: pyproj.CRS,
    origin_x: float,
    origin_y: float,
    horizontal_unit_factor: float,
    bounds: Optional[BoundingBox] = None,
    printf=print,
) -> list[tuple[float, float, dict]]:
    """
    (x, z, tags) in the given local frame for every OSM node tagged
    natural=tree, optionally dropping any outside `bounds` -- the
    tree-only counterpart to osm.py's parse_osm_features, which
    deliberately skips node data entirely (see that module's and this
    one's own docstring). tags is the node's full raw OSM tag dict
    (including natural=tree itself), so callers can read
    TREE_TYPE_TAG/TREE_HEIGHT_TAG or any other custom tag without a
    second parse. Version-independent -- the OSM data itself doesn't
    change per game version, only how build_tree_objects_v2019/_v2021
    turn it into placedObjects2 entries.
    """
    xml_data = Path(osm_xml_path).read_text(encoding="utf-8")
    result = overpy.Overpass().parse_xml(xml_data)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    trees: list[tuple[float, float, dict]] = []
    skipped = 0
    for node in result.nodes:
        if node.tags.get("natural") != "tree":
            continue
        try:
            x, z = latlon_to_local(
                float(node.lat), float(node.lon), origin_x, origin_y, horizontal_unit_factor, transformer,
            )
        except (TypeError, ValueError):
            skipped += 1
            continue
        if bounds is not None and not (bounds.min_x <= x <= bounds.max_x and bounds.min_z <= z <= bounds.max_z):
            continue
        trees.append((x, z, dict(node.tags)))

    if skipped:
        printf(f"Skipped {skipped} tree node(s) with unusable coordinates")
    printf(f"Parsed {len(trees)} tree(s) from {len(result.nodes)} OSM node(s)")
    return trees


def lidar_trees_to_tagged(
    trees: list[tuple[float, float, float, float]],
) -> list[tuple[float, float, dict]]:
    """
    Convert (x, z, radius, height) tuples (see
    ingest/tree_detection.py's detect_trees_from_lidar) into the
    (x, z, tags) shape both build_tree_objects_v2019/_v2021 expect --
    encoding radius/height as TREE_RADIUS_TAG/TREE_HEIGHT_TAG, the
    same tags a hand-tagged OSM node would carry. This is what lets
    LIDAR-detected and OSM-node trees be concatenated into one list
    and built identically regardless of source -- neither builder
    needs to know or care where a given tree came from.
    """
    return [
        (x, z, {TREE_RADIUS_TAG: str(radius), TREE_HEIGHT_TAG: str(height)})
        for x, z, radius, height in trees
    ]


# OSM natural=wood polygons' leaf_type value -> a TREE_TYPE_TAG hint
# for any tree that falls inside without its own explicit type already
# (see apply_area_tree_type_hints). Deliberately small and easy to
# extend -- only what's been directly requested so far -- rather than
# guessing at a mapping for every possible leaf_type value (e.g.
# "broadleaved", "mixed") without a specific asset in mind for each.
LEAF_TYPE_TREE_HINTS = {
    "needleleaved": "pine",
}


def apply_area_tree_type_hints(
    trees: list[tuple[float, float, dict]], wood_features: list[Feature],
) -> list[tuple[float, float, dict]]:
    """
    For every tree in `trees` that does NOT already have its own
    explicit TREE_TYPE_TAG (a hand-tagged OSM node, or a LIDAR-detected
    tree some earlier call already hinted), set one from whichever
    "wood" Feature (natural=wood polygon) it falls inside, via
    LEAF_TYPE_TREE_HINTS -- e.g. a wood polygon tagged
    leaf_type=needleleaved hints every untyped tree inside it as
    "pine". A tree outside every hinted wood polygon, or inside one
    whose leaf_type has no entry in LEAF_TYPE_TREE_HINTS, is left
    unchanged. An explicit per-node tag (from a hand-placed OSM
    pga_tree_type) always wins over an area hint -- this only ever
    fills in a MISSING type, never overrides one that's already there.

    wood_features must already be cropped to the course and in the
    same local [0, COURSE_SIZE_M] frame as `trees` (same convention as
    every other geometry this module handles). Only Polygon-geometry
    "wood" features contribute a hint; anything else is ignored.

    Returns a new list (trees itself is never mutated) -- only the
    dicts for trees that actually get a new tag are copied, everything
    else is passed through as the same tuple.
    """
    hinted_polygons = []
    for f in wood_features:
        if f.kind != "wood" or f.geometry.geom_type != "Polygon":
            continue
        hint = LEAF_TYPE_TREE_HINTS.get(f.tags.get("leaf_type"))
        if hint is not None:
            hinted_polygons.append((f.geometry, hint))
    if not hinted_polygons:
        return trees

    result = []
    for x, z, tags in trees:
        if TREE_TYPE_TAG not in tags:
            point = Point(x, z)
            for polygon, hint in hinted_polygons:
                if polygon.contains(point):
                    tags = dict(tags)
                    tags[TREE_TYPE_TAG] = hint
                    break
        result.append((x, z, tags))
    return result


def _placed_item(x: float, z: float, scale: float, rotation_degrees: float = 0.0) -> dict:
    """One placed-object instance, position shifted into the game's
    origin-centered grid (see module docstring), scale.x = scale.y =
    scale.z = `scale`. Shared by every version's builder -- see module
    docstring on why this shape is assumed version-independent."""
    return {
        "position": {"x": x - GRID_ORIGIN_OFFSET, "y": "-Infinity", "z": z - GRID_ORIGIN_OFFSET},
        "rotation": {"x": 0.0, "y": rotation_degrees, "z": 0.0},
        "scale": {"x": scale, "y": scale, "z": scale},
    }


def _height_scale_lookup(
    trees: list[tuple[float, float, dict]],
) -> tuple[list[float], float, float, float]:
    """
    (heights, min_h, min_scale, scale_multiplier) for the shared
    height-driven uniform-scale rule (see module docstring) --
    scale = (h - min_h) * scale_multiplier + min_scale gives every
    tree's single x=y=z scale factor. Factored out since both
    version-specific builders use the exact same rule, just grouping
    the result differently afterward.
    """
    heights = []
    for _, _, tags in trees:
        try:
            heights.append(float(tags.get(TREE_HEIGHT_TAG, TREE_HEIGHT_M)))
        except (TypeError, ValueError):
            heights.append(TREE_HEIGHT_M)
    min_h, max_h = min(heights), max(heights)
    height_range = max_h - min_h
    if height_range > 0.01:
        return heights, min_h, MIN_HEIGHT_SCALE, (MAX_HEIGHT_SCALE - MIN_HEIGHT_SCALE) / height_range
    # All (nearly) the same height -- e.g. every tree here is still the
    # generic OSM-node placeholder size -- scale to a neutral 1.0
    # rather than an arbitrary point in the range.
    return heights, min_h, 1.0, 0.0


def build_tree_objects_v2019(
    trees: list[tuple[float, float, dict]],
    theme: Optional[int],
    tree_variety: bool = False,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    v2019 placedObjects2 groups -- Key is {"category": 0, "type": id,
    "theme": true} (see module docstring), ported from Chad's
    get_trees(). theme selects which of THEMES_V2019's tree "type" ids
    are available; an unrecognized/None theme falls back to a single
    generic type (id 0), matching Chad's own `.get(theme, [0])`
    fallback exactly. tree_variety=False (the default, matching the
    original) also forces that single generic type regardless of
    theme, and disables the skinny-tree pool entirely.

    A tree's TREE_TYPE_TAG is NOT consulted here -- v2019's catalog is
    numeric ids per theme, not asset names, so there's no meaningful
    way to map an OSM tag value onto it; that per-tree override only
    applies to v2021+ (see build_tree_objects_v2021).

    Classification as "normal" vs. "skinny" uses height/radius (h/r
    >= SKINNY_HEIGHT_TO_RADIUS_RATIO_V2019), same as Chad's original.
    radius comes from a tree's TREE_RADIUS_TAG if present (e.g. a real
    per-tree radius from LIDAR canopy detection -- see
    ingest/tree_detection.py), falling back to the flat TREE_RADIUS_M
    otherwise (matching upstream OSM node tags' own lack of a tree-
    radius concept -- there's nothing to read for a plain OSM-sourced
    tree). Scale is the same shared height-driven uniform x=y=z rule
    every version uses (see _height_scale_lookup / module docstring)
    -- Chad's original v2019 tool scaled x/z from radius independently
    of y from height; this project deliberately does not reproduce
    that (see prior conversation: scale should track height alone,
    uniformly, not stretch/squash per axis).
    """
    if rng is None:
        rng = random.Random()
    if not trees:
        return []

    normal_tree_ids = NORMAL_TREES_V2019.get(theme, [0])
    if (not tree_variety) or len(normal_tree_ids) == 0:
        normal_tree_ids = [0]
    skinny_tree_ids = SKINNY_TREES_V2019.get(theme, normal_tree_ids)
    if (not tree_variety) or len(skinny_tree_ids) == 0:
        skinny_tree_ids = []

    def _group(tree_type: int) -> dict:
        return {"Key": {"category": 0, "type": tree_type, "theme": True}, "Value": {"items": [], "clusters": []}}

    normal_groups = {t: _group(t) for t in normal_tree_ids}
    skinny_groups = {t: _group(t) for t in skinny_tree_ids}

    heights, min_h, min_scale, scale_multiplier = _height_scale_lookup(trees)

    for (x, z, tags), h in zip(trees, heights):
        scale = (h - min_h) * scale_multiplier + min_scale
        item = _placed_item(x, z, scale, rng.uniform(0, 359))
        try:
            radius = float(tags.get(TREE_RADIUS_TAG, TREE_RADIUS_M))
        except (TypeError, ValueError):
            radius = TREE_RADIUS_M
        if radius > 0 and h / radius >= SKINNY_HEIGHT_TO_RADIUS_RATIO_V2019 and skinny_groups:
            group = rng.choice(list(skinny_groups.values()))
        else:
            group = rng.choice(list(normal_groups.values()))
        group["Value"]["items"].append(item)

    return [g for g in list(normal_groups.values()) + list(skinny_groups.values()) if g["Value"]["items"]]


# ---------------------------------------------------------------------------
# v2021+ -- real Unity asset-path scheme, confirmed against this
# project's own sort_objects_v3.py/generate_rough_border_v2.py.
# ---------------------------------------------------------------------------


def _placed_object_group_v2021(asset_path: str) -> dict:
    """One placedObjects2.json group entry, empty of items -- v2021+'s
    Key.path/Value.{items,clusters,splines} schema (see module
    docstring), not v2019's category/type/theme scheme."""
    return {"Key": {"path": asset_path}, "Value": {"items": [], "clusters": [], "splines": []}}


def build_tree_objects_v2021(
    trees: list[tuple[float, float, dict]],
    tree_asset_paths: list[str],
    tree_type_asset_paths: Optional[dict[str, str]] = None,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """
    v2021+ placedObjects2 groups -- Key is {"path": asset_path} (see
    module docstring), one group per asset path actually used.

    tree_asset_paths is the general pool a tree is randomly assigned
    from when it has no more specific pick -- REQUIRED and must be
    non-empty (or tree_type_asset_paths must cover every tree that
    needs one); there's no generic fallback path to invent (see module
    docstring), so this raises ValueError if neither is usable.

    tree_type_asset_paths, if given, maps a tree node's TREE_TYPE_TAG
    value to a specific asset path, overriding the random pool pick
    for just that tree -- e.g. a hand-tagged pga_tree_type=oak node
    always gets whatever path tree_type_asset_paths["oak"] is, while
    untagged trees still draw from tree_asset_paths.

    Scale is the same shared height-driven uniform x=y=z rule every
    version uses (see _height_scale_lookup / module docstring).
    """
    if rng is None:
        rng = random.Random()
    if not trees:
        return []
    tree_type_asset_paths = tree_type_asset_paths or {}
    if not tree_asset_paths and not tree_type_asset_paths:
        raise ValueError(
            "build_tree_objects_v2021 needs at least one real asset path -- pass tree_asset_paths "
            "(a general pool) and/or tree_type_asset_paths (per pga_tree_type tag). There's no "
            "built-in catalog to fall back to; v2021+ placed objects are keyed by real Unity asset "
            "paths, not a numeric id this project could guess at."
        )

    heights, min_h, min_scale, scale_multiplier = _height_scale_lookup(trees)

    groups: dict[str, dict] = {}

    def _group_for(path: str) -> dict:
        if path not in groups:
            groups[path] = _placed_object_group_v2021(path)
        return groups[path]

    for (x, z, tags), h in zip(trees, heights):
        tree_type = tags.get(TREE_TYPE_TAG)
        if tree_type is not None and tree_type in tree_type_asset_paths:
            path = tree_type_asset_paths[tree_type]
        elif tree_asset_paths:
            path = rng.choice(tree_asset_paths)
        else:
            # Tagged with a tree_type that isn't in
            # tree_type_asset_paths, and no general pool to fall back
            # to -- skip rather than guess at a path.
            continue
        scale = (h - min_h) * scale_multiplier + min_scale
        item = _placed_item(x, z, scale, rng.uniform(0, 359))
        _group_for(path)["Value"]["items"].append(item)

    return [g for g in groups.values() if g["Value"]["items"]]


def build_building_stake_objects_v2021(features: list[Feature], stake_asset_path: str) -> list[dict]:
    """
    One stake instance at every exterior vertex of every "building"
    Feature (see ingest/osm.py) -- a single placedObjects2.json group
    (one asset path, so one group covers every building). No rotation
    (a stake has no meaningful facing), scale left at 1.0.

    v2021+ ONLY: this needs an arbitrary asset path, which only
    v2021+'s Key.path scheme supports -- v2019's numeric category/type
    catalog has no known "stake" (or generic decorative object) id, so
    there's currently no way to do this for a v2019 target at all.

    Only Polygon-geometry building features contribute vertices --
    matches how splines.py's feature_to_spline already only handles
    building as an area/fairway-like shape, never a bare line.
    """
    if not stake_asset_path:
        raise ValueError("build_building_stake_objects_v2021 needs a real stake asset path -- see module docstring.")

    group = _placed_object_group_v2021(stake_asset_path)
    for f in features:
        if f.kind != "building" or f.geometry.geom_type != "Polygon":
            continue
        for x, z in f.geometry.exterior.coords[:-1]:  # [:-1] drops the closing repeat of the first point
            group["Value"]["items"].append(_placed_item(x, z, scale=1.0))

    return [group] if group["Value"]["items"] else []


def object_counts(objects: list[dict]) -> list[tuple[str, int, int, int]]:
    """
    (label, item_count, cluster_count, spline_count) per group -- same
    enrichment sort_objects_v3.py computes for its summary table,
    reused here so the GUI's object selector (mirroring the existing
    Splines tab's feature list) can show the same counts without
    duplicating that logic. label is Key["path"] for v2021+ groups, or
    a "category/type" string built from Key["category"]/Key["type"]
    for v2019 groups (which have no "path") -- so this works for
    either version's placedObjects2.json without the caller needing to
    know which one it's looking at.
    """
    counts = []
    for g in objects:
        key = g.get("Key", {})
        if "path" in key:
            label = key["path"]
        else:
            label = f"category={key.get('category')}/type={key.get('type')}"
        value = g.get("Value", {})
        counts.append((
            label,
            len(value.get("items", [])),
            len(value.get("clusters", [])),
            len(value.get("splines", [])),
        ))
    return counts


def save_object_list(trees: list[tuple[float, float, dict]], path: Path) -> None:
    """
    Write the intermediate, VERSION-AGNOSTIC object list (see module
    docstring's version-dispatch discussion) -- the combined result of
    OSM natural=tree parsing and, optionally, LIDAR canopy detection,
    already unified into one (x, z, tags) list (see
    lidar_trees_to_tagged) -- to object_list.json, BEFORE either
    build_tree_objects_v2019 or _v2021 ever runs.

    Deliberately not GeoJSON: these are scalar points with a flat tag
    dict, not polygon/line geometry, so GeoJSON's FeatureCollection
    generality buys nothing here -- a plain JSON list matches this
    project's own convention for non-vector-feature data (see
    initial_stamps.json's Stamp list).

    Kept separate from a specific game_version on purpose: this is the
    one-time-expensive step (OSM parse, LIDAR watershed detection);
    switching game_version later only needs to re-run the (cheap)
    build_tree_objects_v20XX formatting step against this same file,
    not redo detection from scratch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"x": x, "z": z, "tags": tags} for x, z, tags in trees]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def load_object_list(path: Path) -> list[tuple[float, float, dict]]:
    with Path(path).open(encoding="utf-8") as fh:
        entries = json.load(fh)
    return [(e["x"], e["z"], e["tags"]) for e in entries]


def save_placed_objects(objects: list[dict], path: Path) -> None:
    """Write placedObjects2.json -- a plain JSON array, matching the
    same one-key-per-file convention as surfaceSplines.json/holes.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(objects, fh, indent=2)


def load_placed_objects(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
