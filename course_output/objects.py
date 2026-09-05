"""
course_output/objects.py

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

Scale is uniform x=y=z in both versions (established practice for v2021+
already -- sort_objects_v3.py's normalize_scale() does exactly this as a
cleanup pass; building it in at generation time leaves that pass nothing
to fix). Its MAGNITUDE is CALIBRATED where possible: _tree_scale uses
detected_canopy_height / the prefab's measured native height (from
asset_catalog.json's native_height_m -- keyed by asset path for v2021+,
by per-theme type id for v2019, see native_tree_height_v2019), so a tree
detected at its real height renders at ~1.0. Prefabs with no measured
native height (un-measured, non-rustic v2019 themes, OSM placeholder
trees) fall back to the original course-relative height remap
(_height_scale_lookup). Chad's original also stretched crown WIDTH
(scale.x/z) from LiDAR canopy radius; that stays deliberately dropped --
native_canopy_radius_m is captured but unused.

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

Area-based tree-species "hints" ARE built now, both versions: an OSM
natural=wood polygon tagged leaf_type=needleleaved/broadleaved hints
every untyped tree inside it (LEAF_TYPE_TREE_HINTS /
apply_area_tree_type_hints, applied in step_generate_trees), and both
writers then honor the resulting TREE_TYPE_TAG -- v2021+ via
tree_type_asset_paths, v2019 via a per-theme species-bucket table
(tree_themes.json / TreeThemeSpecies / build_tree_objects_v2019's
`species` arg).

Deliberately still NOT built (see conversation, pending v2021+ schema
confirmation beyond what generate_rough_border_v2.py already reverse-
engineered): a generalized object-spline "scatter template" builder
(e.g. a named recipe like "grass1, bush2, tree3" at various
densities, applied to any OSM area instead of the hardcoded NATURE
list) -- v2021+-only (object splines don't exist in v2019 at all).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import overpy
import pyproj
from shapely.geometry import Point

from ingest.osm import Feature, latlon_to_local
from terrain.bounding_box import BoundingBox
from terrain.cart_paths import CART_PATH_WIDTH_M
from course_output.asset_catalog import ASSET_ENTRIES, NATIVE_TREE_HEIGHT_BY_PATH, native_tree_height_v2019
from course_output.userLayers import GRID_ORIGIN_OFFSET
# Re-exported for existing callers (PGA2k_gen.py, PGA2k_gen_gui.py) -- the
# version registry itself now lives in game_versions.py, see its docstring.
from course_output.game_versions import (  # noqa: F401
    DEFAULT_GAME_VERSION, GAME_VERSIONS, IMPLEMENTED_GAME_VERSIONS, VersionSchema, schema_for,
)

# Generic tree size -- used only when a tree has no more specific size
# info (see TREE_HEIGHT_TAG). OSM tree nodes carry no real size data by
# default, so every plain OSM-sourced tree gets the same stand-in
# radius/height, same idea as Chad's OSMTGC.py newTree() used.
TREE_RADIUS_M = 7.0
TREE_HEIGHT_M = 10.0

# Cart-path keepout is intentionally just enough to clear the TRUNK,
# not the canopy -- CART_PATH_WIDTH_M/2 (0.85m) + this clearance, ~1.85m
# total, not the tree's full TREE_RADIUS_M/TREE_RADIUS_TAG canopy
# radius. A canopy-sized keepout (~8m) was flagging far more trees as
# "on the path" than are actually visibly overlapping it, and pushing
# them proportionally farther -- more likely to land on a second,
# nearby path in a connected network. See move_trees_off_cartpaths.
TREE_CARTPATH_CLEARANCE_M = 2.0

# Tag set on a tree that's been detected sitting on a cart path when
# move_trees_off_cartpaths runs in debug_mark_only mode (see that
# function and PGA2k_gen.py's --mark-cartpath-trees) -- left in place
# rather than relocated, so build_tree_objects_v2019 can swap it for an
# oversized, obviously-not-a-tree marker instead, letting you eyeball
# in-game exactly which trees the detector is flagging before trusting
# it to actually move anything.
CARTPATH_DEBUG_MARKER_TAG = "pga_cartpath_debug_marker"
# theme=false, category/type values below confirmed directly by the
# user from a real placedObjects2.json (not guessed): an arbitrary
# decorative prop, not specifically meaningful, just distinct and easy
# to spot -- freely tunable to whatever's easiest to pick out in-game.
# v2019 only -- v2021+ has no known equivalent generic marker id (see
# build_tree_objects_v2021), so debug-tagged trees there are currently
# just built as ordinary trees.
CARTPATH_DEBUG_MARKER_CATEGORY_V2019 = 13
CARTPATH_DEBUG_MARKER_TYPE_V2019 = 34
CARTPATH_DEBUG_MARKER_SCALE = 6.0  # deliberately oversized (real trees scale ~0.5-1.2) so it's unmistakable in-game

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

# Absolute clamp on a placed tree's final uniform scale, applied on BOTH
# the calibrated (detected_height / native_height) and the fallback
# (course-relative remap) path -- see _tree_scale. Wider than the
# fallback curve's own 0.5..1.2 endpoints so genuine calibration isn't
# squeezed back onto the old band, but still a backstop against a wild
# LiDAR detection or a mistyped native_height_m. Taste knobs -- retune
# freely.
TREE_SCALE_CLAMP_MIN = 0.25
TREE_SCALE_CLAMP_MAX = 1.75

# Matches userLayers.py/water.py's own _DECIMALS convention -- every
# value written into placedObjects2.json is rounded to millimeter
# precision, plenty for this project's purposes.
_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)

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


# ---------------------------------------------------------------------------
# Per-theme tree "species" buckets -- lets build_tree_objects_v2019 route a
# tree by its TREE_TYPE_TAG (e.g. "pine" from a leaf_type=needleleaved wood
# hint, see LEAF_TYPE_TREE_HINTS / apply_area_tree_type_hints) into a
# curated subset of the theme's type ids, instead of drawing blindly from
# the whole NORMAL_TREES_V2019 / SKINNY_TREES_V2019 pool. Data lives in
# tree_themes.json (see its own _note); a theme not covered there just
# falls back to the original random-pool behavior.
# ---------------------------------------------------------------------------

TREE_THEMES_JSON = Path(__file__).resolve().parent / "tree_themes.json"


@dataclass(frozen=True, slots=True)
class TreeThemeSpecies:
    """One theme's species-bucket table from tree_themes.json. `species`
    maps a bucket name (matching a TREE_TYPE_TAG value, e.g. "pine",
    "deciduous") to that bucket's list of v2019 category-0 type ids;
    `default_species` is the bucket a tree with no TREE_TYPE_TAG uses."""
    default_species: str
    species: dict[str, tuple[int, ...]]

    def ids_for(self, tree_type: Optional[str]) -> tuple[int, ...]:
        """Type ids a tree tagged `tree_type` (or None/unknown) should draw
        from -- its own bucket if known, else the default bucket, else an
        empty tuple (caller decides the last-ditch fallback)."""
        if tree_type is not None and tree_type in self.species:
            return self.species[tree_type]
        return self.species.get(self.default_species, ())

    def is_known(self, tree_type: Optional[str]) -> bool:
        return tree_type is not None and tree_type in self.species


def _load_tree_themes() -> dict[int, TreeThemeSpecies]:
    if not TREE_THEMES_JSON.exists():
        return {}
    data = json.loads(TREE_THEMES_JSON.read_text(encoding="utf-8"))
    out: dict[int, TreeThemeSpecies] = {}
    for theme_id, block in data.get("themes", {}).items():
        out[int(theme_id)] = TreeThemeSpecies(
            default_species=block["default_species"],
            species={name: tuple(ids) for name, ids in block["species"].items()},
        )
    return out


TREE_THEMES: dict[int, TreeThemeSpecies] = _load_tree_themes()


def load_tree_theme_species(
    theme: Optional[int], config_path: "str | Path | None" = None,
) -> Optional[TreeThemeSpecies]:
    """The TreeThemeSpecies for `theme`, or None if `theme` is None or has
    no block in tree_themes.json (build_tree_objects_v2019 then keeps its
    original random-pool behavior). config_path overrides the default
    tree_themes.json location (e.g. a per-project override); its schema is
    identical."""
    if theme is None:
        return None
    if config_path is None:
        return TREE_THEMES.get(theme)
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    block = data.get("themes", {}).get(str(theme))
    if block is None:
        return None
    return TreeThemeSpecies(
        default_species=block["default_species"],
        species={name: tuple(ids) for name, ids in block["species"].items()},
    )


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
# (see apply_area_tree_type_hints). The hint value is a species-bucket
# name -- it must match a bucket key in tree_themes.json for the active
# theme (v2019) or a key in tree_type_asset_paths (v2021+) to actually
# change which asset the tree gets; an unmatched hint just falls through
# to the default bucket / random pool. "mixed" is deliberately left
# unmapped (ambiguous) -- those trees stay untyped and use the default.
LEAF_TYPE_TREE_HINTS = {
    "needleleaved": "pine",
    "broadleaved": "deciduous",
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


def _nearest_cartpath_line(point: Point, lines: list) -> tuple[object, float]:
    """(line, distance) for whichever of `lines` is closest to `point` --
    factored out of move_trees_off_cartpaths since it's called fresh on
    every relocation attempt (the nearest path can change once a tree
    moves)."""
    best_line, best_dist = lines[0], lines[0].distance(point)
    for line in lines[1:]:
        d = line.distance(point)
        if d < best_dist:
            best_line, best_dist = line, d
    return best_line, best_dist


def _perpendicular_at(line, point: Point) -> Optional[tuple[Point, tuple[float, float]]]:
    """(nearest_point_on_line, unit_perpendicular) at `point`'s
    projection onto `line` -- the perpendicular is line's tangent there
    rotated 90 degrees, found via a small-delta interpolate() step
    either side of the projection (same tangent technique
    terrain/cart_paths.py's generate_cart_path_stamps uses for its
    pearl-to-pearl direction). Returns None if the tangent is
    degenerate (zero-length line, or projection pinned to one end of a
    near-zero-length segment)."""
    proj = line.project(point)
    eps = max(1e-3, min(0.5, line.length / 4.0))
    a = line.interpolate(max(0.0, proj - eps))
    b = line.interpolate(min(line.length, proj + eps))
    tx, tz = b.x - a.x, b.y - a.y
    mag = math.hypot(tx, tz)
    if mag <= 1e-9:
        return None
    tx, tz = tx / mag, tz / mag
    nearest = line.interpolate(proj)
    return nearest, (-tz, tx)


def _min_clearance(x: float, z: float, lines: list) -> float:
    """Distance from (x, z) to the nearest of `lines` -- the "is this
    candidate actually clear of EVERY cart path, not just the one it
    was just pushed off of" check move_trees_off_cartpaths needs at
    every retry (a course's cart paths are usually a connected network
    -- parallel out-and-back paths, loops around tee/green complexes --
    so a push that only checks the path it's dodging can easily land
    the tree on a different, nearby one)."""
    point = Point(x, z)
    return min(line.distance(point) for line in lines)


def move_trees_off_cartpaths(
    trees: list[tuple[float, float, dict]],
    cartpath_lines: list,
    hole_features: list[Feature],
    max_attempts: int = 8,
    debug_mark_only: bool = False,
    printf=print,
) -> list[tuple[float, float, dict]]:
    """
    Push any tree sitting on top of a cart path off to the side, clear
    of the path's real rendered width (terrain.cart_paths.CART_PATH_WIDTH_M)
    plus TREE_CARTPATH_CLEARANCE_M -- ~1.85m total, deliberately just
    enough to clear the TRUNK, not the canopy (a tree's TREE_RADIUS_M/
    TREE_RADIUS_TAG canopy radius is NOT part of this -- that would
    flag/move far more trees than are actually visibly overlapping the
    path). Movement is always perpendicular to the path at the tree's
    nearest point on it (never along the path).

    debug_mark_only=True skips relocation entirely: a tree detected
    within keepout of a path is left exactly where it is and tagged
    with CARTPATH_DEBUG_MARKER_TAG instead, so build_tree_objects_v2019
    can swap it for an oversized, obvious marker object in place of a
    real tree -- lets you eyeball in-game exactly which trees are being
    detected before trusting this function to actually move anything.

    Each retry computes BOTH perpendicular candidates and checks each
    one's clearance against EVERY line in cartpath_lines, not just the
    one being dodged -- a candidate that only clears the nearest path
    can still land squarely on a different, nearby one. Between two
    candidates that both fully clear every path, picks whichever is
    farther from the nearest hole centerline (hole_features) -- pushes
    a tree toward rough rather than the fairway/hole line; with no
    hole_features, falls back to whichever side the tree was already
    leaning toward, so the choice stays deterministic. If only one
    candidate clears, that one wins regardless of hole preference. If
    NEITHER clears (a tight spot -- e.g. two paths closer together than
    2x the keepout distance), the less-bad (larger-clearance) one is
    taken and the loop retries from there.

    Retries up to max_attempts times, tracking the best (max-clearance)
    position seen across all attempts rather than just the last one --
    a tree can legitimately oscillate between two nearby paths without
    strictly improving every single attempt, and returning the actual
    best-found position avoids the final result regressing below an
    earlier attempt. If a tree still hasn't fully cleared every path
    once attempts are exhausted, its best-found (least-bad) position is
    used anyway, and printf (same injectable-print convention
    parse_osm_trees uses) gets one summary line -- not per-tree spam --
    reporting how many trees that happened to.

    cartpath_lines is a plain list of Shapely geometries (LineString or
    Polygon -- a Polygon, e.g. an explicit-area path or cart parking,
    is reduced to its exterior ring internally) -- source-agnostic, so
    callers can pass OSM Feature.geometry, hand-drawn spline geometry,
    or any mix, without this function caring where it came from. KNOWN
    LIMITATION: a Polygon is only measured as distance-to-boundary, not
    distance-to-interior -- a tree deep inside a Polygon much WIDER
    than keepout (e.g. a large cart parking lot) could be several
    meters from every edge and go undetected. Harmless for the common
    case (a real OSM cart path is a LineString, and even an explicit-
    area one is normally only a couple meters wide, well under
    keepout), but worth knowing if this is ever pointed at a genuinely
    wide paved area.

    cartpath_lines/hole_features must already be cropped to the course
    and in the same local [0, COURSE_SIZE_M] frame as `trees`, same
    convention as apply_area_tree_type_hints. Returns a new list
    (trees itself is never mutated); tags is always passed through as
    the same dict reference (this function only ever changes x/z).
    """
    lines = []
    for geom in cartpath_lines:
        line = geom.exterior if geom.geom_type == "Polygon" else geom
        if line.length > 0:
            lines.append(line)
    if not lines:
        return trees

    hole_lines = [f.geometry for f in hole_features if f.geometry.length > 0]

    keepout = CART_PATH_WIDTH_M / 2.0 + TREE_CARTPATH_CLEARANCE_M

    result = []
    n_flagged = 0
    n_relocated = 0
    n_still_stuck = 0
    for x, z, tags in trees:
        best_x, best_z, best_clearance = x, z, _min_clearance(x, z, lines)
        if best_clearance >= keepout:
            result.append((x, z, tags))
            continue

        n_flagged += 1
        if debug_mark_only:
            tags = dict(tags)
            tags[CARTPATH_DEBUG_MARKER_TAG] = True
            result.append((x, z, tags))
            continue

        cur_x, cur_z = x, z
        for _ in range(max_attempts):
            point = Point(cur_x, cur_z)
            line, dist = _nearest_cartpath_line(point, lines)
            if dist >= keepout:
                break
            perp = _perpendicular_at(line, point)
            if perp is None:
                break
            nearest, (px, pz) = perp
            push = keepout + 0.05
            cand_a = (nearest.x + px * push, nearest.y + pz * push)
            cand_b = (nearest.x - px * push, nearest.y - pz * push)
            clearance_a = _min_clearance(cand_a[0], cand_a[1], lines)
            clearance_b = _min_clearance(cand_b[0], cand_b[1], lines)
            a_clear, b_clear = clearance_a >= keepout, clearance_b >= keepout

            if a_clear and b_clear:
                if hole_lines:
                    dist_a = min(hl.distance(Point(*cand_a)) for hl in hole_lines)
                    dist_b = min(hl.distance(Point(*cand_b)) for hl in hole_lines)
                    pick_a = dist_a >= dist_b
                else:
                    ox, oz = point.x - nearest.x, point.y - nearest.y
                    pick_a = (ox * px + oz * pz) >= 0
            elif a_clear:
                pick_a = True
            elif b_clear:
                pick_a = False
            else:
                pick_a = clearance_a >= clearance_b

            cur_x, cur_z = cand_a if pick_a else cand_b
            cur_clearance = clearance_a if pick_a else clearance_b
            if cur_clearance > best_clearance:
                best_x, best_z, best_clearance = cur_x, cur_z, cur_clearance
            if best_clearance >= keepout:
                break

        n_relocated += 1
        if best_clearance < keepout:
            n_still_stuck += 1
        result.append((best_x, best_z, tags))

    if debug_mark_only:
        if n_flagged:
            printf(f"  DEBUG: tagged {n_flagged} tree(s) detected on a cart path for marker "
                   "replacement -- positions left unchanged (course_output/objects.py's "
                   "CARTPATH_DEBUG_MARKER_TAG)")
    elif n_still_stuck:
        printf(f"  WARNING: {n_still_stuck} of {n_relocated} relocated tree(s) could not be fully "
               f"cleared of a cart path after {max_attempts} attempt(s) each -- likely squeezed "
               "between two paths closer together than the keepout distance")

    return result


def _placed_item(
    x: float, z: float, scale: float, rotation_degrees: float = 0.0, y: "float | str" = "-Infinity",
) -> dict:
    """One placed-object instance, position shifted into the game's
    origin-centered grid (see module docstring), scale.x = scale.y =
    scale.z = `scale`. Shared by every version's builder -- see module
    docstring on why this shape is assumed version-independent.

    `y` defaults to the literal "-Infinity" -- the game drops the object
    onto the terrain (with each prefab's own built-in vertical "bleed")
    on load. Pass a real number for an object that carries a designed
    elevation instead (a deck/railing built at a set height; see
    course_output/collections.py's terrain-relative collection members)."""
    return {
        "position": {
            "x": _round(x - GRID_ORIGIN_OFFSET),
            "y": y if isinstance(y, str) else _round(y),
            "z": _round(z - GRID_ORIGIN_OFFSET),
        },
        "rotation": {"x": 0.0, "y": _round(rotation_degrees), "z": 0.0},
        "scale": {"x": _round(scale), "y": _round(scale), "z": _round(scale)},
    }


def _height_scale_lookup(
    trees: list[tuple[float, float, dict]],
) -> tuple[list[float], float, float, float]:
    """
    (heights, min_h, min_scale, scale_multiplier) for the FALLBACK
    course-relative height remap -- the path _tree_scale takes for any
    tree whose chosen prefab has no measured native height (see module
    docstring). `fallback_scale = (h - min_h) * scale_multiplier +
    min_scale` stretches this batch's detected-height span onto
    [MIN_HEIGHT_SCALE, MAX_HEIGHT_SCALE]. Computed once per batch; both
    builders pass the result through _tree_scale per tree.
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


def _tree_scale(
    detected_h: float,
    native_h: Optional[float],
    fb_min_h: float,
    fb_min_scale: float,
    fb_multiplier: float,
) -> float:
    """One tree's uniform x=y=z scale factor.

    CALIBRATED (`native_h` known and > 0): `detected_h / native_h` -- the
    tree renders at its LiDAR-detected canopy height relative to the
    prefab's measured native (scale-1.0) height, so a tree detected at
    its native height lands at ~1.0.

    FALLBACK (`native_h` is None): the original course-relative remap,
    `(detected_h - fb_min_h) * fb_multiplier + fb_min_scale`, with the
    `fb_*` values straight from _height_scale_lookup.

    Either result is clamped to [TREE_SCALE_CLAMP_MIN,
    TREE_SCALE_CLAMP_MAX]. With `native_h` None the result is identical
    to the pre-calibration arithmetic (the fallback curve already lands
    inside the clamp).

    Crown-WIDTH is not scaled independently (x=y=z). If that's ever
    wanted back (Chad's original did scale.x/z from LiDAR canopy radius),
    asset_catalog.json's native_canopy_radius_m is the measured value to
    divide the detected radius by here.
    """
    if native_h is not None and native_h > 0:
        raw = detected_h / native_h
    else:
        raw = (detected_h - fb_min_h) * fb_multiplier + fb_min_scale
    return min(max(raw, TREE_SCALE_CLAMP_MIN), TREE_SCALE_CLAMP_MAX)


def _deterministic_tree_rng(x: float, z: float, salt: int = 0) -> random.Random:
    """
    A per-tree random.Random seeded from its own (rounded) position --
    used by build_tree_objects_v2019/_v2021 whenever no explicit `rng`
    is passed in, so repeated builds of the SAME object_list.json
    (unchanged tree positions) reproduce the exact same rotation/type-
    bucket/asset-path draws every time, instead of a single shared
    unseeded random.Random() reshuffling everything on every call.

    This matters beyond cosmetic reproducibility: PGA2k_gen.py's
    step_import_ingame_edits recomputes this exact construction in
    memory to diff against a previously-written, saved course -- with
    an unseeded shared RNG, EVERY tree would read as "new" (freshly
    drawn) plus "missing" (the old draw), even with zero real in-game
    edits. Seeding per-tree from position fixes that while still
    varying naturally tree-to-tree (unlike one fixed global seed, which
    would make every tree at every position share the same draw
    sequence offset).

    hash() of a tuple of floats/ints (unlike of str/bytes) isn't
    affected by PYTHONHASHSEED randomization, so this is stable across
    separate process runs, not just within one.
    """
    return random.Random(hash((round(x, 3), round(z, 3), salt)))


def build_tree_objects_v2019(
    trees: list[tuple[float, float, dict]],
    theme: Optional[int],
    tree_variety: bool = False,
    rng: Optional[random.Random] = None,
    species: Optional[TreeThemeSpecies] = None,
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

    SPECIES ROUTING (species != None -- a TreeThemeSpecies from
    tree_themes.json, see load_tree_theme_species): each tree is routed
    by its TREE_TYPE_TAG value into that theme's matching species
    bucket (e.g. "pine" from a leaf_type=needleleaved hint -- see
    LEAF_TYPE_TREE_HINTS / apply_area_tree_type_hints), and its type id
    is drawn at random from that bucket. A tree with no TREE_TYPE_TAG
    (or one whose value isn't a bucket name for this theme) uses
    species.default_species. This mode replaces the theme /
    tree_variety / normal-vs-skinny logic below entirely -- the buckets
    ARE the curated pools. A bucket that somehow resolves empty falls
    back to type id 0. This is the v2019 counterpart to
    build_tree_objects_v2021's tree_type_asset_paths.

    Without species (the default), a tree's TREE_TYPE_TAG is NOT
    consulted -- v2019's catalog is numeric ids per theme, and with no
    species table there's no mapping from an OSM tag value onto it;
    classification is height/radius normal-vs-skinny only, as below.

    Classification as "normal" vs. "skinny" uses height/radius (h/r
    >= SKINNY_HEIGHT_TO_RADIUS_RATIO_V2019), same as Chad's original.
    radius comes from a tree's TREE_RADIUS_TAG if present (e.g. a real
    per-tree radius from LIDAR canopy detection -- see
    ingest/tree_detection.py), falling back to the flat TREE_RADIUS_M
    otherwise (matching upstream OSM node tags' own lack of a tree-
    radius concept -- there's nothing to read for a plain OSM-sourced
    tree).

    Scale is uniform x=y=z (see _tree_scale / module docstring): the
    chosen type id is resolved FIRST, then its native height is looked
    up via native_tree_height_v2019(type_id, theme) -- so calibration
    only applies when `theme` is the theme asset_catalog.json was
    captured from (rustic); every other theme's type ids fall back to
    the course-relative remap. Chad's original also scaled x/z from
    canopy radius independently of y from height; this project
    deliberately keeps scale uniform.

    Any tree carrying CARTPATH_DEBUG_MARKER_TAG (see
    move_trees_off_cartpaths' debug_mark_only mode) is pulled out
    before any of the above and built into its own separate group
    instead -- Key {"category": CARTPATH_DEBUG_MARKER_CATEGORY_V2019,
    "type": CARTPATH_DEBUG_MARKER_TYPE_V2019, "theme": False}, items at
    CARTPATH_DEBUG_MARKER_SCALE -- so it shows up in-game as an
    oversized, obviously-not-a-tree prop instead of a real tree.

    rng: pass an explicit random.Random for a single shared draw
    sequence (e.g. deterministic testing); left as None (the default),
    each tree instead gets its OWN random.Random seeded from its
    position (see _deterministic_tree_rng) -- rotation/type-bucket
    draws are then stable across repeated calls on the same trees list,
    rather than reshuffling every call.
    """
    if not trees:
        return []

    debug_marker_items = [
        _placed_item(x, z, CARTPATH_DEBUG_MARKER_SCALE)
        for x, z, tags in trees if tags.get(CARTPATH_DEBUG_MARKER_TAG)
    ]
    trees = [(x, z, tags) for x, z, tags in trees if not tags.get(CARTPATH_DEBUG_MARKER_TAG)]

    def _group(tree_type: int) -> dict:
        return {"Key": {"category": 0, "type": tree_type, "theme": True}, "Value": {"items": [], "clusters": []}}

    groups: list[dict] = []
    if trees and species is not None:
        type_groups: dict[int, dict] = {}
        heights, min_h, min_scale, scale_multiplier = _height_scale_lookup(trees)
        for (x, z, tags), h in zip(trees, heights):
            tree_rng = rng if rng is not None else _deterministic_tree_rng(x, z)
            rotation = tree_rng.uniform(0, 359)  # drawn before the type pick, order unchanged from before
            bucket = species.ids_for(tags.get(TREE_TYPE_TAG)) or (0,)
            tree_type = tree_rng.choice(list(bucket))
            scale = _tree_scale(
                h, native_tree_height_v2019(tree_type, theme), min_h, min_scale, scale_multiplier,
            )
            item = _placed_item(x, z, scale, rotation)
            type_groups.setdefault(tree_type, _group(tree_type))["Value"]["items"].append(item)
        groups = [type_groups[t] for t in sorted(type_groups)]

    elif trees:
        normal_tree_ids = NORMAL_TREES_V2019.get(theme, [0])
        if (not tree_variety) or len(normal_tree_ids) == 0:
            normal_tree_ids = [0]
        skinny_tree_ids = SKINNY_TREES_V2019.get(theme, normal_tree_ids)
        if (not tree_variety) or len(skinny_tree_ids) == 0:
            skinny_tree_ids = []

        normal_groups = {t: _group(t) for t in normal_tree_ids}
        skinny_groups = {t: _group(t) for t in skinny_tree_ids}

        heights, min_h, min_scale, scale_multiplier = _height_scale_lookup(trees)

        for (x, z, tags), h in zip(trees, heights):
            tree_rng = rng if rng is not None else _deterministic_tree_rng(x, z)
            rotation = tree_rng.uniform(0, 359)  # drawn before the group pick, order unchanged from before
            try:
                radius = float(tags.get(TREE_RADIUS_TAG, TREE_RADIUS_M))
            except (TypeError, ValueError):
                radius = TREE_RADIUS_M
            if radius > 0 and h / radius >= SKINNY_HEIGHT_TO_RADIUS_RATIO_V2019 and skinny_groups:
                group = tree_rng.choice(list(skinny_groups.values()))
            else:
                group = tree_rng.choice(list(normal_groups.values()))
            scale = _tree_scale(
                h, native_tree_height_v2019(group["Key"]["type"], theme),
                min_h, min_scale, scale_multiplier,
            )
            group["Value"]["items"].append(_placed_item(x, z, scale, rotation))

        groups = [g for g in list(normal_groups.values()) + list(skinny_groups.values()) if g["Value"]["items"]]

    if debug_marker_items:
        groups.append({
            "Key": {
                "category": CARTPATH_DEBUG_MARKER_CATEGORY_V2019,
                "type": CARTPATH_DEBUG_MARKER_TYPE_V2019,
                "theme": False,
            },
            "Value": {"items": debug_marker_items, "clusters": []},
        })

    return groups


# Same fence-post prop CARTPATH_DEBUG_MARKER_CATEGORY_V2019/TYPE_V2019 use
# for an oversized cart-path debug marker (confirmed directly by the
# user against a real placedObjects2.json: it's literally the same
# category/type ids -- category=13/type=34, not the koi fish an earlier
# guess of 11/43 turned out to be) -- reused here for a building-corner
# stake instead, at a subtle scale rather than an obvious one.
BUILDING_STAKE_CATEGORY_V2019 = CARTPATH_DEBUG_MARKER_CATEGORY_V2019
BUILDING_STAKE_TYPE_V2019 = CARTPATH_DEBUG_MARKER_TYPE_V2019
BUILDING_STAKE_SCALE_V2019 = 0.5  # subtle -- just enough to mark a corner, not call attention to itself


def build_building_stake_objects_v2019(features: list[Feature]) -> list[dict]:
    """
    v2019 counterpart to build_building_stake_objects_v2021 -- one
    stake at every exterior vertex of every "building" Feature (see
    ingest/osm.py), all in a single placedObjects2.json group: Key
    {"category": BUILDING_STAKE_CATEGORY_V2019, "type":
    BUILDING_STAKE_TYPE_V2019, "theme": False}, items scaled to
    BUILDING_STAKE_SCALE_V2019 (0.5x -- vs. the same prop's oversized
    6.0x CARTPATH_DEBUG_MARKER_SCALE when used as a cart-path debug
    marker). No rotation (a stake has no meaningful facing).

    Only Polygon-geometry building features contribute vertices --
    matches how splines.py's feature_to_spline already only handles
    building as an area/fairway-like shape, never a bare line (same
    restriction build_building_stake_objects_v2021 uses).
    """
    group = {
        "Key": {"category": BUILDING_STAKE_CATEGORY_V2019, "type": BUILDING_STAKE_TYPE_V2019, "theme": False},
        "Value": {"items": [], "clusters": []},
    }
    for f in features:
        if f.kind != "building" or f.geometry.geom_type != "Polygon":
            continue
        for x, z in f.geometry.exterior.coords[:-1]:  # [:-1] drops the closing repeat of the first point
            group["Value"]["items"].append(_placed_item(x, z, scale=BUILDING_STAKE_SCALE_V2019))

    return [group] if group["Value"]["items"] else []


# ---------------------------------------------------------------------------
# v2021+ -- real Unity asset-path scheme, confirmed against this
# project's own sort_objects_v3.py/generate_rough_border_v2.py.
# ---------------------------------------------------------------------------


def _placed_object_group_v2021(asset_path: str) -> dict:
    """One placedObjects2.json group entry, empty of items -- v2021+'s
    Key.path/Value.{items,clusters,splines} schema (see module
    docstring), not v2019's category/type/theme scheme."""
    return {"Key": {"path": asset_path}, "Value": {"items": [], "clusters": [], "splines": []}}


# (category, type) -> AssetEntry(path=...) -- own copy of the same lookup
# collections.py's _ENTRY_BY_KEY builds, used here to resolve cluster-fill
# and object-spline-fill records (both still schema-neutral
# {"category","type",...} dicts out of object_clusters.py) to a v2021+
# asset path.
_ENTRY_BY_KEY = {(e.category, e.type): e for e in ASSET_ENTRIES}


def cluster_records_to_v2021_groups(records: list[dict]) -> list[dict]:
    """
    v2021+ counterpart to object_clusters.cluster_records_to_v2019_groups
    -- placedObjects2 groups (Key {"path": ...}) from
    pack_cluster_records' schema-neutral stamp-mode output, one group
    per distinct asset path, via _placed_object_group_v2021. The
    Value.clusters entry shape itself (_cluster_entry_from_record) is
    already version-agnostic (confirmed identical in the hhills3_2019
    vs hhills3_2021 diff), so this only differs from the v2019 version
    in how the Key is resolved. Exists so stamp-mode cluster fills keep
    working under game_version=2021+ even though spline-mode fills are
    the new default there (see the conversation) -- this was previously
    a gap: step_write_objects's v2021+ branch dropped cluster_records
    entirely.
    """
    groups: dict[str, dict] = {}
    for record in records:
        entry = _ENTRY_BY_KEY.get((record["category"], record["type"]))
        if entry is None:
            continue  # asset_catalog.json changed since this record was packed
        group = groups.setdefault(entry.path, _placed_object_group_v2021(entry.path))
        group["Value"]["clusters"].append({
            "position": {
                "x": _round(record["x"] - GRID_ORIGIN_OFFSET), "y": "-Infinity",
                "z": _round(record["z"] - GRID_ORIGIN_OFFSET),
            },
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
            "seed": record["seed"],
            "count": record["count"],
            "radius": _round(record["radius"]),
        })
    return list(groups.values())


def _make_object_spline_waypoint(x: float, z: float) -> dict:
    """
    One waypoint of an object-spline fill's path.waypoints -- degenerate
    bezier handles (pointOne == pointTwo == waypoint), same convention
    ref/generate_rough_border_v2.py's make_waypoint uses for every point
    of a polygon-derived object spline (lets the engine round the
    corners itself rather than this project computing real tangent
    handles the way splines.py's _build_waypoints does for surface
    splines). `z` lands in the waypoint dict's "y" field -- 2-D,
    top-down, same convention surfaceSplines.json's own waypoints use.
    """
    point = {"x": _round(x), "y": _round(z)}
    return {"pointOne": dict(point), "pointTwo": dict(point), "waypoint": dict(point)}


def object_spline_fill_records_to_v2021_groups(records: list[dict]) -> list[dict]:
    """
    v2021+ ONLY -- placedObjects2 groups (Key {"path": ...}) from
    object_clusters.pack_spline_records' schema-neutral output, one
    group per distinct asset path. Each record becomes one
    Value.splines[] entry: a closed, filled object spline around the
    record's (already <= MAX_SPLINE_FILL_PIECE_SIZE_M) polygon piece,
    density fillPct -- see the module docstring and
    ref/generate_rough_border_v2.py's polygon_to_object_spline, which
    this mirrors exactly (state=0, ClosedPath/isClosed/isFilled=True,
    width=1.0 -- the width of a FILLED spline doesn't bound the fill
    area, just the corner rounding, so a fixed value is fine here).
    v2019 has no object-spline concept at all -- there is no v2019
    counterpart to this function (see step_write_objects: those records
    are dropped with a printed note for that game_version instead).
    """
    groups: dict[str, dict] = {}
    for record in records:
        entry = _ENTRY_BY_KEY.get((record["category"], record["type"]))
        if entry is None:
            continue  # asset_catalog.json changed since this record was packed
        group = groups.setdefault(entry.path, _placed_object_group_v2021(entry.path))
        waypoints = [
            _make_object_spline_waypoint(x - GRID_ORIGIN_OFFSET, z - GRID_ORIGIN_OFFSET)
            for x, z in record["waypoints"]
        ]
        group["Value"]["splines"].append({
            "path": {
                "waypoints": waypoints,
                "width": 1.0,
                "state": 0,
                "ClosedPath": True,
                "isClosed": True,
                "isFilled": True,
            },
            "fillPct": _round(record["fill_pct"]),
        })
    return list(groups.values())


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

    Scale is uniform x=y=z (see _tree_scale / module docstring):
    calibrated as detected_height / the prefab's measured native height
    when `path` is in asset_catalog.json's NATIVE_TREE_HEIGHT_BY_PATH,
    else the course-relative height remap.

    Unlike build_tree_objects_v2019, a CARTPATH_DEBUG_MARKER_TAG tree is
    NOT special-cased here -- built as an ordinary tree, same as any
    other -- since there's no known v2021 marker asset path to swap it
    for (v2019's CARTPATH_DEBUG_MARKER_CATEGORY_V2019/TYPE_V2019 debug
    marker has no v2021 equivalent yet).

    rng: pass an explicit random.Random for a single shared draw
    sequence (e.g. deterministic testing); left as None (the default),
    each tree instead gets its OWN random.Random seeded from its
    position (see _deterministic_tree_rng) -- asset-path/rotation draws
    are then stable across repeated calls on the same trees list,
    rather than reshuffling every call.
    """
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
        tree_rng = rng if rng is not None else _deterministic_tree_rng(x, z)
        tree_type = tags.get(TREE_TYPE_TAG)
        if tree_type is not None and tree_type in tree_type_asset_paths:
            path = tree_type_asset_paths[tree_type]
        elif tree_asset_paths:
            path = tree_rng.choice(tree_asset_paths)
        else:
            # Tagged with a tree_type that isn't in
            # tree_type_asset_paths, and no general pool to fall back
            # to -- skip rather than guess at a path.
            continue
        scale = _tree_scale(
            h, NATIVE_TREE_HEIGHT_BY_PATH.get(path), min_h, min_scale, scale_multiplier,
        )
        item = _placed_item(x, z, scale, tree_rng.uniform(0, 359))
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


# Stream drops are placed as a rigid waterfall + splash pair, one per
# seam between two consecutive water tiles (see terrain/streams.py's
# build_stream_records / stream_water_tiles). streams.json freezes the
# geometry -- [x, z, rot_y, level] rows in the course-local frame, the
# anchor already the upper tile's downstream edge centre -- so the
# writer only shifts by GRID_ORIGIN_OFFSET and adds output_height_shift_m.
# The waterfall and its splash share x/z and rot_y exactly; only `level`
# differs (the falls carries the upper tile's surface height, the splash
# the lower tile's).
#
# v2019 category/type/theme confirmed against a real hand-placed 2019
# course: the "low falls" keyed {"category": 16, "type": 3, "theme":
# false} and the "low splash" {"category": 16, "type": 4, "theme": false}
# sitting beside it, items carrying an explicit elevation (not "-Infinity").
WATERFALL_CATEGORY_V2019 = 16
WATERFALL_TYPE_V2019 = 3
WATERSPLASH_CATEGORY_V2019 = 16
WATERSPLASH_TYPE_V2019 = 4

WATERFALL_DEFAULT_ASSET_PATH = "Assets/Effects/Waterfalls/Prefabs/Waterfall_LowFallsPrefab"
# TODO: unverified -- the v2021+ splash path is a best guess (sibling of
# WATERFALL_DEFAULT_ASSET_PATH); only the v2019 form is confirmed.
WATERSPLASH_DEFAULT_ASSET_PATH = "Assets/Effects/Waterfalls/Prefabs/Waterfall_LowSplashPrefab"
WATERFALL_SCALE_V2019 = (1.0, 1.0, 1.0)  # confirmed from the real 2019 sample
WATERFALL_SCALE_V2021 = (0.5, 1.0, 1.0)  # ref/generate_streams.py's value


def _stream_drop_item(x: float, z: float, rot_y: float, level: float, height_shift_m: float,
                      scale: tuple[float, float, float]) -> dict:
    """One waterfall / splash instance at the frozen anchor (x, z), at an
    EXPLICIT elevation level + height_shift_m -- the real 2019 sample
    carries a real y, and these prefabs (like a water plane, unlike a
    tree) aren't ground-snapped by the game. Non-uniform scale, so not
    via _placed_item."""
    sx, sz_width, sy = scale
    return {
        "position": {
            "x": _round(x - GRID_ORIGIN_OFFSET),
            "y": _round(level + height_shift_m),
            "z": _round(z - GRID_ORIGIN_OFFSET),
        },
        "rotation": {"x": 0.0, "y": _round(rot_y), "z": 0.0},
        "scale": {"x": _round(sx), "y": _round(sy), "z": _round(sz_width)},
    }


def _iter_stream_drops(stream_records: list[dict], key: str):
    for record in stream_records:
        for row in record.get(key, []):
            # [x, z, rot_y, level]
            yield row[0], row[1], row[2], row[3]


def _stream_drop_group_v2019(
    stream_records: list[dict], category: int, type_: int,
    scale: tuple[float, float, float], key: str, height_shift_m: float,
) -> list[dict]:
    group = {
        "Key": {"category": category, "type": type_, "theme": False},
        "Value": {"items": [], "clusters": []},
    }
    for x, z, rot_y, level in _iter_stream_drops(stream_records, key):
        group["Value"]["items"].append(_stream_drop_item(x, z, rot_y, level, height_shift_m, scale))
    return [group] if group["Value"]["items"] else []


def _stream_drop_group_v2021(
    stream_records: list[dict], asset_path: str, scale: tuple[float, float, float],
    key: str, height_shift_m: float, what: str,
) -> list[dict]:
    if not asset_path:
        raise ValueError(f"build_{what}_objects_v2021 needs a real asset path -- see the module.")
    group = _placed_object_group_v2021(asset_path)
    for x, z, rot_y, level in _iter_stream_drops(stream_records, key):
        group["Value"]["items"].append(_stream_drop_item(x, z, rot_y, level, height_shift_m, scale))
    return [group] if group["Value"]["items"] else []


def build_waterfall_objects_v2019(stream_records: list[dict], height_shift_m: float = 0.0) -> list[dict]:
    """One "low falls" prefab (Key {"category": 16, "type": 3, "theme":
    false}) per stream-tile seam frozen into streams.json (see
    terrain/streams.py's build_stream_records). height_shift_m is
    project.json's output_height_shift_m -- added to each frozen surface
    level to land in the normalized frame (so run write-terrain first)."""
    return _stream_drop_group_v2019(
        stream_records, WATERFALL_CATEGORY_V2019, WATERFALL_TYPE_V2019,
        WATERFALL_SCALE_V2019, "waterfalls", height_shift_m,
    )


def build_waterfall_objects_v2021(
    stream_records: list[dict], asset_path: str, height_shift_m: float = 0.0,
) -> list[dict]:
    """v2021+ equivalent of build_waterfall_objects_v2019 -- a single
    Key.path group. asset_path defaults (in step_write_objects) to
    WATERFALL_DEFAULT_ASSET_PATH."""
    return _stream_drop_group_v2021(
        stream_records, asset_path, WATERFALL_SCALE_V2021, "waterfalls", height_shift_m, "waterfall",
    )


def build_watersplash_objects_v2019(stream_records: list[dict], height_shift_m: float = 0.0) -> list[dict]:
    """The "low splash" companion (Key {"category": 16, "type": 4,
    "theme": false}) of each waterfall -- same x/z and rotation, dropped
    to the LOWER tile's surface (see build_stream_records "splashes")."""
    return _stream_drop_group_v2019(
        stream_records, WATERSPLASH_CATEGORY_V2019, WATERSPLASH_TYPE_V2019,
        WATERFALL_SCALE_V2019, "splashes", height_shift_m,
    )


def build_watersplash_objects_v2021(
    stream_records: list[dict], asset_path: str, height_shift_m: float = 0.0,
) -> list[dict]:
    """v2021+ equivalent of build_watersplash_objects_v2019. asset_path
    defaults (in step_write_objects) to WATERSPLASH_DEFAULT_ASSET_PATH
    (unverified -- see the constant)."""
    return _stream_drop_group_v2021(
        stream_records, asset_path, WATERFALL_SCALE_V2021, "splashes", height_shift_m, "watersplash",
    )


def merge_object_groups(groups: list[dict]) -> list[dict]:
    """
    Collapse `groups` down to exactly one group per distinct Key,
    concatenating each Value list field (items/clusters/splines) in
    encounter order, first-seen Key order preserved.

    build_tree_objects_v2019/_v2021 and
    object_clusters.cluster_records_to_v2019_groups already merge
    groups sharing a Key WITHIN their own call, but
    step_write_objects concatenates several independent builder calls
    (trees, building stakes, cluster fills) with a plain `+=` -- if two
    of those calls happen to land on the same Key (e.g. a user's
    cluster-fill spec tagged onto the same category/type
    BUILDING_STAKE_CATEGORY_V2019/TYPE_V2019 uses, or a cluster-fill
    category that happens to match the tree category), the result was
    two separate placedObjects2.json nodes for what the game treats as
    one asset, instead of one node with everything merged in. This is
    the single place that's fixed up, applied once after every
    builder's contribution has been concatenated together.
    """
    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for g in groups:
        key = tuple(sorted(g["Key"].items()))
        if key not in merged:
            merged[key] = {"Key": g["Key"], "Value": {}}
            order.append(key)
        dest_value = merged[key]["Value"]
        for field, seq in g.get("Value", {}).items():
            dest_value.setdefault(field, [])
            dest_value[field].extend(seq)
    return [merged[k] for k in order]


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


def save_objects(
    trees: list[tuple[float, float, dict]], cluster_records: list[dict], path: Path,
    collection_objects: Optional[list[dict]] = None,
    object_spline_fill_records: Optional[list[dict]] = None,
    ingame_object_records: Optional[list[dict]] = None,
) -> None:
    """
    Write objects.json -- the pack-objects step's combined, VERSION-
    AGNOSTIC output: `trees` (see save_object_list; passed through
    unchanged, same shape), `cluster_records`
    (course_output/object_clusters.py's pack_cluster_records output --
    already-packed circle positions/counts/RNG seeds, frozen at pack
    time so they don't reroll on every read), `collection_objects`
    (course_output/collections.py -- resolved collection member
    placements, {"x","z","rotation_deg","scale","category","type",
    "theme","path","source_id"}, deterministic, no RNG),
    `object_spline_fill_records` (object_clusters.py's
    pack_spline_records output -- v2021+-only object-spline fills, no
    RNG either, just an already-subdivided polygon per record), and
    `ingame_object_records` (course_output/ingame_objects.py -- objects
    a user placed by hand-editing an exported .course in the game's own
    editor, captured back via PGA2k_gen.py's step_import_ingame_edits;
    read from ingame_objects.json, not overwritten by this step).

    This is what step_write_objects (any game_version) and the GUI's
    live preview both read -- neither needs its own copy of cluster-
    fill packing logic, and switching game_version, or just previewing
    after a Splines-tab Fill/Clear, never re-rolls cluster RNG or
    requires a `course/` extraction. Same "compile once, format/consume
    at read time" split object_list.json already established for trees
    alone (see step_generate_trees' docstring); this extends it to
    cluster fills, object-spline fills, collections, and imported
    in-game objects, which previously had no version-agnostic
    intermediate at all -- they were packed directly into
    placedObjects2.json's schema inside step_write_objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [{"kind": "tree", "x": x, "z": z, "tags": tags} for x, z, tags in trees]
    entries += [dict(record, kind="cluster") for record in cluster_records]
    entries += [dict(obj, kind="collection_object") for obj in (collection_objects or [])]
    entries += [dict(record, kind="object_spline_fill") for record in (object_spline_fill_records or [])]
    entries += [dict(record, kind="ingame_object") for record in (ingame_object_records or [])]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def load_objects(
    path: Path,
) -> tuple[list[tuple[float, float, dict]], list[dict], list[dict], list[dict], list[dict]]:
    """(trees, cluster_records, collection_objects,
    object_spline_fill_records, ingame_object_records) from
    objects.json -- inverse of save_objects, split back into the shapes
    their respective consumers (build_tree_objects_v20XX,
    object_clusters.cluster_records_to_v20XX_groups,
    collections.build_collection_objects_v20XX,
    object_spline_fill_records_to_v2021_groups,
    ingame_objects.build_ingame_objects_v20XX) already expect."""
    with Path(path).open(encoding="utf-8") as fh:
        entries = json.load(fh)
    trees = [(e["x"], e["z"], e["tags"]) for e in entries if e["kind"] == "tree"]
    cluster_records = [{k: v for k, v in e.items() if k != "kind"} for e in entries if e["kind"] == "cluster"]
    collection_objects = [
        {k: v for k, v in e.items() if k != "kind"} for e in entries if e["kind"] == "collection_object"
    ]
    object_spline_fill_records = [
        {k: v for k, v in e.items() if k != "kind"} for e in entries if e["kind"] == "object_spline_fill"
    ]
    ingame_object_records = [
        {k: v for k, v in e.items() if k != "kind"} for e in entries if e["kind"] == "ingame_object"
    ]
    return trees, cluster_records, collection_objects, object_spline_fill_records, ingame_object_records


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
