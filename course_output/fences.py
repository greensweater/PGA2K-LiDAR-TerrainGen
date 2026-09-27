"""
course_output/fences.py

v2023 spline fences and walls -- OSM fence/wall/hedge ways (ingest/osm.py
kinds "fence", "wall", "hedge") become placedObjects3.json objectPaths[]
entries (see V2023_SCHEMA.md "objectPaths rule fields"). v2023 only:
v2019/v2021 have no objectPaths node.

Same "freeze at pack, format at write" split as out_of_bounds.py:
build_fence_records -> FenceRecords in the course-local [0, COURSE_SIZE_M]
frame (fences.json), fence_records_to_groups_v2023 -> placedObjects3
groups ({Key: {path: <fence asset>}, Value: {objectPaths: [...]}}), which
objects.merge_object_groups + placed_object_groups_to_v2023 fold in
alongside items/clusters/splines.

FENCE TYPES (FENCE_TYPES): the 28 entries of the v2023 editor's fence
menu, in menu order (V2023_SCHEMA.md "Fence types"; captured from the
2026-09-27 fence sampler). A type is one or more objectPath PARTS over
the same waypoints -- the game writes picket fence as pickets + posts,
and the two railing walls as a brick wall + an inner metal railing, one
objectPath per part under each part's own asset Key. Every part asset
carries fence_options in asset_catalog.json (FENCE_ENTRIES).

ROUTING (tags -> type), first match wins:
  1. wall=retaining_wall / barrier=retaining_wall -> retaining_wall
  2. hedge kind (barrier=hedge / natural=hedge)   -> hedge
  3. material=* (then wall=*, then fence_type=*, then barrier=*) through
     _MATERIAL_TYPES -- which also takes any FENCE_TYPES name verbatim
     (material=canvas_white, material=picket, ...)
  4. the kind's default (_KIND_DEFAULT_TYPES)

PER-WAY OVERRIDES are this project's own tags on the Feature (what the
GUI's Objects / Fences panel writes; ingest-osm carries them over a
re-ingest -- FENCE_STYLE_TAGS): pga_fence_preset=<FENCE_PRESETS name>,
pga_fence_asset=<FENCE_TYPES name, or a part asset's label or path>, and
pga_fence_<rule field> for each field in RULE_FIELDS (e.g.
pga_fence_spacingRule=3, pga_fence_height=-1.5). Precedence: part
defaults < preset < explicit field tags. On a multi-part type the
SHARED_RULE_FIELDS (width / height / heightRule -- what keeps the parts
lined up) apply to every part, the rest to the first part only. Rule
values are then checked against each part asset's option matrix
(validate_fence_rules).

HANDLES (derived from templates/2023_fences.course, V2023_SCHEMA.md
"objectPath handle rule"): an open run's end waypoints have their outer
handle ON the waypoint and the inner one ENDPOINT_HANDLE_RATIO of the
segment toward the neighbour; a smooth interior waypoint's handles lie
along the tangent (next - prev), each SMOOTH_HANDLE_RATIO of its adjacent
segment long. Interior waypoints turning more than CORNER_ANGLE_DEG (and
every waypoint when hasCurves is false) get corner handles -- each handle
on its own segment, so the curve stays a straight line through a sharp
corner. OSM fences are polylines with real corners; rounding a
rectangular paddock fence into an oval would be wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from course_output.asset_catalog import FENCE_ENTRIES
from course_output.userLayers import GRID_ORIGIN_OFFSET

_DECIMALS = 4

_WALLS = "Assets/CourseGen/Detail/Walls/"
STONE_WALL_ASSET = _WALLS + "StoneWallAPostAPrefab"
WOOD_FENCE_ASSET = _WALLS + "WoodFencesAPostAPrefab"
WOOD_PANELS_ASSET = _WALLS + "WoodFencesBPostCPrefab"
UNI_FENCE_ASSET = _WALLS + "UniFencePostAPrefab"
WIRE_FENCE_ASSET = _WALLS + "WireFenceAPostAPrefab"
TRI_FENCE_WHITE_ASSET = _WALLS + "TriFence01Post01APrefab"
TRI_FENCE_NATURAL_ASSET = _WALLS + "TriFence02Post01APrefab"
PICKET_PICKETS_ASSET = _WALLS + "PicketFencePicket01Prefab"
PICKET_POSTS_ASSET = _WALLS + "PicketFencePost01APrefab"
HEDGE_ASSET = _WALLS + "HedgeSplinePostPrefab"
BRICK_WALL_ASSET = _WALLS + "Asia_Walls/Asia_BrickWalls_PostPrefab"
ASIA_BLACK_CAP_ASSET = _WALLS + "Asia_Walls/Asia_Walls_PostPrefab"
ASIA_PANELS_ASSET = _WALLS + "Asia_Walls/Asia_KoreanWalls_PostPrefab"
BRIT_COBBLE_ASSET = _WALLS + "Brit_Walls/Brit_CobbleWalls_GeneratePostPrefab"
BRIT_PAVERS_ASSET = _WALLS + "Brit_Walls/Brit_LowWalls_GeneratePostPrefab"
BRIT_RED_BRICK_ASSET = _WALLS + "Brit_Walls/Brit_BrickWalls_01_GeneratePostPrefab"
BRIT_BIG_STONE_ASSET = _WALLS + "Brit_Walls/Brit_BrickWalls_02_GeneratePostPrefab"
BRIT_CINDER_ASSET = _WALLS + "Brit_Walls/Brit_BrickWalls_03_GeneratePostPrefab"
BRIT_QUARRIED_ASSET = _WALLS + "Brit_Walls/Brit_Walls_01_GeneratePostPrefab"
BRICK_HIGH_ASSET = _WALLS + "BrickWallsHighAPostAPrefab"
BRICK_LOW_ASSET = _WALLS + "BrickWallsLowAPostAPrefab"
BRICK_HIGH_RAILS_ASSET = _WALLS + "BrickWallsHighRailsAPostAPrefab"
HIGH_METAL_RAILING_ASSET = _WALLS + "HighMetalFenceInnerPost01Prefab"
BRICK_LOW_RAILS_ASSET = _WALLS + "BrickWallsLowRailsAPostAPrefab"
LOW_METAL_RAILING_ASSET = _WALLS + "LowMetalFenceInnerPost01Prefab"
RETAINING_WALL_ASSET = _WALLS + "RetainWallAPostAPrefab"
CANVAS_RED_ASSET = _WALLS + "CanvasFence01AARedPostAPrefab"
CANVAS_BLACK_ASSET = _WALLS + "CanvasFence01AABlackPostAPrefab"
CANVAS_BLUE_ASSET = _WALLS + "CanvasFence01AABluePostAPrefab"
CANVAS_GREEN_ASSET = _WALLS + "CanvasFence01AAPostAPrefab"
CANVAS_WHITE_ASSET = _WALLS + "CanvasFence01AAWhitePostAPrefab"

FENCE_KINDS = ("fence", "wall", "hedge")

# objectPath rule fields (game key names) and their types. width/hasCurves
# live under objectPath.path in the game's JSON; the rest at the top level.
RULE_FIELDS: dict[str, type] = {
    "width": float, "height": float, "spacing": float,
    "spacingRule": int, "flexibilityRule": int, "heightRule": int, "hasCurves": bool,
}
# Fields with a per-asset option matrix in asset_catalog.json fence_options.
_MATRIX_FIELDS = ("spacingRule", "hasCurves", "heightRule", "flexibilityRule")


def _rules(spacingRule: int, *, width: float = 4.0, height: float = 0.0, spacing: float = 4.0,
           flexibilityRule: int = 1, heightRule: int = 0, hasCurves: bool = True) -> dict:
    return {"width": width, "height": height, "spacing": spacing, "spacingRule": spacingRule,
            "flexibilityRule": flexibilityRule, "heightRule": heightRule, "hasCurves": hasCurves}


# Per-part-asset defaults. The first 8 are that asset's plain (contoured,
# unburied) row in templates/2023_fences.course, in-game verified by the
# 2026-09-26 fence test; the rest are what the game's editor writes when
# the type is placed from its menu (the 2026-09-27 fence sampler).
# Retaining wall has only one sample row -- the heightmap-hugging one; its
# -0.715 renders 0.3 m high in-game (fence test, 2026-09-26), so -1.015.
RETAINING_WALL_HEIGHT_M = -1.015
ASSET_DEFAULT_RULES: dict[str, dict] = {
    WOOD_FENCE_ASSET: _rules(2),
    UNI_FENCE_ASSET: _rules(2),
    STONE_WALL_ASSET: _rules(0),
    CANVAS_RED_ASSET: _rules(2),
    CANVAS_BLACK_ASSET: _rules(2),
    HEDGE_ASSET: _rules(1),
    BRICK_WALL_ASSET: _rules(0, flexibilityRule=0, hasCurves=False),
    RETAINING_WALL_ASSET: _rules(0, width=2.5, height=RETAINING_WALL_HEIGHT_M),
    ASIA_BLACK_CAP_ASSET: _rules(1),
    ASIA_PANELS_ASSET: _rules(1),
    CANVAS_BLUE_ASSET: _rules(2),
    CANVAS_GREEN_ASSET: _rules(2),
    CANVAS_WHITE_ASSET: _rules(2),
    BRIT_COBBLE_ASSET: _rules(3),
    BRIT_PAVERS_ASSET: _rules(2),
    BRIT_RED_BRICK_ASSET: _rules(2),
    BRIT_BIG_STONE_ASSET: _rules(2),
    BRIT_CINDER_ASSET: _rules(2),
    BRIT_QUARRIED_ASSET: _rules(2),
    BRICK_HIGH_ASSET: _rules(2),
    BRICK_LOW_ASSET: _rules(2),
    BRICK_HIGH_RAILS_ASSET: _rules(0),
    HIGH_METAL_RAILING_ASSET: _rules(2, spacing=0.1),
    BRICK_LOW_RAILS_ASSET: _rules(0),
    LOW_METAL_RAILING_ASSET: _rules(2, spacing=0.1),
    PICKET_PICKETS_ASSET: _rules(2, spacing=0.1, flexibilityRule=0, hasCurves=False),
    PICKET_POSTS_ASSET: _rules(2, flexibilityRule=0, hasCurves=False),
    TRI_FENCE_WHITE_ASSET: _rules(2),
    TRI_FENCE_NATURAL_ASSET: _rules(2),
    WIRE_FENCE_ASSET: _rules(2),
    WOOD_PANELS_ASSET: _rules(2),
}


@dataclass(frozen=True, slots=True)
class FenceType:
    """One entry of the game's fence menu: `label` is the menu name,
    `parts` the objectPaths it writes -- (part asset, rule overrides on
    top of that asset's ASSET_DEFAULT_RULES) -- all over the same
    waypoints."""
    name: str
    label: str
    parts: tuple[tuple[str, dict], ...]


def _type(name: str, label: str, *parts) -> FenceType:
    return FenceType(name, label, tuple(p if isinstance(p, tuple) else (p, {}) for p in parts))


# The railing walls sit lowered to railing height in the sampler (Andy's
# placement, 2026-09-27) -- both parts at the same height.
_HIGH_RAILS_HEIGHT_M = -0.104
_LOW_RAILS_HEIGHT_M = -0.594

# Menu order (alphabetical by asset name in the game's list).
FENCE_TYPES: dict[str, FenceType] = {t.name: t for t in (
    _type("asian_green_cap", "Brick wall - Asian green cap", BRICK_WALL_ASSET),
    _type("asian_black_cap", "Brick wall - Asian black cap", ASIA_BLACK_CAP_ASSET),
    _type("canvas_black", "Black canvas wall", CANVAS_BLACK_ASSET),
    _type("canvas_blue", "Blue canvas wall", CANVAS_BLUE_ASSET),
    _type("brick_with_railings", "Brick with railings",
          (BRICK_LOW_RAILS_ASSET, {"height": _LOW_RAILS_HEIGHT_M}),
          (LOW_METAL_RAILING_ASSET, {"height": _LOW_RAILS_HEIGHT_M})),
    _type("stone_chunky", "Stone wall - chunky rounded", BRIT_COBBLE_ASSET),
    _type("brick_pavers", "Brick wall - pavers", BRIT_PAVERS_ASSET),
    _type("brick_red", "Brick wall - red brick", BRIT_RED_BRICK_ASSET),
    _type("brick_big_stone", "Brick wall - big stone", BRIT_BIG_STONE_ASSET),
    _type("brick_cinder", "Brick wall - cinder", BRIT_CINDER_ASSET),
    _type("brick_quarried", "Brick wall - quarried", BRIT_QUARRIED_ASSET),
    _type("canvas_green", "Green canvas wall", CANVAS_GREEN_ASSET),
    _type("hedge", "Hedge", HEDGE_ASSET),
    _type("brick_high", "High brick wall - classic brick", BRICK_HIGH_ASSET),
    _type("high_metal_fence", "High metal fence",
          (BRICK_HIGH_RAILS_ASSET, {"height": _HIGH_RAILS_HEIGHT_M}),
          (HIGH_METAL_RAILING_ASSET, {"height": _HIGH_RAILS_HEIGHT_M})),
    _type("asian_panels", "Brick wall - Asian panels", ASIA_PANELS_ASSET),
    _type("brick_low", "Low brick wall - classic brick", BRICK_LOW_ASSET),
    _type("picket", "Picket fence", PICKET_PICKETS_ASSET, PICKET_POSTS_ASSET),
    _type("canvas_red", "Red canvas wall", CANVAS_RED_ASSET),
    _type("retaining_wall", "Retaining wall", RETAINING_WALL_ASSET),
    _type("stone_wall", "Stone wall", STONE_WALL_ASSET),
    _type("three_rail_white", "Fence - 3-rail white", TRI_FENCE_WHITE_ASSET),
    _type("three_rail_natural", "Fence - 3-rail natural", TRI_FENCE_NATURAL_ASSET),
    _type("metal", "Fence - metal", UNI_FENCE_ASSET),
    _type("canvas_white", "White canvas wall", CANVAS_WHITE_ASSET),
    _type("chain_link", "Wire fence - chain-link", WIRE_FENCE_ASSET),
    _type("wood_rustic", "Wooden fence - 2-rail rustic", WOOD_FENCE_ASSET),
    _type("wood_panels", "Wooden panels", WOOD_PANELS_ASSET),
)}

_KIND_DEFAULT_TYPES = {
    "fence": "wood_rustic",
    "wall": "stone_wall",
    "hedge": "hedge",
}

# OSM material / wall / fence_type / barrier values -> FENCE_TYPES name.
# One table for all four keys: the vocabularies overlap (wall=brick,
# material=brick) and don't collide. Every FENCE_TYPES name also matches
# itself (added below), so material=canvas_white etc. work as-is.
_MATERIAL_TYPES = {
    "stone": "stone_wall",
    "dry_stone": "stone_wall",
    "flint": "stone_wall",
    "city_wall": "stone_wall",
    "cobblestone": "stone_chunky",
    "brick": "brick_low",
    "concrete": "brick_cinder",
    "concrete_block": "brick_cinder",
    "cinder_block": "brick_cinder",
    "wood": "wood_rustic",
    "pole": "wood_rustic",
    "rail": "wood_rustic",
    "split_rail": "three_rail_natural",
    "palisade": "wood_panels",
    "wood_panel": "wood_panels",
    "wooden_panels": "wood_panels",
    "panel": "wood_panels",
    "board": "wood_panels",
    "privacy": "wood_panels",
    "chain_link": "chain_link",
    "wire": "chain_link",
    "mesh": "chain_link",
    "barbed_wire": "chain_link",
    "electric": "chain_link",
    "metal": "metal",
    "steel": "metal",
    "metal_bars": "metal",
    "bars": "metal",
    "railing": "metal",
    "guard_rail": "metal",
    "handrail": "metal",
    "chain": "metal",
    "hedge": "hedge",
    "canvas": "canvas_green",
    "white_canvas": "canvas_white",
    "red_canvas": "canvas_red",
    "black_canvas": "canvas_black",
    "blue_canvas": "canvas_blue",
    "green_canvas": "canvas_green",
    **{name: name for name in FENCE_TYPES},
}

# Sample-proven "trick" recipes (V2023_SCHEMA.md "Tricks"): a fence type
# plus rule overrides on top of its part defaults.
FENCE_PRESETS: dict[str, tuple[str, dict]] = {
    "curb": ("stone_wall", {"height": -1.498, "spacingRule": 0}),
    "railroad": ("canvas_black", {"height": -2.688, "spacingRule": 2}),
    "retaining_wall": ("retaining_wall", {"width": 2.5, "height": RETAINING_WALL_HEIGHT_M, "heightRule": 0}),
}

PRESET_TAG = "pga_fence_preset"
ASSET_TAG = "pga_fence_asset"
RULE_TAG_PREFIX = "pga_fence_"
# Every per-way style tag -- what ingest-osm carries over from the previous
# features.geojson when a way is re-parsed (GUI edits survive a re-ingest).
FENCE_STYLE_TAGS = (PRESET_TAG, ASSET_TAG) + tuple(RULE_TAG_PREFIX + name for name in RULE_FIELDS)

ENDPOINT_HANDLE_RATIO = 0.25   # open-run end handle, fraction of its segment (sample: all 2-wp rows)
SMOOTH_HANDLE_RATIO = 0.375    # smooth interior handle, fraction of its adjacent segment (sample: closed retaining wall)
CORNER_ANGLE_DEG = 45.0        # interior turn above this -> corner handles (straight segments through a sharp corner)
FENCE_SIMPLIFY_TOL_M = 0.25    # Douglas-Peucker; OSM fence nodes are deliberate, so only drop near-collinear ones
FENCE_ENDPOINT_TOL_M = 0.5     # way ends within this distance are the same node (joins / closure)
FENCE_MIN_SEGMENT_M = 0.05     # consecutive points closer than this are duplicates

_VALID_ASSETS = {e.path for e in FENCE_ENTRIES}
_FENCE_OPTIONS = {e.path: e.fence_options for e in FENCE_ENTRIES}
_ASSET_BY_LABEL = {e.label: e.path for e in FENCE_ENTRIES}
_TYPE_BY_LABEL = {t.label: t.name for t in FENCE_TYPES.values()}
_unknown = {a for t in FENCE_TYPES.values() for a, _ in t.parts} ^ _VALID_ASSETS
assert not _unknown, f"fence part assets vs asset_catalog fence_options mismatch: {_unknown}"
assert set(ASSET_DEFAULT_RULES) == _VALID_ASSETS, "every fence asset needs ASSET_DEFAULT_RULES"
assert set(_KIND_DEFAULT_TYPES.values()) | set(_MATERIAL_TYPES.values()) \
    | {name for name, _ in FENCE_PRESETS.values()} <= set(FENCE_TYPES)

# Rule fields every part of a multi-part type shares (keeps the railing on
# its wall / the pickets on their posts); the rest override the first part.
SHARED_RULE_FIELDS = ("width", "height", "heightRule")


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def fence_type_for_tags(kind: str, tags: dict) -> str:
    """The FENCE_TYPES name for a fence/wall/hedge Feature from its OSM
    tags alone -- see the module docstring for the precedence. Per-way
    overrides are resolve_fence_style's job."""
    if tags.get("wall") == "retaining_wall" or tags.get("barrier") == "retaining_wall":
        return "retaining_wall"
    if kind == "hedge":
        return "hedge"
    for key in ("material", "wall", "fence_type", "barrier"):
        name = _MATERIAL_TYPES.get(tags.get(key, ""))
        if name is not None:
            return name
    return _KIND_DEFAULT_TYPES[kind]


def _requested_type(requested: str) -> Optional[FenceType]:
    """pga_fence_asset value -> a FenceType: a FENCE_TYPES name or menu
    label, or a single part asset (label or path) as a one-part type."""
    if requested in FENCE_TYPES:
        return FENCE_TYPES[requested]
    if requested in _TYPE_BY_LABEL:
        return FENCE_TYPES[_TYPE_BY_LABEL[requested]]
    path = requested if requested in _VALID_ASSETS else _ASSET_BY_LABEL.get(requested)
    if path is None:
        return None
    return FenceType(path.rsplit("/", 1)[-1], path.rsplit("/", 1)[-1], ((path, {}),))


def _parse_rule(name: str, raw) -> object:
    kind = RULE_FIELDS[name]
    if kind is bool:
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("1", "true", "yes", "curved"):
            return True
        if s in ("0", "false", "no", "straight"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    return kind(float(raw)) if kind is int else float(raw)


def validate_fence_rules(asset: str, rules: dict) -> tuple[dict, list[str]]:
    """
    Check rules against the asset's objectPaths option matrix
    (asset_catalog.json fence_options). A value outside a COMPLETE option
    set is invalid -> warned and reverted to the asset default. A value
    outside an INCOMPLETE set is unverified (the matrix only lists what a
    sample exercised) -> warned but kept, so in-game trials can extend it.
    """
    options = _FENCE_OPTIONS[asset]
    label = asset.rsplit("/", 1)[-1]
    out = dict(rules)
    warnings = []
    for name in _MATRIX_FIELDS:
        opt = options.get(name)
        if opt is None or out[name] in opt["values"]:
            continue
        if opt["complete"]:
            default = ASSET_DEFAULT_RULES[asset][name]
            warnings.append(f"{label}: {name}={out[name]!r} is not a valid option "
                            f"{opt['values']}; using {default!r}")
            out[name] = default
        else:
            warnings.append(f"{label}: {name}={out[name]!r} is unverified in-game "
                            f"(seen: {opt['values']}); keeping it")
    return out, warnings


@dataclass(frozen=True, slots=True)
class FenceStyle:
    """A resolved fence style: the type name and, per part, (asset,
    validated rule dict) -- one objectPath each over the same waypoints."""
    type_name: str
    parts: tuple[tuple[str, dict], ...]

    @property
    def key(self) -> tuple:
        """Hashable identity -- ways join into one run only when this matches."""
        return (self.type_name, tuple((a, tuple(sorted(r.items()))) for a, r in self.parts))


def resolve_fence_style(kind: str, tags: dict) -> tuple[FenceStyle, list[str]]:
    """(FenceStyle, warnings) for one fence Feature: tag routing, then
    preset, then pga_fence_asset, then pga_fence_<field> overrides, then
    each part's option-matrix check."""
    warnings: list[str] = []
    ftype = FENCE_TYPES[fence_type_for_tags(kind, tags)]
    overrides: dict = {}

    preset = tags.get(PRESET_TAG)
    if preset:
        if preset in FENCE_PRESETS:
            type_name, preset_rules = FENCE_PRESETS[preset]
            ftype = FENCE_TYPES[type_name]
            overrides.update(preset_rules)
        else:
            warnings.append(f"unknown {PRESET_TAG}={preset!r} (known: {', '.join(FENCE_PRESETS)})")

    requested = tags.get(ASSET_TAG)
    if requested:
        got = _requested_type(requested)
        if got is None:
            warnings.append(f"{ASSET_TAG}={requested!r} is not a fence type or objectPath fence asset; "
                            f"keeping {ftype.name}")
        else:
            ftype = got

    for name in RULE_FIELDS:
        raw = tags.get(RULE_TAG_PREFIX + name)
        if raw is None:
            continue
        try:
            overrides[name] = _parse_rule(name, raw)
        except ValueError:
            warnings.append(f"{RULE_TAG_PREFIX}{name}={raw!r} is not a valid {RULE_FIELDS[name].__name__}; ignored")

    parts = []
    for i, (asset, part_rules) in enumerate(ftype.parts):
        mine = overrides if i == 0 else {k: v for k, v in overrides.items() if k in SHARED_RULE_FIELDS}
        rules, matrix_warnings = validate_fence_rules(asset, {**ASSET_DEFAULT_RULES[asset], **part_rules, **mine})
        parts.append((asset, rules))
        warnings.extend(matrix_warnings)
    return FenceStyle(ftype.name, tuple(parts)), warnings


@dataclass(slots=True)
class FenceRecord:
    """One objectPath run, in the course-local [0, COURSE_SIZE_M] frame.
    `points` are (x, z); a closed run does NOT repeat its first point.
    `rules` holds every RULE_FIELDS key (game names). `source_ids` are
    the OSM way ids merged into this run, in path order.
    `corner_angle_deg` is frozen here (not read at write time) so
    write-objects stays a pure formatter; None = every interior waypoint
    smooth (see object_path_handles). `fence_type` is the FENCE_TYPES
    name the record is a part of (a multi-part type gives one record per
    part, same points)."""
    asset: str
    points: list[tuple[float, float]]
    closed: bool
    rules: dict
    source_ids: list[Optional[int]] = field(default_factory=list)
    corner_angle_deg: Optional[float] = CORNER_ANGLE_DEG
    fence_type: Optional[str] = None


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clean_points(coords: Sequence, simplify_tol_m: float) -> tuple[list[tuple[float, float]], bool]:
    """(points, closed) for one run: duplicates dropped, Douglas-Peucker
    simplified (closed runs simplified as a ring so the seam isn't
    pinned), closing point removed."""
    from shapely.geometry import LinearRing, LineString

    pts: list[tuple[float, float]] = []
    for x, z in coords:
        p = (float(x), float(z))
        if not pts or _dist(pts[-1], p) >= FENCE_MIN_SEGMENT_M:
            pts.append(p)
    closed = len(pts) >= 4 and _dist(pts[0], pts[-1]) <= FENCE_ENDPOINT_TOL_M
    if closed:
        pts = pts[:-1]
        if simplify_tol_m > 0:
            ring = LinearRing(pts).simplify(simplify_tol_m)
            simplified = list(ring.coords)[:-1]
            if len(simplified) >= 3:
                pts = simplified
    elif simplify_tol_m > 0 and len(pts) > 2:
        pts = list(LineString(pts).simplify(simplify_tol_m).coords)
    return [(float(x), float(z)) for x, z in pts], closed


def _merge_runs(runs: list[tuple[list, list]], tol_m: float) -> list[tuple[list, list]]:
    """
    Join same-style runs end-to-end where exactly two run ends meet
    (within tol_m) -- one objectPath per continuous fence, so a fence
    drawn as several OSM ways gets one post at each shared node, not two,
    and no gap. A node where 3+ ends meet (a T-junction) is left as
    separate runs: there's no single path through it. A run whose two
    ends meet is already a loop and is never joined to another.
    `runs` = [(coords, source_ids)], all the same style.
    """
    runs = [(list(c), list(s)) for c, s in runs]

    def degree(p) -> int:
        return sum((_dist(c[0], p) <= tol_m) + (_dist(c[-1], p) <= tol_m) for c, _ in runs)

    merged = True
    while merged:
        merged = False
        for i in range(len(runs)):
            ci, si = runs[i]
            if _dist(ci[0], ci[-1]) <= tol_m:
                continue
            for j in range(len(runs)):
                if j == i:
                    continue
                cj, sj = runs[j]
                if _dist(cj[0], cj[-1]) <= tol_m:
                    continue
                # Orient so ci's tail meets cj's head.
                for a, b in ((ci, cj), (ci, cj[::-1]), (ci[::-1], cj), (ci[::-1], cj[::-1])):
                    if _dist(a[-1], b[0]) <= tol_m and degree(a[-1]) == 2:
                        sa = si if a is ci else si[::-1]
                        sb = sj if b is cj else sj[::-1]
                        runs[i] = (a + b[1:], sa + sb)
                        del runs[j]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return runs


def build_fence_records(
    features: Sequence,
    *,
    simplify_tol_m: float = FENCE_SIMPLIFY_TOL_M,
    endpoint_tol_m: float = FENCE_ENDPOINT_TOL_M,
    corner_angle_deg: Optional[float] = CORNER_ANGLE_DEG,
) -> tuple[list[FenceRecord], list[str]]:
    """
    FenceRecords from fence/wall/hedge Features (course-local frame --
    crop to the course first, see ingest.osm.crop_features). Ways with
    the same resolved style that share an end node are joined into one
    run (_merge_runs); a run whose ends meet becomes a closed objectPath.
    A multi-part type (picket, railing walls) gives one record per part
    over the same points. Returns (records, warnings) -- warnings are
    per-way style problems (unknown preset, option outside the asset's
    matrix, ...).
    """
    by_style: dict[tuple, tuple[FenceStyle, list]] = {}
    warnings: list[str] = []
    for f in features:
        if f.kind not in FENCE_KINDS or f.geometry.geom_type != "LineString":
            continue
        style, way_warnings = resolve_fence_style(f.kind, f.tags)
        warnings.extend(f"way {f.osm_id}: {w}" for w in way_warnings)
        by_style.setdefault(style.key, (style, []))[1].append(
            ([(float(x), float(z)) for x, z in f.geometry.coords], [f.osm_id]))

    records: list[FenceRecord] = []
    for style, runs in by_style.values():
        for coords, source_ids in _merge_runs(runs, endpoint_tol_m):
            points, closed = _clean_points(coords, simplify_tol_m)
            if len(points) < 2 or (closed and len(points) < 3):
                continue
            for asset, rules in style.parts:
                records.append(FenceRecord(asset=asset, points=list(points), closed=closed,
                                           rules=dict(rules), source_ids=list(source_ids),
                                           corner_angle_deg=corner_angle_deg, fence_type=style.type_name))
    return records, warnings


def _turn_deg(prev, cur, nxt) -> float:
    a = math.atan2(cur[1] - prev[1], cur[0] - prev[0])
    b = math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])
    return abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))


def object_path_handles(
    points: Sequence[tuple[float, float]], closed: bool, has_curves: bool,
    corner_angle_deg: Optional[float] = CORNER_ANGLE_DEG,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(pointOne, pointTwo) per waypoint -- see the module docstring's
    HANDLES section. corner_angle_deg=None never treats a waypoint as a
    corner (every interior waypoint smooth, as the game's own editor
    writes them)."""
    n = len(points)
    out = []
    for i, p in enumerate(points):
        has_prev = closed or i > 0
        has_next = closed or i < n - 1
        prev = points[(i - 1) % n]
        nxt = points[(i + 1) % n]
        if not has_prev:
            out.append((p, (p[0] + (nxt[0] - p[0]) * ENDPOINT_HANDLE_RATIO,
                            p[1] + (nxt[1] - p[1]) * ENDPOINT_HANDLE_RATIO)))
            continue
        if not has_next:
            out.append(((p[0] - (p[0] - prev[0]) * ENDPOINT_HANDLE_RATIO,
                         p[1] - (p[1] - prev[1]) * ENDPOINT_HANDLE_RATIO), p))
            continue
        corner = not has_curves or (corner_angle_deg is not None
                                    and _turn_deg(prev, p, nxt) > corner_angle_deg)
        if corner:
            out.append(((p[0] - (p[0] - prev[0]) * ENDPOINT_HANDLE_RATIO,
                         p[1] - (p[1] - prev[1]) * ENDPOINT_HANDLE_RATIO),
                        (p[0] + (nxt[0] - p[0]) * ENDPOINT_HANDLE_RATIO,
                         p[1] + (nxt[1] - p[1]) * ENDPOINT_HANDLE_RATIO)))
            continue
        tx, tz = nxt[0] - prev[0], nxt[1] - prev[1]
        tlen = math.hypot(tx, tz) or 1.0
        tx, tz = tx / tlen, tz / tlen
        back = _dist(p, prev) * SMOOTH_HANDLE_RATIO
        fwd = _dist(p, nxt) * SMOOTH_HANDLE_RATIO
        out.append(((p[0] - tx * back, p[1] - tz * back), (p[0] + tx * fwd, p[1] + tz * fwd)))
    return out


def fence_record_to_object_path_v2023(record: FenceRecord | dict) -> dict:
    """One placedObjects3 objectPaths[] entry, key order as the game writes
    it. Points shift into the game's origin-centred grid; course z lands in
    the waypoint's "y". `state` 1 = closed loop (inferred: the sample's only
    state=1 row is its only closed run -- V2023_SCHEMA.md), 0 = open."""
    r = record if isinstance(record, FenceRecord) else load_fence_record(record)
    rules = r.rules
    shifted = [(x - GRID_ORIGIN_OFFSET, z - GRID_ORIGIN_OFFSET) for x, z in r.points]
    handles = object_path_handles(shifted, r.closed, rules["hasCurves"], r.corner_angle_deg)

    def pt(p) -> dict:
        return {"x": _round(p[0]), "y": _round(p[1])}

    return {
        "path": {
            "waypoints": [{"pointOne": pt(h1), "pointTwo": pt(h2), "waypoint": pt(p)}
                          for p, (h1, h2) in zip(shifted, handles)],
            "width": float(rules["width"]),
            "hasCurves": bool(rules["hasCurves"]),
            "state": 1 if r.closed else 0,
        },
        "height": float(rules["height"]),
        "spacingRule": int(rules["spacingRule"]),
        "flexibilityRule": int(rules["flexibilityRule"]),
        "heightRule": int(rules["heightRule"]),
        "spacing": float(rules["spacing"]),
    }


def fence_records_to_groups_v2023(records: Sequence[FenceRecord | dict]) -> list[dict]:
    """placedObjects3 groups, one per fence asset, carrying only
    objectPaths -- merge with the rest via objects.merge_object_groups,
    then placed_object_groups_to_v2023 adds the full v2023 envelope."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        r = rec if isinstance(rec, FenceRecord) else load_fence_record(rec)
        groups.setdefault(r.asset, []).append(fence_record_to_object_path_v2023(r))
    return [{"Key": {"path": asset}, "Value": {"objectPaths": paths}} for asset, paths in groups.items()]


LEVELED_SAMPLE_STEP_M = 1.0   # ground sampling pitch along a leveled run


def apply_leveled_heights(
    records: Sequence[FenceRecord | dict], height_at, height_shift_m: float = 0.0,
) -> list[FenceRecord]:
    """
    Resolve leveled (heightRule=1) runs to the game's absolute height. A
    leveled objectPath has ONE height for the whole run, and the game's
    editor sets it to the run's MINIMUM ground height (Andy, in-game,
    2026-09-27). fences.json keeps `height` as an offset in both modes, so
    a leveled record gets min(height_at along the run) + height_shift_m +
    its offset. Ground is sampled every LEVELED_SAMPLE_STEP_M along each
    segment (plus the wrap segment of a closed run), so a dip between
    waypoints still counts. `height_at(x, z)` is the terrain height in the
    pre-shift course-local frame (TerrainModel.evaluate); height_shift_m
    is project.json's output_height_shift_m. Contoured records pass
    through; returns copies, never mutates the input.
    """
    out: list[FenceRecord] = []
    for rec in records:
        r = rec if isinstance(rec, FenceRecord) else load_fence_record(rec)
        r = load_fence_record(asdict(r))
        if int(r.rules["heightRule"]) == 1 and r.points:
            pts = r.points + ([r.points[0]] if r.closed else [])
            ground = [height_at(*pts[0])]
            for a, b in zip(pts, pts[1:]):
                n = max(1, math.ceil(_dist(a, b) / LEVELED_SAMPLE_STEP_M))
                ground += [height_at(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
                           for i in range(1, n + 1)]
            r.rules["height"] = _round(min(ground) + height_shift_m + float(r.rules["height"]))
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# In-game test harness (push-fence-test): a blank course with every fence
# asset / rule variant / preset laid out in labelled rows at one corner, so
# a single in-game load checks them all without hunting through a full
# course. Coordinates below are the GAME frame (origin-centred, "y" =
# course z, as objectPath waypoints are written).
# ---------------------------------------------------------------------------

FENCE_TEST_ORIGIN = (-900.0, 900.0)    # NW area (game frame); the grid is only ±GRID_ORIGIN_OFFSET
FENCE_TEST_EDGE_MARGIN_M = 10.0        # everything must stay this far inside the grid edge
FENCE_TEST_ROW_PITCH_M = 10.0
FENCE_TEST_RUN_M = 20.0
FENCE_TEST_PAD_HALF_M = 20.0           # raised pad: square half-extent
FENCE_TEST_PAD_LIFT_M = 15.0           # pad height above the template datum
FENCE_TEST_PAD_BRUSH = 72              # square brush -- the template's own map-wide datum stamp type


@dataclass(slots=True)
class FenceTestLayout:
    """What build_fence_test_layout lays out. `records` go through the normal
    fence_records_to_groups_v2023 path; `control_groups` are the sample's own
    game-written objectPaths (translated only) and bypass the builder;
    `pad_center`/`pad_half` (game frame) + `pad_lift` describe the raised
    flatten pad the burial check sits on; `legend` is one line per row;
    `labels[i]` is records[i]'s row id ("B7", "D1", ...)."""
    records: list[FenceRecord]
    labels: list[str]
    control_groups: list[dict]
    pad_center: tuple[float, float]
    pad_half: float
    pad_lift: float
    legend: list[str]


def _translate_object_path(op: dict, dx: float, dy: float) -> dict:
    out = json.loads(json.dumps(op))
    for wp in out["path"]["waypoints"]:
        for key in ("pointOne", "pointTwo", "waypoint"):
            wp[key] = {"x": _round(wp[key]["x"] + dx), "y": _round(wp[key]["y"] + dy)}
    return out


def build_fence_test_layout(
    sample_groups: Sequence[dict], origin: tuple[float, float] = FENCE_TEST_ORIGIN, datum: float = 0.0,
) -> FenceTestLayout:
    """
    The push-fence-test layout, growing from `origin` (game frame) toward
    +x / -y (east / south of the NW corner). Five blocks, left to right:

      A. control (x0+5..): the sample course's own objectPaths
         (sample_groups = templates/2023_fences.course placedObjects3),
         translated as one block -- if these render and B doesn't, the
         fault is in our records, not placement.
      B. ours (x0+40..x0+60): one 20 m run per row, 10 m apart --
         every asset at its defaults, rule variants, the trick presets.
      C. orientation L (x0+80): UniFence, short arm +y (15 m), long arm +x
         (40 m). Shows whether waypoint "y" is mirrored.
      D. burial check: a raised flatten pad (pad_lift above the template
         datum) centred at (x0+160, y0-60); a run on the pad, the same run
         with height=+pad_lift, and one crossing the pad's edges. If the
         game grounds objectPaths on the base terrain rather than the
         userLayers stamps, the first vanishes and the second sits on top.
         (Answered 2026-09-26: fences follow the stamps.)
      E. open questions from the first in-game run (x0+230..): stepped
         BrickWall with height = `datum` and datum+3 (is heightRule=1's
         height absolute? B13 at height 0 didn't render), WoodFences
         spacingRule 1 vs 3 on a 4-waypoint zigzag (caps at every waypoint
         vs ends only -- identical on B's 2-point runs), BrickWall
         hasCurves true vs false on a gentle 3-waypoint bend.
    """
    x0, y0 = origin
    legend: list[str] = []

    # --- A. control rows ---
    paths = [(g["Key"]["path"], op) for g in sample_groups for op in (g.get("Value") or {}).get("objectPaths") or []]
    control_groups: list[dict] = []
    if paths:
        xs = [wp["waypoint"]["x"] for _, op in paths for wp in op["path"]["waypoints"]]
        ys = [wp["waypoint"]["y"] for _, op in paths for wp in op["path"]["waypoints"]]
        dx, dy = (x0 + 5.0) - min(xs), (y0 - 5.0) - max(ys)
        by_asset: dict[str, list[dict]] = {}
        for asset, op in paths:
            by_asset.setdefault(asset, []).append(_translate_object_path(op, dx, dy))
        control_groups = [{"Key": {"path": a}, "Value": {"objectPaths": ops}} for a, ops in by_asset.items()]
        legend.append(f"A  control: {len(paths)} sample objectPaths (templates/2023_fences.course), "
                      f"x {min(xs) + dx:.0f}..{max(xs) + dx:.0f}, y {min(ys) + dy:.0f}..{max(ys) + dy:.0f}")

    # --- B. our rows ---
    rows: list[tuple[str, str, dict]] = []
    for asset in ASSET_DEFAULT_RULES:
        rows.append((f"{asset.rsplit('/', 1)[-1]} defaults", asset, {}))
    for sr in range(4):
        rows.append((f"BrickWall spacingRule={sr}", BRICK_WALL_ASSET, {"spacingRule": sr}))
    rows.append(("BrickWall heightRule=1 (stepped)", BRICK_WALL_ASSET, {"heightRule": 1}))
    rows.append(("BrickWall hasCurves=false", BRICK_WALL_ASSET, {"hasCurves": False}))
    for sr in (0, 1, 3):
        rows.append((f"WoodFences spacingRule={sr} (unverified)", WOOD_FENCE_ASSET, {"spacingRule": sr}))
    for name, (type_name, overrides) in FENCE_PRESETS.items():
        rows.append((f"preset {name}", FENCE_TYPES[type_name].parts[0][0], dict(overrides)))

    records: list[FenceRecord] = []
    labels: list[str] = []

    def add(label: str, asset: str, overrides: dict, pts: list[tuple[float, float]]) -> None:
        rules, warnings = validate_fence_rules(asset, {**ASSET_DEFAULT_RULES[asset], **overrides})
        local = [(x + GRID_ORIGIN_OFFSET, y + GRID_ORIGIN_OFFSET) for x, y in pts]
        records.append(FenceRecord(asset=asset, points=local, closed=False, rules=rules))
        labels.append(label.split()[0])
        note = f"  [{'; '.join(warnings)}]" if warnings else ""
        (x_a, y_a), (x_b, y_b) = pts[0], pts[-1]
        legend.append(f"{label}: ({x_a:.0f},{y_a:.0f}) -> ({x_b:.0f},{y_b:.0f}){note}")

    for i, (label, asset, overrides) in enumerate(rows):
        y = y0 - FENCE_TEST_ROW_PITCH_M * (i + 1)
        add(f"B{i + 1:<2} {label}", asset, overrides, [(x0 + 40.0, y), (x0 + 40.0 + FENCE_TEST_RUN_M, y)])

    # --- C. orientation marker ---
    cx, cy = x0 + 80.0, y0 - 40.0
    add("C  orientation L (short arm +y, long arm +x)", UNI_FENCE_ASSET, {},
        [(cx, cy + 15.0), (cx, cy), (cx + 40.0, cy)])

    # --- D. burial check ---
    px, py = x0 + 160.0, y0 - 60.0
    h = FENCE_TEST_PAD_HALF_M
    legend.append(f"D  raised pad: +{FENCE_TEST_PAD_LIFT_M:g} m over datum, "
                  f"x {px - h:.0f}..{px + h:.0f}, y {py - h:.0f}..{py + h:.0f}")
    add("D1 WoodFences on pad", WOOD_FENCE_ASSET, {}, [(px - 15.0, py + 10.0), (px + 15.0, py + 10.0)])
    add(f"D2 WoodFences on pad, height=+{FENCE_TEST_PAD_LIFT_M:g}", WOOD_FENCE_ASSET,
        {"height": FENCE_TEST_PAD_LIFT_M}, [(px - 15.0, py), (px + 15.0, py)])
    add("D3 WoodFences crossing pad edges", WOOD_FENCE_ASSET, {},
        [(px - h - 20.0, py - 10.0), (px + h + 20.0, py - 10.0)])

    # --- E. follow-up questions ---
    ex = x0 + 230.0
    zigzag = lambda y: [(ex, y), (ex + 10.0, y - 6.0), (ex + 20.0, y), (ex + 30.0, y - 6.0)]  # noqa: E731
    bend = lambda y: [(ex, y), (ex + 15.0, y - 4.0), (ex + 30.0, y)]  # noqa: E731
    add(f"E1 BrickWall stepped, height={datum:g} (= datum)", BRICK_WALL_ASSET,
        {"heightRule": 1, "height": round(datum, 3)}, [(ex, y0 - 10.0), (ex + FENCE_TEST_RUN_M, y0 - 10.0)])
    add(f"E2 BrickWall stepped, height={datum + 3:g} (datum+3)", BRICK_WALL_ASSET,
        {"heightRule": 1, "height": round(datum + 3.0, 3)}, [(ex, y0 - 25.0), (ex + FENCE_TEST_RUN_M, y0 - 25.0)])
    add("E3 WoodFences spacingRule=1, zigzag", WOOD_FENCE_ASSET, {"spacingRule": 1}, zigzag(y0 - 40.0))
    add("E4 WoodFences spacingRule=3, zigzag", WOOD_FENCE_ASSET, {"spacingRule": 3}, zigzag(y0 - 55.0))
    add("E5 BrickWall hasCurves=true, bend", BRICK_WALL_ASSET, {"hasCurves": True}, bend(y0 - 70.0))
    add("E6 BrickWall hasCurves=false, bend", BRICK_WALL_ASSET, {"hasCurves": False}, bend(y0 - 85.0))

    # Off-map guard: (-1900, 1900) once built a course with nothing visible
    # at all -- the grid is ±GRID_ORIGIN_OFFSET, not ±2 * that.
    limit = GRID_ORIGIN_OFFSET - FENCE_TEST_EDGE_MARGIN_M
    pts = [(p[k]["x"], p[k]["y"]) for g in fence_records_to_groups_v2023(records) + control_groups
           for op in g["Value"]["objectPaths"] for p in op["path"]["waypoints"]
           for k in ("pointOne", "pointTwo", "waypoint")]
    pts += [(px - h, py - h), (px + h, py + h)]
    worst = max(max(abs(x), abs(y)) for x, y in pts)
    if worst > limit:
        raise ValueError(f"fence test layout from origin ({x0:g}, {y0:g}) reaches {worst:.0f} m from "
                         f"the centre; the course grid is only ±{GRID_ORIGIN_OFFSET:g} m (keep within "
                         f"±{limit:g}). Pick an origin nearer the centre.")

    return FenceTestLayout(records=records, labels=labels, control_groups=control_groups, pad_center=(px, py),
                           pad_half=h, pad_lift=FENCE_TEST_PAD_LIFT_M, legend=legend)


def load_fence_record(d: dict) -> FenceRecord:
    return FenceRecord(asset=d["asset"], points=[tuple(p) for p in d["points"]], closed=d["closed"],
                       rules=dict(d["rules"]), source_ids=list(d.get("source_ids", [])),
                       corner_angle_deg=d.get("corner_angle_deg", CORNER_ANGLE_DEG),
                       fence_type=d.get("fence_type"))


def save_fence_records(records: Sequence[FenceRecord], path: Path) -> None:
    """Write fences.json -- a plain JSON list, same convention as oob.json / parking.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in records], fh, indent=2)


def load_fence_records(path: Path) -> list[FenceRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [load_fence_record(d) for d in json.load(fh)]
