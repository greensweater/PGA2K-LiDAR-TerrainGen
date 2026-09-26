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

ROUTING: only the post prefabs the sample course proved as objectPath
materials (asset_catalog.FENCE_ENTRIES -- the entries carrying
fence_options) are valid targets. The rest of the Walls/** catalog
(panel / post B-D variants, TriFence, PicketFence, ...) are placed-item
pieces, not objectPath materials, and are never routed to here.

Precedence, first match wins:
  1. wall=retaining_wall / barrier=retaining_wall -> retaining wall
  2. hedge kind (barrier=hedge / natural=hedge)   -> hedge
  3. material=* (then wall=*, then fence_type=*) through _MATERIAL_ASSETS
  4. barrier=chain                                 -> UniFence
  5. the kind's default (_KIND_DEFAULT_ASSETS)
The canvas fences have no OSM tag that reaches them -- they're for the
trick presets (FENCE_PRESETS) and per-way overrides.

PER-WAY OVERRIDES are this project's own tags on the Feature (what the
GUI's Objects / Fences panel writes; ingest-osm carries them over a
re-ingest -- FENCE_STYLE_TAGS): pga_fence_preset=<FENCE_PRESETS name>,
pga_fence_asset=<asset label or path>, and pga_fence_<rule field> for
each field in RULE_FIELDS (e.g. pga_fence_spacingRule=3,
pga_fence_height=-1.5). Precedence: asset defaults < preset < explicit
field tags. Rule values are then checked against the asset's option
matrix (validate_fence_rules).

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
UNI_FENCE_ASSET = _WALLS + "UniFencePostAPrefab"
HEDGE_ASSET = _WALLS + "HedgeSplinePostPrefab"
BRICK_WALL_ASSET = _WALLS + "Asia_Walls/Asia_BrickWalls_PostPrefab"
RETAINING_WALL_ASSET = _WALLS + "RetainWallAPostAPrefab"
CANVAS_RED_ASSET = _WALLS + "CanvasFence01AARedPostAPrefab"
CANVAS_BLACK_ASSET = _WALLS + "CanvasFence01AABlackPostAPrefab"

FENCE_KINDS = ("fence", "wall", "hedge")

_KIND_DEFAULT_ASSETS = {
    "fence": WOOD_FENCE_ASSET,
    "wall": STONE_WALL_ASSET,
    "hedge": HEDGE_ASSET,
}

# OSM material / wall / fence_type values -> asset. One table for all
# three keys: the vocabularies overlap (wall=brick, material=brick) and
# don't collide.
_MATERIAL_ASSETS = {
    "stone": STONE_WALL_ASSET,
    "dry_stone": STONE_WALL_ASSET,
    "flint": STONE_WALL_ASSET,
    "brick": BRICK_WALL_ASSET,
    "wood": WOOD_FENCE_ASSET,
    "split_rail": WOOD_FENCE_ASSET,
    "palisade": WOOD_FENCE_ASSET,
    "pole": WOOD_FENCE_ASSET,
    "rail": WOOD_FENCE_ASSET,
    "chain_link": UNI_FENCE_ASSET,
    "metal": UNI_FENCE_ASSET,
    "steel": UNI_FENCE_ASSET,
    "wire": UNI_FENCE_ASSET,
    "barbed_wire": UNI_FENCE_ASSET,
    "electric": UNI_FENCE_ASSET,
    "mesh": UNI_FENCE_ASSET,
    "metal_bars": UNI_FENCE_ASSET,
    "railing": UNI_FENCE_ASSET,
}

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


# Per-asset defaults = that asset's plain (contoured, unburied) sample row.
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
}

# Sample-proven "trick" recipes (V2023_SCHEMA.md "Tricks"): an asset plus
# rule overrides on top of that asset's defaults.
FENCE_PRESETS: dict[str, tuple[str, dict]] = {
    "curb": (STONE_WALL_ASSET, {"height": -1.498, "spacingRule": 0}),
    "railroad": (CANVAS_BLACK_ASSET, {"height": -2.688, "spacingRule": 2}),
    "retaining_wall": (RETAINING_WALL_ASSET, {"width": 2.5, "height": RETAINING_WALL_HEIGHT_M, "heightRule": 0}),
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
_unknown = ({RETAINING_WALL_ASSET, CANVAS_RED_ASSET, CANVAS_BLACK_ASSET}
            | set(_KIND_DEFAULT_ASSETS.values()) | set(_MATERIAL_ASSETS.values())
            | set(ASSET_DEFAULT_RULES)) - _VALID_ASSETS
assert not _unknown, f"fence routing targets missing from asset_catalog fence_options: {_unknown}"
assert set(ASSET_DEFAULT_RULES) == _VALID_ASSETS, "every fence asset needs ASSET_DEFAULT_RULES"


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def fence_asset_for_tags(kind: str, tags: dict) -> str:
    """The v2023 objectPath material (asset path) for a fence/wall/hedge
    Feature from its OSM tags alone -- see the module docstring for the
    precedence. Per-way overrides are resolve_fence_style's job."""
    if tags.get("wall") == "retaining_wall" or tags.get("barrier") == "retaining_wall":
        return RETAINING_WALL_ASSET
    if kind == "hedge":
        return HEDGE_ASSET
    for key in ("material", "wall", "fence_type"):
        asset = _MATERIAL_ASSETS.get(tags.get(key, ""))
        if asset is not None:
            return asset
    if tags.get("barrier") == "chain":
        return UNI_FENCE_ASSET
    return _KIND_DEFAULT_ASSETS[kind]


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


def resolve_fence_style(kind: str, tags: dict) -> tuple[str, dict, list[str]]:
    """(asset path, validated rule dict, warnings) for one fence Feature:
    tag routing, then preset, then pga_fence_<field> overrides, then the
    option-matrix check."""
    warnings: list[str] = []
    asset = fence_asset_for_tags(kind, tags)
    overrides: dict = {}

    preset = tags.get(PRESET_TAG)
    if preset:
        if preset in FENCE_PRESETS:
            asset, preset_rules = FENCE_PRESETS[preset]
            overrides.update(preset_rules)
        else:
            warnings.append(f"unknown {PRESET_TAG}={preset!r} (known: {', '.join(FENCE_PRESETS)})")

    requested = tags.get(ASSET_TAG)
    if requested:
        path = requested if requested in _VALID_ASSETS else _ASSET_BY_LABEL.get(requested)
        if path is None:
            warnings.append(f"{ASSET_TAG}={requested!r} is not an objectPath fence asset; keeping "
                            f"{asset.rsplit('/', 1)[-1]}")
        else:
            asset = path

    for name in RULE_FIELDS:
        raw = tags.get(RULE_TAG_PREFIX + name)
        if raw is None:
            continue
        try:
            overrides[name] = _parse_rule(name, raw)
        except ValueError:
            warnings.append(f"{RULE_TAG_PREFIX}{name}={raw!r} is not a valid {RULE_FIELDS[name].__name__}; ignored")

    rules, matrix_warnings = validate_fence_rules(asset, {**ASSET_DEFAULT_RULES[asset], **overrides})
    return asset, rules, warnings + matrix_warnings


@dataclass(slots=True)
class FenceRecord:
    """One objectPath run, in the course-local [0, COURSE_SIZE_M] frame.
    `points` are (x, z); a closed run does NOT repeat its first point.
    `rules` holds every RULE_FIELDS key (game names). `source_ids` are
    the OSM way ids merged into this run, in path order.
    `corner_angle_deg` is frozen here (not read at write time) so
    write-objects stays a pure formatter; None = every interior waypoint
    smooth (see object_path_handles)."""
    asset: str
    points: list[tuple[float, float]]
    closed: bool
    rules: dict
    source_ids: list[Optional[int]] = field(default_factory=list)
    corner_angle_deg: Optional[float] = CORNER_ANGLE_DEG


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
    Returns (records, warnings) -- warnings are per-way style problems
    (unknown preset, option outside the asset's matrix, ...).
    """
    by_style: dict[tuple, tuple[str, dict, list]] = {}
    warnings: list[str] = []
    for f in features:
        if f.kind not in FENCE_KINDS or f.geometry.geom_type != "LineString":
            continue
        asset, rules, way_warnings = resolve_fence_style(f.kind, f.tags)
        warnings.extend(f"way {f.osm_id}: {w}" for w in way_warnings)
        key = (asset, tuple(sorted(rules.items())))
        by_style.setdefault(key, (asset, rules, []))[2].append(
            ([(float(x), float(z)) for x, z in f.geometry.coords], [f.osm_id]))

    records: list[FenceRecord] = []
    for asset, rules, runs in by_style.values():
        for coords, source_ids in _merge_runs(runs, endpoint_tol_m):
            points, closed = _clean_points(coords, simplify_tol_m)
            if len(points) < 2 or (closed and len(points) < 3):
                continue
            records.append(FenceRecord(asset=asset, points=points, closed=closed,
                                       rules=dict(rules), source_ids=source_ids,
                                       corner_angle_deg=corner_angle_deg))
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
    for name, (asset, overrides) in FENCE_PRESETS.items():
        rows.append((f"preset {name}", asset, dict(overrides)))

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
                       corner_angle_deg=d.get("corner_angle_deg", CORNER_ANGLE_DEG))


def save_fence_records(records: Sequence[FenceRecord], path: Path) -> None:
    """Write fences.json -- a plain JSON list, same convention as oob.json / parking.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in records], fh, indent=2)


def load_fence_records(path: Path) -> list[FenceRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [load_fence_record(d) for d in json.load(fh)]
