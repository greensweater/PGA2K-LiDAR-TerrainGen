"""
course_output/collections.py

Resolve object *collection* templates (course_output/collection_library.py)
into concrete placements, and format those placements for each output.

Flow (see PGA2k_gen.py):
  step_generate_collections:  features.geojson pga_collection lines + library
                              -> collections.json   (frozen, course-local frame)
  step_pack_objects:          collections.json .objects -> objects.json
                              (kind="collection_object")
  step_write_objects:         -> placedObjects2.json groups (v2019 or v2021)
  step_write_splines:         collections.json .splines -> surfaceSplines.json

A collections.json record:

    {"source_id": int | None,          # the OSM way id of the placement line
     "name": str,                      # template name
     "parameter": str | None,          # the placement line's pga_parameter tag
     "x": float, "z": float,           # anchor, course-local metres
     "heading_deg": float,             # 0 = +Z, 90 = +X (see _bearing_deg)
     "objects": [{"x","z","rotation_deg","scale",
                  "category","type","theme","path",
                  "dy"?, "pitch"?, "roll"?}, ...],   # dy present => designed elevation;
                                                     # pitch/roll (deg) => rotation.x/z
                                                     # (used by course_output/parking.py's
                                                     # slope-banked parked cars, which ride
                                                     # this same collection_object path)
     "splines": [ <surfaceSplines.json spline dict, points in course-local frame> ],
     "stamps": [{"x","z","scale_x","scale_z","value","brush","rotation_deg","tool"}, ...]}

Every coordinate is in the course-local [0, COURSE_SIZE_M] frame -- the
object/spline writers apply the usual GRID_ORIGIN_OFFSET shift, same as
every other course_output writer.

PARAMETER: the placement way's optional `pga_parameter` OSM tag lets one
template adapt per placement without a separate template per case -- e.g.
a hole-marker collection whose sign swaps by hole number. An object
member carries an optional `"variants"` map (keyed by parameter value,
with a `"default"` fallback) whose chosen entry is overlaid on the base
member (any field: category/type/theme/path/scale/dx/dz/rotation_deg),
and/or a `"{param}"` token in its asset `path` that's substituted with
the parameter value, and/or a `"param_option"` mapping the parameter to
the game's per-item `options` variant selector. A member with none of
those is placed identically regardless of the parameter. Spline members
are always static.

A `"{param}"` token falls back to the member's optional `"param_default"`
when the placement way carries no `pga_parameter` tag; if that's also
absent the unresolved token survives and `step_generate_collections`
prints a WARNING naming the offending way (a forgotten tag would
otherwise write a broken asset path with no complaint).

`"param_option"` (e.g. `"OffsetIndex"`) is for signs: the ONE
`HoleSign0N`/`MessageSign`/`DirectionSign` prefab picks WHICH sign it
renders via `options.OffsetIndex` (0-based) -- the asset path only picks
the sign STYLE, never the number. The member keeps a fixed path and the
resolved `options[param_option] = int(pga_parameter or param_default) +
param_option_base` (base default 0; set `-1` for a 1-based OSM hole
number). An unresolvable param_option also draws a WARNING.

VERTICAL: object members carrying a `"dy"` (captured from a prop placed
at a designed elevation -- see collection_library.py's VERTICAL
GROUNDING) get an absolute `y` = target terrain height + dy at
write-objects time (apply_terrain_heights). Members with no `"dy"` keep
position.y = "-Infinity" and the game ground-snaps them.

STAMPS: stamp members (raised flowerbed beds, berms) are resolved into
the record's "stamps" list (rotated/translated like objects) and turned
into terrain.stamp.Stamp objects by build_collection_stamps, which
PGA2k_gen.py's _load_normalized_stamps folds into the stamp list every
write-terrain / write-water run -- NOT persisted as a stamps_N.json
layer, so re-running generate-collections never double-applies them.
All are raise-tool (relative); capture never records a flatten.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from course_output.asset_catalog import ASSET_ENTRIES
from course_output.collection_library import Collection
from course_output.objects import _placed_item, _placed_object_group_v2021
from course_output.userLayers import GRID_ORIGIN_OFFSET
from terrain.stamp import TOOL_RAISE, Stamp

_DECIMALS = 3

_ENTRY_BY_KEY = {(e.category, e.type): e for e in ASSET_ENTRIES}
_ENTRY_BY_PATH = {e.path: e for e in ASSET_ENTRIES}


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def _bearing_deg(dx: float, dz: float) -> float:
    """Compass bearing of a direction vector -- 0 = +Z, 90 = +X, sweeping
    clockwise. Same convention as terrain/streams.py, course_output/water.py,
    terrain/cart_paths.py."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _rotate(dx: float, dz: float, heading_deg: float) -> tuple[float, float]:
    """Rotate a local offset so that local +Z (forward) points along
    `heading_deg`. With the (sin, cos) heading vector this module shares
    with streams.py, a local (0, d) maps to (d*sinH, d*cosH)."""
    h = math.radians(heading_deg)
    cos_h, sin_h = math.cos(h), math.sin(h)
    return dx * cos_h + dz * sin_h, -dx * sin_h + dz * cos_h


def _resolve_member(member: dict, parameter: Optional[str]) -> dict:
    """Apply a `pga_parameter` value to one object member (see the module
    docstring's PARAMETER section): overlay the matching entry from the
    member's optional "variants" map (keyed by the parameter value, with
    a "default" fallback) onto the base member, substitute a "{param}"
    token in the asset path, and/or map the parameter onto a per-item
    "options" variant selector. A member with none of those comes back
    unchanged.

    A "{param}" token is filled from the placement way's `pga_parameter`
    tag; if the way has no such tag the member's optional "param_default"
    is used instead. With neither, the literal "{param}" is left in the
    path -- step_generate_collections warns about that (a forgotten tag
    would otherwise ship a broken asset path silently).

    "param_option" (a string, e.g. "OffsetIndex") maps the parameter to
    the game's per-item `options` variant selector instead of the path:
    the ONE hole/message/direction-sign prefab picks WHICH sign it shows
    via `options.OffsetIndex` (0-based), the asset path only picks the
    sign style. The stored value is int(parameter or param_default) +
    "param_option_base" (default 0 -- set it to -1 for a 1-based
    `pga_parameter` like an OSM hole number). If neither a tag nor a
    param_default gives an integer, "options" is left off and
    "options_unresolved" flags it for step_generate_collections to warn.

    "param_default", "param_option", "param_option_base" are all stripped
    from the returned member."""
    effective = dict(member)
    variants = member.get("variants")
    if isinstance(variants, dict):
        chosen = None
        if parameter is not None and str(parameter) in variants:
            chosen = variants[str(parameter)]
        elif "default" in variants:
            chosen = variants["default"]
        if isinstance(chosen, dict):
            effective.update(chosen)
    effective.pop("variants", None)

    token = parameter if parameter is not None else effective.get("param_default")

    path = effective.get("path")
    if path and "{param}" in path and token is not None:
        effective["path"] = path.replace("{param}", str(token))

    option_key = effective.get("param_option")
    if isinstance(option_key, str) and option_key:
        try:
            base = int(effective.get("param_option_base", 0))
            effective["options"] = {option_key: int(str(token)) + base}
        except (TypeError, ValueError):
            effective["options_unresolved"] = option_key

    for k in ("param_default", "param_option", "param_option_base"):
        effective.pop(k, None)
    return effective


def resolve_collection(
    template: Collection, anchor_x: float, anchor_z: float, heading_deg: float,
    source_id: Optional[int] = None, parameter: Optional[str] = None,
) -> dict:
    """One collections.json record: every template member rotated by
    `heading_deg` about the anchor and translated to (anchor_x, anchor_z),
    all in the course-local frame. `parameter` is the placement way's
    `pga_parameter` OSM tag -- see _resolve_member / the module docstring."""
    objects: list[dict] = []
    for raw in template.objects:
        m = _resolve_member(raw, parameter)
        rx, rz = _rotate(m.get("dx", 0.0), m.get("dz", 0.0), heading_deg)
        obj = {
            "x": _round(anchor_x + rx),
            "z": _round(anchor_z + rz),
            "rotation_deg": _round((m.get("rotation_deg", 0.0) + heading_deg) % 360.0),
            "scale": _round(m.get("scale", 1.0)),
            "category": m.get("category"),
            "type": m.get("type"),
            "theme": m.get("theme"),
            "path": m.get("path"),
        }
        # `dy` (height above the source course's flatten datum) rides
        # through unrotated -- heading is a yaw, it doesn't touch
        # elevation. step_write_objects turns it into an absolute y
        # against the target terrain; a member with no `dy` keeps
        # position.y = "-Infinity" (game ground-snap).
        if m.get("dy") is not None:
            obj["dy"] = _round(m["dy"])
        # Per-item variant selector (e.g. a sign's options.OffsetIndex --
        # see _resolve_member's "param_option"). Rides through unrotated;
        # write-objects hands it to objects._placed_item verbatim.
        if m.get("options") is not None:
            obj["options"] = m["options"]
        if m.get("options_unresolved"):
            obj["options_unresolved"] = m["options_unresolved"]
        objects.append(obj)

    splines: list[dict] = []
    for spline in template.splines:
        placed = dict(spline)
        new_waypoints = []
        for wp in spline.get("waypoints", []):
            new_wp = {}
            for part, pt in wp.items():
                rx, rz = _rotate(pt.get("x", 0.0), pt.get("y", 0.0), heading_deg)
                new_wp[part] = {"x": _round(anchor_x + rx), "y": _round(anchor_z + rz)}
            new_waypoints.append(new_wp)
        placed["waypoints"] = new_waypoints
        splines.append(placed)

    stamps: list[dict] = []
    for s in template.stamps:
        rx, rz = _rotate(s.get("dx", 0.0), s.get("dz", 0.0), heading_deg)
        stamps.append({
            "x": _round(anchor_x + rx),
            "z": _round(anchor_z + rz),
            "scale_x": _round(s.get("scale_x", 1.0)),
            "scale_z": _round(s.get("scale_z", 1.0)),
            "value": _round(s.get("value", 0.0)),
            "brush": s.get("brush"),
            "rotation_deg": _round((s.get("rotation_deg", 0.0) + heading_deg) % 360.0),
            "tool": TOOL_RAISE,  # capture only ever records raise; enforce it here too
        })

    return {
        "source_id": source_id,
        "name": template.name,
        "parameter": parameter,
        "x": _round(anchor_x),
        "z": _round(anchor_z),
        "heading_deg": _round(heading_deg % 360.0),
        "objects": objects,
        "splines": splines,
        "stamps": stamps,
    }


def save_collection_records(records: list[dict], path: Path) -> None:
    """Write collections.json -- a plain JSON list, same convention as
    course_output/objects.py's objects.json / terrain/streams.py's
    streams.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def load_collection_records(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def iter_collection_objects(records: list[dict]):
    """Every resolved object placement across every collection record,
    each tagged with its record's source_id -- the flat list
    step_pack_objects folds into objects.json."""
    for record in records:
        for obj in record.get("objects", []):
            yield dict(obj, source_id=record.get("source_id"))


# ---------------------------------------------------------------------------
# write-objects formatting
# ---------------------------------------------------------------------------


def apply_terrain_heights(objects: list[dict], height_at, height_shift_m: float = 0.0) -> list[dict]:
    """For every collection object carrying a `dy` (it had a designed
    scalar y in the source course, recorded as a height above that
    course's flatten datum -- see collection_library.py), set an
    absolute `y` = height_at(x, z) + height_shift_m + dy, so the prop
    sits the same distance above local ground on the target course.
    `height_at(x, z)` is the target terrain height in the pre-shift
    frame (e.g. terrain.terrain_model.TerrainModel.evaluate over the raw
    stamp list); height_shift_m is project.json's output_height_shift_m
    (write-terrain's normalization). Objects with no `dy` are left
    untouched -- they keep position.y = "-Infinity" (game ground-snap).
    Mutates and returns `objects`."""
    for obj in objects:
        dy = obj.get("dy")
        if dy is None:
            continue
        obj["y"] = round(height_at(obj["x"], obj["z"]) + height_shift_m + dy, 3)
    return objects


def _resolve_v2019_key(obj: dict) -> Optional[tuple[int, int, bool]]:
    """(category, type, theme) for a collection object -- straight from
    the record if it was captured from a v2019 course, else reverse-
    resolved from its asset path via the catalog. None (warn + skip) if
    neither is available."""
    if obj.get("category") is not None and obj.get("type") is not None:
        return obj["category"], obj["type"], bool(obj.get("theme"))
    entry = _ENTRY_BY_PATH.get(obj.get("path"))
    if entry is not None:
        return entry.category, entry.type, entry.theme
    return None


def _resolve_v2021_path(obj: dict) -> Optional[str]:
    """Asset path for a collection object -- straight from the record if
    captured from a v2021 course, else forward-resolved from
    (category, type) via the catalog. None (warn + skip) otherwise."""
    if obj.get("path"):
        return obj["path"]
    entry = _ENTRY_BY_KEY.get((obj.get("category"), obj.get("type")))
    return entry.path if entry is not None else None


def build_collection_objects_v2019(objects: list[dict]) -> list[dict]:
    """placedObjects2 v2019 groups (Key {category,type,theme}) from
    resolved collection objects -- one group per distinct key, items via
    objects._placed_item. Unresolvable entries are skipped with a note."""
    groups: dict[tuple[int, int, bool], dict] = {}
    skipped = 0
    for obj in objects:
        key = _resolve_v2019_key(obj)
        if key is None:
            skipped += 1
            continue
        group = groups.setdefault(key, {
            "Key": {"category": key[0], "type": key[1], "theme": key[2]},
            "Value": {"items": [], "clusters": []},
        })
        group["Value"]["items"].append(_placed_item(
            obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0),
            y=obj.get("y", "-Infinity"), options=obj.get("options"),
            pitch_degrees=obj.get("pitch", 0.0), roll_degrees=obj.get("roll", 0.0),
        ))
    if skipped:
        print(f"  NOTE: {skipped} collection object(s) skipped -- no v2019 category/type "
              "(captured from a v2021 course whose asset path isn't in asset_catalog.json)")
    return list(groups.values())


def build_collection_objects_v2021(objects: list[dict]) -> list[dict]:
    """placedObjects2 v2021+ groups (Key {path}) from resolved collection
    objects -- one group per distinct asset path. Unresolvable entries
    are skipped with a note."""
    groups: dict[str, dict] = {}
    skipped = 0
    for obj in objects:
        path = _resolve_v2021_path(obj)
        if not path:
            skipped += 1
            continue
        group = groups.setdefault(path, _placed_object_group_v2021(path))
        group["Value"]["items"].append(_placed_item(
            obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0),
            y=obj.get("y", "-Infinity"), options=obj.get("options"),
            pitch_degrees=obj.get("pitch", 0.0), roll_degrees=obj.get("roll", 0.0),
        ))
    if skipped:
        print(f"  NOTE: {skipped} collection object(s) skipped -- no v2021 asset path "
              "(captured from a v2019 course whose category/type isn't in asset_catalog.json)")
    return list(groups.values())


# ---------------------------------------------------------------------------
# write-splines formatting
# ---------------------------------------------------------------------------


def build_collection_splines(records: list[dict]) -> list[dict]:
    """surfaceSplines.json spline dicts for every spline member across
    every collection record -- the resolved course-local points shifted
    into the game's origin-centred grid (GRID_ORIGIN_OFFSET), same shift
    course_output/splines.py applies. The spline shape (waypoints,
    handles, surface, width, ...) is carried through verbatim from
    capture."""
    out: list[dict] = []
    for record in records:
        for spline in record.get("splines", []):
            shifted = dict(spline)
            shifted["waypoints"] = [
                {
                    part: {
                        "x": _round(pt["x"] - GRID_ORIGIN_OFFSET),
                        "y": _round(pt["y"] - GRID_ORIGIN_OFFSET),
                    }
                    for part, pt in wp.items()
                }
                for wp in spline.get("waypoints", [])
            ]
            out.append(shifted)
    return out


# ---------------------------------------------------------------------------
# write-terrain / write-water formatting
# ---------------------------------------------------------------------------


def build_collection_stamps(records: list[dict]) -> list[Stamp]:
    """terrain.stamp.Stamp objects for every stamp member across every
    collection record, in the course-local [0, COURSE_SIZE_M] frame (same
    frame PGA2k_gen.py's load_all_stamps returns -- the GRID_ORIGIN_OFFSET
    shift happens later, in userLayers.stamp_to_entry). Every one is
    forced to TOOL_RAISE: a collection stamp is a relative delta on top
    of whatever terrain it lands on, never an absolute-height flatten.
    Entries missing a brush id are skipped."""
    out: list[Stamp] = []
    for record in records:
        for s in record.get("stamps", []):
            if s.get("brush") is None:
                continue
            out.append(Stamp(
                x=float(s["x"]), z=float(s["z"]),
                scale_x=float(s.get("scale_x", 1.0)), scale_z=float(s.get("scale_z", 1.0)),
                value=float(s.get("value", 0.0)), brush=int(s["brush"]),
                rotation=float(s.get("rotation_deg", 0.0)), tool=TOOL_RAISE,
            ))
    return out
