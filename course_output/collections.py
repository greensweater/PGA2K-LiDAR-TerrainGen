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
                  "category","type","theme","path"}, ...],
     "splines": [ <surfaceSplines.json spline dict, points in course-local frame> ]}

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
the parameter value. A member with neither is placed identically
regardless of the parameter. Spline members are always static.
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
    a "default" fallback) onto the base member, then substitute a
    "{param}" token in the asset path. A member with no "variants" and
    no "{param}" in its path comes back unchanged."""
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
    path = effective.get("path")
    if path and parameter is not None and "{param}" in path:
        effective["path"] = path.replace("{param}", str(parameter))
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
        objects.append({
            "x": _round(anchor_x + rx),
            "z": _round(anchor_z + rz),
            "rotation_deg": _round((m.get("rotation_deg", 0.0) + heading_deg) % 360.0),
            "scale": _round(m.get("scale", 1.0)),
            "category": m.get("category"),
            "type": m.get("type"),
            "theme": m.get("theme"),
            "path": m.get("path"),
        })

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

    return {
        "source_id": source_id,
        "name": template.name,
        "parameter": parameter,
        "x": _round(anchor_x),
        "z": _round(anchor_z),
        "heading_deg": _round(heading_deg % 360.0),
        "objects": objects,
        "splines": splines,
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
        group["Value"]["items"].append(
            _placed_item(obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0))
        )
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
        group["Value"]["items"].append(
            _placed_item(obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0))
        )
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
