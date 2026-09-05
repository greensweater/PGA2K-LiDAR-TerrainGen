"""
course_output/collection_library.py

Object *collections* -- reusable, named templates that bundle several
placed objects (a bench, a bin, a sign, ...) and several surface splines
(a mulch bed, a path, ...) into one unit that can be dropped onto a
course at a position + heading and rotated as a group.

A collection TEMPLATE lives as one JSON file in a user library directory
(default ~/.pga2k/collections/, overridable via project.json's
"collections_library_dir"). Every member's position is stored as a
metre offset (dx, dz) from the template ANCHOR, and every member
rotation is relative to the template's own heading of 0 -- both
frame-invariant differences, so a template never needs to know the
origin convention of whatever course it was captured from or placed
into.

Templates are AUTHORED by capture: point capture_from_course() at an
extracted .course's CourseDescription_nodes/ directory and it snapshots
*every* placed object (Value.items), *every* surface spline, and every
RAISE-tool terrain stamp (userLayers.json "height", tool == 1 -- a
raised flowerbed bed, a berm) in that course into one template, anchored
at their combined centroid, heading 0. The user then hand-edits the JSON
to trim/tune members.

Flatten-tool terrain stamps (tool == 0) are deliberately NOT captured:
a flatten sets an ABSOLUTE height, so riding one onto another course
would stamp the prop course's elevation onto wherever the collection
lands. Only relative raise stamps compose safely. The map-wide flatten
grounding datum (below) is one of the tool == 0 stamps skipped here.

VERTICAL GROUNDING: most placed props carry position.y = "-Infinity" --
the game drops them onto the terrain on load, each prefab sinking by its
own built-in "bleed" for a gapless join. Those members keep "-Infinity"
and re-ground wherever they're placed. But some props in a hand-built
group carry a real scalar y (a deck/railing bridge built at a set
height). For those, capture records `dy` = that y minus the source
course's map-wide "flatten" datum (the deliberate zero-reference plane
the prop course is built on -- the largest tool=0 stamp in its
userLayers.json "height" array). At placement time
course_output/collections.py + PGA2k_gen.py's step_write_objects turn
each `dy` back into an absolute y = target terrain height at (x, z) +
`dy`, so the deck sits the same distance above local ground on the new
course. If the source course has no map-wide flatten stamp, capture
falls back to `dy` = raw y (datum 0) with a printed note.

Placement (see course_output/collections.py + PGA2k_gen.py's
step_generate_collections): an OSM 2-node way tagged
pga_collection=<template name> -- node 1 is the anchor, node 2 gives the
heading -- is resolved against the library into collections.json, the
frozen per-project record the object/spline writers format at write
time (same "compile once, format at write" split as streams.json).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from course_output.objects import load_placed_objects

# One JSON file per template; a "library" is just the directory holding
# them. Mirrors course_output/asset_catalog.py loading a single catalog
# file -- here it's a directory glob instead, since templates are user
# data that grows over time, not a fixed shipped table.
LIBRARY_GLOB = "*.json"

# userLayers.json "height" entry tool values: 0 = flatten (pulls terrain
# toward an ABSOLUTE height), 1 = raise (adds a RELATIVE delta). Only
# raise stamps can safely ride a collection onto an arbitrary course --
# a flatten would stamp its source course's absolute height onto the
# target. See terrain/stamp.py.
_TOOL_FLATTEN = 0
_TOOL_RAISE = 1
# For a flatten stamp to count as the prop course's map-wide grounding
# datum (see the module docstring's VERTICAL GROUNDING section) its
# footprint has to actually blanket the course -- a small local flatten
# isn't it.
_MAP_WIDE_FLATTEN_MIN_SCALE_M = 400.0


def default_library_dir() -> Path:
    """Where collection templates live when project.json doesn't say
    otherwise. Deliberately outside the repo -- templates are user data
    (captured from the user's own prop courses), not project content."""
    return Path.home() / ".pga2k" / "collections"


@dataclass(slots=True)
class Collection:
    """One collection template. `objects` / `splines` are plain dicts
    (not further dataclasses) so hand-editing the JSON is unconstrained
    and a future member field doesn't need a schema change here.

    object member: {"category": int, "type": int, "theme": bool,
                    "path": str | None,  # v2021 asset path, when known
                    "dx": float, "dz": float,       # metres from anchor
                    "rotation_deg": float,          # relative to heading 0
                    "scale": float,
                    "dy": float}                    # optional -- see VERTICAL GROUNDING
    spline member: a raw surfaceSplines.json spline dict (surface,
        secondarySurface, secondaryWidth, width, state, ClosedPath,
        isClosed, isFilled, waypoints:[{pointOne,pointTwo,waypoint}]),
        with every point already shifted to be relative to the anchor.
    stamp member: {"dx": float, "dz": float,        # metres from anchor
                   "scale_x": float, "scale_z": float,
                   "value": float, "brush": int,
                   "rotation_deg": float,           # relative to heading 0
                   "tool": 1}                       # always raise -- see below
        A terrain stamp (a raised flowerbed bed, a berm, ...). ONLY
        raise-tool stamps are captured -- a flatten stamp would impose
        the prop course's absolute height on wherever the collection
        lands. Placed as one extra terrain stamp per member at
        write-terrain / write-water time (course_output/collections.py's
        build_collection_stamps, folded in by PGA2k_gen.py's
        _load_normalized_stamps -- not persisted as a stamps_N.json
        layer, so re-running generate-collections never double-applies).
    """
    name: str
    objects: list[dict] = field(default_factory=list)
    splines: list[dict] = field(default_factory=list)
    stamps: list[dict] = field(default_factory=list)


def _slug(name: str) -> str:
    """Filesystem-safe stem for a template name -- lowercase, spaces and
    punctuation to underscores. The template's real display name is
    stored inside the file ("name"), not derived back from the stem, so
    two names that slug the same just overwrite (acceptable for a
    user-curated library)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return s or "collection"


def load_library(libdir: Path) -> dict[str, Collection]:
    """Every *.json template in `libdir`, keyed by its own "name" field.
    A missing directory is not an error -- just an empty library.
    A file that doesn't parse, or lacks "name", is skipped with a
    printed note rather than aborting the whole load."""
    libdir = Path(libdir)
    out: dict[str, Collection] = {}
    if not libdir.is_dir():
        return out
    for path in sorted(libdir.glob(LIBRARY_GLOB)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data["name"]
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            print(f"  NOTE: skipping unreadable collection template {path.name}")
            continue
        out[name] = Collection(
            name=name,
            objects=list(data.get("objects", [])),
            splines=list(data.get("splines", [])),
            stamps=list(data.get("stamps", [])),
        )
    return out


def save_collection(collection: Collection, libdir: Path) -> Path:
    """Write one template to <libdir>/<slug>.json, creating the library
    directory if needed. Returns the path written."""
    libdir = Path(libdir)
    libdir.mkdir(parents=True, exist_ok=True)
    path = libdir / f"{_slug(collection.name)}.json"
    payload = {
        "name": collection.name,
        "objects": collection.objects,
        "splines": collection.splines,
        "stamps": collection.stamps,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _iter_placed_items(groups: list[dict]):
    """(key, item) for every Value.items entry across every
    placedObjects2 group -- `key` is the group's own Key dict
    ({"path": ...} for v2021, {"category","type","theme"} for v2019)."""
    for group in groups:
        key = group.get("Key", {})
        for item in group.get("Value", {}).get("items", []):
            yield key, item


def _num(value) -> Optional[float]:
    """A finite float from `value`, or None. position.y is often the
    string "-Infinity" (ground-snap marker) -- and `float("-Infinity")`
    would otherwise succeed -- so non-finite results are rejected too."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _height_entries(course_nodes_dir: Path) -> list[dict]:
    """The userLayers.json "height" array (Landscape-mode terrain stamps)
    for an extracted course, or [] if the file/key is missing or unreadable."""
    path = Path(course_nodes_dir) / "userLayers.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    entries = data.get("height")
    return entries if isinstance(entries, list) else []


def _stamp_member(entry: dict, anchor_x: float, anchor_z: float) -> Optional[dict]:
    """One template stamp member from a userLayers.json "height" raise
    entry (caller has already filtered to tool == 1), positioned relative
    to the anchor. None if the entry is missing usable geometry."""
    pos = entry.get("position", {})
    scale = entry.get("scale", {})
    x, z = _num(pos.get("x")), _num(pos.get("z"))
    sx, sz = _num(scale.get("x")), _num(scale.get("z"))
    value = _num(entry.get("value"))
    if None in (x, z, sx, sz, value):
        return None
    rot_y = _num((entry.get("rotation") or {}).get("y")) or 0.0
    return {
        "dx": round(x - anchor_x, 3),
        "dz": round(z - anchor_z, 3),
        "scale_x": round(sx, 3),
        "scale_z": round(sz, 3),
        "value": round(value, 3),
        "brush": entry.get("type"),
        "rotation_deg": round(rot_y, 3),
        "tool": _TOOL_RAISE,
    }


def _flatten_datum(course_nodes_dir: Path) -> tuple[float, bool]:
    """(datum, found) -- the `value` of the largest map-wide flatten
    stamp in the course's userLayers.json "height" array, the deliberate
    zero-reference plane a prop course is built on (see the module
    docstring's VERTICAL GROUNDING section). (0.0, False) if there's no
    userLayers.json or nothing big enough to count as map-wide."""
    best_value: Optional[float] = None
    best_footprint = -1.0
    for entry in _height_entries(course_nodes_dir):
        if entry.get("tool") != _TOOL_FLATTEN:
            continue
        scale = entry.get("scale", {})
        sx, sz = _num(scale.get("x")), _num(scale.get("z"))
        value = _num(entry.get("value"))
        if sx is None or sz is None or value is None:
            continue
        if min(abs(sx), abs(sz)) < _MAP_WIDE_FLATTEN_MIN_SCALE_M:
            continue
        footprint = abs(sx) * abs(sz)
        if footprint > best_footprint:
            best_value, best_footprint = value, footprint
    if best_value is None:
        return 0.0, False
    return best_value, True


def capture_from_course(course_nodes_dir: Path, name: str, printf=print) -> Collection:
    """
    Snapshot every placed object (Value.items), every surface spline, and
    every RAISE-tool terrain stamp in an extracted course into one
    template, anchored at their combined centroid with heading 0.

    course_nodes_dir is a course/CourseDescription_nodes/ directory (see
    PGA2k_gen.py's step_ingest_course) -- reads placedObjects3.json
    (falling back to placedObjects2.json -- a course extracted from a
    v2021+ save has the former, v2019 the latter), surfaceSplines.json
    (falling back to surfaceSplines2.json), and userLayers.json
    ("height" array). Missing files are treated as "no members of that
    kind", not an error. Flatten-tool stamps (tool == 0) are skipped --
    only relative raise stamps can safely ride a collection onto
    another course.

    Coordinates are taken verbatim from the files and only ever used as
    differences from the centroid, so the source course's own grid
    origin is irrelevant.
    """
    course_nodes_dir = Path(course_nodes_dir)

    groups: list[dict] = []
    for fname in ("placedObjects3.json", "placedObjects2.json"):
        p = course_nodes_dir / fname
        if p.exists():
            groups = load_placed_objects(p)
            break

    splines: list[dict] = []
    for fname in ("surfaceSplines.json", "surfaceSplines2.json"):
        p = course_nodes_dir / fname
        if p.exists():
            splines = json.loads(p.read_text(encoding="utf-8"))
            break

    height_entries = _height_entries(course_nodes_dir)
    raise_entries = [e for e in height_entries if e.get("tool") == _TOOL_RAISE]
    n_flatten_skipped = sum(1 for e in height_entries if e.get("tool") == _TOOL_FLATTEN)

    # --- anchor = centroid of every object position, spline waypoint, and raise-stamp center ---
    xs: list[float] = []
    zs: list[float] = []
    for _key, item in _iter_placed_items(groups):
        pos = item.get("position", {})
        x, z = _num(pos.get("x")), _num(pos.get("z"))
        if x is not None and z is not None:
            xs.append(x)
            zs.append(z)
    for spline in splines:
        for wp in spline.get("waypoints", []):
            pt = wp.get("waypoint", {})
            x, z = _num(pt.get("x")), _num(pt.get("y"))  # spline "y" IS the course z
            if x is not None and z is not None:
                xs.append(x)
                zs.append(z)
    for entry in raise_entries:
        epos = entry.get("position", {})
        x, z = _num(epos.get("x")), _num(epos.get("z"))
        if x is not None and z is not None:
            xs.append(x)
            zs.append(z)

    if not xs:
        anchor_x = anchor_z = 0.0
    else:
        anchor_x = sum(xs) / len(xs)
        anchor_z = sum(zs) / len(zs)

    datum, datum_found = _flatten_datum(course_nodes_dir)

    # --- terrain stamp members (raise only) ---
    stamp_members = [
        m for m in (_stamp_member(e, anchor_x, anchor_z) for e in raise_entries) if m is not None
    ]

    # --- object members ---
    obj_members: list[dict] = []
    n_grounded = n_scalar = 0
    for key, item in _iter_placed_items(groups):
        pos = item.get("position", {})
        x, z = _num(pos.get("x")), _num(pos.get("z"))
        if x is None or z is None:
            continue
        scale = _num(item.get("scale", {}).get("x")) or 1.0
        rot_y = _num(item.get("rotation", {}).get("y")) or 0.0
        member = {
            "category": key.get("category"),
            "type": key.get("type"),
            "theme": key.get("theme"),
            "path": key.get("path"),
            "dx": round(x - anchor_x, 3),
            "dz": round(z - anchor_z, 3),
            "rotation_deg": round(rot_y, 3),
            "scale": round(scale, 3),
        }
        # A finite position.y means the prop was placed at a designed
        # elevation (not "-Infinity" ground-snap) -- record its height
        # above the flatten datum so placement can re-ground it on the
        # target course's terrain (see the module docstring).
        y = _num(pos.get("y"))
        if y is None:
            n_grounded += 1
        else:
            member["dy"] = round(y - datum, 3)
            n_scalar += 1
        obj_members.append(member)

    if n_scalar and not datum_found:
        printf(f"  NOTE: {n_scalar} object(s) have a scalar y but the source course has no "
               "map-wide flatten stamp -- captured 'dy' as offset from y=0. Build the prop "
               "course on a single course-wide flatten stamp for a reliable grounding datum.")
    printf(f"  captured {n_grounded} ground-snapped + {n_scalar} elevated object(s)"
           + (f" (flatten datum y={datum:g})" if datum_found else ""))
    if stamp_members or n_flatten_skipped:
        note = f"  captured {len(stamp_members)} raise-tool terrain stamp(s)"
        if n_flatten_skipped:
            note += (f"; skipped {n_flatten_skipped} flatten stamp(s) (incl. the grounding "
                     "datum) -- collection terrain must use the Raise tool")
        printf(note)

    # --- spline members: shift every point by -anchor, keep everything else ---
    spline_members: list[dict] = []
    for spline in splines:
        shifted = dict(spline)
        shifted["waypoints"] = [
            {
                part: {
                    "x": round(_num(wp[part]["x"]) - anchor_x, 3),
                    "y": round(_num(wp[part]["y"]) - anchor_z, 3),
                }
                for part in ("pointOne", "pointTwo", "waypoint")
                if part in wp
            }
            for wp in spline.get("waypoints", [])
        ]
        spline_members.append(shifted)

    return Collection(name=name, objects=obj_members, splines=spline_members, stamps=stamp_members)
