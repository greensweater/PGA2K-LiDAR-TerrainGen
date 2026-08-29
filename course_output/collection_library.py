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
*every* placed object (Value.items) and *every* surface spline in that
course into one template, anchored at their combined centroid, heading
0. The user then hand-edits the JSON to trim/tune members.

Placement (see course_output/collections.py + PGA2k_gen.py's
step_generate_collections): an OSM 2-node way tagged
pga_collection=<template name> -- node 1 is the anchor, node 2 gives the
heading -- is resolved against the library into collections.json, the
frozen per-project record the object/spline writers format at write
time (same "compile once, format at write" split as streams.json).
"""

from __future__ import annotations

import json
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
                    "scale": float}
    spline member: a raw surfaceSplines.json spline dict (surface,
        secondarySurface, secondaryWidth, width, state, ClosedPath,
        isClosed, isFilled, waypoints:[{pointOne,pointTwo,waypoint}]),
        with every point already shifted to be relative to the anchor.
    """
    name: str
    objects: list[dict] = field(default_factory=list)
    splines: list[dict] = field(default_factory=list)


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
        )
    return out


def save_collection(collection: Collection, libdir: Path) -> Path:
    """Write one template to <libdir>/<slug>.json, creating the library
    directory if needed. Returns the path written."""
    libdir = Path(libdir)
    libdir.mkdir(parents=True, exist_ok=True)
    path = libdir / f"{_slug(collection.name)}.json"
    payload = {"name": collection.name, "objects": collection.objects, "splines": collection.splines}
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
    """A placed item's coordinate is usually a float but position.y is
    sometimes the string "-Infinity" -- coerce or return None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def capture_from_course(course_nodes_dir: Path, name: str) -> Collection:
    """
    Snapshot every placed object (Value.items) and every surface spline
    in an extracted course into one template, anchored at their combined
    centroid with heading 0.

    course_nodes_dir is a course/CourseDescription_nodes/ directory (see
    PGA2k_gen.py's step_ingest_course) -- reads placedObjects2.json and
    surfaceSplines.json (falling back to surfaceSplines2.json). Missing
    files are treated as "no members of that kind", not an error.

    Coordinates are taken verbatim from the files and only ever used as
    differences from the centroid, so the source course's own grid
    origin is irrelevant.
    """
    course_nodes_dir = Path(course_nodes_dir)

    placed_path = course_nodes_dir / "placedObjects2.json"
    groups = load_placed_objects(placed_path) if placed_path.exists() else []

    splines: list[dict] = []
    for fname in ("surfaceSplines.json", "surfaceSplines2.json"):
        p = course_nodes_dir / fname
        if p.exists():
            splines = json.loads(p.read_text(encoding="utf-8"))
            break

    # --- anchor = centroid of every object position + every spline waypoint ---
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

    if not xs:
        anchor_x = anchor_z = 0.0
    else:
        anchor_x = sum(xs) / len(xs)
        anchor_z = sum(zs) / len(zs)

    # --- object members ---
    obj_members: list[dict] = []
    for key, item in _iter_placed_items(groups):
        pos = item.get("position", {})
        x, z = _num(pos.get("x")), _num(pos.get("z"))
        if x is None or z is None:
            continue
        scale = _num(item.get("scale", {}).get("x")) or 1.0
        rot_y = _num(item.get("rotation", {}).get("y")) or 0.0
        obj_members.append({
            "category": key.get("category"),
            "type": key.get("type"),
            "theme": key.get("theme"),
            "path": key.get("path"),
            "dx": round(x - anchor_x, 3),
            "dz": round(z - anchor_z, 3),
            "rotation_deg": round(rot_y, 3),
            "scale": round(scale, 3),
        })

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

    return Collection(name=name, objects=obj_members, splines=spline_members)
