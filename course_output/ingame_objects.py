"""
course_output/ingame_objects.py

Objects a user added by hand-editing an exported .course in PGA Tour
2K's own in-game object tool, captured back into this tool's tracked
state -- see PGA2k_gen.py's step_import_ingame_edits.

A record (ingame_objects.json is a flat list of these):

    {"path": str | None,               # v2021+ asset path
     "category": int | None,           # v2019 catalog key
     "type": int | None,               # v2019 catalog key
     "theme": bool | None,             # v2019 catalog key
     "x": float, "z": float,           # course-local metres (NOT grid-shifted)
     "rotation_deg": float,
     "scale": float,
     "y": float | None,                # raw saved y, verbatim; None = ground-snap
     "group": str}                     # user-visible label, one per import run

`path` is set for a record captured from a v2021+ source course;
`category`/`type`/`theme` for one captured from v2019 -- same dual-
identity shape course_output/collections.py's collection objects
already use, and resolved the same way (_resolve_v2019_key/
_resolve_v2021_path, reused directly from there rather than
duplicated).

`y` is captured VERBATIM from the source course's saved
position.y and replayed unchanged at write-objects time -- no
terrain-relative re-grounding (contrast with collections.py's `dy`-
above-flatten-datum, which IS re-grounded against the target
TerrainModel every write). An object placed at a designed elevation
in-game should come back at exactly that same elevation every time,
regardless of any later terrain change. `y is None` means the source
had the literal "-Infinity" (ground-snap); it keeps ground-snapping.

Kept as its own file, independent of objects.json, because objects.json
is wholesale-overwritten by every pack-objects run (from object_list.json/
features.geojson/collections.json) -- see that step's docstring. This
file is only ever READ by pack-objects (folded into objects.json's kind
union as kind="ingame_object") and appended to by step_import_ingame_edits,
so imported objects survive an OSM re-ingest / regenerate-trees / repack
automatically, same "compile once, format at write time" principle as
everything else.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from course_output.collections import _resolve_v2019_key, _resolve_v2021_path
from course_output.objects import _placed_item, _placed_object_group_v2021

_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def save_ingame_objects(records: list[dict], path: Path) -> None:
    """Write ingame_objects.json -- a plain JSON list, same convention as
    collections.json/streams.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def load_ingame_objects(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def summarize_ingame_object_groups(records: list[dict]) -> list[tuple[str, int]]:
    """(group name, record count), one per distinct `group` label, in
    first-seen order -- for the GUI's Objects tab listing."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for r in records:
        name = r.get("group", "")
        if name not in counts:
            counts[name] = 0
            order.append(name)
        counts[name] += 1
    return [(name, counts[name]) for name in order]


def remove_ingame_object_groups(records: list[dict], group_names: set[str]) -> list[dict]:
    """Every record whose `group` is NOT in `group_names` -- i.e. the
    list to save back after deleting the given group(s). Surgical by
    group label only, same idea as
    PGA2k_gen_gui.py's _remove_cluster_fill_groups."""
    return [r for r in records if r.get("group") not in group_names]


def build_ingame_objects_v2019(records: list[dict]) -> list[dict]:
    """placedObjects2 v2019 groups (Key {category,type,theme}) from
    imported in-game objects, one group per distinct key, items via
    objects._placed_item. Unresolvable entries (no v2019 key and no
    catalog match for a v2021-captured path) are skipped with a note."""
    groups: dict[tuple[int, int, bool], dict] = {}
    skipped = 0
    for obj in records:
        key = _resolve_v2019_key(obj)
        if key is None:
            skipped += 1
            continue
        group = groups.setdefault(key, {
            "Key": {"category": key[0], "type": key[1], "theme": key[2]},
            "Value": {"items": [], "clusters": []},
        })
        y = obj.get("y")
        group["Value"]["items"].append(_placed_item(
            obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0),
            y=y if y is not None else "-Infinity",
        ))
    if skipped:
        print(f"  NOTE: {skipped} imported in-game object(s) skipped -- no v2019 category/type "
              "(captured from a v2021 course whose asset path isn't in asset_catalog.json)")
    return list(groups.values())


def build_ingame_objects_v2021(records: list[dict]) -> list[dict]:
    """placedObjects2 v2021+ groups (Key {path}) from imported in-game
    objects, one group per distinct asset path. Unresolvable entries
    are skipped with a note."""
    groups: dict[str, dict] = {}
    skipped = 0
    for obj in records:
        path = _resolve_v2021_path(obj)
        if not path:
            skipped += 1
            continue
        group = groups.setdefault(path, _placed_object_group_v2021(path))
        y = obj.get("y")
        group["Value"]["items"].append(_placed_item(
            obj["x"], obj["z"], obj.get("scale", 1.0), obj.get("rotation_deg", 0.0),
            y=y if y is not None else "-Infinity",
        ))
    if skipped:
        print(f"  NOTE: {skipped} imported in-game object(s) skipped -- no v2021 asset path "
              "(captured from a v2019 course whose category/type isn't in asset_catalog.json)")
    return list(groups.values())
