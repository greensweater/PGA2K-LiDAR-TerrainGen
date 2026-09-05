"""
util/extract_tree_dimensions.py

Capture per-prefab native tree dimensions from a hand-built .course and
(optionally) write them into course_output/asset_catalog.json's
native_height_m / native_canopy_radius_m fields -- the data
course_output/objects.py._tree_scale uses to size placed trees to their
real height instead of a course-relative guess.

HOW TO BUILD THE CAPTURE COURSE (in-game, one theme at a time):
  1. Flatten the whole working area to a known datum (the reference
     capture used a type-72 flatten stamp, value = 1.0 -> --ground 1.0).
  2. For every category-0 asset in the theme: place it at scale 1,1,1.
  3. Beside each, drop a type-73 hard-round SCULPT (flatten) stamp and
     size its scale.x to match the asset's crown RADIUS.
  4. Beside each, place a water plane and raise its surface Y until it
     sits exactly at the asset's crown TOP.
  5. Save, then extract the .course (util/course_extract.py) to a folder.

Then:  python util/extract_tree_dimensions.py <extracted_dir> --theme 11 --write

native_height_m       = nearest water plane's surface Y  -  --ground
native_canopy_radius_m = nearest type-73 stamp's scale.x

--write only patches asset_catalog.json when --theme matches the file's
own "source_theme_id" (v2019 type ids are per-theme -- a height measured
for rustic's "type 14" must not be written against another theme). With
no --theme, or a mismatch, the table is printed but nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from course_output.asset_catalog import ASSET_ENTRIES  # noqa: E402

ASSET_CATALOG_JSON = _REPO_ROOT / "course_output" / "asset_catalog.json"

# type-73 = hard round; the sculpt/flatten stamp the capture uses to mark
# a crown radius. terrainHeight entries of any other brush type are
# ignored (baseline type-72 flatten, etc).
SCULPT_BRUSH_TYPE = 73
# A stamp / water plane further than this (metres) from its tree is
# almost certainly a different feature (e.g. an unrelated stream) -- warn
# rather than trust it.
MATCH_WARN_DIST_M = 12.0


def _dist(a: dict, x: float, z: float) -> float:
    p = a["position"]
    return math.hypot(p["x"] - x, p["z"] - z)


def _nearest(candidates: list[dict], x: float, z: float) -> tuple[dict | None, float]:
    if not candidates:
        return None, math.inf
    best = min(candidates, key=lambda e: _dist(e, x, z))
    return best, _dist(best, x, z)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("course_dir", help="extracted .course folder (contains CourseDescription_nodes/)")
    ap.add_argument("--ground", type=float, default=1.0,
                    help="the datum the capture ground was flattened to (default 1.0)")
    ap.add_argument("--theme", type=int, default=None,
                    help="theme id of the capture -- required for --write, checked against "
                         "asset_catalog.json's source_theme_id")
    ap.add_argument("--write", action="store_true",
                    help="patch native_height_m / native_canopy_radius_m into asset_catalog.json "
                         "(only if --theme matches source_theme_id)")
    args = ap.parse_args()

    nodes = Path(args.course_dir) / "CourseDescription_nodes"
    placed = json.loads((nodes / "placedObjects2.json").read_text(encoding="utf-8"))
    layers = json.loads((nodes / "userLayers.json").read_text(encoding="utf-8"))

    label = {e.type: e.label for e in ASSET_ENTRIES if e.category == 0}

    objs: list[tuple[int, float, float]] = []
    for group in placed:
        key = group.get("Key", {})
        if key.get("category") != 0:
            continue
        for it in group["Value"]["items"]:
            p = it["position"]
            objs.append((key["type"], p["x"], p["z"]))
    objs.sort()

    stamps = [e for e in layers.get("terrainHeight", []) if e.get("type") == SCULPT_BRUSH_TYPE]
    waters = list(layers.get("water", []))

    if not objs:
        print("No category-0 placed objects found -- nothing to measure.")
        return 1

    rows: dict[int, tuple[float, float]] = {}
    print(f"{'type':>4}  {'asset':<34} {'height_m':>9} {'radius_m':>9}  {'stamp_d':>7} {'water_d':>7}")
    for t, x, z in objs:
        stamp, sd = _nearest(stamps, x, z)
        water, wd = _nearest(waters, x, z)
        if stamp is None or water is None:
            print(f"{t:>4}  {label.get(t, '?'):<34} -- missing {'stamp' if stamp is None else 'water'}")
            continue
        radius = round(stamp["scale"]["x"], 1)
        height = round(water["value"] - args.ground, 1)
        flag = "  <-- far" if max(sd, wd) > MATCH_WARN_DIST_M else ""
        print(f"{t:>4}  {label.get(t, '?'):<34} {height:>9.2f} {radius:>9.2f}  {sd:>7.2f} {wd:>7.2f}{flag}")
        rows[t] = (height, radius)

    if not args.write:
        print("\n(dry run -- pass --write with --theme to update asset_catalog.json)")
        return 0

    data = json.loads(ASSET_CATALOG_JSON.read_text(encoding="utf-8"))
    src = data.get("source_theme_id")
    if args.theme is None:
        print("\n--write needs --theme; nothing written.")
        return 1
    if src != args.theme:
        print(f"\n--theme {args.theme} != asset_catalog.json source_theme_id {src!r}; "
              "type ids are per-theme, so nothing written. Re-capture under that theme, "
              "or (if this IS a fresh catalog) set source_theme_id first.")
        return 1

    n = 0
    for e in data["entries"]:
        if e.get("category") == 0 and e.get("type") in rows:
            h, r = rows[e["type"]]
            e["native_height_m"] = h
            e["native_canopy_radius_m"] = r
            n += 1
    out = json.dumps(data, indent=2).replace("\n", "\r\n")
    ASSET_CATALOG_JSON.write_text(out, encoding="utf-8", newline="")
    json.loads(ASSET_CATALOG_JSON.read_text(encoding="utf-8"))  # validate
    print(f"\nWrote {n} entr{'y' if n == 1 else 'ies'} to {ASSET_CATALOG_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
