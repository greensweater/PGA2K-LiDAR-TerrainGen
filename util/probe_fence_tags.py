"""
util/probe_fence_tags.py

Probe for v2023 task 3.1: runs OSM ways through ingest.osm's real
parse_osm_features (classify_way + geometry build) and
course_output.fences' tag -> asset routing, and prints kind / geometry /
closed / asset per fence-like way.

    python util/probe_fence_tags.py              # built-in synthetic extract, asserts expectations
    python util/probe_fence_tags.py map.osm      # a real OSM extract, report only

The synthetic extract covers every fence/wall/hedge tag variant the
classifier handles, plus regression cases (a barrier=* way that already
classifies as something else must keep its old kind).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyproj  # noqa: E402

from course_output.fences import FENCE_KINDS, fence_asset_for_tags  # noqa: E402
from ingest.osm import parse_osm_features  # noqa: E402

# (tags, expected kind, expected asset label or None, closed ring?)
_CASES = [
    ({"barrier": "fence"}, "fence", "WoodFencesAPostAPrefab", False),
    ({"barrier": "fence", "fence_type": "chain_link"}, "fence", "UniFencePostAPrefab", False),
    ({"barrier": "fence", "material": "metal"}, "fence", "UniFencePostAPrefab", False),
    ({"barrier": "fence", "fence_type": "split_rail"}, "fence", "WoodFencesAPostAPrefab", True),
    ({"barrier": "fence", "material": "stone"}, "fence", "StoneWallAPostAPrefab", False),
    ({"barrier": "chain"}, "fence", "UniFencePostAPrefab", False),
    ({"barrier": "wall"}, "wall", "StoneWallAPostAPrefab", False),
    ({"barrier": "wall", "wall": "brick"}, "wall", "Asia_BrickWalls_PostPrefab", False),
    ({"barrier": "wall", "material": "brick"}, "wall", "Asia_BrickWalls_PostPrefab", False),
    ({"barrier": "wall", "wall": "dry_stone"}, "wall", "StoneWallAPostAPrefab", False),
    ({"barrier": "wall", "wall": "retaining_wall"}, "wall", "RetainWallAPostAPrefab", False),
    ({"barrier": "retaining_wall"}, "wall", "RetainWallAPostAPrefab", False),
    ({"barrier": "city_wall"}, "wall", "StoneWallAPostAPrefab", False),
    ({"barrier": "hedge"}, "hedge", "HedgeSplinePostPrefab", True),
    ({"barrier": "hedge", "area": "yes"}, "hedge", "HedgeSplinePostPrefab", True),
    ({"natural": "hedge"}, "hedge", "HedgeSplinePostPrefab", False),
    # Regressions: existing classifications must win over barrier=*.
    ({"barrier": "range_nets"}, "range_net", None, False),
    ({"building": "yes", "barrier": "wall"}, "building", None, True),
    ({"amenity": "parking", "barrier": "fence"}, "pavement", None, True),
    ({"barrier": "gate"}, None, None, False),
]


def _synthetic_osm() -> str:
    nodes, ways = [], []
    nid = 1
    for i, (tags, _, _, closed) in enumerate(_CASES):
        lat0 = 41.0 + i * 0.001
        ids = []
        for dlat, dlon in ((0, 0), (0, 0.0005), (0.0003, 0.0005), (0.0003, 0)):
            nodes.append(f'<node id="{nid}" lat="{lat0 + dlat}" lon="{-81.0 + dlon}"/>')
            ids.append(nid)
            nid += 1
        if closed:
            ids.append(ids[0])
        nds = "".join(f'<nd ref="{n}"/>' for n in ids)
        tag_xml = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        ways.append(f'<way id="{1000 + i}">{nds}{tag_xml}</way>')
    return f'<?xml version="1.0"?><osm version="0.6">{"".join(nodes)}{"".join(ways)}</osm>'


def _parse(path: Path):
    return parse_osm_features(path, pyproj.CRS("EPSG:3857"), 0.0, 0.0, 1.0, printf=lambda *_: None)


def _row(f) -> tuple:
    closed = f.geometry.geom_type == "LineString" and f.geometry.is_ring
    asset = fence_asset_for_tags(f.kind, f.tags).rsplit("/", 1)[-1] if f.kind in FENCE_KINDS else None
    return f.kind, f.geometry.geom_type, closed, asset


def main() -> int:
    if len(sys.argv) > 1:
        features = [f for f in _parse(Path(sys.argv[1])) if f.kind in FENCE_KINDS]
        for f in features:
            kind, geom, closed, asset = _row(f)
            print(f"way {f.osm_id}: {kind:6} {geom:10} closed={closed!s:5} -> {asset}  "
                  f"tags={ {k: v for k, v in f.tags.items() if k in ('barrier', 'natural', 'material', 'wall', 'fence_type')} }")
        print(f"{len(features)} fence/wall/hedge way(s)")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.osm"
        path.write_text(_synthetic_osm(), encoding="utf-8")
        by_id = {f.osm_id: f for f in _parse(path)}

    failures = 0
    for i, (tags, want_kind, want_asset, closed) in enumerate(_CASES):
        f = by_id.get(1000 + i)
        got = _row(f) if f is not None else (None, None, None, None)
        ok = got[0] == want_kind and got[3] == want_asset
        if want_kind in FENCE_KINDS:
            ok = ok and got[1] == "LineString" and got[2] == closed
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {tags} -> kind={got[0]} geom={got[1]} closed={got[2]} asset={got[3]}")
    print(f"{len(_CASES) - failures}/{len(_CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
