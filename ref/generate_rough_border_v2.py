#!/usr/bin/env python3

from pathlib import Path
from copy import deepcopy
import argparse
import json
import math

from shapely.geometry import (
    LineString,
    Polygon,
    MultiPolygon,
    box
)

# ------------------------------------------------------------
# PGA 2K23 heavy rough border generator
#
# Generates layered vegetation spline borders around
# CLOSED heavy rough polygons (surface == 4)
#
# Uses polygon ring boolean operations:
#
# outer.buffer(width).difference(
#     outer.buffer(offset)
# )
#
# ------------------------------------------------------------

SIZE = 2000
MAP_MIN = -1000
MAP_MAX = 1000

MAX_SPLINE_BOUNDS = 100.0

HEAVY_ROUGH_SAMPLE_SPACING = 8.0
HEAVY_ROUGH_SIMPLIFY = 2.5

ROUND_DIGITS = 3

# NATURE = [
    # {
        # "path": "Assets/CourseGen/Detail/Grass/TallGrassD",
        # "offset": 0,
        # "width": 10,
        # "fillPct": 0.2
    # },
    # {
        # "path": "Assets/CourseGen/Detail/Bushes/GroundCoverC1Prefab",
        # "offset": 0,
        # "width": 10,
        # "fillPct": 0.181
    # },
    # {
        # "path": "Assets/CourseGen/Detail/Bushes/GroundCoverC2Prefab",
        # "offset": 5.0,
        # "width": 20.0,
        # "fillPct": 0.181
    # },
    # {
        # "path": "Assets/Foliage/SpeedTree/Bushes/NaturePackBushes/Prefabs/NPBush08A",
        # "offset": 10.0,
        # "width": 30.0,
        # "fillPct": 0.005
    # }
# ]

NATURE = [
    {
        "path": "Assets/Foliage/SpeedTree/Grass/Long_Grass/Prefabs/Long_Grass_Desktop",
        "offset": 0.0,
        "width": 1.0,
        "fillPct": 0.3
    },
    {
        "path": "Assets/CourseGen/Detail/Grass/DetailGrassKPrefab",
        "offset": 0.0,
        "width": 2.0,
        "fillPct": 0.4
    },
    {
        "path": "Assets/CourseGen/Detail/Grass/DetailGrassSPrefab",
        "offset": 0.0,
        "width": 2.0,
        "fillPct": 0.4
    },
]

# ----------------- Helpers -----------------

def r3(v):

    try:
        return round(float(v), ROUND_DIGITS)
    except Exception:
        return v

def round_dict_floats(obj):

    if isinstance(obj, dict):

        for k, v in obj.items():
            obj[k] = round_dict_floats(v)

        return obj

    if isinstance(obj, list):
        return [round_dict_floats(v) for v in obj]

    if isinstance(obj, float):
        return round(obj, ROUND_DIGITS)

    return obj

def backup(path: Path):

    bak = path.with_suffix(path.suffix + ".bak.json")

    i = 1

    while bak.exists():

        bak = Path(
            str(path.with_suffix(path.suffix + f".bak{i}.json"))
        )

        i += 1

    path.rename(bak)

    return bak

def dist(a, b):

    return math.hypot(
        b[0] - a[0],
        b[1] - a[1]
    )

# ----------------- Surface spline helpers -----------------

def _get_surface_value(spline):

    candidates = ["surface"]

    containers = [
        spline,
        spline.get("Value", {}),
        spline.get("value", {})
    ]

    for c in containers:

        if not isinstance(c, dict):
            continue

        for k in candidates:

            v = c.get(k)

            if v is None:
                continue

            try:
                return int(v)
            except Exception:
                pass

    return None

def _get_state_value(spline):

    candidates = ["state"]

    containers = [spline]

    for c in containers:

        if not isinstance(c, dict):
            continue

        for k in candidates:

            v = c.get(k)

            if v is None:
                continue

            return int(v)

    return None

def _extract_vec2(p):

    if not isinstance(p, dict):
        return None

    try:

        x = float(p["x"])
        z = float(p.get("z", p.get("y")))

        return (x, z)

    except Exception:
        return None

def _cubic_bezier(p0, p1, p2, p3, t):

    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t

    x = (
        mt2 * mt * p0[0] +
        3 * mt2 * t * p1[0] +
        3 * mt * t2 * p2[0] +
        t2 * t * p3[0]
    )

    z = (
        mt2 * mt * p0[1] +
        3 * mt2 * t * p1[1] +
        3 * mt * t2 * p2[1] +
        t2 * t * p3[1]
    )

    return (x, z)

# ----------------- Sampling -----------------

def sample_spline_dense(spline):

    waypoints = spline.get("waypoints", [])

    if len(waypoints) < 2:
        return []

    sampled = []

    for i in range(len(waypoints) - 1):

        a = waypoints[i]
        b = waypoints[i + 1]

        p0 = _extract_vec2(a.get("waypoint"))
        p1 = _extract_vec2(a.get("pointTwo"))
        p2 = _extract_vec2(b.get("pointOne"))
        p3 = _extract_vec2(b.get("waypoint"))

        if not all((p0, p1, p2, p3)):
            continue

        approx_len = (
            dist(p0, p1) +
            dist(p1, p2) +
            dist(p2, p3)
        )

        steps = max(8, int(approx_len))

        for s in range(steps):

            t = s / float(steps)

            sampled.append(
                _cubic_bezier(p0, p1, p2, p3, t)
            )

    sampled.append(p3)

    return sampled

def resample_by_distance(
    points,
    spacing=HEAVY_ROUGH_SAMPLE_SPACING
):

    if len(points) < 2:
        return points

    pearls = [points[0]]

    accum = 0.0

    for i in range(1, len(points)):

        seg = dist(
            points[i - 1],
            points[i]
        )

        accum += seg

        if accum >= spacing:

            pearls.append(points[i])

            accum = 0.0

    return pearls

# ----------------- Polygon helpers -----------------

def subdivide_poly(
    poly,
    max_size=MAX_SPLINE_BOUNDS
):

    minx, miny, maxx, maxy = poly.bounds

    if (
        (maxx - minx) <= max_size and
        (maxy - miny) <= max_size
    ):
        return [poly]

    if (maxx - minx) >= (maxy - miny):

        split = (minx + maxx) * 0.5

        a = box(minx, miny, split, maxy)
        b = box(split, miny, maxx, maxy)

    else:

        split = (miny + maxy) * 0.5

        a = box(minx, miny, maxx, split)
        b = box(minx, split, maxx, maxy)

    out = []

    for clipped in [
        poly.intersection(a),
        poly.intersection(b)
    ]:

        if clipped.is_empty:
            continue

        if isinstance(clipped, MultiPolygon):

            for geom in clipped.geoms:
                out.extend(
                    subdivide_poly(
                        geom,
                        max_size
                    )
                )

        else:

            out.extend(
                subdivide_poly(
                    clipped,
                    max_size
                )
            )

    return out

# ----------------- Object spline generation -----------------

def make_waypoint(x, z):

    x = r3(x)
    z = r3(z)

    return {
        "pointOne": {
            "x": x,
            "y": z
        },
        "pointTwo": {
            "x": x,
            "y": z
        },
        "waypoint": {
            "x": x,
            "y": z
        }
    }

def polygon_to_object_spline(
    poly,
    fillPct
):

    coords = list(poly.exterior.coords)

    return {
        "path": {
            "waypoints": [
                make_waypoint(x, z)
                for (x, z) in coords
            ],
            "width": 1.0,
            "state": 0,
            "ClosedPath": True,
            "isClosed": True,
            "isFilled": True
        },
        "fillPct": r3(fillPct)
    }

# ----------------- Main -----------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('working_folder')
    ap.add_argument('--dry', action='store_true')

    args = ap.parse_args()

    working = Path(args.working_folder)

    splines_path = (
        working /
        'CourseDescription_nodes' /
        'surfaceSplines2.json'
    )

    if not splines_path.exists():

        splines_path = (
            working /
            'CourseDescription_nodes' /
            'surfaceSplines.json'
        )

    placed_path = (
        working /
        'CourseDescription_nodes' /
        'placedObjects3.json'
    )

    splines = json.loads(
        splines_path.read_text(
            encoding='utf-8'
        )
    )

    placed = json.loads(
        placed_path.read_text(
            encoding='utf-8'
        )
    )

    nature_nodes = {}

    for node in placed:

        path = node.get(
            'Key',
            {}
        ).get(
            'path',
            ''
        )

        for obj in NATURE:

            if path == obj['path']:
                nature_nodes[path] = node

    for obj in NATURE:

        path = obj['path']

        if path not in nature_nodes:

            node = {
                "Key": {
                    "path": path
                },
                "Value": {
                    "items": [],
                    "clusters": [],
                    "splines": []
                }
            }

            placed.append(node)

            nature_nodes[path] = node

    generated_splines = 0

    for spline in splines:

        surface = _get_surface_value(spline)
        state = _get_state_value(spline)

        if surface != 4 or state != 3:
            continue

        dense = sample_spline_dense(spline)

        if len(dense) < 3:
            continue

        pearls = resample_by_distance(
            dense,
            spacing=HEAVY_ROUGH_SAMPLE_SPACING
        )

        if len(pearls) < 3:
            continue

        try:

            poly = Polygon(pearls)

            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            poly = poly.simplify(
                HEAVY_ROUGH_SIMPLIFY,
                preserve_topology=True
            )

            if poly.is_empty:
                continue

        except Exception:
            continue

        for obj in sorted(
            NATURE,
            key=lambda o: o['offset'],
            reverse=True
        ):

            try:

                outer = poly.buffer(
                    obj['width'],
                    cap_style=1,
                    join_style=1,
                    resolution=4
                )

                inner_offset = obj.get(
                    'offset',
                    0.0
                )

                if inner_offset > 0:

                    inner = poly.buffer(
                        inner_offset,
                        cap_style=1,
                        join_style=1,
                        resolution=4
                    )

                    ring = outer.difference(inner)

                else:

                    ring = outer.difference(poly)

                ring = ring.buffer(0)

                if ring.is_empty:
                    continue

            except Exception:
                continue

            polys = [ring]

            if isinstance(ring, MultiPolygon):
                polys = list(ring.geoms)

            for geom in polys:

                if geom.is_empty:
                    continue

                if not isinstance(geom, Polygon):
                    continue

                pieces = subdivide_poly(geom)

                for piece in pieces:

                    if piece.is_empty:
                        continue

                    if not isinstance(piece, Polygon):
                        continue

                    spline_obj = polygon_to_object_spline(
                        piece,
                        obj['fillPct']
                    )

                    spline_obj = round_dict_floats(
                        spline_obj
                    )

                    nature_nodes[
                        obj['path']
                    ]['Value']['splines'].append(
                        spline_obj
                    )

                    generated_splines += 1

    placed = round_dict_floats(placed)

    if args.dry:

        print(
            f'Generated vegetation splines: '
            f'{generated_splines}'
        )

        print('DRY RUN: no files written')

        return

    backup(placed_path)

    placed_path.write_text(
        json.dumps(
            placed,
            ensure_ascii=False,
            indent=2
        ),
        encoding='utf-8'
    )

    print(
        f'Generated vegetation splines: '
        f'{generated_splines}'
    )

if __name__ == '__main__':
    main()