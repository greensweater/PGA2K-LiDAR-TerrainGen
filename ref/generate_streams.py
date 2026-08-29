#!/usr/bin/env python3

from pathlib import Path
from copy import deepcopy
import argparse
import json
import math

from shapely.geometry import LineString, Polygon, MultiPolygon, box

# ------------------------------------------------------------
# PGA 2K23 procedural stream generator
#
# Generates:
# 1) Stream terrain stamps (userLayers.height)
# 2) Water objects (userLayers.water)
# 3) Waterfall prefabs (placedObjects3)
# 4) Vegetation spline polygons (placedObjects3)
#
# Source:
# surface splines with surface == 8 and open paths
#
# ------------------------------------------------------------

SIZE = 2000
MAP_MIN = -1000
MAP_MAX = 1000

TARGET_SPACING = 4.0
STREAM_DEPTH = 1.5
STREAM_WIDTH = 4.0
WATER_WIDTH = 4.0
WATERFALL_OFFSET = -3.9
WATERFALL_DROP = 0.1
MAX_SPLINE_BOUNDS = 50.0

STREAM_STAMP_TYPE = 73
WATER_TYPE = 72

WATERFALL_PREFAB = "Assets/Effects/Waterfalls/Prefabs/Waterfall_LowFallsPrefab"

NATURE = [
    {
        "path": "Assets/CourseGen/Detail/Grass/TallGrassD",  # flat rock
        "width": 2.4,
        "fillPct": 0.2
    },
    {
        "path": "Assets/CourseGen/Detail/Grass/TallGrassG2", # flat rock
        "width": 2.2,
        "fillPct": 0.3
    },
    {
        "path": "Assets/CourseGen/Detail/Rocks/RP_Rocks/RP_Rock15Prefab", # flat gray rock
        "width": 1.4,
        "fillPct": 0.15
    },
    {
        "path": "Assets/CourseGen/Detail/Rocks/RP_Rocks/RP_Rock13Prefab", # flat gray rock
        "width": 1.8,
        "fillPct": 0.1
    },
    {
        #"path": "Assets/Foliage/SpeedTree/Bushes/NaturePackBushes/Prefabs/NPBush01A", # branchy cluster
        "path": "Assets/Foliage/SpeedTree/Bushes/NaturePackBushes/Prefabs/NPBush03B",  # low spreader
        "width": 1.6,
        "fillPct": 0.1
    }
]
    

# ----------------- Helpers -----------------

def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak.json")
    i = 1

    while bak.exists():
        bak = Path(str(path.with_suffix(path.suffix + f".bak{i}.json")))
        i += 1

    path.rename(bak)
    return bak

def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def normalize(dx, dz):
    mag = math.hypot(dx, dz)

    if mag <= 1e-9:
        return 0.0, 0.0

    return dx / mag, dz / mag

def world_to_pixel_round(x, z):
    px = round(x - MAP_MIN)
    pz = round(MAP_MAX - z)
    return int(px), int(pz)

def clone_height(meta_template, x, z, scale_x, scale_z, stamp_type, height, rot_y):

    clone = deepcopy(meta_template)

    clone.setdefault('position', {})
    clone.setdefault('scale', {})
    clone.setdefault('rotation', {})
    
    clone['tool'] = 1

    clone['position']['x'] = float(x)
    clone['position']['z'] = float(z)

    clone['scale']['x'] = float(scale_x)
    clone['scale']['y'] = 1.0
    clone['scale']['z'] = float(scale_z)

    clone['rotation']['x'] = 0.0
    clone['rotation']['y'] = float(rot_y)
    clone['rotation']['z'] = 0.0

    clone['value'] = float(height)
    clone['type'] = int(stamp_type)

    return clone

def clone_water(template, x, y, z, rot_y):

    water = deepcopy(template)

    water.setdefault('position', {})
    water.setdefault('rotation', {})
    water.setdefault('scale', {})
    water.setdefault('options', {})

    water['surfaceCategory'] = 9

    water['position']['x'] = float(x)
    water['position']['y'] = "-Infinity"
    water['position']['z'] = float(z)

    water['rotation']['x'] = 0.0
    water['rotation']['y'] = float(rot_y)
    water['rotation']['z'] = 0.0

    water['scale']['x'] = 2.0
    water['scale']['y'] = 1.0
    water['scale']['z'] = WATER_WIDTH

    water['type'] = WATER_TYPE
    water['value'] = float(y)

    flow = (180.0 + rot_y) % 360.0

    water['options']['flowOrientation'] = float(flow)
    water['options']['flowSpeed'] = 50

    return water

def make_waterfall_item(x, y, z, rot_y):

    rad = math.radians(rot_y)

    fx = x - math.sin(rad) * WATERFALL_OFFSET
    fz = z - math.cos(rad) * WATERFALL_OFFSET
    

    return {
        "position": {
            "x": float(fx),
            "y": "-Infinity",
            "z": float(fz)
        },
        "rotation": {
            "x": 0.0,
            "y": float(rot_y),
            "z": 0.0
        },
        "scale": {
            "x": 0.5,
            "y": 1.0,
            "z": 1.0
        }
    }

# ----------------- Surface spline helpers -----------------

def _get_surface_value(spline):

    candidates = ["surface"]
    containers = [spline, spline.get("Value", {}), spline.get("value", {})]

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

def _get_is_closed(spline):

    candidates = ["isClosed", "state"]
    containers = [spline, spline.get("Value", {}), spline.get("value", {})]

    for c in containers:

        if not isinstance(c, dict):
            continue

        for k in candidates:

            v = c.get(k)

            if v is None:
                continue

            return bool(v)

    return False

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

def resample_by_distance(points, spacing=TARGET_SPACING):

    if len(points) < 2:
        return points

    pearls = [points[0]]

    accum = 0.0

    for i in range(1, len(points)):

        seg = dist(points[i - 1], points[i])
        accum += seg

        if accum >= spacing:
            pearls.append(points[i])
            accum = 0.0

    return pearls

# ----------------- Height lookup -----------------

def build_height_lookup(height_entries):

    lookup = {}

    for st in height_entries:

        try:
            x = float(st['position']['x'])
            z = float(st['position']['z'])
        except Exception:
            continue

        if not x.is_integer() or not z.is_integer():
            continue

        sc = st.get('scale', {})

        try:
            sx = float(sc.get('x', 0.0))
            sz = float(sc.get('z', 0.0))
        except Exception:
            continue

        if sx == 80.0 or sz == 80.0:
            continue

        px, pz = world_to_pixel_round(x, z)

        lookup[(px, pz)] = float(st.get('value', 0.0))

    return lookup

def find_height(lookup, x, z):

    px, pz = world_to_pixel_round(x, z)

    if (px, pz) in lookup:
        return lookup[(px, pz)]

    for radius in range(1, 100):

        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):

                key = (px + dx, pz + dz)

                if key in lookup:
                    return lookup[key]

    return None

# ----------------- Polygon helpers -----------------

def subdivide_poly(poly, max_size=MAX_SPLINE_BOUNDS):

    minx, miny, maxx, maxy = poly.bounds

    if (maxx - minx) <= max_size and (maxy - miny) <= max_size:
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

    for clipped in [poly.intersection(a), poly.intersection(b)]:

        if clipped.is_empty:
            continue

        if isinstance(clipped, MultiPolygon):
            for geom in clipped.geoms:
                out.extend(subdivide_poly(geom, max_size))
        else:
            out.extend(subdivide_poly(clipped, max_size))

    return out

# ----------------- Object spline generation -----------------

def make_waypoint(x, z):

    return {
        "pointOne": {
            "x": float(x),
            "y": float(z)
        },
        "pointTwo": {
            "x": float(x),
            "y": float(z)
        },
        "waypoint": {
            "x": float(x),
            "y": float(z)
        }
    }

def polygon_to_object_spline(poly, fillPct):

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
        "fillPct": fillPct
    }

# ----------------- Main -----------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('working_folder')
    ap.add_argument('--dry', action='store_true')

    args = ap.parse_args()

    working = Path(args.working_folder)

    userlayers_path = working / 'CourseDescription_nodes' / 'userLayers2.json'

    if not userlayers_path.exists():
        userlayers_path = working / 'CourseDescription_nodes' / 'userLayers.json'

    splines_path = working / 'CourseDescription_nodes' / 'surfaceSplines2.json'

    if not splines_path.exists():
        splines_path = working / 'CourseDescription_nodes' / 'surfaceSplines.json'

    placed_path = working / 'CourseDescription_nodes' / 'placedObjects3.json'

    userlayers = json.loads(userlayers_path.read_text(encoding='utf-8'))
    splines = json.loads(splines_path.read_text(encoding='utf-8'))
    placed = json.loads(placed_path.read_text(encoding='utf-8'))

    height = userlayers.get('height', [])
    water = userlayers.setdefault('water', [])

    height_lookup = build_height_lookup(height)
    print(f"height: {len(height_lookup)}")

    height_template = height[0]

    water_template = {
        "surfaceCategory": 9,
        "position": {"x": 0, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "_orientation": 0.0,
        "scale": {"x": 4, "y": 1, "z": 10},
        "type": 72,
        "value": 0,
        "holeId": -1,
        "options": {
            "flowOrientation": 0,
            "flowSpeed": 50
        },
        "radius": 0,
        "orientation": 0
    }

    waterfall_node = None
    nature_nodes = {}

    for node in placed:

        path = node.get('Key', {}).get('path', '')

        if path == WATERFALL_PREFAB:
            waterfall_node = node

        for obj in NATURE:

            if path == obj['path']:
                nature_nodes[path] = node

    if waterfall_node is None:

        waterfall_node = {
            "Key": {
                "path": WATERFALL_PREFAB
            },
            "Value": {
                "items": [],
                "clusters": [],
                "splines": []
            }
        }

        placed.append(waterfall_node)

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

    generated_heights = 0
    generated_water = 0
    generated_falls = 0
    generated_splines = 0

    for spline in splines:

        surface = _get_surface_value(spline)
        closed = _get_is_closed(spline)

        if surface != 8 or closed:
            continue

        dense = sample_spline_dense(spline)

        if len(dense) < 2:
            continue

        pearls = resample_by_distance(dense)

        if len(pearls) < 2:
            continue

        # ----- terrain + water -----

        prev_h = None

        for i in range(len(pearls) - 1):

            cur = pearls[i]
            nxt = pearls[i + 1]

            dx = nxt[0] - cur[0]
            dz = nxt[1] - cur[1]

            ndx, ndz = normalize(dx, dz)

            rot_y = math.degrees(math.atan2(ndx, ndz))

            terrain_h = find_height(height_lookup, cur[0], cur[1])

            if terrain_h is None:
                terrain_h = prev_h
                
                if terrain_h is None:
                    print("terrain none")
                    continue

            stream_h = terrain_h - STREAM_DEPTH

            height.append(
                clone_height(
                    height_template,
                    cur[0],
                    cur[1],
                    2.0,
                    STREAM_WIDTH,
                    STREAM_STAMP_TYPE,
                    -STREAM_DEPTH,
                    rot_y
                )
            )

            generated_heights += 1

            water.append(
                clone_water(
                    water_template,
                    cur[0],
                    0.0,
                    cur[1],
                    rot_y
                )
            )

            generated_water += 1

            if prev_h is not None:

                drop = prev_h - stream_h

                if True or drop >= WATERFALL_DROP:

                    waterfall_node['Value']['items'].append(
                        make_waterfall_item(
                            cur[0],
                            stream_h -0.15,
                            cur[1],
                            rot_y
                        )
                    )

                    generated_falls += 1

            prev_h = stream_h

        # ----- vegetation polygons -----

        line = LineString(pearls)

        for object in NATURE:
            poly = line.buffer(
                object['width'],
                cap_style=1,
                join_style=1,
                resolution=4
            )

            poly = poly.buffer(0)

            if poly.is_empty:
                continue

            polys = [poly]

            if isinstance(poly, MultiPolygon):
                polys = list(poly.geoms)

            for geom in polys:

                pieces = subdivide_poly(geom)

                for piece in pieces:

                    if piece.is_empty:
                        continue

                    if not isinstance(piece, Polygon):
                        continue

                    nature_nodes[object['path']]['Value']['splines'].append(
                        polygon_to_object_spline(piece, object['fillPct'])
                    )

                    generated_splines += 1

    if args.dry:

        print(f'Generated stream terrain stamps: {generated_heights}')
        print(f'Generated water objects: {generated_water}')
        print(f'Generated waterfall prefabs: {generated_falls}')
        print(f'Generated vegetation splines: {generated_splines}')
        print('DRY RUN: no files written')

        return

    backup(userlayers_path)
    backup(placed_path)

    userlayers_path.write_text(
        json.dumps(userlayers, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    placed_path.write_text(
        json.dumps(placed, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f'Generated stream terrain stamps: {generated_heights}')
    print(f'Generated water objects: {generated_water}')
    print(f'Generated waterfall prefabs: {generated_falls}')
    print(f'Generated vegetation splines: {generated_splines}')

if __name__ == '__main__':
    main()
