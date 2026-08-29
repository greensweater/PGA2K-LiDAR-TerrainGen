import argparse
import copy
import json
from pathlib import Path

from shapely.geometry import Polygon


BUFFER_WIDTH = 20.0
TARGET_SURFACES = {1, 2}  # 1=fairway, 2=green
OUTPUT_SURFACE = 4        # heavy rough
ROUND_DIGITS = 4


# PGA map coordinates are centered on 0,0 with bounds +/-1000.
MAP_MIN = -1000.0
MAP_MAX = 1000.0
MAP_SIZE = 2000.0


# -----------------------------------------------------------------------------
# Coordinate helpers
# -----------------------------------------------------------------------------

def world_to_local(x_world, z_world):
    """
    Convert PGA world coordinates to local shapely coordinates.
    Shapely operates in a normal Cartesian plane where +Y is upward.
    PGA uses X/Z.
    """
    return (x_world, z_world)



def local_to_world(x_local, y_local):
    return (x_local, y_local)


# -----------------------------------------------------------------------------
# Waypoint helpers
# -----------------------------------------------------------------------------

def _extract_point_from_wp(wp):
    """
    Extract waypoint position.
    Supports both:
      wp['point'] = {x,z}
    and:
      wp['waypoint'] = {x,z}
    """
    point = wp.get("point") or wp.get("waypoint")
    if not point:
        return None

    x = point.get("x")
    z = point.get("y")

    if x is None or z is None:
        return None

    return (float(x), float(z))



def _set_point(target, x, z):
    target["x"] = round(float(x), ROUND_DIGITS)
    target["y"] = round(float(z), ROUND_DIGITS)


# -----------------------------------------------------------------------------
# Polygon helpers
# -----------------------------------------------------------------------------

def spline_to_polygon(spline):
    """
    Convert a surface spline into a shapely polygon.
    Uses waypoint positions only.
    """
    coords = []

    for wp in spline.get("waypoints", []):
        pt = _extract_point_from_wp(wp)
        if pt is None:
            print("pt is None")
            continue

        coords.append(world_to_local(*pt))

    if len(coords) < 3:
        print("len coords < 3")
        return None

    # Remove duplicate closing point if present.
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        print("len coords < 3 pt2")
        return None

    try:
        poly = Polygon(coords)
    except Exception:
        print("Polygon exception")
        return None

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        print("poly is_empty after buffer")
        return None

    return poly



def polygon_to_waypoints(poly, template_waypoint):
    """
    Convert shapely polygon exterior back into PGA waypoints.

    Bezier handles are flattened to the waypoint position.
    This produces clean linear splines.
    """
    coords = list(poly.exterior.coords)

    # Shapely closes polygons by repeating first point.
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    waypoints = []

    for x_local, y_local in coords:
        x_world, z_world = local_to_world(x_local, y_local)

        wp = copy.deepcopy(template_waypoint)

        point = wp.get("point") or wp.get("waypoint")
        if point is None:
            point = {}
            wp["point"] = point

        _set_point(point, x_world, z_world)

        # Flatten bezier handles.
        for handle_key in ["pointOne", "pointTwo"]:
            if handle_key in wp and isinstance(wp[handle_key], dict):
                _set_point(wp[handle_key], x_world, z_world)

        waypoints.append(wp)

    return waypoints


# -----------------------------------------------------------------------------
# Main generation
# -----------------------------------------------------------------------------

def create_heavy_rough_spline(source_spline, buffer_width):
    poly = spline_to_polygon(source_spline)
    if poly is None:
        print(f"poly is None")
        return None

    buffered = poly.buffer(buffer_width, join_style=1)

    if buffered.is_empty:
        print(f"buffered is empty")
        return None

    # If buffer produces multiple polygons, use largest.
    if buffered.geom_type == "MultiPolygon":
        buffered = max(buffered.geoms, key=lambda g: g.area)

    new_spline = copy.deepcopy(source_spline)

    new_spline["surface"] = OUTPUT_SURFACE

    # Preserve existing secondary settings if present.
    if "secondarySurface" in new_spline:
        new_spline["secondarySurface"] = -1

    if "secondaryWidth" in new_spline:
        new_spline["secondaryWidth"] = -1.0

    template_waypoint = None
    for wp in source_spline.get("waypoints", []):
        if _extract_point_from_wp(wp) is not None:
            template_waypoint = wp
            break

    if template_waypoint is None:
        print(f"template_waypoint is None")
        return None

    new_spline["waypoints"] = polygon_to_waypoints(
        buffered,
        template_waypoint,
    )

    return new_spline



def main():
    parser = argparse.ArgumentParser(
        description="Generate heavy rough splines from fairways and greens."
    )

    parser.add_argument(
        "input_json",
        help="Input surfaceSplines2.json",
    )

    parser.add_argument(
        "output_json",
        help="Output surfaceSplines2.json",
    )

    parser.add_argument(
        "--buffer",
        type=float,
        default=BUFFER_WIDTH,
        help="Buffer width in meters (default: 20.0)",
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError("Expected root JSON array.")

    generated = []

    for spline in data:
        surface = spline.get("surface")

        if surface not in TARGET_SURFACES:
            continue

        new_spline = create_heavy_rough_spline(
            spline,
            args.buffer,
        )

        if new_spline is not None:
            generated.append(new_spline)

    print(f"Generated {len(generated)} heavy rough splines.")

    data.extend(generated)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()

