import json
import sys
import numpy as np
import argparse
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon, LineString
from pathlib import Path

CANVAS_SIZE = 2000
MAP_MIN, MAP_MAX = -1000, 1000  # Coordinate bounds for x and z

def world_to_pixel(x, z):
    # Flip vertically so north is at the top of the image
    px = int(((x - MAP_MIN) / (MAP_MAX - MAP_MIN)) * (CANVAS_SIZE - 1))
    pz = int(((MAP_MAX - z) / (MAP_MAX - MAP_MIN)) * (CANVAS_SIZE - 1))
    return px, pz

def parse_width(value):
    """Accepts a single float or comma/space-separated list of floats."""
    try:
        # Allow JSON-style arrays like "[10,20,30]"
        if value.startswith("[") and value.endswith("]"):
            return [float(v) for v in value.strip("[]").split(",")]
        # Allow comma-separated or space-separated
        parts = [v for v in value.replace(",", " ").split() if v]
        vals = [float(v) for v in parts]
        return vals if len(vals) > 1 else vals[0]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid width value: {value} ({e})")

def _get_surface_value(spline):
    """
    Try common locations / key names for a surface value. Return int if found,
    otherwise None.
    """
    candidates = ["surface", "Surface", "surfaceType", "surface_type"]
    # also check nested containers often named 'Value' or similar
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
            except (ValueError, TypeError):
                # sometimes floats or strings like "2"
                try:
                    return int(float(v))
                except Exception:
                    continue
    return None
    
def _get_is_closed(spline):
    """
    Try common locations / key names for spline closed-state.
    Return bool if found, otherwise False.
    """
    candidates = ["isClosed", "closed", "IsClosed"]
    containers = [spline, spline.get("Value", {}), spline.get("value", {})]

    for c in containers:
        if not isinstance(c, dict):
            continue

        for k in candidates:
            v = c.get(k)
            if v is None:
                continue

            if isinstance(v, bool):
                return v

            if isinstance(v, (int, float)):
                return bool(v)

            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes")

    return False
    
def _extract_point_from_wp(wp):
    """
    Given a waypoint entry (which may contain keys 'pointOne', 'pointTwo', 'waypoint'),
    return (x,z) as floats if found, otherwise None.
    """
    for key in ("pointOne", "pointTwo", "waypoint"):
        if key in wp and isinstance(wp[key], dict):
            p = wp[key]
            # coordinate naming varies; try x/z, x/y, or xpos/zpos
            for (kx, kz) in (("x", "z"), ("x", "y"), ("xPos", "zPos"), ("posX", "posZ")):
                if kx in p and kz in p:
                    try:
                        return float(p[kx]), float(p[kz])
                    except Exception:
                        continue
            # try x and fallback to z if 'z' missing but 'y' present (seen in original)
            if "x" in p and ("z" in p or "y" in p):
                try:
                    x = float(p["x"])
                    z = float(p.get("z", p.get("y")))
                    return x, float(z)
                except Exception:
                    pass
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

def cubic_bezier_samples(spline, steps=4):

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

        for s in range(steps):
            t = s / float(steps)

            x, z = _cubic_bezier(p0, p1, p2, p3, t)

            px, pz = world_to_pixel(x, z)
            sampled.append((px, pz))

    # ensure final endpoint included
    final_pt = _extract_vec2(waypoints[-1].get("waypoint"))
    if final_pt:
        sampled.append(world_to_pixel(*final_pt))

    return sampled
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('working_folder')
    ap.add_argument('--objects', help='draw objects and stuff')
    ap.add_argument('--width', type=parse_width, default=10, help='width of mask steps (number or array)')
    if len(sys.argv) < 2:
        print("Usage: python generate_map_points.py <working_folder>")
        sys.exit(1)
    args = ap.parse_args()
    working_folder = Path(args.working_folder)
    userlayers_path = working_folder / "CourseDescription_nodes" / "userLayers2.json"
    if not(userlayers_path.exists()):
        userlayers_path = working_folder / "CourseDescription_nodes" / "userLayers.json"
    placedobjects_path = working_folder / "CourseDescription_nodes" / "placedObjects3.json"
    surfacesplines_path = working_folder / "CourseDescription_nodes" / "surfaceSplines2.json"
    if not(surfacesplines_path.exists()):
        surfacesplines_path = working_folder / "CourseDescription_nodes" / "surfaceSplines.json"
    output_file = working_folder / "map_points.png"
    
    # define colors
    colors = [(255,0,255),(0,255,255),(255,255,0),(0,0,255)]
    # build widths
    if isinstance(args.width, (int, float)):
        widths = [args.width] * len(colors)
    else:
        widths = args.width
        if len(widths) < len(colors):
            widths += [widths[-1]] * (len(colors) - len(widths))  # pad if shorter

    # Start with blue background
    img_array = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    img_array[:, :, 2] = 255  # Blue

    # --- Height points ---
    if userlayers_path.exists():
        with open(userlayers_path, "r", encoding="utf-8") as f:
            userlayers2 = json.load(f)

        height = userlayers2.get("height", [])
        min_x, max_x = float("inf"), float("-inf")
        min_z, max_z = float("inf"), float("-inf")

        for entry in height:
            pos = entry.get("position", {})
            try:
                x = float(pos["x"])
                z = float(pos["z"])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_z, max_z = min(min_z, z), max(max_z, z)
                px, pz = world_to_pixel(x, z)
                if 0 <= px < CANVAS_SIZE and 0 <= pz < CANVAS_SIZE:
                    img_array[pz, px] = [64, 64, 255] # lighter blue
            except (KeyError, ValueError, TypeError):
                continue

        print(f"Height data bounding box:")
        print(f"  X range: {min_x:.2f} → {max_x:.2f}")
        print(f"  Z range: {min_z:.2f} → {max_z:.2f}")
        print(f"  Approx size: {max_x - min_x:.2f} x {max_z - min_z:.2f}")
    else:
        print(f"No userLayers found: {working_folder}")

    # --- Placed objects ---
    if args.objects:
        if placedobjects_path.exists():
            with open(placedobjects_path, "r", encoding="utf-8") as f:
                placed_objects = json.load(f)
            object_count = 0
            for obj in placed_objects:
                items = obj.get("Value", {}).get("items", [])
                for item in items:
                    pos = item.get("position", {})
                    try:
                        x = float(pos["x"])
                        z = float(pos["z"])
                        px, pz = world_to_pixel(x, z)
                        if 0 <= px < CANVAS_SIZE and 0 <= pz < CANVAS_SIZE:
                            img_array[pz, px] = [255, 255, 255]  # White
                            object_count += 1
                    except (KeyError, ValueError, TypeError):
                        continue
            print(f"Overlayed {object_count} placed objects.")
        else:
            print("No placedObjects found.")

    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # --- Splines ---
    if surfacesplines_path.exists():
        with open(surfacesplines_path, "r", encoding="utf-8") as f:
            splines = json.load(f)

        splines_sorted = sorted(
            splines,
            key=lambda s: _get_surface_value(s) if _get_surface_value(s) is not None else -1
        )
        
        for i, color in enumerate(colors):
            for spline in splines_sorted:
                
                # Collect ordered points for this spline
                points = cubic_bezier_samples(spline, steps=4)

                surface_val = _get_surface_value(spline)
                is_closed = _get_is_closed(spline)
                
                # Water bodies
                if surface_val == 8 and is_closed and len(points) >= 3:

                    try:
                        poly = Polygon(points)

                        if poly.is_valid and not poly.is_empty:

                            # Interior = low-res
                            draw.polygon(points, fill=(255,0,0))

                            # Edge = preserve highest-res shoreline
                            edge_points = points + [points[0]]
                            draw.line(edge_points, fill=colors[3], width=6)

                    except Exception:
                        pass
                
                elif surface_val == 8 and len(points) > 1:
                    draw.line(points, fill=colors(3), width=8)
                        
                # Regular surfaces
                elif surface_val is not None and 0 <= surface_val <= 3 and len(points) >= 2:
                    poly = LineString(points)
                    expanded = poly.buffer(widths[i] * (len(colors)-i))
                    # Handle both Polygon and MultiPolygon results
                    if isinstance(expanded, Polygon):
                        polys = [expanded]
                    elif isinstance(expanded, MultiPolygon):
                        polys = list(expanded.geoms)
                    else:
                        polys = []

                    for geom in polys:
                        # Draw the exterior ring
                        draw.polygon(list(geom.exterior.coords), fill=color)

                        # Optionally fill interiors (holes) with background color
                        for interior in geom.interiors:
                            draw.polygon(list(interior.coords), fill=color)
                else:
                    if args.objects:
                        draw.line(points, fill=(0, 0, 0))
        print(f"Overlayed {len(splines)} splines.")
    else:
        print("No surfaceSplines found.")

    img.save(output_file)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
