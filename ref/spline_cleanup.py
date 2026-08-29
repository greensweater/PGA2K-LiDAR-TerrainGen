#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
from copy import deepcopy
from math import hypot

# ------------------------------------------------------------
# PGA 2K23 spline cleanup utility
#
# Goals:
#   1) Merge overlapping spline points which cause polygon artifacts.
#   2) Auto-smooth bezier handles (pointOne / pointTwo).
#
# Notes:
#   - Uses waypoint as the canonical spline vertex.
#   - If neighboring waypoints overlap within tolerance, they are merged.
#   - Handles are regenerated symmetrically along the local tangent.
#   - Closed splines are handled circularly.
#   - Original file is backed up automatically.
# ------------------------------------------------------------

MERGE_EPSILON = 0.5
DEFAULT_HANDLE_SCALE = 0.3

# ----------------- Helpers -----------------


def distance(a, b):
    return hypot(a["x"] - b["x"], a["y"] - b["y"])



def lerp(a, b, t):
    return {
        "x": a["x"] + (b["x"] - a["x"]) * t,
        "y": a["y"] + (b["y"] - a["y"]) * t,
    }



def normalize(dx, dy):
    mag = hypot(dx, dy)
    if mag <= 1e-9:
        return 0.0, 0.0
    return dx / mag, dy / mag



def clone_point(pt):
    return {
        "x": float(pt["x"]),
        "y": float(pt["y"]),
    }



def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak.json")
    i = 1
    while bak.exists():
        bak = Path(str(path.with_suffix(path.suffix + f".bak{i}.json")))
        i += 1
    path.rename(bak)
    return bak


# ----------------- Extraction -----------------


def get_waypoint_xy(wp):
    p = wp.get("waypoint", {})
    return {
        "x": float(p.get("x", 0.0)),
        "y": float(p.get("y", 0.0)),
    }



def make_waypoint(center, point_one=None, point_two=None):
    if point_one is None:
        point_one = center
    if point_two is None:
        point_two = center

    return {
        "pointOne": clone_point(point_one),
        "pointTwo": clone_point(point_two),
        "waypoint": clone_point(center),
    }


# ----------------- Overlap merge -----------------

def merge_overlapping_waypoints(waypoints, is_closed, epsilon=MERGE_EPSILON):

    # We only care about first/last overlap when the path is explicitly closed
    if is_closed and len(waypoints) > 2:
        first_center = get_waypoint_xy(waypoints[0])
        last_center = get_waypoint_xy(waypoints[-1])

        if distance(first_center, last_center) <= epsilon:
            # Remove the last waypoint to prevent duplication at the closure point
            # (returns a new list to avoid mutating the original)
            return waypoints[:-1], 1

    # If not closed or endpoints don't overlap, return unchanged
    return list(waypoints), 0


# ----------------- Bezier smoothing -----------------

def auto_smooth_handles(waypoints, is_closed, handle_scale=DEFAULT_HANDLE_SCALE):
    """
    Rebuilds bezier handles using neighboring waypoint tangents.

    For each waypoint:
      tangent = normalize(next - prev)
      pointOne = center - tangent * handle_length
      pointTwo = center + tangent * handle_length

    Handle length is based on neighboring segment lengths.
    """

    count = len(waypoints)

    if count < 2:
        return waypoints

    result = []

    for i in range(count):
        center = get_waypoint_xy(waypoints[i])

        # Endpoint behavior for open splines.
        if not is_closed:
            if i == 0:
                prev_pt = center
                next_pt = get_waypoint_xy(waypoints[i + 1])
            elif i == count - 1:
                prev_pt = get_waypoint_xy(waypoints[i - 1])
                next_pt = center
            else:
                prev_pt = get_waypoint_xy(waypoints[i - 1])
                next_pt = get_waypoint_xy(waypoints[i + 1])
        else:
            prev_pt = get_waypoint_xy(waypoints[(i - 1) % count])
            next_pt = get_waypoint_xy(waypoints[(i + 1) % count])

        dx = next_pt["x"] - prev_pt["x"]
        dy = next_pt["y"] - prev_pt["y"]

        tx, ty = normalize(dx, dy)

        prev_len = distance(center, prev_pt)
        next_len = distance(center, next_pt)

        handle_len = min(prev_len, next_len) * handle_scale

        # Round handle coordinates to 3 decimal places
        point_one = {
            "x": round(center["x"] - tx * handle_len, 3),
            "y": round(center["y"] - ty * handle_len, 3),
        }

        point_two = {
            "x": round(center["x"] + tx * handle_len, 3),
            "y": round(center["y"] + ty * handle_len, 3),
        }

        result.append(make_waypoint(center, point_one, point_two))

    return result


# ----------------- Pipeline -----------------


def process_spline(spline, epsilon, handle_scale):
    is_closed = bool(spline.get("ClosedPath") or spline.get("isClosed"))

    original_waypoints = spline.get("waypoints", [])

    merged_waypoints, removed = merge_overlapping_waypoints(
        original_waypoints,
        is_closed=is_closed,
        epsilon=epsilon,
    )

    smoothed = auto_smooth_handles(
        merged_waypoints,
        is_closed=is_closed,
        handle_scale=handle_scale,
    )

    spline["waypoints"] = smoothed
    
    if spline["surface"] == 10:
        spline["width"] = 1.5

    return removed, len(smoothed)


# ----------------- Main -----------------


def main():
    ap = argparse.ArgumentParser(description="Clean and smooth PGA 2K23 surface splines")

    ap.add_argument("working_folder")

    ap.add_argument(
        "--epsilon",
        type=float,
        default=MERGE_EPSILON,
        help="distance threshold for overlapping points",
    )

    ap.add_argument(
        "--handle-scale",
        type=float,
        default=DEFAULT_HANDLE_SCALE,
        help="bezier handle length multiplier",
    )

    ap.add_argument(
        "--dry",
        action="store_true",
        help="preview only",
    )

    args = ap.parse_args()

    working = Path(args.working_folder)

    spline_path = working / "CourseDescription_nodes" / "surfaceSplines2.json"

    if not spline_path.exists():
        spline_path = working / "CourseDescription_nodes" / "surfaceSplines.json"

    if not spline_path.exists():
        raise SystemExit(f"surfaceSplines.json not found: {spline_path}")

    data = json.loads(spline_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise SystemExit("Expected top-level spline list")

    total_removed = 0

    for spline in data:
        removed, remaining = process_spline(
            spline,
            epsilon=args.epsilon,
            handle_scale=args.handle_scale,
        )

        total_removed += removed

    print(f"Processed splines: {len(data)}")
    print(f"Removed overlapping waypoints: {total_removed}")
    print(f"Handle scale: {args.handle_scale}")
    print(f"Merge epsilon: {args.epsilon}")

    if args.dry:
        print("DRY RUN: no file written")
        return

    bak = backup(spline_path)

    spline_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Backup saved → {bak}")
    print(f"Updated spline file → {spline_path}")


if __name__ == "__main__":
    main()
