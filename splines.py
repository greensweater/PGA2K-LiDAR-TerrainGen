"""
splines.py

Generates PGA surface splines (surfaceSplines.json) from this
project's own Feature objects (see ingest/osm.py), using the spline-
construction algorithm and per-surface-type parameters (width, handle
length, tight-vs-loose curves, secondary blend surface) from Chad
Rockey's TGC-Designer-Tools (OSMTGC.py's newSpline family), reverse-
engineered/tuned against real in-game testing:
https://github.com/chadrockey/TGC-Designer-Tools

Scope: green/tee/fairway/rough/bunker/cartpath/path/building/wood.

Water is deliberately excluded here -- Chad's own approach fills water
hazards with a placeholder "mulch" surface spline (confirmed directly
in his source: newWaterHazard sets surface=surface2 "as a placeholder"),
which is exactly the awkward workaround this project's roadmap already
flagged wanting to replace with dedicated water-body handling (auto-
fitted water objects + a real elevation subroutine), not another mulch
spline.

"hole" is also excluded: Chad's hole object is a much richer gameplay/
scoring structure (tee/pin positions, radii, par) than the simple
routing linestring our OSM ingest currently captures, and needs its
own dedicated construction step later.

Ported algorithm notes (see _build_spline):
  - Winding direction (clockwise vs not) is computed on the ORIGINAL
    points, before any shrinking -- matches Chad's own call order
    (splineIsClockWise runs right after building raw waypoints, before
    shrinkSplineNormals mutates them).
  - Tangent angles for the Bezier handles are computed from the
    ALREADY-SHRUNK points, not the original ones -- in Chad's own code,
    completeSpline() reads positions back out of the same spline_json
    dict that shrinkSplineNormals() already mutated in place, so the
    (unused) `points` parameter it's called with is actually dead code;
    the real data flow uses shrunk positions throughout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from ingest.osm import Feature
from writer import GRID_ORIGIN_OFFSET

# From tgc_definitions.py (confirmed via direct re-fetch, no drift from
# an earlier fetch): surface name -> numeric ID the game expects.
FEATURES_TO_SURFACES = {
    "bunker": 0,
    "green": 1,
    "fairway": 2,
    "rough": 3,
    "heavyrough": 4,
    "clearobjects": 5,
    "cleartrees": 6,
    "surface1": 7,
    "surface2": 8,
    "water": 9,
    "surface3": 10,
    "cartpath": 10,
}

# Per-kind spline parameters, matching Chad's own newBunker/newGreen/
# newTeeBox/newFairway/newRough/newBuilding/newForest exactly (surface
# name here refers to FEATURES_TO_SURFACES keys, not our own Feature.kind).
_STATIC_SPLINE_PARAMS: dict[str, dict] = {
    "bunker": dict(surface="bunker", path_width=0.01, handle_length=1.0,
                   tight_splines=True, secondary_surface="heavyrough", secondary_width=2.5),
    "green": dict(surface="green", path_width=1.7, handle_length=0.2,
                  tight_splines=True, secondary_surface="heavyrough", secondary_width=2.5),
    # Tees are output as surface=green (confirmed: newTeeBox sets
    # surface to featuresToSurfaces["green"], not a dedicated tee ID --
    # there isn't one), just via a separately-nameable spline_json key.
    "tee": dict(surface="green", path_width=1.7, handle_length=0.2,
                tight_splines=True, secondary_surface="heavyrough", secondary_width=2.5),
    "fairway": dict(surface="fairway", path_width=3.0, handle_length=3.0,
                     tight_splines=False, secondary_surface="rough", secondary_width=5.0),
    "rough": dict(surface="rough", path_width=1.7, handle_length=3.0,
                   tight_splines=False, secondary_surface="", secondary_width=0.0),
    "building": dict(surface="surface2", path_width=0.01, handle_length=0.2,
                      tight_splines=True, secondary_surface="", secondary_width=0.0),
    "wood": dict(surface="surface1", path_width=0.01, handle_length=0.2,
                  tight_splines=True, secondary_surface="", secondary_width=0.0),
}


def _tangent_angle(p: tuple[float, float], n: tuple[float, float]) -> float:
    return math.atan2(n[1] - p[1], n[0] - p[0])


def _is_clockwise(points: list[tuple[float, float]]) -> bool:
    """https://stackoverflow.com/questions/1165647 -- same shoelace-sign test Chad's tool uses."""
    edge_sum = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x0, y0 = points[i - 1]
        edge_sum += (x1 - x0) * (y1 + y0)
    return edge_sum >= 0.0


def _shrink_normals(
    points: list[tuple[float, float]], shrink_distance: float, is_clockwise: bool,
) -> list[tuple[float, float]]:
    """
    Move every point inward along its local normal by shrink_distance --
    compensates for the game expanding every spline outward by its own
    width when rendering (it treats all splines like filled cartpaths).
    """
    if not shrink_distance:
        return list(points)
    n = len(points)
    result = []
    for i in range(n):
        p = points[i - 1]
        t = points[i]
        nxt = points[(i + 1) % n]
        tangent_angle = _tangent_angle(p, nxt)
        normal_angle = tangent_angle - math.pi / 2.0
        if not is_clockwise:
            normal_angle += math.pi
        result.append((
            t[0] + math.cos(normal_angle) * shrink_distance,
            t[1] + math.sin(normal_angle) * shrink_distance,
        ))
    return result


def _build_waypoints(
    shrunk_points: list[tuple[float, float]], handle_length: float,
    is_clockwise: bool, tight_splines: bool,
) -> list[dict]:
    n = len(shrunk_points)
    waypoints = []
    for i in range(n):
        p = shrunk_points[i - 1]
        t = shrunk_points[i]
        nxt = shrunk_points[(i + 1) % n]
        angle = _tangent_angle(p, nxt)
        if tight_splines:
            # Pull handles perpendicular and inward -- tight, low-
            # smoothing curves that hug the source shape closely.
            angle_one = angle - 1.1 * math.pi / 2.0
            angle_two = angle - 0.9 * math.pi / 2.0
            if not is_clockwise:
                angle_one, angle_two = angle_two + math.pi, angle_one + math.pi
        else:
            # Loose, smooth splines.
            angle_one = angle + math.pi
            angle_two = angle
        waypoints.append({
            "pointOne": {
                "x": round(t[0] + handle_length * math.cos(angle_one), 3),
                "y": round(t[1] + handle_length * math.sin(angle_one), 3),
            },
            "pointTwo": {
                "x": round(t[0] + handle_length * math.cos(angle_two), 3),
                "y": round(t[1] + handle_length * math.sin(angle_two), 3),
            },
            "waypoint": {"x": round(t[0], 3), "y": round(t[1], 3)},
        })
    return waypoints


def _build_spline(
    points: list[tuple[float, float]],
    surface: str,
    path_width: float = 0.01,
    shrink_distance: Optional[float] = None,
    handle_length: float = 0.5,
    tight_splines: bool = True,
    secondary_surface: str = "",
    secondary_width: float = 0.0,
    state: int = 3,
    is_closed: bool = True,
    is_filled: bool = True,
) -> dict:
    is_clockwise = _is_clockwise(points)
    if shrink_distance is None:
        shrink_distance = path_width / 2.0
    shrunk = _shrink_normals(points, shrink_distance, is_clockwise)
    waypoints = _build_waypoints(shrunk, handle_length, is_clockwise, tight_splines)
    return {
        "surface": FEATURES_TO_SURFACES[surface],
        "secondarySurface": FEATURES_TO_SURFACES.get(secondary_surface, 11),
        "secondaryWidth": secondary_width,
        "waypoints": waypoints,
        "width": path_width,
        "state": state,
        "ClosedPath": False,  # matches Chad's own template -- never toggled by any surface type
        "isClosed": is_closed,
        "isFilled": is_filled,
    }


def feature_to_spline(feature: Feature) -> Optional[dict]:
    """
    Build one PGA spline dict from a single Feature, or None if this
    feature's kind isn't handled by this writer (water, hole -- see
    module docstring).
    """
    geom = feature.geometry
    is_area = geom.geom_type == "Polygon"
    raw_points = list(geom.exterior.coords[:-1]) if is_area else list(geom.coords)
    # PGA's grid is centered on the origin ([-1000, 1000] for a 2000 m
    # course); this compiler works in a local [0, COURSE_SIZE_M] frame
    # throughout, same as the terrain stamps (see writer.py's
    # GRID_ORIGIN_OFFSET / module docstring). Splines were missing this
    # same conversion -- every spline coordinate was being written raw
    # in [0, 2000] space instead, offset +1000 from where the game
    # expects it, which is what was causing the game to expand the
    # whole playfield to fit them.
    points = [(x - GRID_ORIGIN_OFFSET, z - GRID_ORIGIN_OFFSET) for x, z in raw_points]
    if len(points) < 2:
        return None

    if feature.kind in _STATIC_SPLINE_PARAMS:
        return _build_spline(points, **_STATIC_SPLINE_PARAMS[feature.kind])

    if feature.kind == "cartpath":
        return _build_spline(
            points, surface="cartpath", path_width=2.0,
            shrink_distance=None if is_area else 0.0,
            handle_length=4.0, tight_splines=False,
            secondary_surface="", secondary_width=0.0,
            state=3 if is_area else 0, is_closed=is_area, is_filled=is_area,
        )

    if feature.kind == "path":
        return _build_spline(
            points, surface="surface1", path_width=1.7,
            shrink_distance=None if is_area else 0.0,
            handle_length=2.0, tight_splines=False,
            secondary_surface="rough", secondary_width=0.0,
            state=3 if is_area else 0, is_closed=is_area, is_filled=is_area,
        )

    return None  # water, hole, or anything unrecognized -- see module docstring


_BEZIER_CIRCLE_KAPPA = 0.5522847498307936  # standard 4-anchor-point cubic-bezier circle approximation


def _circle_spline(cx: float, cz: float, radius: float) -> dict:
    """
    A near-perfect circle (4 anchor points -- N/E/S/W -- with cubic-
    bezier handles sized by the standard "kappa" constant; numerically
    verified to track a true circle to within ~0.03% of radius) as a
    cart path spline, for registration marks (see
    writer.py's build_registration_mark_stamps for the matching
    terrain bump at the same position).

    Reuses _build_spline/_build_waypoints rather than bespoke bezier
    math: with exactly 4 equally-spaced points on a circle, the
    existing tangent-from-neighbors formula (_tangent_angle(prev,
    next), used for every other spline kind) already gives the
    mathematically correct tangent direction at each of the 4 points --
    the chord between a point's two neighbors on a circle is parallel
    to the true tangent at that point when equally spaced -- so no
    circle-specific tangent handling was needed, only the right
    handle_length.

    shrink_distance=0.0 (not the usual path-width-based default) keeps
    the radius exact -- the whole point of a registration mark is
    precise, known geometry.

    cx, cz are in the local [0, COURSE_SIZE_M] frame, same as every
    other input to this module -- shifted by GRID_ORIGIN_OFFSET here,
    same as feature_to_spline does, since this bypasses
    feature_to_spline entirely (there's no Feature/OSM geometry behind
    a registration mark) and would otherwise skip that conversion.
    """
    cx -= GRID_ORIGIN_OFFSET
    cz -= GRID_ORIGIN_OFFSET
    points = [
        (cx, cz + radius),  # N
        (cx + radius, cz),  # E
        (cx, cz - radius),  # S
        (cx - radius, cz),  # W
    ]
    return _build_spline(
        points, surface="cartpath", path_width=1.7,
        shrink_distance=0.0, handle_length=radius * _BEZIER_CIRCLE_KAPPA,
        tight_splines=False, secondary_surface="", secondary_width=0.0,
        state=1, is_closed=True, is_filled=False,
    )


def build_registration_mark_splines(course_size_m: float) -> list[dict]:
    """
    4 circle splines (cart path surface), one at each course corner --
    the spline counterpart to writer.py's
    build_registration_mark_stamps, at the exact same corner positions
    (registration_mark_corners), for visually confirming in-game that
    terrain and splines land where expected relative to each other.
    """
    from writer import registration_mark_corners, REGISTRATION_MARK_CIRCLE_RADIUS_M
    return [
        _circle_spline(x, z, REGISTRATION_MARK_CIRCLE_RADIUS_M)
        for x, z in registration_mark_corners(course_size_m)
    ]


def build_surface_splines(features: list[Feature]) -> list[dict]:
    """
    One spline per handled Feature (see feature_to_spline for which
    kinds are supported -- water/hole are not). mask is NOT checked
    here at all -- every feature that feature_to_spline can handle
    gets exported regardless of its own mask value. mask only affects
    height_mask.geojson membership (see ingest/osm.py's
    merge_height_mask_features) and, separately, hole export (see
    write_holes) -- it was never meant to gate spline output for
    fairway/green/tee/bunker/etc.
    """
    splines = []
    for f in features:
        spline = feature_to_spline(f)
        if spline is not None:
            splines.append(spline)
    return splines


def save_surface_splines(splines: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(splines, f, indent=2)
