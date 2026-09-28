"""
course_output/clear_objects.py

"Clear generated objects" paint: fills tagged polygon splines with
brush stamps that stop the game's own procedural scatter (trees,
plants, grass, rocks) from spawning inside them. Our own placed
objects (LIDAR trees, clusters, ...) are NOT affected -- in-game, the
paint only clears the game's scatter (V2023_SCHEMA.md "In-game
results"), and generator-side suppression was deliberately not added
(Andy, 2026-09-27: paint only).

Storage (confirmed in v2019/v2021/v2023 editor exports, see
V2023_SCHEMA.md "Three clear-stamp categories"): entries in the
userLayers(2).json "surfaces" array with surfaceCategory 5 (clear
generated objects). The same array also holds category 6 (clear trees)
and 11 (clear heavy rough, v2023 only) painted in the editor; we only
emit 5.

Fill: each polygon is rasterised on a cell_m grid aligned to its
minimum rotated rectangle (so a rotated rectangular area fills with a
handful of stamps, not a staircase), cells whose centre falls inside the
polygon are merged into horizontal runs, and identical runs in adjacent
rows are merged into rectangles. Each rectangle is one type-72 hard
square stamp -- type 72's edge sits exactly at +-scale (the
scale_test1.course ruler test), so scale = half the rectangle's side and
the painted edge is within cell_m / 2 of the polygon boundary.

Rotation convention matches out_of_bounds.py / terrain/cart_paths.py:
local +z runs along (sin r, cos r), rotation.y = r in degrees.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from course_output.game_versions import DEFAULT_GAME_VERSION, schema_for
from course_output.userLayers import (
    CLEAR_OBJECTS_CATEGORY, GRID_ORIGIN_OFFSET, POSITION_Y, _round,
)
from terrain.bounding_box import BoundingBox

# This project's own tag (not an OSM standard) on a polygon spline in
# features.geojson marking it to be filled with clear-objects stamps.
# Set from the GUI's Splines tab; carried across OSM re-ingest like
# pga_cluster_fills.
PGA_CLEAR_OBJECTS_TAG = "pga_clear_objects"

CLEAR_HARD_SQUARE_BRUSH = 72   # hard square: edge exactly at +-scale
CLEAR_STAMP_VALUE = 1.0        # what the editor writes for categories 5/6

# Grid cell (m). The painted edge lands within cell_m / 2 of the
# polygon boundary; smaller cells follow curves more closely at the cost
# of more stamps.
CLEAR_CELL_M = 2.0

# Each stamp's half-extent is grown by this much (m) so adjacent
# rectangles overlap instead of meeting on a shared edge -- guards
# against hairline gaps in the game's own rasterisation of the paint.
CLEAR_OVERLAP_M = 0.25

_CLEAR_JSON_INDENT = 2


@dataclass(slots=True)
class ClearRecord:
    """One clear-objects brush placement, in the course-local [0, COURSE_SIZE_M] frame."""
    x: float
    z: float
    scale_x: float
    scale_z: float
    rotation: float
    brush: int = CLEAR_HARD_SQUARE_BRUSH
    category: int = CLEAR_OBJECTS_CATEGORY


def _iter_polygons(geom: Optional[BaseGeometry]) -> Iterable[BaseGeometry]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def _grid_bearing_deg(poly: BaseGeometry) -> float:
    """Bearing (deg) of the longest edge of poly's minimum rotated rectangle."""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return 0.0
    coords = list(rect.exterior.coords)
    best, best_len = (0.0, 1.0), -1.0
    for (ax, az), (bx, bz) in zip(coords[:-1], coords[1:]):
        length = math.hypot(bx - ax, bz - az)
        if length > best_len:
            best, best_len = (bx - ax, bz - az), length
    return math.degrees(math.atan2(best[0], best[1])) % 180.0


def _rectangles(inside: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Merge a boolean (rows, cols) grid into rectangles (row0, row1, col0,
    col1), half-open: horizontal runs per row, then a run continues an
    open rectangle from the row below when its (col0, col1) matches.
    """
    done: list[tuple[int, int, int, int]] = []
    open_rects: dict[tuple[int, int], int] = {}   # (col0, col1) -> row0
    n_rows = inside.shape[0]
    for r in range(n_rows + 1):
        runs: set[tuple[int, int]] = set()
        if r < n_rows:
            row = inside[r]
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
            runs = {(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])}
        for key in [k for k in open_rects if k not in runs]:
            done.append((open_rects.pop(key), r, key[0], key[1]))
        for key in runs:
            open_rects.setdefault(key, r)
    return done


def _fill_polygon(poly: BaseGeometry, cell_m: float, overlap_m: float) -> list[ClearRecord]:
    bearing = _grid_bearing_deg(poly)
    rad = math.radians(bearing)
    v_axis = np.array([math.sin(rad), math.cos(rad)])   # local +z (along)
    u_axis = np.array([math.cos(rad), -math.sin(rad)])  # local +x (across)

    pts = np.asarray(poly.exterior.coords)
    us, vs = pts @ u_axis, pts @ v_axis
    u0, v0 = us.min(), vs.min()
    n_cols = max(1, int(math.ceil((us.max() - u0) / cell_m)))
    n_rows = max(1, int(math.ceil((vs.max() - v0) / cell_m)))

    cu = u0 + (np.arange(n_cols) + 0.5) * cell_m
    cv = v0 + (np.arange(n_rows) + 0.5) * cell_m
    gu, gv = np.meshgrid(cu, cv)                        # (rows, cols)
    gx = gu * u_axis[0] + gv * v_axis[0]
    gz = gu * u_axis[1] + gv * v_axis[1]
    inside = shapely.contains_xy(poly, gx, gz)

    records: list[ClearRecord] = []
    for row0, row1, col0, col1 in _rectangles(inside):
        mu = u0 + (col0 + col1) / 2.0 * cell_m
        mv = v0 + (row0 + row1) / 2.0 * cell_m
        records.append(ClearRecord(
            x=float(mu * u_axis[0] + mv * v_axis[0]),
            z=float(mu * u_axis[1] + mv * v_axis[1]),
            scale_x=(col1 - col0) * cell_m / 2.0 + overlap_m,
            scale_z=(row1 - row0) * cell_m / 2.0 + overlap_m,
            rotation=bearing,
        ))
    return records


def build_clear_records(
    regions: Sequence[BaseGeometry],
    *,
    cell_m: float = CLEAR_CELL_M,
    overlap_m: float = CLEAR_OVERLAP_M,
    course_bounds: Optional[BoundingBox] = None,
) -> list[ClearRecord]:
    """
    Fill every polygon in `regions` (course-local frame) with type-72
    clear-objects stamps (each grown by overlap_m, so the painted edge
    is within cell_m / 2 + overlap_m of the boundary). Regions are
    clipped to course_bounds first;
    lines/points and empty geometry are ignored. Returns [] if nothing
    fills.
    """
    if cell_m <= 0:
        raise ValueError(f"cell_m must be > 0 (got {cell_m})")
    clip = None
    if course_bounds is not None:
        clip = box(course_bounds.min_x, course_bounds.min_z,
                   course_bounds.max_x, course_bounds.max_z)

    records: list[ClearRecord] = []
    for region in regions:
        geom = region.intersection(clip) if clip is not None else region
        for poly in _iter_polygons(geom):
            records.extend(_fill_polygon(poly, cell_m, max(0.0, overlap_m)))
    return records


def clear_records_to_entries(
    records: Sequence[ClearRecord | dict], game_version: str = DEFAULT_GAME_VERSION,
) -> list[dict]:
    """
    Format records into userLayers(2).json "surfaces" entries, in the
    shape the editor itself writes: v2023 has surfaceCategory/position/
    rotation/scale/type/value/holeId only (no "tool"); v2019/v2021 add
    "_orientation"/"radius"/"orientation" (both versions' exports carry
    them, even v2019 whose height entries we write without -- see
    V2023_SCHEMA.md "v2019/v2021 storage").
    """
    full_shape = schema_for(game_version).has_radius_field
    entries: list[dict] = []
    for rec in records:
        r = rec if isinstance(rec, ClearRecord) else ClearRecord(**rec)
        entry = {
            "surfaceCategory": int(r.category),
            "position": {
                "x": _round(r.x - GRID_ORIGIN_OFFSET),
                "y": POSITION_Y,
                "z": _round(r.z - GRID_ORIGIN_OFFSET),
            },
            "rotation": {"x": 0.0, "y": _round(r.rotation), "z": 0.0},
        }
        if full_shape:
            entry["_orientation"] = 0.0
        entry.update({
            "scale": {"x": _round(r.scale_x), "y": 1.0, "z": _round(r.scale_z)},
            "type": int(r.brush),
            "value": CLEAR_STAMP_VALUE,
            "holeId": -1,
        })
        if full_shape:
            entry["radius"] = 0.0
            entry["orientation"] = 0.0
        entries.append(entry)
    return entries


def save_clear_records(records: Sequence[ClearRecord], path: Path) -> None:
    """Write clear_objects.json -- a plain JSON list, same convention as oob.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in records], fh, indent=_CLEAR_JSON_INDENT)


def load_clear_records(path: Path) -> list[ClearRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ClearRecord(**d) for d in raw]
