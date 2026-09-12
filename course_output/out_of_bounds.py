"""
course_output/out_of_bounds.py

Auto-generated out-of-bounds (OOB) paint, laid down with terrain
BRUSHES -- no splines. PGA Tour 2K marks OOB by painting the
userLayers.json "outOfBounds" layer with brush stamps; an entry there
is geometrically identical to a "height" stamp entry (see
course_output/userLayers.py's stamp_to_entry) but carries NO "tool"
field and its "value" is pinned to 0.0 -- it is a paint marker, not a
height. OOB entries are never fed to TerrainModel, so they are their
own record type here, not terrain.stamp.Stamp objects.

Shape (confirmed against a blank .course saved from the editor with a
hand-painted OOB strip): a centerline band with round caps --

    o========o========o========o
    ^        ^                  ^
  type 8   type 15            type 8
  (cap)    (stretched square)  (cap)

one round brush (type 8) stamp at each vertex of the (simplified)
course-boundary polyline, plus one "smooth square" (type 15) stamp
stretched along each edge between consecutive vertices, rotated to the
edge bearing. One stamp == one placement.

build_oob_records() takes the playable-area union (the same shape the
height mask is built from -- fairway/green/tee + hole corridors),
buffers it outward by an inner buffer, then walks a curve running down
the middle of a fixed-width band just outside that buffer. The band
width, inner buffer and how closely the band hugs the boundary wiggles
(simplify tolerance) are all knobs.

The along-path/across-path scale convention matches
terrain/cart_paths.py: local +z is along the path, local +x is across
it (the band width), and rotation.y = degrees(atan2(dx, dz)).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from shapely.geometry import LineString, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from course_output.game_versions import DEFAULT_GAME_VERSION, schema_for
from course_output.userLayers import GRID_ORIGIN_OFFSET, POSITION_Y, _round
from terrain.bounding_box import BoundingBox
from terrain.cart_paths import TYPE15_PLATEAU_PX, TYPE15_TEXTURE_PX

# Brush ids -- confirmed from the sample .course.
OOB_ROUND_BRUSH = 8       # circular brush, used for the caps at each vertex
OOB_SQUARE_BRUSH = 15     # "smooth/blurred square", stretched along each edge

# Distance (m) the playable-area union is buffered outward before the
# OOB band starts -- the "optional buffer" between play and OOB.
OOB_INNER_BUFFER_M = 8.0

# Painted width (m) of the OOB band, across the boundary.
OOB_BAND_WIDTH_M = 15.0

# Morphological-closing radius (m) applied to the playable union before
# the perimeter is traced: bridges gaps up to ~2x this between adjacent
# playfield pieces (fairway <-> tee <-> next fairway) so the OOB line
# follows the OUTER course boundary instead of looping every fragment.
# Does not grow the overall footprint. 0 disables.
OOB_MERGE_GAP_M = 45.0

# Playfield parts smaller than this (m^2), after closing, are dropped --
# keeps a stray isolated tee/sliver from getting its own OOB loop.
OOB_MIN_PART_AREA_M2 = 1000.0

# Douglas-Peucker tolerance (m) applied to the boundary curve before it
# is walked -- higher = fewer, longer square stamps and a coarser OOB
# line that cuts corners; lower = follows every wiggle of the mask.
OOB_SIMPLIFY_TOL_M = 8.0

# Cap (type 8) half-extent as a fraction of the square's across-path
# half-extent. ~0.8 in the sample (cap scale ~42 vs square scale.x ~53).
# Tunable: type 8's own falloff differs from type 15's, so confirm the
# corner coverage in-game and adjust.
OOB_CAP_SCALE_RATIO = 0.8

# How far (m) each square stamp is extended past its edge endpoints, so
# consecutive squares overlap through the caps rather than just meeting.
OOB_SEGMENT_OVERLAP_M = 1.0

OOB_INCLUDE_CAPS = True

_OOB_JSON_INDENT = 2

# type 15's plateau is TYPE15_PLATEAU_PX / TYPE15_TEXTURE_PX of its full
# footprint; to make the PLATEAU span a desired width, the stamp's
# center-to-edge scale is (width / 2) * TEXTURE / PLATEAU -- exactly the
# reasoning behind terrain/cart_paths.py's CART_PATH_STAMP_RADIUS.
_TYPE15_SCALE_PER_HALF_WIDTH = TYPE15_TEXTURE_PX / TYPE15_PLATEAU_PX


@dataclass(slots=True)
class OOBRecord:
    """One OOB brush placement, in the course-local [0, COURSE_SIZE_M] frame."""
    x: float
    z: float
    scale_x: float
    scale_z: float
    rotation: float
    brush: int


def _iter_linestrings(geom: Optional[BaseGeometry]) -> Iterable[LineString]:
    """Flatten any geometry into its component LineStrings (boundaries, multiparts)."""
    if geom is None or geom.is_empty:
        return
    gtype = geom.geom_type
    if gtype == "LineString":
        yield geom
    elif gtype == "LinearRing":
        yield LineString(geom.coords)
    elif gtype in ("MultiLineString", "GeometryCollection", "MultiPolygon"):
        for part in geom.geoms:
            yield from _iter_linestrings(part)
    elif gtype == "Polygon":
        yield from _iter_linestrings(geom.boundary)
    # points / empty -> nothing


def _iter_polygons(geom: Optional[BaseGeometry]) -> Iterable[BaseGeometry]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            yield from _iter_polygons(part)


def build_oob_records(
    playable_union: Optional[BaseGeometry],
    *,
    inner_buffer_m: float = OOB_INNER_BUFFER_M,
    band_width_m: float = OOB_BAND_WIDTH_M,
    merge_gap_m: float = OOB_MERGE_GAP_M,
    min_part_area_m2: float = OOB_MIN_PART_AREA_M2,
    simplify_tol_m: float = OOB_SIMPLIFY_TOL_M,
    cap_scale_ratio: float = OOB_CAP_SCALE_RATIO,
    segment_overlap_m: float = OOB_SEGMENT_OVERLAP_M,
    include_caps: bool = OOB_INCLUDE_CAPS,
    exteriors_only: bool = True,
    course_bounds: Optional[BoundingBox] = None,
) -> list[OOBRecord]:
    """
    Build the OOB brush placements for a constant-width band sitting
    immediately outside the playable-area perimeter.

    playable_union is the pre-buffer union of the playable Features
    (ingest.osm.merge_height_mask_features) in the course-local frame.
    It is first morphologically closed by merge_gap_m (bridging gaps
    between adjacent playfield pieces without growing the footprint),
    tiny parts are dropped, then the result is buffered outward by
    inner_buffer_m and a curve down the middle of a band_width_m band
    just outside that is walked. With exteriors_only, interior holes
    (ponds/bunkers inside fairways) get no OOB loop.

    Returns [] if there's nothing to work with.
    """
    if playable_union is None or playable_union.is_empty:
        return []
    if band_width_m <= 0:
        return []

    merged = playable_union
    if merge_gap_m and merge_gap_m > 0:
        merged = merged.buffer(merge_gap_m).buffer(-merge_gap_m)

    parts = [p for p in _iter_polygons(merged) if p.area >= max(0.0, min_part_area_m2)]
    if not parts:
        parts = list(_iter_polygons(merged))
    if not parts:
        return []

    inner = unary_union([p.buffer(max(0.0, inner_buffer_m)) for p in parts])
    if inner.is_empty:
        return []

    # The curve running down the middle of the band -- offset outward
    # from `inner` by half the band width, so the whole band lies
    # outside `inner`. exteriors_only: skip interior holes.
    band = inner.buffer(band_width_m / 2.0)
    if exteriors_only:
        centerline = unary_union([
            LineString(p.exterior.coords) for p in _iter_polygons(band)
        ])
    else:
        centerline = band.boundary

    clip = None
    if course_bounds is not None:
        clip = box(course_bounds.min_x, course_bounds.min_z,
                   course_bounds.max_x, course_bounds.max_z)

    half_across = (band_width_m / 2.0) * _TYPE15_SCALE_PER_HALF_WIDTH
    cap_scale = half_across * cap_scale_ratio

    records: list[OOBRecord] = []
    for line in _iter_linestrings(centerline):
        piece = line
        if clip is not None:
            piece = line.intersection(clip)
        for seg in _iter_linestrings(piece):
            if simplify_tol_m > 0:
                seg = seg.simplify(simplify_tol_m)
            coords = list(seg.coords)
            if len(coords) < 2:
                continue
            closed = (
                math.hypot(coords[0][0] - coords[-1][0], coords[0][1] - coords[-1][1])
                < 1e-6
            )
            cap_coords = coords[:-1] if closed else coords

            if include_caps:
                for (cx, cz) in cap_coords:
                    records.append(OOBRecord(
                        x=float(cx), z=float(cz),
                        scale_x=cap_scale, scale_z=cap_scale,
                        rotation=0.0, brush=OOB_ROUND_BRUSH,
                    ))

            for (ax, az), (bx, bz) in zip(coords[:-1], coords[1:]):
                dx, dz = bx - ax, bz - az
                length = math.hypot(dx, dz)
                if length < 1e-6:
                    continue
                records.append(OOBRecord(
                    x=float((ax + bx) / 2.0), z=float((az + bz) / 2.0),
                    scale_x=half_across,
                    scale_z=length / 2.0 + segment_overlap_m,
                    rotation=math.degrees(math.atan2(dx, dz)),
                    brush=OOB_SQUARE_BRUSH,
                ))

    return records


def oob_records_to_entries(
    records: Sequence[OOBRecord | dict], game_version: str = DEFAULT_GAME_VERSION,
) -> list[dict]:
    """
    Format OOB records into userLayers.json "outOfBounds" array entries.

    Same shape as course_output.userLayers.stamp_to_entry MINUS the
    "tool" field, with "value" pinned to 0.0. "_orientation" /
    "orientation" (both 0.0) are added only on schemas that carry them
    (v2021+), exactly as stamp_to_entry does for "height".
    """
    has_orientation = schema_for(game_version).has_orientation_fields
    entries: list[dict] = []
    for rec in records:
        r = rec if isinstance(rec, OOBRecord) else OOBRecord(**rec)
        entry = {
            "position": {
                "x": _round(r.x - GRID_ORIGIN_OFFSET),
                "y": POSITION_Y,
                "z": _round(r.z - GRID_ORIGIN_OFFSET),
            },
            "rotation": {"x": 0.0, "y": _round(r.rotation), "z": 0.0},
        }
        if has_orientation:
            entry["_orientation"] = 0.0
        entry.update({
            "scale": {"x": _round(r.scale_x), "y": 1.0, "z": _round(r.scale_z)},
            "type": int(r.brush),
            "value": 0.0,
            "holeId": -1,
            "radius": 0.0,
        })
        if has_orientation:
            entry["orientation"] = 0.0
        entries.append(entry)
    return entries


def save_oob_records(records: Sequence[OOBRecord], path: Path) -> None:
    """Write oob.json -- a plain JSON list, same convention as streams.json / parking.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in records], fh, indent=_OOB_JSON_INDENT)


def load_oob_records(path: Path) -> list[OOBRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [OOBRecord(**d) for d in raw]
