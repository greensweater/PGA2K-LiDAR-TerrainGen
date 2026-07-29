"""
osm.py

Parses an OSM XML export into this compiler's internal vector Feature
representation -- Shapely geometries tagged with a `kind` and the raw
OSM tags, not PGA spline JSON directly (see the architecture doc's
"Internal Feature Representation": "Everything uses the same vector
geometry... No repeated spline parsing").

Classification -- which OSM tag combination maps to which `kind`, and
whether it's an area (polygon) or a line -- is ported from Chad
Rockey's TGC-Designer-Tools (OSMTGC.py's addOSMToTGC), proven,
already-tuned tag dispatch for golf-course OSM tagging conventions:
https://github.com/chadrockey/TGC-Designer-Tools

Coordinate transformation is deliberately NOT ported from that tool,
though: this project already has its own tested LAZ-CRS-based
coordinate pipeline (ingest.laz_reader), including the US-survey-feet
unit-conversion fix that tool's separate GeoPointCloud transform
doesn't share. OSM features need to land in the exact same local
frame the point cloud and stamps already use, not a second,
independently-computed one.

Tree nodes (`natural=tree`) are deliberately not handled here -- the
architecture doc treats tree placement as staying "largely based on
Chad's algorithm... independent of terrain optimization," a separate
concern from this vector-feature ingest.

Output is a features.geojson FeatureCollection (matching the
architecture doc's project artifact list), with "kind" and the raw OSM
"tags" stashed in each feature's properties -- surface splines, terrain
masks, and everything else downstream all read from this same file
rather than re-parsing OSM XML repeatedly (see the doc's "No repeated
spline parsing").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import overpy
import pyproj
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry

from terrain.bounding_box import BoundingBox


@dataclass(slots=True)
class Feature:
    geometry: BaseGeometry
    kind: str
    tags: dict


def classify_way(tags: dict) -> Optional[tuple[str, bool]]:
    """
    Return (kind, is_area) for a way's OSM tags, or None if this way
    isn't a golf-course-relevant feature. Ported from OSMTGC.py's
    addOSMToTGC tag dispatch (see module docstring) -- kept at the
    classification level only; Chad's tool's spline-shaping parameters
    (path width, handle length, tight vs. loose curves) are a later
    concern for whatever actually builds PGA splines from these
    Features, not part of ingest.
    """
    golf_type = tags.get("golf")
    waterway_type = tags.get("waterway")
    building_type = tags.get("building")
    natural_type = tags.get("natural")
    highway_type = tags.get("highway")
    golf_cart_type = tags.get("golf_cart")
    foot_type = tags.get("foot")
    amenity_type = tags.get("amenity")
    explicit_area = tags.get("area") == "yes"

    if golf_type is not None:
        if golf_type in ("green", "tee"):
            return (golf_type, True)
        if golf_type == "bunker":
            return ("bunker", True)
        if golf_type in ("fairway", "driving_range"):
            return ("fairway", True)
        if golf_type == "rough":
            return ("rough", True)
        if golf_type in ("water_hazard", "lateral_water_hazard"):
            # Golf-tagged water hazards are always filled polygons in
            # Chad's tool (newWaterHazard(nds, area=True)), unlike
            # waterway=* below, which respects the area tag.
            return ("water", True)
        if golf_type == "cartpath":
            return ("cartpath", explicit_area)
        if golf_type == "path":
            return ("path", explicit_area)
        if golf_type == "clubhouse":
            return ("building", True)
        if golf_type == "hole":
            return ("hole", False)
        return None

    if waterway_type is not None:
        # A plain waterway (stream, ditch, ...) defaults to a
        # centerline, not a filled polygon, unless explicitly tagged
        # area=yes -- matches Chad's area=area (not area=True) here.
        return ("water", explicit_area)

    if building_type is not None:
        return ("building", True)

    if natural_type == "wood":
        return ("wood", True)

    if highway_type is not None and highway_type not in ("proposed", "construction"):
        implicit_foot_access = {
            "motorway": "no", "motorway_link": "no", "trunk": "no", "trunk_link": "no",
        }
        way_foot_access = foot_type if foot_type is not None else implicit_foot_access.get(highway_type, "yes")
        if golf_cart_type not in (None, "no"):
            return ("cartpath", explicit_area)
        if way_foot_access != "no":
            return ("path", explicit_area)
        return None

    if amenity_type == "parking" and golf_cart_type not in (None, "no"):
        return ("cartpath", True)

    return None


def latlon_to_local(
    lat: float, lon: float,
    origin_x: float, origin_y: float, horizontal_unit_factor: float,
    transformer: pyproj.Transformer,
) -> tuple[float, float]:
    """
    Project a lat/lon into this compiler's local course frame -- the
    same true-meters, origin-aligned frame ingest.laz_reader.PointCloud
    uses (see module docstring). `transformer` should already be built
    from EPSG:4326 to the course's own CRS (see parse_osm_features).
    """
    native_x, native_y = transformer.transform(lon, lat)
    return native_x * horizontal_unit_factor - origin_x, native_y * horizontal_unit_factor - origin_y


def parse_osm_features(
    osm_xml_path: Path,
    crs: pyproj.CRS,
    origin_x: float,
    origin_y: float,
    horizontal_unit_factor: float,
    bounds: Optional[BoundingBox] = None,
    printf=print,
) -> list[Feature]:
    """
    Parse an OSM XML export into Features in the local course frame.

    If `bounds` is given, ways that don't intersect it are dropped
    (same "off of map, skip" idea as Chad's tool, but checked against
    our own course bounds rather than the LAZ extent).
    """
    xml_data = Path(osm_xml_path).read_text(encoding="utf-8")
    result = overpy.Overpass().parse_xml(xml_data)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    bbox = box(bounds.min_x, bounds.min_z, bounds.max_x, bounds.max_z) if bounds is not None else None

    features: list[Feature] = []
    skipped_nodes = 0
    for way in result.ways:
        classification = classify_way(way.tags)
        if classification is None:
            continue
        kind, is_area = classification

        try:
            nodes = way.get_nodes(resolve_missing=False)
        except overpy.exception.OverPyException:
            skipped_nodes += 1
            continue

        coords = [
            latlon_to_local(float(n.lat), float(n.lon), origin_x, origin_y, horizontal_unit_factor, transformer)
            for n in nodes
        ]
        if len(coords) < 2:
            continue

        if is_area and len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            geometry: BaseGeometry = Polygon(coords)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)  # common fix for self-intersecting OSM rings
        else:
            geometry = LineString(coords)

        if bbox is not None and not geometry.intersects(bbox):
            continue

        features.append(Feature(geometry=geometry, kind=kind, tags=dict(way.tags)))

    if skipped_nodes:
        printf(f"Skipped {skipped_nodes} way(s) with unresolvable nodes "
               "(referenced nodes not included in this OSM export)")
    printf(f"Parsed {len(features)} golf-course-relevant feature(s) from {len(result.ways)} way(s)")
    return features


def save_features(features: list[Feature], path: Path) -> None:
    """Write features as a GeoJSON FeatureCollection (features.geojson)."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(f.geometry),
                "properties": {"kind": f.kind, "tags": f.tags},
            }
            for f in features
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(collection, fh)


def load_features(path: Path) -> list[Feature]:
    with Path(path).open(encoding="utf-8") as fh:
        collection = json.load(fh)
    return [
        Feature(geometry=shape(f["geometry"]), kind=f["properties"]["kind"], tags=f["properties"]["tags"])
        for f in collection["features"]
    ]
