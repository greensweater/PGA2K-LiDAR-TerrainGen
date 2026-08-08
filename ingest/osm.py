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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import overpy
import pyproj
import numpy as np
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.affinity import translate
import shapely.vectorized

from terrain.bounding_box import BoundingBox

DEFAULT_HEIGHT_MASK_KINDS = ("fairway", "green", "tee")
DEFAULT_HOLE_CORRIDOR_BUFFER_PX = 10.0
GOLF_OBJECT_KINDS = ("fairway", "green", "tee", "hole")  # mask (=excluded) defaults False for these, True otherwise
DEFAULT_HEIGHT_MASK_BUFFER_PX = 50.0  # see build_height_mask's docstring for what "pixel" means here


@dataclass(slots=True)
class Feature:
    geometry: BaseGeometry
    kind: str
    tags: dict
    osm_id: Optional[int] = None  # the original OSM way ID -- lets a specific feature be targeted/re-selected later
    mask: bool = False  # EXCLUDE this feature? True = leave it out of height_mask.geojson, and (holes
                        # only) leave it out of holes.json too -- e.g. a duplicate hole bleeding in
                        # from a neighboring course. Non-hole splines always export regardless of mask.
                        # Defaults set per-kind at parse time -- see parse_osm_features/GOLF_OBJECT_KINDS.


def classify_way(tags: dict) -> Optional[tuple[str, bool]]:
    """
    Return (kind, is_area) for a way's OSM tags, or None if this way
    isn't a golf-course-relevant feature. Ported from OSMTGC.py's
    addOSMToTGC tag dispatch (see module docstring) -- kept at the
    classification level only; Chad's tool's spline-shaping parameters
    (path width, handle length, tight vs. loose curves) are a later
    concern for whatever actually builds PGA splines from these
    Features, not part of ingest.

    pga_scatter is this project's own tag (not an OSM standard), for
    hand-drawn ways meant to seed a scattered planting -- fescue,
    brambles, a named tree grove, etc. -- rather than a single golf
    surface. Always classified as an area (a scatter needs a region to
    fill, never a bare line) regardless of any area=yes/no tag, with
    kind fixed at "vegetation" for every pga_scatter value -- the
    specific value stays in the Feature's own tags (see
    parse_osm_features) for whatever actually builds the scatter
    (individual tree instances, or ring/fillPct object-splines like
    generate_rough_border_v2.py's NATURE pattern) to read. That
    builder doesn't exist yet -- this is groundwork so pga_scatter ways
    survive ingest (features.geojson) rather than being silently
    dropped, not a claim the scatter itself is implemented.
    """
    scatter_type = tags.get("pga_scatter")
    if scatter_type is not None:
        return ("vegetation", True)

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


def _clip_to_bbox(geometry: BaseGeometry, bbox: BaseGeometry) -> Optional[BaseGeometry]:
    """
    Clip geometry to bbox (not just check intersection and keep it
    whole) -- a feature that only partially overlaps some bounds
    (very common with real OSM data; golf courses aren't neatly
    bounded rectangles) would otherwise carry its entire, unclipped
    extent through to spline export, and the game responds to an
    out-of-bounds spline by silently expanding the whole playfield to
    fit it -- a real bug, confirmed by a course loading in-game at
    over 3330m instead of 2000m.

    Returns None if geometry doesn't overlap bbox at all (caller
    should drop the feature).
    """
    if not geometry.intersects(bbox):
        return None
    clipped = geometry.intersection(bbox)
    if clipped.is_empty:
        return None
    # Intersection can return a Multi*/GeometryCollection if the
    # original crossed the boundary in a complex way (rare, but
    # possible for a long hole routing line or an oddly-shaped
    # fairway) -- keep just the largest piece, since downstream code
    # (feature_to_spline, hole waypoint reduction, corridor buffering)
    # expects one simple Polygon/LineString, not a disconnected
    # multi-part shape.
    if clipped.geom_type not in ("Polygon", "LineString"):
        parts = [g for g in getattr(clipped, "geoms", []) if g.geom_type in ("Polygon", "LineString")]
        if not parts:
            return None
        clipped = max(parts, key=lambda g: g.area if g.geom_type == "Polygon" else g.length)
    return clipped


def crop_features(features: list[Feature], bounds: BoundingBox) -> list[Feature]:
    """
    Clip every feature's geometry to bounds, dropping any with no
    overlap at all. Used at the point of use (write-splines,
    write-holes, height-mask building, the course-cropped OSM preview
    overlay) rather than at parse time -- see parse_osm_features's own
    docstring for why features.geojson itself is kept in the full
    point cloud's frame, uncropped: so a future manually-repositioned
    course crop doesn't need OSM re-parsed from scratch, just re-cropped
    from this same stored, stable feature set.
    """
    bbox = box(bounds.min_x, bounds.min_z, bounds.max_x, bounds.max_z)
    cropped = []
    for f in features:
        clipped_geom = _clip_to_bbox(f.geometry, bbox)
        if clipped_geom is None:
            continue
        cropped.append(replace(f, geometry=clipped_geom))
    return cropped


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
    Parse an OSM XML export into Features, shifted into a local frame
    by (origin_x, origin_y) -- but deliberately NOT cropped to the
    course's own [0, COURSE_SIZE_M] bounds here, even though `bounds`
    can still be given to drop/clip against some other, larger extent
    (e.g. the full merged point cloud's own bounds, to filter out
    anything OSM returned well outside it).

    Cropping to the course-crop bounds specifically happens later, at
    the point of use (see crop_features) -- write-splines,
    write-holes, height-mask building, and the course-cropped preview
    overlay all crop a shifted copy right before they need it. Keeping
    features.geojson itself in the full point cloud's frame,
    uncropped, means a future manually-repositioned course crop (see
    project design notes on selecting the crop within the larger LAZ
    extent) can just re-crop from this same stored set instead of
    needing OSM re-parsed from scratch -- cropping at parse time would
    permanently discard whatever fell outside whichever crop position
    happened to be current when Ingest OSM was run.
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

        if bbox is not None:
            clipped = _clip_to_bbox(geometry, bbox)
            if clipped is None:
                continue
            geometry = clipped

        features.append(Feature(
            geometry=geometry, kind=kind, tags=dict(way.tags), osm_id=way.id,
            mask=(kind not in GOLF_OBJECT_KINDS),
        ))

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
                "properties": {
                    "kind": f.kind, "tags": f.tags, "osm_id": f.osm_id, "mask": f.mask,
                },
            }
            for f in features
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(collection, fh)


def shift_features(features: list[Feature], dx: float, dz: float) -> list[Feature]:
    """
    Translate every feature's geometry by (dx, dz) -- a pure coordinate
    shift, not a reprojection. Used to move Features from the
    course-crop-relative frame they're parsed in (see parse_osm_features)
    into a different local frame that shares the same real-world origin
    but a different (0, 0), e.g. the full merged point cloud's own frame
    used by the LIDAR previews (see PGA2k_gen.py's step_ingest_osm for
    how that shift is derived from the two clouds' own origin_x/origin_y).
    """
    return [
        Feature(geometry=translate(f.geometry, xoff=dx, yoff=dz), kind=f.kind, tags=f.tags,
                osm_id=f.osm_id, mask=f.mask)
        for f in features
    ]


def set_feature_mask(features: list[Feature], osm_id: int, mask: bool) -> bool:
    """Set the mask flag on the feature with this osm_id, in place. Returns True if a match was found."""
    for f in features:
        if f.osm_id == osm_id:
            f.mask = mask
            return True
    return False


def load_features(path: Path) -> list[Feature]:
    with Path(path).open(encoding="utf-8") as fh:
        collection = json.load(fh)
    return [
        Feature(
            geometry=shape(f["geometry"]), kind=f["properties"]["kind"], tags=f["properties"]["tags"],
            osm_id=f["properties"].get("osm_id"), mask=f["properties"].get("mask", False),
        )
        for f in collection["features"]
    ]


def merge_height_mask_features(
    features: list[Feature],
    hole_corridor_buffer_px: Optional[float] = DEFAULT_HOLE_CORRIDOR_BUFFER_PX,
) -> Optional[BaseGeometry]:
    """
    Merge every Feature with mask=False (i.e. NOT excluded) into one
    shape (the unary_union), *before* any buffering. Split out from
    build_height_mask so this (parsing + union, the relatively
    expensive part) can be done once and cached, then buffered
    repeatedly at different distances cheaply -- e.g. for a live GUI
    slider, where re-parsing features.geojson and re-unioning on every
    tick would be wasted, unnecessary work.

    mask=True means EXCLUDE -- fairway/green/tee/hole default to
    mask=False (not excluded, so included here) at parse time (see
    GOLF_OBJECT_KINDS), everything else defaults to mask=True
    (excluded). It's a per-feature, independently toggleable flag
    though -- a masked-out (mask=True) bunker stays out, and
    unmasking one (mask=False) would include it here too, regardless
    of kind.

    hole_corridor_buffer_px, if not None, buffers "hole" Features (a
    routing line, tee-to-green centerline) into a "corridor" polygon
    rather than including the bare centerline -- golfers don't walk a
    perfectly straight line, so the unbuffered centerline alone would
    leave most of the actual playing corridor outside the mask.
    Buffering a LineString directly with shapely's own .buffer()
    already produces exactly this ribbon-shaped polygon; no separate
    line-to-polygon "outline" conversion is needed. Pass None to
    buffer nothing (hole Features contribute just their bare
    centerline instead).
    """
    relevant = []
    for f in features:
        if f.mask:
            continue
        if f.kind == "hole" and hole_corridor_buffer_px is not None:
            relevant.append(f.geometry.buffer(hole_corridor_buffer_px))
        else:
            relevant.append(f.geometry)
    if not relevant:
        return None
    return unary_union(relevant)


def build_height_mask(
    features: list[Feature],
    buffer_px: float = DEFAULT_HEIGHT_MASK_BUFFER_PX,
    hole_corridor_buffer_px: Optional[float] = DEFAULT_HOLE_CORRIDOR_BUFFER_PX,
) -> Optional[BaseGeometry]:
    """
    Merge every Feature with mask=False, i.e. NOT excluded (default:
    fairway/green/tee/hole -- the areas adaptive refinement should
    actually spend effort on -- plus buffered hole-path corridors, see
    merge_height_mask_features) into one shape, then buffer it
    outward -- the "merge filled splines, then buffer to create a
    simple outline" approach.

    buffer_px follows the same 1-pixel-per-meter convention the earlier
    generate_height_mask_v3.py script used for its CANVAS_SIZE=2000
    raster over a 2000-unit map -- our course is likewise exactly
    2000x2000 m, so a pixel and a meter are the same number here, and
    buffer_px is used directly as meters. If a genuinely different
    raster resolution is ever introduced elsewhere, this would need
    converting explicitly rather than assumed equivalent.

    Returns None if no matching features exist (nothing to mask).
    """
    merged = merge_height_mask_features(features, hole_corridor_buffer_px)
    if merged is None:
        return None
    return merged.buffer(buffer_px)


def save_height_mask(geometry: Optional[BaseGeometry], path: Path) -> None:
    """Write the height mask as a single-geometry GeoJSON file, or an explicit null if there's nothing to mask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(mapping(geometry) if geometry is not None else None, fh)


def load_height_mask(path: Path) -> Optional[BaseGeometry]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    return shape(data) if data is not None else None


def rasterize_mask(geometry: Optional[BaseGeometry], bounds: BoundingBox, resolution: int) -> np.ndarray:
    """
    Boolean grid (resolution x resolution, rows=z cols=x -- same
    convention as adaptive_refine.py's error grid) that's True wherever
    a cell's center falls inside `geometry`, False everywhere else.
    Rasterized fresh at whatever resolution/bounds are asked for, since
    the mask is stored as a vector geometry, not tied to one fixed grid.

    If geometry is None (no fairway/green features found), returns an
    all-True grid -- "no mask" means "don't restrict anything", not
    "restrict everything".
    """
    if geometry is None:
        return np.ones((resolution, resolution), dtype=bool)

    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    xx, zz = np.meshgrid(x_centers, z_centers)  # shape (resolution, resolution), rows=z cols=x

    return shapely.vectorized.contains(geometry, xx, zz)


def rasterize_mask_rgba(
    geometry: Optional[BaseGeometry],
    bounds: BoundingBox,
    width_px: int,
    height_px: int,
    color: tuple[int, int, int] = (255, 0, 255),
    opacity: float = 0.3,
    invert: bool = True,
) -> np.ndarray:
    """
    Fast (no matplotlib) RGBA raster of `geometry`, colored `color` at
    `opacity`, for interactive preview use -- a GUI slider redrawing on
    every tick needs something much cheaper than matplotlib's full
    figure/savefig/reload pipeline. Returns a (height_px, width_px, 4)
    uint8 array with row 0 = top (max z), matching normal top-down image
    convention directly -- unlike rasterize_mask's row-0-is-min-z grid,
    which is fine for algorithm-internal use but wouldn't line up with
    a screen image without flipping.

    invert follows the same "rasterize, then invert, then use as alpha"
    recipe as a Photoshop selection-to-mask conversion: with
    invert=True (the default here), the cells *outside* `geometry` end
    up colored/opaque and the inside stays fully transparent; pass
    invert=False to highlight the inside instead.

    geometry=None renders fully transparent (nothing to highlight).
    """
    if geometry is None:
        return np.zeros((height_px, width_px, 4), dtype=np.uint8)

    edges_x = np.linspace(bounds.min_x, bounds.max_x, width_px + 1)
    edges_z = np.linspace(bounds.max_z, bounds.min_z, height_px + 1)  # descending: row 0 = max z (top)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
    xx, zz = np.meshgrid(x_centers, z_centers)

    inside = shapely.vectorized.contains(geometry, xx, zz)
    gray = np.where(inside, 255, 0).astype(np.uint8)
    if invert:
        gray = 255 - gray

    rgba = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (gray.astype(np.float64) * opacity).astype(np.uint8)
    return rgba
