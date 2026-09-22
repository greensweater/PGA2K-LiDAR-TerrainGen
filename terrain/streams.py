"""
terrain/streams.py

Stream generation: turns an OSM stream/ditch centerline (a Shapely
LineString, already in the course's local [0, COURSE_SIZE_M] frame) into

  1. a "pearl necklace" of directionally-rotated terrain Stamps that
     carve a downhill streambed (flatten-to-absolute-value, type 73), and
  2. a frozen, version-agnostic "stream record" (see build_stream_records)
     holding the pearl chain, a single flow-orientation bearing, and the
     subset of pearls where the bed drops sharply enough to warrant a
     waterfall prefab.

Same split as terrain/cart_paths.py: this module is pure algorithm over
this project's own primitives (terrain.stamp.Stamp, Shapely geometry,
the real heightmap array). All I/O -- reading features.geojson /
heightmap.npz, writing stamps_N.json / streams.json -- lives in
PGA2k_gen.py's step_generate_streams. Water-plane strips
(course_output/water.py) and waterfall placed-objects
(course_output/objects.py) are formatted from the stream record at
write time, against the terrain as it stands then, exactly like the
rest of the "compile once, format at write" pipeline.

Deliberately NOT ported from ref/generate_streams.py: the trench uses
tool=flatten valued at an absolute bed height (idempotent -- re-running
generate-streams replaces its own layer and lands the same terrain),
not tool=raise with a -depth delta (which would dig another 1.5 m
deeper every run). The bed height itself is a cumulative-minimum sweep
of the sampled ground height in flow order, so the channel never runs
uphill even where the raw LIDAR surface briefly does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import json
import numpy as np
from shapely.geometry import LineString

from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

# Circular "hard round" brush -- present in viz/brushes/brush_profiles.json
# (so TerrainModel can evaluate it, unlike cart_paths.py's type 15) and a
# SHAPE_CIRCLE, so scale_x must equal scale_z (see terrain/stamp.py).
STREAM_STAMP_BRUSH = 73

STREAM_DEPTH_M = 1.0            # how far below original grade the bed is cut
STREAM_HALF_WIDTH_M = 2.0      # trench stamp radius (center-to-edge); channel reads ~4 m wide
PEARL_SPACING_M = 2.5           # along-centerline pearl spacing -- < 2*radius so stamps overlap

# Flowing-water-tile geometry (consumed by course_output/water.py's
# build_stream_water_objects). Stream water tiles are the SAME type-72
# objects as pond planes -- perfectly horizontal, carrying a real
# elevation -- so a descending stream can't be one tile: it's cut into
# short ELEVATION BANDS (each tile drops at most STREAM_WATER_TILE_DROP_M
# of bed). Because each tile is horizontal, the further its bed has
# dropped the deeper (and wider) the water sits -- hence the width grows
# both with the tile's own local depth and with total descent from the
# source. The carved channel's banks clip the oversized plane back to the
# visible water. (v2025's elevation-changing water splines will replace
# this per-version formatting; streams.json stays the source of truth.)
STREAM_WATER_TILE_DROP_M = 0.6        # max bed descent covered by one horizontal tile
STREAM_WATER_TILE_MAX_LEN_M = 200.0   # ...also cap tile length (near-flat reaches), safety
STREAM_WATER_TILE_MAX_TURN_DEG = 35.0  # ...also break where the heading swings this far (don't cut bends)
STREAM_WATER_FILL_DEPTH_M = 0.25      # FALLBACK-ONLY water film depth above the (raw-heightmap) bed --
                                     # the real write path now fits each tile's level to the actual
                                     # carved TerrainModel (see stream_water_tiles' bed_sampler,
                                     # STREAM_WATER_LEVEL_* below); still feeds the deep_end_depth
                                     # width term.
STREAM_WATER_BASE_WIDTH_M = 4.5      # cross-flow span at zero extra depth
STREAM_WATER_WIDEN_PER_DEPTH = 2.0   # + this * (water depth at the tile's deep end), meters
STREAM_WATER_WIDEN_PER_DESCENT = 0.2  # + this * (total bed descent from the stream source), meters
STREAM_WATER_OVERLAP_M = 6.0         # each tile's length = run length + this, so adjacent tiles overlap
STREAM_WATER_FLOW_SPEED = 37.0       # options.flowSpeed (real stream tile ~37; 50 is the engine max)

# Terrain-fit water level (same sampling idea as course_output/water.py's
# pond _water_level_from_way_boundary, but percentile-based rather than
# mean-based -- a stream's centerline runs along the channel's deepest
# line, not a real shoreline, so it has no reason to cluster around the
# target level the way a pond's own tagged boundary does): a stream water
# tile is written at the P-th percentile of the ACTUAL carved terrain
# height sampled along the tile's centerline, minus a small margin, so
# the plane sits just inside the channel and can't float. Only the
# fallback path uses STREAM_WATER_FILL_DEPTH_M.
STREAM_WATER_LEVEL_PERCENTILE = 15.0       # low percentile of centerline terrain samples (not strict min)
STREAM_WATER_LEVEL_MARGIN_M = -0.2        # subtracted from that percentile -- negative lifts the surface
                                         # a visible film ABOVE the deepest carved bed (tuned in-game)
STREAM_WATER_LEVEL_SAMPLE_SPACING_M = 1.0  # arc-length spacing of the centerline sample points

# The seam between two consecutive horizontal water tiles emits a
# waterfall + splash prefab group when the upper tile's surface sits at
# least this far above the lower tile's (tiles that split on a bend or
# the length cap rather than a real drop are skipped). Tiles are the
# natural spacing, so no separate thinning pass is needed.
WATERFALL_MIN_DROP_M = 0.3

# The "low falls" prefab's pivot sits this far BELOW the water surface it
# pours from -- so the waterfall's y is (upper tile surface - this),
# applied uniformly to every fall. Measured ~1.0 m against a hand-placed
# 2019 sample (falls y 4.261 vs its water tile 5.255). The splash prefab
# instead sits right on the surface it lands on (lower tile), no drop.
WATERFALL_LIP_DROP_M = 1.0

# Centerline buffer (each side) for the stream-bank vegetation polygon
# that step_generate_streams tags with cluster fills.
BANK_VEG_WIDTH_M = 2.0

STREAMS_JSON_INDENT = 2


@dataclass
class StreamCenterline:
    """One stream/ditch centerline, reduced to a Shapely LineString in
    the course's local frame. `source_id` is the OSM way id (for
    diagnostics / back-reference); `waterway` is the raw tag value
    ("stream" or "ditch")."""
    line: LineString
    source_id: Optional[int] = None
    waterway: str = "stream"


def _height_sampler(heights: np.ndarray, bounds: BoundingBox):
    """Returns average_height(x, z, radius_m) -> Optional[float], the
    mean of finite heightmap cells within radius_m of (x, z). Copied
    from terrain/cart_paths.py's generate_cart_path_stamps -- same cell
    grid math, same "None where a real LIDAR gap" policy."""
    n_rows, n_cols = heights.shape
    cell_x = (bounds.max_x - bounds.min_x) / n_cols
    cell_z = (bounds.max_z - bounds.min_z) / n_rows
    x_centers = bounds.min_x + (np.arange(n_cols) + 0.5) * cell_x
    z_centers = bounds.min_z + (np.arange(n_rows) + 0.5) * cell_z
    # A probe disc smaller than the cell diagonal can fall entirely
    # between cell centers (real heightmaps are ~1 m/px so this never
    # bites there, but a coarse grid otherwise loses whole streams).
    min_radius_m = 1.5 * math.hypot(cell_x, cell_z)

    def average_height(x: float, z: float, radius_m: float) -> Optional[float]:
        radius_m = max(radius_m, min_radius_m)
        col_min = max(0, int((x - radius_m - bounds.min_x) / cell_x))
        col_max = min(n_cols, int((x + radius_m - bounds.min_x) / cell_x) + 1)
        row_min = max(0, int((z - radius_m - bounds.min_z) / cell_z))
        row_max = min(n_rows, int((z + radius_m - bounds.min_z) / cell_z) + 1)
        if col_min >= col_max or row_min >= row_max:
            return None
        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)
        within = np.hypot(xx - x, zz - z) <= radius_m
        sub_heights = heights[row_min:row_max, col_min:col_max]
        valid = within & np.isfinite(sub_heights)
        if not valid.any():
            return None
        return float(np.mean(sub_heights[valid]))

    return average_height


def _bearing_deg(dx: float, dz: float) -> float:
    """Compass bearing of a direction vector -- 0 = +Z (north), 90 = +X
    (east), sweeping clockwise. Matches terrain/cart_paths.py and
    course_output/water.py's rotation convention (atan2(dx, dz))."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def resample_centerline(line: LineString, spacing_m: float = PEARL_SPACING_M) -> list[tuple[float, float]]:
    """Evenly-spaced points along `line` (endpoints included), via
    Shapely's own interpolate -- the geometry itself is never modified."""
    length = line.length
    if length <= 0:
        return []
    n = max(2, int(length / spacing_m) + 1)
    return [(p.x, p.y) for p in (line.interpolate(d) for d in np.linspace(0.0, length, n))]


def orient_downhill(
    line: LineString, average_height, probe_radius_m: float = STREAM_HALF_WIDTH_M,
) -> Optional[LineString]:
    """Return `line` (or its reverse) so that walking its vertices runs
    downhill -- the direction water actually flows, which fixes the
    flow-orientation bearing and the "which pearl does a waterfall sit
    at" question. Returns None if neither endpoint has finite heightmap
    data nearby (can't tell which way is down)."""
    coords = list(line.coords)
    h_start = average_height(coords[0][0], coords[0][1], probe_radius_m)
    h_end = average_height(coords[-1][0], coords[-1][1], probe_radius_m)
    if h_start is None or h_end is None:
        return None
    if h_end > h_start:
        return LineString(coords[::-1])
    return line


def streambed_profile(
    pearls: list[tuple[float, float]], average_height, depth_m: float = STREAM_DEPTH_M,
    probe_radius_m: float = STREAM_HALF_WIDTH_M,
) -> list[Optional[float]]:
    """Per-pearl carved bed height: sampled mean ground height, swept to
    a running minimum in flow order (so the bed only ever descends),
    then lowered by depth_m. Entries are None where a pearl has no
    finite heightmap data nearby -- the caller drops those pearls."""
    ground = [average_height(x, z, probe_radius_m) for (x, z) in pearls]
    bed: list[Optional[float]] = []
    running_min: Optional[float] = None
    for g in ground:
        if g is None:
            bed.append(None)
            continue
        running_min = g if running_min is None else min(running_min, g)
        bed.append(running_min - depth_m)
    return bed


def _pearl_rotations(pearls: list[tuple[float, float]]) -> list[float]:
    """Compass bearing from each pearl toward the next (the last pearl
    reuses the previous bearing)."""
    rots: list[float] = []
    for i in range(len(pearls)):
        a = pearls[i]
        b = pearls[min(i + 1, len(pearls) - 1)]
        dx, dz = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dz) <= 1e-9:
            rots.append(rots[-1] if rots else 0.0)
        else:
            rots.append(_bearing_deg(dx, dz))
    return rots


def segment_stream_pearls_by_elevation(
    pearls: list[tuple[float, float, float]],
    max_drop_m: float = STREAM_WATER_TILE_DROP_M,
    max_len_m: float = STREAM_WATER_TILE_MAX_LEN_M,
    max_turn_deg: float = STREAM_WATER_TILE_MAX_TURN_DEG,
) -> list[list[tuple[float, float, float]]]:
    """Cut a pearl chain (each pearl (x, z, bed_h)) into consecutive runs
    (>= 2 pearls, overlapping by one shared pearl) for horizontal water
    tiles: a new run starts once the bed has descended more than
    max_drop_m from the run's start, OR the run passes max_len_m, OR the
    heading has swung more than max_turn_deg from the run's first leg."""
    if len(pearls) < 2:
        return []
    segments: list[list[tuple[float, float, float]]] = []
    cur = [pearls[0]]
    seg_len = 0.0
    base_heading: Optional[float] = None
    for prev, p in zip(pearls, pearls[1:]):
        dx, dz = p[0] - prev[0], p[1] - prev[1]
        leg = math.hypot(dx, dz)
        heading = _bearing_deg(dx, dz) if leg > 1e-9 else (base_heading or 0.0)
        if base_heading is None:
            base_heading = heading
        turn = abs((heading - base_heading + 180.0) % 360.0 - 180.0)
        drop = cur[0][2] - p[2]
        if len(cur) >= 2 and (drop > max_drop_m or seg_len + leg > max_len_m or turn > max_turn_deg):
            segments.append(cur)
            cur = [prev]
            seg_len = 0.0
            base_heading = heading
        cur.append(p)
        seg_len += leg
    if len(cur) >= 2:
        segments.append(cur)
    return segments


@dataclass
class StreamWaterTile:
    """One horizontal type-72 water tile along a stream -- the rectangle
    course_output/water.py's _water_entry renders. All lengths in meters,
    course-local frame; `level` is PRE height-normalization (add
    output_height_shift_m at write time, exactly like a pearl's bed_h).

    `bearing` is the compass heading of the band's upstream->downstream
    chord (0 = +Z, clockwise) and is EXACTLY the tile's written
    rotation.y. `rendered_length` (band chord + the overlap pad that
    becomes scale.z) is the tile's full along-flow extent, so
    `edge_offset(+1)` is the downstream edge center and `edge_offset(-1)`
    the upstream edge center -- the anchor a waterfall / splash prefab
    hangs on."""
    cx: float
    cz: float
    bearing: float
    band_length: float          # upstream->downstream chord of the elevation band
    width_m: float              # cross-flow footprint (before _water_entry's /1.8)
    level: float                # water surface = band upstream bed + fill depth, pre-shift
    flow_orientation: float     # options.flowOrientation (leg circular mean + 180)

    @property
    def rendered_length(self) -> float:
        return self.band_length + STREAM_WATER_OVERLAP_M

    def edge_offset(self, sign: float) -> tuple[float, float]:
        """Center of the downstream (sign=+1) or upstream (sign=-1) short
        edge of the rendered rectangle."""
        rad = math.radians(self.bearing)
        half = 0.5 * self.rendered_length
        return (self.cx + math.sin(rad) * half * sign,
                self.cz + math.cos(rad) * half * sign)


def _fitted_tile_level(
    ax: float, az: float, bx: float, bz: float,
    bed_sampler: Callable[[np.ndarray], np.ndarray],
    percentile: float, margin_m: float, spacing_m: float,
) -> Optional[float]:
    """Water level for a tile whose centerline runs (ax, az) -> (bx, bz):
    the `percentile`-th percentile of the actual terrain height sampled
    every `spacing_m` along that chord, minus `margin_m` -- same dense-
    sampling approach as course_output/water.py's
    _water_level_from_way_boundary for ponds, but percentile- rather
    than mean-based (see STREAM_WATER_LEVEL_PERCENTILE's own comment for
    why). None if the sampler returns nothing finite (caller falls
    back)."""
    length = math.hypot(bx - ax, bz - az)
    n = max(2, int(math.ceil(length / spacing_m)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    pts = np.column_stack((ax + (bx - ax) * ts, az + (bz - az) * ts))
    heights = np.asarray(bed_sampler(pts), dtype=float)
    heights = heights[np.isfinite(heights)]
    if heights.size == 0:
        return None
    return float(np.percentile(heights, percentile)) - margin_m


def stream_water_tiles(
    pearls_bed: list[tuple[float, float, float]], source_bed: float,
    *,
    bed_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fill_depth_m: float = STREAM_WATER_FILL_DEPTH_M,
    base_width_m: float = STREAM_WATER_BASE_WIDTH_M,
    widen_per_depth: float = STREAM_WATER_WIDEN_PER_DEPTH,
    widen_per_descent: float = STREAM_WATER_WIDEN_PER_DESCENT,
    level_margin_m: float = STREAM_WATER_LEVEL_MARGIN_M,
    level_percentile: float = STREAM_WATER_LEVEL_PERCENTILE,
    level_sample_spacing_m: float = STREAM_WATER_LEVEL_SAMPLE_SPACING_M,
) -> list[StreamWaterTile]:
    """The chain of horizontal water tiles for one stream -- the single
    source of tile geometry, shared by course_output/water.py (renders
    them as type-72 planes) and build_stream_records (hangs a waterfall +
    splash on the seam between consecutive tiles). `pearls_bed` is the
    stream's (x, z, bed_h) pearl chain in downhill order; `source_bed` is
    the stream's own first-pearl bed height (drives the widen-with-total-
    descent term).

    `bed_sampler(points (N,2)) -> heights (N,)` -- when given, each tile's
    `.level` is fit to the ACTUAL terrain (percentile of samples along the
    tile centerline minus `level_margin_m`), exactly like a pond; the
    result is in whatever frame the sampler returns. When omitted (or a
    tile has no finite sample), `.level` falls back to the raw-heightmap
    `bed_up + fill_depth_m`. The width/geometry kwargs default to the
    module constants but are overridable per-project (persisted by
    step_generate_streams, re-read by step_write_water / write-objects)."""
    tiles: list[StreamWaterTile] = []
    for segment in segment_stream_pearls_by_elevation(pearls_bed):
        (ax, az, bed_up), (bx, bz, bed_down) = segment[0], segment[-1]
        length = math.hypot(bx - ax, bz - az)
        if length < 1e-6:
            continue
        legs = [
            math.atan2(p2[0] - p1[0], p2[1] - p1[1])
            for p1, p2 in zip(segment, segment[1:])
        ]
        flow_orientation = (math.degrees(math.atan2(
            sum(math.sin(a) for a in legs), sum(math.cos(a) for a in legs),
        )) + 180.0) % 360.0
        deep_end_depth = max(0.0, (bed_up + fill_depth_m) - bed_down)
        total_descent = max(0.0, source_bed - bed_down)
        level = None
        if bed_sampler is not None:
            level = _fitted_tile_level(
                ax, az, bx, bz, bed_sampler,
                level_percentile, level_margin_m, level_sample_spacing_m,
            )
        if level is None:
            level = bed_up + fill_depth_m
        tiles.append(StreamWaterTile(
            cx=0.5 * (ax + bx), cz=0.5 * (az + bz),
            bearing=_bearing_deg(bx - ax, bz - az),
            band_length=length,
            width_m=(
                base_width_m
                + widen_per_depth * deep_end_depth
                + widen_per_descent * total_descent
            ),
            level=level,
            flow_orientation=flow_orientation,
        ))
    return tiles


def _pearls_bed_rots(
    stream: StreamCenterline, average_height, spacing_m: float,
    depth_m: float = STREAM_DEPTH_M, probe_radius_m: float = STREAM_HALF_WIDTH_M,
):
    """Shared front half of generate_stream_stamps / build_stream_records:
    orient downhill, resample, compute bed profile, drop dead-data
    pearls. Returns (pearls, bed, rotations) -- all three lists aligned,
    or (None, None, None) if the stream can't be used."""
    line = orient_downhill(stream.line, average_height, probe_radius_m)
    if line is None:
        return None, None, None
    raw_pearls = resample_centerline(line, spacing_m)
    if len(raw_pearls) < 2:
        return None, None, None
    raw_bed = streambed_profile(raw_pearls, average_height, depth_m, probe_radius_m)
    pearls = [p for p, b in zip(raw_pearls, raw_bed) if b is not None]
    bed = [b for b in raw_bed if b is not None]
    if len(pearls) < 2:
        return None, None, None
    return pearls, bed, _pearl_rotations(pearls)


def generate_stream_stamps(
    streams: list[StreamCenterline], heights: np.ndarray, bounds: BoundingBox,
    spacing_m: float = PEARL_SPACING_M, half_width_m: float = STREAM_HALF_WIDTH_M,
    depth_m: float = STREAM_DEPTH_M, brush: int = STREAM_STAMP_BRUSH, printf=print,
) -> list[Stamp]:
    """One flatten Stamp per usable pearl, valued at that pearl's
    absolute carved bed height (see streambed_profile). scale_x ==
    scale_z (circular brush). Rotation is recorded for parity with the
    water strips but doesn't affect a circular stamp's footprint."""
    average_height = _height_sampler(heights, bounds)
    stamps: list[Stamp] = []
    skipped = 0
    for stream in streams:
        pearls, bed, rots = _pearls_bed_rots(
            stream, average_height, spacing_m, depth_m, half_width_m,
        )
        if pearls is None:
            skipped += 1
            continue
        for (x, z), bed_h, rot in zip(pearls, bed, rots):
            stamps.append(Stamp(
                x=float(x), z=float(z), scale_x=float(half_width_m), scale_z=float(half_width_m),
                value=float(bed_h), brush=brush, rotation=float(rot), tool=TOOL_FLATTEN,
            ))
    if skipped:
        printf(f"  {skipped} stream(s) skipped -- no usable heightmap data along the centerline")
    return stamps


def stream_drop_rows(
    tiles: list[StreamWaterTile], waterfall_min_drop_m: float = WATERFALL_MIN_DROP_M,
    lip_drop_m: float = WATERFALL_LIP_DROP_M,
) -> tuple[list[list[float]], list[list[float]]]:
    """(waterfalls, splashes) for one stream's tile chain -- one pair per
    consecutive tile seam where the upper tile's surface sits at least
    `waterfall_min_drop_m` above the lower tile's. Each row is
    [x, z, rot_y, level] in the tiles' own `level` frame: anchor x/z is
    the UPPER tile's downstream short-edge centre, rot_y its bearing; the
    waterfall's level is (upper surface - lip_drop_m), the splash's is the
    lower tile surface. The two objects coincide in x/z and rotation."""
    waterfalls: list[list[float]] = []
    splashes: list[list[float]] = []
    for upper, lower in zip(tiles, tiles[1:]):
        if upper.level - lower.level < waterfall_min_drop_m:
            continue
        ex, ez = upper.edge_offset(+1.0)
        rot_y = round(upper.bearing, 3)
        waterfalls.append([round(ex, 3), round(ez, 3), rot_y, round(upper.level - lip_drop_m, 3)])
        splashes.append([round(ex, 3), round(ez, 3), rot_y, round(lower.level, 3)])
    return waterfalls, splashes


def build_stream_records(
    streams: list[StreamCenterline], heights: np.ndarray, bounds: BoundingBox,
    spacing_m: float = PEARL_SPACING_M, waterfall_min_drop_m: float = WATERFALL_MIN_DROP_M,
    depth_m: float = STREAM_DEPTH_M, half_width_m: float = STREAM_HALF_WIDTH_M,
    water_fill_depth_m: float = STREAM_WATER_FILL_DEPTH_M,
    water_base_width_m: float = STREAM_WATER_BASE_WIDTH_M,
    water_widen_per_depth: float = STREAM_WATER_WIDEN_PER_DEPTH,
    water_widen_per_descent: float = STREAM_WATER_WIDEN_PER_DESCENT,
    printf=print,
) -> list[dict]:
    """Frozen streams.json payload -- one record per usable stream:

        {"source_id": int | None,
         "waterway": "stream" | "ditch",
         "pearls": [[x, z, rot_y, bed_h], ...],     # course frame, downhill order
         "flow_orientation": <compass bearing deg>,
         "waterfalls": [[x, z, rot_y, level], ...],  # one per seam between consecutive
                                                     # water tiles whose surfaces differ
                                                     # by >= waterfall_min_drop_m
         "splashes":   [[x, z, rot_y, level], ...]}  # 1:1 with waterfalls, SAME x/z/rot_y

    A waterfall + its splash are placed as a rigid group on the shared
    edge of two adjacent water tiles (see stream_water_tiles): x/z is the
    centre of the UPPER tile's downstream short edge (tile centre shifted
    by half its rendered length along its rotation), rot_y is that tile's
    bearing (== its written rotation.y). The two objects coincide in x/z
    and rotation; only `level` differs -- the waterfall's is the UPPER
    tile surface minus WATERFALL_LIP_DROP_M (the prefab pivot sits ~1 m
    below the water it pours from), the splash's is the LOWER tile
    surface.

    The `waterfalls`/`splashes` heights frozen here are HEIGHTMAP-BASED
    (pre height-normalization) and serve as a fallback / record of which
    seams get a drop -- both write-water (course_output/water.py's
    build_stream_water_objects) and write-objects
    (rebuild_stream_drop_rows) re-fit tile levels against the real carved
    TerrainModel at write time, adding output_height_shift_m only where
    the sampler is a pre-shift model. `pearls` (bed_h) is the stable
    input to that re-fit, so run write-terrain before write-water /
    write-objects."""
    average_height = _height_sampler(heights, bounds)
    records: list[dict] = []
    for stream in streams:
        pearls, bed, rots = _pearls_bed_rots(
            stream, average_height, spacing_m, depth_m, half_width_m,
        )
        if pearls is None:
            continue
        pearl_rows = [
            [float(x), float(z), float(r), float(b)]
            for (x, z), r, b in zip(pearls, rots, bed)
        ]

        # Mean of the sin/cos of each leg's bearing -> a stable average
        # heading even across the 0/360 wrap. The +180 offset matches
        # ref/generate_streams.py's own `(180 + rot_y) % 360` for
        # options.flowOrientation -- verified in-game: PGA reads
        # flowOrientation as a "flow-from" heading (points upstream), so
        # it's the downstream bearing mirrored.
        rads = [math.radians(r) for r in rots[:-1]] or [math.radians(rots[0])]
        mean_bearing = math.degrees(math.atan2(
            sum(math.sin(a) for a in rads), sum(math.cos(a) for a in rads),
        ))
        flow_orientation = (mean_bearing + 180.0) % 360.0

        # Waterfall + splash on the seam of every consecutive water-tile
        # pair (same tiles course_output/water.py renders -- one shared
        # geometry helper) -- see stream_drop_rows. These frozen rows are
        # heightmap-based; write-objects re-derives them against the real
        # carved TerrainModel via rebuild_stream_drop_rows.
        tiles = stream_water_tiles(
            [(px, pz, pb) for (px, pz), pb in zip(pearls, bed)], float(bed[0]),
            fill_depth_m=water_fill_depth_m, base_width_m=water_base_width_m,
            widen_per_depth=water_widen_per_depth, widen_per_descent=water_widen_per_descent,
        )
        waterfalls, splashes = stream_drop_rows(tiles, waterfall_min_drop_m)

        records.append({
            "source_id": stream.source_id,
            "waterway": stream.waterway,
            "pearls": pearl_rows,
            "flow_orientation": round(flow_orientation, 3),
            "waterfalls": waterfalls,
            "splashes": splashes,
        })

    n_falls = sum(len(r["waterfalls"]) for r in records)
    printf(f"  {len(records)} stream record(s), {n_falls} waterfall(s)")
    return records


def rebuild_stream_drop_rows(
    stream_records: list[dict],
    bed_sampler: Callable[[np.ndarray], np.ndarray],
    *,
    water_fill_depth_m: float = STREAM_WATER_FILL_DEPTH_M,
    water_base_width_m: float = STREAM_WATER_BASE_WIDTH_M,
    water_widen_per_depth: float = STREAM_WATER_WIDEN_PER_DEPTH,
    water_widen_per_descent: float = STREAM_WATER_WIDEN_PER_DESCENT,
    waterfall_min_drop_m: float = WATERFALL_MIN_DROP_M,
    level_margin_m: float = STREAM_WATER_LEVEL_MARGIN_M,
) -> list[dict]:
    """Return a copy of `stream_records` with each record's
    `waterfalls`/`splashes` recomputed from its own `pearls` against
    `bed_sampler` (a TerrainModel-backed `evaluate_many`), so the drops
    land on the SAME terrain-fit tile levels write-water renders (see
    stream_water_tiles' bed_sampler). Levels come out in the sampler's
    frame -- the caller adds output_height_shift_m iff that sampler is a
    pre-shift (raw) model. Records with <2 usable pearls pass through
    untouched."""
    out: list[dict] = []
    for record in stream_records:
        pearls_bed = [
            (float(p[0]), float(p[1]), float(p[3]))
            for p in record.get("pearls", []) if len(p) >= 4
        ]
        if len(pearls_bed) < 2:
            out.append(record)
            continue
        tiles = stream_water_tiles(
            pearls_bed, pearls_bed[0][2], bed_sampler=bed_sampler,
            fill_depth_m=water_fill_depth_m, base_width_m=water_base_width_m,
            widen_per_depth=water_widen_per_depth, widen_per_descent=water_widen_per_descent,
            level_margin_m=level_margin_m,
        )
        waterfalls, splashes = stream_drop_rows(tiles, waterfall_min_drop_m)
        out.append({**record, "waterfalls": waterfalls, "splashes": splashes})
    return out


def save_stream_records(records: list[dict], path: Path) -> None:
    """Write streams.json -- a plain JSON list, same convention as
    course_output/objects.py's objects.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=STREAMS_JSON_INDENT)


def load_stream_records(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
