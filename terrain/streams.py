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
from typing import Optional

import json
import numpy as np
from shapely.geometry import LineString

from terrain.bounding_box import BoundingBox
from terrain.stamp import TOOL_FLATTEN, Stamp

# Circular "hard round" brush -- present in viz/brushes/brush_profiles.json
# (so TerrainModel can evaluate it, unlike cart_paths.py's type 15) and a
# SHAPE_CIRCLE, so scale_x must equal scale_z (see terrain/stamp.py).
STREAM_STAMP_BRUSH = 73

STREAM_DEPTH_M = 1.5            # how far below original grade the bed is cut
STREAM_HALF_WIDTH_M = 3.0      # trench stamp radius (center-to-edge); channel reads ~6 m wide
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
STREAM_WATER_FILL_DEPTH_M = 0.35      # water film depth at a tile's shallow (upstream) end
STREAM_WATER_BASE_WIDTH_M = 7.0      # cross-flow span at zero extra depth
STREAM_WATER_WIDEN_PER_DEPTH = 3.0   # + this * (water depth at the tile's deep end), meters
STREAM_WATER_WIDEN_PER_DESCENT = 0.4  # + this * (total bed descent from the stream source), meters
STREAM_WATER_OVERLAP_M = 6.0         # each tile's length = run length + this, so adjacent tiles overlap
STREAM_WATER_FLOW_SPEED = 37.0       # options.flowSpeed (real stream tile ~37; 50 is the engine max)

# A bed drop of at least this much between consecutive pearls emits a
# waterfall record at the lower pearl; consecutive falls are then thinned
# so no two sit within WATERFALL_MIN_SPACING_M of each other (a single
# cliff smeared across a few pearls by the height-sampling radius would
# otherwise stack several overlapping prefabs).
WATERFALL_MIN_DROP_M = 0.3
WATERFALL_MIN_SPACING_M = 12.0

# Centerline buffer (each side) for the stream-bank vegetation polygon
# that step_generate_streams tags with cluster fills.
BANK_VEG_WIDTH_M = 2.5

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


def _pearls_bed_rots(stream: StreamCenterline, average_height, spacing_m: float):
    """Shared front half of generate_stream_stamps / build_stream_records:
    orient downhill, resample, compute bed profile, drop dead-data
    pearls. Returns (pearls, bed, rotations) -- all three lists aligned,
    or (None, None, None) if the stream can't be used."""
    line = orient_downhill(stream.line, average_height)
    if line is None:
        return None, None, None
    raw_pearls = resample_centerline(line, spacing_m)
    if len(raw_pearls) < 2:
        return None, None, None
    raw_bed = streambed_profile(raw_pearls, average_height)
    pearls = [p for p, b in zip(raw_pearls, raw_bed) if b is not None]
    bed = [b for b in raw_bed if b is not None]
    if len(pearls) < 2:
        return None, None, None
    return pearls, bed, _pearl_rotations(pearls)


def generate_stream_stamps(
    streams: list[StreamCenterline], heights: np.ndarray, bounds: BoundingBox,
    spacing_m: float = PEARL_SPACING_M, half_width_m: float = STREAM_HALF_WIDTH_M,
    brush: int = STREAM_STAMP_BRUSH, printf=print,
) -> list[Stamp]:
    """One flatten Stamp per usable pearl, valued at that pearl's
    absolute carved bed height (see streambed_profile). scale_x ==
    scale_z (circular brush). Rotation is recorded for parity with the
    water strips but doesn't affect a circular stamp's footprint."""
    average_height = _height_sampler(heights, bounds)
    stamps: list[Stamp] = []
    skipped = 0
    for stream in streams:
        pearls, bed, rots = _pearls_bed_rots(stream, average_height, spacing_m)
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


def build_stream_records(
    streams: list[StreamCenterline], heights: np.ndarray, bounds: BoundingBox,
    spacing_m: float = PEARL_SPACING_M, waterfall_min_drop_m: float = WATERFALL_MIN_DROP_M,
    printf=print,
) -> list[dict]:
    """Frozen streams.json payload -- one record per usable stream:

        {"source_id": int | None,
         "waterway": "stream" | "ditch",
         "pearls": [[x, z, rot_y, bed_h], ...],     # course frame, downhill order
         "flow_orientation": <compass bearing deg>,
         "waterfalls": [[x, z, rot_y, bed_h], ...]}  # subset of pearls where the bed
                                                     # drops >= waterfall_min_drop_m

    bed_h everywhere is the ABSOLUTE carved bed height (pre height-
    normalization). course_output/water.py and objects.py add
    project.json's output_height_shift_m (persisted by write-terrain) to
    reach the normalized frame -- so run write-terrain before write-water
    / write-objects. Stored rather than re-derived because neither writer
    has a TerrainModel of its own and the trench bed doesn't move unless
    generate-streams re-runs (which rewrites this file)."""
    average_height = _height_sampler(heights, bounds)
    records: list[dict] = []
    for stream in streams:
        pearls, bed, rots = _pearls_bed_rots(stream, average_height, spacing_m)
        if pearls is None:
            continue
        pearl_rows = [
            [float(x), float(z), float(r), float(b)]
            for (x, z), r, b in zip(pearls, rots, bed)
        ]

        # Mean of the sin/cos of each leg's bearing -> a stable average
        # heading even across the 0/360 wrap. The +180 offset matches
        # ref/generate_streams.py's own `(180 + rot_y) % 360` for
        # options.flowOrientation -- unverified in-game here; if the
        # current visibly flows the wrong way, drop the +180.
        rads = [math.radians(r) for r in rots[:-1]] or [math.radians(rots[0])]
        mean_bearing = math.degrees(math.atan2(
            sum(math.sin(a) for a in rads), sum(math.cos(a) for a in rads),
        ))
        flow_orientation = (mean_bearing + 180.0) % 360.0

        waterfalls: list[list[float]] = []
        last_xz: Optional[tuple[float, float]] = None
        for i in range(1, len(bed)):
            if bed[i - 1] - bed[i] < waterfall_min_drop_m:
                continue
            x, z, r, _bed = pearl_rows[i]
            if last_xz is not None and math.dist(last_xz, (x, z)) < WATERFALL_MIN_SPACING_M:
                continue
            waterfalls.append([x, z, r, float(bed[i])])
            last_xz = (x, z)

        records.append({
            "source_id": stream.source_id,
            "waterway": stream.waterway,
            "pearls": pearl_rows,
            "flow_orientation": round(flow_orientation, 3),
            "waterfalls": waterfalls,
        })

    n_falls = sum(len(r["waterfalls"]) for r in records)
    printf(f"  {len(records)} stream record(s), {n_falls} waterfall(s)")
    return records


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
