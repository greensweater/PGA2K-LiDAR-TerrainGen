"""
course_output/parking.py

Auto-fill a parking lot: take an OSM aisle/service-road line (a LineString
already in the course-local [0, COURSE_SIZE_M] frame) and line it with
parked-car props from course_output/vehicle_catalog.json.

The aisle line is DIRECTIONAL. By default cars go on its LEFT side only
(PARKING_SIDES) -- the mapper draws the line the way that puts the stalls
where they want them, and reverses it to flip. "right" / "both" are
available too.

Per-aisle OCCUPANCY: the way's `ref` tag (0..1) scales how full that
aisle parks -- populated fraction = (1 - skip_prob) * ref. No tag == ref
1.0 (normal fill); ref 0.5 halves it; ref 0 empties the aisle.

Vehicle POOL: cars only (no vans -- PARKING_EXCLUDE_CLASSES). Colour is
WEIGHTED (PARKING_COLOR_WEIGHTS: black 0.40 / gray 0.15 / white 0.15,
plus PARKING_ACCENT_COUNT accent colours drawn once per course from
PARKING_ACCENT_COLORS at PARKING_ACCENT_WEIGHT each -- default totals to
1.0). A colour not in the effective map never spawns. Each car picks a
colour by weight then a prefab of that colour uniformly. The number of
DISTINCT prefabs placed course-wide is still capped at
PARKING_MAX_VARIANTS (>= 1 per active colour) so parked cars don't eat
the game's placed-object type budget. A `pga_parking` tag value further
restricts an aisle -- "yes"/"all"/empty = the default weighted mix;
otherwise a comma list of colour names and/or "car"/"van" (the weights
still apply among survivors; "van" opts vans back in for that aisle).

Same "compile once, format at write time" split as course_output/collections.py
and terrain/streams.py:

  step_generate_parking:  features.geojson pga_parking lines + heightmap
                          -> parking.json         (frozen, course-local frame)
  step_pack_objects:      parking.json .cars -> objects.json
                          (kind="collection_object", carrying pitch/roll)
  step_write_objects:     -> placedObjects2/3.json groups (v2021 path-keyed;
                          v2019 skips -- vehicle prefabs aren't in
                          asset_catalog.json yet)

A parking.json record:

    {"source_id": int | None,        # OSM way id of the aisle line
     "pool": [<asset path>, ...],    # the variant pool this aisle drew from
     "fill": float,                  # the way's `ref` tag, 0..1 (occupancy scale)
     "cars": [{"x","z","rotation_deg","pitch_deg","roll_deg","scale",
               "path"}, ...]}        # course-local metres; y is always
                                     # ground-snap ("-Infinity") at write time

Every car is placed with position.y = "-Infinity" (the game drops it onto
the terrain). pitch_deg / roll_deg bank the car to the local ground slope,
clamped to the editor's +-10 deg limit (see vehicle_catalog). They are
frozen here from the heightmap -- a parking lot is usually close to flat
and the terrain under it is already carved by the time this runs, so a
write-time re-fit (as streams do for water) isn't worth the coupling.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import LineString

from course_output.vehicle_catalog import (
    VEHICLE_VARIANTS,
    VEHICLE_PITCH_LIMIT_DEG,
    VEHICLE_ROLL_LIMIT_DEG,
)
from terrain.bounding_box import BoundingBox

_DECIMALS = 3

# Defaults (metres / probability / degrees). Overridable per project
# (project.json "parking_*" keys) and per CLI flag -- see
# PGA2k_gen.py step_generate_parking.
PARKING_SPACING_M = 2.6         # along-row centre-to-centre car spacing
PARKING_OFFSET_M = 3.0          # perpendicular distance from the aisle line to the row centre
PARKING_SIDES = "left"         # "left" | "right" | "both" -- which side of the DIRECTED line.
                              # Default "left": the aisle line is drawn directional on purpose,
                              # cars go on its left only (draw the line the other way to flip).
PARKING_ORIENTATION = "perpendicular"  # "perpendicular" (nose-in) | "parallel" (kerbside)
PARKING_SKIP_PROB = 0.12       # chance a given stall is left empty
PARKING_POS_JITTER_M = 0.12    # +- uniform position noise (both axes)
PARKING_YAW_JITTER_DEG = 2.5   # +- uniform heading noise
PARKING_SCALE = 1.0
PARKING_BANK_TO_SLOPE = True   # tilt pitch/roll to the ground normal
PARKING_SEED = 20260908

# PGA 2K budgets placed-object *types* (one per distinct prefab path), so
# drawing from all ~28 car prefabs would burn 28 type slots on parked
# cars alone. Cap the number of DISTINCT prefabs actually placed across
# the whole course to this many -- a subset is chosen once (seeded by
# PARKING_SEED) and every aisle draws from it. 0 = no cap.
PARKING_MAX_VARIANTS = 6

# Classes excluded from the default pool. Vans are big and add a
# distinct prefab per colour -- opt back in explicitly with
# pga_parking=van (or pga_parking=<colour>,van).
PARKING_EXCLUDE_CLASSES = ("van",)

# Colour mix for parked cars: {colour: relative weight}. Not required to
# sum to 1 -- weights are normalised. A colour NOT in the effective map
# never spawns. Base map (realistic "car park" palette):
PARKING_COLOR_WEIGHTS = {"black": 0.40, "gray": 0.15, "white": 0.15}
# ...plus PARKING_ACCENT_COUNT accent colours drawn once per course
# (seeded by PARKING_SEED) from this list, each added at
# PARKING_ACCENT_WEIGHT. 0.40 + 0.15 + 0.15 + 3*0.10 = 1.0.
PARKING_ACCENT_COLORS = ("red", "green", "blue", "yellow")
PARKING_ACCENT_COUNT = 3
PARKING_ACCENT_WEIGHT = 0.10

# Sign of the slope->tilt mapping. Best-effort without an in-game check;
# flip a value to -1 if cars lean the wrong way on a slope.
PARKING_PITCH_SIGN = -1.0      # rotation.x: ground rising ahead -> nose up (negative x)
PARKING_ROLL_SIGN = 1.0       # rotation.z

# How far (m) to step when estimating the ground gradient by finite
# difference.
_SLOPE_PROBE_M = 2.0


@dataclass
class ParkingAisle:
    """One aisle to line with cars. `line` is a course-local LineString;
    `pool` is the already-resolved list of vehicle asset paths this aisle
    may draw from (empty = whole catalogue); `source_id` is the OSM way
    id for back-reference. `fill` (0..1, from the way's `ref` tag) scales
    how full THIS aisle parks -- populated fraction = (1 - skip_prob) *
    fill, so fill=1 (or no tag) is the normal fill, fill=0.5 halves it."""
    line: LineString
    source_id: Optional[int] = None
    pool: list[str] = field(default_factory=list)
    fill: float = 1.0


def _round(v: float) -> float:
    return round(float(v), _DECIMALS)


def _bearing_deg(dx: float, dz: float) -> float:
    """Compass bearing of a direction vector -- 0 = +Z, 90 = +X, clockwise.
    Same convention as terrain/streams.py / course_output/collections.py."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def default_pool() -> list[str]:
    """The catalogue minus PARKING_EXCLUDE_CLASSES (vans). Not used by
    the weighted selection (see build_parking_records) -- kept as a
    convenience for callers that just want "every car"."""
    return [v.path for v in VEHICLE_VARIANTS if v.vehicle_class not in PARKING_EXCLUDE_CLASSES]


def resolve_pool(pool_filter: Optional[str]) -> list[str]:
    """Turn a `pga_parking` tag value into an allow-list of vehicle asset
    paths that RESTRICTS the aisle (the colour weights still apply among
    whatever survives -- see build_parking_records).
    "", "yes", "all", None -> [] (no restriction; the default weighted
    colour mix applies). Otherwise a comma list of colour names and/or
    classes ("car"/"van"), e.g. "black,white,red".

    Union, not intersection: "black,white,red" keeps a variant matching
    ANY token (by colour or class). Vans are dropped unless "van" is an
    explicit token (PARKING_EXCLUDE_CLASSES) -- so "black" gives black
    cars, not the black van too."""
    if pool_filter is None:
        return []
    tokens = {t.strip().lower() for t in str(pool_filter).split(",") if t.strip()}
    if not tokens or tokens <= {"yes", "all", "true"}:
        return []
    excluded = {c for c in PARKING_EXCLUDE_CLASSES if c not in tokens}
    return [
        v.path for v in VEHICLE_VARIANTS
        if (v.color in tokens or v.vehicle_class in tokens)
        and v.vehicle_class not in excluded
    ]


def parse_color_weights(text: "str | dict | None") -> "dict[str, float] | None":
    """Parse a "colour=weight,colour=weight" string (or pass a dict
    through) into {colour: float}. None / "" / unparseable -> None (the
    caller falls back to PARKING_COLOR_WEIGHTS). Weights need not sum to
    1; percentages ("black=40") are fine -- they're normalised at use."""
    if text is None or isinstance(text, dict):
        return text or None
    out: dict[str, float] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, _, v = part.partition("=")
        try:
            out[k.strip().lower()] = float(v)
        except ValueError:
            continue
    return out or None


class _HeightField:
    """Bilinear sampler + finite-difference gradient over a heightmap grid.
    NaN cells are treated as 'no data' -- a sample touching one returns
    None (caller leaves the car flat there)."""

    def __init__(self, heights: np.ndarray, bounds: BoundingBox):
        self._h = np.asarray(heights, dtype=float)
        self._n_rows, self._n_cols = self._h.shape
        self._min_x, self._min_z = bounds.min_x, bounds.min_z
        self._cell_x = (bounds.max_x - bounds.min_x) / max(1, self._n_cols)
        self._cell_z = (bounds.max_z - bounds.min_z) / max(1, self._n_rows)

    def height(self, x: float, z: float) -> Optional[float]:
        fx = (x - self._min_x) / self._cell_x - 0.5
        fz = (z - self._min_z) / self._cell_z - 0.5
        x0, z0 = int(math.floor(fx)), int(math.floor(fz))
        if not (0 <= x0 < self._n_cols - 1 and 0 <= z0 < self._n_rows - 1):
            # Clamp to the nearest valid cell rather than fail at the edge.
            xi = min(max(x0, 0), self._n_cols - 1)
            zi = min(max(z0, 0), self._n_rows - 1)
            v = self._h[zi, xi]
            return None if not math.isfinite(v) else float(v)
        tx, tz = fx - x0, fz - z0
        h00, h10 = self._h[z0, x0], self._h[z0, x0 + 1]
        h01, h11 = self._h[z0 + 1, x0], self._h[z0 + 1, x0 + 1]
        if not all(math.isfinite(v) for v in (h00, h10, h01, h11)):
            return None
        return float(
            h00 * (1 - tx) * (1 - tz) + h10 * tx * (1 - tz)
            + h01 * (1 - tx) * tz + h11 * tx * tz
        )

    def gradient(self, x: float, z: float, probe_m: float = _SLOPE_PROBE_M) -> Optional[tuple[float, float]]:
        """(dh/dx, dh/dz) at (x, z), central difference over +-probe_m.
        None if any of the four probes has no data."""
        hxp, hxm = self.height(x + probe_m, z), self.height(x - probe_m, z)
        hzp, hzm = self.height(x, z + probe_m), self.height(x, z - probe_m)
        if None in (hxp, hxm, hzp, hzm):
            return None
        return (hxp - hxm) / (2 * probe_m), (hzp - hzm) / (2 * probe_m)


def _tilt_for(
    field: _HeightField, x: float, z: float, heading_deg: float,
    pitch_limit: float, roll_limit: float,
) -> tuple[float, float]:
    """(pitch_deg, roll_deg) banking a car at (x, z) facing `heading_deg`
    to the local ground slope, clamped to the editor limits. (0, 0) where
    the heightmap has no data."""
    grad = field.gradient(x, z)
    if grad is None:
        return 0.0, 0.0
    gx, gz = grad
    h = math.radians(heading_deg)
    fwd = (math.sin(h), math.cos(h))            # car forward, (x, z)
    right = (math.cos(h), -math.sin(h))         # car right
    slope_fwd = gx * fwd[0] + gz * fwd[1]        # rise per metre forward
    slope_right = gx * right[0] + gz * right[1]  # rise per metre to the right
    pitch = PARKING_PITCH_SIGN * math.degrees(math.atan(slope_fwd))
    roll = PARKING_ROLL_SIGN * math.degrees(math.atan(slope_right))
    return (
        max(-pitch_limit, min(pitch_limit, pitch)),
        max(-roll_limit, min(roll_limit, roll)),
    )


def _row_points(line: LineString, spacing_m: float) -> list[tuple[float, float, float]]:
    """Evenly-spaced (x, z, forward_bearing_deg) along `line`."""
    length = line.length
    if length <= 1e-6:
        return []
    n = max(2, int(length / spacing_m) + 1)
    ds = np.linspace(0.0, length, n)
    pts = [line.interpolate(float(d)) for d in ds]
    out: list[tuple[float, float, float]] = []
    for i, p in enumerate(pts):
        nxt = pts[min(i + 1, len(pts) - 1)]
        prv = pts[max(i - 1, 0)]
        dx, dz = nxt.x - prv.x, nxt.y - prv.y
        bearing = _bearing_deg(dx, dz) if math.hypot(dx, dz) > 1e-9 else 0.0
        out.append((p.x, p.y, bearing))
    return out


def _resolve_selection(
    seed: int, max_variants: int,
    color_weights: dict[str, float], accent_colors, accent_count: int, accent_weight: float,
    tag_paths: "set[str] | None",
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Course-wide vehicle selection: pick the accent colours (seeded),
    build {colour: [prefab paths]} (vans excluded, colours restricted to
    `tag_paths` when the aisles carry an explicit pga_parking filter),
    then trim the total distinct-prefab count to `max_variants` keeping
    >= 1 prefab per colour (or, when max_variants < #colours, keep only
    the highest-weight colours). Returns (by_colour, weights)."""
    pool_rng = random.Random(f"{seed}:pool")
    weights = {c: float(w) for c, w in color_weights.items() if w > 0}
    for c in pool_rng.sample(list(accent_colors), min(accent_count, len(accent_colors))):
        weights.setdefault(c, accent_weight)

    by_color: dict[str, list[str]] = {}
    for v in VEHICLE_VARIANTS:
        if v.vehicle_class in PARKING_EXCLUDE_CLASSES:
            continue
        if tag_paths is not None and v.path not in tag_paths:
            continue
        if v.color in weights:
            by_color.setdefault(v.color, []).append(v.path)
    # A tag colour with no weight entry: give it the accent weight.
    if tag_paths is not None:
        for v in VEHICLE_VARIANTS:
            if v.path in tag_paths and v.vehicle_class not in PARKING_EXCLUDE_CLASSES \
                    and v.color not in by_color:
                by_color.setdefault(v.color, []).append(v.path)
                weights.setdefault(v.color, accent_weight)

    by_color = {c: sorted(ps) for c, ps in by_color.items() if ps}
    weights = {c: weights[c] for c in by_color}
    if not by_color:
        return {}, {}

    if 0 < max_variants < len(by_color):
        keep = sorted(by_color, key=lambda c: weights[c], reverse=True)[:max_variants]
        by_color = {c: by_color[c] for c in keep}
        weights = {c: weights[c] for c in keep}

    total = sum(len(ps) for ps in by_color.values())
    if 0 < max_variants < total:
        capped = {c: [pool_rng.choice(ps)] for c, ps in by_color.items()}
        extras = [(c, p) for c, ps in by_color.items() for p in ps if p not in capped[c]]
        pool_rng.shuffle(extras)
        for c, p in extras[: max_variants - len(capped)]:
            capped[c].append(p)
        by_color = {c: sorted(ps) for c, ps in capped.items()}
    return by_color, weights


def build_parking_records(
    aisles: list[ParkingAisle], heights: np.ndarray, bounds: BoundingBox, *,
    spacing_m: float = PARKING_SPACING_M,
    offset_m: float = PARKING_OFFSET_M,
    sides: str = PARKING_SIDES,
    orientation: str = PARKING_ORIENTATION,
    skip_prob: float = PARKING_SKIP_PROB,
    pos_jitter_m: float = PARKING_POS_JITTER_M,
    yaw_jitter_deg: float = PARKING_YAW_JITTER_DEG,
    scale: float = PARKING_SCALE,
    bank_to_slope: bool = PARKING_BANK_TO_SLOPE,
    seed: int = PARKING_SEED,
    max_variants: int = PARKING_MAX_VARIANTS,
    color_weights: "dict[str, float] | None" = None,
    accent_count: int = PARKING_ACCENT_COUNT,
    accent_weight: float = PARKING_ACCENT_WEIGHT,
    pitch_limit: float = VEHICLE_PITCH_LIMIT_DEG,
    roll_limit: float = VEHICLE_ROLL_LIMIT_DEG,
    printf=print,
) -> list[dict]:
    """Frozen parking.json payload -- one record per aisle. RNG (colour,
    prefab, skipped stalls, jitter) is seeded per aisle from `seed` +
    source_id so a re-run reproduces the same lot and adding one aisle
    doesn't reshuffle the others.

    COLOUR MIX: `color_weights` (default PARKING_COLOR_WEIGHTS -- black
    0.40 / gray 0.15 / white 0.15) plus `accent_count` accent colours
    drawn once per course from PARKING_ACCENT_COLORS at `accent_weight`
    each. A colour not in the effective map never spawns. Each car picks
    a colour by weight, then a prefab of that colour uniformly.

    `max_variants` (> 0) caps the DISTINCT prefab count across the WHOLE
    course, keeping >= 1 prefab per active colour -- so parked cars never
    spend more than `max_variants` placed-object type slots.

    A `pga_parking` tag on an aisle further restricts which paths (hence
    colours) that aisle may use; the weights still apply among survivors."""
    field = _HeightField(heights, bounds) if (heights is not None and bank_to_slope) else None
    side_signs = {"both": (-1.0, 1.0), "left": (-1.0,), "right": (1.0,)}.get(sides, (-1.0, 1.0))

    weight_map = dict(color_weights) if color_weights else dict(PARKING_COLOR_WEIGHTS)
    any_untagged = any(not a.pool for a in aisles)
    tag_union = {p for a in aisles for p in a.pool} or None
    by_color, weights = _resolve_selection(
        seed, max_variants, weight_map, PARKING_ACCENT_COLORS, accent_count, accent_weight,
        None if any_untagged else tag_union,
    )
    if not by_color:
        printf("  NOTE: no vehicles match the colour weights / pga_parking filters -- nothing placed")
        return [{"source_id": a.source_id, "pool": [], "fill": 1.0, "cars": []} for a in aisles]
    n_prefabs = sum(len(ps) for ps in by_color.values())
    printf(f"  vehicle pool: {n_prefabs} distinct prefab(s) across {len(by_color)} colour(s) "
           f"{{{', '.join(f'{c} {weights[c]:.2f}' for c in by_color)}}} (cap {max_variants or 'off'})")

    def _pick_path(rng: random.Random, allow: dict[str, list[str]]) -> str:
        # Straight weighted pick -- no adjacent-dedup: a parked lot with
        # two of the same car next to each other is fine, and deduping
        # here would visibly pull the dominant colour below its weight
        # (black is one prefab at 0.40).
        cols = list(allow)
        wts = [weights.get(c, accent_weight) for c in cols]
        c = rng.choices(cols, weights=wts, k=1)[0]
        return rng.choice(allow[c])

    records: list[dict] = []
    total_cars = 0
    for aisle in aisles:
        line = aisle.line
        if line.geom_type != "LineString" or line.length <= 1e-6:
            continue
        if aisle.pool:
            allow = {c: [p for p in ps if p in set(aisle.pool)] for c, ps in by_color.items()}
            allow = {c: ps for c, ps in allow.items() if ps} or by_color
        else:
            allow = by_color
        aisle_paths = sorted({p for ps in allow.values() for p in ps})
        rng = random.Random(f"{seed}:{aisle.source_id if aisle.source_id is not None else 0}")

        # Per-aisle occupancy: populated fraction = (1 - skip_prob) *
        # fill (the way's `ref` tag, default 1.0), so the effective skip
        # is 1 - that.
        fill = max(0.0, min(1.0, float(aisle.fill)))
        eff_skip = 1.0 - (1.0 - skip_prob) * fill

        cars: list[dict] = []
        for (px, pz, bearing) in _row_points(line, spacing_m):
            hb = math.radians(bearing)
            right_vec = (math.cos(hb), -math.sin(hb))
            for sign in side_signs:
                if rng.random() < eff_skip:
                    continue
                ox = right_vec[0] * sign * offset_m + rng.uniform(-pos_jitter_m, pos_jitter_m)
                oz = right_vec[1] * sign * offset_m + rng.uniform(-pos_jitter_m, pos_jitter_m)
                cx, cz = px + ox, pz + oz
                if orientation == "parallel":
                    heading = (bearing + (0.0 if sign > 0 else 180.0)) % 360.0
                else:  # perpendicular / nose-in: face away from the aisle
                    heading = _bearing_deg(right_vec[0] * sign, right_vec[1] * sign)
                heading = (heading + rng.uniform(-yaw_jitter_deg, yaw_jitter_deg)) % 360.0
                path = _pick_path(rng, allow)
                pitch = roll = 0.0
                if field is not None:
                    pitch, roll = _tilt_for(field, cx, cz, heading, pitch_limit, roll_limit)
                cars.append({
                    "x": _round(cx), "z": _round(cz),
                    "rotation_deg": _round(heading),
                    "pitch_deg": _round(pitch), "roll_deg": _round(roll),
                    "scale": _round(scale), "path": path,
                })
        records.append({
            "source_id": aisle.source_id,
            "pool": aisle_paths,
            "fill": round(fill, 3),
            "cars": cars,
        })
        total_cars += len(cars)
        fill_note = f", ref/fill {fill:.2f}" if fill < 0.999 else ""
        printf(f"  aisle osm_id={aisle.source_id}: {len(cars)} car(s) "
               f"along {line.length:.0f} m ({len(aisle_paths)} variant pool{fill_note})")

    printf(f"  {len(records)} parking aisle(s), {total_cars} car(s) total")
    return records


def iter_parking_cars(records: list[dict]):
    """Every car placement across every parking record, shaped as a
    collections.py-style object dict (x, z, rotation_deg, scale, path)
    plus pitch/roll and its record's source_id -- the flat list
    step_pack_objects folds into objects.json as kind="collection_object"."""
    for record in records:
        for car in record.get("cars", []):
            yield {
                "x": car["x"], "z": car["z"],
                "rotation_deg": car.get("rotation_deg", 0.0),
                "pitch": car.get("pitch_deg", 0.0),
                "roll": car.get("roll_deg", 0.0),
                "scale": car.get("scale", 1.0),
                "category": None, "type": None, "theme": False,
                "path": car["path"],
                "source_id": record.get("source_id"),
            }


def save_parking_records(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def load_parking_records(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
