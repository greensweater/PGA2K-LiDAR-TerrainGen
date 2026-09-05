"""
course_output/object_clusters.py

Fills user-selected spline areas (Feature geometry, features.geojson)
with placedObjects2 "clusters" -- density-fill scatter stamps, not
individually placed instances (see objects.py's module docstring) --
using real per-asset density data from asset_catalog.py.

PACKING: a graduated multi-tier scheme per asset (see _pack_circles),
same overall "coarse first, later passes mop up what's left" idea
terrain/contour_layers.py's tiered band-fill already uses for terrain
stamps, adapted from a heightmap raster to a vector shapely polygon:

  Tier 1: circles at the category's own cluster_radius (the largest a
    stamp is ever allowed to be) -- dart-thrown (random candidate,
    reject on overlap) same rejection-sampling idea as
    terrain/adaptive_refine.py's scatter_stamps ("the poisson fill
    elsewhere in here"), reimplemented directly against a shapely
    polygon rather than called as-is, since that function is tightly
    coupled to heightmap rasters/Stamp objects that don't apply here.
  Tier 2, 3, ...: each tier's radius is the previous tier's radius
    times TIER_STEP_RATIO, dart-thrown the same way, but only into
    whatever the previous tiers left uncovered (see _pack_circles for
    why this is dart-thrown too, not a deterministic grid, now that
    CONTAINMENT below can leave that remainder an irregular sliver).
    Tiers stop once the radius drops below MIN_TIER_RADIUS or
    MAX_TIERS is reached. A narrow region (e.g. a border ring much
    thinner than 2x cluster_radius) simply skips however many of the
    largest tiers don't fit and starts placing at whichever tier
    first does -- rather than the old 2-pass scheme's cliff, where a
    region too narrow for cluster_radius fell straight to one fixed
    half-radius tier with nothing in between.

CONTAINMENT: every stamp's WHOLE circle must land inside the spline --
never just its center, and never allowed to spill past the spline's
own exterior boundary, in any tier. Enforced by testing candidate
centers against geometry eroded by that stamp's own radius
(geometry.buffer(-radius)) -- the standard "does a disc of this radius
centered here fit entirely inside geometry" test -- rather than the
raw geometry. An earlier version tested plain center-inside-geometry
(tier 1) or even geometry DILATED by radius (later tiers' coverage
mop-up, deliberately allowing edge overspill for fuller coverage) --
both replaced after real fills showed stamps overlapping well past
spline edges, which looked wrong regardless of any coverage benefit. The
trade-off this accepts: a strip within one stamp's radius of the
boundary can never be covered at all, since no valid circle of that
size can be centered close enough to reach it without crossing the
edge -- each successive, smaller tier covers some of what the larger
ones couldn't reach this way, down to MIN_TIER_RADIUS, but perfect
edge-to-edge coverage is not the goal here.

A cluster stamp's radius is always one of the tier values _tier_radii
produces -- never independently tunable per stamp beyond that, since
cluster_radius is an engine property of the category, not a
per-placement choice. `ratio`
(the GUI's "Raster ratio" knob) controls how much overlap sibling
circles tolerate between each other (never the boundary, which is
never crossed regardless of ratio): minimum required center-to-center
separation is `(r1 + r2) * ratio` -- 1.0 means circles may only just
touch, <1 lets them overlap more (denser fill), >1 spaces them out
more. count per stamp is asset_catalog.cluster_count(that stamp's own
radius, asset.spacing) -- computed per-tier, since a half-radius stamp
covers a quarter the area and should get proportionally fewer instances
at the same measured density.

WALK: border rings (see build_border_ring_geometry) get a different
packer entirely, _pack_ring_walk, not the tiered dart-throw above.
A border ring is a buffered LINE, not a blob -- its width is usually
much narrower than a nature category's cluster_radius (a ring stroked
10m wide against a tree category whose cluster_radius is 45m), so
dart-throw's tier ladder (45, 22.5, 11.25, 5.625, ...) mostly finds
nowhere to fit: every tier above roughly half the ring's own width is
rejected everywhere except the odd self-overlap bulge at a concave
corner of the source boundary (see util/viz/border_ring_viz.py), which
is exactly the "sparse at large radii, tiny stamps clumped at
corners" result this replaces. Since a ring DOES have a natural path
through it -- the original centerline it was stroked from -- walking
that path directly is both simpler and better suited.

_pack_ring_walk (via _pack_line_squeeze) is a two-phase packer along
that centerline, not a single fixed-step march:

  Phase 1, SEED: sample the ring's locally-achievable radius (distance
    to ring_geometry's own boundary, capped at the category's
    cluster_radius -- see _max_local_radius) at a fine arc-length step
    along the whole centerline, then take the local maxima of that
    width profile as candidate seed circles (_select_ring_seeds),
    accepted largest-first with the usual (r1+r2)*ratio separation
    rejecting anything too close to an already-accepted seed. This is
    the "lovely large fill" -- one circle per real local width peak,
    nothing smaller squeezed in between yet.
  Phase 2, SQUEEZE: for every ADJACENT PAIR of accepted seeds, solve
    directly (no search) for the "secondary" circle on each of the
    gap's two lateral sides -- one leaning toward each of the ring's two
    edges -- via _secondary_circle: the classic closed-form construction
    for a circle tangent to two given circles and a line (the line-as-
    zero-curvature-circle case of the Apollonius-problem/Descartes-
    circle-theorem family, generalized to this module's usual
    ratio-scaled "tangent-or-looser" separation instead of exact
    tangency -- see _solve_tangent_circle for the algebra), where the
    "line" is a local linearization of ring_geometry's own REAL boundary
    near the gap, refined by re-anchoring the linearization at the
    solved point and re-solving a few times (real boundary curvature
    converges in 1-2 rounds) rather than assumed flat outright. Exactly
    ONE circle per side per pair -- placed if it clears MIN_TIER_RADIUS,
    dropped otherwise (same CONTAINMENT trade-off the tiered dart-throw
    above already accepts for a side too tight for even one such
    circle) -- no recursion into further leftover sub-gaps beyond that,
    a deliberate simplification (an earlier version did recurse; traded
    away for a flat pass once the closed-form replaced the search it
    used to recurse around).

    Two EARLIER versions of this same phase are worth knowing existed,
    since both were replaced for concrete, confirmed reasons rather than
    style preference: a purely centerline-confined search (undersized
    every squeeze circle, since a centerline point is by construction
    roughly equidistant from BOTH ring edges and so can never lean
    toward whichever edge has the real room); and a 2D grid search (not
    confined to the centerline, but a brute-force shrinking-box search,
    ~300-600 shapely evaluations per gap-side) that fixed the
    undersizing but was slow and only heuristically accurate. The
    closed-form construction here replaces the grid search outright --
    same or better accuracy (confirmed: the grid search's coarse
    quantization slightly UNDERSHOT the true optimum on a curved test
    case, since the closed form converged to a larger, still fully
    valid, radius there) at a fraction of the cost (~1 quadratic solve
    plus a few cheap re-anchoring iterations, not hundreds of point
    evaluations).

    The two OPEN ENDS of a non-closed line (before the first seed, after
    the last) have only one neighbor, not a pair, so _secondary_circle's
    two-circles-and-a-line construction doesn't apply there -- they keep
    the simpler single-neighbor _squeeze_centerline_seed floor instead
    (one on-centerline circle, no two-sided secondaries), a deliberate
    scope limit matching what actually needed fixing here.

Requires the centerline itself, which build_border_ring_geometry's
OUTPUT (the buffered ring polygon) does not retain -- see
SYNTHETIC_BORDER_CENTERLINE_KIND for how the GUI carries it through as
a second, linked synthetic Feature so it survives the
shift_features/_crop_features_to_course trip from features.geojson to
write-objects time.

DENSITY: a fill spec's `density` field (the GUI's "Fill density (%)"
knob, default DEFAULT_FILL_DENSITY = 100) scales the instance COUNT
placed inside each stamp circle -- asset_catalog.cluster_count(radius,
spacing) times density/100, floored at 1 -- without touching the
circles themselves (their radius/position/count of STAMPS is still
whatever the packer above produced). Deliberately a separate knob from
`ratio`: ratio changes how many/how large the circles are (packing
geometry), density changes how many instances render inside a given
circle (measured-density scale), independent of each other.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from course_output.asset_catalog import ASSET_CATEGORIES, ASSET_ENTRIES, AssetCategory, AssetEntry, cluster_count
from course_output.userLayers import GRID_ORIGIN_OFFSET
from ingest.osm import Feature

# Feature.tags key the GUI's "Fill with Clusters" action sets on a
# selected spline (see PGA2k_gen_gui.py) -- pga_-prefixed, same custom-
# tag convention as objects.py's TREE_TYPE_TAG. Value is a LIST of
# {"category": int, "type": int, "ratio": float} dicts, not a single
# scalar tag, specifically so multiple assets (e.g. grass understory +
# scattered rocks) can be layered onto the same spline -- each Fill
# action appends to it rather than overwriting it. Feature.tags is a
# plain dict JSON-dumped as-is by ingest.osm.save_features, so a list
# value round-trips fine (not restricted to OSM-style string tags).
PGA_CLUSTER_FILLS_TAG = "pga_cluster_fills"

# Feature.kind values for the two synthetic, GUI-generated Features this
# module's border/masked-manual fill modes create (see PGA2k_gen_gui.py's
# _open_cluster_fill_dialog) -- neither exists anywhere in ingest/osm.py's
# classify_way, so both are guaranteed never to be picked up by any
# f.kind == "..." check elsewhere in the pipeline (splines/holes/water/mask
# building all filter by a fixed allow-list of real OSM-derived kinds).
SYNTHETIC_BORDER_KIND = "pga_cluster_border"
SYNTHETIC_MASKED_KIND = "pga_cluster_masked"

# Feature.kind for the centerline companion Feature a border fill creates
# alongside its SYNTHETIC_BORDER_KIND ring Feature (see the GUI's
# _open_cluster_fill_dialog and this module's WALK docstring section) --
# carries the ORIGINAL mask boundary line (pre-buffer) as its own
# .geometry, purely so it rides the normal shift_features/
# _crop_features_to_course pipeline every other Feature's geometry
# already goes through, rather than needing a second, shift-aware code
# path of its own. Never carries PGA_CLUSTER_FILLS_TAG itself, so it's
# invisible to every "tagged = [f for f in features if f.tags.get(...)]"
# filter already in place (e.g. PGA2k_gen.py's step_pack_objects) --
# only pack_cluster_records (which sees the FULL features list, not a
# pre-filtered one) knows to look it up, via the ring Feature's own
# PGA_CLUSTER_CENTERLINE_REF_TAG. Also never in _AREA_GEOM_TYPES (it's a
# Line, not a Polygon), so the ordinary "no area to fill" skip in both
# fill_feature_with_clusters and pack_cluster_records already excludes
# it from ever being treated as a fillable spline in its own right.
SYNTHETIC_BORDER_CENTERLINE_KIND = "pga_cluster_border_centerline"

# Tag key on a SYNTHETIC_BORDER_KIND ring Feature: the osm_id (int) of
# its linked SYNTHETIC_BORDER_CENTERLINE_KIND companion Feature (see
# above). Absent (or unresolvable, e.g. a features.geojson saved before
# this existed) just means _clusters_for_spec falls back to the ordinary
# dart-throw tiered packing for that spec, same as any other polygon.
PGA_CLUSTER_CENTERLINE_REF_TAG = "pga_cluster_centerline_ref"

# Fill spec dicts' "source" field -- distinguishes a fill tagged directly
# onto a real spline's own geometry (manual, the original/default behavior)
# from one tagged onto a synthetic ring/clipped Feature (border). Purely a
# display/bookkeeping field: packing code (below) never reads it, so a spec
# predating this field's existence is fine to treat as "manual" wherever it
# does get read (see PGA2k_gen_gui.py's _build_cluster_fill_rows).
CLUSTER_FILL_SOURCE_MANUAL = "manual"
CLUSTER_FILL_SOURCE_BORDER = "border"
CLUSTER_FILL_SOURCE_STREAM = "stream"  # tagged onto a synthetic stream-bank Feature (see step_generate_streams)


def next_synthetic_osm_id(features) -> int:
    """First unused negative int for a synthetic (non-OSM) Feature --
    real OSM way ids are always non-negative, so a negative id can never
    collide with one. Shared by PGA2k_gen_gui.py's cluster-fill dialog
    and PGA2k_gen.py's step_generate_streams; osm_id on a synthetic
    Feature is only ever used for selection/display and for linking a
    border ring to its centerline companion, never for real OSM
    cross-reference."""
    existing = {f.osm_id for f in features if f.osm_id is not None}
    candidate = -1
    while candidate in existing:
        candidate -= 1
    return candidate

DEFAULT_RASTER_RATIO = 1.0
DEFAULT_FILL_DENSITY = 100.0  # percent; scales cluster_count(...) -- see DENSITY in the module docstring

# Fill spec dicts' "mode" field -- "stamps" (default, back-compat with
# every spec predating this field) packs circle-scatter stamps into
# Value.clusters (see pack_cluster_records); "spline" instead emits an
# engine-auto-scattered object-spline fill region into Value.splines,
# v2021+ only (see pack_spline_records). One Feature can carry both --
# each fill spec picks its own mode independently.
CLUSTER_FILL_MODE_STAMPS = "stamps"
CLUSTER_FILL_MODE_SPLINE = "spline"

# The engine appears to cap a single object-spline's bounding box --
# see ref/generate_rough_border_v2.py's MAX_SPLINE_BOUNDS (100m there).
# An oversized spline-mode fill polygon is chopped into pieces no
# larger than this on either side first (see subdivide_polygon).
MAX_SPLINE_FILL_PIECE_SIZE_M = 50.0
TIER_STEP_RATIO = 0.5  # each tier's radius = previous tier's radius * this
MIN_TIER_RADIUS = 1.0  # meters; smallest radius a tier (or a ring-walk circle) is ever allowed to target
MAX_TIERS = 6  # hard cap on tier count, safety net -- not expected to bind for any current catalog cluster_radius
MAX_CONSECUTIVE_FAILURES = 30  # a dart-throw pass gives up once this many candidates in a row are rejected
MAX_PACK_ATTEMPTS = 500  # hard cap on candidate draws per dart-throw pass, safety net for pathological geometry
_MIN_USEFUL_RADIUS = 0.1  # below this a pass is pointless (every real category's radius is well above it)
_AREA_GEOM_TYPES = ("Polygon", "MultiPolygon")

# _pack_ring_walk (see WALK in the module docstring)
_SEED_SNAP_ITERS = 20  # binary-search iterations tightening a seed's spacing from its left neighbor -- plenty given the search span is at most one scan_step wide
_SECONDARY_SOLVE_MAX_ITERS = 4  # re-anchor/re-solve rounds for _secondary_circle's tangent-circle construction -- converges in 1-2 for real boundary curvature, this is headroom not a expected ceiling
_SECONDARY_SOLVE_TOLERANCE = 0.01  # meters; re-anchoring stops early once consecutive solves land within this of each other
_SECONDARY_PROBE_FACTOR = 0.5  # initial anchor push distance = (raw, uncapped boundary distance at the pair midpoint) * this -- enough to break the tie between the ring's two edges without overshooting past the correct nearby one (see _secondary_circle; a version of this sized off max_radius instead could overshoot badly whenever border_width was large relative to max_radius)

# Matches userLayers.py/water.py's own _DECIMALS convention -- every
# value written into placedObjects2.json is rounded to millimeter
# precision, plenty for this project's purposes.
_DECIMALS = 3


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)

_ENTRIES_BY_KEY = {(e.category, e.type): e for e in ASSET_ENTRIES}


def _resolve_spec(spec: dict) -> Optional[tuple[AssetCategory, AssetEntry]]:
    """(category, entry) for a {"category","type",...} fill spec, or
    None if it doesn't resolve to a real clusterable catalog entry --
    e.g. a stale tag left over after asset_catalog.json changed."""
    category = ASSET_CATEGORIES.get(spec.get("category"))
    entry = _ENTRIES_BY_KEY.get((spec.get("category"), spec.get("type")))
    if category is None or category.cluster_radius is None or entry is None or not entry.spacing:
        return None
    return category, entry


def _dart_throw_pass(
    geometry, radius: float, already_placed: list[tuple[float, float, float]], ratio: float, rng: random.Random,
) -> list[tuple[float, float, float]]:
    """
    One organic, non-overlapping-by-default packing pass at a fixed
    `radius`: repeatedly draws a random candidate center inside the
    bounding box of geometry ERODED by `radius` (see module docstring's
    CONTAINMENT), accepts it if the candidate's WHOLE circle therefore
    fits inside `geometry` and it's at least
    `(radius + other_radius) * ratio` away from every circle in
    `already_placed` (this pass's own accepted circles included, as
    they accumulate). Returns newly accepted (x, z, radius) triples;
    does not mutate `already_placed`. [] immediately if nothing this
    large fits inside geometry at all (the erosion comes back empty).

    Stops once MAX_CONSECUTIVE_FAILURES candidates in a row are
    rejected (this radius no longer fits anywhere) or
    MAX_PACK_ATTEMPTS total draws are used up, whichever comes first --
    same failure-count termination idea as
    terrain/adaptive_refine.py's scatter_stamps.
    """
    eroded = geometry.buffer(-radius)
    if eroded.is_empty:
        return []

    min_x, min_z, max_x, max_z = eroded.bounds
    all_circles = list(already_placed)
    accepted: list[tuple[float, float, float]] = []
    consecutive_failures = 0
    attempts = 0
    while consecutive_failures < MAX_CONSECUTIVE_FAILURES and attempts < MAX_PACK_ATTEMPTS:
        attempts += 1
        x = rng.uniform(min_x, max_x)
        z = rng.uniform(min_z, max_z)
        if not eroded.contains(Point(x, z)):
            consecutive_failures += 1
            continue
        if any(math.hypot(x - ox, z - oz) < (radius + orad) * ratio for ox, oz, orad in all_circles):
            consecutive_failures += 1
            continue
        triple = (x, z, radius)
        all_circles.append(triple)
        accepted.append(triple)
        consecutive_failures = 0
    return accepted


def _subtract_circles(geometry, circles: list[tuple[float, float, float]]):
    """geometry with every (x,z,radius) circle's disc cut out -- the
    "still uncovered" remainder the next pass targets. `geometry`
    itself if `circles` is empty (nothing to subtract)."""
    if not circles:
        return geometry
    covered = unary_union([Point(x, z).buffer(r) for x, z, r in circles])
    return geometry.difference(covered)


def _tier_radii(max_radius: float, min_radius: float, step_ratio: float, max_tiers: int) -> list[float]:
    """
    Geometric ladder of tier radii from max_radius down to (but not
    below) min_radius: [max_radius, max_radius*step_ratio,
    max_radius*step_ratio**2, ...], capped at max_tiers entries.

    Deliberately simpler than terrain/contour_layers.py's own
    _tier_radii (which blends in a min_step_m floor to keep steps
    meaningful over a wide, user-tunable slider range): here
    max_radius is always one of a handful of fixed engine constants
    from asset_catalog.json's cluster_radius, so a plain multiplicative
    ladder is sufficient, and pulling in that terrain-package-private
    helper (plus its numpy return type) would add cross-package
    coupling this module doesn't otherwise have for no real benefit.
    """
    radii = [max_radius]
    while len(radii) < max_tiers:
        next_radius = radii[-1] * step_ratio
        if next_radius < min_radius:
            break
        radii.append(next_radius)
    return radii


def _pack_circles(geometry, max_radius: float, ratio: float, rng: random.Random) -> list[tuple[float, float, float]]:
    """
    Graduated multi-tier circle packing over `geometry`'s area -- see
    module docstring. Every tier is dart-thrown (not a deterministic
    grid): once containment requires a stamp's WHOLE circle to fit
    inside geometry (see CONTAINMENT), the valid region left for a
    later tier -- geometry eroded by that tier's radius, after earlier
    tiers' circles are subtracted out -- can be an irregular, thin
    sliver (e.g. the narrow middle of a tight spline, or a border ring
    much thinner than 2x max_radius), which a fixed-phase grid can miss
    entirely regardless of resolution. Random sampling instead finds it
    with probability proportional to its actual area, same as every
    other tier.

    A tier that fits nowhere in what's left (radius(-)erosion comes
    back empty, or the failure-count cutoff hits with zero placements)
    contributes nothing and the next, smaller tier is tried against the
    same `remaining` region -- this is what lets a narrow region skip
    straight to whichever tier actually fits, rather than the old
    hardcoded-2-pass scheme's cliff from "full size" to one fixed half
    size with nothing in between.
    """
    placed: list[tuple[float, float, float]] = []
    remaining = geometry

    for radius in _tier_radii(max_radius, MIN_TIER_RADIUS, TIER_STEP_RATIO, MAX_TIERS):
        if remaining.is_empty or radius < _MIN_USEFUL_RADIUS:
            break
        tier_circles = _dart_throw_pass(remaining, radius, placed, ratio, rng)
        if not tier_circles:
            continue  # this tier doesn't fit anywhere left -- a smaller tier still might
        placed += tier_circles
        remaining = _subtract_circles(remaining, tier_circles)

    return placed


def _max_local_radius(ring_geometry, point: Point, max_radius: float) -> float:
    """
    Largest disc radius (capped at max_radius) that fits entirely inside
    ring_geometry when centered at `point` -- exactly the distance from
    `point` to ring_geometry's own nearest boundary edge (outer or
    inner/hole ring, whichever is closer), the standard "radius of the
    largest circle inscribed at this point" identity for a point already
    known to be inside the geometry. No search needed here, unlike
    _dart_throw_pass's erosion test -- that one has to search for a valid
    CENTER at a fixed radius; the walk (see _pack_ring_walk) already
    fixes the center by construction and only needs the radius.
    """
    return min(max_radius, ring_geometry.boundary.distance(point))


def _scan_step(max_radius: float) -> float:
    """
    Shared arc-length sampling step for every fixed-step scan in this
    module's WALK packer (_sample_ring_radii's width profile,
    _squeeze_centerline_seed's seed search) -- fine relative to
    MIN_TIER_RADIUS (this module's own "not worth a stamp" floor) so it
    can't step clean over the smallest circle that would matter, and
    fine relative to max_radius so a large category radius still gets
    reasonable resolution across a long span.
    """
    return max(MIN_TIER_RADIUS * 0.25, max_radius * 0.02)


def _sample_ring_radii(line, ring_geometry, max_radius: float, scan_step: float) -> tuple[list[float], list[float]]:
    """
    Arc-length positions `ss` (0, scan_step, 2*scan_step, ..., line.length,
    the endpoint always included even if it falls short of a full step)
    and the locally-achievable radius (_max_local_radius) at each --
    the raw width profile _select_ring_seeds picks local maxima from.
    """
    length = line.length
    ss = []
    s = 0.0
    while s < length:
        ss.append(s)
        s += scan_step
    ss.append(length)
    radii = [_max_local_radius(ring_geometry, line.interpolate(s), max_radius) for s in ss]
    return ss, radii


def _select_ring_seeds(line, ss: list[float], radii: list[float], ratio: float) -> list[tuple[float, Point, float]]:
    """
    Phase 1 (SEED) of _pack_line_squeeze -- see module docstring's WALK
    section. Candidate peaks are local maxima of the sampled width
    profile, including both endpoints (so a line that simply narrows or
    widens monotonically from one end still seeds there, same as the old
    march-based walk did) and requiring at least MIN_TIER_RADIUS (this
    module's shared "not worth a stamp" floor) to even be a candidate.

    Accepted SEQUENTIALLY, in arc-length order -- a candidate is dropped
    if it's too close to the immediately PRECEDING accepted seed
    (< (r1+r2)*ratio apart, same separation formula as everywhere else
    in this module), and the walk simply continues forward to the next
    candidate rather than reconsidering it. This is a genuine "walk the
    path, placing the next stamp that fits" -- not a global largest-
    first greedy (an earlier version of this function): sorting every
    candidate peak on the whole ring by height first meant a genuinely
    tall peak could suppress several smaller-but-still-legitimate
    nearby peaks purely because it got processed first, leaving oversize
    unfilled gaps and skewing the ring toward mostly-touching primaries
    with only a "kissing circle" (see _secondary_circle) between them
    almost everywhere, and the intended "pair of big primaries with a
    pair of real secondaries between them" pattern only where two
    similarly-tall peaks happened to survive next to each other. A
    left-to-right walk instead takes every peak in the order it's
    actually encountered, so consecutive primaries (and the gaps
    between them) track the ring's own real width variation directly.

    For a flat/uniform-width run (every sample tied), sequential accept
    reproduces the exact same result the old largest-first version
    already did there (Python's stable sort of an all-tied list is
    itself just left-to-right order) -- this only changes behavior where
    the ring's width genuinely varies from peak to peak.

    Returns (s, point, radius) triples in arc-length order -- the order
    _pack_line_squeeze's gap-filling pass walks them in.
    """
    n = len(radii)
    peak_idxs = [
        i for i in range(n)
        if radii[i] >= MIN_TIER_RADIUS
        and (i == 0 or radii[i] >= radii[i - 1])
        and (i == n - 1 or radii[i] >= radii[i + 1])
    ]

    accepted: list[tuple[float, Point, float]] = []
    for i in peak_idxs:
        s, r = ss[i], radii[i]
        point = line.interpolate(s)
        if accepted:
            _, prev_point, prev_r = accepted[-1]
            if point.distance(prev_point) < (r + prev_r) * ratio:
                continue  # too close to the PRECEDING seed -- keep walking, don't reconsider later
        accepted.append((s, point, r))

    return accepted


def _tighten_seed_spacing(
    line, ring_geometry, max_radius: float, ratio: float, seeds: list[tuple[float, Point, float]],
) -> list[tuple[float, Point, float]]:
    """
    Snaps each seed (after the first, in arc-length order) as close to
    its LEFT neighbor as the geometry actually allows, instead of
    leaving it at whatever scan-grid sample _select_ring_seeds happened
    to accept it at. The coarse scan can only ever land within one
    scan_step of the true minimum-touching position, so every accepted
    seed pair on a flat plateau (a uniform-width ring, the common case
    once border_width is small relative to the category's cluster_radius
    -- the ORIGINAL border-fill scenario this whole packer was built
    for) leaves a small, purely artificial residual gap after nearly
    every seed. Left alone, the squeeze phase (see _secondary_circle)
    correctly finds that gap and -- because leaning off-axis toward
    either ring edge can amplify even a small arc-length slack into a
    non-trivial radius -- dutifully fills it with a small circle, on
    BOTH sides, EVERYWHERE around the ring: exactly the "tons of dinky
    stamps" result this packer exists to avoid, just from a new source
    (seed-grid quantization) instead of the ones already fixed
    (concave-corner clustering, centerline confinement).

    Binary search (not a direct formula, since the true achievable
    radius at a candidate position need not vary linearly, or even
    monotonically, with s in general -- good enough here regardless,
    since it's always searching a span at most one scan_step wide) for
    the smallest s where the new position clears (r_prev + r) * ratio
    from the (already-tightened) left neighbor, then re-measures the
    true local radius there via _max_local_radius (never just reusing
    the original scanned r -- tightening moves the point, which can
    shrink the true achievable radius slightly if the ring's width is
    decreasing in that direction) and re-clamps to it.

    Applied only against the LEFT neighbor, greedily, left to right --
    this can't create a validity problem against the RIGHT neighbor,
    since tightening only ever shrinks a seed's own radius or moves it
    FURTHER from the right neighbor, never closer. The net effect is
    that a long plateau run's total quantization slack collapses to a
    single leftover gap at its far (right) end -- which the squeeze
    phase can still fill if that residual turns out to be real -- rather
    than being smeared as a tiny extra gap after every single seed in
    the run.

    Only fires when the excess slack (distance beyond min_gap) is
    within one scan_step of the minimum -- exactly the amount pure
    grid-sampling quantization can ever produce. A GENUINE gap between
    two real, separated local-maxima peaks (not a plateau) is a
    geometric feature, not an artifact, and is almost always wider than
    that: tightening it anyway would drag a real peak off its own true
    apex and shrink its radius for no reason (confirmed directly: doing
    this unconditionally moved a true r=45 peak seed off-apex down to
    r=37.76, discarding real height it should have kept). The
    scan_step bound is what tells the two cases apart without needing
    to inspect the re-measured radius to guess.
    """
    if len(seeds) < 2:
        return seeds
    max_artifact_slack = _scan_step(max_radius)
    tightened: list[tuple[float, Point, float]] = [seeds[0]]
    # Eligibility (is this pair's gap small enough to be pure scan-grid
    # quantization, not a real feature-driven gap?) is judged against the
    # ORIGINAL, untightened predecessor -- that's the invariant the accept
    # loop in _select_ring_seeds actually guarantees is <= one scan_step.
    # The ACTUAL shift target still anchors on the already-tightened
    # predecessor (tightened[-1]), so a run of several artifact gaps in a
    # row correctly cascades (each one closes fully, not just the first)
    # instead of alternating tighten/skip: comparing against the
    # TIGHTENED predecessor's growing leftward drift would make each
    # pair's measured slack balloon by however much its predecessor
    # already moved, even though the underlying original gap never
    # changed -- exactly that bug, caught by testing this against a
    # uniform-width ring (a long run of identical seeds): every other
    # pair alternated between fully tightened and left completely alone.
    for (_, point_orig_prev, r_orig_prev), (s, point, r) in zip(seeds, seeds[1:]):
        s_prev, point_prev, r_prev = tightened[-1]
        original_slack = point.distance(point_orig_prev) - (r_orig_prev + r) * ratio
        if original_slack <= 0.0 or original_slack > max_artifact_slack:
            tightened.append((s, point, r))  # no slack, or a real (non-artifact) gap -- leave as-is
            continue
        min_gap = (r_prev + r) * ratio
        lo, hi = s_prev, s
        for _ in range(_SEED_SNAP_ITERS):
            mid = (lo + hi) / 2.0
            if line.interpolate(mid).distance(point_prev) >= min_gap:
                hi = mid
            else:
                lo = mid
        point_tight = line.interpolate(hi)
        r_tight = min(r, _max_local_radius(ring_geometry, point_tight, max_radius))
        tightened.append((hi, point_tight, r_tight))
    return tightened


def _achievable_radius(
    x: float, z: float, ring_geometry, max_radius: float, ratio: float,
    point_lo: Optional[Point], r_lo: float, point_hi: Optional[Point], r_hi: float,
) -> float:
    """
    Radius of the largest circle centered at (x, z) that both fits
    inside ring_geometry (0.0 if (x, z) isn't even inside ring_geometry
    at all -- a point outside the ring has no valid placement regardless
    of what its raw boundary distance happens to read, since that
    distance identity only holds for points already known to be
    interior -- see _max_local_radius) and stays tangent-or-looser
    (scaled by `ratio`, this module's usual (r1+r2)*ratio separation)
    against whichever of the two neighbor circles are given. Either
    neighbor may be None (an open end of a non-ring line).
    """
    point = Point(x, z)
    if not ring_geometry.contains(point):
        return 0.0
    r = _max_local_radius(ring_geometry, point, max_radius)
    if point_lo is not None:
        r = min(r, point.distance(point_lo) / ratio - r_lo)
    if point_hi is not None:
        r = min(r, point.distance(point_hi) / ratio - r_hi)
    return max(r, 0.0)


def _squeeze_centerline_seed(
    line, ring_geometry, max_radius: float, ratio: float,
    s_lo: float, point_lo: Optional[Point], r_lo: float,
    s_hi: float, point_hi: Optional[Point], r_hi: float,
) -> tuple[Point, float]:
    """
    Scans arc-length positions along `line` within (s_lo, s_hi), step
    _scan_step(max_radius) apart (endpoint always included even if it
    falls short of a full step -- same convention as _sample_ring_radii),
    for the one where _achievable_radius comes out largest -- i.e. the
    best ON-CENTERLINE circle achievable against whichever of the two
    neighbors are given. `_pack_line_squeeze` uses this directly (not as
    a seed for further refinement) for its two OPEN-END spans, which
    have only one neighbor and so can't use _secondary_circle's
    two-circles-and-a-line construction (see module docstring's WALK
    section) -- a plain centerline floor is what those get instead.

    A plain fixed-step SCAN, not golden-section search (an earlier
    version of this function) -- _achievable_radius is nowhere near
    unimodal in general. Once the one neighbor circle is close to
    touching the far end of the span (the COMMON case, since the seed
    phase already packs most of a ring densely), the window where a
    centerline point clears both the neighbor AND the ring boundary
    collapses to a narrow spike surrounded by a wide flat ZERO region on
    both sides. Golden-section search has no way to locate a hidden
    narrow spike in a mostly-flat landscape -- worse, on the tied
    (both-zero) comparisons that dominate such a landscape it
    deterministically shrinks the SAME direction every time, so it can
    walk cleanly past a real, positive spike and report 0.0 even though
    one exists a few meters away (confirmed directly: a hand-built
    two-neighbor gap with a genuine r=4.95 window golden-section
    reported as entirely unachievable). A scan can't miss a spike wider
    than its own step.
    """
    def achievable(s: float) -> float:
        point = line.interpolate(s)
        return _achievable_radius(point.x, point.y, ring_geometry, max_radius, ratio, point_lo, r_lo, point_hi, r_hi)

    step = _scan_step(max_radius)
    ss = []
    s = s_lo
    while s < s_hi:
        ss.append(s)
        s += step
    ss.append(s_hi)

    best_s, best_r = ss[0], achievable(ss[0])
    for s in ss[1:]:
        r = achievable(s)
        if r > best_r:
            best_s, best_r = s, r
    return line.interpolate(best_s), best_r


def _local_tangent(line, s: float, eps: float = 0.05) -> tuple[float, float]:
    """
    Approximate unit tangent direction of `line` at arc-length `s`
    (central difference, `eps` meters each way, clamped into the line's
    own [0, length] bounds -- degenerates gracefully to a one-sided
    difference near an open end). `line` need not be the centerline --
    _secondary_circle calls this against individual components of
    ring_geometry's own REAL boundary too (see _boundary_components/
    _nearest_boundary_point -- always a single-component LineString,
    NEVER the combined multi-component `ring_geometry.boundary` itself,
    whose `.project`/`.interpolate` are not reliable across components)
    to build the local (tangent, normal) frame its closed-form solve
    runs in. Falls back to an arbitrary but valid unit vector (never
    raises, never returns a zero vector) if even that comes out
    degenerate -- a genuinely zero-length probe window on a zero-length
    line, which _pack_line_squeeze's own `line.length <= 0` guard
    already keeps this from being called on for the centerline case.
    """
    length = line.length
    s0 = max(0.0, s - eps)
    s1 = min(length, s + eps)
    p0, p1 = line.interpolate(s0), line.interpolate(s1)
    dx, dz = p1.x - p0.x, p1.y - p0.y
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return (1.0, 0.0)
    return (dx / norm, dz / norm)


def _solve_tangent_circle(
    x1: float, y1: float, r1: float, x2: float, y2: float, r2: float, ratio: float,
) -> Optional[tuple[float, float]]:
    """
    Closed-form solve for a circle (x, r) tangent to the LINE y=0 (from
    the y>0 side -- so its center's own y-coordinate is exactly r, one
    unknown eliminated for free) and tangent-or-looser (scaled by
    `ratio`, this module's usual separation convention) to two given
    circles (x1,y1,r1) and (x2,y2,r2) -- the classic "circle tangent to
    two circles and a line" construction (the line-as-zero-curvature-
    circle special case of the Apollonius-problem/Descartes-circle-
    theorem family), generalized to the ratio-scaled (not just exactly
    tangent) case this module needs everywhere else too.

    Derivation: subtracting the two squared-distance equations
    ((x-x1)^2+(r-y1)^2 = (ratio*(r+r1))^2, same for point 2) eliminates
    both the x^2 and the r^2(1-ratio^2) terms, leaving x as a LINEAR
    function of r (x = A + B*r). Substituting back gives a standard
    quadratic a*r^2 + b*r + c = 0 in r alone.

    Returns (x, r) for the SMALLEST positive real root, or None if no
    such root exists (no real solution, or every real root is <= 0).
    The other root, when both are positive, is the algebraically valid
    but physically spurious "large circle enclosing both neighbors from
    outside" solution the same tangency equations also admit --
    confirmed numerically during design (e.g. r~19.3 vs a spurious
    r~649 for one test configuration) -- discarded by always taking the
    smaller one, same convention Descartes' circle theorem itself uses
    to pick the "inscribed" branch over the "circumscribing" one.

    x1~=x2 (the two circles project to nearly the same point along the
    line) makes the elimination step's division by (x2-x1) blow up --
    treated as "no solution" (returns None) rather than raising, same
    as every other "nothing valid found" case in this module; expected
    to be rare in practice since seed-phase primaries are spaced along
    the ring's own arc, not stacked along its normal.
    """
    denom = x2 - x1
    if abs(denom) < 1e-9:
        return None

    ratio2 = ratio * ratio
    A = (x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1 + ratio2 * (r1 * r1 - r2 * r2)) / (2.0 * denom)
    B = ((y1 - y2) + ratio2 * (r1 - r2)) / denom

    a = B * B + 1.0 - ratio2
    b = 2.0 * B * (A - x1) - 2.0 * y1 - 2.0 * ratio2 * r1
    c = (A - x1) * (A - x1) + y1 * y1 - ratio2 * r1 * r1

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        roots = [-c / b]
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sq = math.sqrt(disc)
        roots = [(-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)]

    positive = sorted(r for r in roots if r > 0.0)
    if not positive:
        return None
    r = positive[0]
    return A + B * r, r


def _boundary_components(ring_geometry) -> list[LineString]:
    """
    Every individual boundary ring of `ring_geometry` -- a Polygon's own
    exterior plus each of its interiors/holes, across every part if
    `ring_geometry` is a MultiPolygon (several disjoint border rings) --
    as a plain SINGLE-component LineString each, never the combined
    `ring_geometry.boundary` MultiLineString.

    This split exists because shapely's `.project()`/`.interpolate()` on
    a MultiLineString are NOT a reliable "nearest point across every
    component" operation -- confirmed directly during debugging: a point
    sitting exactly ON a ring's own interior (hole) boundary component,
    round-tripped through `.project()`/`.interpolate()` on the combined
    `ring_geometry.boundary`, came back on the EXTERIOR ring instead,
    over 100m away (the interior component apparently isn't addressed
    correctly by the combined MultiLineString's own linear referencing).
    Operating on each component's own plain LineString individually (see
    _nearest_boundary_point) sidesteps that bug entirely -- project/
    interpolate on an actual single-component LineString are reliable,
    verified separately.
    """
    parts = list(ring_geometry.geoms) if hasattr(ring_geometry, "geoms") else [ring_geometry]
    components: list[LineString] = []
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        components.append(LineString(part.exterior.coords))
        for interior in part.interiors:
            components.append(LineString(interior.coords))
    return components


def _nearest_boundary_point(components: list[LineString], point: Point) -> tuple[Point, tuple[float, float]]:
    """
    (nearest point, local tangent there) to `point` across every one of
    `components` (see _boundary_components) -- the TRUE nearest point on
    ring_geometry's real boundary, found by checking each component's
    own plain LineString individually rather than the combined
    MultiLineString, which isn't reliable across components (see
    _boundary_components). Computes each component's own arc-length
    position once and reuses it for both the nearest point and its
    tangent, rather than projecting twice.
    """
    best_point, best_tangent, best_dist = None, (1.0, 0.0), float("inf")
    for comp in components:
        s = comp.project(point)
        candidate = comp.interpolate(s)
        d = point.distance(candidate)
        if d < best_dist:
            best_dist = d
            best_point = candidate
            best_tangent = _local_tangent(comp, s)
    return best_point, best_tangent


def _secondary_circle(
    line, ring_geometry, max_radius: float, ratio: float,
    point_a: Point, r_a: float, point_b: Point, r_b: float, side: float,
) -> tuple[Point, float]:
    """
    The secondary circle for one lateral SIDE (+1.0/-1.0) of the gap
    between two adjacent primary (seed-phase) circles -- see module
    docstring's WALK section. Tangent-or-looser (scaled by `ratio`) to
    BOTH primaries and tangent to the ring's own REAL (curved) boundary,
    via _solve_tangent_circle's closed form against a LOCAL LINEARIZATION
    of that boundary, refined by re-anchoring where the boundary is
    actually curved:

      1. Start `anchor` at the primary pair's midpoint, pushed along
         `side`'s normal (using the CENTERLINE's local tangent at the
         midpoint's arc-length, via _local_tangent -- this is just what
         picks which of the ring's two edges "side" means; the actual
         solve below uses the real boundary, not this direction, once
         anchored) by _SECONDARY_PROBE_FACTOR of the RAW (uncapped by
         max_radius) boundary distance at the midpoint -- enough to
         break the tie between the ring's two edges (which, for a point
         near the centerline, are otherwise roughly equidistant -- a
         coin flip for the nearest-point query below to resolve
         correctly on its own), but deliberately kept well short of the
         wall itself: pushing past max_radius * a few (an earlier
         version of this push, sized off max_radius rather than the
         ring's own local width) OVERSHOT past the correct nearby edge
         entirely whenever border_width was large relative to max_radius
         (this session's real border-fill scenario) and landed nearer to
         some UNRELATED, distant part of the ring's boundary instead --
         confirmed directly: a 400x200 rect at border_width=144,
         max_radius=45 pushed to (45, 180) and anchored to (72, 128), a
         corner of the ring's own inner hole nowhere near the actual
         gap, silently producing a degenerate (zero-radius) result for
         that entire side on every straight-edge primary pair.
      2. Up to _SECONDARY_SOLVE_MAX_ITERS times: find the nearest point
         on ring_geometry's own boundary to `anchor` and its local
         tangent there (_nearest_boundary_point -- checks each boundary
         component individually, see _boundary_components; NOT
         `ring_geometry.boundary` as a whole, whose `.project()`/
         `.interpolate()` are unreliable across components), orient the
         normal to point back into ring_geometry (a one-point
         containment probe, flipped if wrong), transform both primaries
         into that local (tangent, normal) frame, and solve. Stop early
         once consecutive solves land within _SECONDARY_SOLVE_TOLERANCE
         of each other -- real boundary curvature converges in 1-2
         rounds in practice (see module docstring), the remaining
         iterations are headroom.
      3. Final safety clamp: re-measure the TRUE achievable radius at
         the converged point via _achievable_radius (real containment
         against ring_geometry, not the local linearization) and cap
         the result there -- guarantees CONTAINMENT is never violated
         even on boundary shapes irregular enough that the iteration
         hasn't fully converged by the time it runs out.

    Returns (point, 0.0) -- never raises, never returns a circle bigger
    than what's genuinely achievable -- if no valid tangent circle
    exists on this side at all (no real root, degenerate primary
    alignment, or the clamped result comes out non-positive); caller is
    responsible for the usual MIN_TIER_RADIUS floor check.
    """
    mid = Point((point_a.x + point_b.x) / 2.0, (point_a.y + point_b.y) / 2.0)
    mid_tangent = _local_tangent(line, line.project(mid))
    mid_normal = (-mid_tangent[1], mid_tangent[0])
    push = max(ring_geometry.boundary.distance(mid) * _SECONDARY_PROBE_FACTOR, MIN_TIER_RADIUS)
    anchor = Point(mid.x + side * mid_normal[0] * push, mid.y + side * mid_normal[1] * push)

    components = _boundary_components(ring_geometry)
    result_point, result_r = anchor, 0.0
    for _ in range(_SECONDARY_SOLVE_MAX_ITERS):
        nearest, tangent = _nearest_boundary_point(components, anchor)
        normal = (-tangent[1], tangent[0])
        probe = Point(nearest.x + normal[0] * 0.5, nearest.y + normal[1] * 0.5)
        if not ring_geometry.contains(probe):
            normal = (-normal[0], -normal[1])

        def to_local(p: Point) -> tuple[float, float]:
            dx, dz = p.x - nearest.x, p.y - nearest.y
            return dx * tangent[0] + dz * tangent[1], dx * normal[0] + dz * normal[1]

        x1, y1 = to_local(point_a)
        x2, y2 = to_local(point_b)
        solved = _solve_tangent_circle(x1, y1, r_a, x2, y2, r_b, ratio)
        if solved is None:
            return anchor, 0.0
        xl, r = solved
        r = min(r, max_radius)
        world = Point(nearest.x + tangent[0] * xl + normal[0] * r, nearest.y + tangent[1] * xl + normal[1] * r)

        converged = world.distance(anchor) < _SECONDARY_SOLVE_TOLERANCE
        anchor = world
        result_point, result_r = world, r
        if converged:
            break

    true_r = _achievable_radius(
        result_point.x, result_point.y, ring_geometry, max_radius, ratio, point_a, r_a, point_b, r_b,
    )
    return result_point, min(result_r, true_r)


def _place_if_valid(point: Point, r: float, ratio: float, placed: list[tuple[float, float, float]]) -> None:
    """Appends (point.x, point.y, r) to `placed` iff it clears MIN_TIER_RADIUS and doesn't collide with anything already there (the usual (r1+r2)*ratio separation) -- shared by every squeeze-phase placement site in _pack_line_squeeze."""
    if r < MIN_TIER_RADIUS:
        return
    if any(point.distance(Point(ox, oz)) < (r + orad) * ratio for ox, oz, orad in placed):
        return
    placed.append((point.x, point.y, r))


def _pack_line_squeeze(
    line, ring_geometry, max_radius: float, ratio: float, placed: list[tuple[float, float, float]],
) -> None:
    """
    Packs one connected component of a border ring's centerline -- see
    module docstring's WALK section for the two-phase (seed, then
    squeeze the gaps) design this implements. Appends directly into
    `placed`, shared across every component of a multi-part centerline
    (see _pack_ring_walk) so two components passing close to each other
    (e.g. a narrow isthmus between two masked regions) still respect
    each other's circles, not just their own component's chain.

    SQUEEZE is a single FLAT pass now, not recursive: exactly one
    inward + one outward secondary circle (see _secondary_circle) per
    adjacent pair of primary (seed-phase) circles, no chasing further
    leftover sub-gaps after that -- a deliberate simplification over an
    earlier recursive version, since the closed-form tangent-circle
    construction _secondary_circle uses only solves for TWO given
    neighbors at a time, not an open-ended "keep subdividing" gap-fill.

    A component with no seeds at all (every sample below MIN_TIER_RADIUS
    -- the whole line is narrower than one useful circle) still gets one
    whole-line _squeeze_centerline_seed attempt with no neighbors on
    either side, which correctly comes up empty rather than silently
    skipping the component.

    The two OPEN ENDS (before the first seed, after the last) have only
    ONE neighbor, not a pair -- _secondary_circle's tangent-to-two-
    circles-and-a-line construction needs both to pin down a unique
    circle, so it doesn't apply there. Open ends keep the simpler,
    single-neighbor _squeeze_centerline_seed floor instead (one
    on-centerline circle, not two-sided secondaries) -- a deliberate
    scope limit, not an oversight: this packer's whole redesign this
    session was driven by the PAIR case, which is what _secondary_circle
    targets.

    Doesn't special-case the seam where a CLOSED ring (LinearRing) wraps
    back to meet its own starting point -- the two open-end spans (0 to
    the first seed, the last seed to line.length) are filled
    independently, same as for a non-closed line, so the seam can end up
    with a slightly different neighbor spacing than everywhere else.
    Accepted rather than special-cased, same rationale as every earlier
    version of this packer: real border rings are long enough that one
    seam's worth of imperfection isn't worth a lookback-and-adjust pass.
    """
    length = line.length
    if length <= 0:
        return
    ss, radii = _sample_ring_radii(line, ring_geometry, max_radius, _scan_step(max_radius))
    seeds = _select_ring_seeds(line, ss, radii, ratio)
    seeds = _tighten_seed_spacing(line, ring_geometry, max_radius, ratio, seeds)

    if not seeds:
        point, r = _squeeze_centerline_seed(
            line, ring_geometry, max_radius, ratio, 0.0, None, 0.0, length, None, 0.0,
        )
        _place_if_valid(point, r, ratio, placed)
        return

    for _, point, radius in seeds:
        placed.append((point.x, point.y, radius))

    first_s, first_point, first_r = seeds[0]
    point, r = _squeeze_centerline_seed(
        line, ring_geometry, max_radius, ratio, 0.0, None, 0.0, first_s, first_point, first_r,
    )
    _place_if_valid(point, r, ratio, placed)

    for (_, point_a, r_a), (_, point_b, r_b) in zip(seeds, seeds[1:]):
        for side in (1.0, -1.0):
            sec_point, sec_r = _secondary_circle(
                line, ring_geometry, max_radius, ratio, point_a, r_a, point_b, r_b, side,
            )
            _place_if_valid(sec_point, sec_r, ratio, placed)

    last_s, last_point, last_r = seeds[-1]
    point, r = _squeeze_centerline_seed(
        line, ring_geometry, max_radius, ratio, last_s, last_point, last_r, length, None, 0.0,
    )
    _place_if_valid(point, r, ratio, placed)


def _pack_ring_walk(centerline, ring_geometry, max_radius: float, ratio: float) -> list[tuple[float, float, float]]:
    """
    Seed-and-squeeze packing along a border ring's own centerline -- see
    module docstring's WALK section for why this exists instead of
    _pack_circles' dart-throw tiers. `centerline` is the ORIGINAL line
    ring_geometry was stroked from (build_border_ring_geometry's own
    `mask_geometry.boundary` argument), not derived from ring_geometry
    itself -- shapely has no medial-axis/skeleton operation to
    reconstruct a walkable line back out of an already-buffered polygon,
    so the caller has to have kept it (see
    SYNTHETIC_BORDER_CENTERLINE_KIND).

    A LineString/LinearRing packs as a single component; a
    MultiLineString (a mask with a hole, or several disjoint masked
    regions) packs each component independently via _pack_line_squeeze,
    sharing one `placed` accumulator across all of them.
    """
    parts = list(centerline.geoms) if hasattr(centerline, "geoms") else [centerline]
    placed: list[tuple[float, float, float]] = []
    for part in parts:
        if part.geom_type not in ("LineString", "LinearRing") or part.length <= 0:
            continue
        _pack_line_squeeze(part, ring_geometry, max_radius, ratio, placed)
    return placed


def build_border_ring_geometry(mask_geometry, border_width: float):
    """
    Stroke mask_geometry's own outline (its .boundary) by border_width --
    a SYMMETRIC stroke, since Shapely buffers a LineString/MultiLineString
    outward on both sides: half of border_width extends inward from the
    outline, half extends outward. If border_width/2 exceeds however far
    mask_geometry's own outline already sits from the original spline(s)
    it was buffered from, the inward half can reach back past that
    original edge -- intentional, lets a ring dip inside the source shape
    rather than only ever sitting entirely outside it.

    mask_geometry.boundary on a MultiPolygon is a MultiLineString --
    buffering it still produces one separate ring per disjoint region,
    which is what e.g. several individually-marked lake splines want.

    Returns an empty Polygon (never raises) if mask_geometry is None/empty
    or border_width <= 0 -- callers must check .is_empty before using the
    result (e.g. before persisting it into a Feature).
    """
    if mask_geometry is None or mask_geometry.is_empty or border_width <= 0:
        return Polygon()
    return mask_geometry.boundary.buffer(border_width / 2.0)


def _scaled_count(radius: float, spacing: float, density: float) -> int:
    """cluster_count(...) scaled by a fill spec's `density` percent knob (see DENSITY in the module docstring), floored at 1 same as cluster_count's own floor."""
    return max(1, round(cluster_count(radius, spacing) * density / 100.0))


def _cluster_entry_from_record(record: dict) -> dict:
    """
    One placedObjects2 Value.clusters entry (schema confirmed directly
    against a real placedObjects2.json) built from an already-packed,
    schema-neutral record (see pack_cluster_records) -- position/seed/
    count/radius all just carried straight through from pack time, only
    the GRID_ORIGIN_OFFSET shift and placedObjects2's own key names are
    applied here. Keeping the seed/position/count FROZEN at pack time
    (rather than re-rolled on every write-objects run, as an earlier
    version did) is what lets objects.json (see course_output/objects.py)
    serve as a stable, version-agnostic preview/export source -- a
    re-serialize into a different game_version's schema reproduces the
    exact same in-game layout, not a fresh random one.
    """
    return {
        "position": {
            "x": _round(record["x"] - GRID_ORIGIN_OFFSET), "y": "-Infinity",
            "z": _round(record["z"] - GRID_ORIGIN_OFFSET),
        },
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "seed": record["seed"],
        "count": record["count"],
        "radius": _round(record["radius"]),
    }


def _pack_spec(
    feature: Feature, category: AssetCategory, entry: AssetEntry, ratio: float, density: float, rng: random.Random,
    centerline_by_id: Optional[dict] = None,
) -> list[dict]:
    """
    One packing run for a single {"category","type","ratio","density"}
    spec against `feature`'s own geometry -- walk-packed (see
    _pack_ring_walk) if `feature` is a border ring AND its linked
    centerline resolves via `centerline_by_id` (osm_id -> centerline
    geometry, built by pack_cluster_records from the FULL features
    list -- see SYNTHETIC_BORDER_CENTERLINE_KIND), dart-throw tiered
    packed (see _pack_circles) otherwise -- covers every non-border
    spec, plus a border spec whose centerline is missing (a
    features.geojson saved before this existed) or came out empty.

    Returns schema-neutral records -- {"category", "type", "x", "z",
    "radius", "count", "seed", "spline_id"} -- not yet wrapped into any
    game-version's output shape (see pack_cluster_records/
    cluster_records_to_v2019_groups). The seed is drawn here, once, at
    pack time (see _cluster_entry_from_record's docstring for why).
    """
    centerline = None
    if centerline_by_id is not None and feature.kind == SYNTHETIC_BORDER_KIND:
        centerline = centerline_by_id.get(feature.tags.get(PGA_CLUSTER_CENTERLINE_REF_TAG))

    if centerline is not None and not centerline.is_empty:
        circles = _pack_ring_walk(centerline, feature.geometry, category.cluster_radius, ratio)
    else:
        circles = _pack_circles(feature.geometry, category.cluster_radius, ratio, rng)
    return [
        {
            "category": category.id, "type": entry.type, "x": x, "z": z, "radius": radius,
            "count": _scaled_count(radius, entry.spacing, density), "seed": rng.randrange(0, 2**31 - 1),
            "spline_id": feature.osm_id,
        }
        for x, z, radius in circles
    ]


def fill_feature_with_clusters(feature: Feature, rng: Optional[random.Random] = None) -> list[dict]:
    """
    Schema-neutral packed records (see _pack_spec) covering
    `feature.geometry`'s area, one packing run per
    {"category","type","ratio"} spec in feature.tags[PGA_CLUSTER_FILLS_TAG].
    [] if the feature isn't tagged, its geometry has no area (e.g. a
    bare line), or a spec doesn't resolve (skipped with a printed note
    rather than raising -- a stale tag shouldn't break the whole
    pack-objects step).

    Single-Feature API: if `feature` is a border ring, this can't
    resolve its linked centerline (that lives on a SEPARATE Feature --
    see SYNTHETIC_BORDER_CENTERLINE_KIND -- and this function only ever
    sees the one), so it always falls back to dart-throw tiered packing
    for borders. pack_cluster_records, which takes the full features
    list, is the one real caller in the pack-objects pipeline and gets
    the walk-packed version; this function has no callers of its own
    currently, kept as a single-Feature convenience API.
    """
    if rng is None:
        rng = random.Random()
    if feature.geometry.geom_type not in _AREA_GEOM_TYPES:
        return []

    specs = feature.tags.get(PGA_CLUSTER_FILLS_TAG)
    if not specs:
        return []

    records: list[dict] = []
    for spec in specs:
        if spec.get("mode", CLUSTER_FILL_MODE_STAMPS) != CLUSTER_FILL_MODE_STAMPS:
            continue  # mode="spline" -- see pack_spline_records, not this stamp packer
        resolved = _resolve_spec(spec)
        if resolved is None:
            print(f"  NOTE: skipping unresolvable cluster fill spec "
                  f"category={spec.get('category')}/type={spec.get('type')} "
                  "(stale tag or asset_catalog.json no longer has it)")
            continue
        category, entry = resolved
        ratio = spec.get("ratio", DEFAULT_RASTER_RATIO)
        density = spec.get("density", DEFAULT_FILL_DENSITY)
        records.extend(_pack_spec(feature, category, entry, ratio, density, rng))

    return records


def pack_cluster_records(features: list[Feature], rng: Optional[random.Random] = None) -> list[dict]:
    """
    Schema-neutral packed cluster records (see _pack_spec) for every
    Feature carrying PGA_CLUSTER_FILLS_TAG -- the pack-objects step's
    own contribution to objects.json (see course_output/objects.py),
    consumed from there by cluster_records_to_v2019_groups (or a future
    v2021+ equivalent) at write-objects time, and directly by the GUI's
    preview -- neither has to re-run this packing itself.

    Takes the FULL features list (not pre-filtered to tagged ones) --
    unlike every other tagged-feature scan in this module, this one also
    needs to see any untagged SYNTHETIC_BORDER_CENTERLINE_KIND Features
    so it can build centerline_by_id (osm_id -> centerline geometry) up
    front, for _pack_spec to resolve each border ring's own
    PGA_CLUSTER_CENTERLINE_REF_TAG against.

    Each record carries "spline_id" (the source Feature's own osm_id) --
    a real back-reference objects.json gets "for free" out of packing
    per-Feature, unlike placedObjects2.json's merged-by-asset groups
    (see build_cluster_objects_v2019... now cluster_records_to_v2019_groups),
    which lose it. The GUI's Objects-tab cluster-fill-row highlight used
    to have to reconstruct this via point-in-polygon geometry checks
    against the packed positions (see PGA2k_gen_gui.py's
    _on_object_selected); with objects.json as the preview source, a
    plain spline_id filter replaces that.
    """
    if rng is None:
        rng = random.Random()

    centerline_by_id = {
        f.osm_id: f.geometry for f in features
        if f.kind == SYNTHETIC_BORDER_CENTERLINE_KIND and f.osm_id is not None and not f.geometry.is_empty
    }

    records: list[dict] = []
    for feature in features:
        if feature.geometry.geom_type not in _AREA_GEOM_TYPES:
            continue
        specs = feature.tags.get(PGA_CLUSTER_FILLS_TAG)
        if not specs:
            continue
        for spec in specs:
            if spec.get("mode", CLUSTER_FILL_MODE_STAMPS) != CLUSTER_FILL_MODE_STAMPS:
                continue  # mode="spline" -- see pack_spline_records, not this stamp packer
            resolved = _resolve_spec(spec)
            if resolved is None:
                print(f"  NOTE: skipping unresolvable cluster fill spec "
                      f"category={spec.get('category')}/type={spec.get('type')} "
                      "(stale tag or asset_catalog.json no longer has it)")
                continue
            category, entry = resolved
            ratio = spec.get("ratio", DEFAULT_RASTER_RATIO)
            density = spec.get("density", DEFAULT_FILL_DENSITY)
            records.extend(_pack_spec(feature, category, entry, ratio, density, rng, centerline_by_id))

    return records


def subdivide_polygon(poly: Polygon, max_size: float = MAX_SPLINE_FILL_PIECE_SIZE_M) -> list[Polygon]:
    """
    Recursively splits poly's bounding box in half along its longer
    axis, clipping poly into each half, until every piece's bbox is
    <= max_size on both axes -- this project's own port of
    ref/generate_rough_border_v2.py's subdivide_poly (same recursive-
    bisection approach), used by pack_spline_records to keep an
    oversized object-spline fill under the engine's apparent per-spline
    bounding-box cap (see MAX_SPLINE_FILL_PIECE_SIZE_M).

    poly must be a Polygon, not MultiPolygon -- a caller with
    MultiPolygon feature geometry (e.g. a "Use mask" clip split into
    several disjoint pieces) subdivides each of its .geoms separately.
    """
    min_x, min_z, max_x, max_z = poly.bounds
    if (max_x - min_x) <= max_size and (max_z - min_z) <= max_size:
        return [poly]

    if (max_x - min_x) >= (max_z - min_z):
        split = (min_x + max_x) * 0.5
        half_a = box(min_x, min_z, split, max_z)
        half_b = box(split, min_z, max_x, max_z)
    else:
        split = (min_z + max_z) * 0.5
        half_a = box(min_x, min_z, max_x, split)
        half_b = box(min_x, split, max_x, max_z)

    pieces: list[Polygon] = []
    for clipped in (poly.intersection(half_a), poly.intersection(half_b)):
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiPolygon):
            for geom in clipped.geoms:
                pieces.extend(subdivide_polygon(geom, max_size))
        elif isinstance(clipped, Polygon):
            pieces.extend(subdivide_polygon(clipped, max_size))
        # else: a degenerate sliver intersection (Point/LineString/
        # GeometryCollection) -- discard, not a fillable area.
    return pieces


def pack_spline_records(features: list[Feature]) -> list[dict]:
    """
    Schema-neutral object-spline-fill records -- {"category", "type",
    "waypoints", "fill_pct", "spline_id"} -- for every mode="spline"
    fill spec (see pack_cluster_records's mode="stamps" counterpart).

    Unlike stamp-mode packing, there's no circle placement here and
    nothing to freeze via RNG: an object-spline fill is just the
    feature's own polygon (already the buffered ring for a border
    fill, already the clipped shape for a masked fill -- the exact
    same geometry _pack_spec's dart-throw/ring-walk packing consumes
    for stamp mode) chopped into <= MAX_SPLINE_FILL_PIECE_SIZE_M pieces
    (subdivide_polygon) and reported as-is. fill_pct is the spec's
    existing density percent / 100 -- reusing the same knob stamp-mode
    already exposes rather than adding a second one (see the
    conversation's decision).

    waypoints are the piece's exterior ring, course-local frame, closed
    point dropped (shapely repeats the first point at the end) --
    course_output/objects.py's object_spline_fill_records_to_v2021_groups
    turns these into the game's degenerate-handle waypoint dicts and
    applies GRID_ORIGIN_OFFSET at write time, same "freeze at pack,
    format at write" split as pack_cluster_records.
    """
    records: list[dict] = []
    for feature in features:
        if feature.geometry.geom_type not in _AREA_GEOM_TYPES:
            continue
        specs = feature.tags.get(PGA_CLUSTER_FILLS_TAG)
        if not specs:
            continue
        polys = (
            list(feature.geometry.geoms) if feature.geometry.geom_type == "MultiPolygon"
            else [feature.geometry]
        )
        for spec in specs:
            if spec.get("mode", CLUSTER_FILL_MODE_STAMPS) != CLUSTER_FILL_MODE_SPLINE:
                continue
            resolved = _resolve_spec(spec)
            if resolved is None:
                print(f"  NOTE: skipping unresolvable object-spline fill spec "
                      f"category={spec.get('category')}/type={spec.get('type')} "
                      "(stale tag or asset_catalog.json no longer has it)")
                continue
            category, entry = resolved
            fill_pct = spec.get("density", DEFAULT_FILL_DENSITY) / 100.0
            for poly in polys:
                for piece in subdivide_polygon(poly, MAX_SPLINE_FILL_PIECE_SIZE_M):
                    if piece.is_empty or not isinstance(piece, Polygon):
                        continue
                    waypoints = list(piece.exterior.coords[:-1])
                    if len(waypoints) < 3:
                        continue
                    records.append({
                        "category": category.id, "type": entry.type,
                        "waypoints": waypoints, "fill_pct": fill_pct,
                        "spline_id": feature.osm_id,
                    })
    return records


def cluster_records_to_v2019_groups(records: list[dict]) -> list[dict]:
    """
    placedObjects2 groups (v2019 scheme) from pack_cluster_records'
    schema-neutral output -- one {"Key": {"category","type","theme"},
    "Value": {"items": [], "clusters": [...]}} group per distinct
    (category,type,theme) seen across `records`, merging records that
    share an asset (same per-key grouping idea as objects.py's
    build_tree_objects_v2019). Pure formatting -- no packing, no RNG --
    so it's cheap enough to re-run on every write-objects call even
    though the positions/seeds themselves are already frozen.
    """
    groups: dict[tuple[int, int, bool], dict] = {}
    for record in records:
        entry = _ENTRIES_BY_KEY.get((record["category"], record["type"]))
        category = ASSET_CATEGORIES.get(record["category"])
        if entry is None or category is None:
            continue  # asset_catalog.json changed since this record was packed
        key = (category.id, entry.type, entry.theme)
        group = groups.setdefault(key, {
            "Key": {"category": category.id, "type": entry.type, "theme": entry.theme},
            "Value": {"items": [], "clusters": []},
        })
        group["Value"]["clusters"].append(_cluster_entry_from_record(record))
    return list(groups.values())
