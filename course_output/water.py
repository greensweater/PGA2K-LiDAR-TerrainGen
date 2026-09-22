"""
course_output/water.py

Builds userLayers.json's "water" entries (see userLayers.py's
_BLANK_USER_LAYERS_SCHEMA) from OSM water Features -- one or more flat,
rotated, horizontal plane objects per water body, sized/positioned to
best-fit its polygon, elevated to a robust high point of the terrain
that already covers that area. Default is one object per body (fit_
water_rectangle, a single minimum-rotated-rectangle); build_water_
objects' opt-in multi_tile_water instead fills the polygon with
several smaller, possibly-overlapping tiles that hug its real boundary
far more closely (fit_water_tiles) -- see that function's docstring
and "Fit shape" below. Every tile belonging to one water body shares
that body's single water level and flow settings; only the per-tile
position/rotation/scale differ.

Schema confirmed directly (not derived/guessed) against a real water
entry from userLayers.json:
    surfaceCategory: 9        same surface id as splines.py's
                               FEATURES_TO_SURFACES["water"]
    position: {x, y, z}       y is the water LEVEL (elevation) -- and,
                               critically, an EXPLICIT absolute number,
                               not a sentinel like userLayers.py's
                               terrain stamps' "-Infinity". Terrain
                               stamps can get away with "-Infinity"
                               because the game computes their actual
                               rendered height from `value` plus every
                               other stamp's blend at load time; a
                               water plane has no such computation --
                               position.y IS the elevation it renders
                               at, verbatim. That means it has to
                               already be fully, correctly shifted by
                               whatever normalize_stamp_heights applied
                               to the rest of the course -- see
                               "Water level" below for why this was
                               getting missed.
    rotation: {x:0, y, z:0}   y is the plane's yaw, degrees -- a compass
                               BEARING (0=north/+Z, 90=east/+X, sweeping
                               clockwise), NOT a standard math CCW angle
                               from +X. See build_water_objects' inline
                               comment on the rotation.y assignment for
                               how this was confirmed (a real course
                               with a ~46-degree-oriented pond rendering
                               a clean 90 degrees off, long axis on the
                               wrong side, while every more-cardinal-
                               aligned pond looked fine -- the telltale
                               signature of a reflection-type convention
                               mismatch, not a uniform swap/offset bug).
    _orientation: 0.0         always 0.0 in the reference sample --
                               unlike userLayers.py's terrain stamps,
                               which set this to the actual rotation;
                               reproduced as given here, not assumed
    scale: {x, y:1.0, z}      see "Geometry" below -- NOT simply the
                               fitted rectangle's width/depth in meters
    type: 72                  fixed, confirmed from the reference sample
    value: same as position.y -- redundant, both carry the water
                               level; reproduced as given
    holeId: -1
    options: {flowOrientation: 0.0, flowSpeed: 1.0} -- reproduced as
                               given; no per-water-body customization
                               implemented yet (still/uniform flow)
    radius: 0.0
    orientation: 0.0          NOT the same value as rotation.y in the
                               reference sample (unlike terrain stamps'
                               orientation/_orientation, which
                               duplicate each other) -- reproduced as
                               given rather than assumed to mirror
                               rotation.y

Geometry: two rounds of guessing at the base plane mesh's real-world
size, both from inference against an observed discrepancy, before a
real measurement settled it. First guess: a flat 1x1 m unit quad, so
scale.x/scale.z could just BE the real-world width/depth in meters
directly. Confirmed wrong in practice (water planes rendering "way
bigger" than their actual pond) -- Unity's own built-in Plane
primitive is a well-known 10x10 unit mesh by default, so
WATER_MESH_BASE_SIZE_M was set to 10.0 to correct for that. ALSO
confirmed wrong in practice (water planes still rendering roughly 1/5
the correct size -- too small -- after that fix): dividing by a number
5x too large gives scale values 5x too small, which points at the real
base size being closer to 2.0. WATER_MESH_BASE_SIZE_M became 2.0 --
still an inference at that point, which (per direct measurement, see
below) undersized every water plane by ~7.8% per side (~14.9% in
area): plausible-looking mid-magnitude errors like this are exactly
why the earlier "still an inference" note asked for a real measurement
rather than a fourth round-number guess. Now CONFIRMED by direct
in-game measurement (see that constant's own comment for the full
method): the mesh's real, human-authored base size is a clean 1.8 m
once its dead edge border is accounted for.
WATER_MESH_BASE_SIZE_M's value is back-derived from that 1.8 m figure,
not hardcoded directly, so it stays correct if WATER_EDGE_BORDER_
FRACTION's own measurement ever changes. A small WATER_RECT_MARGIN is
also applied on top, so the plane errs toward "same size or slightly
larger" than the fitted rectangle rather than an exact, zero-tolerance
fit that could leave a sliver of real pond peeking out at the corners.

The mapping from the fitted rectangle's two edge lengths to scale.x
vs. scale.z (edge1/"width" -> scale.x, edge2/"depth" -> scale.z) is
this module's own convention and is NOT independently at issue here --
fit_water_rectangle's output was confirmed correct by plotting it
directly against a real pond's source polygon (matched exactly,
including which edge is longer). What WAS wrong, and is now fixed, is
purely the rotation.y value paired with that convention -- see the
compass-bearing note on rotation above and build_water_objects' inline
comment.

Fit shape (single-tile, fit_water_rectangle): shapely's
minimum_rotated_rectangle -- the smallest-area rectangle (at any
rotation, not just axis-aligned) that fully contains the water polygon.
Matches "a square plane, placed and scaled" -- one rotated rectangle,
not a closer per-vertex fit -- while wasting much less area than a
plain axis-aligned bounding box would on any non-axis-aligned or
elongated pond. Still leaves real overshoot area outside the polygon
for irregular pond shapes, though -- confirmed on a real course as the
cause of water visibly rendering over/into land the polygon doesn't
actually cover, since nothing guarantees surrounding terrain stays
above the water level outside the polygon (dig-water only carves an
INWARD-buffered basin, and there's no separate bank-building step).

Fit shape (multi-tile, fit_water_tiles, opt-in via multi_tile_water):
simplifies the polygon boundary, then places one rectangle per boundary
edge (ray-cast into the interior for depth -- same technique as
terrain/contour_layers.py's "rect" fill mode/_fallline_edge_stamps, the
template this was adapted from), followed by a greedy dedup pass that
drops a candidate once it stops contributing meaningful new coverage.
Each per-edge tile is also extended by a tunable ABSOLUTE overlap_m at
both ends (still centered the same, per direct instruction), so
neighboring tiles genuinely overlap along the boundary rather than
meeting exactly corner-to-corner -- an exact meet measured as visibly
"inset" once real float rounding and the dedup pass's own ordering
were involved. Overlap between accepted tiles is expected and fine --
PGA renders overlapping same-height water planes with a clean seam --
only overshoot past the polygon boundary is minimized (to a tunable
tolerance, not eliminated: very concave pond shapes can still leave
small gaps). Measured against real ponds (default overlap_m=1.0): cuts
overshoot from roughly 20-65% of the pond's own area down to a few
percent for most ponds -- a SMALL pond can still see it run higher
(observed ~10% on a ~130 sqm pond), since overlap_m being an absolute
meter value rather than a fraction of pond size is the whole point
(the seam gap it's fixing doesn't scale with pond size either), not a
bug to chase toward zero by shrinking the default.

Fit shape (multi-tile, fit_water_stripes, opt-in via multi_tile_water
+ water_fill_mode="stripe"): a second multi-tile algorithm, addressing
a real limitation of the edge-fill mode above -- its greedy dedup pass
can reject a candidate tile that was the only thing that would have
covered some patch of a sufficiently complex pond, leaving a real,
unpredictable gap ("hole"). By direct instruction, 100% coverage is a
HARD requirement for this mode (overshoot/stripe-count/aspect-ratio
are NOT -- "1:100 or more is fine"), so this algorithm is built around
that guarantee rather than balancing it against tightness of fit the
way edge-fill does.

It walks outward from the pond's own fitted-rectangle center in both
directions (using that rectangle's rotation as a FIXED orientation for
every stripe, unlike edge-fill's per-edge-varying rotation; no
separate "seed" step -- both walks simply start a hairline before the
exact center). Each stripe's WIDTH is never guessed from a single
probe: it's the true footprint of a real shapely intersection between
the polygon and the stripe's own candidate depth-band (_ws_band_
u_extent), which by construction can never miss real polygon area
inside that band, no matter how the boundary bulges -- an earlier,
since-replaced version sized width from a handful of sampled points
and left measurable real gaps wherever the true boundary bulged out
BETWEEN samples. Each stripe's DEPTH is fit adaptively: try the
largest remaining candidate (out to the true wall, itself found via
the same real-intersection technique, not a single ray -- a single ray
was also confirmed to sometimes stop short of a curving tip, leaving a
gap there too), and if a width sized to that whole candidate would
overshoot the true boundary by more than overlap_m at either of its
own edges, bisect the candidate and retry until it passes or hits
min_edge_m -- at which point it's accepted regardless of remaining
overshoot, since by direct instruction coverage always wins over
tightness.

overlap_m (by direct instruction, one knob, not two) applies ONLY to
the true polygon boundary -- how far a stripe's edge may overshoot it,
checked at every bisection step. Consecutive stripes overlap by a
separate, tiny, FIXED, non-tunable amount (_WS_SEAM_OVERLAP_M, a
couple cm) instead -- that seam only needs to guard against float-
precision gaps at the join, not express any real design tolerance.
Verified against every real pond in a full course: 0.0% gap on every
single one, at a total overshoot cost of ~9% of pond area (vs. edge-
fill's ~3%) -- prefer "edge" for a fairly regular pond shape where
tight overshoot matters most, "stripe" whenever guaranteed complete
coverage matters more than how tightly the tiles hug the shape.

Water level: the MEAN (not a low percentile or the true minimum -- see
point 6 below) of the actual rendered terrain anywhere along the water
body's own real OUTLINE (the boundary of the raw OSM water WAY polygon
itself, not the union of tiles/stripes fitted to it -- see point 6;
that fitted footprint is deliberately inflated past the way's own edge,
e.g. WATER_RECT_MARGIN, so sampling it instead of the way undershoots
the real requirement of tracking the actual, tagged shoreline), minus
a small fixed safety margin (DEFAULT_WATER_LEVEL_SAFETY_MARGIN_M).
Deliberately the OUTLINE, not the whole interior area -- see point 4
below, a real correction, not just a design choice made right the
first time. Computed by sampling TerrainModel densely (TerrainModel.
evaluate_many) at points spaced along the water way polygon's own
boundary curve (DEFAULT_WATER_LEVEL_BOUNDARY_SPACING_M apart) -- see
_water_level_from_way_boundary.

This replaced an earlier version (still worth understanding, since the
lessons generalize) that only evaluated at STAMP CENTERS falling
inside the raw polygon, then took the 90th percentile as a "robust
high point." Four things changed to get here:

  1. It read stamp.value directly. That's wrong: normalize_stamp_
     heights (see userLayers.py) does NOT rewrite existing stamps'
     .value fields to shift them -- it APPENDS one new course-wide
     raise-tool stamp whose effect only shows up once the WHOLE stamp
     list is evaluated together through a TerrainModel. Every other
     stamp's own .value is still its pre-shift number. Reading .value
     directly therefore silently used pre-shift elevations for nearly
     every stamp -- which is exactly why some water bodies lined up
     fine (courses needing little or no shift) while others were off
     by "hundreds of feet" (courses needing a large one, e.g. real-
     world elevations far from a local zero). Fixed by evaluating a
     TerrainModel built from the (already-normalized) stamp list this
     module receives, instead of trusting any individual stamp's own
     stored value -- still true of the current approach.

  2. It took the single MINIMUM instead of a robust high point. Made
     sense when only hand-identified error hotspots got stamps; wrong
     once "scatter" mode (see terrain/adaptive_refine.py) started
     blanketing the whole course, including in and around water bodies
     themselves, where real RAW LIDAR returns are frequently noisy
     (partial penetration, surface bounce) -- the single lowest RAW
     reading inside a water polygon was disproportionately likely to
     be a bad-data artifact, not the real basin floor. Switched to a
     90th-percentile HIGH point instead, reasoning a pond's surrounding
     bank is a better-behaved signal than its noisy interior.

  3. Confirmed in practice (direct report): sampling only at stamp
     CENTERS, even with a robust-high-point percentile, still isn't
     enough -- some ponds rendered visibly floating above the terrain
     at their edges, wherever a coarse/high-RMS-error stamp left the
     actual resolved terrain lower than any nearby stamp CENTER
     happened to capture. Switched to a dense-render minimum WITHOUT
     reintroducing the raw-LIDAR-noise problem #2 originally guarded
     against: TerrainModel.render() reflects the already-FITTED stamps
     (the actual in-game terrain, warts and all), not raw per-point
     LIDAR returns -- there's no equivalent noisy-single-point-artifact
     risk to guard against with a percentile here, since a stamp-fitted
     surface is already far smoother than raw LIDAR.

  4. That dense-render minimum was first taken over the footprint's
     WHOLE INTERIOR, not just its outline -- wrong, caught by direct
     correction: a real pond's basin is typically deepest in the
     middle (natural bowl shape, or an actual dug depression), so a
     whole-interior minimum sinks the plane far below the shoreline
     instead of just below it, burying the water under the surrounding
     terrain entirely rather than fixing the floating-at-the-edges
     symptom it was meant to fix. Only the immediate area around the
     pond's EDGE determines whether the plane looks like it floats
     there -- the deep interior is irrelevant, since the water surface
     covers it regardless of how deep it is. Restricting the minimum to
     a thin band hugging the footprint's boundary (not its interior)
     is what actually answers "how high can this plane sit without any
     part of its visible edge poking above the surrounding ground."

  5. Restricting to the OUTLINE still wasn't enough: a real course had
     one pond (a small hazard right next to a tee) rendering visibly
     buried, 3m below its true rim, confirmed via real in-game stake
     markers placed along the pond's actual visible edge -- the stakes
     read a tight, consistent ~79.9m band directly off TerrainModel,
     but the rasterized boundary-band approach above returned 76.7m.
     Two compounding causes, both boiling down to "a single outlier
     point along the boundary, not real widespread low terrain": (a)
     dig-water (ingest/heightmap.py) leaves only a thin, ~1m undug
     collar around the true polygon edge before dropping to the dug
     basin floor -- a rasterized band ~1 grid cell wide can straddle
     that thin collar and sample already-dug cells while nominally
     still being "the boundary"; (b) independent of the rasterization,
     coarse/sparse terrain stamps in that specific area didn't cleanly
     preserve the sharp dig-buffer edge, so even a single ordinary
     boundary point (nothing geometrically unusual about it) read a
     basin-floor height instead of the true rim height. Taking the
     STRICT MINIMUM across the boundary means any one such artifact
     point -- real terrain or not -- single-handedly sets the whole
     pond's level, exactly the same "single point is disproportionately
     likely to be a bad-data artifact" lesson point 2 above already
     learned once, just resurfacing along the boundary curve instead of
     the raw interior. Median was tried and rejected: most real ponds
     on a real course have several meters of LEGITIMATE rim relief
     (not artifacts), so a mid-distribution statistic reintroduces
     genuine floating on a large fraction of the boundary. Switched to
     the 5th PERCENTILE of dense points sampled directly along the
     boundary CURVE (TerrainModel.evaluate_many at fixed arc-length
     spacing, not a rasterized grid band -- avoids the thin-collar
     bleed-through in (a) outright) instead of the strict minimum --
     by direct instruction, chosen over a fancier local neighbor-
     window outlier-rejection alternative that tested as more surgical
     (zero effect on ponds without a real artifact, vs. every pond
     shifting slightly under a global percentile) but was more code for
     the same practical result on the one pond that actually needed it.

  6. Confirmed in practice (direct report): even after all of the
     above, ponds still rendered a bit too low across the board -- not
     one outlier pond, most of them, by a modest but consistent amount.
     Two more things were wrong, independent of each other: (a) the
     boundary being sampled was the FITTED footprint (the union of the
     actual rectangle(s)/tile(s) placed for the pond, deliberately
     inflated past the raw OSM polygon by WATER_RECT_MARGIN/tile
     overlap -- see fit_water_rectangle), not the real OSM water WAY
     itself -- so every sample point sat outside the pond's true tagged
     edge, on surrounding bank/dig-collar terrain that has no reason to
     track the real water height as closely as the way's own traced
     boundary does; (b) the 5th PERCENTILE from point 5, while a real
     fix for the single-outlier problem, is still a statistic that
     deliberately reaches toward the low tail of the sampled
     distribution on EVERY pond, not just the rare one with an
     artifact -- a small systematic downward bias applied uniformly,
     exactly matching "every pond a bit too low" rather than "one pond
     very wrong." Fixed by switching to (a) the raw way polygon
     (`f.geometry`, before any tile-fitting inflation) as the boundary
     sampled, and (b) the MEAN of those dense boundary samples instead
     of a low percentile: a real water way's traced boundary is,
     definitionally, drawn at the actual shoreline, so the large
     majority of dense samples along it already cluster tightly around
     the true water height, and a mean uses that whole clustered
     distribution instead of deliberately discarding it in favor of the
     low tail. This also improves on point 5's own single-artifact-
     point concern rather than reintroducing it: one outlier sample
     pulls a mean of many dense points proportionally less than it
     pulls a percentile that's already anchored near the tail where
     outliers live. Renamed _water_level_from_footprint ->
     _water_level_from_way_boundary to match (the function's argument
     is no longer a "footprint" at all).

This means water objects must be built AFTER terrain generation/
refinement AND normalize_stamp_heights have both already run (see
PGA2k_gen.py's step_write_water, which re-runs that same load/
normalize pipeline itself via _load_normalized_stamps rather than
trusting whatever's already on disk) -- not before, and not against the
pre-normalization stamp list. A water body whose footprint yields no
valid render grid (degenerate geometry) has no level to compute from
and is skipped -- logged by the caller, not silently dropped here.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient

from ingest.osm import Feature
from course_output.userLayers import GRID_ORIGIN_OFFSET
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel

WATER_SURFACE_CATEGORY = 9  # same id as splines.py's FEATURES_TO_SURFACES["water"]
WATER_TYPE = 72
_DECIMALS = 3


def _measured_edge_border_fraction(brush_id: int) -> float:
    """
    Fraction of a square brush's own half-width (radius, in the
    Chebyshev/SHAPE_SQUARE sense -- see terrain/brush_profiles.py) that
    renders at ZERO weight right at the outer edge -- a real, dead
    border baked into the brush's own 512x512 PNG asset, measured
    directly from BRUSH_PROFILES[brush_id]'s real pixel-scan data
    (terrain/brush_profiles.py), not another guess. For brush 72 this
    comes out to 6 px out of a 256 px half-width (~2.35%) -- the exact
    same border userLayers.py's build_course_wide_stamp docstring
    already estimates at "3%", now precisely measured instead.

    WATER_TYPE == 72, the exact same brush/type id as terrain's "hard
    square" stamps -- confirmed in practice (water planes rendering
    consistently undersized vs. their fitted rectangle) that the water
    plane reuses that same square decal, dead border included. Terrain
    itself never needed this correction: type 72 there is only ever
    used hugely oversized (see brush_profiles.py's own docstring), so
    every rendered point sits deep in the saturated interior, nowhere
    near this edge. A water plane's edge IS the pond's actual boundary,
    so here it's directly visible as "every pond a bit too small."
    """
    samples = BRUSH_PROFILES[brush_id].sorted_samples()  # (r, weight), r=0 center -> r=1 edge
    saturated = samples[samples[:, 1] > 0.5]
    return 1.0 - float(saturated[:, 0].max())


# The dead-border fraction measured above, converted into a scale
# multiplier: if the outer WATER_EDGE_BORDER_FRACTION of the mesh's own
# radius never renders, the plane's ACTUAL visible footprint is only
# (1 - WATER_EDGE_BORDER_FRACTION) of whatever scale is set -- so scale
# must be inflated by the reciprocal for the VISIBLE water surface to
# actually match the fitted rectangle, not just the invisible mesh
# bounds. Applied in build_water_objects, on top of (not instead of)
# WATER_MESH_BASE_SIZE_M below and WATER_RECT_MARGIN.
WATER_EDGE_BORDER_FRACTION = _measured_edge_border_fraction(WATER_TYPE)
WATER_BORDER_SCALE_CORRECTION = 1.0 / (1.0 - WATER_EDGE_BORDER_FRACTION)

# CONFIRMED by direct in-game measurement, not another guess -- the
# two earlier rounds here (a 1x1 unit quad, then Unity's stock 10x10
# Plane primitive) were both inferences from an observed size
# discrepancy, and this module used to say a direct measurement would
# settle it outright. That measurement: two water objects hand-placed
# directly into userLayers.json (one axis-aligned square, scale=
# 56.1218452; one rotated rectangle, rotation.y=49.8288841 deg,
# scale.x=21.4116669/scale.z=58.8747635), then their FOUR RENDERED
# CORNERS each marked in-game with a stake object and read back by
# position. Comparing each stake-measured span against its water
# object's own scale value gives four independent real meters-per-
# scale-unit readings (square's width and depth, rectangle's short and
# long side): 1.79977, 1.80152, 1.79966, 1.80049 -- agreeing to within
# 0.03% of each other, and unmistakably clustered on a clean human-
# authored 1.8 m, not a coincidence worth chasing to more decimal
# places (the residual spread is measurement/stake-placement noise, on
# the order of millimeters). Same dataset, for free, also cross-checked
# two OTHER things this module depends on against real geometry rather
# than "no longer looks wrong": the rotated rectangle's measured
# long-axis direction matched the rotation.y compass-bearing formula
# (see build_water_objects' rotation.y comment) to within 0.052
# degrees, and both objects' measured centers (averaged over their 4
# stakes) matched their own JSON position.x/z to within a few
# centimeters.
#
# Back-derived from that clean 1.8 m figure by dividing out
# WATER_EDGE_BORDER_FRACTION above (real, independently texture-
# measured -- not implicated by this correction). The previous
# WATER_MESH_BASE_SIZE_M=2.0 guess OVER-estimated how many real meters
# one unit of scale.x/scale.z actually renders as (1.9529 assumed vs.
# 1.8 true) -- which, since scale is computed by DIVIDING the intended
# real-world width by that assumed per-unit figure, means every scale
# value came out too SMALL, and every water plane rendered UNDERSIZED
# by ~7.8% per side (~14.9% in area) versus its fitted rectangle. That
# direction matches the original symptom this constant was chasing --
# "every pond a little too small" -- confirmed by direct arithmetic
# check against a real pond, not just by construction: verify before
# assuming a percentage's sign, an inverse (divide-by) relationship
# flips it easily.
WATER_MESH_BASE_SIZE_M = 1.8 / (1.0 - WATER_EDGE_BORDER_FRACTION)

# Small safety margin on the fitted rectangle so the water plane errs
# toward "same size or slightly larger" than the real pond outline,
# not an exact zero-tolerance fit -- see module docstring.
WATER_RECT_MARGIN = 1.02

# _water_level_from_way_boundary's tunables -- see module docstring's
# "Water level" and that function's own docstring. Not exposed as CLI/
# GUI parameters: internal, well-reasoned defaults, matching this
# module's other non-tunable internal constants (e.g. _WS_SEAM_
# OVERLAP_M) rather than proliferating knobs with no strong reason to
# differ per-course.
DEFAULT_WATER_LEVEL_BOUNDARY_SPACING_M = 0.25  # arc-length spacing between sampled boundary points
DEFAULT_WATER_LEVEL_SAFETY_MARGIN_M = 0.15  # subtracted from the mean of the boundary samples --
                                              # sits slightly BELOW the sampled value, not just at it
                                              # (even dense sampling can still miss the exact true
                                              # minimum between sample points) -- see module
                                              # docstring's "Water level" point 6

# fit_water_tiles' tunables -- see that function's docstring. Follows
# the same plain DEFAULT_*_M module constant + matching keyword-arg
# convention terrain/contour_layers.py already uses for its own
# rect-fill knobs (DEFAULT_RECT_TOLERANCE_M etc.).
DEFAULT_WATER_TILE_TOLERANCE_M = 1.5
DEFAULT_WATER_TILE_MIN_EDGE_M = 2.0
DEFAULT_WATER_TILE_MAX_SEARCH_M = 200.0
DEFAULT_WATER_TILE_WIDTH_SAMPLES = 5
DEFAULT_WATER_TILE_REDUNDANCY_RATIO = 0.05
DEFAULT_WATER_TILE_OVERLAP_M = 1.0

# fit_water_stripes' tunables -- see that function's docstring. By
# direct instruction, this applies ONLY to the true polygon boundary
# (the max a stripe's width may overshoot the real pond edge, which
# _ws_walk_direction enforces via adaptive depth-bisection -- see that
# function) -- NOT to the seam between adjacent stripes, which is a
# separate, tiny, fixed, non-tunable amount (_WS_SEAM_OVERLAP_M) purely
# to guard against float-precision gaps at the join, not a real design
# tolerance. Also a separate constant from DEFAULT_WATER_TILE_OVERLAP_M
# (edge-fill's own knob) since the two modes tune against different
# failure modes even though the name rhymes.
DEFAULT_WATER_STRIPE_OVERLAP_M = 1.5

# Boundary-simplify tolerance applied ONLY to reduce probe-line noise
# (extra MultiLineString fragments from closely-spaced near-duplicate
# LIDAR/OSM vertices) -- NOT structural the way DEFAULT_WATER_TILE_
# TOLERANCE_M is for fit_water_tiles (which walks boundary EDGES
# directly, so an unsimplified wiggly boundary means one rectangle per
# wiggle). fit_water_stripes never iterates boundary vertices -- every
# placement comes from a shapely polygon/line intersection against the
# real geometry -- so simplifying first is a pure perf/robustness
# trade against the TRUE shape, never something correctness depends
# on. Defaults OFF (0.0): real ponds' vertex counts are small enough
# that shapely's line intersection is fast regardless, so the more
# defensible default is probing the true, unsimplified polygon unless
# someone actually hits a real cost problem on an unusually
# vertex-dense polygon.
DEFAULT_WATER_STRIPE_TOLERANCE_M = 0.0

# Hard safety cap on stripes placed per walk direction (+v and -v are
# capped independently) -- purely defensive, so a pathological/
# degenerate polygon can't spin the outward walk indefinitely. By
# direct instruction, 100% coverage is the actual requirement and
# stripe count is otherwise unconstrained (a real, complex boundary
# legitimately needing many thin stripes is an accepted outcome, not a
# problem), so this is set generously and not expected to ever bind on
# a real pond.
DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE = 200

# Fudge-factor buffer (m) applied to the pond's OWN polygon before any
# fitting happens (fit_water_rectangle's center/rotation, then every
# stripe's own probe against the resulting shape) -- NOT the same thing
# as DEFAULT_WATER_STRIPE_OVERLAP_M's overshoot-past-the-boundary knob,
# which only bounds how far a stripe may extend past whatever polygon
# it's given. This instead grows the polygon itself first. Exists
# because the real OSM way and the real LIDAR-derived terrain don't
# always line up exactly -- confirmed in practice on a real course,
# stripes ending visibly short of the actual terrain intersection at
# the pond edge, consistent with the OSM trace sitting slightly inside
# where the ground actually starts sloping into the water. Defaults to
# 0.0 (byte-identical output to before this existed) since most ponds
# don't need it -- a per-course dial for the ones that do, not a
# blanket correction.
DEFAULT_WATER_STRIPE_BUFFER_M = 0.0


def _round(value: float) -> float:
    return round(float(value), _DECIMALS)


def _water_entry(
    cx: float, cz: float, width_m: float, depth_m: float, rotation_deg: float, level: float,
    flow_orientation: float = 0.0, flow_speed: float = 1.0,
) -> dict:
    """
    One userLayers.json "water" entry. width_m/depth_m are the real-world
    meters the plane should span along its local x/z axes; rotation_deg
    is a standard math CCW angle from +X describing the tile's real
    geometry.

    rotation_deg is NEGATED into PGA's rotation.y, which is a compass
    BEARING (0=north/+Z, 90=east/+X, clockwise) -- a mirror-image
    convention, not an additive offset. Confirmed against a real course:
    an un-negated write only visibly mis-renders a water body far from
    axis-aligned (a reflection error is invisible at 0/90/180/270, worst
    near 45) -- one ~46-degree pond rendered a clean 90 off, long axis
    on the wrong side. Cross-checked against terrain/cart_paths.py's
    atan2(dx, dz) stamp rotation (same bearing convention, args swapped
    vs. this module's atan2(dz, dx)), confirmed working in-game.

    WATER_BORDER_SCALE_CORRECTION inflates scale so the mesh's ~2.35%
    dead outer border (see that constant) doesn't eat into the visible
    surface.

    flow_orientation/flow_speed default to still water (a pond);
    build_stream_water_objects passes real flow values.
    """
    return {
        "surfaceCategory": WATER_SURFACE_CATEGORY,
        "position": {
            "x": _round(cx - GRID_ORIGIN_OFFSET),
            "y": _round(level),
            "z": _round(cz - GRID_ORIGIN_OFFSET),
        },
        "rotation": {"x": 0.0, "y": _round((-rotation_deg) % 360.0), "z": 0.0},
        "_orientation": 0.0,
        "scale": {
            "x": _round(width_m * WATER_BORDER_SCALE_CORRECTION / WATER_MESH_BASE_SIZE_M),
            "y": 1.0,
            "z": _round(depth_m * WATER_BORDER_SCALE_CORRECTION / WATER_MESH_BASE_SIZE_M),
        },
        "type": WATER_TYPE,
        "value": _round(level),
        "holeId": -1,
        "options": {"flowOrientation": _round(flow_orientation), "flowSpeed": _round(flow_speed)},
        "radius": 0.0,
        "orientation": 0.0,
    }


def _water_level_from_way_boundary(
    way_polygon, model: TerrainModel,
    spacing_m: float = DEFAULT_WATER_LEVEL_BOUNDARY_SPACING_M,
    safety_margin_m: float = DEFAULT_WATER_LEVEL_SAFETY_MARGIN_M,
) -> Optional[float]:
    """
    The water level (Y) that keeps the water plane close to the ACTUAL
    rendered terrain along `way_polygon`'s own OUTLINE, without
    floating over most of it -- see module docstring's "Water level".
    Deliberately NOT the minimum over the way's whole INTERIOR: a real
    pond's basin is typically deepest in the middle (real digging, or
    just natural bowl shape), so a whole-interior minimum would sink
    the plane far below the shoreline instead of just below it (module
    docstring point 4).

    Computed by sampling TerrainModel.evaluate_many at points spaced
    spacing_m apart along `way_polygon.boundary` (every rung of a
    MultiLineString/LinearRing walked in curve order), then taking the
    MEAN of those heights, minus safety_margin_m. A real water way's
    traced boundary is drawn at the actual shoreline, so the large
    majority of dense samples along it already cluster tightly around
    the true water height -- the mean uses that whole distribution
    directly instead of reaching for a low percentile or the strict
    minimum (both tried and rejected; see module docstring points 5
    and 6).

    `way_polygon` should be the raw OSM water way's own polygon
    geometry (`Feature.geometry`, before any rectangle/tile fitting) --
    NOT the fitted footprint actually rendered (the union of tiles/
    stripes), which is deliberately inflated past the way's own edge
    and so doesn't track the real, tagged shoreline as closely (module
    docstring point 6). `way_polygon` and `model` must already be in
    the same local [0, COURSE_SIZE_M] frame (neither is
    GRID_ORIGIN_OFFSET-shifted yet); `model` must be built from the
    ALREADY-height-normalized stamp list (see userLayers.py's
    normalize_stamp_heights) -- evaluating it, rather than reading any
    stamp's own .value, is what actually captures that normalization
    shift (see module docstring's "Water level" #1).

    None if the way has no boundary length to sample at all (degenerate
    geometry).
    """
    boundary = way_polygon.boundary
    lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    points: list[tuple[float, float]] = []
    for line in lines:
        length = line.length
        if length < 1e-9:
            continue
        n = max(2, int(math.ceil(length / spacing_m)))
        points.extend(line.interpolate(i / n, normalized=True).coords[0] for i in range(n))
    if not points:
        return None

    heights = model.evaluate_many(np.array(points))
    return float(heights.mean()) - safety_margin_m


def fit_water_rectangle(polygon) -> Optional[tuple[float, float, float, float, float]]:
    """
    (center_x, center_z, width, depth, rotation_degrees) for the
    minimum-area rotated rectangle enclosing `polygon`, expanded by
    WATER_RECT_MARGIN -- see module docstring's "Fit shape"/"Geometry".
    None if the geometry is too degenerate to fit one (e.g. collapses
    to a point or line). width/depth here are the real-world meters
    the plane should SPAN -- converting that into the mesh's own scale
    units (see WATER_MESH_BASE_SIZE_M) happens in build_water_objects,
    not here, so this function's output stays independently checkable
    against the source geometry in real-world units.

    Public (not module-private) so the GUI's live "Show objects"
    preview can call it directly against the current in-memory feature
    list -- same "pure helper, not a step function" reuse as
    object_clusters.pack_cluster_records/ingest.osm.build_height_mask
    (see PGA2k_gen_gui.py's _get_water_preview_rects) -- rather than
    duplicating this geometry fit or requiring a full write-water run
    just to see whether a fitted rectangle lines up with its pond.
    """
    rect = polygon.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return None
    coords = list(rect.exterior.coords[:-1])
    if len(coords) != 4:
        return None

    cx = sum(c[0] for c in coords) / 4.0
    cz = sum(c[1] for c in coords) / 4.0
    edge1 = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    edge2 = (coords[2][0] - coords[1][0], coords[2][1] - coords[1][1])
    width = math.hypot(*edge1) * WATER_RECT_MARGIN
    depth = math.hypot(*edge2) * WATER_RECT_MARGIN
    rotation_deg = math.degrees(math.atan2(edge1[1], edge1[0])) % 360.0
    return cx, cz, width, depth, rotation_deg


def _wt_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _wt_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _wt_scale(a, s):
    return (a[0] * s, a[1] * s)


def _wt_length(a):
    return math.hypot(a[0], a[1])


def _wt_normalize(a):
    ln = _wt_length(a)
    if ln < 1e-9:
        return (0.0, 0.0)
    return (a[0] / ln, a[1] / ln)


def _wt_rotate90(a, sign):
    return (-sign * a[1], sign * a[0])


def _wt_ring_edges(ring: list[tuple[float, float]]) -> list[tuple[tuple, tuple]]:
    n = len(ring)
    return [(ring[i], ring[(i + 1) % n]) for i in range(n)]


def _wt_ray_edges_intersection(origin, direction, edges, max_distance):
    """
    Nearest intersection of the ray (origin + t*direction, t > 0) with
    any segment in `edges` (a flat list of (p0, p1) pairs -- may span
    multiple rings), within max_distance. Returns t or None. Ported
    from terrain/contour_layers.py's _fl_ray_edges_intersection (itself
    lifted verbatim from fallline_fill_viz.py) -- duplicated locally
    rather than imported across the course_output/terrain package
    boundary for ~40 lines of dependency-free vector geometry.
    """
    best_t = None
    ox, oz = origin
    dx, dz = direction

    for p0, p1 in edges:
        ex, ez = p1[0] - p0[0], p1[1] - p0[1]

        denom = dx * ez - dz * ex
        if abs(denom) < 1e-9:
            continue

        qx, qz = p0[0] - ox, p0[1] - oz
        t = (qx * ez - qz * ex) / denom
        u = (qx * dz - qz * dx) / denom
        if t > 1e-6 and 0.0 <= u <= 1.0 and t <= max_distance:
            if best_t is None or t < best_t:
                best_t = t

    return best_t


def _water_tile_corners(
    cx: float, cz: float, width: float, depth: float, rotation_deg: float,
) -> list[tuple[float, float]]:
    """
    The 4 real-world corners of a (cx, cz, width, depth, rotation_deg)
    water-tile tuple -- same corner math PGA2k_gen_gui.py's
    _composite_objects_layer already uses to DRAW these tuples, reused
    here (not reimplemented) so the polygon fit_water_tiles reasons
    about during its dedup pass is exactly the shape that ends up on
    screen/in-game.
    """
    angle = math.radians(rotation_deg)
    width_dir = (math.cos(angle), math.sin(angle))
    depth_dir = (-math.sin(angle), math.cos(angle))
    half_w, half_d = width / 2.0, depth / 2.0
    return [
        (
            cx + sw * half_w * width_dir[0] + sd * half_d * depth_dir[0],
            cz + sw * half_w * width_dir[1] + sd * half_d * depth_dir[1],
        )
        for sw, sd in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def fit_water_tiles(
    polygon,
    tolerance_m: float = DEFAULT_WATER_TILE_TOLERANCE_M,
    min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    max_search_m: float = DEFAULT_WATER_TILE_MAX_SEARCH_M,
    width_samples: int = DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    redundancy_ratio: float = DEFAULT_WATER_TILE_REDUNDANCY_RATIO,
    overlap_m: float = DEFAULT_WATER_TILE_OVERLAP_M,
) -> Optional[list[tuple[float, float, float, float, float]]]:
    """
    Fill `polygon` with several smaller, possibly-overlapping rotated
    rectangles hugging its real boundary, instead of fit_water_rectangle's
    single minimum-rotated-rectangle -- see module docstring and
    PGA2k_gen.py's --multi-tile-water. Returns a list of (cx, cz, width,
    depth, rotation_deg) tuples in the EXACT SAME convention as
    fit_water_rectangle (rotation_deg is a standard math CCW angle from
    +X of the tile's "width" edge -- deliberately NOT terrain/
    contour_layers.py's atan2(interior_perp_x, interior_perp_z) Stamp-
    rotation convention, which is written straight to Stamp.rotation
    with no negation anywhere downstream; this tuple instead keeps
    flowing through build_water_objects' existing PGA-compass-bearing
    negation unchanged, same as fit_water_rectangle's output already
    does) so build_water_objects can treat both fitters identically.

    Never returns None/[] for anything fit_water_rectangle itself can
    handle -- always at least one tile for a valid, non-degenerate
    polygon. Algorithm (see module docstring's "Fit shape" and
    terrain/contour_layers.py's "rect" fill mode, the direct template
    this was adapted from -- see _fallline_edge_stamps there):

      1. A simple polygon (<=4 exterior vertices, no holes) is already
         near-rectangular -- skip straight to fit_water_rectangle,
         there's nothing a per-edge pass would improve.
      2. Simplify the boundary (Douglas-Peucker, tolerance_m) so real
         LIDAR/OSM wiggle (a modest pond can easily have 40+ vertices)
         collapses into a handful of long edges before per-edge
         placement -- unlike contour_layers.py's raster-traced masks,
         no rasterize/re-trace round trip is needed since this already
         starts from a real shapely polygon.
      3. One candidate rectangle per boundary edge (exterior CCW,
         holes CW -- shapely's orient() makes the interior-side sign a
         fixed +1/-1 per ring): width = the edge's own length EXTENDED
         by overlap_m at both ends (still centered on the edge's own
         midpoint, so this is a plain symmetric grow, not a shift) --
         without this, adjacent tiles from neighboring edges meet
         exactly corner-to-corner with zero margin, which measured as a
         visibly "inset" seam once real float rounding and the dedup
         pass's own candidate ordering are involved. A tweakable
         absolute overlap (not a multiplicative margin like
         WATER_RECT_MARGIN -- a % of a short edge is too small to
         matter) fixes that by direct instruction. depth = ray-cast
         from several points along the (un-extended) edge into the
         polygon interior to the opposite wall (floored at min_edge_m,
         capped at max_search_m, then WATER_RECT_MARGIN-expanded same
         as fit_water_rectangle) -- same technique as
         _fallline_edge_stamps, against these vector edges directly.
      4. Greedy area-descending dedup: keep a candidate only if it
         still contributes at least redundancy_ratio of its own area as
         genuinely new coverage against the shrinking "not yet covered"
         remainder (shapely .difference(), same remainder-tracking
         idiom as object_clusters.py's _subtract_circles) -- overlap
         with what's ALREADY placed is fine (the whole point), this
         only rejects a candidate that would add almost nothing new.
      5. If every candidate gets rejected (degenerate tolerance
         settings, or a polygon simple enough that one big rectangle
         already covers it about as well), fall back to
         fit_water_rectangle's result -- guarantees at least one tile.
    """
    if polygon.geom_type != "Polygon" or polygon.is_empty:
        return None

    exterior_coords = list(polygon.exterior.coords[:-1])
    if len(exterior_coords) <= 4 and not list(polygon.interiors):
        single = fit_water_rectangle(polygon)
        return [single] if single is not None else None

    simplified = polygon.simplify(tolerance_m, preserve_topology=True)
    if not simplified.is_valid:
        simplified = simplified.buffer(0)
    if simplified.is_empty or simplified.geom_type != "Polygon":
        single = fit_water_rectangle(polygon)
        return [single] if single is not None else None

    simplified = orient(simplified, sign=1.0)
    rings = [list(simplified.exterior.coords[:-1])] + [
        list(ring.coords[:-1]) for ring in simplified.interiors
    ]
    all_edges = [e for ring in rings for e in _wt_ring_edges(ring)]

    candidates: list[tuple[float, float, float, float, float]] = []
    for ring_index, ring in enumerate(rings):
        sign = 1 if ring_index == 0 else -1
        for p0, p1 in _wt_ring_edges(ring):
            seg = _wt_sub(p1, p0)
            width = _wt_length(seg)
            if width < 1e-9:
                continue  # degenerate zero-width segment

            direction = _wt_normalize(seg)
            interior_perp = _wt_rotate90(direction, sign)

            sample_ts = np.linspace(0.05, 0.95, width_samples)
            hits = []
            for t in sample_ts:
                sample_origin = _wt_add(p0, _wt_scale(seg, float(t)))
                d = _wt_ray_edges_intersection(sample_origin, interior_perp, all_edges, max_search_m)
                if d is not None:
                    hits.append(d)
            depth = max(min_edge_m, min(hits)) if hits else min_edge_m

            midpoint = _wt_scale(_wt_add(p0, p1), 0.5)
            center = _wt_add(midpoint, _wt_scale(interior_perp, depth / 2.0))
            rotation_deg = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
            # Extend width by overlap_m at BOTH ends, still centered on
            # the same midpoint -- gives this tile real overlap with
            # its neighbors along the boundary instead of an exact (or,
            # after rounding, slightly short) corner-to-corner meet.
            candidates.append((
                center[0], center[1],
                width + 2.0 * overlap_m, depth * WATER_RECT_MARGIN,
                rotation_deg,
            ))

    if not candidates:
        single = fit_water_rectangle(polygon)
        return [single] if single is not None else None

    candidates.sort(key=lambda c: c[2] * c[3], reverse=True)  # largest area first
    remaining = polygon  # overshoot accounting targets the REAL, un-simplified pond
    accepted: list[tuple[float, float, float, float, float]] = []
    for cx, cz, width, depth, rotation_deg in candidates:
        rect = Polygon(_water_tile_corners(cx, cz, width, depth, rotation_deg))
        if not rect.is_valid or rect.area < 1e-9:
            continue
        new_area = rect.intersection(remaining).area
        if new_area >= rect.area * redundancy_ratio:
            accepted.append((cx, cz, width, depth, rotation_deg))
            remaining = remaining.difference(rect)

    if not accepted:
        single = fit_water_rectangle(polygon)
        return [single] if single is not None else None
    return accepted


def _ws_probe_segment(polygon, origin, direction, half_length):
    """
    The extent of `polygon`'s intersection with a long line through
    `origin` along `direction` (unit vector), as (t_min, t_max) signed
    distances from `origin` along `direction` -- t=0 is `origin`
    itself. Used by fit_water_stripes to find where the pond's real
    boundary actually is, in any direction, from any interior point --
    a different access pattern than _wt_ray_edges_intersection (built
    for "ray from a point ON the boundary, into the interior"), so a
    live shapely polygon/line intersection is used directly here
    instead of adapting that helper.

    Robust to concave/multi-lobed polygons: constructs the FULL
    intersection (not a boundary-edge ray-cast), then -- when that
    intersection is a MultiLineString/GeometryCollection (a crescent or
    dogleg pond can have the probe line cross it more than once) --
    picks the ONE segment closest to `origin`, never the min/max across
    every fragment (which would silently span a real gap in the pond).

    None if no usable segment exists near `origin` at all (this probe
    position doesn't pass through the polygon here -- e.g. `origin`
    sits in a concave notch, or just outside the boundary).
    """
    ux, uz = direction
    ox, oz = origin
    p0 = (ox - ux * half_length, oz - uz * half_length)
    p1 = (ox + ux * half_length, oz + uz * half_length)
    inter = polygon.intersection(LineString([p0, p1]))
    if inter.is_empty:
        return None

    if inter.geom_type == "LineString":
        segments = [inter]
    elif inter.geom_type == "MultiLineString":
        segments = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        segments = [g for g in inter.geoms if g.geom_type == "LineString"]
    else:  # Point/MultiPoint -- a tangent touch, not a usable span
        return None
    if not segments:
        return None

    def _t_of(pt):
        return (pt[0] - ox) * ux + (pt[1] - oz) * uz

    best_range, best_dist = None, None
    for seg in segments:
        ts = [_t_of(c) for c in seg.coords]
        tmin, tmax = min(ts), max(ts)
        dist = 0.0 if tmin <= 1e-6 <= tmax else min(abs(tmin), abs(tmax))
        if best_dist is None or dist < best_dist:
            best_range, best_dist = (tmin, tmax), dist
    return best_range


# The deliberate overlap between ADJACENT stripes along the stacking
# axis -- NOT the true polygon boundary (see DEFAULT_WATER_STRIPE_
# OVERLAP_M's own comment: that one governs boundary overshoot only,
# by direct instruction). This seam only needs to guard against float-
# precision gaps at the stripe-to-stripe join, not express any real
# design tolerance, so a couple of cm is plenty and it isn't worth
# exposing as a tunable.
_WS_SEAM_OVERLAP_M = 0.02

# Safety cap on adaptive depth-halving per stripe attempt -- purely
# defensive (this file's usual "never loop forever" ethos), not
# expected to bind: by direct instruction, 100% coverage is the actual
# requirement and stripe count/aspect ratio is unconstrained, so this
# exists only to guarantee termination on a pathological polygon, not
# to cap how thin a stripe is allowed to get.
_WS_MAX_BISECTIONS = 24

# How far inside a candidate range's own [v0, v1] edges to probe when
# measuring overshoot (see _ws_fit_width_for_range) -- probing EXACTLY
# at v1 fails essentially always on a walk's first (largest) candidate,
# since v1 is very often the exact true-wall coordinate found by
# _ws_max_v_extent, where the polygon's true width is exactly zero (a
# tangent point). A tiny inward nudge keeps the measurement meaningful
# without landing on a degenerate point.
_WS_EDGE_PROBE_EPS_M = 0.01


def _ws_band_intersection_coords(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_u_length):
    """
    All (x, z) world coordinates of `polygon`'s intersection with the
    band [min(v0,v1), max(v0,v1)] (sign*v_dir distance from (cx,cz)) x
    [-half_u_length, +half_u_length] (u_dir) -- a real shapely polygon/
    polygon intersection, not discrete point sampling. Shared by
    _ws_band_u_extent/_ws_max_v_extent, which each just project these
    coordinates onto a different axis. None if the band doesn't
    intersect the polygon at all.
    """
    v_lo, v_hi = sign * min(v0, v1), sign * max(v0, v1)
    lo_base = _wt_add((cx, cz), _wt_scale(v_dir, v_lo))
    hi_base = _wt_add((cx, cz), _wt_scale(v_dir, v_hi))
    band = Polygon([
        _wt_add(lo_base, _wt_scale(u_dir, -half_u_length)),
        _wt_add(lo_base, _wt_scale(u_dir, half_u_length)),
        _wt_add(hi_base, _wt_scale(u_dir, half_u_length)),
        _wt_add(hi_base, _wt_scale(u_dir, -half_u_length)),
    ])
    if not band.is_valid:
        band = band.buffer(0)
    inter = polygon.intersection(band)
    if inter.is_empty:
        return None

    geoms = list(inter.geoms) if hasattr(inter, "geoms") else [inter]
    coords: list[tuple[float, float]] = []
    for g in geoms:
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            coords.extend(g.exterior.coords)
        elif g.geom_type == "LineString":
            coords.extend(g.coords)
        elif g.geom_type == "Point":
            coords.append((g.x, g.y))
    return coords or None


def _ws_band_u_extent(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_u_length):
    """
    (u_min, u_max) of polygon's TRUE footprint anywhere within the
    v-band [min(v0,v1), max(v0,v1)] (sign*v_dir distance from (cx,cz)),
    computed via a real shapely polygon/polygon intersection (see
    _ws_band_intersection_coords) -- not discrete point sampling along
    v. This replaced an earlier version that sampled only a handful of
    points across the band and sized width to the widest SAMPLE:
    confirmed on real ponds to leave real, measurable gaps whenever the
    true boundary bulges out somewhere BETWEEN two samples (common on
    LIDAR/OSM-derived boundaries, which are rarely smooth) -- that
    bulge was invisible to point sampling no matter how many samples
    were used, since it could always fall between them. A full band
    intersection has no such blind spot: it sees the polygon's ENTIRE
    true shape within the band, continuously.

    None if the band doesn't intersect the polygon at all.
    """
    coords = _ws_band_intersection_coords(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_u_length)
    if coords is None:
        return None
    us = [(px - cx) * u_dir[0] + (pz - cz) * u_dir[1] for px, pz in coords]
    return min(us), max(us)


def _ws_max_v_extent(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_u_length):
    """
    The TRUE farthest v (sign*v_dir distance from (cx,cz)) reached by
    any part of `polygon` within the band [min(v0,v1), max(v0,v1)] x
    [-half_u_length, +half_u_length] -- same real band-intersection
    technique as _ws_band_u_extent (axes swapped), used to robustly
    find how far the pond TRULY extends in the walk direction, across
    its FULL width, rather than along one single ray from one point.
    Confirmed on real ponds: a single ray can miss real pond area that
    curves away from it, silently leaving a gap at the far tip -- the
    exact same failure mode _ws_band_u_extent already fixed for the
    width dimension, here applied to the depth/walk dimension too
    (this is what determines where _ws_walk_direction's outward walk
    actually stops).

    None if the band doesn't intersect the polygon at all.
    """
    coords = _ws_band_intersection_coords(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_u_length)
    if coords is None:
        return None
    vs = [((px - cx) * v_dir[0] + (pz - cz) * v_dir[1]) * sign for px, pz in coords]
    return max(vs)


def _ws_fit_width_for_range(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_length):
    """
    Fit ONE stripe's width to the v-range [v0, v1] (sign*v_dir distance
    from (cx, cz)). Returns (width, u_center, max_overshoot):
      - width/u_center describe the SMALLEST rectangle (no padding)
        that fully contains the band's ENTIRE true footprint (see
        _ws_band_u_extent) -- guarantees no gap ANYWHERE in [v0, v1],
        not just at a finite set of sampled points.
      - max_overshoot is how far that rectangle's u-edges extend past
        the polygon's true extent specifically AT v0 and v1 (the
        band's own edges) -- what the caller checks against overlap_m
        to decide whether this v-range is thin enough to accept as one
        stripe, or needs to be bisected into thinner ones (see
        _ws_walk_direction). A band that bulges out in the middle
        relative to its own edges will show real overshoot here,
        correctly triggering a bisection to chase a tighter fit --
        while still never risking a gap, since width always covers the
        full band regardless of the bisection outcome.

    Only None if the band doesn't intersect the polygon at ALL (width
    itself, the part that guarantees coverage, is unmeasurable). The
    v0/v1 edge probes used for max_overshoot are a single ray each, and
    CAN individually fail right at a true polygon tip (numerically
    fragile exactly at a boundary) even when the band intersection
    above succeeds fine -- confirmed as a real bug on a real pond: an
    earlier version returned None whenever EITHER edge probe failed,
    which (at the floor, where the caller has nothing better to fall
    back on) discarded real, already-known-good coverage and left a
    genuine gap at the tip. A failed edge probe is instead treated as
    "unmeasured, don't penalize it" here -- coverage (from the band)
    is never in question either way.
    """
    band_extent = _ws_band_u_extent(polygon, cx, cz, u_dir, v_dir, sign, v0, v1, half_length)
    if band_extent is None:
        return None
    u_min_band, u_max_band = band_extent

    # Probe a hair INSIDE v0/v1, not exactly at them -- v1 in particular
    # is very often the exact true wall coordinate found by _ws_max_v_
    # extent (on a walk's very first, largest candidate), where the
    # polygon's true width is exactly zero (a tangent point) -- a
    # single-ray probe placed EXACTLY there fails essentially always,
    # not as a rare edge case (confirmed directly on a real pond).
    # Nudging inward keeps the measurement meaningfully close to the
    # true edge while landing somewhere the polygon actually has real
    # width to find.
    eps = min(_WS_EDGE_PROBE_EPS_M, (v1 - v0) / 4.0) if v1 > v0 else 0.0
    edge0 = _ws_probe_segment(polygon, _wt_add((cx, cz), _wt_scale(v_dir, sign * (v0 + eps))), u_dir, half_length)
    edge1 = _ws_probe_segment(polygon, _wt_add((cx, cz), _wt_scale(v_dir, sign * (v1 - eps))), u_dir, half_length)
    edges = [e for e in (edge0, edge1) if e is not None]
    if not edges:
        width = u_max_band - u_min_band
        u_center = (u_max_band + u_min_band) / 2.0
        return width, u_center, 0.0

    left_overshoot = max(e[0] for e in edges) - u_min_band
    right_overshoot = u_max_band - min(e[1] for e in edges)
    width = u_max_band - u_min_band
    u_center = (u_max_band + u_min_band) / 2.0
    return width, u_center, max(left_overshoot, right_overshoot)


def _ws_walk_direction(
    polygon, cx, cz, u_dir, v_dir, sign, near_t0,
    overlap_m, min_edge_m, half_length, max_stripes, rotation_deg,
):
    """
    One outward walk (sign=+1 or -1) along v_dir from near_t0 (a
    distance from (cx, cz) along sign*v_dir). Returns a list of
    (cx, cz, width, depth, rotation_deg) stripe tuples.

    How far the walk can go at all is itself found robustly:
    _ws_max_v_extent checks the polygon's TRUE remaining extent across
    its FULL width via a real band intersection, not a single ray from
    one point (an earlier version used a single ray -- confirmed on
    real ponds to sometimes stop short of the true tip, silently
    leaving a gap there, whenever the tip curves away from that one
    ray's direction).

    Each stripe's depth is then fit ADAPTIVELY: try the largest
    remaining candidate first (out to that true wall), check via
    _ws_fit_width_for_range whether a single width sized to that whole
    candidate range would overshoot the true boundary by more than
    overlap_m at either of its own edges -- if so, bisect the candidate
    depth and retry, repeating until it passes or hits min_edge_m. This
    guarantees 100% coverage (width always covers the band's ENTIRE
    true footprint, never an average or a sampled guess that could
    leave part of the true boundary uncovered) while keeping overshoot
    bounded, at the cost of possibly many thin stripes on a fast-
    changing boundary -- by direct instruction, no cap on stripe count
    or aspect ratio beyond the max_stripes safety net; a very long,
    thin stripe (or many of them) is an accepted, expected outcome
    here, not a problem to avoid. A pond with long, fairly-constant-
    width sections still gets a few large, efficient stripes there,
    since the largest-candidate-first approach only pays the thin-
    stripe cost where the boundary genuinely curves quickly.

    Consecutive stripes overlap by a small FIXED _WS_SEAM_OVERLAP_M
    (not overlap_m -- these are different concerns, see that constant's
    own comment), except the LAST stripe in this direction: its far
    edge actually touches the true polygon boundary, not another
    stripe, so it gets the same overlap_m tolerance the width dimension
    uses, for a consistent small margin there too.

    Never raises; stops (returns whatever it already has) the moment no
    more pond remains beyond the current near edge, the walk reaches
    the true wall, or max_stripes is hit (a defensive cap, not expected
    to bind on a real pond).
    """
    stripes: list[tuple[float, float, float, float, float]] = []
    near_t = near_t0

    for _ in range(max_stripes):
        true_far_t = _ws_max_v_extent(
            polygon, cx, cz, u_dir, v_dir, sign, near_t, near_t + 2.0 * half_length, half_length,
        )
        if true_far_t is None or true_far_t - near_t < 1e-6:
            break

        candidate_far = true_far_t
        fit = None
        for _bisect in range(_WS_MAX_BISECTIONS):
            trial_fit = _ws_fit_width_for_range(
                polygon, cx, cz, u_dir, v_dir, sign, near_t, candidate_far, half_length,
            )
            if trial_fit is not None and trial_fit[2] <= overlap_m:
                fit = trial_fit
                break
            if candidate_far - near_t <= min_edge_m:
                fit = trial_fit  # floor reached -- accept whatever this is (even excess overshoot)
                break
            candidate_far = near_t + (candidate_far - near_t) / 2.0

        if fit is None:
            break  # couldn't fit anything usable even at the floor -- stop this walk

        width, u_center_local, _overshoot = fit
        width = max(min_edge_m, width)

        reached_true_wall = (true_far_t - candidate_far) <= 1e-6
        if reached_true_wall:
            candidate_far = true_far_t + overlap_m  # true boundary edge -- small tolerance pad
        depth = max(min_edge_m, candidate_far - near_t)
        far_edge_t = near_t + depth
        center_t = (near_t + far_edge_t) / 2.0

        stripe_center = _wt_add(_wt_add((cx, cz), _wt_scale(v_dir, sign * center_t)),
                                 _wt_scale(u_dir, u_center_local))
        stripes.append((stripe_center[0], stripe_center[1], width, depth, rotation_deg))

        if reached_true_wall:
            break
        near_t = far_edge_t - _WS_SEAM_OVERLAP_M

    return stripes


def fit_water_stripes(
    polygon,
    overlap_m: float = DEFAULT_WATER_STRIPE_OVERLAP_M,
    min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    tolerance_m: float = DEFAULT_WATER_STRIPE_TOLERANCE_M,
    max_stripes_per_side: int = DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
    buffer_m: float = DEFAULT_WATER_STRIPE_BUFFER_M,
) -> Optional[list[tuple[float, float, float, float, float]]]:
    """
    An alternative to fit_water_tiles' per-edge-plus-dedup fill --
    starts from the pond's own center and walks outward in both
    directions, each stripe's own width/depth fit adaptively (see
    _ws_walk_direction) directly against the real boundary. Where
    fit_water_tiles can leave real, unpredictable gaps (its greedy
    redundancy-dedup pass can reject a candidate that was the only
    thing that would have covered some patch, on a sufficiently complex
    boundary), this algorithm never has a rejection step at all -- every
    stripe it places stays placed, and every stripe's width is sized to
    the WIDEST point it covers (never an average), so 100% coverage is
    structural, not a tuning outcome. Overshoot past the true boundary
    is bounded by overlap_m via adaptive depth-bisection, not avoided
    by capping stripe count or aspect ratio -- by direct instruction,
    coverage is the only hard requirement; a very long, thin stripe (or
    many of them) is an accepted outcome, not a problem.

    buffer_m (default DEFAULT_WATER_STRIPE_BUFFER_M, 0.0 -- byte-
    identical output when left at default) grows `polygon` by this many
    meters (shapely .buffer(), all directions) BEFORE anything else
    happens -- the pond's own center/rotation (from fit_water_rectangle)
    and every stripe's boundary probe all then run against the grown
    shape, not the original. A per-course fudge factor for when the OSM
    way and the real LIDAR-derived terrain don't quite agree on where
    the pond edge actually is, leaving stripes ending visibly short of
    where the ground really starts sloping into the water -- distinct
    from overlap_m, which only bounds how far a stripe may extend past
    whatever polygon it's given, not the polygon itself.

    Returns a list of (cx, cz, width, depth, rotation_deg) tuples, same
    convention as fit_water_rectangle/fit_water_tiles (rotation_deg a
    standard math CCW angle from +X) so build_water_objects can treat
    all three fitters identically. rotation_deg is FIXED across every
    stripe in one pond (taken from fit_water_rectangle's own minimum_
    rotated_rectangle -- only its rotation is used here, not its width/
    depth, since every stripe sizes itself independently against the
    real boundary). Never None/[] for anything fit_water_rectangle
    itself can fit -- falls back to [fit_water_rectangle(polygon)] on a
    degenerate/too-simple polygon or a center point that (per a very
    concave polygon) doesn't actually land inside it.

    Algorithm:
      1. cx, cz, _, _, rotation_deg = fit_water_rectangle(polygon) --
         only the rotation and the literal center point are used.
      2. u_dir/v_dir: the same "width"/"depth" basis convention
         fit_water_rectangle/_water_tile_corners already use, at the
         fixed rotation_deg.
      3. Walk outward independently in +v_dir and -v_dir (_ws_walk_
         direction), each starting from a hairline before the exact
         center (-_WS_SEAM_OVERLAP_M/2) so the innermost +stripe and
         -stripe overlap at the middle by that same small seam amount,
         same as every other internal seam -- no separate "seed" step
         needed; the first stripe from each walk IS the center stripe
         for that half.
    """
    if buffer_m:
        polygon = polygon.buffer(buffer_m)

    if polygon.geom_type != "Polygon" or polygon.is_empty:
        return None

    exterior_coords = list(polygon.exterior.coords[:-1])
    if len(exterior_coords) <= 4 and not list(polygon.interiors):
        single = fit_water_rectangle(polygon)
        return [single] if single is not None else None

    outer = fit_water_rectangle(polygon)
    if outer is None:
        return None
    cx, cz, _outer_width, _outer_depth, rotation_deg = outer

    if not polygon.contains(Point(cx, cz)):
        return [outer]

    probe_polygon = polygon
    if tolerance_m > 0:
        simplified = polygon.simplify(tolerance_m, preserve_topology=True)
        if not simplified.is_valid:
            simplified = simplified.buffer(0)
        if not simplified.is_empty and simplified.geom_type == "Polygon":
            probe_polygon = simplified
        # else: simplification here is a pure perf/noise-reduction aid
        # (see DEFAULT_WATER_STRIPE_TOLERANCE_M's own comment), never
        # structural -- silently keep probing the real polygon rather
        # than falling back to a single tile just because simplify
        # itself failed.

    angle = math.radians(rotation_deg)
    u_dir = (math.cos(angle), math.sin(angle))
    v_dir = (-math.sin(angle), math.cos(angle))

    minx, miny, maxx, maxy = probe_polygon.bounds
    half_length = math.hypot(maxx - minx, maxy - miny)
    if half_length < 1e-6:
        return [outer]

    plus = _ws_walk_direction(
        probe_polygon, cx, cz, u_dir, v_dir, +1, -_WS_SEAM_OVERLAP_M / 2.0,
        overlap_m, min_edge_m, half_length, max_stripes_per_side, rotation_deg,
    )
    minus = _ws_walk_direction(
        probe_polygon, cx, cz, u_dir, v_dir, -1, -_WS_SEAM_OVERLAP_M / 2.0,
        overlap_m, min_edge_m, half_length, max_stripes_per_side, rotation_deg,
    )

    stripes = plus + minus
    return stripes if stripes else [outer]


def build_water_objects(
    water_features: Sequence[Feature], stamps: Sequence[Stamp], printf=print,
    multi_tile_water: bool = False,
    water_fill_mode: str = "edge",
    water_tile_tolerance_m: float = DEFAULT_WATER_TILE_TOLERANCE_M,
    water_tile_min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    water_tile_max_search_m: float = DEFAULT_WATER_TILE_MAX_SEARCH_M,
    water_tile_width_samples: int = DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    water_tile_redundancy_ratio: float = DEFAULT_WATER_TILE_REDUNDANCY_RATIO,
    water_tile_overlap_m: float = DEFAULT_WATER_TILE_OVERLAP_M,
    water_stripe_overlap_m: float = DEFAULT_WATER_STRIPE_OVERLAP_M,
    water_stripe_min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    water_stripe_tolerance_m: float = DEFAULT_WATER_STRIPE_TOLERANCE_M,
    water_stripe_max_stripes_per_side: int = DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
    water_stripe_buffer_m: float = DEFAULT_WATER_STRIPE_BUFFER_M,
) -> list[dict]:
    """
    One or more water entries per "water" Feature with Polygon geometry
    (see module docstring) -- water_features must already be cropped to
    the course (see PGA2k_gen.py's _crop_features_to_course); `stamps`
    must be the FULLY height-normalized list (see userLayers.py's
    normalize_stamp_heights -- this must run BEFORE calling this
    function, not after) and in the same local [0, COURSE_SIZE_M] frame
    as water_features' geometry. GRID_ORIGIN_OFFSET is applied here, at
    the point of writing, same as every other writer in this project.

    multi_tile_water (default False, i.e. today's behavior, byte-
    identical output) swaps fit_water_rectangle's single minimum-
    rotated-rectangle per pond for a multi-tile fill -- water_fill_mode
    selects WHICH ONE, and is only consulted when multi_tile_water is
    True: "edge" (default, matches this flag's original behavior byte-
    for-byte) is fit_water_tiles' per-boundary-edge-plus-dedup fill;
    "stripe" is fit_water_stripes' seed-from-center-and-walk-outward
    fill, which trades away edge-fill's tighter worst-case overshoot
    for structurally-near-impossible gaps (no rejection step at all) --
    see that function's docstring for when to prefer it. Every tile
    belonging to the SAME pond Feature shares the SAME water level
    (computed once per Feature, exactly as before -- it was already
    polygon-level, not tile-level) and the SAME options.flowOrientation/
    flowSpeed (both are absolute compass values, unaffected by any
    individual tile's own rotation -- confirmed by direct instruction,
    not derived). The water_tile_*/water_stripe_* keyword args are only
    consulted for their own respective fill mode.

    Skips, with a printed reason (not silently and not an error), any
    water feature that isn't a Polygon, that no rectangle/tile set can
    be fit to, or that contains no stamp centers to determine a water
    level from.
    """
    if water_fill_mode not in ("edge", "stripe"):
        raise ValueError(f"water_fill_mode must be 'edge' or 'stripe', got {water_fill_mode!r}")

    model = TerrainModel(stamps)
    entries = []
    ponds_built = 0
    skipped = 0
    for f in water_features:
        if f.kind != "water":
            continue
        if f.geometry.geom_type != "Polygon":
            printf(f"  Skipping non-polygon water feature (geom_type={f.geometry.geom_type}) -- "
                   "only filled water bodies get a water object, not centerlines.")
            skipped += 1
            continue

        if multi_tile_water and water_fill_mode == "stripe":
            fits = fit_water_stripes(
                f.geometry, overlap_m=water_stripe_overlap_m, min_edge_m=water_stripe_min_edge_m,
                tolerance_m=water_stripe_tolerance_m, max_stripes_per_side=water_stripe_max_stripes_per_side,
                buffer_m=water_stripe_buffer_m,
            )
        elif multi_tile_water:
            fits = fit_water_tiles(
                f.geometry, tolerance_m=water_tile_tolerance_m, min_edge_m=water_tile_min_edge_m,
                max_search_m=water_tile_max_search_m, width_samples=water_tile_width_samples,
                redundancy_ratio=water_tile_redundancy_ratio, overlap_m=water_tile_overlap_m,
            )
        else:
            single = fit_water_rectangle(f.geometry)
            fits = [single] if single is not None else None

        if not fits:
            printf("  Skipping a water feature -- couldn't fit any rectangle(s) to its geometry "
                   "(likely degenerate/too small).")
            skipped += 1
            continue

        level = _water_level_from_way_boundary(f.geometry, model)
        if level is None:
            cx0, cz0 = fits[0][0], fits[0][1]
            printf(f"  Skipping a water feature near ({cx0:.0f}, {cz0:.0f}) -- its way geometry "
                   "is too degenerate to render a water level from.")
            skipped += 1
            continue

        ponds_built += 1
        for cx, cz, width_m, depth_m, rotation_deg in fits:
            entries.append(_water_entry(cx, cz, width_m, depth_m, rotation_deg, level))

    if multi_tile_water:
        printf(f"  {ponds_built} water pond(s) -> {len(entries)} water tile object(s) built" +
               (f", {skipped} skipped" if skipped else ""))
    else:
        printf(f"  {len(entries)} water object(s) built" + (f", {skipped} skipped" if skipped else ""))
    return entries


def build_stream_water_objects(
    stream_records: Sequence[dict], height_shift_m: float = 0.0,
    *,
    bed_sampler=None,
    water_fill_depth_m: float | None = None,
    water_base_width_m: float | None = None,
    water_widen_per_depth: float | None = None,
    water_widen_per_descent: float | None = None,
    level_margin_m: float | None = None,
    printf=print,
) -> list[dict]:
    """
    Flowing type-72 water tiles along every stream centerline (see
    terrain/streams.py's build_stream_records / streams.json). These are
    the SAME objects as pond planes -- perfectly horizontal, carrying a
    real elevation -- so a descending stream becomes a CHAIN of tiles,
    one per shallow elevation band. terrain.streams.stream_water_tiles is
    the shared geometry helper (each tile drops <= STREAM_WATER_TILE_DROP_M
    of bed; level = band upstream bed + STREAM_WATER_FILL_DEPTH_M; width
    grows with the tile's own deep-end depth AND total descent from the
    source, because a horizontal plane further below the original grade
    spreads wider before the banks clip it). build_stream_records reuses
    the same helper to hang a waterfall + splash on each tile seam.

    `bed_sampler(points (N,2)) -> heights (N,)` fits each tile's level to
    the ACTUAL carved terrain (percentile of centerline samples minus
    level_margin_m), exactly like a pond -- step_write_water passes the
    same already-normalized TerrainModel.evaluate_many the pond fit uses,
    with height_shift_m=0. Without a sampler, tile level falls back to
    streams.json's pre-normalization bed_h + STREAM_WATER_FILL_DEPTH_M and
    height_shift_m (project.json's output_height_shift_m, persisted by
    write-terrain) lifts it into the normalized frame. Either way, run
    write-terrain before write-water.

    Kept a separate builder (not merged into build_water_objects) on
    purpose: streams stay their own collection so a later v2025 target
    can format streams.json as elevation-changing water splines instead.

    rotation.y is the flow's compass bearing; scale.z (long axis) runs
    along the flow (see _water_entry, rotation_deg = -bearing).
    options.flowOrientation is that bearing + 180 (PGA treats it as a
    "flow-from" heading -- verified in-game). options.flowSpeed is
    STREAM_WATER_FLOW_SPEED (50 is the engine max).
    """
    from terrain.streams import STREAM_WATER_FLOW_SPEED, stream_water_tiles

    # Only forward the overrides that were actually given -- omitted ones
    # fall through to stream_water_tiles' own module-constant defaults, so
    # a bare call stays identical to before.
    tile_kwargs = {
        k: v for k, v in {
            "bed_sampler": bed_sampler,
            "fill_depth_m": water_fill_depth_m,
            "base_width_m": water_base_width_m,
            "widen_per_depth": water_widen_per_depth,
            "widen_per_descent": water_widen_per_descent,
            "level_margin_m": level_margin_m,
        }.items() if v is not None
    }

    entries: list[dict] = []
    tiles = 0
    for record in stream_records:
        pearls = [
            (float(p[0]), float(p[1]), float(p[3]))  # (x, z, bed_h); p = [x, z, rot, bed_h]
            for p in record.get("pearls", []) if len(p) >= 4
        ]
        if len(pearls) < 2:
            continue
        source_bed = pearls[0][2]

        for tile in stream_water_tiles(pearls, source_bed, **tile_kwargs):
            # _water_entry mirrors rotation_deg into a compass bearing
            # (rotation.y = -rotation_deg % 360), so pre-negate the tile
            # bearing to land on it exactly; long axis (scale.z) then runs
            # along the flow.
            entries.append(_water_entry(
                tile.cx, tile.cz,
                width_m=tile.width_m, depth_m=tile.rendered_length,
                rotation_deg=(-tile.bearing) % 360.0,
                level=tile.level + height_shift_m,
                flow_orientation=tile.flow_orientation,
                flow_speed=STREAM_WATER_FLOW_SPEED,
            ))
            tiles += 1

    printf(f"  {tiles} stream water tile(s) built across {len(stream_records)} stream(s)")
    return entries
