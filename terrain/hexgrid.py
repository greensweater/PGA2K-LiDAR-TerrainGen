"""
terrain/hexgrid.py

Generates the initial coarse stamp layout: a flat-top hexagonal
(triangular) lattice of overlapping stamp positions covering the
playable course.

Lattice geometry (hex circle-packing spacing):
    col_spacing (width)  = 2 * pitch   -- same-row neighbor spacing
    row_spacing (height) = sqrt(3) * pitch -- row-to-row spacing,
                                              alternate rows offset by pitch

HEX_LATTICE_PITCH_M = 50 m (doubled from an earlier 100 m pass) is
fixed by the corner-alignment requirement: col_spacing (2 * pitch =
100 m) must divide the 2000 m course into exactly 20 equal intervals,
giving 21 evenly spaced centers per row that land exactly on the left
and right corners. Row spacing at that pitch (sqrt(3) * 50 ~= 86.6 m)
doesn't divide 2000 m evenly -- no pitch makes both spacings hit exact
corners simultaneously, since that requires sqrt(3) to be rational --
so rows are centered vertically instead, leaving an equal margin above
and below.

Stamp radius, under pull-toward-value semantics (see terrain_model.py):
there's no additive-overshoot concern here -- a lerp can never push
terrain past a stamp's own value, so unlike an additive model, more
overlap is basically free (it just means smoother blending between
neighboring targets, not risk of a ridge). What actually matters is
coverage: a point untouched by any stamp just stays at 0, unmoved --
a real, literal hole, not a shape-blending nuance. The worst-covered
point in the lattice is the "deep hole" at the center of each
triangular gap between three mutual nearest-neighbor stamps, at
distance 2*pitch/sqrt(3) from each. Setting stamp radius = 2 * pitch
(each stamp reaches exactly to its nearest neighbors' centers) gets
that point pulled by all three overlapping stamps to ~97.6% combined
weight (1 - (1-w)^3, compounding each stamp's sequential pull) --
solid coverage without needing extra filler stamps, and this ratio is
scale-invariant (2*pitch/sqrt(3) / (2*pitch) = 1/sqrt(3) regardless of
pitch), so it holds at any resolution, not just the original 100 m one.

Edge coverage (HEX_BLEED_M): even rows reach exactly to bounds.min_x
and bounds.max_x by construction, but odd (offset) rows are shifted
inward by `pitch` on both ends -- inherent to proper hex tiling, not a
bug -- leaving a strip near the east/west edges covered only by
vertically-neighboring stamps rather than a same-row one too. Rather
than guess at a margin, this was checked numerically: scanning the
whole play area for its true worst-covered point (not just the interior
deep holes) found it sitting exactly at an odd row's missing corner,
at ~89% combined pull weight -- weaker than interior coverage, but not
a literal hole. HEX_BLEED_M = radius / 2 (= pitch) allows exactly one
extra stamp per side on odd rows only (verified: adds exactly 24
stamps at this pitch, all outside [0, 2000]), restoring the same
coverage odd rows would have gotten from an uncropped, infinite hex
lattice. Bleed is horizontal (east/west) only -- vertical centering
doesn't have the same alternating-row asymmetry, so top/bottom edges
don't need it.

Stamps here carry a placeholder value=0.0 -- this module only decides
*where* stamps go and how big they are, never what height they pull
toward. That's terrain height-fitting's job, against real LIDAR data,
and it has to run before these stamps mean anything. Bled stamps
(centered outside [0, 2000]) need LIDAR coverage slightly beyond the
course crop to fit well -- see the note in PGA2k_gen.py's
generate-terrain step.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from terrain.bounding_box import BoundingBox
from terrain.stamp import Stamp

HEX_LATTICE_PITCH_M = 50.0             # doubled resolution; derived from 2*pitch * 40 == course width
HEX_STAMP_RADIUS_M = 2.0 * HEX_LATTICE_PITCH_M  # reaches nearest-neighbor centers
HEX_BLEED_M = HEX_STAMP_RADIUS_M / 2.0  # = pitch; closes the offset-row edge gap, see module docstring
DEFAULT_BRUSH = 9
PLACEHOLDER_VALUE = 0.0                # overwritten by height fitting

_EPS = 1e-6


def generate_hex_grid(
    bounds: BoundingBox,
    pitch: float = HEX_LATTICE_PITCH_M,
    stamp_radius: float = HEX_STAMP_RADIUS_M,
    brush: int = DEFAULT_BRUSH,
    bleed: float = HEX_BLEED_M,
) -> list[Stamp]:
    """
    Generate a flat-top hexagonal (triangular) lattice of Stamps covering
    `bounds`, with placeholder value=0.0 (see module docstring).

    Columns are anchored to bounds.min_x so a pitch that evenly divides
    the width lands exactly on both left and right corners. Rows are
    centered vertically within bounds, since row spacing generally
    doesn't divide the height evenly (see module docstring).

    Offset (odd) rows are shifted inward by `pitch` on both ends
    relative to even rows -- inherent to proper hex tiling, not a bug --
    which leaves a strip near the east/west edges covered only by
    vertically-neighboring stamps rather than a same-row one. `bleed`
    (default pitch, i.e. radius/2) allows one extra stamp per side on
    odd rows only, centered outside [min_x, max_x], so its radius still
    reaches the true edge without changing any other row's positions.
    Bleed is horizontal (east/west) only -- vertical centering doesn't
    have the same alternating-row asymmetry, so top/bottom edges don't
    need it.
    """
    col_spacing = 2.0 * pitch
    row_spacing = math.sqrt(3.0) * pitch

    width = bounds.max_x - bounds.min_x
    height = bounds.max_z - bounds.min_z

    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate bounds for hex grid: {bounds}")

    # Floor (not round) so the row span never exceeds the available height.
    n_rows = int(math.floor(height / row_spacing)) + 1
    total_row_span = (n_rows - 1) * row_spacing
    z_start = bounds.min_z + (height - total_row_span) / 2.0

    stamps: list[Stamp] = []
    for row in range(n_rows):
        z = z_start + row * row_spacing
        x_offset = pitch if (row % 2 == 1) else 0.0
        first_x = bounds.min_x + x_offset

        # Walk backward from the row's natural first point, prepending
        # one extra stamp per step as long as it stays within `bleed`
        # of the true boundary. For even rows (first_x == min_x) this
        # never fires, since one step back is a full col_spacing past
        # the bleed allowance -- only odd rows, whose first point is
        # already `pitch` in from the edge, pick up an extra point here.
        x = first_x
        while x - col_spacing >= bounds.min_x - bleed - _EPS:
            x -= col_spacing

        while x <= bounds.max_x + bleed + _EPS:
            stamps.append(
                Stamp(x=x, z=z, radius=stamp_radius, value=PLACEHOLDER_VALUE, brush=brush)
            )
            x += col_spacing

    return stamps


def find_deep_holes(
    stamps: list[Stamp],
    pitch: float = HEX_LATTICE_PITCH_M,
    tolerance: float = 0.02,
) -> list[tuple[float, float]]:
    """
    Find the lattice's deep-hole points: centroids of every equilateral
    triangle formed by three mutual nearest-neighbor stamp centers
    (neighbor spacing = 2 * pitch). Each is the worst-covered point in
    the coarse layout -- useful as a diagnostic (e.g. flagging where
    adaptive refinement should look first), not needed to place fillers
    now that stamp radius alone gets these points solid coverage (see
    module docstring).
    """
    positions = np.array([[s.x, s.z] for s in stamps])
    tree = cKDTree(positions)
    neighbor_dist = 2.0 * pitch

    holes: set[tuple[float, float]] = set()
    for i, p in enumerate(positions):
        candidate_idx = tree.query_ball_point(p, neighbor_dist * (1.0 + tolerance))
        neighbors = [
            j for j in candidate_idx
            if j != i
            and abs(np.linalg.norm(positions[j] - p) - neighbor_dist) < neighbor_dist * tolerance
        ]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                j, k = neighbors[a], neighbors[b]
                d_jk = np.linalg.norm(positions[j] - positions[k])
                if abs(d_jk - neighbor_dist) < neighbor_dist * tolerance:
                    centroid = (p + positions[j] + positions[k]) / 3.0
                    holes.add((round(float(centroid[0]), 3), round(float(centroid[1]), 3)))

    return sorted(holes)
