"""
terrain/hexgrid.py

Generates the initial coarse stamp layout: a flat-top hexagonal
(triangular) lattice of overlapping stamp positions covering the
playable course.

Lattice geometry (hex circle-packing spacing):
    col_spacing (width)  = 2 * pitch   -- same-row neighbor spacing
    row_spacing (height) = sqrt(3) * pitch -- row-to-row spacing,
                                              alternate rows offset by pitch

HEX_LATTICE_PITCH_M = 100 m is fixed by the corner-alignment requirement:
col_spacing (2 * pitch = 200 m) must divide the 2000 m course into
exactly 10 equal intervals, giving 11 evenly spaced centers per row that
land exactly on the left and right corners. Row spacing at that pitch
(sqrt(3) * 100 ~= 173.2 m) doesn't divide 2000 m evenly -- no pitch
makes both spacings hit exact corners simultaneously (that requires
sqrt(3) to be rational) -- so rows are centered vertically instead,
leaving an equal margin above and below.

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
solid coverage without needing extra filler stamps.

Stamps here carry a placeholder value=0.0 -- this module only decides
*where* stamps go and how big they are, never what height they pull
toward. That's terrain height-fitting's job, against real LIDAR data,
and it has to run before these stamps mean anything.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from terrain.bounding_box import BoundingBox
from terrain.stamp import Stamp

HEX_LATTICE_PITCH_M = 50.0            # derived from 2*pitch * 10 == course width
HEX_STAMP_RADIUS_M = 2.0 * HEX_LATTICE_PITCH_M  # reaches nearest-neighbor centers
DEFAULT_BRUSH = 9
PLACEHOLDER_VALUE = 0.0                # overwritten by height fitting

_EPS = 1e-6


def generate_hex_grid(
    bounds: BoundingBox,
    pitch: float = HEX_LATTICE_PITCH_M,
    stamp_radius: float = HEX_STAMP_RADIUS_M,
    brush: int = DEFAULT_BRUSH,
) -> list[Stamp]:
    """
    Generate a flat-top hexagonal (triangular) lattice of Stamps covering
    `bounds`, with placeholder value=0.0 (see module docstring).

    Columns are anchored to bounds.min_x so a pitch that evenly divides
    the width lands exactly on both left and right corners. Rows are
    centered vertically within bounds, since row spacing generally
    doesn't divide the height evenly (see module docstring).
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

        x = bounds.min_x + x_offset
        while x <= bounds.max_x + _EPS:
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
