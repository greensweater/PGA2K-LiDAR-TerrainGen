"""
terrain/hexgrid.py

Generates the initial coarse stamp layout:
  1. A flat-top hexagonal (triangular) lattice of overlapping stamps.
  2. Small filler stamps at the lattice's leftover triangular gaps.

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

The lattice pitch and the actual Stamp radius are deliberately separate
numbers. With col_spacing = 2 * pitch, same-row neighbors at
radius = pitch are exactly tangent, never overlapping -- that's
structural, not something pitch can fix. Since we're packing circles
(not hexagons), and the circles are meant to overlap, HEX_STAMP_RADIUS_M
is solved for instead of just reusing the pitch: it's the radius at
which two overlapping Type 9 falloffs (the flattest of the four
profiles) sum back up to exactly the flat plateau height at their
shared midpoint -- the "ridge limit". Any larger and the seam becomes
a visible ridge above the surrounding plateau; any smaller and it's a
valley below it.

Even at the ridge-limit radius, the lattice's deep holes -- the points
equidistant from three mutual nearest-neighbor stamps, at distance
2*pitch/sqrt(3) from each -- are still under-covered (three overlapping
falloffs there sum to ~75% of plateau, not 100%). generate_hole_fillers()
adds one small Type 10 (or 54) stamp per hole, sized from the model's
own predicted deficit at that point, since those brushes' cosine-like
falloff (no flat plateau) suits patching a localized dip better than
Type 8/9's plateau shape would.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES, REFERENCE_AMPLITUDE_M
from terrain.stamp import Stamp
from terrain.terrain_kernel import TerrainKernel
from terrain.terrain_model import TerrainModel

HEX_LATTICE_PITCH_M = 100.0    # r: derived from 2r * 10 == course width
DEFAULT_BRUSH = 9
DEFAULT_FILLER_BRUSH = 10
DEFAULT_AMPLITUDE = REFERENCE_AMPLITUDE_M

_EPS = 1e-6


def solve_ridge_radius(
    pitch: float = HEX_LATTICE_PITCH_M,
    brush: int = DEFAULT_BRUSH,
) -> float:
    """
    Solve for the stamp radius at which two nearest-neighbor stamps'
    falloffs sum to exactly the flat plateau height at their shared
    midpoint (distance = pitch from each center, since neighbor spacing
    is 2 * pitch).
    """
    profile = BRUSH_PROFILES[brush]
    kernel = TerrainKernel(profile)
    plateau = profile.samples[0, 1]  # normalized center height

    def ridge_gap(radius: float) -> float:
        return 2.0 * kernel.sample(pitch / radius) - plateau

    # Just above `pitch` the midpoint sits right at the stamp edge (r->1,
    # height->0); a few pitches out it's deep in the flat plateau on both
    # sides (well above the ridge limit) -- root lies in between.
    return brentq(ridge_gap, pitch + 1e-6, pitch * 10.0)


HEX_STAMP_RADIUS_M = solve_ridge_radius()


def generate_hex_grid(
    bounds: BoundingBox,
    pitch: float = HEX_LATTICE_PITCH_M,
    stamp_radius: float = HEX_STAMP_RADIUS_M,
    brush: int = DEFAULT_BRUSH,
    amplitude: float = DEFAULT_AMPLITUDE,
) -> list[Stamp]:
    """
    Generate a flat-top hexagonal (triangular) lattice of Stamps covering
    `bounds`.

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
                Stamp(x=x, z=z, radius=stamp_radius, amplitude=amplitude, brush=brush)
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
    (neighbor spacing = 2 * pitch). Each is the point farthest from any
    stamp center within its triangle -- the worst-covered spot in the
    coarse layout.
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


def generate_hole_fillers(
    stamps: list[Stamp],
    pitch: float = HEX_LATTICE_PITCH_M,
    filler_brush: int = DEFAULT_FILLER_BRUSH,
    filler_radius: Optional[float] = None,
    target_height: Optional[float] = None,
) -> list[Stamp]:
    """
    Add one small filler stamp at each deep-hole gap, sized so its own
    center contribution closes the model's predicted deficit there.

    filler_radius defaults to the hole-to-neighbor distance
    (2 * pitch / sqrt(3)), reaching roughly to the three surrounding
    stamp centers without extending much further. target_height
    defaults to the Type 9 plateau (the reference "fully covered"
    height the rest of the layout is tuned to).
    """
    if filler_radius is None:
        filler_radius = 2.0 * pitch / math.sqrt(3.0)

    if target_height is None:
        target_height = BRUSH_PROFILES[DEFAULT_BRUSH].samples[0, 1] * REFERENCE_AMPLITUDE_M

    filler_center_normalized = BRUSH_PROFILES[filler_brush].samples[0, 1]
    model = TerrainModel(stamps)

    fillers: list[Stamp] = []
    for hx, hz in find_deep_holes(stamps, pitch):
        deficit = target_height - model.evaluate(hx, hz)
        if deficit <= 0.0:
            continue  # already at or above the target; nothing to add
        amplitude = deficit / filler_center_normalized
        fillers.append(
            Stamp(x=hx, z=hz, radius=filler_radius, amplitude=amplitude, brush=filler_brush)
        )
    return fillers
