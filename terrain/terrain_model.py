"""
terrain/terrain_model.py

TerrainModel: predicts terrain height at any (x, z) by replaying every
affecting Stamp's "flatten" pull, in placement order, starting from a
flat 0 height grid.

This is the core function everything else depends on -- the optimizer,
adaptive refinement, and the writer all read predicted height through
evaluate() / evaluate_many(), never by inspecting stamps directly (see
"Terrain Evaluation" in the architecture doc).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES, SHAPE_SQUARE
from terrain.stamp import TOOL_RAISE, Stamp
from terrain.terrain_kernel import TerrainKernel


class TerrainModel:
    """
    Evaluates predicted terrain height by folding every affecting
    Stamp's update over a starting height of 0. Each stamp's own
    `tool` selects which update rule applies (see terrain/stamp.py):

        flatten (tool=0): new_height = old_height + (stamp.value - old_height) * weight
        raise   (tool=1): new_height = old_height + stamp.value * weight

    Flatten is a lerp toward stamp.value (an absolute height), so a
    single stamp can never push terrain past its own value. Raise adds
    a delta scaled by the brush weight, preserving whatever relief
    already exists rather than overriding it with one flat target --
    better suited to correcting a uniform bias over an area that's
    already roughly the right shape (see terrain/adaptive_refine.py).

    Either way the result depends on the order stamps are applied in
    -- two stamps overlapping at different values give a different
    blended result depending on which one is folded in last. That's
    fine as long as this model replays stamps in the same order the
    in-game renderer will (list order, i.e. placement/generation
    order); the order itself doesn't need to be canonical, just
    consistent between the two.

    Terrain noise (terrainNoise.json) is a separate, always-on in-game
    layer per the architecture doc's "Terrain Noise" section and is
    intentionally not modeled here -- evaluate() predicts the
    stamp-driven base terrain only.
    """

    def __init__(self, stamps: Sequence[Stamp]):
        self.stamps: list[Stamp] = list(stamps)

        if self.stamps:
            centers = np.array([[s.x, s.z] for s in self.stamps])
            self._tree: Optional[cKDTree] = cKDTree(centers)
            # A square stamp's corners sit radius*sqrt(2) away in
            # Euclidean terms, even though its own (Chebyshev) radius
            # is just `radius` -- the KD-tree query below is always
            # Euclidean (that's what cKDTree does), so it needs a
            # pruning radius generous enough to still catch a point
            # near a square stamp's corner, or that stamp would never
            # even be considered as a candidate for such a point.
            self._max_radius = max(self._euclidean_reach(s) for s in self.stamps)
        else:
            self._tree = None
            self._max_radius = 0.0

        # One TerrainKernel per brush type actually in use, built once
        # rather than per-stamp or per-evaluate-call.
        self._kernels: dict[int, TerrainKernel] = {}
        for stamp in self.stamps:
            if stamp.brush not in self._kernels:
                if stamp.brush not in BRUSH_PROFILES:
                    raise ValueError(
                        f"No BrushProfile registered for brush type {stamp.brush}"
                    )
                self._kernels[stamp.brush] = TerrainKernel(BRUSH_PROFILES[stamp.brush])

    @staticmethod
    def _euclidean_reach(stamp: Stamp) -> float:
        """Furthest Euclidean distance from center this stamp can affect -- radius for circular, radius*sqrt(2) for square (its corners)."""
        profile = BRUSH_PROFILES.get(stamp.brush)
        if profile is not None and profile.shape == SHAPE_SQUARE:
            return stamp.radius * math.sqrt(2.0)
        return stamp.radius

    def _affecting_stamp_indices(self, x: float, z: float) -> np.ndarray:
        """
        Indices of stamps whose radius reaches (x, z), sorted ascending.

        Ascending index order recovers original list order (the
        placement/application order), which matters here since pulls
        must be replayed in that order, not summed unordered.
        """
        if self._tree is None:
            return np.empty(0, dtype=np.int64)
        idx = np.asarray(
            self._tree.query_ball_point([x, z], self._max_radius), dtype=np.int64
        )
        idx.sort()
        return idx

    def evaluate(self, x: float, z: float, start_height: float = 0.0) -> float:
        """
        Predicted terrain height at a single (x, z), in meters.

        start_height, if given, is where the sequential fold begins
        instead of the implicit 0.0 baseline -- lets a *separate*,
        smaller TerrainModel (e.g. just the hotspots added so far
        within one find_error_hotspots pass) be folded on top of this
        model's own result, without needing to rebuild this model's
        (potentially large) KD-tree to include those newer stamps too.
        """
        height = start_height
        for i in self._affecting_stamp_indices(x, z):
            stamp = self.stamps[i]
            dx = x - stamp.x
            dz = z - stamp.z
            profile = BRUSH_PROFILES.get(stamp.brush)
            if profile is not None and profile.shape == SHAPE_SQUARE:
                # Square footprint, axis-aligned (rotation=0 only --
                # see terrain/stamp.py): Chebyshev/L-infinity distance,
                # not Euclidean. A rotated square would need dx/dz
                # rotated into the stamp's local frame first; not
                # supported yet since nothing uses rotation != 0.
                dist = max(abs(dx), abs(dz))
            else:
                dist = (dx * dx + dz * dz) ** 0.5
            if dist > stamp.radius:
                continue
            r = dist / stamp.radius
            weight = self._kernels[stamp.brush].sample(r)
            if stamp.tool == TOOL_RAISE:
                height += stamp.value * weight
            else:
                height += (stamp.value - height) * weight
        return height

    def evaluate_many(self, points: np.ndarray, start_heights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predicted terrain height at many (x, z) points.

        points: shape (N, 2), columns (x, z).
        Returns: shape (N,) heights, in meters.

        start_heights, if given (shape (N,)), is where each point's
        fold begins instead of the implicit zero baseline -- same
        purpose as evaluate()'s start_height: folding a separate,
        smaller model's stamps on top of this one's own result without
        rebuilding this model's KD-tree to include them too.

        Different points are independent of each other (a stamp's
        effect on one point doesn't depend on its effect on another),
        so this loops over *stamps*, not points -- for each stamp,
        finding and updating every point it affects in one vectorized
        numpy operation, rather than calling evaluate() (which itself
        loops over stamps) once per point. Only the fold *within* one
        point's own affecting stamps needs to preserve order, which
        this does, since stamps are still visited in original list
        order -- just with the per-stamp update applied to every
        affected point at once instead of one point at a time.

        Stamps are pre-filtered to only those whose reach could
        possibly overlap this batch of points at all (via the same
        KD-tree built in __init__, queried once against the batch's
        bounding-box center) -- without that, a cumulative stamp list
        that's grown into the thousands over many refinement passes
        would mean scanning every stamp for every batch, even when a
        given batch (e.g. a few hundred LIDAR points within one
        hotspot's small radius) is nowhere near most of them.
        """
        points = np.asarray(points, dtype=np.float64)
        n = points.shape[0]
        heights = np.zeros(n, dtype=np.float64) if start_heights is None else np.array(start_heights, dtype=np.float64)
        if not self.stamps or n == 0 or self._tree is None:
            return heights

        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_z, max_z = points[:, 1].min(), points[:, 1].max()
        center = [(min_x + max_x) / 2.0, (min_z + max_z) / 2.0]
        half_diag = math.hypot(max_x - min_x, max_z - min_z) / 2.0

        # Safe (if slightly loose) superset: any stamp actually
        # affecting any point in this batch must have its center
        # within half_diag + that stamp's own Euclidean reach of the
        # batch's bounding-box center (triangle inequality) -- and
        # self._max_radius already accounts for square stamps' corners
        # reaching further than their radius (see _euclidean_reach),
        # so using it here is the same safety margin _affecting_stamp_indices
        # relies on for the single-point case.
        candidate_idx = np.asarray(
            self._tree.query_ball_point(center, half_diag + self._max_radius), dtype=np.int64
        )
        candidate_idx.sort()  # preserve original placement/application order

        px, pz = points[:, 0], points[:, 1]
        for i in candidate_idx:
            stamp = self.stamps[i]
            dx = px - stamp.x
            dz = pz - stamp.z
            profile = BRUSH_PROFILES.get(stamp.brush)
            if profile is not None and profile.shape == SHAPE_SQUARE:
                dist = np.maximum(np.abs(dx), np.abs(dz))
            else:
                dist = np.hypot(dx, dz)

            mask = dist <= stamp.radius
            if not np.any(mask):
                continue

            r = dist[mask] / stamp.radius
            weight = self._kernels[stamp.brush].sample_many(r)
            if stamp.tool == TOOL_RAISE:
                heights[mask] += stamp.value * weight
            else:
                heights[mask] += (stamp.value - heights[mask]) * weight

        return heights

    def render(self, resolution: int, bounds: Optional[BoundingBox] = None) -> np.ndarray:
        """
        Render a resolution x resolution height grid.

        This is for diagnostics only (e.g. preview_height.png). It is
        never used as terrain data internally -- the compiler never
        rasterizes for real optimization work (see "Important Design
        Rules": never rasterize terrain internally).

        Samples at histogram bin *centers* (edges = linspace(min, max,
        resolution+1), centers = midpoints of consecutive edges) --
        not at linspace(min, max, resolution) directly, which was a
        real bug found and fixed here: that endpoint-inclusive sampling
        doesn't match _bin_actual_elevation's bin-center convention
        (adaptive_refine.py), so `predicted - actual` in
        find_error_hotspots was comparing two differently-gridded
        arrays -- up to half a cell width (5 m at resolution=200)
        misaligned at the course edges, tapering to near-zero at the
        center. Both now use the exact same grid.

        Implemented via direct per-stamp bounding-box index arithmetic
        rather than calling evaluate_many() over every grid point: on a
        *regular* grid, a stamp's affected cells are directly computable
        from its center/radius, with no need for evaluate_many's
        KD-tree candidate search (built for arbitrary, non-grid-aligned
        point batches, which this isn't). Measured ~13x faster than the
        previous evaluate_many-based implementation at 3000 stamps,
        200x200 -- confirmed to produce bit-identical output first.
        """
        if bounds is None:
            if not self.stamps:
                raise ValueError(
                    "Cannot infer render bounds from an empty TerrainModel; "
                    "pass bounds explicitly."
                )
            xs = [s.x for s in self.stamps]
            zs = [s.z for s in self.stamps]
            bounds = BoundingBox(min_x=min(xs), min_z=min(zs), max_x=max(xs), max_z=max(zs))

        edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
        edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
        x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
        z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0
        cell_size_x = (bounds.max_x - bounds.min_x) / resolution
        cell_size_z = (bounds.max_z - bounds.min_z) / resolution

        accum = np.zeros((resolution, resolution), dtype=np.float64)

        for stamp in self.stamps:
            profile = BRUSH_PROFILES.get(stamp.brush)
            is_square = profile is not None and profile.shape == SHAPE_SQUARE
            # Bounding box uses the stamp's full Euclidean reach (radius
            # for circles, radius*sqrt(2) for squares' corners) -- the
            # exact per-cell shape test below (Chebyshev vs Euclidean
            # distance) is what actually decides inclusion; this box
            # only needs to be a safe superset, same principle as
            # TerrainModel._euclidean_reach for the KD-tree case.
            reach = stamp.radius * math.sqrt(2.0) if is_square else stamp.radius

            col_min = max(0, int((stamp.x - reach - bounds.min_x) / cell_size_x))
            col_max = min(resolution, int((stamp.x + reach - bounds.min_x) / cell_size_x) + 1)
            row_min = max(0, int((stamp.z - reach - bounds.min_z) / cell_size_z))
            row_max = min(resolution, int((stamp.z + reach - bounds.min_z) / cell_size_z) + 1)
            if col_min >= col_max or row_min >= row_max:
                continue

            sub_x = x_centers[col_min:col_max]
            sub_z = z_centers[row_min:row_max]
            xx, zz = np.meshgrid(sub_x, sub_z)
            dx = xx - stamp.x
            dz = zz - stamp.z
            dist = np.maximum(np.abs(dx), np.abs(dz)) if is_square else np.hypot(dx, dz)

            mask = dist <= stamp.radius
            if not np.any(mask):
                continue
            r = dist[mask] / stamp.radius
            weight = self._kernels[stamp.brush].sample_many(r)

            sub_accum = accum[row_min:row_max, col_min:col_max]
            local = sub_accum[mask]
            if stamp.tool == TOOL_RAISE:
                sub_accum[mask] = local + stamp.value * weight
            else:
                sub_accum[mask] = local + (stamp.value - local) * weight
            accum[row_min:row_max, col_min:col_max] = sub_accum

        return accum
