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

    def evaluate(self, x: float, z: float) -> float:
        """Predicted terrain height at a single (x, z), in meters."""
        height = 0.0
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

    def evaluate_many(self, points: np.ndarray) -> np.ndarray:
        """
        Predicted terrain height at many (x, z) points.

        points: shape (N, 2), columns (x, z).
        Returns: shape (N,) heights, in meters.

        NOTE: this loops per-point today for correctness/simplicity. If
        it becomes a bottleneck, it's worth profiling before optimizing
        rather than guessing -- and any vectorization has to preserve
        per-point stamp ordering, which a naive batched approach won't
        do for free.
        """
        points = np.asarray(points, dtype=np.float64)
        heights = np.empty(points.shape[0], dtype=np.float64)
        for i in range(points.shape[0]):
            heights[i] = self.evaluate(points[i, 0], points[i, 1])
        return heights

    def render(self, resolution: int, bounds: Optional[BoundingBox] = None) -> np.ndarray:
        """
        Render a resolution x resolution height grid.

        This is for diagnostics only (e.g. preview_height.png). It is
        never used as terrain data internally -- the compiler never
        rasterizes for real optimization work (see "Important Design
        Rules": never rasterize terrain internally).
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

        grid_x, grid_z = np.meshgrid(
            np.linspace(bounds.min_x, bounds.max_x, resolution),
            np.linspace(bounds.min_z, bounds.max_z, resolution),
        )
        points = np.column_stack((grid_x.ravel(), grid_z.ravel()))
        heights = self.evaluate_many(points)
        return heights.reshape(resolution, resolution)
