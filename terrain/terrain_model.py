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

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from terrain.bounding_box import BoundingBox
from terrain.brush_profiles import BRUSH_PROFILES
from terrain.stamp import Stamp
from terrain.terrain_kernel import TerrainKernel


class TerrainModel:
    """
    Evaluates predicted terrain height by folding every affecting
    Stamp's pull over a starting height of 0.

    Under PGA's flatten tool, a stamp doesn't add to the terrain -- it
    pulls nearby height toward stamp.value, weighted by the brush's
    falloff:

        new_height = old_height + (stamp.value - old_height) * kernel.sample(r)

    This is a lerp toward stamp.value, not a sum, so a single stamp can
    never push terrain past its own value. It also means the result
    depends on the order stamps are applied in -- two stamps overlapping
    at different values give a different blended result depending on
    which one is folded in last. That's fine as long as this model
    replays stamps in the same order the in-game renderer will (list
    order, i.e. placement/generation order); the order itself doesn't
    need to be canonical, just consistent between the two.

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
            self._max_radius = max(s.radius for s in self.stamps)
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
            dist = (dx * dx + dz * dz) ** 0.5
            if dist > stamp.radius:
                continue
            r = dist / stamp.radius
            weight = self._kernels[stamp.brush].sample(r)
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
