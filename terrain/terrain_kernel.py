"""
terrain/terrain_kernel.py

TerrainKernel: samples one BrushProfile's normalized falloff curve at an
arbitrary radius via linear interpolation between the measured rings.
"""

from __future__ import annotations

import numpy as np

from terrain.brush_profiles import BrushProfile


class TerrainKernel:
    """
    Wraps a single BrushProfile with fast radius sampling.

    r is normalized: 0 = stamp center, 1 = stamp edge. Every profile in
    brush_profiles.py is anchored to height 0.0 at r=1.0, so sampling
    past the edge naturally falls off to zero rather than needing special
    handling here.
    """

    def __init__(self, profile: BrushProfile):
        self.profile = profile
        samples = profile.sorted_samples()
        self._radii = samples[:, 0]
        self._heights = samples[:, 1]

    def sample(self, r: float) -> float:
        """
        Sample normalized brush amplitude.

        Parameters
        ----------
        r
            Normalized distance from center.
            0 = center
            1 = brush edge

        Returns
        -------
        float
            Normalized height (0..1)
        """
        r_clamped = min(max(r, 0.0), 1.0)
        return float(np.interp(r_clamped, self._radii, self._heights))

    def sample_many(self, r: np.ndarray) -> np.ndarray:
        """
        Vectorized form of sample(), for evaluating many radii at once
        (terrain_model.py's evaluate_many / render will need this).
        """
        r_clamped = np.clip(r, 0.0, 1.0)
        return np.interp(r_clamped, self._radii, self._heights)
