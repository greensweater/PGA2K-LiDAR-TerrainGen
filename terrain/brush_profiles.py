"""
terrain/brush_profiles.py

Empirically measured brush falloff curves, used as pull-toward-value
weights under PGA's "flatten" tool semantics (see terrain_model.py):

    new_height = old_height + (stamp.value - old_height) * kernel.sample(r)

Each BrushProfile.samples holds (r, weight) rows where:
    r      -- normalized radial distance from the stamp center, 0 (center)
              to 1 (full scale radius)
    weight -- normalized pull weight (0..1). At r=0 (dead center, under
              the cursor), weight should be exactly 1.0: placing a stamp
              should fully commit the terrain under the cursor to the
              typed value, not just partially pull toward it.

Measurement method (see the architecture doc's "ADDENDUM: Stamp
measurements"): stakes were placed every 10 m out from center, and the
in-game feet value recorded at each stake is the height *drop from the
previous stake toward the edge* -- i.e. the values are ring-to-ring
deltas, read from the outside in. r=1.0 (the full scale radius) is true
ground level (weight 0) by definition; every ring's weight is built by
walking inward from that zero point and subtracting each recorded delta:

    height(r=0.9) = 0        - delta[9]
    height(r=0.8) = height(r=0.9) - delta[8]
    ...
    height(r=0.0) = height(r=0.1) - delta[0]

Type 10 has no flat plateau (a cosine-like falloff), so its steepest
ring-to-ring drop lands right at the edge, not the center. Type 54
shares that same shape. Only 9 of type 54's 10 rings were measured
(r=0..0.8); its r=0.9 ring was back-solved so the reconstructed center
matched its separately-measured 392 ft plateau.

No terrain-noise correction is applied (an earlier version corrected
for terrainNoise.json's always-on ambient bias, back-solved from these
same stake measurements). That correction doesn't make sense once
these measured-and-reconstructed profiles are replaced by the real
brush alpha masks extracted directly from the game's own assets (see
the "stamp and check" rework this module is slated for) -- those are
the raw, noise-free brush data itself, not a reading taken through the
in-game noise layer, so there's nothing left to correct for. In the
meantime, the reconstructed centers for types 8 and 9 come out just
under 1.0 (raw ~0.986 / ~0.988) rather than exactly 1.0, purely from
compounding rounding error across many independently-measured ring
deltas -- used as-is rather than force-normalized to exactly 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEET_TO_METERS = 0.3048
REFERENCE_AMPLITUDE_M = 200.0


SHAPE_CIRCLE = "circle"
SHAPE_SQUARE = "square"


@dataclass(frozen=True, slots=True)
class BrushProfile:
    brush_id: int
    samples: np.ndarray  # shape (N, 2): columns are (r, normalized_weight)
    shape: str = SHAPE_CIRCLE  # SHAPE_CIRCLE (Euclidean r) or SHAPE_SQUARE (Chebyshev r)

    def sorted_samples(self) -> np.ndarray:
        """Samples sorted by r, for safe use with np.interp."""
        return self.samples[np.argsort(self.samples[:, 0])]


def _profile_from_ring_deltas(
    brush_id: int,
    delta_ft: list[float],
) -> BrushProfile:
    """
    Convert measured ring-to-ring feet deltas (r = 0, 10, 20, ... m, read
    outside-in) into a normalized (r, weight) sample table, anchored at
    weight=0 at r=1.0 (the true edge).
    """
    deltas_m = np.asarray(delta_ft, dtype=np.float64) * FEET_TO_METERS
    heights_m = -np.cumsum(deltas_m[::-1])[::-1]

    radii = np.arange(len(delta_ft), dtype=np.float64) * 0.1
    heights = heights_m / REFERENCE_AMPLITUDE_M

    # Append the r=1.0 anchor point itself -- unshifted, since it was
    # never a measurement (see module docstring).
    radii = np.append(radii, 1.0)
    heights = np.append(heights, 0.0)

    return BrushProfile(brush_id=brush_id, samples=np.column_stack((radii, heights)))


# ---------------------------------------------------------------------------
# Raw measurements: ring-to-ring drops in in-game feet, r = 0, 10, ... 90 m
# ---------------------------------------------------------------------------

# Type 8 -- large plateau, smooth S-shaped falloff (measured center ~650 ft)
_TYPE_8_DELTAS_FT = [0, 0, 0, 0, -2.1, -55.7, -209.7, -239.7, -117.7, -22.1]

# Type 9 -- smaller plateau, smoother falloff (measured center ~650 ft)
_TYPE_9_DELTAS_FT = [0, 0, 0, -7.5, -62.8, -154.7, -183.9, -138.8, -78.7, -21.8]

# Type 10 -- approximately cosine radial falloff, no flat plateau
# (measured center ~630 ft)
_TYPE_10_DELTAS_FT = [-28.1, -66.5, -91.3, -102.8, -97.5, -79.7, -63.2, -50.8, -44.8, -14.8]

# Type 54 -- same profile shape as type 10, ~62% vertical amplitude.
# Only 9 rings were measured (r=0..0.8); r=0.9 is back-solved below so the
# reconstructed center matches the separately-measured 392 ft plateau.
_TYPE_54_DELTAS_FT = [-19.9, -48.5, -71.6, -63.9, -51.3, -33.4, -21.6, -7.5, -0.4]
_TYPE_54_MEASURED_CENTER_FT = 392.0
_TYPE_54_R90_DELTA_FT = -(_TYPE_54_MEASURED_CENTER_FT + sum(_TYPE_54_DELTAS_FT))


def _hard_edge_profile(brush_id: int, shape: str) -> BrushProfile:
    """
    A "hard" stamp: full weight (1.0) everywhere within the stamp,
    with no falloff ramp at all -- not even a thin one. An earlier
    version of this modeled a small bevel near the edge, on the
    (reasonable-sounding) assumption that "vertical walls with a tiny
    beveled edge" meant the brush's own alpha mask had a soft edge.
    Direct evidence points the other way: the apparent softness is
    much more likely an artifact of the game's own sub-1m terrain
    surface smoothing, not the stamp itself -- confirmed by directly
    counting individual pixels on a hard-edged circle stamp scaled to
    R=100m (128x128px), which wouldn't be possible to do cleanly if
    the mask itself were blurred at the edge. TerrainModel.evaluate()
    already treats anything beyond the stamp's radius as unaffected
    (dist > stamp.radius skips it entirely), so this only ever gets
    sampled for r in [0, 1] -- a flat 1.0 across that whole range is
    the correct hard-edge shape, not an approximation of one.
    """
    samples = np.array([
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    return BrushProfile(brush_id=brush_id, samples=samples, shape=shape)


# ---------------------------------------------------------------------------
# Build the lookup table
# ---------------------------------------------------------------------------

BRUSH_PROFILES: dict[int, BrushProfile] = {
    8: _profile_from_ring_deltas(8, _TYPE_8_DELTAS_FT),
    9: _profile_from_ring_deltas(9, _TYPE_9_DELTAS_FT),
    10: _profile_from_ring_deltas(10, _TYPE_10_DELTAS_FT),
    54: _profile_from_ring_deltas(54, _TYPE_54_DELTAS_FT + [_TYPE_54_R90_DELTA_FT]),
    72: _hard_edge_profile(72, SHAPE_SQUARE),  # hard square -- ESTIMATED, see _hard_edge_profile
    73: _hard_edge_profile(73, SHAPE_CIRCLE),  # hard circle -- ESTIMATED, see _hard_edge_profile
}
