"""
terrain/brush_profiles.py

Empirically measured brush falloff curves.

Each BrushProfile.samples holds (r, height) rows where:
    r      -- normalized radial distance from the stamp center, 0 (center)
              to 1 (full scale radius)
    height -- normalized stamp height as a fraction of the *input*
              amplitude parameter used during measurement (amplitude=200),
              not the measured center-plateau height. This assumes brush
              height scales linearly with the amplitude parameter -- an
              unverified but reasonable default until measured otherwise.

Measurement method (see the architecture doc's "ADDENDUM: Stamp
measurements"): stakes were placed every 10 m out from center, and the
in-game feet value recorded at each stake is the height *drop from the
previous stake toward the edge* -- i.e. the values are ring-to-ring
deltas, read from the outside in. r=1.0 (the full scale radius) is true
ground level (height 0) by definition; every ring's height is built by
walking inward from that zero point and subtracting each recorded delta:

    height(r=0.9) = 0        - delta[9]
    height(r=0.8) = height(r=0.9) - delta[8]
    ...
    height(r=0.0) = height(r=0.1) - delta[0]

Summing all ring deltas and negating reproduces the separately-measured
center-plateau height to within ~1.5% for types 8/9/10, confirming this
interpretation (rather than each delta being independently measured
relative to the center).

Type 10 has no flat plateau (a cosine-like falloff), so its steepest
ring-to-ring drop lands right at the edge, not the center. Type 54 shares
that same shape. Only 9 of type 54's 10 rings were measured (r=0..0.8);
its r=0.9 ring is back-solved so the reconstructed center height matches
its separately-measured 392 ft plateau (see _TYPE_54_R90_DELTA_FT below).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEET_TO_METERS = 0.3048
REFERENCE_AMPLITUDE_M = 200.0


@dataclass(frozen=True, slots=True)
class BrushProfile:
    brush_id: int
    samples: np.ndarray  # shape (N, 2): columns are (r, normalized_height)

    def sorted_samples(self) -> np.ndarray:
        """Samples sorted by r, for safe use with np.interp."""
        return self.samples[np.argsort(self.samples[:, 0])]


def _profile_from_ring_deltas(brush_id: int, delta_ft: list[float]) -> BrushProfile:
    """
    Convert measured ring-to-ring feet deltas (r = 0, 10, 20, ... m, read
    outside-in) into a normalized (r, height) sample table, anchored at
    height=0 at r=1.0 (the true edge).
    """
    deltas_m = np.asarray(delta_ft, dtype=np.float64) * FEET_TO_METERS
    # Walk inward from the r=1.0 zero anchor: height(k) = -sum(deltas[k:])
    heights_m = -np.cumsum(deltas_m[::-1])[::-1]

    radii = np.arange(len(delta_ft), dtype=np.float64) * 0.1
    # Append the r=1.0 anchor point itself.
    radii = np.append(radii, 1.0)
    heights_m = np.append(heights_m, 0.0)

    heights = heights_m / REFERENCE_AMPLITUDE_M
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


# ---------------------------------------------------------------------------
# Build the lookup table
# ---------------------------------------------------------------------------

BRUSH_PROFILES: dict[int, BrushProfile] = {
    8: _profile_from_ring_deltas(8, _TYPE_8_DELTAS_FT),
    9: _profile_from_ring_deltas(9, _TYPE_9_DELTAS_FT),
    10: _profile_from_ring_deltas(10, _TYPE_10_DELTAS_FT),
    54: _profile_from_ring_deltas(54, _TYPE_54_DELTAS_FT + [_TYPE_54_R90_DELTA_FT]),
}"""
terrain/brush_profiles.py

Empirically measured brush falloff curves.

Each BrushProfile.samples holds (r, height) rows where:
    r      -- normalized radial distance from the stamp center, 0 (center)
              to 1 (full scale radius)
    height -- normalized stamp height as a fraction of the *input*
              amplitude parameter used during measurement (amplitude=200),
              not the measured center-plateau height. This assumes brush
              height scales linearly with the amplitude parameter -- an
              unverified but reasonable default until measured otherwise.

Measurements were taken in-game feet, at 10 m radial steps, from
scale=100 (100 m brush radius), amplitude=200. See the architecture doc's
"ADDENDUM: Stamp measurements" for the raw feet values and method.

Type 54's r=0.9 sample was not measured; it is extrapolated toward
baseline from the measured r=0.8 value (-0.4 ft).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEET_TO_METERS = 0.3048
REFERENCE_AMPLITUDE_M = 200.0


@dataclass(frozen=True, slots=True)
class BrushProfile:
    brush_id: int
    samples: np.ndarray  # shape (N, 2): columns are (r, normalized_height)

    def sorted_samples(self) -> np.ndarray:
        """Samples sorted by r, for safe use with np.interp."""
        return self.samples[np.argsort(self.samples[:, 0])]


def _profile_from_feet(
    brush_id: int,
    center_height_ft: float,
    delta_ft: list[float],
) -> BrushProfile:
    """
    Convert measured (center height, per-ring delta) feet values, taken at
    10 m steps starting at r=0, into a normalized (r, height) sample table.
    """
    radii = np.arange(len(delta_ft), dtype=np.float64) * 0.1
    center_m = center_height_ft * FEET_TO_METERS
    deltas_m = np.asarray(delta_ft, dtype=np.float64) * FEET_TO_METERS
    heights = (center_m + deltas_m) / REFERENCE_AMPLITUDE_M
    return BrushProfile(brush_id=brush_id, samples=np.column_stack((radii, heights)))


# ---------------------------------------------------------------------------
# Raw measurements (in-game feet), r = 0, 10, 20, ... m
# ---------------------------------------------------------------------------

# Type 8 -- large plateau, smooth S-shaped falloff
_TYPE_8_CENTER_FT = 650.0
_TYPE_8_DELTAS_FT = [0, 0, 0, 0, -2.1, -55.7, -209.7, -239.7, -117.7, -22.1]

# Type 9 -- smaller plateau, smoother falloff
_TYPE_9_CENTER_FT = 650.0
_TYPE_9_DELTAS_FT = [0, 0, 0, -7.5, -62.8, -154.7, -183.9, -138.8, -78.7, -21.8]

# Type 10 -- approximately cosine radial falloff (no flat plateau)
_TYPE_10_CENTER_FT = 630.0
_TYPE_10_DELTAS_FT = [-28.1, -66.5, -91.3, -102.8, -97.5, -79.7, -63.2, -50.8, -44.8, -14.8]

# Type 54 -- same profile shape as type 10, ~62% vertical amplitude.
# Only 9 rings were measured (r=0..80); r=0.9 is extrapolated below.
_TYPE_54_CENTER_FT = 392.0
_TYPE_54_DELTAS_FT = [-19.9, -48.5, -71.6, -63.9, -51.3, -33.4, -21.6, -7.5, -0.4]
_TYPE_54_R90_DELTA_M = -0.05  # extrapolated directly in meters (not feet)


# ---------------------------------------------------------------------------
# Build the lookup table
# ---------------------------------------------------------------------------

BRUSH_PROFILES: dict[int, BrushProfile] = {
    8: _profile_from_feet(8, _TYPE_8_CENTER_FT, _TYPE_8_DELTAS_FT),
    9: _profile_from_feet(9, _TYPE_9_CENTER_FT, _TYPE_9_DELTAS_FT),
    10: _profile_from_feet(10, _TYPE_10_CENTER_FT, _TYPE_10_DELTAS_FT),
}

_type_54_measured = _profile_from_feet(54, _TYPE_54_CENTER_FT, _TYPE_54_DELTAS_FT)
_type_54_center_m = _TYPE_54_CENTER_FT * FEET_TO_METERS
_type_54_r90_height = (_type_54_center_m + _TYPE_54_R90_DELTA_M) / REFERENCE_AMPLITUDE_M

BRUSH_PROFILES[54] = BrushProfile(
    brush_id=54,
    samples=np.vstack([
        _type_54_measured.samples,
        [[0.9, _type_54_r90_height]],
    ]),
)
