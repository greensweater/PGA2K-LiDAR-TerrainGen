"""
terrain/brush_profiles.py

Brush falloff curves, used as pull-toward-value weights under PGA's
"flatten" tool semantics (see terrain_model.py):

    new_height = old_height + (stamp.value - old_height) * kernel.sample(r)

Each BrushProfile.samples holds (r, weight) rows where:
    r      -- normalized radial distance from the stamp center, 0 (center)
              to 1 (full scale radius)
    weight -- normalized pull weight (0..1). At r=0 (dead center, under
              the cursor), weight should be exactly 1.0: placing a stamp
              should fully commit the terrain under the cursor to the
              typed value, not just partially pull toward it.

Sourced directly from each brush's real PNG asset now, not sparse
manual stake measurements: a one-off, offline script
(extract_brush_profiles.py) scans a single radius of each 512x512
brush PNG, edge to center, giving 256 real, precisely measured
grayscale values per brush -- committed here as brush_profiles.json,
loaded once at import time. This replaced an earlier version based on
manually-measured ring-to-ring drops (stakes placed every 10 m, only
10 rings per brush) for types 8/9/10/54, and an analytical flat-1.0
approximation for 72/73 (no measured data existed for those at all).
Both are superseded entirely by this real data: confirmed directly
(via composite_render.py's independent, real-2D-image-compositing
renderer) that the kernel-based model this module feeds was already
close to what real brush compositing produces, so this isn't fixing a
significant accuracy problem -- it's replacing a 10-point interpolated
approximation with the real, exact curve now that it's available, at
effectively no cost.

The one thing this can't do that direct 2D PNG compositing (see
composite_render.py) can: represent a brush that ISN'T radially
symmetric. Type 72 (square) is the one real example -- a single-radius
scan can't capture "square", so this profile is only ever a faithful
model of type 72 at exactly the angle it was measured along, elsewhere
it's an approximation. In practice this doesn't matter: type 72 is
only ever used for the whole-course-covering special-purpose stamps
(baseline flatten, the write-time normalization shim, registration
marks) with a radius so much larger than the course itself that every
point actually rendered sits deep in the brush's fully-saturated
interior, never near the corner-vs-edge transition where its squareness
would actually show up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHAPE_CIRCLE = "circle"
SHAPE_SQUARE = "square"

# Real brush PNG assets (and, alongside them, brush_profiles.json --
# the output of running extract_brush_profiles.py once against those
# same PNGs) live in a "brushes/" folder at the project root -- see
# composite_render.py's DEFAULT_BRUSH_DIR, the same convention.
BRUSH_ASSETS_DIR = Path(__file__).resolve().parent.parent / "viz/brushes"
BRUSH_PROFILES_JSON = BRUSH_ASSETS_DIR / "brush_profiles.json"


@dataclass(frozen=True, slots=True)
class BrushProfile:
    brush_id: int
    samples: np.ndarray  # shape (N, 2): columns are (r, normalized_weight)
    shape: str = SHAPE_CIRCLE  # SHAPE_CIRCLE (Euclidean r) or SHAPE_SQUARE (Chebyshev r)

    def sorted_samples(self) -> np.ndarray:
        """Samples sorted by r, for safe use with np.interp."""
        return self.samples[np.argsort(self.samples[:, 0])]


def _profile_from_pixel_scan(brush_id: int, values: list[int], shape: str = SHAPE_CIRCLE) -> BrushProfile:
    """
    Build a BrushProfile from one real brush PNG's own pixel values --
    `values` is grayscale intensity (0-255), one sample per pixel,
    ordered from the brush's outer edge (index 0) in to its center
    (the last index) -- extract_brush_profiles.py's own output format,
    a plain horizontal scan across the source 512x512 image's middle
    row from the left edge to the center.
    """
    n = len(values)
    # Reverse to center-to-edge (matching this module's own r=0-at-center
    # convention) and normalize the raw 0-255 grayscale scale to a 0-1
    # weight.
    weights = np.asarray(values[::-1], dtype=np.float64) / 255.0
    radii = np.linspace(0.0, 1.0, n)
    return BrushProfile(brush_id=brush_id, samples=np.column_stack((radii, weights)), shape=shape)


def _load_all_profiles() -> dict[int, BrushProfile]:
    entries = json.loads(BRUSH_PROFILES_JSON.read_text())
    shapes = {72: SHAPE_SQUARE}  # every other brush is a circle; see SHAPE_CIRCLE/SHAPE_SQUARE above
    return {
        entry["type"]: _profile_from_pixel_scan(
            entry["type"], entry["values"], shape=shapes.get(entry["type"], SHAPE_CIRCLE),
        )
        for entry in entries
    }


BRUSH_PROFILES: dict[int, BrushProfile] = _load_all_profiles()
