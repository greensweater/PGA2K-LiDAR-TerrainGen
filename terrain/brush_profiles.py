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

Type 74 is a second, more visible instance of the same approximation:
a directional gradient brush (also SHAPE_SQUARE), used at real editing
scale via contour_layers.py's rect-fill path rather than as a giant
flatten stamp -- so unlike type 72, its non-radially-symmetric shape
isn't hidden deep in a saturated interior. Accepted tradeoff, not
solved here; see BrushProfile.rotation_offset_deg/scale_x_mult/
scale_z_mult below for the per-brush corrections it does need
(directionality and non-square aspect), applied at Stamp-construction
time via apply_brush_adjustment().
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

    # Per-brush corrections some assets need beyond shape/falloff -- a
    # fixed rotation offset (degrees) for brushes whose own gradient
    # axis isn't aligned with the geometry that computes a stamp's
    # rotation, and independent x/z scale multipliers for brushes whose
    # source asset isn't square. Both default to no-op. Applied via
    # apply_brush_adjustment() at Stamp-construction time -- callers
    # that build Stamps directly are responsible for calling it; these
    # fields are inert data, not enforced automatically (see that
    # function's docstring for why).
    rotation_offset_deg: float = 0.0
    scale_x_mult: float = 1.0
    scale_z_mult: float = 1.0

    # Per-brush texture-space offset for assets whose own bright/effective
    # point isn't at the source image's geometric center (unlike every
    # other brush here, where it is). Normalized to the brush's own
    # radius (+center_offset_x = toward the image's higher-column side,
    # +center_offset_z = toward its higher-row side), in the brush's own
    # un-rotated local frame -- so it stays glued to the brush's axes
    # (and therefore rotates along with a stamp's own rotation) rather
    # than being a fixed world-space shift. A rendering-only concern:
    # consumed by viz/composite_render.py's direct 2D pixel sampling
    # (the only renderer with a real, non-radially-symmetric picture of
    # the asset); NOT applied to a Stamp's own x/z (there's nothing to
    # write differently to userLayers.json -- the real game samples the
    # same texture the same way around a stamp's placement point).
    center_offset_x: float = 0.0
    center_offset_z: float = 0.0

    def sorted_samples(self) -> np.ndarray:
        """Samples sorted by r, for safe use with np.interp."""
        return self.samples[np.argsort(self.samples[:, 0])]


def _profile_from_pixel_scan(brush_id: int, values: list[int], shape: str = SHAPE_CIRCLE, **overrides) -> BrushProfile:
    """
    Build a BrushProfile from one real brush PNG's own pixel values --
    `values` is grayscale intensity (0-255), one sample per pixel,
    ordered from the brush's outer edge (index 0) in to its center
    (the last index) -- extract_brush_profiles.py's own output format,
    a plain horizontal scan across the source 512x512 image's middle
    row from the left edge to the center.

    `overrides` forwards any of BrushProfile's hand-authored per-brush
    corrections (rotation_offset_deg, scale_x_mult, scale_z_mult) --
    see _load_all_profiles.
    """
    n = len(values)
    # Reverse to center-to-edge (matching this module's own r=0-at-center
    # convention) and normalize the raw 0-255 grayscale scale to a 0-1
    # weight.
    weights = np.asarray(values[::-1], dtype=np.float64) / 255.0
    radii = np.linspace(0.0, 1.0, n)
    return BrushProfile(brush_id=brush_id, samples=np.column_stack((radii, weights)), shape=shape, **overrides)


def _load_all_profiles() -> dict[int, BrushProfile]:
    entries = json.loads(BRUSH_PROFILES_JSON.read_text())
    shapes = {72: SHAPE_SQUARE, 74: SHAPE_SQUARE}  # every other brush is a circle; see SHAPE_CIRCLE/SHAPE_SQUARE above
    # Hand-authored per-brush corrections (see BrushProfile.rotation_offset_deg/
    # scale_x_mult/scale_z_mult) -- kept separate from `entries`, which is
    # purely measured pixel-scan data from extract_brush_profiles.py.
    overrides = {74: {
        "rotation_offset_deg": 90.0, "scale_x_mult": 3.0, "scale_z_mult": 1.0,
        # Measured directly from type74.png: bilaterally symmetric about
        # its own horizontal midline (row 255.5, so no vertical offset),
        # but its bright region sits ~156.5px toward the high-column
        # side of the image's geometric center -- see composite_render.py.
        "center_offset_x": 156.5 / 256.0, "center_offset_z": 0.0,
    }}
    return {
        entry["type"]: _profile_from_pixel_scan(
            entry["type"], entry["values"], shape=shapes.get(entry["type"], SHAPE_CIRCLE),
            **overrides.get(entry["type"], {}),
        )
        for entry in entries
    }


BRUSH_PROFILES: dict[int, BrushProfile] = _load_all_profiles()


def apply_brush_adjustment(
    rotation_deg: float, scale_x: float, scale_z: float, brush: int,
) -> tuple[float, float, float]:
    """
    Fold a brush's rotation_offset_deg/scale_x_mult/scale_z_mult (see
    BrushProfile) into a stamp's own rotation/scale, once, at the point
    a Stamp is first constructed for that brush. Not applied via
    Stamp.__post_init__/a dataclass hook: height_fit.py re-invokes
    Stamp's constructor via dataclasses.replace() to tweak an existing
    stamp's value, which would silently re-apply any construction-time
    hook a second time (double rotation, double-multiplied scale) on a
    stamp that's already had it applied once. Calling this explicitly,
    only where rotation_deg/scale_x/scale_z are first computed, avoids
    that -- so every other call site that builds a Stamp directly must
    call this itself if the brush it uses needs the correction.

    A rotation_offset_deg that's an odd multiple of 90 degrees also
    swaps which caller-supplied extent (scale_x vs scale_z) ends up
    driving which of the brush's own local axes: rotating a stamp's
    local frame by a quarter turn necessarily swaps which of its two
    original world-space directions each local axis now points along
    (see local_square_offsets -- "across"/scale_x and "along"/scale_z
    are always perpendicular, so a 90 degree turn of the frame maps
    each one onto where the OTHER used to point; a half/full turn
    doesn't, it only flips sign along the same axis). Without this, the
    caller's own physical measurements (e.g. along-edge width vs
    into-interior depth) end up applied to the wrong visual extent
    post-rotation -- confirmed directly: without the swap, a fallline
    edge stamp using this offset stretched into/out of the interior
    using its edge-length measurement instead of its depth measurement,
    and vice versa.
    """
    profile = BRUSH_PROFILES.get(brush)
    if profile is None:
        return rotation_deg, scale_x, scale_z
    if round(profile.rotation_offset_deg / 90.0) % 2 == 1:
        scale_x, scale_z = scale_z, scale_x
    return (
        rotation_deg + profile.rotation_offset_deg,
        scale_x * profile.scale_x_mult,
        scale_z * profile.scale_z_mult,
    )
