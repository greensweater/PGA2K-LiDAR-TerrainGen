"""
viz/composite_render.py

An independent, alternate terrain renderer: composites the REAL brush
PNG assets (not terrain/terrain_kernel.py's measured/interpolated 1D
radial approximation) onto a 16-bit-equivalent canvas, using actual
image-layer blending -- Normal (alpha blend toward a target value) for
flatten, Linear Dodge/Add for raise -- matching our existing tool
semantics exactly, just sourcing each stamp's per-pixel weight from a
real 2D image instead of a 1D kernel sample.

Deliberately independent of TerrainModel/TerrainKernel: same
sequential-fold algorithm, same stamp list, same tool semantics, but
this is a fully separate, reusable function, not tied to matplotlib or
preview generation at all -- specifically so it could later be dropped
into terrain/adaptive_refine.py's error-scoring/candidate-fitting path
too, not just used as a one-off diagnostic preview, without dragging
in any rendering-specific code to do so.

Opt-in, not part of the normal refine-terrain loop or its
auto-visualize: rendering this way means a per-stamp PIL resize/paste
loop over potentially thousands of stamps, meaningfully slower than
TerrainModel's vectorized numpy approach -- fine for an explicit,
manually-triggered preview, not for something regenerated automatically
after every pass.

Requires the 6 real brush PNG assets (type8.png, type9.png, type10.png,
type54.png, type72.png, type73.png -- 512x512 grayscale, stored as RGB
with R==G==B) to be present in a "brushes/" directory. These aren't
included with the pipeline itself (not free to redistribute) -- place
your own copies in a "brushes" folder alongside these source files, or
pass an explicit brush_dir to composite_stamps_to_canvas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from terrain.bounding_box import BoundingBox
from terrain.stamp import Stamp, TOOL_RAISE

DEFAULT_BRUSH_DIR = Path(__file__).parent / "brushes"

# All 6 brush types this pipeline uses have a real PNG asset -- see
# terrain/brush_profiles.py for the same set used by the kernel-based
# model.
BRUSH_FILENAMES = {
    8: "type8.png",
    9: "type9.png",
    10: "type10.png",
    54: "type54.png",
    72: "type72.png",
    73: "type73.png",
}

# 16-bit canvas convention: 0-65535 maps to [0, MAX_INGAME_HEIGHT_M]
# meters -- same ceiling writer.py's normalize_stamp_heights already
# enforces for the real export, believed to match the in-game editor's
# own internal working buffer (per direct confirmation this is a 16-bit
# format, not floating-point meters).
MAX_INGAME_HEIGHT_M = 275.0
UINT16_MAX = 65535


def load_brush_image(brush: int, brush_dir: Path = DEFAULT_BRUSH_DIR) -> np.ndarray:
    """
    Load a brush's real PNG asset as a single-channel float array in
    [0, 1] -- the image's own grayscale intensity (confirmed directly:
    all 6 brush assets have R==G==B everywhere, i.e. genuinely
    grayscale despite being stored as 3-channel RGB PNGs).
    """
    filename = BRUSH_FILENAMES.get(brush)
    if filename is None:
        raise ValueError(
            f"No brush PNG asset known for brush type {brush} -- "
            f"expected one of {sorted(BRUSH_FILENAMES)}."
        )
    path = Path(brush_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Brush asset not found: {path}. The composite renderer needs the 6 real brush "
            f"PNGs ({', '.join(sorted(BRUSH_FILENAMES.values()))}) in {brush_dir} -- "
            "these aren't bundled with the pipeline itself."
        )
    im = Image.open(path).convert("L")  # grayscale; confirmed R==G==B already, this is just type safety
    return np.asarray(im, dtype=np.float64) / 255.0


def composite_stamps_to_canvas(
    stamps: Sequence[Stamp],
    bounds: BoundingBox,
    resolution: int,
    brush_dir: Path = DEFAULT_BRUSH_DIR,
    max_height_m: float = MAX_INGAME_HEIGHT_M,
) -> np.ndarray:
    """
    Render stamps onto a resolution x resolution canvas by compositing
    each stamp's real brush PNG, in 16-bit-equivalent units (float64
    holding values on a 0-65535 scale, not an actual uint16 array --
    see below for why).

    Sequential fold, same as TerrainModel: canvas starts at all-0 and
    each stamp blends into whatever's already there, in stamp-list
    order -- flatten is Normal/alpha-blend-toward-value
    (`canvas = canvas + (target - canvas) * weight`), raise is Linear
    Dodge/Add (`canvas = canvas + target * weight`), matching our
    existing tool semantics exactly (confirmed as already correct
    against real in-game results) -- the only thing that changes here
    is that `weight` comes from a real brush PNG's per-pixel intensity
    instead of TerrainKernel's 1D radial sample.

    Held as float64 rather than true uint16 specifically to avoid
    Python-level intermediate overflow/wraparound while still working
    in the same 0-65535 unit granularity the real editor is believed to
    use -- this reproduces the QUANTIZATION (rounding to the nearest of
    65536 discrete levels) without also committing to a specific
    overflow/clamping behavior we haven't independently verified.
    Only rounded and clipped to a true uint16 range at the point of
    display (see visualize.py's render_composite_preview).

    For each stamp, samples the brush image DIRECTLY at each affected
    canvas cell's own coordinate (mapped into the brush's own 512x512
    pixel space via scipy.ndimage.map_coordinates, bilinear, 0 outside
    the brush's own extent) -- not a pre-resize-then-crop of the whole
    512x512 image. An earlier version did resize first; confirmed
    directly this was a real, significant bug: PIL's resize to a very
    small target (e.g. a sub-meter-radius stamp, common late in a
    heavily-refined session once DEC has shrunk hotspot radius a lot)
    averages the ENTIRE source image -- including the brush's own
    far-outside-the-circle black background -- into each output pixel,
    rather than representing the true value at that precise location.
    Measured directly: resizing type8's PNG to 1x1 gave 138, not the
    true center value of 255. Direct per-cell sampling is correct
    regardless of how small or large the stamp's own pixel footprint
    is, so this isn't a special case for tiny stamps -- it replaces
    the resize approach entirely, for every stamp size.

    Same bounding-box clipping principle as TerrainModel.render(): only
    the canvas sub-region actually covered by each stamp's own radius
    gets touched, not the whole canvas.

    rotation (Stamp.rotation) is accepted but not yet applied -- every
    stamp we've seen in practice so far uses rotation=0; stubbed in
    for later rather than silently ignored, so it's an easy, contained
    change when it's actually needed (rotate the normalized offset
    (nx, nz) below by -rotation before mapping to brush pixel
    coordinates, matching how a rotated stamp's footprint would need
    to be sampled).
    """
    cell_size_x = (bounds.max_x - bounds.min_x) / resolution
    cell_size_z = (bounds.max_z - bounds.min_z) / resolution
    edges_x = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    edges_z = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)
    x_centers = (edges_x[:-1] + edges_x[1:]) / 2.0
    z_centers = (edges_z[:-1] + edges_z[1:]) / 2.0

    canvas = np.zeros((resolution, resolution), dtype=np.float64)

    brush_cache: dict[int, np.ndarray] = {}

    def get_brush(brush_type: int) -> np.ndarray:
        if brush_type not in brush_cache:
            # Keep on a 0-255 scale here (not the [0,1] load_brush_image
            # itself returns) since map_coordinates below samples this
            # directly -- normalizing to a 0-1 weight happens once, after
            # sampling, not before.
            brush_cache[brush_type] = load_brush_image(brush_type, brush_dir) * 255.0
        return brush_cache[brush_type]

    brush_size = None  # set from the first loaded brush; all 6 assets are the same 512x512
    print(f"  composite_render: direct-coordinate-sampling method, {len(stamps)} stamps, "
          f"resolution={resolution}")

    for stamp in stamps:
        brush_img = get_brush(stamp.brush)
        if brush_size is None:
            brush_size = brush_img.shape[0]
        brush_center_px = (brush_size - 1) / 2.0
        brush_radius_px = brush_size / 2.0  # brush's own radius reaches to the image edge

        if stamp.rotation != 0.0:
            # Stubbed in, not yet applied -- see docstring.
            pass

        col_min = max(0, int((stamp.x - stamp.radius - bounds.min_x) / cell_size_x))
        col_max = min(resolution, int((stamp.x + stamp.radius - bounds.min_x) / cell_size_x) + 1)
        row_min = max(0, int((stamp.z - stamp.radius - bounds.min_z) / cell_size_z))
        row_max = min(resolution, int((stamp.z + stamp.radius - bounds.min_z) / cell_size_z) + 1)
        if col_min >= col_max or row_min >= row_max:
            continue  # entirely off-canvas

        sub_x = x_centers[col_min:col_max]
        sub_z = z_centers[row_min:row_max]
        xx, zz = np.meshgrid(sub_x, sub_z)  # shape (n_rows, n_cols)

        # Normalized offset from stamp center, in [-1, 1] across the
        # stamp's own radius.
        nx = (xx - stamp.x) / stamp.radius
        nz = (zz - stamp.z) / stamp.radius

        # Map into the brush image's own pixel coordinates.
        brush_rows = brush_center_px + nz * brush_radius_px
        brush_cols = brush_center_px + nx * brush_radius_px

        weight = ndimage.map_coordinates(
            brush_img, [brush_rows, brush_cols], order=1, mode="constant", cval=0.0,
        ) / 255.0

        region = canvas[row_min:row_max, col_min:col_max]
        target = stamp.value / max_height_m * UINT16_MAX

        if stamp.tool == TOOL_RAISE:
            canvas[row_min:row_max, col_min:col_max] = region + target * weight
        else:
            canvas[row_min:row_max, col_min:col_max] = region + (target - region) * weight

    return canvas


def canvas_to_meters(canvas: np.ndarray, max_height_m: float = MAX_INGAME_HEIGHT_M) -> np.ndarray:
    """
    Convert a composited 16-bit-equivalent canvas (see
    composite_stamps_to_canvas) to meters, rounding and clipping to a
    true uint16's representable range first -- this is the point where
    the quantization this whole renderer exists to reproduce actually
    gets applied; composite_stamps_to_canvas itself stays in
    unclamped float64 throughout the fold (see its own docstring).
    """
    clipped = np.clip(np.round(canvas), 0, UINT16_MAX).astype(np.uint16)
    return clipped.astype(np.float64) / UINT16_MAX * max_height_m
