"""
ingest/tree_detection.py

Detects individual trees directly from classified LIDAR points --
ported from Chad Rockey's TGC-Designer-Tools (tree_mapper.py's
getTreeCoordinates), same canopy-height-model + watershed-segmentation
technique, retargeted onto this project's own heightmap/scipy stack
instead of his cv2/imutils pipeline. Confirmed against his actual
source, not guessed:

  1. Rasterize ground-classified points (see laz_reader.py's
     BARE_EARTH_CLASSIFICATIONS) into a ground heightmap -- already
     have this, ingest/heightmap.py's rasterize_ground_heightmap.
  2. Rasterize VEGETATION-classified points (see VEGETATION_CLASSIFICATIONS
     below) into a canopy heightmap -- see rasterize_canopy_heightmap.
     MAX-per-cell, deliberately not mean-per-cell like the ground
     raster: a canopy's top surface is what a tree-height measurement
     actually means, and averaging returns from gaps between branches
     with returns off the actual crown top would understate real tree
     height. This is a design choice made porting this, not something
     directly confirmed in Chad's own rasterizer (which this module
     never had visibility into, only tree_mapper.py's consumer side).
  3. canopy_height = canopy - ground (height above ground at every
     cell -- a standard Canopy Height Model / CHM). Implausible spikes
     (see outlier_height_m) are dropped, gaps filled (reusing
     ingest/heightmap.py's fill_heightmap_gaps -- same harmonic-
     inpainting job Chad's own infill_image.infill_image_scipy did,
     just already built here for a different reason), then smoothed
     (so one real tree's crown doesn't fracture into several small
     detections).
  4. Adaptive local threshold -> binary "is this a crown pixel" mask,
     distance transform, peak_local_max seeds one point per local
     blob peak, watershed floods outward from each seed to split
     touching/overlapping crowns into separate instances.
  5. Per watershed instance: centroid/equivalent-radius via
     skimage.measure.regionprops (replaces Chad's cv2.findContours +
     minEnclosingCircle -- these aren't quite the same geometric
     construction, equivalent-area radius vs. smallest enclosing
     circle, but both are just "a representative radius for a roughly
     round blob" and the difference doesn't matter at tree-crown
     scale), height sampled from the CHM at that centroid, dropped if
     under min_height_m (filters out low vegetation/underbrush that
     isn't really a discrete tree).

One real bug in Chad's original, independent of this port: his
`from skimage.morphology import watershed` is a stale import path --
current skimage (confirmed directly, 0.26) moved watershed to
skimage.segmentation and dropped peak_local_max's indices=False mode
entirely (it now always returns coordinates, never a boolean image).
Neither issue is a port judgment call, both are just what current
skimage actually requires -- if anyone tried running his original code
against a current skimage install, the import alone would fail.

NEW to this port, not in Chad's original: mask_geometry. Detected
trees are filtered to those falling inside the course's own
height_mask.geojson polygon (see ingest/osm.py's build_height_mask) --
i.e. the buffered fairway/green/tee/hole-corridor "core play area",
the same mask adaptive_refine.py already uses for terrain-refinement
tolerance. Outside that area, the game's own procedural vegetation
fill is expected to populate trees on its own; detecting and placing
real trees there too would double up with that, not add to it.
Filtering happens AFTER detection (against each instance's final
centroid), not by masking the canopy raster before watershed -- an
artificial hard raster boundary right through a real tree crown would
corrupt its watershed segmentation and give a wrong radius/height for
any tree straddling the mask edge; a post-filter just drops or keeps
each already-correctly-segmented tree whole.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed

from ingest.heightmap import fill_heightmap_gaps
from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox

# ASPRS classification codes for vegetation returns -- see
# laz_reader.py's BARE_EARTH_CLASSIFICATIONS for the ground-side
# equivalent this is meant to sit alongside.
CLASS_LOW_VEGETATION = 3
CLASS_MEDIUM_VEGETATION = 4
CLASS_HIGH_VEGETATION = 5
VEGETATION_CLASSIFICATIONS = (CLASS_LOW_VEGETATION, CLASS_MEDIUM_VEGETATION, CLASS_HIGH_VEGETATION)

# Tuning constants -- Chad's own defaults (tree_mapper.py), kept as
# the starting point, but redefined in real-world units (meters)
# rather than his raw pixel counts, since this port takes an explicit
# `resolution` rather than assuming one fixed raster size -- a pixel
# count only means the same thing at a fixed resolution, meters don't.
DEFAULT_BLUR_SIGMA_M = 1.5  # approximate real-world equivalent of Chad's kernel-size-5 Gaussian blur
DEFAULT_THRESHOLD_BLOCK_M = 20.0  # approximate real-world equivalent of his adaptiveThreshold blockSize=101
DEFAULT_MIN_TREE_DISTANCE_M = 2.0
DEFAULT_MIN_HEIGHT_M = 3.5
DEFAULT_OUTLIER_HEIGHT_M = 40.0
# Absolute CHM floor (m) a cell must clear before it's even eligible to
# be a threshold candidate, independent of the relative/local
# threshold below -- see detect_trees_from_lidar's binary-mask step
# for why this matters: without it, a genuinely flat (no vegetation)
# region's own value and its local sliding-window mean are
# mathematically identical (both exactly 0), so a strict floating-
# point `>` comparison between two different computation paths that
# should be equal can flip essentially at random across that entire
# flat region from sub-epsilon rounding noise -- confirmed directly
# (39% of a 2000x2000 synthetic test's cells were flagged "candidate"
# with only 60 well-separated trees present, versus ~0.5% expected).
# Deliberately well below DEFAULT_MIN_HEIGHT_M: this only needs to
# exclude "definitely not vegetation" cells, not do the real
# height filtering, which still happens per-detected-instance at the
# end (using the real, non-normalized CHM value, not this floor).
DEFAULT_CANDIDATE_FLOOR_M = 1.0


def rasterize_canopy_heightmap(
    cloud: PointCloud,
    bounds: BoundingBox,
    resolution: int,
    classifications: tuple[int, ...] = VEGETATION_CLASSIFICATIONS,
) -> np.ndarray:
    """
    MAX bare-earth-equivalent elevation per cell, over vegetation-
    classified points only -- the canopy-top-surface counterpart to
    ingest/heightmap.py's rasterize_ground_heightmap, which is mean-
    per-cell (correct for a ground DTM, wrong for a canopy top -- see
    module docstring). NaN where a cell has no vegetation-classified
    points at all (no canopy there, not "canopy at height 0").
    """
    mask = np.isin(cloud.classification, classifications)
    x, z, elevation = cloud.x[mask], cloud.z[mask], cloud.elevation[mask]

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)

    canopy = np.full((resolution, resolution), np.nan, dtype=np.float64)
    if x.size == 0:
        return canopy

    col = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, resolution - 1)
    row = np.clip(np.searchsorted(z_edges, z, side="right") - 1, 0, resolution - 1)
    flat_idx = row * resolution + col

    # Max-per-bin via a sorted-then-deduplicated reduction: sort points
    # by (bin, elevation), then for each bin the last entry is its max.
    # Vectorized -- no per-point Python loop -- and correct regardless
    # of how many points land in any one cell (a dense canopy cell
    # might have hundreds of vegetation returns).
    order = np.lexsort((elevation, flat_idx))
    flat_idx_sorted = flat_idx[order]
    elevation_sorted = elevation[order]
    is_last_in_bin = np.empty(flat_idx_sorted.shape, dtype=bool)
    is_last_in_bin[:-1] = flat_idx_sorted[1:] != flat_idx_sorted[:-1]
    is_last_in_bin[-1] = True

    flat_canopy = canopy.reshape(-1)
    flat_canopy[flat_idx_sorted[is_last_in_bin]] = elevation_sorted[is_last_in_bin]
    return canopy


def _cell_size(bounds: BoundingBox, resolution: int) -> float:
    """Meters per cell -- both axes are assumed equal (a square course crop over a square raster)."""
    return (bounds.max_x - bounds.min_x) / resolution


def detect_trees_from_lidar(
    ground_heights: np.ndarray,
    canopy_heights: np.ndarray,
    bounds: BoundingBox,
    mask_geometry: Optional["shapely.geometry.base.BaseGeometry"] = None,  # noqa: F821
    blur_sigma_m: float = DEFAULT_BLUR_SIGMA_M,
    threshold_block_m: float = DEFAULT_THRESHOLD_BLOCK_M,
    min_tree_distance_m: float = DEFAULT_MIN_TREE_DISTANCE_M,
    min_height_m: float = DEFAULT_MIN_HEIGHT_M,
    outlier_height_m: float = DEFAULT_OUTLIER_HEIGHT_M,
    candidate_floor_m: float = DEFAULT_CANDIDATE_FLOOR_M,
    printf=print,
) -> list[tuple[float, float, float, float]]:
    """
    (x, z, radius, height) for every tree detected in canopy_heights
    above ground_heights -- see module docstring for the full
    algorithm (canopy height model -> threshold -> watershed) and its
    relationship to Chad's tree_mapper.py.

    ground_heights / canopy_heights must be the same shape, same
    bounds/resolution convention as every other heightmap in this
    project (row=z, col=x -- see ingest/heightmap.py). Gaps (NaN) in
    either are filled internally (fill_heightmap_gaps) before use, so
    callers don't need to pre-fill either raster themselves.

    mask_geometry, if given, drops any detected tree whose centroid
    falls outside it (see module docstring's "NEW to this port" note)
    -- pass the course's loaded height_mask.geojson geometry
    (ingest/osm.py's load_height_mask) to keep LIDAR-detected trees
    confined to the core play area, leaving everything else for the
    game's own procedural vegetation fill.
    """
    if ground_heights.shape != canopy_heights.shape:
        raise ValueError(
            f"ground_heights {ground_heights.shape} and canopy_heights {canopy_heights.shape} "
            "must be the same shape."
        )
    resolution = ground_heights.shape[0]
    cell_size = _cell_size(bounds, resolution)

    printf("Filling gaps in ground/canopy heightmaps before computing canopy height model...")
    ground_filled = fill_heightmap_gaps(ground_heights, bounds) if np.isnan(ground_heights).any() else ground_heights
    # A canopy raster legitimately has huge NaN regions (fairways,
    # greens, bunkers -- anywhere with no vegetation at all), unlike
    # the ground raster's occasional small gaps. Filling those in
    # wholesale would fabricate phantom canopy height across the
    # entire open course. Instead: treat "no vegetation return here"
    # as canopy height == ground height (0 m above ground, i.e. no
    # tree) directly, which is the actually-correct interpretation --
    # only fill_heightmap_gaps the (much smaller) remaining NaNs, if
    # any, that come from small true sensor gaps within vegetated
    # areas rather than "not vegetated at all".
    canopy_filled = np.where(np.isnan(canopy_heights), ground_filled, canopy_heights)
    if np.isnan(canopy_filled).any():
        canopy_filled = fill_heightmap_gaps(canopy_filled, bounds)

    chm = canopy_filled - ground_filled

    # Drop implausible spikes (stray high-altitude noise returns, per
    # Chad's own outlier handling) before smoothing, so one bad point
    # doesn't drag its whole smoothing neighborhood upward.
    chm[chm > outlier_height_m] = np.nan
    if np.isnan(chm).any():
        chm = fill_heightmap_gaps(chm, bounds)
    chm[chm < 0] = 0.0  # canopy can't be below ground; small negative noise -> flat ground

    blur_sigma_px = max(blur_sigma_m / cell_size, 0.5)
    smoothed = ndimage.gaussian_filter(chm, sigma=blur_sigma_px)

    smoothed_range = smoothed.max() - smoothed.min()
    if smoothed_range < 1e-9:
        printf("Canopy height model is completely flat -- no trees to detect.")
        return []
    normalized = (smoothed - smoothed.min()) / smoothed_range

    # Local-mean adaptive threshold -- the scipy-only equivalent of
    # Chad's cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C,
    # blockSize=101, C=0): a pixel counts as "crown" if it's above its
    # own local neighborhood's mean, not one fixed global threshold
    # (a global threshold would miss short trees in an otherwise flat
    # area and false-positive on the shoulders of one very tall one).
    threshold_block_px = max(int(round(threshold_block_m / cell_size)), 3)
    if threshold_block_px % 2 == 0:
        threshold_block_px += 1  # odd size, matching Chad's own blockSize=101 convention
    local_mean = ndimage.uniform_filter(normalized, size=threshold_block_px)
    # See DEFAULT_CANDIDATE_FLOOR_M: gate on genuine absolute CHM
    # height first (cheap, exact -- no floating-point-equality trap),
    # THEN apply the relative/local adaptive comparison only within
    # cells that already clear that floor. A flat, unvegetated region
    # never reaches this comparison at all now, instead of relying on
    # a strict `>` between two values that are supposed to be equal
    # there.
    above_floor = chm > candidate_floor_m
    binary = above_floor & (normalized > local_mean)

    if not binary.any():
        printf("No crown-candidate cells found -- no trees to detect.")
        return []

    min_tree_distance_px = max(int(round(min_tree_distance_m / cell_size)), 1)
    distance = ndimage.distance_transform_edt(binary)
    peak_coords = peak_local_max(
        distance, min_distance=min_tree_distance_px, labels=binary, exclude_border=True,
    )
    if peak_coords.size == 0:
        printf("No local crown peaks found -- no trees to detect.")
        return []

    markers_bool = np.zeros(distance.shape, dtype=bool)
    markers_bool[tuple(peak_coords.T)] = True
    markers, _ = ndimage.label(markers_bool, structure=np.ones((3, 3)))
    labels = watershed(-distance, markers, mask=binary)

    x_edges = np.linspace(bounds.min_x, bounds.max_x, resolution + 1)
    z_edges = np.linspace(bounds.min_z, bounds.max_z, resolution + 1)

    def _to_world(row: float, col: float) -> tuple[float, float]:
        x = bounds.min_x + (col + 0.5) * (x_edges[1] - x_edges[0])
        z = bounds.min_z + (row + 0.5) * (z_edges[1] - z_edges[0])
        return x, z

    printf(f"Watershed found {labels.max()} candidate crown(s); filtering by height"
           + (" and mask" if mask_geometry is not None else "") + "...")

    trees: list[tuple[float, float, float, float]] = []
    for region in regionprops(labels):
        row, col = region.centroid
        # equivalent-area radius -- see module docstring on why this
        # (not Chad's smallest-enclosing-circle) is used here.
        radius_px = np.sqrt(region.area / np.pi)
        radius_m = radius_px * cell_size

        row_i, col_i = int(round(row)), int(round(col))
        row_i = min(max(row_i, 0), resolution - 1)
        col_i = min(max(col_i, 0), resolution - 1)
        height = float(chm[row_i, col_i])
        if height < min_height_m:
            continue

        x, z = _to_world(row, col)
        if mask_geometry is not None:
            import shapely.vectorized
            if not shapely.vectorized.contains(mask_geometry, np.array([x]), np.array([z]))[0]:
                continue

        trees.append((x, z, radius_m, height))

    printf(f"{len(trees)} tree(s) kept after height/mask filtering.")
    return trees
