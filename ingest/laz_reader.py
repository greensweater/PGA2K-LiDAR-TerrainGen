"""
io/laz_reader.py

Reads USGS LAZ/LAS tiles into a merged, origin-aligned point cloud.

The LAZ files are authoritative: they define the CRS, world coordinates,
and bounding box that every other pipeline stage aligns to (see the
"Important Design Rules" in the architecture doc).

This module never rasterizes. It reads raw LIDAR points, merges tiles
that share a common CRS, and hands back plain numpy arrays plus a
KD-tree for spatial queries. Any raster image (preview PNGs) is a
diagnostic artifact produced by visualize.py, never terrain data
consumed by the optimizer.

Coordinate convention, matching terrain.bounding_box and terrain.stamp:
    x  -- horizontal ground-plane axis (meters, local/origin-aligned)
    z  -- horizontal ground-plane axis (meters, local/origin-aligned),
          mapped from the LAZ file's projected northing ("y")
    elevation -- raw LIDAR height (meters), untouched vertical datum

Projected-CRS coordinates are only ever used internally, to compute the
merged extent and the local origin. Everything past load_point_cloud()
works in local x/z.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import laspy
import numpy as np
import pyproj
from scipy.spatial import cKDTree

from constants import EPSILON
from terrain.bounding_box import BoundingBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASPRS classification codes, as used by USGS 3DEP LAZ tiles
# ---------------------------------------------------------------------------

CLASS_GROUND = 2
CLASS_MODEL_KEY_POINT = 8
BARE_EARTH_CLASSIFICATIONS = (CLASS_GROUND, CLASS_MODEL_KEY_POINT)


class LazReadError(RuntimeError):
    """Raised when LAZ tiles can't be read, trusted, or merged safely."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TileExtent:
    """Header-only extent of a single LAZ tile, in its native projected CRS."""

    path: Path
    crs: pyproj.CRS
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    point_count: int


@dataclass(slots=True)
class PointCloud:
    """
    A merged LIDAR point cloud in local, origin-aligned coordinates.

    origin_x / origin_y are the projected-CRS coordinates that map to
    local (0, 0). Keep these around: they're the only link back to the
    source CRS, and OSM features must be reprojected through the same
    origin/CRS before they'll line up with this terrain.
    """

    x: np.ndarray
    z: np.ndarray
    elevation: np.ndarray
    classification: np.ndarray

    crs: pyproj.CRS
    origin_x: float
    origin_y: float

    _tree: Optional[cKDTree] = field(default=None, repr=False, compare=False)

    @property
    def count(self) -> int:
        return int(self.x.shape[0])

    @property
    def bounds(self) -> BoundingBox:
        if self.count == 0:
            raise LazReadError("Cannot compute bounds of an empty PointCloud")
        return BoundingBox(
            min_x=float(np.min(self.x)),
            min_z=float(np.min(self.z)),
            max_x=float(np.max(self.x)),
            max_z=float(np.max(self.z)),
        )

    @property
    def tree(self) -> cKDTree:
        """Lazily-built KD-tree over the (x, z) horizontal plane."""
        if self._tree is None:
            self._tree = cKDTree(np.column_stack((self.x, self.z)))
        return self._tree

    def query_radius(self, x: float, z: float, radius: float) -> np.ndarray:
        """Indices of points within `radius` meters of (x, z)."""
        return np.asarray(self.tree.query_ball_point([x, z], radius), dtype=np.int64)

    def bare_earth_mask(
        self,
        classifications: tuple[int, ...] = BARE_EARTH_CLASSIFICATIONS,
    ) -> np.ndarray:
        """Boolean mask selecting only ground / model-key-point returns."""
        return np.isin(self.classification, classifications)

    def subset(self, mask: np.ndarray) -> "PointCloud":
        """Return a new PointCloud containing only points where mask is True."""
        return PointCloud(
            x=self.x[mask],
            z=self.z[mask],
            elevation=self.elevation[mask],
            classification=self.classification[mask],
            crs=self.crs,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )

    def cropped(self, bounds: BoundingBox) -> "PointCloud":
        """Return a new PointCloud containing only points inside `bounds`."""
        mask = (
            (self.x >= bounds.min_x) & (self.x <= bounds.max_x)
            & (self.z >= bounds.min_z) & (self.z <= bounds.max_z)
        )
        return self.subset(mask)

    def save(self, path: Path) -> None:
        """Write to the project's cached pointcloud.npz artifact."""
        np.savez_compressed(
            path,
            x=self.x,
            z=self.z,
            elevation=self.elevation,
            classification=self.classification,
            crs_wkt=self.crs.to_wkt(),
            origin_x=self.origin_x,
            origin_y=self.origin_y,
        )
        logger.info("Wrote %d points to %s", self.count, path)

    @classmethod
    def load(cls, path: Path) -> "PointCloud":
        """Read back a previously cached pointcloud.npz artifact."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                x=data["x"],
                z=data["z"],
                elevation=data["elevation"],
                classification=data["classification"],
                crs=pyproj.CRS.from_wkt(str(data["crs_wkt"])),
                origin_x=float(data["origin_x"]),
                origin_y=float(data["origin_y"]),
            )


# ---------------------------------------------------------------------------
# Tile discovery + header scan
# ---------------------------------------------------------------------------

def find_laz_tiles(laz_dir: Path) -> list[Path]:
    """List every .laz/.las tile in laz_dir, sorted for stable ordering."""
    laz_dir = Path(laz_dir)
    tiles = sorted(
        p for p in laz_dir.iterdir()
        if p.suffix.lower() in (".laz", ".las")
    )
    if not tiles:
        raise LazReadError(f"No .laz/.las files found in {laz_dir}")
    return tiles


def scan_tile_extent(path: Path, force_crs: Optional[pyproj.CRS] = None) -> TileExtent:
    """
    Read only the header of a LAZ tile -- fast, no point decoding.

    Used to determine the merged bounds before committing to a full
    point-cloud load (the "determine merged bounds" pipeline stage,
    which runs before the OSM download).
    """
    with laspy.open(path) as reader:
        header = reader.header
        crs = force_crs or header.parse_crs()
        if crs is None:
            raise LazReadError(
                f"{path.name}: no CRS found in the LAZ header and no force_crs "
                "was given. USGS 3DEP tiles usually carry this in a GeoTIFF or "
                "WKT VLR; if it's missing, pass force_crs explicitly."
            )
        return TileExtent(
            path=path,
            crs=crs,
            min_x=header.x_min,
            max_x=header.x_max,
            min_y=header.y_min,
            max_y=header.y_max,
            point_count=header.point_count,
        )


def scan_merged_extent(
    laz_dir: Path,
    force_crs: Optional[pyproj.CRS] = None,
) -> tuple[list[TileExtent], BoundingBox, pyproj.CRS]:
    """
    Scan every tile's header and return the merged bounds, still in
    projected-CRS coordinates (not yet localized to an origin).
    """
    tiles = find_laz_tiles(laz_dir)
    extents = [scan_tile_extent(p, force_crs=force_crs) for p in tiles]

    crs = extents[0].crs
    for extent in extents[1:]:
        if extent.crs != crs:
            raise LazReadError(
                f"{extent.path.name} has CRS {extent.crs} but "
                f"{extents[0].path.name} has {crs}. All tiles must share one "
                "CRS -- reproject mismatched tiles before merging, or pass "
                "force_crs if the mismatch is spurious."
            )

    bounds = BoundingBox(
        min_x=min(e.min_x for e in extents),
        min_z=min(e.min_y for e in extents),
        max_x=max(e.max_x for e in extents),
        max_z=max(e.max_y for e in extents),
    )
    logger.info(
        "Scanned %d tiles (%d points), merged extent %.1f x %.1f m, CRS %s",
        len(extents),
        sum(e.point_count for e in extents),
        bounds.max_x - bounds.min_x,
        bounds.max_z - bounds.min_z,
        crs,
    )
    return extents, bounds, crs


# ---------------------------------------------------------------------------
# Full point cloud load
# ---------------------------------------------------------------------------

def load_point_cloud(
    laz_dir: Path,
    origin_x: Optional[float] = None,
    origin_y: Optional[float] = None,
    force_crs: Optional[pyproj.CRS] = None,
) -> PointCloud:
    """
    Load every tile in `laz_dir` into one origin-aligned PointCloud.

    If origin_x / origin_y are not given, the lower-left corner of the
    merged extent is used as local (0, 0) -- a zero-lower-left frame,
    same convention Chad's tool used, so bearings stay comparable.
    """
    extents, bounds, crs = scan_merged_extent(laz_dir, force_crs=force_crs)

    if origin_x is None:
        origin_x = bounds.min_x
    if origin_y is None:
        origin_y = bounds.min_z  # min_z holds the scanned min northing ("y")

    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    elevations: list[np.ndarray] = []
    classifications: list[np.ndarray] = []

    for extent in extents:
        las = laspy.read(extent.path)
        tile_crs = force_crs or las.header.parse_crs()
        if tile_crs is None or tile_crs != crs:
            raise LazReadError(
                f"{extent.path.name}: CRS changed or was lost between the "
                "header scan and the full read. Re-run scan_merged_extent."
            )

        xs.append(np.asarray(las.x, dtype=np.float64) - origin_x)
        zs.append(np.asarray(las.y, dtype=np.float64) - origin_y)
        elevations.append(np.asarray(las.z, dtype=np.float64))
        classifications.append(np.asarray(las.classification, dtype=np.uint8))

    cloud = PointCloud(
        x=np.concatenate(xs),
        z=np.concatenate(zs),
        elevation=np.concatenate(elevations),
        classification=np.concatenate(classifications),
        crs=crs,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    logger.info("Loaded %d points into local frame (origin %.1f, %.1f)",
                cloud.count, origin_x, origin_y)
    return cloud


# ---------------------------------------------------------------------------
# Crop selection
# ---------------------------------------------------------------------------

def select_crop(
    bounds: BoundingBox,
    size_m: float,
    center_x: Optional[float] = None,
    center_z: Optional[float] = None,
) -> BoundingBox:
    """
    Compute a size_m x size_m crop window inside `bounds`, centered on
    (center_x, center_z) and clamped so it never asks for terrain outside
    the LAZ extent.

    With no center given, defaults to the center of `bounds` -- a
    reasonable starting crop before the user has picked one (playable
    terrain is always 2000 x 2000 m; see constants.COURSE_SIZE_M).
    """
    available_x = bounds.max_x - bounds.min_x
    available_z = bounds.max_z - bounds.min_z
    if size_m > available_x + EPSILON or size_m > available_z + EPSILON:
        raise LazReadError(
            f"Requested {size_m:.1f} m crop doesn't fit inside the "
            f"{available_x:.1f} x {available_z:.1f} m LAZ extent."
        )

    if center_x is None:
        center_x = (bounds.min_x + bounds.max_x) / 2.0
    if center_z is None:
        center_z = (bounds.min_z + bounds.max_z) / 2.0

    half = size_m / 2.0
    min_x, max_x = center_x - half, center_x + half
    min_z, max_z = center_z - half, center_z + half

    # Shift (never shrink) the window so it stays inside bounds.
    if min_x < bounds.min_x:
        min_x, max_x = bounds.min_x, bounds.min_x + size_m
    elif max_x > bounds.max_x:
        min_x, max_x = bounds.max_x - size_m, bounds.max_x

    if min_z < bounds.min_z:
        min_z, max_z = bounds.min_z, bounds.min_z + size_m
    elif max_z > bounds.max_z:
        min_z, max_z = bounds.max_z - size_m, bounds.max_z

    return BoundingBox(min_x=min_x, min_z=min_z, max_x=max_x, max_z=max_z)


def recentered_crop(
    cloud: PointCloud,
    size_m: float,
    center_x: Optional[float] = None,
    center_z: Optional[float] = None,
) -> PointCloud:
    """
    Crop `cloud` to a size_m x size_m window (see select_crop) and shift
    the result so its local (0, 0) sits at the crop's lower-left corner.

    This is what makes a loaded LAZ point cloud directly usable with
    TerrainModel / hexgrid.generate_hex_grid: both already work in a
    local [0, size_m] x [0, size_m] frame (playable terrain is always
    2000 x 2000 m; see constants.COURSE_SIZE_M), so after this call,
    stamp coordinates and point cloud coordinates line up with no
    further translation needed anywhere else in the pipeline.

    origin_x / origin_y absorb the crop shift, so they keep being the
    single source of truth for converting local (0, 0) back to real
    projected-CRS coordinates -- needed later to align OSM features,
    which are pulled in real-world coordinates, not local ones.
    """
    crop_bounds = select_crop(cloud.bounds, size_m, center_x=center_x, center_z=center_z)
    cropped = cloud.cropped(crop_bounds)
    return PointCloud(
        x=cropped.x - crop_bounds.min_x,
        z=cropped.z - crop_bounds.min_z,
        elevation=cropped.elevation,
        classification=cropped.classification,
        crs=cropped.crs,
        origin_x=cropped.origin_x + crop_bounds.min_x,
        origin_y=cropped.origin_y + crop_bounds.min_z,
    )
