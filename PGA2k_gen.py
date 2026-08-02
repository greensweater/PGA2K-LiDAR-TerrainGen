#!/usr/bin/env python3
"""
PGA2k_gen.py

CLI orchestrator for the terrain compiler. Operates on a working
directory, running one pipeline step at a time:

    PGA2k_gen.py <working_dir>                       (same as --step init)
    PGA2k_gen.py <working_dir> --step init
    PGA2k_gen.py <working_dir> --step ingest-laz [--projection <EPSG>]
    PGA2k_gen.py <working_dir> --step ingest-osm
    PGA2k_gen.py <working_dir> --step ingest-course --course-file <path>
    PGA2k_gen.py <working_dir> --step generate-terrain
    PGA2k_gen.py <working_dir> --step refine-terrain [--error-tolerance M] [--resolution N]
    PGA2k_gen.py <working_dir> --step output-terrain
    PGA2k_gen.py <working_dir> --step repack --repack-filename <name>

Each step reads/writes plain-file artifacts in <working_dir> instead of
holding state in memory across invocations -- this is a CLI today, a
GUI eventually (per the architecture doc), so every step needs to be
independently resumable and inspectable, never a black box.

<working_dir> layout:
    laz/                    input LAZ/LAS tiles
    map.osm                 input OSM export (user-downloaded, using
                             the lat/lon bbox ingest-laz prints)
    features.geojson         ingest-osm output (see osm.py): classified
                             vector Features in the course's local frame
    project.json             small state manifest (projection, merged
                             bounds, course origin, course_name) carried
                             between steps so they don't need re-specifying
    pointcloud.npz            ingest-laz output (ingest.laz_reader.PointCloud)
    initial_stamps.json      generate-terrain output (Stamp list)
    refine_stamps_N.json      refine-terrain output: only the stamps
                             pass N added (not a cumulative snapshot);
                             deleting the highest N undoes that pass
    course/                  extracted blank .course, always at this
                             fixed path (see ingest-course / output-terrain)

Step ordering is enforced with clear errors (e.g. generate-terrain
without a pointcloud.npz on disk yet) rather than letting a later step
fail on a confusing missing-file exception.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyproj

SCRIPT_DIR = Path(__file__).resolve().parent

from constants import (
    COURSE_SIZE_M, PREVIEW_ERROR, PREVIEW_HEIGHT, PREVIEW_HEX,
    PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_MASK, PREVIEW_OSM, PREVIEW_OSM_FULL, PREVIEW_STAMPS,
    POINTCLOUD_FILE, PREVIEW_DIR, PROJECT_FILE, STAMPS_DIR,
)
import visualize as viz
from ingest.laz_reader import LazReadError, PointCloud, load_point_cloud, recentered_crop
from ingest.heightmap import DEFAULT_HEIGHTMAP_RESOLUTION, load_heightmap, rasterize_ground_heightmap, save_heightmap
from ingest.osm import (
    DEFAULT_HEIGHT_MASK_BUFFER_PX, build_height_mask, load_features, load_height_mask,
    parse_osm_features, rasterize_mask, save_features, save_height_mask, shift_features,
)
from splines import build_surface_splines, feature_to_spline, save_surface_splines
from terrain.adaptive_refine import (
    DEFAULT_CLAIM_RADIUS_FRACTION,
    DEFAULT_BRUSH_RADIUS_SPREAD_RATIO,
    DEFAULT_MAX_HOTSPOT_RADIUS_M,
    DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    DEFAULT_MIN_HOTSPOT_RADIUS_M,
    DEFAULT_RADIUS_DECAY_PER_PASS,
    DEFAULT_RESOLUTION,
    refine_stamps,
)
from terrain.bounding_box import BoundingBox
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import generate_hex_grid
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel
from writer import build_baseline_flatten_stamp, normalize_stamp_heights, write_user_layers

INITIAL_STAMPS_FILE = "initial_stamps.json"
FEATURES_FILE = "features.geojson"
HEIGHT_MASK_FILE = "height_mask.geojson"
HEIGHTMAP_FILE = "heightmap.npz"
REFINE_STAMPS_PATTERN = "refine_stamps_{n}.json"


def _stamps_dir(working_dir: Path) -> Path:
    return working_dir / STAMPS_DIR


def _refine_stamps_files(working_dir: Path) -> list[Path]:
    """Every refine_stamps_N.json present under stamps/, in order (N=1, 2, 3, ...)."""
    files = []
    n = 1
    while (_stamps_dir(working_dir) / REFINE_STAMPS_PATTERN.format(n=n)).exists():
        files.append(_stamps_dir(working_dir) / REFINE_STAMPS_PATTERN.format(n=n))
        n += 1
    return files


def load_all_stamps(working_dir: Path) -> list[Stamp]:
    """
    Reconstruct the full, current stamp list: initial_stamps.json plus
    every refine_stamps_N.json in order, all under stamps/.

    Each refine-terrain pass writes only the stamps *it* added, not a
    cumulative snapshot -- so deleting the highest-numbered
    refine_stamps_N.json is a natural undo of just the most recent
    pass, and every earlier pass's file stays exactly as it was
    (nothing gets rewritten/renumbered by later passes).
    """
    initial_path = _stamps_dir(working_dir) / INITIAL_STAMPS_FILE
    if not initial_path.exists():
        raise StepError(
            f"No {INITIAL_STAMPS_FILE} found under {_stamps_dir(working_dir)}. "
            "Run --step generate-terrain first."
        )

    stamps, _ = load_stamp_file(initial_path)
    for path in _refine_stamps_files(working_dir):
        more_stamps, _ = load_stamp_file(path)
        stamps.extend(more_stamps)
    return stamps


def load_latest_refine_metadata(working_dir: Path) -> dict | None:
    """
    Metadata (step/parameters/timestamp/hotspot_count -- see
    save_stamp_file) from the most recent refine_stamps_N.json, or
    None if no refine pass has run yet. Used to label previews with
    whatever settings actually produced the terrain being looked at.
    """
    files = _refine_stamps_files(working_dir)
    if not files:
        return None
    _, metadata = load_stamp_file(files[-1])
    return metadata


class StepError(RuntimeError):
    """A step couldn't run -- missing prerequisites, bad input, etc."""


# ---------------------------------------------------------------------------
# project.json manifest
# ---------------------------------------------------------------------------

def load_project(working_dir: Path) -> dict:
    path = working_dir / PROJECT_FILE
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_project(working_dir: Path, updates: dict) -> None:
    data = load_project(working_dir)
    data.update(updates)
    path = working_dir / PROJECT_FILE
    with path.open("w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Stamp list <-> JSON (internal artifact format, distinct from writer.py's
# userLayers.json -- this is our own working representation, not PGA's).
#
# Each file is self-contained: whatever step/parameters produced these
# stamps travels with them in the same file, rather than living in a
# separate history in project.json. That matters specifically because
# refine_stamps_N.json files can be deleted individually (undoing one
# pass) -- a separate history would leave orphaned entries referencing
# files that no longer exist, needing its own cleanup logic to stay in
# sync. Keeping metadata and stamps in the same file means deleting the
# file removes its metadata too, automatically, with nothing to orphan.
# ---------------------------------------------------------------------------

def save_stamp_file(
    stamps: list[Stamp], path: Path, step: str, parameters: dict, extra: dict | None = None,
) -> None:
    payload = {
        "step": step,
        "parameters": parameters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(extra or {}),
        "stamps": [dataclasses.asdict(s) for s in stamps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_stamp_file(path: Path) -> tuple[list[Stamp], dict]:
    """Returns (stamps, metadata) -- metadata is everything in the file except "stamps" itself."""
    with path.open() as f:
        payload = json.load(f)
    stamps = [Stamp(**entry) for entry in payload["stamps"]]
    metadata = {k: v for k, v in payload.items() if k != "stamps"}
    return stamps, metadata


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_init(working_dir: Path) -> None:
    """
    Scaffold a fresh working directory: create it (and laz/) if they
    don't exist yet. Safe to run multiple times -- never touches
    anything that's already there, including an existing project.json
    from a prior ingest-laz run.
    """
    working_dir.mkdir(parents=True, exist_ok=True)
    laz_dir = working_dir / "laz"
    laz_dir.mkdir(exist_ok=True)

    print(f"Working directory ready: {working_dir}")
    if any(laz_dir.iterdir()):
        print(f"  {laz_dir} already has files in it.")
    else:
        print(f"  Put your LAZ/LAS tiles in {laz_dir}")

    if (working_dir / PROJECT_FILE).exists():
        print(f"  {PROJECT_FILE} already exists -- this working directory has been used before.")
    else:
        print("  Next: PGA2k_gen.py "
              f"{working_dir} --step ingest-laz --projection <EPSG code>")


def step_visualize(working_dir: Path) -> None:
    """
    Generate every diagnostic preview PNG this pipeline can currently
    produce, against whatever artifacts already exist in working_dir.
    Never a prerequisite for other steps -- purely for inspection (see
    "never behave as a black box").
    """

    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    preview_dir = working_dir / PREVIEW_DIR
    full_cloud = PointCloud.load(pointcloud_path)
    print(f"Loaded {pointcloud_path} ({full_cloud.count:,} points)")

    pointcloud_mtime = pointcloud_path.stat().st_mtime
    latest_lidar = viz.find_latest_preview(preview_dir, PREVIEW_LIDAR)
    latest_lidar_heightmap = viz.find_latest_preview(preview_dir, PREVIEW_LIDAR_HEIGHTMAP)
    lidar_previews_stale = (
        latest_lidar is None or latest_lidar.stat().st_mtime < pointcloud_mtime
        or latest_lidar_heightmap is None or latest_lidar_heightmap.stat().st_mtime < pointcloud_mtime
    )

    if lidar_previews_stale:
        print(f"Writing {PREVIEW_LIDAR} and {PREVIEW_LIDAR_HEIGHTMAP} "
              "(full merged point cloud, not just the course crop)...")
        viz.render_lidar_preview(full_cloud, preview_dir / PREVIEW_LIDAR)
        viz.render_lidar_heightmap(full_cloud, full_cloud.bounds, preview_dir / PREVIEW_LIDAR_HEIGHTMAP)
    else:
        print(f"{PREVIEW_LIDAR} / {PREVIEW_LIDAR_HEIGHTMAP} already up to date with "
              f"{POINTCLOUD_FILE} -- skipping (re-run --step ingest-laz to force a refresh)")

    if not (_stamps_dir(working_dir) / INITIAL_STAMPS_FILE).exists():
        print(f"No {INITIAL_STAMPS_FILE} yet -- run --step generate-terrain for the "
              "hex/stamps/height/error previews. Stopping after the LIDAR previews.")
        return

    stamps = load_all_stamps(working_dir)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    model = TerrainModel(stamps)

    # Label the terrain-related previews with whatever refine-terrain
    # parameters actually produced the latest stamps, if any pass has
    # run -- self-documenting, without cross-referencing a separate
    # log (see terrain/adaptive_refine.py's stamp-file metadata).
    extra_label = None
    mask_grid = None
    latest_refine = load_latest_refine_metadata(working_dir)
    if latest_refine is not None:
        p = latest_refine["parameters"]
        extra_label = (
            f"tol={p['tolerance']} res={p['resolution']} hot={p['min_hotspot_radius_cells']} "
            f"claim={p['claim_radius_fraction']} spread={p['brush_radius_spread_ratio']}"
        )
        if p.get("use_height_mask"):
            buffer_note = f" buffer={p['mask_buffer_px']:.0f}px" if p.get("mask_buffer_px") is not None else ""
            extra_label += f" mask=on{buffer_note}"
            mask_path = working_dir / HEIGHT_MASK_FILE
            if mask_path.exists():
                mask_geometry = load_height_mask(mask_path)
                mask_grid = rasterize_mask(mask_geometry, bounds, p["resolution"])

    print(f"Writing {PREVIEW_HEX}...")
    viz.render_hex_preview(stamps, bounds, preview_dir / PREVIEW_HEX, extra_label=extra_label)
    print(f"Writing {PREVIEW_STAMPS}...")
    viz.render_stamps_preview(stamps, bounds, preview_dir / PREVIEW_STAMPS, extra_label=extra_label)
    print(f"Writing {PREVIEW_HEIGHT}...")
    viz.render_height_preview(model, bounds, preview_dir / PREVIEW_HEIGHT, extra_label=extra_label)

    print(f"Writing {PREVIEW_ERROR} (course-cropped point cloud vs. TerrainModel)...")
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    error_resolution = latest_refine["parameters"]["resolution"] if latest_refine is not None else 200
    viz.render_error_preview(
        model, course_cloud, bounds, preview_dir / PREVIEW_ERROR,
        resolution=error_resolution, extra_label=extra_label, mask=mask_grid,
    )

    print(f"All previews written to {preview_dir}")


def step_ingest_laz(working_dir: Path, projection: int | None) -> None:
    laz_dir = working_dir / "laz"
    if not laz_dir.is_dir():
        raise StepError(f"No laz/ folder found under {working_dir} -- expected {laz_dir}")

    force_crs = None
    if projection is not None:
        try:
            force_crs = pyproj.CRS.from_epsg(projection)
        except pyproj.exceptions.CRSError as e:
            raise StepError(f"--projection {projection} is not a valid EPSG code: {e}") from e
        print(f"Reading LAZ tiles from {laz_dir} (forcing CRS EPSG:{projection})...")
    else:
        print(f"Reading LAZ tiles from {laz_dir} (auto-detecting CRS from LAZ headers)...")

    cloud = load_point_cloud(laz_dir, force_crs=force_crs)
    print(f"  detected CRS: {cloud.crs}")
    if abs(cloud.horizontal_unit_factor - 1.0) > 1e-9:
        print(f"  NOTE: this CRS's native unit is not meters -- detected horizontal "
              f"conversion factor {cloud.horizontal_unit_factor:.6f} to meters, applied.")
    if cloud.vertical_unit_source == "assumed-matches-horizontal" and abs(cloud.vertical_unit_factor - 1.0) > 1e-9:
        print(f"  WARNING: elevation unit could not be read from the CRS directly "
              f"(not a compound CRS) -- ASSUMED to match the horizontal factor "
              f"({cloud.vertical_unit_factor:.6f}). Verify against a known site "
              "elevation if that seems off.")
    print(f"  {cloud.count} points loaded, local bounds {cloud.bounds} (meters)")

    cloud.save(working_dir / POINTCLOUD_FILE)
    print(f"  wrote {POINTCLOUD_FILE}")

    # Rasterize once, here, rather than every consumer (height_fit.py,
    # adaptive_refine.py) separately querying the raw point cloud's
    # KD-tree -- a regular grid supports direct bounding-box index
    # arithmetic, no tree traversal needed at all (same idea as the
    # render() optimization, applied to the "ground truth" side of
    # every error/fit computation instead of just the "predicted" side).
    # Cropped to the course area specifically, matching what those
    # consumers actually operate on.
    course_cloud_for_heightmap = recentered_crop(cloud, size_m=COURSE_SIZE_M)
    print(f"Rasterizing ground heightmap ({DEFAULT_HEIGHTMAP_RESOLUTION}x"
          f"{DEFAULT_HEIGHTMAP_RESOLUTION}, bare-earth points only)...")
    heightmap = rasterize_ground_heightmap(
        course_cloud_for_heightmap,
        BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M),
        resolution=DEFAULT_HEIGHTMAP_RESOLUTION,
    )
    coverage = np.mean(np.isfinite(heightmap))
    save_heightmap(
        heightmap,
        BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M),
        working_dir / HEIGHTMAP_FILE,
    )
    print(f"  wrote {HEIGHTMAP_FILE} ({coverage:.1%} of cells have at least one bare-earth point)")

    # Report the merged extent as a lat/lon bbox, since that's what's
    # needed to manually pull an OSM export before the next step.
    # pyproj's Transformer expects coordinates in the CRS's own native
    # unit, so convert our true-meters coordinates back before feeding
    # them in -- NOT the same as cloud.origin_x/bounds directly, which
    # are already in meters (see ingest.laz_reader module docstring).
    h_factor = cloud.horizontal_unit_factor
    proj_min_x = (cloud.origin_x + cloud.bounds.min_x) / h_factor
    proj_max_x = (cloud.origin_x + cloud.bounds.max_x) / h_factor
    proj_min_z = (cloud.origin_y + cloud.bounds.min_z) / h_factor
    proj_max_z = (cloud.origin_y + cloud.bounds.max_z) / h_factor
    corners_proj = [
        (proj_min_x, proj_min_z), (proj_min_x, proj_max_z),
        (proj_max_x, proj_min_z), (proj_max_x, proj_max_z),
    ]
    to_wgs84 = pyproj.Transformer.from_crs(cloud.crs, "EPSG:4326", always_xy=True)
    corners_ll = [to_wgs84.transform(x, y) for x, y in corners_proj]
    lons = [c[0] for c in corners_ll]
    lats = [c[1] for c in corners_ll]

    print()
    print("Lat/Lon bounding box (for OSM export):")
    print(f"  min_lon={min(lons):.6f}  min_lat={min(lats):.6f}")
    print(f"  max_lon={max(lons):.6f}  max_lat={max(lats):.6f}")
    print()
    print("Download an OSM export covering this box and save it as "
          f"{working_dir / 'map.osm'}, then run --step ingest-osm.")

    save_project(working_dir, {
        "projection_epsg": cloud.crs.to_epsg(),
        "projection_source": "forced" if force_crs is not None else "auto-detected",
        "horizontal_unit_factor": cloud.horizontal_unit_factor,
        "vertical_unit_factor": cloud.vertical_unit_factor,
        "vertical_unit_source": cloud.vertical_unit_source,
        "crs_wkt": cloud.crs.to_wkt(),
        "point_count": cloud.count,
        "merged_bounds_local": dataclasses.asdict(cloud.bounds),
        "origin_x": cloud.origin_x,
        "origin_y": cloud.origin_y,
        "lat_lon_bbox": {
            "min_lon": min(lons), "max_lon": max(lons),
            "min_lat": min(lats), "max_lat": max(lats),
        },
    })


def step_ingest_osm(working_dir: Path, height_mask_buffer_px: float) -> None:
    osm_path = working_dir / "map.osm"
    if not osm_path.exists():
        raise StepError(
            f"No map.osm found at {osm_path}. Run --step ingest-laz first to get "
            "the lat/lon bbox, download an OSM export covering it, and save it there."
        )

    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    print(f"Found {osm_path} ({osm_path.stat().st_size:,} bytes).")

    # OSM features need to land in the *course-cropped* local frame
    # (the same one stamps/terrain use, established by recentered_crop),
    # not the full merged LAZ extent's frame.
    full_cloud = PointCloud.load(pointcloud_path)
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    print("Parsing OSM features into the course's local frame...")
    features = parse_osm_features(
        osm_path, course_cloud.crs, course_cloud.origin_x, course_cloud.origin_y,
        course_cloud.horizontal_unit_factor, bounds=bounds,
    )

    counts: dict[str, int] = {}
    for f in features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        print(f"  {kind}: {n}")

    out_path = working_dir / FEATURES_FILE
    save_features(features, out_path)
    print(f"  wrote {out_path}")

    preview_path = working_dir / PREVIEW_DIR / PREVIEW_OSM
    viz.render_osm_features(features, bounds, preview_path)
    print(f"  wrote {preview_path} (transparent overlay -- composite over the course-cropped "
          "previews [hex/stamps/height/error] in the GUI, doesn't stand alone)")

    # The LIDAR previews (preview_lidar.png / preview_lidar_heightmap.png)
    # render the *full* merged point cloud, in its own local frame --
    # not the course crop's [0, COURSE_SIZE_M] frame the overlay above
    # was built in. Both frames share the same real-world origin, just
    # offset from each other by however far the course crop sits inside
    # the larger merged extent, so a plain coordinate shift (not a
    # reprojection) is enough to correctly place the same features
    # against the full cloud's own bounds.
    full_shift_x = course_cloud.origin_x - full_cloud.origin_x
    full_shift_z = course_cloud.origin_y - full_cloud.origin_y
    full_features = shift_features(features, dx=full_shift_x, dz=full_shift_z)
    full_preview_path = working_dir / PREVIEW_DIR / PREVIEW_OSM_FULL
    viz.render_osm_features(full_features, full_cloud.bounds, full_preview_path)
    print(f"  wrote {full_preview_path} (same overlay, shifted to align with the "
          "LIDAR previews' full-point-cloud frame instead)")

    mask_geometry = build_height_mask(features, buffer_px=height_mask_buffer_px)
    mask_path = working_dir / HEIGHT_MASK_FILE
    save_height_mask(mask_geometry, mask_path)
    if mask_geometry is None:
        print(f"  wrote {mask_path} (no fairway/green/tee/hole features found -- mask is empty, "
              "--use-height-mask on refine-terrain would restrict everything)")
    else:
        print(f"  wrote {mask_path} (fairway + green + tee, plus buffered hole-path corridors, "
              f"then buffered {height_mask_buffer_px} m/px)")

    mask_preview_path = working_dir / PREVIEW_DIR / PREVIEW_MASK
    viz.render_mask_preview(mask_geometry, bounds, mask_preview_path)
    print(f"  wrote {mask_preview_path} (black/white -- multiply-blend over another "
          "course-cropped preview in the GUI's 'Show mask' toggle)")

    save_project(working_dir, {
        "osm_feature_count": len(features), "osm_feature_kinds": counts,
        "height_mask_buffer_px": height_mask_buffer_px,
    })


def step_write_splines(working_dir: Path) -> None:
    """
    Generate PGA surface splines from features.geojson (see splines.py)
    and write them to course/CourseDescription_nodes/surfaceSplines.json.

    Scope: green/tee/fairway/rough/bunker/cartpath/path/building/wood.
    Water and hole are deliberately excluded (see splines.py's module
    docstring) -- neither is handled by this generic writer yet.
    Features marked "ignored" (e.g. a duplicate hole bleeding in from a
    neighboring course) are skipped, not just the ones with an
    unsupported kind.

    This overwrites surfaceSplines.json wholesale -- it's the primary
    generator for these surface types now, not a merge with whatever
    was already there (from the blank course template or prior manual
    edits in the PGA editor).
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    features = load_features(features_path)
    splines = build_surface_splines(features)

    ignored_count = sum(1 for f in features if f.ignored)
    unsupported: dict[str, int] = {}
    for f in features:
        if not f.ignored and feature_to_spline(f) is None:
            unsupported[f.kind] = unsupported.get(f.kind, 0) + 1

    print(f"Generated {len(splines)} splines from {len(features)} features "
          f"({ignored_count} ignored, {sum(unsupported.values())} unsupported kind: {unsupported})")

    nodes_dir = working_dir / "course" / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(f"No {nodes_dir} found under {working_dir}. Run --step ingest-course first.")

    out_path = nodes_dir / "surfaceSplines.json"
    save_surface_splines(splines, out_path)
    print(f"Wrote {out_path}")


def step_generate_terrain(working_dir: Path) -> None:
    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    print(f"Loading {pointcloud_path}...")
    full_cloud = PointCloud.load(pointcloud_path)

    print(f"Cropping to the center {COURSE_SIZE_M:.0f} x {COURSE_SIZE_M:.0f} m...")
    try:
        course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    except LazReadError as e:
        raise StepError(f"Couldn't crop to a {COURSE_SIZE_M:.0f} m course: {e}") from e

    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first.")
    heightmap, _ = load_heightmap(heightmap_path)

    print("Generating hex grid...")
    stamps = generate_hex_grid(bounds)
    print(f"  {len(stamps)} stamps placed")

    print("Fitting stamp heights from the rasterized ground heightmap...")
    fitted = fit_stamp_heights(stamps, heightmap, bounds)
    n_unfitted = sum(1 for s in fitted if s.value == 0.0)
    if n_unfitted:
        print(f"  WARNING: {n_unfitted} stamps had too few nearby heightmap cells and kept "
              "their placeholder value=0.0")

    mean_elevation = float(np.nanmean(heightmap))
    print(f"Prepending a course-wide baseline-flatten stamp at the mean ground "
          f"elevation ({mean_elevation:.2f} m)...")
    baseline_stamp = build_baseline_flatten_stamp(bounds, mean_elevation)
    fitted = [baseline_stamp] + fitted

    out_path = _stamps_dir(working_dir) / INITIAL_STAMPS_FILE
    save_stamp_file(
        fitted, out_path, step="generate-terrain",
        parameters={"course_size_m": COURSE_SIZE_M},
    )
    print(f"  wrote {out_path}")

    save_project(working_dir, {
        "course_origin_x": course_cloud.origin_x,
        "course_origin_y": course_cloud.origin_y,
        "stamp_count": len(fitted),
    })

    print("Refreshing previews...")
    step_visualize(working_dir)


def step_refine_terrain(
    working_dir: Path,
    tolerance: float,
    resolution: int,
    min_hotspot_radius_cells: float,
    max_new_stamps: int | None,
    claim_radius_fraction: float | None,
    brush_radius_spread_ratio: float | None,
    radius_decay_per_pass: float | None,
    use_height_mask: bool | None,
    mask_buffer_px: float | None = None,
) -> None:
    """
    One adaptive refinement pass (see terrain/adaptive_refine.py): find
    contiguous regions of the binned error grid exceeding `tolerance`
    (same grid preview_error.png visualizes), add one stamp per region
    centered and sized on it, write the newly-added stamps (only --
    not the whole cumulative list) to the next refine_stamps_N.json.

    Safe to run repeatedly -- each call reconstructs the full current
    terrain via load_all_stamps() (initial_stamps.json plus every
    prior refine_stamps_N.json) and scores against that, so "run this
    a few times, watch the hotspot count drop" is the expected way to
    iterate (see module docstring's "largest error -> split -> repeat").
    Deleting the highest-numbered refine_stamps_N.json undoes just
    that pass.

    Without radius_decay_per_pass, every pass uses the exact same
    min/max hotspot radius clamps (see adaptive_refine.py's
    DEFAULT_MIN/MAX_HOTSPOT_RADIUS_M) regardless of how many prior
    passes have already run -- confirmed in practice to cause a real
    problem: a second pass over the same course reproduced hotspots at
    nearly the same shape/radius as the first, just at lower error,
    because min_radius=12.5 still floors every stamp up to at least
    that size even where the residual error region is now much smaller
    than that. radius_decay_per_pass > 1.0 shrinks both clamps by
    decay**(pass_number - 1) each successive pass (pass 1 unchanged,
    pass 2 divided by decay, pass 3 by decay^2, ...), so later passes
    are only allowed to add progressively finer detail rather than
    re-covering the same ground at a smaller value. 1.0 disables this
    (every pass uses the same clamps -- the old default behavior).

    claim_radius_fraction / brush_radius_spread_ratio / radius_decay_per_pass
    are feature-flagged via project.json rather than always needing a
    CLI value: pass None here to use whatever was last saved
    (defaulting to the old/off behavior if never set), or an explicit
    value to override for this run and persist it as the new default
    for next time.
    """
    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(
            f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    project = load_project(working_dir)
    if claim_radius_fraction is None:
        claim_radius_fraction = project.get(
            "refine_claim_radius_fraction", DEFAULT_CLAIM_RADIUS_FRACTION
        )
    if brush_radius_spread_ratio is None:
        brush_radius_spread_ratio = project.get(
            "refine_brush_radius_spread_ratio", DEFAULT_BRUSH_RADIUS_SPREAD_RATIO
        )
    if radius_decay_per_pass is None:
        radius_decay_per_pass = project.get(
            "refine_radius_decay_per_pass", DEFAULT_RADIUS_DECAY_PER_PASS
        )
    if use_height_mask is None:
        use_height_mask = project.get("refine_use_height_mask", False)

    stamps = load_all_stamps(working_dir)
    pass_number = len(_refine_stamps_files(working_dir)) + 1
    print(f"  {len(stamps)} stamps (cumulative: initial + {pass_number - 1} prior refine pass(es))")
    print(f"  claim_radius_fraction={claim_radius_fraction}  "
          f"brush_radius_spread_ratio={brush_radius_spread_ratio}  "
          f"radius_decay_per_pass={radius_decay_per_pass} (this is pass {pass_number})  "
          f"use_height_mask={use_height_mask}")

    decay = radius_decay_per_pass ** (pass_number - 1)
    min_radius = DEFAULT_MIN_HOTSPOT_RADIUS_M / decay
    max_radius = DEFAULT_MAX_HOTSPOT_RADIUS_M / decay
    if decay != 1.0:
        print(f"  min/max hotspot radius this pass: {min_radius:.2f} / {max_radius:.2f} m "
              f"(decayed {decay:.2f}x from {DEFAULT_MIN_HOTSPOT_RADIUS_M}/{DEFAULT_MAX_HOTSPOT_RADIUS_M})")

    heights, _ = load_heightmap(heightmap_path)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    mask_grid = None
    if use_height_mask:
        mask_path = working_dir / HEIGHT_MASK_FILE
        if not mask_path.exists():
            raise StepError(
                f"use_height_mask is on but no {HEIGHT_MASK_FILE} found under {working_dir}. "
                "Run --step ingest-osm first."
            )
        mask_geometry = load_height_mask(mask_path)
        mask_grid = rasterize_mask(mask_geometry, bounds, resolution)
        print(f"  height mask covers {mask_grid.mean():.1%} of the course at this resolution")

    if COURSE_SIZE_M % resolution != 0:
        cell_size = COURSE_SIZE_M / resolution
        print(f"  NOTE: resolution={resolution} doesn't evenly divide the {COURSE_SIZE_M:.0f} m course "
              f"({cell_size:.3f} m cells) -- cell boundaries won't land on whole-meter positions "
              "matching the ground heightmap's own 1 px = 1 m grid. Not an error, just imprecise; "
              "an exact divisor (200, 250, 400, 500, 1000, 2000, ...) avoids this.")

    print(f"Scanning the error grid ({resolution}x{resolution}, tolerance={tolerance} m)...")
    refined, hotspots = refine_stamps(
        stamps, heights, bounds, tolerance=tolerance,
        resolution=resolution, min_hotspot_radius_cells=min_hotspot_radius_cells,
        min_radius=min_radius, max_radius=max_radius,
        claim_radius_fraction=claim_radius_fraction,
        brush_radius_spread_ratio=brush_radius_spread_ratio,
        max_new_stamps=max_new_stamps,
        mask=mask_grid,
    )

    new_stamps = refined[len(stamps):]
    if hotspots:
        worst = hotspots[0]
        print(f"  {len(hotspots)} hotspots over tolerance (worst: {worst.peak_error:.3f} m "
              f"at ({worst.x:.1f}, {worst.z:.1f}), {worst.n_cells} cells)")
    else:
        print("  0 hotspots over tolerance -- nothing to refine, terrain already meets it")

    parameters = {
        "tolerance": tolerance,
        "resolution": resolution,
        "min_hotspot_radius_cells": min_hotspot_radius_cells,
        "max_new_stamps": max_new_stamps,
        "claim_radius_fraction": claim_radius_fraction,
        "brush_radius_spread_ratio": brush_radius_spread_ratio,
        "use_height_mask": use_height_mask,
        "mask_buffer_px": mask_buffer_px,
    }

    if new_stamps:
        next_n = len(_refine_stamps_files(working_dir)) + 1
        out_path = _stamps_dir(working_dir) / REFINE_STAMPS_PATTERN.format(n=next_n)
        save_stamp_file(
            new_stamps, out_path, step="refine-terrain", parameters=parameters,
            extra={"hotspot_count": len(hotspots)},
        )
        print(f"  wrote {out_path} ({len(new_stamps)} new stamps; "
              f"{len(stamps)} -> {len(refined)} total)")
    else:
        print(f"  nothing written ({len(stamps)} total, unchanged)")

    save_project(working_dir, {
        "last_refine_tolerance_m": tolerance,
        "last_refine_hotspot_count": len(hotspots),
        "last_refine_added_count": len(new_stamps),
        "total_stamp_count": len(refined),
        "refine_claim_radius_fraction": claim_radius_fraction,
        "refine_brush_radius_spread_ratio": brush_radius_spread_ratio,
        "refine_radius_decay_per_pass": radius_decay_per_pass,
        "refine_use_height_mask": use_height_mask,
    })

    print("Refreshing previews (parameters used above are now the header on the terrain previews)...")
    step_visualize(working_dir)


def _set_course_name_in_file(path: Path, course_name: str, key: str) -> None:
    """
    Set `key` to `course_name` in the JSON object at `path`, preserving
    every other key already there (same preserve-everything-else
    pattern write_user_layers.py uses for userLayers.json's sibling
    keys). No-ops with a note if `path` doesn't exist yet, rather than
    creating a file whose overall structure we don't actually know.
    """
    if not path.exists():
        print(f"NOTE: course_name is set ('{course_name}') but {path} doesn't exist yet "
              "-- run --step ingest-course first if you want the name applied there.")
        return

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data[key] = course_name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Set course name to '{course_name}' in {path}")


def step_output_terrain(working_dir: Path) -> None:
    course_dir = working_dir / "course"

    print(f"Loading stamps from {working_dir} (initial + all refine passes)...")
    stamps = load_all_stamps(working_dir)
    print(f"  {len(stamps)} stamps")

    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    heights = TerrainModel(stamps).render(resolution=200, bounds=bounds)
    true_min, true_max = float(heights.min()), float(heights.max())
    print(f"Normalizing heights: actual resolved range [{true_min:.3f}, {true_max:.3f}] m "
          f"-> shifting by {-true_min:.3f} m so minimum lands at 0")
    try:
        stamps = normalize_stamp_heights(stamps, bounds)
    except ValueError as e:
        raise StepError(str(e)) from e

    nodes_dir = course_dir / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(
            f"{nodes_dir} doesn't exist. Run --step ingest-course to extract a blank "
            f"starting .course into {course_dir} first."
        )

    out_path = nodes_dir / "userLayers.json"
    write_user_layers(stamps, out_path)
    print(f"Wrote {out_path}")

    # If a course name has been set (see the GUI's "Course name" field /
    # project.json), write it into both CourseDescription.json and
    # CourseMetadata.json -- which one the game actually reads from
    # depends on version: confirmed CourseMetadata.json for 2019, with
    # CourseDescription.json believed to be what later versions (2K21+)
    # use instead. Writing both covers either case rather than guessing
    # which version a given course targets.
    project = load_project(working_dir)
    course_name = project.get("course_name")
    if course_name:
        _set_course_name_in_file(course_dir / "CourseDescription.json", course_name, "name")
        # ASSUMPTION: using the same "name" key here as CourseDescription.json --
        # unconfirmed for CourseMetadata.json specifically. If the game doesn't
        # pick up the name after this, that key name is the first thing to check.
        _set_course_name_in_file(course_dir / "CourseMetadata.json", course_name, "name")

    save_project(working_dir, {
        "output_height_shift_m": -true_min,
        "output_height_range_m": true_max - true_min,
    })


def step_ingest_course(working_dir: Path, course_file: Path) -> None:
    """
    Extract a .course file into working_dir/course via util/course_extract.py,
    invoked as a subprocess -- same "one place behavior lives" reasoning as
    the GUI's own subprocess-per-step design (see PGA2k_gen_gui.py).
    """
    if not course_file.exists():
        raise StepError(f"Course file not found: {course_file}")

    script = SCRIPT_DIR / "util" / "course_extract.py"
    if not script.exists():
        raise StepError(f"course_extract.py not found at {script}")

    course_dir = working_dir / "course"
    course_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {course_file} -> {course_dir} ...")
    result = subprocess.run(
        [sys.executable, str(script), str(course_file), str(course_dir)],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise StepError(f"course_extract.py failed (exit {result.returncode})")

    print(f"Extracted to {course_dir}")
    save_project(working_dir, {"source_course_file": str(course_file)})


def step_repack(working_dir: Path, filename: str) -> None:
    """
    Repack working_dir/course into a .course file via util/course_repack.py,
    invoked as a subprocess (see step_ingest_course).
    """
    course_dir = working_dir / "course"
    if not course_dir.is_dir():
        raise StepError(f"No course/ folder under {working_dir}. Run --step ingest-course first.")

    script = SCRIPT_DIR / "util" / "course_repack.py"
    if not script.exists():
        raise StepError(f"course_repack.py not found at {script}")

    filename = filename.strip()
    if not filename:
        raise StepError("Repack filename can't be empty.")
    if filename.lower().endswith(".course"):
        filename = filename[: -len(".course")]

    out_path = working_dir / f"{filename}.course"

    print(f"Repacking {course_dir} -> {out_path} ...")
    result = subprocess.run(
        [sys.executable, str(script), str(course_dir), str(out_path)],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise StepError(f"course_repack.py failed (exit {result.returncode})")

    print(f"Wrote {out_path}")
    save_project(working_dir, {"repack_filename": filename})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEPS = {
    "init": step_init,
    "ingest-laz": step_ingest_laz,
    "ingest-osm": step_ingest_osm,
    "ingest-course": step_ingest_course,
    "generate-terrain": step_generate_terrain,
    "refine-terrain": step_refine_terrain,
    "output-terrain": step_output_terrain,
    "write-splines": step_write_splines,
    "repack": step_repack,
    "visualize": step_visualize,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PGA2K terrain compiler CLI")
    parser.add_argument("working_dir", type=Path, help="Project working directory")
    parser.add_argument("--step", default="init", choices=sorted(STEPS.keys()),
                         help="Pipeline step to run (default: init)")
    parser.add_argument("--projection", type=int, default=None,
                         help="EPSG code to force for ingest-laz (optional -- "
                              "auto-detected from LAZ headers if omitted)")
    parser.add_argument("--error-tolerance", type=float, default=2.0,
                         help="refine-terrain: |predicted - actual| (m) above which a grid "
                              "cell counts as a hotspot (default: 2.0)")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                         help=f"refine-terrain: error grid resolution, same grid "
                              f"preview_error.png uses (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--min-hotspot-radius-cells", type=float, default=DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
                         help=f"refine-terrain: drop hotspot regions smaller than this many "
                              f"cells (pre-clamp), likely noise not a feature (default: {DEFAULT_MIN_HOTSPOT_RADIUS_CELLS})")
    parser.add_argument("--claim-radius-fraction", type=float, default=None,
                         help="refine-terrain: fraction of placement radius to mark claimed "
                              f"(< 1.0 lets neighboring stamps overlap; default: use whatever's "
                              f"saved in project.json, or {DEFAULT_CLAIM_RADIUS_FRACTION} if never set)")
    parser.add_argument("--brush-radius-spread-ratio", type=float, default=None,
                         help="refine-terrain: radius multiplier per brush rank "
                              "(spread_ratio ** rank, ranks 0..3 for types 8/9/10/54); "
                              "1.0 disables it (default: use whatever's saved in project.json, "
                              f"or {DEFAULT_BRUSH_RADIUS_SPREAD_RATIO} if never set)")
    parser.add_argument("--radius-decay-per-pass", type=float, default=None,
                         help="refine-terrain: shrink min/max hotspot radius by this factor per "
                              "prior refine pass already run (pass 2 divided by this once, pass 3 "
                              "twice, ...), so later passes add progressively finer detail instead "
                              "of re-covering the same ground at lower error; 1.0 disables it "
                              "(default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RADIUS_DECAY_PER_PASS} if never set)")
    parser.add_argument("--use-height-mask", action=argparse.BooleanOptionalAction, default=None,
                         help="refine-terrain: restrict hotspot placement to inside height_mask.geojson "
                              "(fairway/green, see ingest-osm) -- everything outside is treated like "
                              "no-data, never becoming a hotspot (default: use whatever's saved in "
                              "project.json, or off if never set)")
    parser.add_argument("--mask-buffer-px", type=float, default=None,
                         help="refine-terrain: record-keeping only -- the buffer distance (m) the "
                              "current height_mask.geojson was built with, saved alongside this pass's "
                              "other parameters so it shows up in preview titles and stamp-file "
                              "metadata. Doesn't affect the mask itself (already baked into "
                              "height_mask.geojson) or any computation here.")
    parser.add_argument("--height-mask-buffer-px", type=float, default=DEFAULT_HEIGHT_MASK_BUFFER_PX,
                         help="ingest-osm: buffer (grow) the merged fairway+green outline by this many "
                              "pixels before rasterizing -- 1 pixel = 1 m, since the course is exactly "
                              f"2000x2000 m (default: {DEFAULT_HEIGHT_MASK_BUFFER_PX})")
    parser.add_argument("--max-new-stamps", type=int, default=None,
                         help="refine-terrain: cap on new detail stamps per pass (default: no cap)")
    parser.add_argument("--course-file", type=Path, default=None,
                         help="ingest-course: path to the .course file to extract")
    parser.add_argument("--repack-filename", type=str, default=None,
                         help="repack: output filename (without .course extension)")
    args = parser.parse_args(argv)

    working_dir: Path = args.working_dir

    if args.step == "init":
        step_init(working_dir)
        return 0

    if not working_dir.is_dir():
        print(f"error: {working_dir} is not a directory -- run --step init first", file=sys.stderr)
        return 1

    try:
        if args.step == "ingest-laz":
            step_ingest_laz(working_dir, args.projection)
        elif args.step == "ingest-osm":
            step_ingest_osm(working_dir, args.height_mask_buffer_px)
        elif args.step == "ingest-course":
            if args.course_file is None:
                print("error: --step ingest-course requires --course-file <path>", file=sys.stderr)
                return 1
            step_ingest_course(working_dir, args.course_file)
        elif args.step == "generate-terrain":
            step_generate_terrain(working_dir)
        elif args.step == "refine-terrain":
            step_refine_terrain(working_dir, args.error_tolerance, args.resolution,
                                 args.min_hotspot_radius_cells, args.max_new_stamps,
                                 args.claim_radius_fraction, args.brush_radius_spread_ratio,
                                 args.radius_decay_per_pass, args.use_height_mask, args.mask_buffer_px)
        elif args.step == "output-terrain":
            step_output_terrain(working_dir)
        elif args.step == "write-splines":
            step_write_splines(working_dir)
        elif args.step == "repack":
            if not args.repack_filename:
                print("error: --step repack requires --repack-filename <name>", file=sys.stderr)
                return 1
            step_repack(working_dir, args.repack_filename)
        elif args.step == "visualize":
            step_visualize(working_dir)
    except StepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
