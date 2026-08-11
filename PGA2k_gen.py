#!/usr/bin/env python3
"""
PGA2k_gen.py

CLI orchestrator for the terrain compiler. Operates on a working
directory, running one pipeline step at a time:

    PGA2k_gen.py <working_dir>                       (same as --step init)
    PGA2k_gen.py <working_dir> --step init
    PGA2k_gen.py <working_dir> --step ingest-laz [--projection <EPSG>] [--no-fill-heightmap-gaps]
    PGA2k_gen.py <working_dir> --step ingest-osm
    PGA2k_gen.py <working_dir> --step ingest-course --course-file <path>
    PGA2k_gen.py <working_dir> --step dig-water [--dig-depth M] [--dig-buffer M]
    PGA2k_gen.py <working_dir> --step generate-terrain
    PGA2k_gen.py <working_dir> --step refine-terrain [--error-tolerance M] [--resolution N]
                                 [--method adaptive|scatter] [--rad-m M]
    PGA2k_gen.py <working_dir> --step output-terrain
    PGA2k_gen.py <working_dir> --step generate-trees [--detect-lidar-trees]
    PGA2k_gen.py <working_dir> --step write-objects [--game-version <2019|2021|2023|2025>]
                                 [--theme <id-or-name>] [--tree-variety]              (2019)
                                 [--tree-asset-path <path>]...
                                 [--tree-type-asset-path <TAG=path>]... [--stake-asset-path <path>]  (2021+)
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
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent

from constants import (
    COURSE_SIZE_M, PREVIEW_COMPOSITE, PREVIEW_ERROR, PREVIEW_HEIGHT, PREVIEW_HEX,
    PREVIEW_LIDAR, PREVIEW_LIDAR_GROUND, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_MASK, PREVIEW_OSM,
    PREVIEW_OSM_FULL, PREVIEW_STAMPS,
    POINTCLOUD_FILE, PREVIEW_DIR, PROJECT_FILE, STAMPS_DIR,
)
import viz.visualize as viz
from ingest.laz_reader import LazReadError, PointCloud, load_point_cloud, recentered_crop
from ingest.heightmap import (
    DEFAULT_FILL_MAX_ITERATIONS_PER_LEVEL, DEFAULT_FILL_MIN_COARSE_RESOLUTION,
    DEFAULT_FILL_TOLERANCE, DEFAULT_HEIGHTMAP_RESOLUTION,
    dig_water_into_heightmap, fill_heightmap_gaps, load_heightmap, rasterize_ground_heightmap,
    save_heightmap,
)
from ingest.tree_detection import (
    DEFAULT_MIN_HEIGHT_M as DEFAULT_LIDAR_TREE_MIN_HEIGHT_M,
    detect_trees_from_lidar, rasterize_canopy_heightmap_with_fallback,
)
from ingest.osm import (
    DEFAULT_HEIGHT_MASK_BUFFER_PX, DEFAULT_HOLE_CORRIDOR_BUFFER_PX, build_height_mask, crop_features,
    load_features, load_height_mask, parse_osm_features, rasterize_mask, save_features,
    save_height_mask, shift_features,
)
from course_output.splines import (
    build_registration_mark_splines, build_surface_splines, feature_to_spline, save_surface_splines,
)
from course_output.holes import build_holes, save_holes
from course_output.objects import (
    DEFAULT_GAME_VERSION, GAME_VERSIONS, IMPLEMENTED_GAME_VERSIONS, THEMES_V2019, TREE_TYPE_TAG,
    apply_area_tree_type_hints, build_building_stake_objects_v2021, build_tree_objects_v2019,
    build_tree_objects_v2021, lidar_trees_to_tagged, load_object_list, object_counts,
    parse_osm_trees, save_object_list, save_placed_objects,
)
from terrain.adaptive_refine import (
    DEFAULT_CLAIM_RADIUS_FRACTION,
    DEFAULT_BRUSH_RADIUS_SPREAD_RATIO,
    DEFAULT_MAX_HOTSPOT_RADIUS_M,
    DEFAULT_MAX_PLANAR_RMS,
    DEFAULT_MIN_HOTSPOT_RADIUS_CELLS,
    DEFAULT_MIN_HOTSPOT_RADIUS_M,
    DEFAULT_MODEL_REBUILD_INTERVAL,
    DEFAULT_PLANAR_SHRINK_FACTOR,
    DEFAULT_RAD_M,
    DEFAULT_RESOLUTION,
    DEFAULT_VARIATION_CONTRAST_GAMMA,
    DEFAULT_SUBPIXEL_JITTER_FRACTION,
    refine_stamps,
    scatter_refine_stamps,
)
from terrain.bounding_box import BoundingBox
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import HEX_LATTICE_PITCH_M, generate_hex_grid
from terrain.contour_layers import (
    DEFAULT_BAND_SPACING_M,
    DEFAULT_FILL_BRUSH,
    DEFAULT_MIN_RADIUS_M,
    DEFAULT_MAX_RADIUS_M,
    DEFAULT_RADIUS_STEP_RATIO,
    DEFAULT_EDGE_DISTANCE_M,
    DEFAULT_SMOOTHING_BRUSH,
    DEFAULT_SMOOTHING_MIN_RADIUS_M,
    DEFAULT_CRUMB_SCATTER_MULTIPLIER,
    DEFAULT_SMOOTH_CLAIM_FRACTION,
    DEFAULT_SWEET_SPOT_STAMP_RATIO,
    DEFAULT_SWEET_SPOT_SAMPLE_BANDS,
    DEFAULT_SWEET_SPOT_SEEDS,
    DEFAULT_SWEET_SPOT_MAX_CANDIDATES,
    DEFAULT_SWEET_SPOT_TIME_BUDGET_S,
    DEFAULT_RANDOM_SEED,
    DEFAULT_DENOISE_PX,
    generate_contour_layers,
)
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel
from course_output.userLayers import (
    build_baseline_flatten_stamp, build_registration_mark_stamps, normalize_stamp_heights,
    write_user_layers,
)
from course_output.water import build_water_objects

INITIAL_STAMPS_FILE = "initial_stamps.json"
FEATURES_FILE = "features.geojson"
HEIGHT_MASK_FILE = "height_mask.geojson"
HEIGHTMAP_FILE = "heightmap.npz"
REFINE_STAMPS_PATTERN = "refine_stamps_{n}.json"
PLACED_OBJECTS_FILE = "placedObjects2.json"
OBJECT_LIST_FILE = "object_list.json"

DEFAULT_DIG_WATER_DEPTH_M = 3.0
DEFAULT_DIG_WATER_BUFFER_M = 1.0


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
# Stamp list <-> JSON (internal artifact format, distinct from userLayers.py's
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


def step_visualize(
    working_dir: Path, overwrite_current_version: bool = False, error_resolution: int | None = None,
) -> None:
    """
    Generate every diagnostic preview PNG this pipeline can currently
    produce, against whatever artifacts already exist in working_dir.
    Never a prerequisite for other steps -- purely for inspection (see
    "never behave as a black box").

    error_resolution overrides preview_error.png's own grid resolution
    directly. Left at None (default), it falls back to whatever refine-
    terrain last used (letting the error preview match the resolution
    refine actually tuned against) -- but that inherits a completely
    unrelated step's own setting, and silently drops to a hardcoded 200
    if refine-terrain hasn't run yet at all, which is far too coarse to
    localize a specific small feature (confirmed: at RES~1000, a 200x200
    error grid averages ~5x5 real cells into one, hiding exactly the
    kind of localized error a generate-terrain contour-method run needs
    to debug before refine-terrain has even run once). Pass this
    explicitly to decouple preview_error.png from refine-terrain
    entirely and control it directly.

    overwrite_current_version distinguishes this function's two kinds
    of caller. generate-terrain/refine-terrain call this automatically
    right after producing genuinely new stamp data -- that should keep
    appending a new preview version, matching the new stamp version
    they just created (the default, False). But a manual, standalone
    --step visualize / GUI "Visualize" click doesn't change stamps at
    all -- appending yet another preview version there would leave a
    "phantom" version with no corresponding stamp file, which broke
    Undo: it independently finds the latest stamp file and the latest
    preview of each kind, so a mismatched extra preview version meant
    undoing the actual latest refine pass left the *previous* pass's
    preview on screen instead of reverting to it, silently looking
    like undo had done nothing. Passing True here instead overwrites
    the current latest version in place (delete-then-rerender, so the
    same version number gets reused, not incremented), keeping preview
    version aligned with stamp version the way Undo assumes.

    This also forces the LIDAR previews to regenerate unconditionally,
    bypassing their own pointcloud-mtime staleness check -- a manual
    "Visualize" is an explicit refresh-everything action, and that
    check exists to avoid redundant work on the frequent, automatic
    calls above, not to second-guess an explicit one.
    """

    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    preview_dir = working_dir / PREVIEW_DIR
    full_cloud = PointCloud.load(pointcloud_path)
    print(f"Loaded {pointcloud_path} ({full_cloud.count:,} points)")

    def _overwrite_latest(kind: str) -> None:
        latest = viz.find_latest_preview(preview_dir, kind)
        if latest is not None:
            latest.unlink()

    pointcloud_mtime = pointcloud_path.stat().st_mtime
    latest_lidar = viz.find_latest_preview(preview_dir, PREVIEW_LIDAR)
    latest_lidar_heightmap = viz.find_latest_preview(preview_dir, PREVIEW_LIDAR_HEIGHTMAP)
    lidar_previews_stale = (
        overwrite_current_version
        or latest_lidar is None or latest_lidar.stat().st_mtime < pointcloud_mtime
        or latest_lidar_heightmap is None or latest_lidar_heightmap.stat().st_mtime < pointcloud_mtime
    )

    if lidar_previews_stale:
        print(f"Writing {PREVIEW_LIDAR} and {PREVIEW_LIDAR_HEIGHTMAP} "
              "(full merged point cloud, not just the course crop)...")
        if overwrite_current_version:
            _overwrite_latest(PREVIEW_LIDAR)
            _overwrite_latest(PREVIEW_LIDAR_HEIGHTMAP)
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
            f"method={p.get('method', 'adaptive')} rad={p.get('rad_m', 'n/a')} "
            f"tol={p['tolerance']} res={p['resolution']} hot={p['min_hotspot_radius_cells']} "
            f"claim={p['claim_radius_fraction']} spread={p['brush_radius_spread_ratio']}"
        )
        if p.get("max_planar_rms") is not None:
            extra_label += f" planar_rms={p['max_planar_rms']} shrink={p.get('planar_shrink_factor')}"
        if p.get("use_height_mask"):
            buffer_note = f" buffer={p['mask_buffer_px']:.0f}px" if p.get("mask_buffer_px") is not None else ""
            extra_label += f" mask=on{buffer_note}"
            mask_path = working_dir / HEIGHT_MASK_FILE
            if mask_path.exists():
                mask_geometry = load_height_mask(mask_path)
                mask_grid = rasterize_mask(mask_geometry, bounds, p["resolution"])

    if overwrite_current_version:
        _overwrite_latest(PREVIEW_HEX)
    print(f"Writing {PREVIEW_HEX}...")
    viz.render_hex_preview(stamps, bounds, preview_dir / PREVIEW_HEX, extra_label=extra_label)

    if overwrite_current_version:
        _overwrite_latest(PREVIEW_STAMPS)
    print(f"Writing {PREVIEW_STAMPS}...")
    viz.render_stamps_preview(stamps, bounds, preview_dir / PREVIEW_STAMPS, extra_label=extra_label)

    if overwrite_current_version:
        _overwrite_latest(PREVIEW_HEIGHT)
        _overwrite_latest(PREVIEW_LIDAR_GROUND)
    # Shared color scale between predicted height and actual ground-only
    # LIDAR height, so the two are directly comparable at a glance when
    # flipped back and forth -- each auto-scaling to its own independent
    # range could make an identical-looking pair appear different, or a
    # real discrepancy look subtle, purely from the color mapping.
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    heightmap_path = working_dir / HEIGHTMAP_FILE
    ground_heights = load_heightmap(heightmap_path)[0] if heightmap_path.exists() else None
    model_grid_for_range = model.render(resolution=400, bounds=bounds)
    model_min, model_max = float(np.nanmin(model_grid_for_range)), float(np.nanmax(model_grid_for_range))

    # Normalize height/ground-lidar/composite all the same way before
    # computing the shared color scale -- same shift userLayers.py's
    # normalize_stamp_heights applies for the real export (so the
    # minimum resolved height lands at 0), applied here to all 3
    # terrain-comparison previews' own data too, not just the
    # composite one. Without this, preview_height.png/
    # preview_lidar_ground.png show raw, un-normalized real-world
    # elevation (e.g. 268-393m) while the composite preview (which
    # DOES need to normalize, to match what the real 16-bit export
    # actually operates on) ends up in a completely different range --
    # confirmed directly as a real bug: a real course's composite
    # canvas correctly computed 0-124m after normalizing, but the
    # shared color scale was still 268-393m from the un-normalized
    # model, making the (correctly computed!) composite look like a
    # solid, washed-out color with no visible variation at all.
    shift = -model_min
    normalized_stamps = normalize_stamp_heights(stamps, bounds)
    normalized_model = TerrainModel(normalized_stamps)
    model_min, model_max = model_min + shift, model_max + shift
    shifted_ground_heights = ground_heights + shift if ground_heights is not None else None

    if shifted_ground_heights is not None and np.isfinite(shifted_ground_heights).any():
        shared_vmin = min(model_min, float(np.nanmin(shifted_ground_heights)))
        shared_vmax = max(model_max, float(np.nanmax(shifted_ground_heights)))
    else:
        shared_vmin, shared_vmax = model_min, model_max

    print(f"Writing {PREVIEW_HEIGHT}...")
    viz.render_height_preview(
        normalized_model, bounds, preview_dir / PREVIEW_HEIGHT, extra_label=extra_label,
        vmin=shared_vmin, vmax=shared_vmax,
    )

    if shifted_ground_heights is not None:
        print(f"Writing {PREVIEW_LIDAR_GROUND} (from the saved, gap-filled {HEIGHTMAP_FILE} -- for "
              f"comparing against {PREVIEW_HEIGHT})...")
        viz.render_ground_lidar_preview(
            shifted_ground_heights, bounds, preview_dir / PREVIEW_LIDAR_GROUND, extra_label=extra_label,
            vmin=shared_vmin, vmax=shared_vmax,
        )
    else:
        print(f"  No {HEIGHTMAP_FILE} found -- skipping {PREVIEW_LIDAR_GROUND} (run --step ingest-laz first).")

    # Composite preview (real brush PNG compositing, see
    # composite_render.py) is manual/opt-in ONLY -- never generated by
    # the automatic auto-visualize inside generate-terrain/refine-terrain,
    # only here, when this is the explicit, standalone Visualize step
    # (same treatment as the LIDAR previews' own force-refresh).
    if overwrite_current_version:
        _overwrite_latest(PREVIEW_COMPOSITE)
        print(f"Writing {PREVIEW_COMPOSITE} (real brush PNG compositing, cross-check against "
              f"{PREVIEW_HEIGHT})...")
        try:
            viz.render_composite_preview(
                normalized_stamps, bounds, preview_dir / PREVIEW_COMPOSITE, extra_label=extra_label,
                vmin=shared_vmin, vmax=shared_vmax,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  skipped -- {e}")

    if overwrite_current_version:
        _overwrite_latest(PREVIEW_ERROR)
    print(f"Writing {PREVIEW_ERROR} (course-cropped point cloud vs. TerrainModel)...")
    if error_resolution is None:
        error_resolution = latest_refine["parameters"]["resolution"] if latest_refine is not None else 200
    error_stats = viz.render_error_preview(
        model, course_cloud, bounds, preview_dir / PREVIEW_ERROR,
        resolution=error_resolution, extra_label=extra_label, mask=mask_grid,
    )
    print(f"  RMS={error_stats['rms']:.2f} m, bias={error_stats['bias']:+.2f} m "
          "(bias = mean(predicted - actual): positive means the map sits above the real LIDAR "
          "on average, negative means below)")
    if error_stats["masked_rms"] is not None:
        print(f"  masked area: RMS={error_stats['masked_rms']:.2f} m, "
              f"bias={error_stats['masked_bias']:+.2f} m")

    print(f"All previews written to {preview_dir}")


def step_ingest_laz(
    working_dir: Path, projection: int | None, fill_heightmap: bool = True,
) -> None:
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
    # float(), not the raw numpy scalar np.mean() returns -- otherwise
    # `coverage < 1.0` below is numpy.bool_, not a real Python bool.
    # json.dump can't serialize numpy.bool_ at all (a real, confirmed
    # failure: it broke save_project's "heightmap_gaps_filled" entry,
    # since `fill_heightmap and coverage < 1.0`'s `and` returns that
    # second operand -- the numpy.bool_ -- completely unconverted
    # whenever fill_heightmap is truthy).
    coverage = float(np.mean(np.isfinite(heightmap)))
    print(f"  {coverage:.1%} of cells have at least one bare-earth point "
          f"({(1 - coverage):.1%} gap -- water, buildings, other no-data areas)")

    if fill_heightmap and coverage < 1.0:
        print("  Filling gaps via harmonic (Laplace) inpainting -- iterative "
              "neighbor-average relaxation, not a single-pass flood-fill (see "
              "ingest/heightmap.py's fill_heightmap_gaps for why that "
              "distinction matters). Water and buildings get the same "
              "treatment: water is a separately-placed, sized plane object in "
              "the game, so what's needed here is a plausible recessed-basin "
              "shape under it, not a flat constant fill.")
        heightmap = fill_heightmap_gaps(
            heightmap,
            BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M),
        )
        print(f"  filled -- {np.mean(np.isfinite(heightmap)):.1%} of cells now valid")
    elif not fill_heightmap and coverage < 1.0:
        print("  --no-fill-heightmap-gaps set -- leaving gaps as NaN "
              "(excluded from error scoring/fitting downstream, as before).")

    save_heightmap(
        heightmap,
        BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M),
        working_dir / HEIGHTMAP_FILE,
    )
    print(f"  wrote {HEIGHTMAP_FILE}")

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
        "heightmap_gaps_filled": fill_heightmap and coverage < 1.0,
        "heightmap_raw_coverage": float(coverage),
        "lat_lon_bbox": {
            "min_lon": min(lons), "max_lon": max(lons),
            "min_lat": min(lats), "max_lat": max(lats),
        },
    })


def step_ingest_osm(
    working_dir: Path, height_mask_buffer_px: float,
    hole_corridor_buffer_px: float = DEFAULT_HOLE_CORRIDOR_BUFFER_PX,
) -> None:
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

    # features.geojson is now stored in the FULL merged point cloud's
    # frame, uncropped -- not the course-cropped [0, COURSE_SIZE_M]
    # frame stamps/terrain use. See parse_osm_features's docstring:
    # this is so a future manually-repositioned course crop can just
    # re-crop from this same stored set, not need OSM re-parsed from
    # scratch. Cropping happens later, at the point of use, via
    # crop_features.
    full_cloud = PointCloud.load(pointcloud_path)
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    print("Parsing OSM features into the full point cloud's local frame...")
    features = parse_osm_features(
        osm_path, full_cloud.crs, full_cloud.origin_x, full_cloud.origin_y,
        full_cloud.horizontal_unit_factor, bounds=full_cloud.bounds,
    )

    counts: dict[str, int] = {}
    for f in features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        print(f"  {kind}: {n}")

    out_path = working_dir / FEATURES_FILE
    save_features(features, out_path)
    print(f"  wrote {out_path} (full point cloud frame, uncropped)")

    # The course crop's own (0, 0), expressed in the full cloud's
    # frame -- both this shift (full-frame position of the course
    # crop's origin) and its negation (course-frame position of a
    # full-frame point) are needed below.
    course_origin_in_full_x = course_cloud.origin_x - full_cloud.origin_x
    course_origin_in_full_z = course_cloud.origin_y - full_cloud.origin_y

    course_features = shift_features(features, dx=-course_origin_in_full_x, dz=-course_origin_in_full_z)
    course_features = crop_features(course_features, course_bounds)
    preview_path = working_dir / PREVIEW_DIR / PREVIEW_OSM
    viz.render_osm_features(course_features, course_bounds, preview_path)
    print(f"  wrote {preview_path} (transparent overlay -- composite over the course-cropped "
          "previews [hex/stamps/height/error] in the GUI, doesn't stand alone)")

    # Unlike the course-cropped overlay above, this one is deliberately
    # NOT cropped -- it's composited over the *full* LIDAR previews,
    # where seeing OSM detail beyond the current course crop is exactly
    # the point (e.g. deciding where a future manually-repositioned
    # crop should actually go). crop_box draws the current [0, 2000]
    # crop's own position as a visible rectangle on top, so it's clear
    # where the boundary sits relative to that detail -- currently
    # always centered on the point cloud (see recentered_crop), but
    # this will show a manually-chosen position just as well once that
    # exists.
    full_preview_path = working_dir / PREVIEW_DIR / PREVIEW_OSM_FULL
    course_crop_box_in_full = BoundingBox(
        min_x=course_origin_in_full_x, min_z=course_origin_in_full_z,
        max_x=course_origin_in_full_x + COURSE_SIZE_M, max_z=course_origin_in_full_z + COURSE_SIZE_M,
    )
    viz.render_osm_features(features, full_cloud.bounds, full_preview_path, crop_box=course_crop_box_in_full)
    print(f"  wrote {full_preview_path} (same features, uncropped, in the LIDAR previews' "
          "full-point-cloud frame instead -- plus the current course crop's own position)")

    mask_geometry = build_height_mask(
        course_features, buffer_px=height_mask_buffer_px,
        hole_corridor_buffer_px=hole_corridor_buffer_px,
    )
    mask_path = working_dir / HEIGHT_MASK_FILE
    save_height_mask(mask_geometry, mask_path)
    if mask_geometry is None:
        print(f"  wrote {mask_path} (no not-excluded features found -- mask is empty, "
              "--use-height-mask on refine-terrain would restrict everything)")
    else:
        print(f"  wrote {mask_path} (every feature with mask=False, i.e. NOT excluded -- defaults to "
              f"fairway/green/tee/hole, individually overridable per-feature in the GUI's Splines tab -- "
              f"then buffered {height_mask_buffer_px} m/px, with hole routing centerlines corridor-"
              f"buffered by {hole_corridor_buffer_px} m/px first)")

    mask_preview_path = working_dir / PREVIEW_DIR / PREVIEW_MASK
    viz.render_mask_preview(mask_geometry, course_bounds, mask_preview_path)
    print(f"  wrote {mask_preview_path} (black/white -- multiply-blend over another "
          "course-cropped preview in the GUI's 'Show mask' toggle)")

    save_project(working_dir, {
        "osm_feature_count": len(features), "osm_feature_kinds": counts,
        "height_mask_buffer_px": height_mask_buffer_px,
        "hole_corridor_buffer_px": hole_corridor_buffer_px,
        # The course crop's own origin, expressed in the full point
        # cloud's frame -- features.geojson is stored in that full
        # frame (see parse_osm_features), so any step that needs the
        # course-cropped version (write-splines, write-holes) can
        # shift+crop_features with this saved value instead of
        # reloading the entire point cloud just to recompute it.
        "course_crop_origin_in_full_frame_x": course_origin_in_full_x,
        "course_crop_origin_in_full_frame_z": course_origin_in_full_z,
    })


def _crop_features_to_course(working_dir: Path, features: list) -> list:
    """
    Shift features (as stored in features.geojson -- the full point
    cloud's frame, uncropped, see parse_osm_features) into the course
    crop's own [0, COURSE_SIZE_M] frame, then crop to it -- shared by
    every step that needs the course-cropped version (write-splines,
    write-holes) without reloading the entire point cloud just to
    recompute the shift ingest-osm already saved.
    """
    project = load_project(working_dir)
    shift_x = project.get("course_crop_origin_in_full_frame_x")
    shift_z = project.get("course_crop_origin_in_full_frame_z")
    if shift_x is None or shift_z is None:
        raise StepError(
            f"No course crop position found in {working_dir}/project.json -- "
            "run --step ingest-osm again (this project may predate storing it)."
        )
    course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    shifted = shift_features(features, dx=-shift_x, dz=-shift_z)
    return crop_features(shifted, course_bounds)


def step_write_splines(working_dir: Path, registration_marks: bool = False) -> None:
    """
    Generate PGA surface splines from features.geojson (see splines.py)
    and write them to course/CourseDescription_nodes/surfaceSplines.json.

    Scope: green/tee/fairway/rough/bunker/cartpath/path/building/wood.
    Water and hole are deliberately excluded (see splines.py's module
    docstring) -- neither is handled by this generic writer yet. mask
    is NOT checked here -- every feature feature_to_spline can handle
    exports regardless of its own mask value (mask only affects
    height_mask.geojson membership and, separately, hole export --
    see holes.py's step_write_holes).

    This overwrites surfaceSplines.json wholesale -- it's the primary
    generator for these surface types now, not a merge with whatever
    was already there (from the blank course template or prior manual
    edits in the PGA editor).
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    features = load_features(features_path)
    features = _crop_features_to_course(working_dir, features)
    splines = build_surface_splines(features)

    if registration_marks:
        marks = build_registration_mark_splines(COURSE_SIZE_M)
        splines = splines + marks
        print(f"  + {len(marks)} registration-mark circle splines (one per corner)")

    unsupported: dict[str, int] = {}
    for f in features:
        if feature_to_spline(f) is None:
            unsupported[f.kind] = unsupported.get(f.kind, 0) + 1

    print(f"Generated {len(splines)} splines from {len(features)} features "
          f"({sum(unsupported.values())} unsupported kind: {unsupported})")

    nodes_dir = working_dir / "course" / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(f"No {nodes_dir} found under {working_dir}. Run --step ingest-course first.")

    out_path = nodes_dir / "surfaceSplines.json"
    save_surface_splines(splines, out_path)
    print(f"Wrote {out_path}")


def step_write_holes(working_dir: Path) -> None:
    """
    Generate holes.json (routing waypoints + par/tee/pin metadata --
    see holes.py) from features.geojson's "hole" ways, matching Chad's
    TGC-Designer-Tools OSMTGC.py newHole() conversion exactly.

    Only "hole" features with mask=False (NOT excluded) are included --
    a masked-out (mask=True) hole is treated as a duplicate/extra hole
    bleeding in from a neighboring course on the same OSM map, and PGA
    can't import more than 18 holes. This is the one place mask
    actually gates export (everything else in surfaceSplines.json
    exports regardless -- see step_write_splines).

    Deliberately separate from step_write_splines/step_output_terrain --
    lets mask settings be tweaked and holes.json regenerated on its
    own, without redoing the terrain height export or surface splines.

    This overwrites holes.json wholesale, same as step_write_splines
    does for surfaceSplines.json.
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    features = load_features(features_path)
    features = _crop_features_to_course(working_dir, features)
    holes = build_holes(features)

    total_hole_features = sum(1 for f in features if f.kind == "hole")
    excluded_count = sum(1 for f in features if f.kind == "hole" and f.mask)
    print(f"Generated {len(holes)} holes from {total_hole_features} hole features "
          f"({excluded_count} excluded via mask)")
    if len(holes) > 18:
        print(f"  WARNING: {len(holes)} holes exceeds PGA's 18-hole limit -- "
              "mask off (exclude) the extras in the GUI's Splines tab before importing")

    nodes_dir = working_dir / "course" / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(f"No {nodes_dir} found under {working_dir}. Run --step ingest-course first.")

    out_path = nodes_dir / "holes.json"
    save_holes(holes, out_path)
    print(f"Wrote {out_path}")


def _resolve_theme(theme_arg: str | None) -> int | None:
    """
    Accept either a theme id (e.g. "7") or a theme name (e.g.
    "countryside", case-insensitive) on the CLI -- see objects.py's
    THEMES_V2019 for the full list (v2019 only -- v2021+ doesn't use
    a theme concept, see build_tree_objects_v2021). Returns None
    (meaning "unrecognized/not set", which build_tree_objects_v2019
    treats as a single generic tree type, not an error) if theme_arg
    is None or doesn't match either.
    """
    if theme_arg is None:
        return None
    theme_arg = theme_arg.strip()
    if theme_arg.isdigit():
        return int(theme_arg)
    for theme_id, name in THEMES_V2019.items():
        if name.lower() == theme_arg.lower():
            return theme_id
    return None


def step_generate_trees(working_dir: Path, detect_lidar_trees: bool | None = None) -> None:
    """
    Generate the intermediate, VERSION-AGNOSTIC object_list.json (see
    objects.py's save_object_list) -- trees parsed from map.osm's
    natural=tree nodes, plus, optionally, trees individually detected
    from LIDAR canopy points (ingest/tree_detection.py), combined into
    one list.

    Deliberately does NOT know about game_version or write
    placedObjects2.json at all -- that's step_write_objects' job, kept
    separate so switching game_version later only needs to re-run the
    (cheap) version-specific formatting step, not repeat this one
    (which does the actually-expensive work: OSM parsing, and LIDAR
    watershed detection if enabled). Same "compile once, format at
    write time" split this project already uses for terrain (Stamp
    objects vs. userLayers.json) and features (features.geojson vs.
    splines/holes).

    detect_lidar_trees (feature-flagged via project.json, same
    None-means-use-saved pattern used throughout this file) adds
    individually-detected trees straight from the LIDAR canopy -- see
    ingest/tree_detection.py -- on top of whatever OSM natural=tree
    nodes were found, combined into one list (real per-tree radius/
    height are carried through as TREE_RADIUS_TAG/TREE_HEIGHT_TAG, see
    objects.py's lidar_trees_to_tagged, so both sources build
    identically regardless of origin). Defaults to True: OSM alone
    typically finds few or no individually-tagged trees on a real
    course (confirmed directly on a real course extract: 0 of ~5000
    OSM nodes were natural=tree), so leaving this off by default would
    silently produce an almost-empty tree list for most courses.
    LIDAR-detected trees are dropped outside height_mask.geojson's
    polygon if that file exists (the "core play area" mask adaptive-
    refine already uses) -- the game's own procedural vegetation fill
    is expected to populate everywhere else, so detecting real trees
    there too would double up rather than add detail. Requires
    heightmap.npz and pointcloud.npz (both from --step ingest-laz);
    raises StepError if either is missing while this is on.

    This overwrites object_list.json wholesale.
    """
    osm_path = working_dir / "map.osm"
    if not osm_path.exists():
        raise StepError(f"No map.osm found at {osm_path}. Run --step ingest-laz first.")

    project = load_project(working_dir)
    if detect_lidar_trees is None:
        detect_lidar_trees = project.get("objects_detect_lidar_trees", True)

    required = ["crs_wkt", "origin_x", "origin_y", "horizontal_unit_factor", "merged_bounds_local"]
    missing = [k for k in required if k not in project]
    if missing:
        raise StepError(
            f"project.json is missing {missing} -- run --step ingest-laz first (or again, "
            "if this project predates saving them)."
        )
    shift_x = project.get("course_crop_origin_in_full_frame_x")
    shift_z = project.get("course_crop_origin_in_full_frame_z")
    if shift_x is None or shift_z is None:
        raise StepError(
            f"No course crop position found in {working_dir}/project.json -- run --step ingest-osm first."
        )

    crs = pyproj.CRS.from_wkt(project["crs_wkt"])
    full_bounds = BoundingBox(**project["merged_bounds_local"])

    print(f"Parsing tree nodes from {osm_path}...")
    full_frame_trees = parse_osm_trees(
        osm_path, crs, project["origin_x"], project["origin_y"], project["horizontal_unit_factor"],
        bounds=full_bounds,
    )

    course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    trees = []
    for x, z, tags in full_frame_trees:
        cx, cz = x - shift_x, z - shift_z
        if course_bounds.min_x <= cx <= course_bounds.max_x and course_bounds.min_z <= cz <= course_bounds.max_z:
            trees.append((cx, cz, tags))
    print(f"  {len(trees)} of {len(full_frame_trees)} tree(s) fall inside the current "
          f"{COURSE_SIZE_M:.0f}x{COURSE_SIZE_M:.0f} m course crop")

    if detect_lidar_trees:
        heightmap_path = working_dir / HEIGHTMAP_FILE
        pointcloud_path = working_dir / POINTCLOUD_FILE
        if not heightmap_path.exists() or not pointcloud_path.exists():
            raise StepError(
                f"--detect-lidar-trees needs both {HEIGHTMAP_FILE} and {POINTCLOUD_FILE} under "
                f"{working_dir} -- run --step ingest-laz first."
            )
        print("Detecting individual trees from LIDAR canopy points "
              "(ingest/tree_detection.py)...")
        ground_heights, _ = load_heightmap(heightmap_path)
        resolution = ground_heights.shape[0]

        full_cloud = PointCloud.load(pointcloud_path)
        course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
        canopy_heights = rasterize_canopy_heightmap_with_fallback(course_cloud, course_bounds, resolution)

        mask_path = working_dir / HEIGHT_MASK_FILE
        mask_geometry = load_height_mask(mask_path) if mask_path.exists() else None
        if mask_geometry is None:
            print(f"  NOTE: no {HEIGHT_MASK_FILE} found -- LIDAR-detected trees will NOT be "
                  "confined to a core play area (run --step ingest-osm to generate one).")

        lidar_trees = detect_trees_from_lidar(
            ground_heights, canopy_heights, course_bounds, mask_geometry=mask_geometry,
        )
        trees += lidar_trees_to_tagged(lidar_trees)
        print(f"  {len(lidar_trees)} LIDAR-detected tree(s) added "
              f"({len(trees)} total tree(s) now)")

    features_path = working_dir / FEATURES_FILE
    if features_path.exists() and trees:
        features = load_features(features_path)
        features = _crop_features_to_course(working_dir, features)
        wood_features = [f for f in features if f.kind == "wood"]
        if wood_features:
            untyped_before = sum(1 for _, _, tags in trees if TREE_TYPE_TAG not in tags)
            trees = apply_area_tree_type_hints(trees, wood_features)
            untyped_after = sum(1 for _, _, tags in trees if TREE_TYPE_TAG not in tags)
            if untyped_before != untyped_after:
                print(f"  applied area-based tree-type hints from {len(wood_features)} wood "
                      f"polygon(s): {untyped_before - untyped_after} tree(s) tagged "
                      "(course_output/objects.py's LEAF_TYPE_TREE_HINTS)")
    elif trees:
        print(f"  No {FEATURES_FILE} found -- skipping area-based tree-type hints "
              "(run --step ingest-osm first if you want wood-polygon leaf_type hints applied).")

    out_path = working_dir / OBJECT_LIST_FILE
    save_object_list(trees, out_path)
    print(f"Wrote {out_path} ({len(trees)} tree(s))")

    save_project(working_dir, {
        "objects_detect_lidar_trees": detect_lidar_trees,
        "objects_tree_count": len(trees),
    })


def step_write_objects(
    working_dir: Path,
    game_version: str | None = None,
    theme: int | None = None,
    tree_variety: bool | None = None,
    tree_asset_paths: list[str] | None = None,
    tree_type_asset_paths: dict[str, str] | None = None,
    stake_asset_path: str | None = None,
) -> None:
    """
    Generate placedObjects2.json -- formats object_list.json (see
    step_generate_trees; run that first) into the target game_version's
    schema, plus, optionally (v2021+ only), a stake at every building
    corner (from features.geojson's "building" ways).

    game_version selects which of objects.py's two confirmed schemas
    to write (see that module's docstring: v2019 is Chad Rockey's
    numeric category/type/theme catalog; v2021+ is a real Unity asset
    path). It's a project-level setting, same as course_name -- the
    GUI sets it once near the top (not per-step) and it's read from
    project.json here, same as every other value below. Raises
    StepError up front if game_version isn't in
    objects.IMPLEMENTED_GAME_VERSIONS (v2023/v2025 aren't confirmed
    yet -- see objects.py's module docstring) rather than silently
    guessing at an unconfirmed schema.

    theme / tree_variety (v2019) and tree_asset_paths /
    tree_type_asset_paths / stake_asset_path (v2021+) are all feature-
    flagged via project.json, same pattern as refine-terrain's
    parameters: pass None here to use whatever was last saved, or an
    explicit value (an empty list/dict counts as explicit) to override
    for this run and persist it as the new default for next time. Only
    the parameters relevant to the resolved game_version are actually
    used; the others are still accepted (and persisted, if given) so a
    project can carry both versions' settings across a future
    game_version switch without losing them.

    tree_variety defaults to True (not just "off unless set") -- there's
    no real reason to want the flat, single-generic-type result it
    disables to; it exists mainly as an override for testing.

    This overwrites placedObjects2.json wholesale, same as
    step_write_splines/step_write_holes do for their own files.
    """
    object_list_path = working_dir / OBJECT_LIST_FILE
    if not object_list_path.exists():
        raise StepError(
            f"No {OBJECT_LIST_FILE} found under {working_dir}. Run --step generate-trees first."
        )

    project = load_project(working_dir)
    if game_version is None:
        game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    if game_version not in IMPLEMENTED_GAME_VERSIONS:
        raise StepError(
            f"game_version={game_version!r} isn't implemented yet (only {IMPLEMENTED_GAME_VERSIONS} "
            "are) -- see objects.py's module docstring. Set --game-version explicitly, or fix "
            "project.json's saved 'game_version' if this project predates it."
        )
    if theme is None:
        theme = project.get("objects_theme")
    if tree_variety is None:
        tree_variety = project.get("objects_tree_variety", True)
    if tree_asset_paths is None:
        tree_asset_paths = project.get("objects_tree_asset_paths", [])
    if tree_type_asset_paths is None:
        tree_type_asset_paths = project.get("objects_tree_type_asset_paths", {})
    if stake_asset_path is None:
        stake_asset_path = project.get("objects_stake_asset_path")

    trees = load_object_list(object_list_path)
    print(f"game_version={game_version}  loaded {len(trees)} tree(s) from {OBJECT_LIST_FILE}")

    placed_objects: list[dict] = []

    if game_version == "2019":
        if trees:
            print(f"  theme={theme}  tree_variety={tree_variety}")
            placed_objects += build_tree_objects_v2019(trees, theme=theme, tree_variety=tree_variety)
        if stake_asset_path:
            print("  NOTE: --stake-asset-path is set but ignored for game_version=2019 -- "
                  "building stakes need v2021+'s asset-path scheme (see objects.py's "
                  "build_building_stake_objects_v2021 docstring).")
    else:  # 2021+ (only "2021" itself is in IMPLEMENTED_GAME_VERSIONS right now)
        if trees:
            if not tree_asset_paths and not tree_type_asset_paths:
                raise StepError(
                    f"{len(trees)} tree(s) found in object_list.json, but no tree asset path "
                    "is set -- pass --tree-asset-path (repeatable) and/or --tree-type-asset-path "
                    "TAG=path (repeatable). See objects.py's module docstring: there's no built-in "
                    "catalog to fall back to, v2021+ placed objects need real Unity asset paths."
                )
            print(f"  tree_asset_paths={tree_asset_paths}  tree_type_asset_paths={tree_type_asset_paths}")
            placed_objects += build_tree_objects_v2021(trees, tree_asset_paths, tree_type_asset_paths)

        if stake_asset_path:
            features_path = working_dir / FEATURES_FILE
            if not features_path.exists():
                raise StepError(
                    f"--stake-asset-path was given but no {FEATURES_FILE} found under {working_dir} "
                    "(needed for building corners) -- run --step ingest-osm first."
                )
            features = load_features(features_path)
            features = _crop_features_to_course(working_dir, features)
            building_count = sum(1 for f in features if f.kind == "building")
            stakes = build_building_stake_objects_v2021(features, stake_asset_path)
            stake_count = sum(len(g["Value"]["items"]) for g in stakes)
            print(f"  {stake_count} stake(s) at corners of {building_count} building(s)")
            placed_objects += stakes

    for label, item_count, cluster_count, spline_count in object_counts(placed_objects):
        print(f"    {label}: {item_count} item(s), {cluster_count} cluster(s), {spline_count} spline(s)")

    nodes_dir = working_dir / "course" / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(f"No {nodes_dir} found under {working_dir}. Run --step ingest-course first.")

    out_path = nodes_dir / PLACED_OBJECTS_FILE
    save_placed_objects(placed_objects, out_path)
    print(f"Wrote {out_path}")

    save_project(working_dir, {
        "game_version": game_version,
        "objects_theme": theme,
        "objects_tree_variety": tree_variety,
        "objects_tree_asset_paths": tree_asset_paths,
        "objects_tree_type_asset_paths": tree_type_asset_paths,
        "objects_stake_asset_path": stake_asset_path,
    })


def step_dig_water(
    working_dir: Path, dig_depth_m: float | None = None, buffer_m: float | None = None,
) -> None:
    """
    Lowers heightmap.npz by dig_depth_m wherever an (inward-buffered)
    OSM water polygon covers it -- see ingest/heightmap.py's
    dig_water_into_heightmap and course_output/water.py's own module
    docstring for the companion "water plane clips slightly into the
    bank" half of this same idea. Meant to run once, after both Ingest
    LAZ and Ingest OSM, before Generate/Refine Terrain -- everything
    downstream (adaptive/scatter refinement, water-level lookup) just
    sees the resulting recessed heightmap and needs no water-specific
    awareness of its own; letting refine-terrain do the rest is the
    whole point, not a separate water-aware terrain algorithm.

    buffer_m shrinks each water polygon INWARD (negative buffer) before
    determining which cells to lower, so the dug recess ends up
    slightly SMALLER than the water body's actual mapped outline --
    letting the water plane object (built from the ORIGINAL, un-
    buffered polygon; see water.py) clip a little into the surrounding
    terrain at the edges instead of floating exactly at the rim of a
    perfectly-matching recess with a visible seam.

    Modifies heightmap.npz IN PLACE (overwrites it) -- there's no
    separate "pristine" checkpoint kept here. Running this a second
    time therefore compounds the dig (lowers already-dug cells by
    dig_depth_m again, not to a fixed target level) -- if you want to
    change dig_depth_m/buffer_m after already digging once, re-run
    Ingest LAZ first to regenerate a clean heightmap.npz, then dig
    again. A project.json flag triggers a loud warning (not a hard
    block) if this looks like a second run without that reset, so a
    compounded dig is a deliberate choice, not an accident.
    """
    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first.")
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    project = load_project(working_dir)
    if dig_depth_m is None:
        dig_depth_m = project.get("dig_water_depth_m", DEFAULT_DIG_WATER_DEPTH_M)
    if buffer_m is None:
        buffer_m = project.get("dig_water_buffer_m", DEFAULT_DIG_WATER_BUFFER_M)

    if project.get("water_dig_applied"):
        print(f"  WARNING: water digging was already applied to this {HEIGHTMAP_FILE} -- running "
              "again will compound the dig (lower already-dug cells a second time), not re-dig to a "
              "fixed level. Re-run Ingest LAZ first for a clean slate if that's not what you want.")

    heights, bounds = load_heightmap(heightmap_path)

    features = load_features(features_path)
    features = _crop_features_to_course(working_dir, features)
    water_features = [f for f in features if f.kind == "water"]
    if not water_features:
        print("  No water features found -- nothing to dig.")
        save_project(working_dir, {
            "water_dig_applied": True, "dig_water_depth_m": dig_depth_m, "dig_water_buffer_m": buffer_m,
        })
        return

    buffered_geoms = []
    skipped = 0
    for f in water_features:
        buffered = f.geometry.buffer(-buffer_m)
        if buffered.is_empty:
            skipped += 1
            continue
        buffered_geoms.append(buffered)
    if skipped:
        print(f"  {skipped} water polygon(s) collapsed to nothing under a {buffer_m} m inward "
              "buffer (too small) -- skipped.")
    if not buffered_geoms:
        print("  No water polygons survived the inward buffer -- nothing to dig.")
        save_project(working_dir, {
            "water_dig_applied": True, "dig_water_depth_m": dig_depth_m, "dig_water_buffer_m": buffer_m,
        })
        return

    union_geom = unary_union(buffered_geoms)
    resolution = heights.shape[0]
    mask = rasterize_mask(union_geom, bounds, resolution)
    dug_cell_count = int(mask.sum())
    print(f"Digging {dig_depth_m} m into {dug_cell_count:,} heightmap cell(s) "
          f"({dug_cell_count / mask.size:.2%} of the course) under {len(buffered_geoms)} water "
          f"polygon(s), each buffered inward by {buffer_m} m...")

    new_heights = dig_water_into_heightmap(heights, mask, dig_depth_m)
    save_heightmap(new_heights, bounds, heightmap_path)
    print(f"  wrote {heightmap_path}")

    save_project(working_dir, {
        "water_dig_applied": True,
        "dig_water_depth_m": dig_depth_m,
        "dig_water_buffer_m": buffer_m,
    })

    print("Refreshing previews...")
    step_visualize(working_dir)


def step_generate_terrain(
    working_dir: Path,
    pitch: float | None = None,
    method: str | None = None,
    band_spacing_m: float | None = None,
    fill_brush: int | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    radius_step_ratio: float | None = None,
    edge_distance_m: float | None = None,
    smoothing_brush: int | None = None,
    smoothing_min_radius: float | None = None,
    smooth_ratio: float | None = None,
    smooth_claim_fraction: float | None = None,
    candidates_per_radius: int | None = None,
    sweet_spot_ratio: float | None = None,
    sweet_spot_sample_bands: int | None = None,
    sweet_spot_seeds: int | None = None,
    sweet_spot_max_candidates: int | None = None,
    sweet_spot_time_budget_s: float | None = None,
    random_seed: int | None = None,
    denoise_px: int | None = None,
    max_stamps: int | None = None,
) -> None:
    """
    pitch (feature-flagged via project.json, same None-means-use-saved
    pattern used throughout this file) is terrain/hexgrid.py's
    HEX_LATTICE_PITCH_M, exposed here rather than hardcoded -- controls
    the spacing of the initial coarse hex-grid stamp lattice (smaller
    pitch = more, smaller, more tightly-packed initial stamps). Stamp
    radius and edge bleed both derive from pitch using the exact same
    ratios hexgrid.py's own module-level constants encode (radius =
    2*pitch, bleed = radius/2 = pitch) -- generate_hex_grid() itself
    only defaults those two to the ORIGINAL fixed pitch's values (Python
    default arguments are evaluated once, not re-derived from whatever
    `pitch` is actually passed), so both are computed explicitly here
    for whatever pitch is in play, not left to fall back silently.

    method ("hex", default, or "contour") picks the initial-layout
    generator: "hex" is the flat lattice above; "contour" is terrain/
    contour_layers.py's two-pass-per-band fill -- see that module's
    docstring for the full design. Briefly: PASS 1 is a fast random-
    candidate poisson pack with a hard, high-plateau fill_brush (type
    8/73), then PASS 2 is an oversized, heavily-overlapping scatter
    fill with a softer smoothing_brush over whatever pass 1 leaves as
    genuine crumbs -- pass 2 IS exhaustive, so overall coverage stays
    complete by construction even though pass 1 trades a per-call
    guarantee for speed. Only "hex"'s parameters (pitch) apply in "hex"
    mode and vice versa for the contour_* parameters below.

    Unlike hex mode, contour mode's stamps already carry their exact
    fitted value (the local heightmap mean within each stamp's own
    footprint, computed inside generate_contour_layers itself) -- so
    the fit_stamp_heights() pass below only runs in "hex" mode.
    Re-running it against contour stamps would be redundant at best.

    max_radius is the main contour-mode tuning knob for level of
    detail: it caps how large pass 1's biggest stamps can be, so it
    should scale with how much real terrain variation exists at your
    chosen band_spacing_m.

    candidates_per_radius (left unset) auto-tunes itself once at the
    start of the run by searching a handful of sample bands for the
    point of diminishing returns -- see the --candidates-per-radius
    and --sweet-spot-* CLI help text, and generate_contour_layers'
    own docstring, for the full calibration design. Set it explicitly
    once you've seen a good auto-tuned value to skip re-running that
    search on every subsequent run -- it is NOT auto-persisted from an
    auto-tuned run, you have to note the value yourself.

    max_stamps (contour method only) stops generation once that many
    stamps have been placed in total -- a quick way to sanity-check a
    parameter combination before committing to the full run, not a real
    generation mode: bands process ascending by elevation, so the cutoff
    always lands on the low-elevation end and most of the course will
    genuinely be unfilled, not just coarser. Deliberately NOT persisted
    to project.json (unlike every other contour-mode parameter here) --
    it's meant to be set explicitly each time you want a quick partial
    preview, not silently inherited by your next real run.
    """
    if method is None:
        project = load_project(working_dir)
        method = project.get("generate_terrain_method", "hex")
    if method not in ("hex", "contour"):
        raise StepError(f"method must be 'hex' or 'contour', got {method!r}")

    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    project = load_project(working_dir)
    if pitch is None:
        pitch = project.get("generate_terrain_pitch_m", HEX_LATTICE_PITCH_M)
    if band_spacing_m is None:
        band_spacing_m = project.get("generate_terrain_band_spacing_m", DEFAULT_BAND_SPACING_M)
    if fill_brush is None:
        fill_brush = project.get("generate_terrain_fill_brush", DEFAULT_FILL_BRUSH)
    if min_radius is None:
        min_radius = project.get("generate_terrain_min_radius_m", DEFAULT_MIN_RADIUS_M)
    if max_radius is None:
        max_radius = project.get("generate_terrain_max_radius_m", DEFAULT_MAX_RADIUS_M)
    if radius_step_ratio is None:
        radius_step_ratio = project.get("generate_terrain_radius_step_ratio", DEFAULT_RADIUS_STEP_RATIO)
    if edge_distance_m is None:
        edge_distance_m = project.get("generate_terrain_edge_distance_m", DEFAULT_EDGE_DISTANCE_M)
    if smoothing_brush is None:
        smoothing_brush = project.get("generate_terrain_smoothing_brush", DEFAULT_SMOOTHING_BRUSH)
    if smoothing_min_radius is None:
        smoothing_min_radius = project.get(
            "generate_terrain_smoothing_min_radius_m", DEFAULT_SMOOTHING_MIN_RADIUS_M
        )
    if smooth_ratio is None:
        smooth_ratio = project.get("generate_terrain_smooth_ratio", DEFAULT_CRUMB_SCATTER_MULTIPLIER)
    if smooth_claim_fraction is None:
        smooth_claim_fraction = project.get(
            "generate_terrain_smooth_claim_fraction", DEFAULT_SMOOTH_CLAIM_FRACTION
        )
    if candidates_per_radius is None:
        # Unlike every other knob here, NOT resolved to a fixed default
        # if absent from project.json -- staying None means
        # generate_contour_layers auto-tunes it fresh (see that
        # function and _auto_tune_candidates). Set this explicitly
        # (CLI/GUI) once you've seen a good auto-tuned value to skip
        # re-running the calibration search on every subsequent run.
        candidates_per_radius = project.get("generate_terrain_candidates_per_radius", None)
    if sweet_spot_ratio is None:
        sweet_spot_ratio = project.get("generate_terrain_sweet_spot_ratio", DEFAULT_SWEET_SPOT_STAMP_RATIO)
    if sweet_spot_sample_bands is None:
        sweet_spot_sample_bands = project.get(
            "generate_terrain_sweet_spot_sample_bands", DEFAULT_SWEET_SPOT_SAMPLE_BANDS
        )
    if sweet_spot_seeds is None:
        sweet_spot_seeds = project.get("generate_terrain_sweet_spot_seeds", DEFAULT_SWEET_SPOT_SEEDS)
    if sweet_spot_max_candidates is None:
        sweet_spot_max_candidates = project.get(
            "generate_terrain_sweet_spot_max_candidates", DEFAULT_SWEET_SPOT_MAX_CANDIDATES
        )
    if sweet_spot_time_budget_s is None:
        sweet_spot_time_budget_s = project.get(
            "generate_terrain_sweet_spot_time_budget_s", DEFAULT_SWEET_SPOT_TIME_BUDGET_S
        )
    if random_seed is None:
        random_seed = project.get("generate_terrain_random_seed", DEFAULT_RANDOM_SEED)
    if denoise_px is None:
        denoise_px = project.get("generate_terrain_denoise_px", DEFAULT_DENOISE_PX)

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

    if method == "contour":
        print(f"Two-pass poisson band fill (band_spacing_m={band_spacing_m}, "
              f"pass1_radius=[{min_radius}, {max_radius}] m, step_ratio={radius_step_ratio}, "
              f"pass2_radius={smoothing_min_radius * smooth_ratio} m, "
              f"candidates_per_radius={'auto-tuning...' if candidates_per_radius is None else candidates_per_radius})...")
        if max_stamps is not None:
            print(f"  max_stamps={max_stamps} -- PARTIAL PREVIEW RUN, not a real generation: "
                  "bands fill ascending by elevation, so this will stop somewhere on the low-"
                  "elevation end and most of the course will be genuinely unfilled, not just "
                  "coarser. Re-run without --max-stamps for the real thing.")

        progress_start_time = time.time()

        def _print_contour_progress(stamp_count: int, fraction: float) -> None:
            elapsed = time.time() - progress_start_time
            print(f"  ... {elapsed:.0f}s elapsed: {stamp_count} stamps so far, "
                  f"{fraction:.1%} complete (bands ascending, lowest to highest elevation)")

        fitted = generate_contour_layers(
            heightmap, bounds,
            band_spacing_m=band_spacing_m,
            fill_brush=fill_brush,
            min_radius=min_radius,
            max_radius=max_radius,
            radius_step_ratio=radius_step_ratio,
            edge_distance_m=edge_distance_m,
            smoothing_brush=smoothing_brush,
            smoothing_min_radius=smoothing_min_radius,
            smooth_ratio=smooth_ratio,
            smooth_claim_fraction=smooth_claim_fraction,
            candidates_per_radius=candidates_per_radius,
            sweet_spot_ratio=sweet_spot_ratio,
            sweet_spot_sample_bands=sweet_spot_sample_bands,
            sweet_spot_seeds=sweet_spot_seeds,
            sweet_spot_max_candidates=sweet_spot_max_candidates,
            sweet_spot_time_budget_s=sweet_spot_time_budget_s,
            random_seed=random_seed,
            denoise_px=denoise_px,
            max_stamps=max_stamps,
            progress_callback=_print_contour_progress,
        )
        if max_stamps is not None and len(fitted) >= max_stamps:
            print(f"  {len(fitted)} stamps placed -- STOPPED at max_stamps={max_stamps}, "
                  "course is only partially filled (see note above)")
        else:
            print(f"  {len(fitted)} stamps placed (tiered band fill + crumb smoothing, all already fitted)")
    else:
        stamp_radius = 2.0 * pitch
        bleed = stamp_radius / 2.0
        print(f"Generating hex grid (pitch={pitch} m, stamp_radius={stamp_radius} m, bleed={bleed} m)...")
        stamps = generate_hex_grid(bounds, pitch=pitch, stamp_radius=stamp_radius, bleed=bleed)
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
        parameters={"course_size_m": COURSE_SIZE_M, "pitch_m": pitch, "method": method,
                     "band_spacing_m": band_spacing_m},
    )
    print(f"  wrote {out_path}")


    save_project(working_dir, {
        "course_origin_x": course_cloud.origin_x,
        "course_origin_y": course_cloud.origin_y,
        "stamp_count": len(fitted),
        "generate_terrain_pitch_m": pitch,
        "generate_terrain_method": method,
        "generate_terrain_band_spacing_m": band_spacing_m,
        "generate_terrain_fill_brush": fill_brush,
        "generate_terrain_min_radius_m": min_radius,
        "generate_terrain_max_radius_m": max_radius,
        "generate_terrain_radius_step_ratio": radius_step_ratio,
        "generate_terrain_edge_distance_m": edge_distance_m,
        "generate_terrain_smoothing_brush": smoothing_brush,
        "generate_terrain_smoothing_min_radius_m": smoothing_min_radius,
        "generate_terrain_smooth_ratio": smooth_ratio,
        "generate_terrain_smooth_claim_fraction": smooth_claim_fraction,
        "generate_terrain_candidates_per_radius": candidates_per_radius,
        "generate_terrain_sweet_spot_ratio": sweet_spot_ratio,
        "generate_terrain_sweet_spot_sample_bands": sweet_spot_sample_bands,
        "generate_terrain_sweet_spot_seeds": sweet_spot_seeds,
        "generate_terrain_sweet_spot_max_candidates": sweet_spot_max_candidates,
        "generate_terrain_sweet_spot_time_budget_s": sweet_spot_time_budget_s,
        "generate_terrain_random_seed": random_seed,
        "generate_terrain_denoise_px": denoise_px,
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
    method: str | None,
    use_height_mask: bool | None,
    mask_buffer_px: float | None = None,
    model_rebuild_interval: int | None = None,
    candidate_brushes: tuple[int, ...] | None = None,
    max_planar_rms: float | None = None,
    planar_shrink_factor: float | None = None,
    rad_m: float | None = None,
    use_slope_radius: bool | None = None,
    use_variation_radius: bool | None = None,
    variation_contrast_gamma: float | None = None,
    density_weighted: bool | None = None,
    subpixel_jitter_fraction: float | None = None,
) -> None:
    """
    One refinement pass (see terrain/adaptive_refine.py), in one of two
    methods:

      "adaptive" (default) -- find contiguous regions of the binned
      error grid exceeding `tolerance` (same grid preview_error.png
      visualizes), add one stamp per region centered and sized on it.

      "scatter" -- ignore error entirely; place stamps at randomly-
      chosen, well-spaced sites, each flattened to the real local
      LIDAR average -- see adaptive_refine.py's scatter_stamps for the
      full rationale (closer in spirit to Chad Rockey's fixed-grid
      raster approach, just organically spaced).

    Either way, only the newly-added stamps (not the whole cumulative
    list) are written to the next refine_stamps_N.json. Safe to run
    repeatedly -- each call reconstructs the full current terrain via
    load_all_stamps() (initial_stamps.json plus every prior
    refine_stamps_N.json) and builds against that, so "run this a few
    times, watch coverage improve" is the expected way to iterate.
    Deleting the highest-numbered refine_stamps_N.json undoes just
    that pass.

    rad_m ("RAD") is the literal target stamp radius (m) for THIS
    pass, replacing the old radius_decay_per_pass percentage as the
    direct, primary size control (see adaptive_refine.py's
    DEFAULT_RAD_M docstring) -- no more indirect "decay compounds
    across N passes" math to reason about; you just say how big you
    want stamps this run. In "adaptive" mode this becomes max_radius
    (min_radius = rad_m * the fixed 0.5 ratio DEFAULT_MIN/
    MAX_HOTSPOT_RADIUS_M already used); in "scatter" mode it's the
    literal per-stamp placement radius before jitter. The old implied
    "decay" (how much smaller this pass's stamps are than the last
    pass's) is now a DERIVED, informational value only -- computed
    from last_refine_rad_m / rad_m and saved to project.json for
    display, never fed back into the computation.

    claim_radius_fraction / brush_radius_spread_ratio / rad_m /
    max_planar_rms / planar_shrink_factor are feature-flagged via
    project.json rather than always needing a CLI value: pass None
    here to use whatever was last saved (defaulting to the old/off
    behavior if never set), or an explicit value to override for this
    run and persist it as the new default for next time.

    max_planar_rms (adaptive only) shrinks a hotspot's radius (before
    claim_radius_fraction / brush_radius_spread_ratio are applied to
    it -- see adaptive_refine.py) until the region's actual LIDAR
    heights fit a single tilted plane within this RMS (m), catching
    cases the error-sign-based sizing above can't: a valley's V-shaped
    cross-section is one contiguous same-sign error region from floor
    to rim, so it grows a stamp radius all the way to the rim with no
    planarity check, pulling the floor up and the rim down under one
    averaged stamp. None (default) disables this -- old behavior.

    planar_shrink_factor ("SHR%") does double duty depending on
    method: in "adaptive" it's max_planar_rms's shrink-loop step size
    (see above); in "scatter" it's repurposed as radius jitter
    magnitude instead (see adaptive_refine.py's scatter_stamps) --
    there's no planar-fit concept in scatter mode at all, so reusing
    the same GUI/CLI knob for a different, mode-appropriate purpose
    avoids adding a redundant parameter.

    model_rebuild_interval only applies to "adaptive" -- "scatter"
    doesn't build or evaluate a TerrainModel at all (see
    adaptive_refine.py's scatter_stamps for why that's provably
    unnecessary there), so this is silently ignored when method is
    "scatter".

    use_variation_radius (scatter only) is the corrected replacement
    for use_slope_radius: site radius is driven by RMS-from-local-mean
    at lag=rad_m (real curvature AND macro-scale slope carried across
    the window) rather than raw gradient magnitude, which a tilted
    fairway reads as "steep" everywhere even though a wide flat stamp
    would represent it fine at a small enough window. If both this and
    use_slope_radius are set, use_variation_radius wins (see
    adaptive_refine.py's scatter_stamps). variation_contrast_gamma (>1
    sharpens toward the extremes, default 2.0) reshapes the percentile
    map planar_shrink_factor/SHR% bounds.

    density_weighted (scatter only) fixes what shrinking radius alone
    can't: a smaller target radius previously only changed how big an
    accepted dart was, never how often darts landed there, so small
    high-detail regions ended up with isolated small stamps rather than
    a tightly-packed cluster. With this on, candidate sites are drawn
    from a precomputed density field (~1/radius^2) instead of uniform-
    random over the whole course. Requires use_slope_radius or
    use_variation_radius to be meaningful -- with neither set, every
    site wants the same radius and density-weighting degenerates to
    uniform anyway. subpixel_jitter_fraction (default 0.5, i.e. up to
    half a cell width) jitters density-weighted draws off the exact
    cell center -- dither only, to avoid visibly grid-aligned stamp
    centers, not for precision (the course never needs sub-cell
    accuracy).
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
    if method is None:
        method = project.get("refine_method", "adaptive")
    if method not in ("adaptive", "scatter"):
        raise StepError(f"method must be 'adaptive' or 'scatter', got {method!r}")
    if rad_m is None:
        rad_m = project.get("refine_rad_m", DEFAULT_RAD_M)
    if use_height_mask is None:
        use_height_mask = project.get("refine_use_height_mask", False)
    if model_rebuild_interval is None:
        model_rebuild_interval = project.get(
            "refine_model_rebuild_interval", DEFAULT_MODEL_REBUILD_INTERVAL
        )
    if candidate_brushes is None:
        saved_brushes = project.get("refine_candidate_brushes")
        candidate_brushes = tuple(saved_brushes) if saved_brushes is not None else None
    if max_planar_rms is None:
        max_planar_rms = project.get("refine_max_planar_rms", DEFAULT_MAX_PLANAR_RMS)
    if planar_shrink_factor is None:
        planar_shrink_factor = project.get(
            "refine_planar_shrink_factor", DEFAULT_PLANAR_SHRINK_FACTOR
        )
    if use_slope_radius is None:
        use_slope_radius = project.get("refine_use_slope_radius", False)
    if use_variation_radius is None:
        use_variation_radius = project.get("refine_use_variation_radius", False)
    if variation_contrast_gamma is None:
        variation_contrast_gamma = project.get(
            "refine_variation_contrast_gamma", DEFAULT_VARIATION_CONTRAST_GAMMA
        )
    if density_weighted is None:
        density_weighted = project.get("refine_density_weighted", False)
    if subpixel_jitter_fraction is None:
        subpixel_jitter_fraction = project.get(
            "refine_subpixel_jitter_fraction", DEFAULT_SUBPIXEL_JITTER_FRACTION
        )

    stamps = load_all_stamps(working_dir)
    pass_number = len(_refine_stamps_files(working_dir)) + 1
    print(f"  {len(stamps)} stamps (cumulative: initial + {pass_number - 1} prior refine pass(es))")
    print(f"  method={method}  rad_m={rad_m}  claim_radius_fraction={claim_radius_fraction}  "
          f"brush_radius_spread_ratio={brush_radius_spread_ratio}  use_height_mask={use_height_mask}")

    last_rad_m = project.get("last_refine_rad_m")
    implied_decay = (last_rad_m / rad_m) if last_rad_m and rad_m else None
    if implied_decay is not None:
        print(f"  implied decay vs. last run: {implied_decay:.3f}x "
              f"(last_refine_rad_m={last_rad_m} -> rad_m={rad_m})")

    if max_planar_rms is not None and method == "adaptive":
        print(f"  max_planar_rms={max_planar_rms}  planar_shrink_factor={planar_shrink_factor}")
    if use_slope_radius and method == "scatter" and not use_variation_radius:
        print(f"  use_slope_radius=on -- stamp radius driven by real local terrain slope "
              f"(np.gradient), not random jitter; planar_shrink_factor={planar_shrink_factor} is "
              "reused as the 'how small can it shrink on steep ground' floor.")
    if use_variation_radius and method == "scatter":
        print(f"  use_variation_radius=on -- stamp radius driven by RMS-from-local-mean at "
              f"lag={rad_m}m (real curvature + macro-scale slope carried across the window), not "
              f"random jitter; planar_shrink_factor={planar_shrink_factor} is the shrink floor, "
              f"variation_contrast_gamma={variation_contrast_gamma} sharpens the map toward the "
              "extremes.")
        if use_slope_radius:
            print("  (use_slope_radius is also on -- use_variation_radius takes priority)")
    if density_weighted and method == "scatter":
        print(f"  density_weighted=on -- candidate sites drawn from a ~1/radius^2 density field "
              f"instead of uniform-random; subpixel_jitter_fraction={subpixel_jitter_fraction} "
              "dithers draws off exact cell centers.")
        if not (use_slope_radius or use_variation_radius):
            print("  NOTE: density_weighted has no effect without use_slope_radius or "
                  "use_variation_radius -- every site wants the same radius, so density-weighting "
                  "degenerates to uniform draws anyway.")

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

    progress_start_time = time.time()

    def _print_adaptive_progress(hotspot_count: int, claimed_fraction: float) -> None:
        elapsed = time.time() - progress_start_time
        print(f"  ... {elapsed:.0f}s elapsed: {hotspot_count} stamps so far, "
              f"{claimed_fraction:.1%} of the searchable area claimed")

    def _print_scatter_progress(hotspot_count: int, failure_fraction: float) -> None:
        # Not "searchable area" -- scatter mode never scans/claims a
        # grid at all (see adaptive_refine.py's scatter_stamps); this
        # is dart-throwing, so the only meaningful "how close to done"
        # signal is how much of the consecutive-failure budget before
        # giving up has been spent on the CURRENT run of rejections.
        elapsed = time.time() - progress_start_time
        print(f"  ... {elapsed:.0f}s elapsed: {hotspot_count} stamps placed, "
              f"{failure_fraction:.1%} of the way through the current run of rejected placements "
              "(stops once that reaches 100% -- the space is full)")

    if method == "scatter":
        print(f"Scattering stamps ({resolution}x{resolution} grid, rad_m={rad_m})...")
        refined, hotspots = scatter_refine_stamps(
            stamps, heights, bounds, rad_m=rad_m,
            resolution=resolution,
            claim_radius_fraction=claim_radius_fraction,
            brush_radius_spread_ratio=brush_radius_spread_ratio,
            jitter_factor=planar_shrink_factor,
            use_slope_radius=use_slope_radius,
            use_variation_radius=use_variation_radius,
            variation_contrast_gamma=variation_contrast_gamma,
            density_weighted=density_weighted,
            subpixel_jitter_fraction=subpixel_jitter_fraction,
            max_new_stamps=max_new_stamps,
            mask=mask_grid,
            candidate_brushes=candidate_brushes,
            progress_callback=_print_scatter_progress,
        )
    else:
        max_radius = rad_m
        min_radius = rad_m * (DEFAULT_MIN_HOTSPOT_RADIUS_M / DEFAULT_MAX_HOTSPOT_RADIUS_M)
        print(f"  min/max hotspot radius this pass: {min_radius:.2f} / {max_radius:.2f} m")
        print(f"Scanning the error grid ({resolution}x{resolution}, tolerance={tolerance} m)...")
        refined, hotspots = refine_stamps(
            stamps, heights, bounds, tolerance=tolerance,
            resolution=resolution, min_hotspot_radius_cells=min_hotspot_radius_cells,
            min_radius=min_radius, max_radius=max_radius,
            claim_radius_fraction=claim_radius_fraction,
            brush_radius_spread_ratio=brush_radius_spread_ratio,
            max_new_stamps=max_new_stamps,
            mask=mask_grid,
            model_rebuild_interval=model_rebuild_interval,
            candidate_brushes=candidate_brushes,
            max_planar_rms=max_planar_rms,
            planar_shrink_factor=planar_shrink_factor,
            progress_callback=_print_adaptive_progress,
        )

    new_stamps = refined[len(stamps):]
    fit_rms_values = [h.fit_rms for h in hotspots]
    mean_fit_rms = float(np.mean(fit_rms_values)) if fit_rms_values else None
    max_fit_rms = float(np.max(fit_rms_values)) if fit_rms_values else None
    if hotspots:
        worst = hotspots[0]
        label = "hotspots over tolerance" if method == "adaptive" else "stamps placed"
        print(f"  {len(hotspots)} {label} (worst peak_error: {worst.peak_error:.3f} m "
              f"at ({worst.x:.1f}, {worst.z:.1f}), {worst.n_cells} cells); "
              f"fit_rms mean={mean_fit_rms:.3f} max={max_fit_rms:.3f}")
    else:
        print("  nothing placed this pass")

    parameters = {
        "method": method,
        "tolerance": tolerance,
        "resolution": resolution,
        "min_hotspot_radius_cells": min_hotspot_radius_cells,
        "max_new_stamps": max_new_stamps,
        "claim_radius_fraction": claim_radius_fraction,
        "brush_radius_spread_ratio": brush_radius_spread_ratio,
        "rad_m": rad_m,
        "use_height_mask": use_height_mask,
        "mask_buffer_px": mask_buffer_px,
        "model_rebuild_interval": model_rebuild_interval,
        "candidate_brushes": list(candidate_brushes) if candidate_brushes is not None else None,
        "max_planar_rms": max_planar_rms,
        "planar_shrink_factor": planar_shrink_factor,
    }

    if new_stamps:
        next_n = len(_refine_stamps_files(working_dir)) + 1
        out_path = _stamps_dir(working_dir) / REFINE_STAMPS_PATTERN.format(n=next_n)
        save_stamp_file(
            new_stamps, out_path, step="refine-terrain", parameters=parameters,
            extra={"hotspot_count": len(hotspots), "mean_fit_rms": mean_fit_rms, "max_fit_rms": max_fit_rms},
        )
        print(f"  wrote {out_path} ({len(new_stamps)} new stamps; "
              f"{len(stamps)} -> {len(refined)} total)")
    else:
        print(f"  nothing written ({len(stamps)} total, unchanged)")

    save_project(working_dir, {
        "last_refine_method": method,
        "last_refine_rad_m": rad_m,
        "last_refine_implied_decay": implied_decay,
        "last_refine_tolerance_m": tolerance,
        "last_refine_hotspot_count": len(hotspots),
        "last_refine_added_count": len(new_stamps),
        "last_refine_mean_fit_rms": mean_fit_rms,
        "last_refine_max_fit_rms": max_fit_rms,
        "total_stamp_count": len(refined),
        "refine_claim_radius_fraction": claim_radius_fraction,
        "refine_brush_radius_spread_ratio": brush_radius_spread_ratio,
        "refine_method": method,
        "refine_rad_m": rad_m,
        "refine_use_height_mask": use_height_mask,
        "refine_model_rebuild_interval": model_rebuild_interval,
        "refine_candidate_brushes": list(candidate_brushes) if candidate_brushes is not None else None,
        "refine_max_planar_rms": max_planar_rms,
        "refine_planar_shrink_factor": planar_shrink_factor,
        "refine_use_slope_radius": use_slope_radius,
        "refine_use_variation_radius": use_variation_radius,
        "refine_variation_contrast_gamma": variation_contrast_gamma,
        "refine_density_weighted": density_weighted,
        "refine_subpixel_jitter_fraction": subpixel_jitter_fraction,
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


def step_output_terrain(working_dir: Path, registration_marks: bool = False) -> None:
    course_dir = working_dir / "course"

    print(f"Loading stamps from {working_dir} (initial + all refine passes)...")
    stamps = load_all_stamps(working_dir)
    print(f"  {len(stamps)} stamps")

    if registration_marks:
        marks = build_registration_mark_stamps(COURSE_SIZE_M)
        stamps = list(stamps) + marks
        print(f"  + {len(marks)} registration-mark stamps (one per corner)")

    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    heights = TerrainModel(stamps).render(resolution=200, bounds=bounds)
    true_min, true_max = float(heights.min()), float(heights.max())
    print(f"Normalizing heights: actual resolved range [{true_min:.3f}, {true_max:.3f}] m "
          f"-> shifting by {-true_min:.3f} m so minimum lands at 0")
    try:
        stamps = normalize_stamp_heights(stamps, bounds)
    except ValueError as e:
        raise StepError(str(e)) from e

    water_entries: list[dict] = []
    features_path = working_dir / FEATURES_FILE
    if features_path.exists():
        print("Building water objects from OSM water features (course_output/water.py)...")
        features = load_features(features_path)
        features = _crop_features_to_course(working_dir, features)
        water_features = [f for f in features if f.kind == "water"]
        water_entries = build_water_objects(water_features, stamps)
    else:
        print(f"  No {FEATURES_FILE} found -- skipping water objects (run --step ingest-osm first "
              "if this course has water hazards).")

    nodes_dir = course_dir / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(
            f"{nodes_dir} doesn't exist. Run --step ingest-course to extract a blank "
            f"starting .course into {course_dir} first."
        )

    out_path = nodes_dir / "userLayers.json"
    write_user_layers(stamps, out_path, water=water_entries)
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
    "dig-water": step_dig_water,
    "generate-terrain": step_generate_terrain,
    "refine-terrain": step_refine_terrain,
    "output-terrain": step_output_terrain,
    "write-splines": step_write_splines,
    "write-holes": step_write_holes,
    "generate-trees": step_generate_trees,
    "write-objects": step_write_objects,
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
    parser.add_argument("--fill-heightmap-gaps", action=argparse.BooleanOptionalAction, default=True,
                         help="ingest-laz: fill NaN heightmap gaps (water, buildings, other no-ground-"
                              "point areas) via harmonic inpainting -- iterative neighbor-average "
                              "relaxation, converging coarse-to-fine rather than a single-pass "
                              "flood-fill (see ingest/heightmap.py's fill_heightmap_gaps). On by "
                              "default; pass --no-fill-heightmap-gaps to leave gaps as NaN, excluded "
                              "from error scoring/fitting downstream (old behavior).")
    parser.add_argument("--pitch", type=float, default=None,
                         help="generate-terrain, hex method only: spacing (m) of the initial coarse "
                              "hex-grid stamp lattice (terrain/hexgrid.py's HEX_LATTICE_PITCH_M) -- "
                              "smaller pitch means more, smaller, more tightly-packed initial stamps. "
                              "Stamp radius and edge bleed both derive from this automatically "
                              "(radius=2*pitch, bleed=pitch). Default: use whatever's saved in "
                              f"project.json, or {HEX_LATTICE_PITCH_M} if never set.")
    parser.add_argument("--generate-terrain-method", type=str, default=None, choices=["hex", "contour"],
                         help="generate-terrain: 'hex' (default) is the flat hex lattice. 'contour' "
                              "traces elevation-band contours of the real heightmap and places stamps "
                              "along each ring, spaced by local curvature, with a distance-transform "
                              "gap-fill pass for flat interiors ring-tracing can't reach on its own -- "
                              "see terrain/contour_layers.py. Default: use whatever's saved in "
                              "project.json, or 'hex' if never set.")
    parser.add_argument("--band-spacing-m", type=float, default=None,
                         help="generate-terrain, contour method only: elevation spacing (m) defining "
                              "each band -- smaller means more, narrower bands. Default: use whatever's "
                              f"saved in project.json, or {DEFAULT_BAND_SPACING_M} if never set.")
    parser.add_argument("--fill-brush", type=int, default=None,
                         help="generate-terrain, contour method only: brush for the main tiered multi-"
                              "scale band fill -- type 8 (wide flat plateau) recommended, has the best "
                              "plateau fraction (least overhang) of the four brush types. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_FILL_BRUSH} if never set.")
    parser.add_argument("--min-radius", type=float, default=None,
                         help="generate-terrain, contour method only: smallest tier in the multi-scale "
                              "fill scan (m). Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_MIN_RADIUS_M} if never set.")
    parser.add_argument("--max-radius", type=float, default=None,
                         help="generate-terrain, contour method only: largest tier in the multi-scale "
                              "fill scan (m) -- the main level-of-detail knob: how big the biggest "
                              "stamps in a band are allowed to be. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_MAX_RADIUS_M} if never set.")
    parser.add_argument("--radius-step-ratio", type=float, default=None,
                         help="generate-terrain, contour method only: geometric (multiplicative) step "
                              "between tiers, scanning from --max-radius down to --min-radius -- each "
                              "tier's radius is the previous tier's radius times this ratio (0-1, not "
                              "a fixed meters step). Closer to 1.0 means more, finer-grained tiers at "
                              "the cost of more candidate lookups; automatically scales with whatever "
                              "--min-radius/--max-radius range you choose, unlike a fixed meters step. "
                              "Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RADIUS_STEP_RATIO} if never set.")
    parser.add_argument("--edge-distance-m", type=float, default=None,
                         help="generate-terrain, contour method only: pass-1-only buffer (m) past the "
                              "true band boundary that every candidate's plateau must additionally "
                              "clear, on top of just fitting within it -- leaves a strip along every "
                              "band edge for pass 2's finer crumb fill to handle instead of pass 1's "
                              "large hard stamps. Pass 2 always ignores this. 0 disables. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_EDGE_DISTANCE_M} if never "
                              "set.")
    parser.add_argument("--smoothing-brush", type=int, default=None,
                         help="generate-terrain, contour method only: brush for pass 2's crumb-scatter "
                              "fill over whatever pass 1 leaves as genuine crumbs -- a softer brush "
                              "(type 10 default) so small scattered crumbs blend rather than showing a "
                              "hard-edged patch. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_SMOOTHING_BRUSH} if never set.")
    parser.add_argument("--smoothing-min-radius", type=float, default=None,
                         help="generate-terrain, contour method only: pass 2's OWN radius floor, "
                              "independent of --min-radius (pass 1's) -- the crumb stage's scale is a "
                              "property of how it does its own job, not of how finely pass 1 happened "
                              "to be tiered. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_SMOOTHING_MIN_RADIUS_M} if never set.")
    parser.add_argument("--smooth-ratio", type=float, default=None,
                         help="generate-terrain, contour method only: pass 2's crumb-scatter radius as "
                              "a multiple of --smoothing-min-radius (default 4.0 -- a 16m scatter "
                              "radius at the 4m default floor). Deliberately a ratio, not an "
                              "independent absolute value: the crumb stage exists to reach across "
                              "whatever pass 1 couldn't fit, so its own scale should track its own "
                              "floor directly rather than needing separate re-tuning. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_CRUMB_SCATTER_MULTIPLIER} "
                              "if never set.")
    parser.add_argument("--smooth-claim-fraction", type=float, default=None,
                         help="generate-terrain, contour method only: \"eat\" -- how much of each "
                              "pass-2 crumb-scatter stamp's placed radius gets claimed. Deliberately "
                              "much heavier overlap (claim less) than pass 1's real-plateau-derived "
                              "claim, since pass 2's whole job is blanket-covering whatever pass 1's "
                              "large hard stamps couldn't reach, not precise packing. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_SMOOTH_CLAIM_FRACTION} "
                              "if never set.")
    parser.add_argument("--candidates-per-radius", type=int, default=None,
                         help="generate-terrain, contour method only: pass 1's random-candidate cap "
                              "per tier. Left unset, this is AUTO-TUNED once at the start of the run "
                              "(see --sweet-spot-* flags below) by searching a handful of sample bands "
                              "for the point of diminishing returns, rather than defaulting to a fixed "
                              "number -- set this explicitly, once you've seen a good auto-tuned value, "
                              "to skip re-running that calibration search on every subsequent run. Not "
                              "auto-persisted from an auto-tuned run -- you have to note the value "
                              "yourself and pass it back in.")
    parser.add_argument("--sweet-spot-ratio", type=float, default=None,
                         help="generate-terrain, contour method only, auto-tune only (ignored if "
                              "--candidates-per-radius is set explicitly): keep doubling "
                              "candidates_per_radius until pass 2's own stamp count drops to this "
                              "fraction of pass 1's -- past that point more candidates buys "
                              "diminishing pass-1 coverage while pass 2 does proportionally less "
                              "mop-up work. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_SWEET_SPOT_STAMP_RATIO} if never set.")
    parser.add_argument("--sweet-spot-sample-bands", type=int, default=None,
                         help="generate-terrain, contour method only, auto-tune only: how many "
                              "regularly-spaced bands to calibrate against, not every band -- bands "
                              "are similar enough in character that per-band tuning would mostly "
                              "repeat the same search for no benefit. Default: use whatever's saved "
                              f"in project.json, or {DEFAULT_SWEET_SPOT_SAMPLE_BANDS} if never set.")
    parser.add_argument("--sweet-spot-seeds", type=int, default=None,
                         help="generate-terrain, contour method only, auto-tune only: random seeds per "
                              "sampled band, so one lucky/unlucky seed doesn't skew the calibration -- "
                              "the MAX candidate count found across every band/seed combination is what "
                              "actually gets used. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_SWEET_SPOT_SEEDS} if never set.")
    parser.add_argument("--sweet-spot-max-candidates", type=int, default=None,
                         help="generate-terrain, contour method only, auto-tune only: safety cap on "
                              "the calibration search itself. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_SWEET_SPOT_MAX_CANDIDATES} if never set.")
    parser.add_argument("--sweet-spot-time-budget-s", type=float, default=None,
                         help="generate-terrain, contour method only, auto-tune only: hard wall-clock "
                              "ceiling (seconds) on the whole calibration search -- once exceeded, "
                              "whatever's best so far gets used, rather than calibration being able to "
                              "dominate total run time unpredictably. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_SWEET_SPOT_TIME_BUDGET_S} if never set.")
    parser.add_argument("--random-seed", type=int, default=None,
                         help="generate-terrain, contour method only: seeds pass 1's own randomness -- "
                              "each band gets random_seed + its own index, so the whole run is "
                              "reproducible given the same inputs, while still varying naturally band "
                              "to band. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RANDOM_SEED} if never set.")
    parser.add_argument("--denoise-px", type=int, default=None,
                         help="generate-terrain, contour method only: morphological open+close radius "
                              "(heightmap pixels) applied to each band's mask before filling -- trims "
                              "isolated few-pixel bumps and fills isolated few-pixel gaps, simplifying "
                              "the boundary before it fragments the fill into unnecessary tiny stamps. "
                              "0 disables. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_DENOISE_PX} if never set.")
    parser.add_argument("--max-stamps", type=int, default=None,
                         help="generate-terrain, contour method only: stop once this many stamps have "
                              "been placed in total -- a quick way to sanity-check a parameter "
                              "combination before committing to the full run. Bands fill ascending by "
                              "elevation, so the cutoff always lands on the low-elevation end and most "
                              "of the course will be genuinely unfilled, not just coarser -- this is a "
                              "partial-preview tool, not a real generation mode. Not saved to "
                              "project.json -- pass it explicitly each time.")
    parser.add_argument("--error-resolution", type=int, default=None,
                         help="visualize: grid resolution for preview_error.png, overriding the "
                              "default of inheriting whatever --resolution refine-terrain last used "
                              "(or a hardcoded 200 if refine-terrain hasn't run at all yet -- far too "
                              "coarse to localize a specific small feature; at RES~1000 a 200x200 "
                              "error grid averages ~5x5 real cells into one). Not saved to project.json "
                              "-- pass it explicitly each time you want a non-default resolution. "
                              "Higher values cost more (predicted-vs-actual is evaluated at every "
                              "cell), so start moderate (e.g. 500) before jumping to 1000+.")
    parser.add_argument("--dig-depth", type=float, default=None,
                         help="dig-water: how much (m) to lower heightmap.npz under each water "
                              "polygon. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_DIG_WATER_DEPTH_M} if never set.")
    parser.add_argument("--dig-buffer", type=float, default=None,
                         help="dig-water: inward (negative) buffer (m) applied to each water polygon "
                              "before determining which cells to lower -- lets the water plane object "
                              "(built from the ORIGINAL un-buffered polygon) clip slightly into the "
                              "surrounding terrain at the edges. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_DIG_WATER_BUFFER_M} if never set.")
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
    parser.add_argument("--method", type=str, default=None, choices=["adaptive", "scatter"],
                         help="refine-terrain: 'adaptive' (default) targets error hotspots; "
                              "'scatter' ignores error and places well-spaced random stamps flattened "
                              "to real local LIDAR average -- see adaptive_refine.py's scatter_stamps. "
                              "Default: use whatever's saved in project.json, or 'adaptive' if never set.")
    parser.add_argument("--rad-m", type=float, default=None,
                         help="refine-terrain: literal target stamp radius (m) for this pass -- "
                              "'adaptive' uses this as max_radius (min_radius derives from the fixed "
                              "0.5 ratio DEFAULT_MIN/MAX_HOTSPOT_RADIUS_M already used); 'scatter' "
                              "uses it as the literal per-stamp placement radius before jitter. "
                              "Replaces the old radius_decay_per_pass percentage -- the implied decay "
                              "vs. the last run (last_refine_rad_m / rad_m) is now a derived, "
                              "informational value only, saved to project.json for display. Default: "
                              f"use whatever's saved in project.json, or {DEFAULT_RAD_M} if never set.")
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
    parser.add_argument("--model-rebuild-interval", type=int, default=None,
                         help="refine-terrain: fold every hotspot placed so far this pass into the "
                              "model (and re-derive the error grid from it) every N new hotspots, "
                              "instead of only ever fitting against the pre-pass baseline -- fixes a "
                              "real staleness gap where claim_radius_fraction<1 lets same-pass "
                              "hotspots overlap, but candidates were fit blind to each other. Lower "
                              "= more accurate but slower (rebuilds the whole error grid each time); "
                              f"default: use whatever's saved in project.json, or {DEFAULT_MODEL_REBUILD_INTERVAL} "
                              "if never set.")
    parser.add_argument("--candidate-brushes", type=str, default=None,
                         help="refine-terrain: comma-separated brush types to consider per hotspot, "
                              "e.g. '10,54' to restrict to only the two brushes with no flat plateau "
                              "(smooth, cosine-like falloff the whole way from center to edge) -- "
                              "excluding 8/9 avoids the flat-topped 'crater' look densely-packed small "
                              "8/9 stamps can produce, at some cost to how precisely a wide flat area "
                              "can hit an exact target height. Default: use whatever's saved in "
                              "project.json, or all four (8,9,10,54) if never set.")
    parser.add_argument("--registration-marks", action="store_true",
                         help="output-terrain/write-splines: add a small type-73 (circle) raise "
                              "stamp and a matching 5m circle spline (cart path surface) at each of "
                              "the 4 course corners (5m inset from each edge) -- for visually "
                              "confirming in-game that terrain and splines land exactly where "
                              "expected, and that the game isn't scaling/repositioning either one "
                              "unexpectedly. Opt-in; off by default.")
    parser.add_argument("--height-mask-buffer-px", type=float, default=DEFAULT_HEIGHT_MASK_BUFFER_PX,
                         help="ingest-osm: buffer (grow) the merged fairway+green outline by this many "
                              "pixels before rasterizing -- 1 pixel = 1 m, since the course is exactly "
                              f"2000x2000 m (default: {DEFAULT_HEIGHT_MASK_BUFFER_PX})")
    parser.add_argument("--hole-corridor-buffer-px", type=float, default=DEFAULT_HOLE_CORRIDOR_BUFFER_PX,
                         help="ingest-osm: buffer each hole routing centerline (tee-to-green line) by "
                              "this many pixels/meters before it contributes to the height mask -- an "
                              "unbuffered centerline alone would leave most of the actual playing "
                              "corridor outside the mask. This is a BUFFER (roughly half the resulting "
                              f"corridor width), not the total width (default: {DEFAULT_HOLE_CORRIDOR_BUFFER_PX})")
    parser.add_argument("--max-planar-rms", type=float, default=None,
                         help="refine-terrain: shrink a hotspot's radius (before claim_radius_fraction/ "
                              "brush_radius_spread_ratio apply) until the region's actual LIDAR heights "
                              "fit a single tilted plane within this RMS (m) -- catches valleys/ridges/"
                              "creases that an error-sign-only region never stops growing across (a "
                              "V-shaped cross-section stays one sign from floor to rim, so it gets "
                              "averaged into one stamp that pulls the floor up and the rim down). "
                              "Default: use whatever's saved in project.json, or off (None) if never set.")
    parser.add_argument("--planar-shrink-factor", type=float, default=None,
                         help="refine-terrain: multiplier (< 1.0) applied to a hotspot's radius each "
                              "time it fails the max_planar_rms check, until it passes or hits "
                              "min_radius. Only matters when --max-planar-rms is set. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_PLANAR_SHRINK_FACTOR} if never set.")
    parser.add_argument("--use-slope-radius", action=argparse.BooleanOptionalAction, default=None,
                         help="refine-terrain, scatter method only: drive each stamp's radius from "
                              "real local terrain slope (np.gradient over the whole grid, computed "
                              "once) instead of random jitter -- flat ground gets large stamps, steep "
                              "ground (valleys, ridges) gets small ones. Poisson-disc spacing becomes "
                              "radius-aware to match (two stamps of different sizes need to be farther "
                              "apart than two small or two large ones alike -- see terrain/"
                              "adaptive_refine.py's scatter_stamps). --planar-shrink-factor is reused "
                              "as the 'how small can it shrink on the steepest ground' floor, same "
                              "role it already plays for plain random jitter when this is off. Default: "
                              "use whatever's saved in project.json, or off if never set.")
    parser.add_argument("--use-variation-radius", action=argparse.BooleanOptionalAction, default=None,
                         help="refine-terrain, scatter method only: drive each stamp's radius from "
                              "RMS-from-local-mean at lag=RAD (real curvature AND macro-scale slope "
                              "carried across the window) instead of raw gradient magnitude -- the "
                              "corrected replacement for --use-slope-radius (a tilted-plane-forgiving "
                              "slope reading treats a gentle multi-km fairway grade as 'steep' "
                              "everywhere, even though a wide flat stamp represents it fine at a small "
                              "enough radius; RMS-from-local-mean instead only shrinks once the "
                              "accumulated rise across that radius actually matters). Wins over "
                              "--use-slope-radius if both are set. --planar-shrink-factor is the shrink "
                              "floor, same role as for --use-slope-radius. Default: use whatever's "
                              "saved in project.json, or off if never set.")
    parser.add_argument("--variation-contrast-gamma", type=float, default=None,
                         help="refine-terrain, scatter + --use-variation-radius only: exponent applied "
                              "to the normalized variation field before mapping into [RAD * SHR%%, RAD] "
                              "-- >1 sharpens toward the extremes (only genuinely high-variation cells "
                              "shrink much; mid-variation terrain stays closer to full radius), 1.0 is "
                              "a plain linear map. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_VARIATION_CONTRAST_GAMMA} if never set.")
    parser.add_argument("--density-weighted", action=argparse.BooleanOptionalAction, default=None,
                         help="refine-terrain, scatter method only: draw candidate sites from a "
                              "~1/radius^2-weighted density field (over the same heightmap grid) "
                              "instead of uniform-random over the whole course -- fixes what shrinking "
                              "radius alone can't: a smaller target radius previously only changed how "
                              "big an accepted dart was, never how often darts landed there, so small "
                              "high-detail regions got isolated small stamps reading as random bumps "
                              "instead of a tightly-packed cluster. Has no effect without "
                              "--use-slope-radius or --use-variation-radius also set. Default: use "
                              "whatever's saved in project.json, or off if never set.")
    parser.add_argument("--subpixel-jitter-fraction", type=float, default=None,
                         help="refine-terrain, scatter + --density-weighted only: fraction of a "
                              "heightmap cell's width to jitter density-weighted draws by, off the "
                              "exact cell center -- dither only, to avoid visibly grid-aligned stamp "
                              "centers (the course never needs sub-cell precision on its own). Default: "
                              f"use whatever's saved in project.json, or {DEFAULT_SUBPIXEL_JITTER_FRACTION} "
                              "if never set.")
    parser.add_argument("--max-new-stamps", type=int, default=None,
                         help="refine-terrain: cap on new detail stamps per pass (default: no cap)")
    parser.add_argument("--course-file", type=Path, default=None,
                         help="ingest-course: path to the .course file to extract")
    parser.add_argument("--game-version", type=str, default=None, choices=GAME_VERSIONS,
                         help="Project-level target game version -- selects which of objects.py's "
                              f"placedObjects2 schemas write-objects writes (implemented: "
                              f"{IMPLEMENTED_GAME_VERSIONS}; the rest are accepted here but will "
                              "raise a clear error until their schema is confirmed -- see "
                              "objects.py's module docstring). Persists to project.json when given, "
                              "same as the GUI's top-level Game version selector. Default: use "
                              f"whatever's saved in project.json, or {DEFAULT_GAME_VERSION} if never set.")
    parser.add_argument("--theme", type=str, default=None,
                         help="write-objects (game_version=2019 only): theme id or name (see "
                              "objects.py's THEMES_V2019) controlling which tree types are available. "
                              "Default: use whatever's saved in project.json, or a single generic tree "
                              "type if never set (not an error).")
    parser.add_argument("--tree-variety", action=argparse.BooleanOptionalAction, default=None,
                         help="write-objects (game_version=2019 only): use the full set of the "
                              "theme's tree types (normal + skinny) instead of one generic type. "
                              "Default: use whatever's saved in project.json, or ON if never set.")
    parser.add_argument("--tree-asset-path", dest="tree_asset_paths", action="append", default=None,
                         help="write-objects (game_version=2021+ only): a Unity asset path (e.g. "
                              "'Assets/Trees/OakA') to draw "
                              "from for any tree without a more specific --tree-type-asset-path match. "
                              "Repeatable for a variety pool. Default: use whatever's saved in "
                              "project.json, or none if never set.")
    parser.add_argument("--tree-type-asset-path", dest="tree_type_asset_paths", action="append", default=None,
                         metavar="TAG=PATH",
                         help="write-objects (game_version=2021+ only): map a tree node's "
                              "pga_tree_type tag value to a specific "
                              "asset path, e.g. 'oak=Assets/Trees/BigOak' -- overrides the general "
                              "--tree-asset-path pool for just that tree. Repeatable. Default: use "
                              "whatever's saved in project.json, or none if never set.")
    parser.add_argument("--stake-asset-path", type=str, default=None,
                         help="write-objects: Unity asset path for a stake placed at every corner of "
                              "every 'building' feature. Omit to skip stakes entirely. Default: use "
                              "whatever's saved in project.json, or none if never set.")
    parser.add_argument("--detect-lidar-trees", action=argparse.BooleanOptionalAction, default=None,
                         help="generate-trees: also detect individual trees directly from LIDAR canopy "
                              "points (ingest/tree_detection.py), added on top of any OSM natural=tree "
                              "nodes. Confined to height_mask.geojson's core-play-area polygon if one "
                              "exists (the game's own procedural vegetation fill is expected to handle "
                              "everywhere else). Needs heightmap.npz and pointcloud.npz (--step "
                              "ingest-laz). Default: use whatever's saved in project.json, or ON if "
                              "never set (OSM alone typically finds few or no trees on a real course).")
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
            step_ingest_laz(working_dir, args.projection, args.fill_heightmap_gaps)
        elif args.step == "ingest-osm":
            step_ingest_osm(working_dir, args.height_mask_buffer_px, args.hole_corridor_buffer_px)
        elif args.step == "ingest-course":
            if args.course_file is None:
                print("error: --step ingest-course requires --course-file <path>", file=sys.stderr)
                return 1
            step_ingest_course(working_dir, args.course_file)
        elif args.step == "dig-water":
            step_dig_water(working_dir, args.dig_depth, args.dig_buffer)
        elif args.step == "generate-terrain":
            step_generate_terrain(
                working_dir,
                pitch=args.pitch,
                method=args.generate_terrain_method,
                band_spacing_m=args.band_spacing_m,
                fill_brush=args.fill_brush,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                radius_step_ratio=args.radius_step_ratio,
                edge_distance_m=args.edge_distance_m,
                smoothing_brush=args.smoothing_brush,
                smoothing_min_radius=args.smoothing_min_radius,
                smooth_ratio=args.smooth_ratio,
                smooth_claim_fraction=args.smooth_claim_fraction,
                candidates_per_radius=args.candidates_per_radius,
                sweet_spot_ratio=args.sweet_spot_ratio,
                sweet_spot_sample_bands=args.sweet_spot_sample_bands,
                sweet_spot_seeds=args.sweet_spot_seeds,
                sweet_spot_max_candidates=args.sweet_spot_max_candidates,
                sweet_spot_time_budget_s=args.sweet_spot_time_budget_s,
                random_seed=args.random_seed,
                denoise_px=args.denoise_px,
                max_stamps=args.max_stamps,
            )
        elif args.step == "refine-terrain":
            parsed_candidate_brushes = (
                tuple(int(b.strip()) for b in args.candidate_brushes.split(","))
                if args.candidate_brushes else None
            )
            step_refine_terrain(working_dir, args.error_tolerance, args.resolution,
                                 args.min_hotspot_radius_cells, args.max_new_stamps,
                                 args.claim_radius_fraction, args.brush_radius_spread_ratio,
                                 args.method, args.use_height_mask, args.mask_buffer_px,
                                 args.model_rebuild_interval, parsed_candidate_brushes,
                                 args.max_planar_rms, args.planar_shrink_factor, args.rad_m,
                                 args.use_slope_radius, args.use_variation_radius,
                                 args.variation_contrast_gamma, args.density_weighted,
                                 args.subpixel_jitter_fraction)
        elif args.step == "output-terrain":
            step_output_terrain(working_dir, registration_marks=args.registration_marks)
        elif args.step == "write-splines":
            step_write_splines(working_dir, registration_marks=args.registration_marks)
        elif args.step == "write-holes":
            step_write_holes(working_dir)
        elif args.step == "generate-trees":
            step_generate_trees(working_dir, args.detect_lidar_trees)
        elif args.step == "write-objects":
            tree_type_asset_paths = None
            if args.tree_type_asset_paths is not None:
                tree_type_asset_paths = {}
                for pair in args.tree_type_asset_paths:
                    if "=" not in pair:
                        raise StepError(f"--tree-type-asset-path expects TAG=PATH, got: {pair!r}")
                    tag, _, asset_path = pair.partition("=")
                    tree_type_asset_paths[tag] = asset_path
            step_write_objects(
                working_dir, args.game_version, _resolve_theme(args.theme), args.tree_variety,
                args.tree_asset_paths, tree_type_asset_paths, args.stake_asset_path,
            )
        elif args.step == "repack":
            if not args.repack_filename:
                print("error: --step repack requires --repack-filename <name>", file=sys.stderr)
                return 1
            step_repack(working_dir, args.repack_filename)
        elif args.step == "visualize":
            step_visualize(working_dir, overwrite_current_version=True, error_resolution=args.error_resolution)
    except StepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
