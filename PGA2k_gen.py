#!/usr/bin/env python3
"""
PGA2k_gen.py

CLI orchestrator for the terrain compiler. Operates on a working
directory, running one pipeline step at a time:

    PGA2k_gen.py <working_dir>                       (same as --step init)
    PGA2k_gen.py <working_dir> --step init
    PGA2k_gen.py <working_dir> --step ingest-laz [--projection <EPSG>] [--no-fill-heightmap-gaps]
    PGA2k_gen.py <working_dir> --step ingest-osm
    PGA2k_gen.py <working_dir> --step ingest-course [--course-theme <name>]
                                 (optional -- resets course/ from the bundled template;
                                 every other write/repack step provisions it automatically)
    PGA2k_gen.py <working_dir> --step push-blank-template [--blank-course-name <name>]
                                 (builds a fresh named blank .course from the current
                                 game_version + theme template -> working_dir/blank_template.course;
                                 does not touch course/)
    PGA2k_gen.py <working_dir> --step dig-water [--dig-depth M] [--dig-buffer M]
    PGA2k_gen.py <working_dir> --step generate-terrain
    PGA2k_gen.py <working_dir> --step generate-streams
    PGA2k_gen.py <working_dir> --step generate-oob [--oob-inner-buffer M] [--oob-band-width M]
                                 [--oob-merge-gap M] [--oob-simplify-tol M] [--oob-cap-ratio R]
                                 [--oob-no-caps] [--oob-clear]
                                 (auto out-of-bounds paint: a brush-stamp band -- round type-8
                                 caps + stretched type-15 squares -- just outside the playable
                                 area -> oob.json; folded into userLayers.json by write-terrain.
                                 --oob-clear deletes it; re-run write-terrain to apply)
    PGA2k_gen.py <working_dir> --step generate-collections [--collection-library <dir>]
    PGA2k_gen.py <working_dir> --step generate-parking [--parking-spacing M] [--parking-offset M]
                                 [--parking-sides left|right|both] [--parking-orientation
                                 perpendicular|parallel] [--parking-skip-prob P]
                                 [--parking-max-variants N] [--no-parking-bank]
                                 (lines every OSM pga_parking=<pool> way with parked-car props
                                 -> parking.json; folded into objects.json by pack-objects)
    PGA2k_gen.py <working_dir> --step push-collection --collection-name <name>
                                 [--collection-library <dir>] [--blank-course-name <name>]
                                 (builds a fresh .course holding one collection template's
                                 objects/splines/stamps at the course centre for in-game
                                 editing -> working_dir/pushed_collection.course; does not
                                 touch course/)
    PGA2k_gen.py <working_dir> --step refine-terrain [--error-tolerance M] [--resolution N]
                                 [--method adaptive|scatter] [--rad-m M]
    PGA2k_gen.py <working_dir> --step write-terrain [--registration-marks] [--direct-height-shift]
    PGA2k_gen.py <working_dir> --step write-water [--registration-marks] [--direct-height-shift]
                                 [--multi-tile-water]
                                 [--water-tile-tolerance-m M] [--water-tile-min-edge-m M]
                                 [--water-tile-max-search-m M] [--water-tile-width-samples N]
                                 [--water-tile-redundancy-ratio R] [--water-tile-overlap-m M]
                                 [--water-fill-mode edge|stripe] [--water-stripe-overlap-m M]
                                 [--water-stripe-min-edge-m M] [--water-stripe-tolerance-m M]
                                 [--water-stripe-max-stripes-per-side N]
    PGA2k_gen.py <working_dir> --step generate-trees [--detect-lidar-trees] [--mark-cartpath-trees]
    PGA2k_gen.py <working_dir> --step write-objects [--game-version <2019|2021|2023|2025>]
                                 [--theme <id-or-name>] [--tree-variety] [--stake-buildings]
                                 [--tree-asset-path <path>]...
                                 [--tree-type-asset-path <TAG=path>]... [--stake-asset-path <path>]  (2021+)
    PGA2k_gen.py <working_dir> --step repack --repack-filename <name>
    PGA2k_gen.py <working_dir> --step import-ingame-edits --edited-course <path.course> [--commit]
                                 [--import-group <label>]
                                 (reconciles a saved, hand-edited .course against what this tool
                                 currently tracks -- dry-run by default, prints a diff summary;
                                 --commit appends new objects to ingame_objects.json and new
                                 terrain stamps as the next stamps_N.json layer)

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
    stamps/stamps_N.json      one file per layering pass (generate-terrain,
                             refine-terrain, generate-cart-paths, ...), in
                             the order they were run -- each holds only the
                             stamps THAT pass added (not a cumulative
                             snapshot), and later layers take precedence
                             over earlier ones wherever they overlap;
                             deleting the highest N undoes that pass
    course/                  extracted blank .course, always at this fixed
                             path -- auto-provisioned from the bundled
                             templates/{game_version}_{theme}.course on first
                             write/repack (see _ensure_course_baseline);
                             --step ingest-course resets it explicitly
    ingame_objects.json       objects a user placed by hand-editing an
                             exported .course in the game's own editor,
                             captured back via --step import-ingame-edits;
                             read (not overwritten) by pack-objects

Step ordering is enforced with clear errors (e.g. generate-terrain
without a pointcloud.npz on disk yet) rather than letting a later step
fail on a confusing missing-file exception.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent

from constants import (
    COURSE_SIZE_M, PREVIEW_COMPOSITE, PREVIEW_ERROR, PREVIEW_HEIGHT, PREVIEW_HEX,
    PREVIEW_LIDAR, PREVIEW_LIDAR_GROUND, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_MASK, PREVIEW_OOB, PREVIEW_OSM,
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
    DEFAULT_HEIGHT_MASK_BUFFER_PX, DEFAULT_HOLE_CORRIDOR_BUFFER_PX, Feature, build_height_mask,
    crop_features, load_features, load_height_mask, merge_height_mask_features, parse_osm_features,
    rasterize_mask, save_features, save_height_mask, shift_features,
)
from course_output.splines import (
    build_registration_mark_splines, build_surface_splines, feature_to_spline, save_surface_splines,
)
from course_output.holes import build_holes, save_holes
from course_output.object_clusters import (
    CLUSTER_FILL_MODE_AUTO, CLUSTER_FILL_MODE_STAMPS, CLUSTER_FILL_SOURCE_STREAM,
    PGA_CLUSTER_FILLS_TAG, SYNTHETIC_BORDER_CENTERLINE_KIND,
    SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND,
    cluster_records_to_v2019_groups, next_synthetic_osm_id, pack_cluster_records, pack_spline_records,
)
from course_output.objects import (
    DEFAULT_GAME_VERSION, DEFAULT_STAKE_ASSET_PATH_V2021, GAME_VERSIONS, IMPLEMENTED_GAME_VERSIONS,
    THEMES_V2019, TREE_TYPE_TAG,
    WATERFALL_DEFAULT_ASSET_PATH, WATERSPLASH_DEFAULT_ASSET_PATH, apply_area_tree_type_hints,
    build_building_stake_objects_v2019,
    build_building_stake_objects_v2021, build_tree_objects_v2019, build_tree_objects_v2021,
    build_waterfall_objects_v2019, build_waterfall_objects_v2021,
    build_watersplash_objects_v2019, build_watersplash_objects_v2021, cluster_records_to_v2021_groups,
    default_tree_asset_paths_v2021,
    lidar_trees_to_tagged, load_object_list,
    load_objects, load_placed_objects, load_tree_theme_species, merge_object_groups, move_trees_off_cartpaths,
    object_counts, object_spline_fill_records_to_v2021_groups, parse_osm_trees, schema_for,
    save_object_list, save_objects, save_placed_objects,
)
from course_output.course_templates import resolve_course_template
from course_output.game_versions import VERSION_SCHEMAS
from course_output.collection_library import (
    _flatten_datum as _collection_flatten_datum,
    _slug as _collection_slug,
    default_library_dir, load_library,
)
from course_output.collections import (
    _bearing_deg as _collection_bearing_deg,
    apply_terrain_heights, build_collection_objects_v2019, build_collection_objects_v2021,
    build_collection_splines, build_collection_stamps, iter_collection_objects, load_collection_records,
    resolve_collection, save_collection_records,
)
from course_output.parking import (
    PARKING_ACCENT_COUNT, PARKING_COLOR_WEIGHTS, PARKING_MAX_VARIANTS, PARKING_OFFSET_M,
    PARKING_ORIENTATION, PARKING_SIDES, PARKING_SKIP_PROB, PARKING_SPACING_M,
    ParkingAisle, build_parking_records, iter_parking_cars, load_parking_records,
    parse_color_weights, resolve_pool, save_parking_records,
)
from course_output.ingame_objects import (
    build_ingame_objects_v2019, build_ingame_objects_v2021, load_ingame_objects, remove_ingame_object_groups,
    save_ingame_objects, summarize_ingame_object_groups,
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
from terrain.hexgrid import (
    DEFAULT_BRUSH as HEX_DEFAULT_BRUSH,
    HEX_DEFAULT_SPREAD_RATIO,
    HEX_LATTICE_PITCH_M,
    generate_hex_grid,
)
from terrain.rastergrid import (
    RASTER_BRUSH,
    RASTER_SIZES,
    DEFAULT_RASTER_SIZE,
    DEFAULT_RASTER_SPREAD_RATIO,
    DEFAULT_RASTER_CENTER_BIAS_RATIO,
    generate_raster_grid,
)
from terrain.contour_layers import (
    DEFAULT_BAND_SPACING_M,
    DEFAULT_FILL_MODE,
    DEFAULT_FILL_BRUSH,
    DEFAULT_MIN_RADIUS_M,
    DEFAULT_MAX_RADIUS_M,
    DEFAULT_RADIUS_STEP_RATIO,
    DEFAULT_EDGE_DISTANCE_M,
    DEFAULT_RECT_BRUSH,
    DEFAULT_RECT_TOLERANCE_M,
    DEFAULT_RECT_MIN_LENGTH_M,
    DEFAULT_RECT_MAX_SEARCH_DISTANCE_M,
    DEFAULT_RECT_WIDTH_SAMPLES,
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
    DEFAULT_N_WORKERS,
    TYPE72_PLATEAU_FRACTION,
    generate_contour_layers,
)
from terrain.stamp import TOOL_FLATTEN, TOOL_RAISE, Stamp
from terrain.cart_paths import (
    CART_PATH_STAMP_TYPE, CART_PATH_STAMP_RADIUS, CART_PATH_SPACING_M,
    CART_PATH_HEIGHT_AVG_RADIUS_M, CartPathSpline, bezier_waypoints_to_linestring,
    generate_cart_path_stamps,
)
from terrain.terrain_model import TerrainModel
from terrain.stamp_containment import stamps_fully_within
from terrain.streams import (
    BANK_VEG_WIDTH_M, STREAM_DEPTH_M, STREAM_HALF_WIDTH_M, STREAM_WATER_BASE_WIDTH_M,
    STREAM_WATER_FILL_DEPTH_M, STREAM_WATER_LEVEL_MARGIN_M, STREAM_WATER_WIDEN_PER_DEPTH,
    STREAM_WATER_WIDEN_PER_DESCENT, StreamCenterline, build_stream_records, generate_stream_stamps,
    load_stream_records, rebuild_stream_drop_rows, save_stream_records,
)
from course_output.userLayers import (
    GRID_ORIGIN_OFFSET, build_baseline_flatten_stamp, build_registration_mark_stamps,
    normalize_stamp_heights, normalize_stamp_heights_by_value_shift, stamp_to_entry, write_user_layers,
)
from course_output.out_of_bounds import (
    OOB_BAND_WIDTH_M, OOB_CAP_SCALE_RATIO, OOB_INCLUDE_CAPS, OOB_INNER_BUFFER_M, OOB_MERGE_GAP_M,
    OOB_SIMPLIFY_TOL_M, build_oob_records, load_oob_records, oob_records_to_entries, save_oob_records,
)
from course_output.water import (
    build_water_objects, build_stream_water_objects,
    DEFAULT_WATER_TILE_TOLERANCE_M, DEFAULT_WATER_TILE_MIN_EDGE_M,
    DEFAULT_WATER_TILE_MAX_SEARCH_M, DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    DEFAULT_WATER_TILE_REDUNDANCY_RATIO, DEFAULT_WATER_TILE_OVERLAP_M,
    DEFAULT_WATER_STRIPE_OVERLAP_M, DEFAULT_WATER_STRIPE_TOLERANCE_M,
    DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
)

FEATURES_FILE = "features.geojson"
HEIGHT_MASK_FILE = "height_mask.geojson"
HEIGHTMAP_FILE = "heightmap.npz"
STAMPS_PATTERN = "stamps_{n}.json"
OBJECT_LIST_FILE = "object_list.json"
OBJECTS_FILE = "objects.json"
STREAMS_FILE = "streams.json"
COLLECTIONS_FILE = "collections.json"
PARKING_FILE = "parking.json"
OOB_FILE = "oob.json"
INGAME_OBJECTS_FILE = "ingame_objects.json"

# This project's own OSM tag (not an OSM standard) on a 2-node way that
# places a reusable object/spline collection -- value is the template
# name (see course_output/collection_library.py, ingest/osm.py's
# classify_way "collection" kind, and step_generate_collections).
PGA_COLLECTION_TAG = "pga_collection"

# Optional companion tag on the same way -- a free-form value passed
# through to resolve_collection so one template can adapt per placement
# (a hole sign that swaps by hole number, etc.) via member "variants" /
# a "{param}" asset-path token. See course_output/collections.py.
PGA_PARAMETER_TAG = "pga_parameter"

# This project's own OSM tag (not an OSM standard) on a way to be lined
# with parked-car props -- value is a pool filter ("yes"/"all" or a comma
# list of colours and/or "car"/"van"). See course_output/parking.py,
# ingest/osm.py's classify_way "parking" kind, and step_generate_parking.
PGA_PARKING_TAG = "pga_parking"

# Companion tag on the same way -- an occupancy scale in [0, 1]: the
# populated fraction of that aisle is (1 - parking_skip_prob) * ref
# (default 1.0). Named `ref` (not pga_-prefixed) at the user's request.
PGA_PARKING_REF_TAG = "ref"

# OSM waterway tag values treated as linear streams (carved bed + flowing
# water + bank vegetation), as opposed to filled water bodies.
STREAM_WATERWAY_KINDS = ("stream", "ditch")

# Marker tag on the synthetic stream-bank Features step_generate_streams
# creates, so a re-run can find and replace its own (see that step).
STREAM_BANK_MARKER_TAG = "pga_stream_bank"

# Default cluster-fill specs tagged onto each stream-bank polygon --
# grass + waterside rock + lily ground-cover, all real
# course_output/asset_catalog.json (category, type) pairs. Overridable
# per project via project.json's "streams_bank_veg_specs".
# grass + ground-cover default to mode="auto" (object-spline fill on
# v2021+, circle-scatter on v2019 -- see object_clusters.resolve_fill_mode);
# rock stays circle-scatter on every version (reads better that way).
DEFAULT_STREAM_BANK_VEG_SPECS = [
    {"category": 2, "type": 0, "ratio": 1.0, "density": 60.0,
     "mode": CLUSTER_FILL_MODE_AUTO, "source": CLUSTER_FILL_SOURCE_STREAM},
    {"category": 1, "type": 7, "ratio": 1.0, "density": 25.0,
     "mode": CLUSTER_FILL_MODE_STAMPS, "source": CLUSTER_FILL_SOURCE_STREAM},
    {"category": 3, "type": 1, "ratio": 1.0, "density": 30.0,
     "mode": CLUSTER_FILL_MODE_AUTO, "source": CLUSTER_FILL_SOURCE_STREAM},
]

DEFAULT_DIG_WATER_DEPTH_M = 3.0
DEFAULT_DIG_WATER_BUFFER_M = 1.0

DEFAULT_REMOVE_COVERED_MARGIN_M = 2.0


def _stamps_dir(working_dir: Path) -> Path:
    return working_dir / STAMPS_DIR


def _stamps_files(working_dir: Path) -> list[Path]:
    """Every stamps_N.json present under stamps/, in order (N=1, 2, 3, ...)."""
    files = []
    n = 1
    while (_stamps_dir(working_dir) / STAMPS_PATTERN.format(n=n)).exists():
        files.append(_stamps_dir(working_dir) / STAMPS_PATTERN.format(n=n))
        n += 1
    return files


def load_all_stamps(working_dir: Path) -> list[Stamp]:
    """
    Reconstruct the full, EFFECTIVE current stamp list: every
    stamps_N.json under stamps/, in order, excluding any stamp whose
    `blocked_by` names a layer_id that belongs to a layer file still on
    disk right now (see _flag_previous_layer_blocked).

    Every layering pass (generate-terrain, refine-terrain, generate-
    cart-paths, ...) writes only the stamps *it* added, not a
    cumulative snapshot -- so deleting the highest-numbered
    stamps_N.json is a natural undo of just the most recent pass, and
    every earlier pass's file stays exactly as it was, EXCEPT for the
    one narrow case where a later masked pass annotated some of its
    stamps with `blocked_by` (never a deletion/reorder, so still
    "the same layer, plus one optional field on specific entries").
    That's also why blocking is undo/redo-transparent for free: the
    check below is purely "does layer_id X currently exist," so
    renaming a blocking layer away (undo) or back (redo) toggles every
    stamp it blocked without touching anything else.
    """
    files = _stamps_files(working_dir)
    if not files:
        raise StepError(
            f"No stamp layers found under {_stamps_dir(working_dir)}. "
            "Run --step generate-terrain first."
        )

    layers = [load_stamp_file(path) for path in files]  # [(stamps, blocked_by, metadata), ...]
    active_layer_ids = {metadata["layer_id"] for _, _, metadata in layers if metadata.get("layer_id")}

    stamps: list[Stamp] = []
    for layer_stamps, blocked_by, _metadata in layers:
        for stamp, blocker in zip(layer_stamps, blocked_by):
            if blocker is not None and blocker in active_layer_ids:
                continue
            stamps.append(stamp)
    return stamps


def _load_collection_terrain_stamps(working_dir: Path, verbose: bool = True) -> list[Stamp]:
    """Every placed object-collection's raise-tool terrain stamps
    (collections.json -- see step_generate_collections /
    course_output/collections.py), or [] if none are placed. Factored
    out of _load_all_stamps_incl_collections so callers that need just
    the collection stamps (without re-walking stamps_N.json) can get
    them directly, without persisting them as a stamps_N.json layer."""
    collections_path = working_dir / COLLECTIONS_FILE
    if not collections_path.exists():
        return []
    coll_stamps = build_collection_stamps(load_collection_records(collections_path))
    if coll_stamps and verbose:
        print(f"  + {len(coll_stamps)} terrain stamp(s) from placed collections")
    return coll_stamps


def _load_all_stamps_incl_collections(working_dir: Path, verbose: bool = True) -> tuple[list[Stamp], bool]:
    """(stamps, added_collection_stamps) -- load_all_stamps plus every
    placed object-collection's raise-tool terrain stamps (collections.json
    -- see step_generate_collections / course_output/collections.py).
    Appended last, so they compose on top of the generated terrain, and
    formatted at write time rather than persisted as a stamps_N.json layer
    -- re-running generate-collections never double-applies them. Shared by
    _load_normalized_stamps (write-terrain / write-water),
    step_write_objects' elevation lookup, and step_visualize so they all
    agree on the same terrain.
    NOTE deliberately NOT used by step_refine_terrain -- collection berms
    aren't in the LIDAR target, so the refiner shouldn't try to fit them."""
    stamps = load_all_stamps(working_dir)
    coll_stamps = _load_collection_terrain_stamps(working_dir, verbose=verbose)
    if coll_stamps:
        stamps = list(stamps) + coll_stamps
        return stamps, True
    return stamps, False


def load_latest_stamp_metadata(working_dir: Path) -> dict | None:
    """
    Metadata (step/parameters/timestamp/... -- see save_stamp_file)
    from the most recent stamps_N.json, or None if no layer has been
    written yet. Used to label previews with whatever settings
    actually produced the terrain being looked at, whichever step
    (generate-terrain, refine-terrain, generate-cart-paths, ...) wrote
    that layer.
    """
    files = _stamps_files(working_dir)
    if not files:
        return None
    _, _, metadata = load_stamp_file(files[-1])
    return metadata


# The terrain previews step_visualize versions one-per-layering-pass (the
# same set the GUI's _find_undo_group renames aside on Undo). Kept in
# lockstep with the stamps_N.json count.
_TERRAIN_PREVIEW_KINDS = (
    PREVIEW_HEX, PREVIEW_STAMPS, PREVIEW_HEIGHT, PREVIEW_LIDAR_GROUND,
    PREVIEW_COMPOSITE, PREVIEW_ERROR,
)


def _clear_trailing_stream_layers(working_dir: Path, printf=print) -> int:
    """Delete the contiguous run of generate-streams stamp layer(s) sitting
    at the TOP of the stack, plus the newest preview version of each terrain
    kind per layer removed (so preview-count / stamp-count stay in lockstep,
    exactly like the GUI's Undo). Stops at the first non-generate-streams
    layer -- a buried stream layer is left alone (the caller warns). Lets a
    re-run of generate-streams REPLACE its trench instead of stacking a
    second (possibly deeper/wider) carve under it. Returns the count removed.
    """
    files = _stamps_files(working_dir)
    trailing: list[Path] = []
    for path in reversed(files):
        if load_stamp_file(path)[2].get("step") == "generate-streams":
            trailing.append(path)
        else:
            break
    if not trailing:
        return 0
    for path in trailing:
        path.unlink()
        printf(f"  cleared previous generate-streams layer {path.name}")
    preview_dir = working_dir / PREVIEW_DIR
    for kind in _TERRAIN_PREVIEW_KINDS:
        for version in viz.find_all_preview_versions(preview_dir, kind)[:len(trailing)]:
            version.unlink()
    return len(trailing)


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
# stamps_N.json files can be deleted individually (undoing one
# pass) -- a separate history would leave orphaned entries referencing
# files that no longer exist, needing its own cleanup logic to stay in
# sync. Keeping metadata and stamps in the same file means deleting the
# file removes its metadata too, automatically, with nothing to orphan.
# ---------------------------------------------------------------------------

def _new_layer_id() -> str:
    """A fresh, globally-unique layer identity -- see the module-level
    note above save_stamp_file for why this can't just be the
    filename (stamps_N.json slot numbers get reused after undo +
    regenerate) or a project.json-backed counter (would couple a
    layer's own identity to a separate file's state, exactly what this
    project's self-contained-layer-file design avoids elsewhere)."""
    return secrets.token_hex(8)


def save_stamp_file(
    stamps: list[Stamp], path: Path, step: str, parameters: dict, extra: dict | None = None,
    layer_id: str | None = None,
) -> None:
    """`layer_id` identifies this layer well enough for a LATER layer to
    reference it via a stamp's `blocked_by` field (see
    _flag_previous_layer_blocked) -- auto-generated if the caller
    doesn't supply one (every layer gets one, whether or not blocking
    is ever used against it), so existing call sites need no changes."""
    payload = {
        "step": step,
        "parameters": parameters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layer_id": layer_id or _new_layer_id(),
        **(extra or {}),
        "stamps": [dataclasses.asdict(s) for s in stamps],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _migrate_stamp_entry(entry: dict) -> dict:
    """
    Stamp.radius was replaced by independent scale_x/scale_z (see
    terrain/stamp.py) -- stamps_N.json files saved before that change
    still have "radius" instead. Every stamp this compiler has ever
    written was isotropic (scale_x == scale_z) at save time, so this
    is a lossless, not approximate, migration.
    """
    if "radius" in entry and "scale_x" not in entry:
        entry = dict(entry)
        radius = entry.pop("radius")
        entry["scale_x"] = radius
        entry["scale_z"] = radius
    return entry


def load_stamp_file(path: Path) -> tuple[list[Stamp], list["str | None"], dict]:
    """
    Returns (stamps, blocked_by, metadata). `blocked_by` is a parallel
    list (same order/length as `stamps`): another layer's `layer_id`
    for any stamp a later masked pass flagged as blocked (see
    _flag_previous_layer_blocked), else None. `metadata` is everything
    in the file except "stamps" itself (includes this layer's own
    `layer_id`, per save_stamp_file).
    """
    with path.open() as f:
        payload = json.load(f)
    entries = [_migrate_stamp_entry(entry) for entry in payload["stamps"]]
    blocked_by = [entry.get("blocked_by") for entry in entries]
    stamps = [Stamp(**{k: v for k, v in entry.items() if k != "blocked_by"}) for entry in entries]
    metadata = {k: v for k, v in payload.items() if k != "stamps"}
    return stamps, blocked_by, metadata


def _flag_previous_layer_blocked(
    working_dir: Path, mask_geometry: BaseGeometry | None, margin_m: float, new_layer_id: str,
) -> int:
    """
    If `mask_geometry` is given and at least one stamps_N.json layer
    already exists, finds every stamp in the IMMEDIATELY-PRECEDING
    layer (skipping any already blocked_by something) whose whole
    footprint -- exact shapely containment, terrain/stamp_containment.py,
    no discretization -- sits inside mask_geometry shrunk inward by
    margin_m, and sets blocked_by=new_layer_id on those entries.

    Atomically rewrites that ONE layer file (temp file + rename) -- the
    one narrow, deliberate exception to "stamps_N.json is immutable
    once written" (see CLAUDE.md): never deletes/reorders anything,
    only adds an optional field to specific entries, and only ever
    touches this file once. Undo/redo of the NEW layer (the one that
    calls this) needs no extra handling here -- load_all_stamps's
    active-layer-id check is what makes blocking toggle automatically
    when a layer file is renamed away/back.

    Backfills the previous layer's own layer_id first if it doesn't
    have one yet (every layer in every project created before this
    feature existed has none) -- but only when something is actually
    flagged, so a no-op call never rewrites a file for nothing.

    Returns the count flagged (0 if no mask, no previous layer, or
    nothing qualifies).
    """
    if mask_geometry is None or mask_geometry.is_empty:
        return 0
    files = _stamps_files(working_dir)
    if not files:
        return 0
    prev_path = files[-1]

    with prev_path.open() as f:
        payload = json.load(f)
    entries = [_migrate_stamp_entry(entry) for entry in payload["stamps"]]

    candidate_idx = [i for i, e in enumerate(entries) if e.get("blocked_by") is None]
    if not candidate_idx:
        return 0
    candidate_stamps = [
        Stamp(**{k: v for k, v in entries[i].items() if k != "blocked_by"}) for i in candidate_idx
    ]

    shrunk = mask_geometry.buffer(-margin_m)
    flags = stamps_fully_within(candidate_stamps, shrunk)
    flagged_idx = [candidate_idx[j] for j, ok in enumerate(flags) if ok]
    if not flagged_idx:
        return 0

    for i in flagged_idx:
        entries[i]["blocked_by"] = new_layer_id
    payload["stamps"] = entries
    payload.setdefault("layer_id", _new_layer_id())

    tmp_path = prev_path.with_name(prev_path.name + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(prev_path)

    print(f"  Flagged {len(flagged_idx)} of {len(entries)} stamp(s) in {prev_path.name} "
          f"(layer_id={payload['layer_id']}) as blocked_by this new layer (margin={margin_m:g}m)")
    return len(flagged_idx)


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

    if not _stamps_files(working_dir):
        print("No stamp layers yet -- run --step generate-terrain for the "
              "hex/stamps/height/error previews. Stopping after the LIDAR previews.")
        return

    stamps, _ = _load_all_stamps_incl_collections(working_dir, verbose=False)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    model = TerrainModel(stamps)

    # Label the terrain-related previews with whatever parameters
    # actually produced the latest layer, if any layer has been
    # written -- self-documenting, without cross-referencing a
    # separate log (see save_stamp_file's stamp-file metadata). Which
    # fields are meaningful depends on which step wrote that layer
    # (generate-terrain vs. refine-terrain vs. generate-cart-paths all
    # save different parameter shapes), so branch on the "step" field
    # rather than assuming refine-terrain's shape unconditionally --
    # unification means the latest layer is no longer necessarily a
    # refine pass.
    extra_label = None
    mask_grid = None
    latest_layer = load_latest_stamp_metadata(working_dir)
    if latest_layer is not None:
        p = latest_layer["parameters"]
        step_name = latest_layer.get("step", "?")
        if step_name == "refine-terrain":
            extra_label = (
                f"refine: method={p.get('method', 'adaptive')} rad={p.get('rad_m', 'n/a')} "
                f"tol={p.get('tolerance')} res={p.get('resolution')} hot={p.get('min_hotspot_radius_cells')} "
                f"claim={p.get('claim_radius_fraction')} spread={p.get('brush_radius_spread_ratio')}"
            )
            if p.get("max_planar_rms") is not None:
                extra_label += f" planar_rms={p['max_planar_rms']} shrink={p.get('planar_shrink_factor')}"
        elif step_name == "generate-terrain":
            extra_label = (
                f"generate: method={p.get('method')} pitch={p.get('pitch_m')} "
                f"band_spacing={p.get('band_spacing_m')}"
            )
            if p.get("method") == "hex":
                tool_name = "raise" if p.get("hex_tool") == 1 else "flatten"
                extra_label += f" hex_brush={p.get('hex_brush')} hex_tool={tool_name}"
            elif p.get("method") == "raster":
                extra_label += (
                    f" raster_size={p.get('raster_size')} raster_spread={p.get('raster_spread_ratio')} "
                    f"raster_center_bias_x={p.get('raster_center_bias_ratio_x')} "
                    f"raster_center_bias_z={p.get('raster_center_bias_ratio_z')} "
                    f"raster_brush={p.get('raster_brush')}"
                )
        elif step_name == "generate-cart-paths":
            extra_label = (
                f"cart-paths: stamp_radius={p.get('stamp_radius_m')} spacing={p.get('spacing_m')}"
            )
        elif step_name == "generate-streams":
            extra_label = f"streams: {p.get('stream_count')} centerline(s) {p.get('waterway_kinds')}"
        else:
            extra_label = f"step={step_name}"
        if p.get("use_height_mask"):
            buffer_note = f" buffer={p['mask_buffer_px']:.0f}px" if p.get("mask_buffer_px") is not None else ""
            extra_label += f" mask=on{buffer_note}"
            mask_path = working_dir / HEIGHT_MASK_FILE
            if mask_path.exists():
                # Rasterized at error_resolution (resolved here, once,
                # rather than at render_error_preview's own call site
                # below) -- mask_grid is only ever actually consumed by
                # render_error_preview, which requires it to match the
                # `error` array's own shape exactly (both index the same
                # resolution x resolution grid). Previously rasterized at
                # p.get("resolution", DEFAULT_RESOLUTION) instead -- a key
                # only refine-terrain's saved parameters ever have, so a
                # generate-terrain layer always fell back to
                # DEFAULT_RESOLUTION regardless of what error_resolution
                # (explicitly passed, or refine-terrain's own inherited
                # value) actually was, a real shape-mismatch bug confirmed
                # directly: a 1000x1000 error grid against a 200x200 mask.
                if error_resolution is None:
                    error_resolution = (
                        latest_layer["parameters"].get("resolution", 200)
                        if latest_layer is not None else 200
                    )
                mask_geometry = load_height_mask(mask_path)
                mask_grid = rasterize_mask(mask_geometry, bounds, error_resolution)

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
        error_resolution = (
            latest_layer["parameters"].get("resolution", 200) if latest_layer is not None else 200
        )
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
    preserve_synthetic: bool = True,
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

    # Re-parsing map.osm rebuilds every OSM-derived Feature from scratch,
    # but features.geojson also holds GUI-authored content that has no OSM
    # way behind it (cluster-fill border rings + their centerline
    # companions, "Use mask" clipped fills, generate-streams bank
    # vegetation) plus GUI edits layered onto real OSM Features (a flipped
    # mask flag, a pga_cluster_fills spec list). Fold all of that back in
    # so a re-ingest doesn't silently wipe it -- unless --no-preserve-
    # synthetic asked for the old clean-slate rebuild. `features` stays
    # OSM-only for the counts/previews/mask below (mask & fill overrides
    # are re-applied to it in place); the synthetic Features are appended
    # only to what's written to disk -- they have no viz style, so drawing
    # them into the OSM overlay PNG would bake in an un-toggleable magenta
    # blob.
    out_path = working_dir / FEATURES_FILE
    preserved_synthetic: list = []
    if preserve_synthetic and out_path.exists():
        preserved_synthetic, preserved = _preserved_synthetic_features(
            features, load_features(out_path)
        )
        if any(preserved.values()):
            print(f"  preserved {preserved['synthetic']} synthetic feature(s), re-applied "
                  f"{preserved['mask_reapplied']} mask override(s) and "
                  f"{preserved['fills_reapplied']} cluster-fill edit(s) from the existing "
                  f"{FEATURES_FILE} (--no-preserve-synthetic rebuilds purely from map.osm)")

    counts: dict[str, int] = {}
    for f in features:
        counts[f.kind] = counts.get(f.kind, 0) + 1
    for kind, n in sorted(counts.items()):
        print(f"  {kind}: {n}")

    save_features(features + preserved_synthetic, out_path)
    print(f"  wrote {out_path} (full point cloud frame, uncropped"
          + (f"; {len(preserved_synthetic)} preserved synthetic feature(s) appended)"
             if preserved_synthetic else ")"))

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


_SYNTHETIC_FEATURE_KINDS = (
    SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND, SYNTHETIC_BORDER_CENTERLINE_KIND,
)


def _is_synthetic_feature(f) -> bool:
    """
    A GUI/generate-streams-authored Feature with no OSM way behind it --
    identified purely by its `kind` (every synthetic Feature is created
    with one of `_SYNTHETIC_FEATURE_KINDS`: cluster-fill border rings and
    their centerline companions, "Use mask" clipped fills, generate-
    streams bank vegetation).

    Deliberately NOT keyed off a negative osm_id: an OSM editor (JOSM,
    Level0, ...) gives every not-yet-uploaded way a negative id, so a
    hand-authored `pga_collection` / `pga_scatter` marker way legitimately
    carries one. Treating those as synthetic made every re-ingest
    *preserve the old copy AND re-parse a fresh one*, compounding the
    feature list on each run.
    """
    return f.kind in _SYNTHETIC_FEATURE_KINDS


def _preserved_synthetic_features(parsed: list, previous: list) -> tuple[list, dict]:
    """
    Reconcile GUI/generate-streams-authored edits from a prior
    features.geojson (`previous`) against a freshly parsed OSM list
    (`parsed`):

      * `parsed` is mutated IN PLACE -- for each real OSM Feature that
        still exists (matched by osm_id), a saved mask override is
        re-applied and a GUI-added pga_cluster_fills spec list is copied
        back;
      * the purely synthetic Features (cluster-fill border rings + their
        centerline companions, "Use mask" clipped fills, generate-
        streams bank vegetation -- identified by `_is_synthetic_feature`,
        i.e. a synthetic `kind`) are returned as a SEPARATE list, verbatim
        (ids unchanged, so a border ring's PGA_CLUSTER_CENTERLINE_REF_TAG
        back-reference to its companion stays valid).

    A hand-authored `pga_collection` / `pga_scatter` marker way is a real
    OSM way (even with a JOSM negative id) -- it is NOT synthetic, so it
    is re-parsed fresh from map.osm every ingest, never preserved-and-
    duplicated, and its mask / pga_cluster_fills edits transplant via
    prev_by_id like any other real way.

    Returned separately rather than concatenated so the caller can write
    the synthetic Features to features.geojson but keep them OUT of the
    OSM preview renders (they have no viz feature style -> would draw as
    a magenta unknown-kind blob baked into the overlay PNG, with no way
    to toggle it off) and the OSM feature-kind counts. Both lists are in
    the same full uncropped point-cloud frame, so the caller just
    appends -- no shift/crop.

    Accepted heuristics: any difference between the old and freshly
    parsed `mask` is treated as a deliberate user override (if a way is
    re-tagged in OSM so its default mask flips, the stale override still
    transplants -- re-toggle in the GUI). pga_cluster_fills is only
    copied when the fresh Feature has none, so a spec coming straight
    from OSM tags wins and GUI-only spec lists (always written whole)
    never get duplicated.

    Returns (synthetic_features, {"synthetic", "mask_reapplied", "fills_reapplied"}).
    """
    synthetic = [f for f in previous if _is_synthetic_feature(f)]

    prev_by_id = {
        f.osm_id: f for f in previous
        if f.osm_id is not None and not _is_synthetic_feature(f)
    }
    mask_reapplied = fills_reapplied = 0
    for f in parsed:
        old = prev_by_id.get(f.osm_id)
        if old is None:
            continue
        if old.mask != f.mask:
            f.mask = old.mask
            mask_reapplied += 1
        old_fills = old.tags.get(PGA_CLUSTER_FILLS_TAG)
        if old_fills and not f.tags.get(PGA_CLUSTER_FILLS_TAG):
            f.tags[PGA_CLUSTER_FILLS_TAG] = old_fills
            fills_reapplied += 1

    return synthetic, {
        "synthetic": len(synthetic),
        "mask_reapplied": mask_reapplied,
        "fills_reapplied": fills_reapplied,
    }


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

    # Spline members of any placed object collection (see
    # step_generate_collections). Carried through verbatim from capture,
    # just rotated/translated into place and shifted into the game grid.
    collections_path = working_dir / COLLECTIONS_FILE
    if collections_path.exists():
        collection_splines = build_collection_splines(load_collection_records(collections_path))
        if collection_splines:
            splines = splines + collection_splines
            print(f"  + {len(collection_splines)} spline(s) from placed collections")

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

    _ensure_course_baseline(working_dir)
    nodes_dir = working_dir / "course" / "CourseDescription_nodes"

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

    Deliberately separate from step_write_splines/step_write_terrain --
    lets mask settings be tweaked and holes.json regenerated on its
    own, without redoing the terrain height export or surface splines.

    This overwrites holes.json wholesale, same as step_write_splines
    does for surfaceSplines.json.
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    game_version = load_project(working_dir).get("game_version", DEFAULT_GAME_VERSION)
    features = load_features(features_path)
    features = _crop_features_to_course(working_dir, features)
    holes = build_holes(features, game_version)

    total_hole_features = sum(1 for f in features if f.kind == "hole")
    excluded_count = sum(1 for f in features if f.kind == "hole" and f.mask)
    print(f"Generated {len(holes)} holes from {total_hole_features} hole features "
          f"({excluded_count} excluded via mask)")
    if len(holes) > 18:
        print(f"  WARNING: {len(holes)} holes exceeds PGA's 18-hole limit -- "
              "mask off (exclude) the extras in the GUI's Splines tab before importing")

    _ensure_course_baseline(working_dir)
    nodes_dir = working_dir / "course" / "CourseDescription_nodes"

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


def step_generate_trees(
    working_dir: Path, detect_lidar_trees: bool | None = None, mark_cartpath_trees: bool | None = None,
) -> None:
    """
    Generate the intermediate, VERSION-AGNOSTIC object_list.json (see
    objects.py's save_object_list) -- trees parsed from map.osm's
    natural=tree nodes, plus, optionally, trees individually detected
    from LIDAR canopy points (ingest/tree_detection.py), combined into
    one list.

    mark_cartpath_trees is a DEBUG mode: trees detected sitting on a
    cart path are left in place and tagged instead of relocated, so
    step_write_objects (v2019 only) swaps them for an oversized, obvious
    marker object instead of a real tree -- see
    objects.py's move_trees_off_cartpaths(debug_mark_only=...). Use this
    to eyeball in-game exactly which trees the detector is flagging
    before trusting it to actually move anything. Deliberately NOT the
    same None-means-use-saved-project.json pattern detect_lidar_trees
    uses -- always defaults to False (normal relocation behavior)
    unless explicitly passed True, so a debug run never silently
    becomes the sticky default for every future run. The last value
    used IS still saved to project.json, but only for the record (e.g.
    a GUI could show it), never read back as this run's default.

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
    raises StepError if either is missing while this is on. OSM
    building footprints (features.geojson's kind == "building") are
    also masked out of the canopy raster before detection -- some LAZ
    sources never classify buildings separately (ASPRS class 6), so
    their flat roof returns can land in the same bucket
    rasterize_canopy_heightmap_with_fallback falls back to, which
    watershed would otherwise mistake for a crown and place a "tree"
    on the roof.

    This overwrites object_list.json wholesale.
    """
    osm_path = working_dir / "map.osm"
    if not osm_path.exists():
        raise StepError(f"No map.osm found at {osm_path}. Run --step ingest-laz first.")

    project = load_project(working_dir)
    if detect_lidar_trees is None:
        detect_lidar_trees = project.get("objects_detect_lidar_trees", True)
    if mark_cartpath_trees is None:
        mark_cartpath_trees = False

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

    features_path = working_dir / FEATURES_FILE
    features = []
    building_geometries = []
    if features_path.exists():
        features = load_features(features_path)
        features = _crop_features_to_course(working_dir, features)
        building_geometries = [f.geometry for f in features if f.kind == "building"]

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

        if building_geometries:
            # OSM building footprints, not LIDAR classification -- some LAZ
            # deliveries never classify buildings (ASPRS class 6) separately,
            # so their roof returns land in the same "unclassified" bucket
            # rasterize_canopy_heightmap_with_fallback falls back to,
            # producing a flat elevated blob that watershed mistakes for a
            # crown. Buildings aren't real canopy regardless of why a LAZ
            # source misclassified them, so mask footprint cells out of the
            # canopy raster entirely -- routes through detect_trees_from_lidar's
            # existing "no vegetation return here" -> ground-height handling.
            building_mask = rasterize_mask(unary_union(building_geometries), course_bounds, resolution)
            canopy_heights[building_mask] = np.nan
            print(f"  masked out {len(building_geometries)} OSM building footprint(s) from the "
                  "canopy raster (ingest/osm.py) to avoid rooftop false-positive trees")

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

    if trees:
        wood_features = [f for f in features if f.kind == "wood"]
        if wood_features:
            untyped_before = sum(1 for _, _, tags in trees if TREE_TYPE_TAG not in tags)
            trees = apply_area_tree_type_hints(trees, wood_features)
            untyped_after = sum(1 for _, _, tags in trees if TREE_TYPE_TAG not in tags)
            if untyped_before != untyped_after:
                print(f"  applied area-based tree-type hints from {len(wood_features)} wood "
                      f"polygon(s): {untyped_before - untyped_after} tree(s) tagged "
                      "(course_output/objects.py's LEAF_TYPE_TREE_HINTS)")

        cartpath_features = [f for f in features if f.kind == "cartpath"]
        if cartpath_features:
            cartpath_lines = [f.geometry for f in cartpath_features]
            hole_features = [f for f in features if f.kind == "hole"]
            before_xz = [(x, z) for x, z, _ in trees]
            trees = move_trees_off_cartpaths(
                trees, cartpath_lines, hole_features, debug_mark_only=mark_cartpath_trees,
            )
            moved = sum(1 for (bx, bz), (x, z, _) in zip(before_xz, trees) if (bx, bz) != (x, z))
            if moved:
                print(f"  moved {moved} tree(s) off {len(cartpath_features)} cart path feature(s), "
                      "perpendicular to the path and away from the nearest hole centerline "
                      "(course_output/objects.py's move_trees_off_cartpaths)")
    elif trees:
        print(f"  No {FEATURES_FILE} found -- skipping area-based tree-type hints "
              "(run --step ingest-osm first if you want wood-polygon leaf_type hints applied).")

    out_path = working_dir / OBJECT_LIST_FILE
    save_object_list(trees, out_path)
    print(f"Wrote {out_path} ({len(trees)} tree(s))")

    save_project(working_dir, {
        "objects_detect_lidar_trees": detect_lidar_trees,
        "objects_mark_cartpath_trees_debug": mark_cartpath_trees,
        "objects_tree_count": len(trees),
    })


def step_pack_objects(working_dir: Path) -> None:
    """
    Generate objects.json (see course_output/objects.py's save_objects)
    -- the intermediate, VERSION-AGNOSTIC combined object list: trees,
    passed through unchanged from object_list.json (see
    step_generate_trees; run that first), plus packed cluster-fill
    records from every features.geojson Feature carrying
    PGA_CLUSTER_FILLS_TAG (see course_output/object_clusters.py's
    pack_cluster_records -- tiered dart-throw or ring-walk circle
    placement, RNG seed draws, all done here and FROZEN into the file).

    Extends the "compile once, format at write time" split
    step_generate_trees' own docstring describes to cluster fills too:
    step_write_objects (any game_version) now just formats this file's
    contents into that version's schema -- no packing, no RNG, no
    `course/` dependency -- so it's cheap enough for a GUI Fill/Clear
    action to re-run synchronously for an instant preview, and
    switching game_version later reuses the exact same in-game layout
    rather than re-rolling a fresh random one.

    Also folds in resolved object-collection placements from
    collections.json (see step_generate_collections) as
    kind="collection_object" records -- deterministic, no RNG.

    Also packs object-spline-fill records (see
    course_output/object_clusters.py's pack_spline_records) from every
    mode="spline" PGA_CLUSTER_FILLS_TAG spec -- v2021+-only, no RNG
    (there's nothing to pack, just an already-subdivided polygon per
    piece), formatted at write-objects time same as cluster fills.

    Also passes ingame_objects.json through unchanged (see
    course_output/ingame_objects.py / step_import_ingame_edits) as
    kind="ingame_object" records -- this step only READS that file,
    never writes it, so objects a user imported from an in-game edit
    survive every re-pack (a re-ingest, a fresh Generate Trees run, a
    Fill/Clear) automatically.

    Requires object_list.json to exist (same convention
    step_write_objects used before this step existed) -- an empty tree
    list still needs the file to be there. features.geojson,
    collections.json, and ingame_objects.json are all optional: a
    missing file just means zero records of that kind, not an error.

    This overwrites objects.json wholesale.
    """
    object_list_path = working_dir / OBJECT_LIST_FILE
    if not object_list_path.exists():
        raise StepError(f"No {OBJECT_LIST_FILE} found under {working_dir}. Run --step generate-trees first.")
    trees = load_object_list(object_list_path)

    # "auto"/"spline" fill specs resolve against game_version at pack time
    # (see object_clusters.resolve_fill_mode), so objects.json depends on
    # it -- re-run pack-objects after a version switch (the GUI does this
    # for its live preview automatically).
    game_version = load_project(working_dir).get("game_version", DEFAULT_GAME_VERSION)

    cluster_records: list[dict] = []
    object_spline_fill_records: list[dict] = []
    features_path = working_dir / FEATURES_FILE
    if features_path.exists():
        features = _crop_features_to_course(working_dir, load_features(features_path))
        tagged = [f for f in features if f.tags.get(PGA_CLUSTER_FILLS_TAG)]
        if tagged:
            cluster_records = pack_cluster_records(features, game_version=game_version)
            object_spline_fill_records = pack_spline_records(features, game_version=game_version)
            print(f"  packed {len(cluster_records)} cluster stamp(s) + "
                  f"{len(object_spline_fill_records)} object-spline fill piece(s) "
                  f"across {len(tagged)} tagged spline(s) (game_version={game_version})")

    # Resolved collection member placements (see step_generate_collections
    # -- run that first). Deterministic, no RNG: just flattened out of
    # collections.json's per-record "objects" lists.
    collection_objects: list[dict] = []
    collections_path = working_dir / COLLECTIONS_FILE
    if collections_path.exists():
        records = load_collection_records(collections_path)
        collection_objects = list(iter_collection_objects(records))
        if collection_objects:
            print(f"  {len(collection_objects)} collection object(s) across {len(records)} placement(s)")

    # Parked-car props from parking.json (see step_generate_parking) --
    # folded in as kind="collection_object" records too (same shape plus
    # pitch/roll), so write-objects formats them with no extra branch.
    parking_path = working_dir / PARKING_FILE
    if parking_path.exists():
        parking_records = load_parking_records(parking_path)
        parking_cars = list(iter_parking_cars(parking_records))
        if parking_cars:
            collection_objects += parking_cars
            print(f"  {len(parking_cars)} parked car(s) across {len(parking_records)} aisle(s)")

    ingame_object_records: list[dict] = []
    ingame_objects_path = working_dir / INGAME_OBJECTS_FILE
    if ingame_objects_path.exists():
        ingame_object_records = load_ingame_objects(ingame_objects_path)
        if ingame_object_records:
            print(f"  {len(ingame_object_records)} imported in-game object(s) from {INGAME_OBJECTS_FILE}")

    out_path = working_dir / OBJECTS_FILE
    save_objects(
        trees, cluster_records, out_path, collection_objects, object_spline_fill_records,
        ingame_object_records,
    )
    print(f"Wrote {out_path} ({len(trees)} tree(s), {len(cluster_records)} cluster stamp(s), "
          f"{len(object_spline_fill_records)} object-spline fill piece(s), "
          f"{len(collection_objects)} collection object(s), "
          f"{len(ingame_object_records)} imported in-game object(s))")


def step_write_objects(
    working_dir: Path,
    game_version: str | None = None,
    theme: int | None = None,
    tree_variety: bool | None = None,
    tree_asset_paths: list[str] | None = None,
    tree_type_asset_paths: dict[str, str] | None = None,
    stake_asset_path: str | None = None,
    stake_buildings: bool | None = None,
    waterfall_asset_path: str | None = None,
    splash_asset_path: str | None = None,
    tree_theme_config: str | None = None,
) -> None:
    """
    Generate placedObjects2.json -- formats objects.json (see
    step_pack_objects; run that first, after step_generate_trees) into
    the target game_version's schema, plus, optionally, a stake at
    every building corner (from features.geojson's "building" ways):
    stake_buildings turns them on for either game_version. v2019 uses
    the fence-post prop at category=objects.BUILDING_STAKE_CATEGORY_V2019/
    type=objects.BUILDING_STAKE_TYPE_V2019, scaled to objects.
    BUILDING_STAKE_SCALE_V2019 (see build_building_stake_objects_v2019);
    v2021+ uses objects.DEFAULT_STAKE_ASSET_PATH_V2021 (the same prop),
    or stake_asset_path when given to override it (see
    build_building_stake_objects_v2021). stake_asset_path is v2021+-only
    and, if set, also implies stakes even without stake_buildings.

    Purely a formatting step now -- no packing, no RNG: tree positions
    come straight from objects.json's tree records, and cluster-fill
    circle positions/counts/seeds were already frozen in there by
    step_pack_objects, so re-running this (e.g. after switching
    game_version) reproduces the exact same in-game layout rather than
    rerolling a fresh random one. Building stakes are the one exception
    -- cheap and non-randomized, so they're still computed directly
    from features.geojson here rather than routed through objects.json.

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

    theme / tree_variety / stake_buildings and tree_asset_paths /
    tree_type_asset_paths / stake_asset_path (v2021+) are all feature-
    flagged via project.json, same pattern as refine-terrain's
    parameters: pass None here to use whatever was last saved, or an
    explicit value (an empty list/dict counts as explicit) to override
    for this run and persist it as the new default for next time. Only
    the parameters relevant to the resolved game_version are actually
    used; the others are still accepted (and persisted, if given) so a
    project can carry both versions' settings across a future
    game_version switch without losing them.

    stake_buildings defaults to False -- unlike tree_variety, there's no
    reason to want stakes on by default. The GUI's "Stake Buildings" /
    "Clear Building Stakes" buttons are both just this same step run
    with --stake-buildings / --no-stake-buildings.

    tree_variety defaults to True (not just "off unless set") -- there's
    no real reason to want the flat, single-generic-type result it
    disables to; it exists mainly as an override for testing.

    This overwrites placedObjects2.json wholesale, same as
    step_write_splines/step_write_holes do for their own files.
    """
    project = load_project(working_dir)
    (game_version, theme, tree_variety, tree_theme_config, tree_asset_paths, tree_type_asset_paths,
     stake_asset_path, stake_buildings, waterfall_asset_path, splash_asset_path) = _resolve_write_objects_params(
        project, game_version, theme, tree_variety, tree_theme_config, tree_asset_paths,
        tree_type_asset_paths, stake_asset_path, stake_buildings, waterfall_asset_path, splash_asset_path,
    )

    placed_objects = _build_placed_objects(
        working_dir, game_version, theme, tree_variety, tree_theme_config, tree_asset_paths,
        tree_type_asset_paths, stake_asset_path, stake_buildings, waterfall_asset_path, splash_asset_path,
        project,
    )

    _ensure_course_baseline(working_dir)
    nodes_dir = working_dir / "course" / "CourseDescription_nodes"

    out_path = nodes_dir / schema_for(game_version).objects_filename
    save_placed_objects(placed_objects, out_path)
    print(f"Wrote {out_path}")

    save_project(working_dir, {
        "game_version": game_version,
        "objects_theme": theme,
        "objects_tree_variety": tree_variety,
        "objects_tree_theme_config": tree_theme_config,
        "objects_tree_asset_paths": tree_asset_paths,
        "objects_tree_type_asset_paths": tree_type_asset_paths,
        "objects_stake_asset_path": stake_asset_path,
        "objects_stake_buildings": stake_buildings,
        "streams_waterfall_asset_path": waterfall_asset_path,
        "streams_splash_asset_path": splash_asset_path,
    })


def _resolve_write_objects_params(
    project: dict,
    game_version: str | None,
    theme: int | None,
    tree_variety: bool | None,
    tree_theme_config: str | None,
    tree_asset_paths: list[str] | None,
    tree_type_asset_paths: dict[str, str] | None,
    stake_asset_path: str | None,
    stake_buildings: bool | None,
    waterfall_asset_path: str | None,
    splash_asset_path: str | None,
) -> tuple:
    """
    Resolve every step_write_objects/_build_placed_objects parameter
    against project.json, exactly like step_write_objects always did --
    factored out so step_import_ingame_edits can recompute the same
    "expected" placed-objects build (against whatever's currently
    saved) without duplicating this resolution logic. Raises StepError
    if the resolved game_version isn't implemented.
    """
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
    if tree_theme_config is None:
        tree_theme_config = project.get("objects_tree_theme_config")
    if tree_asset_paths is None:
        tree_asset_paths = project.get("objects_tree_asset_paths", [])
    if tree_type_asset_paths is None:
        tree_type_asset_paths = project.get("objects_tree_type_asset_paths", {})
    if stake_asset_path is None:
        stake_asset_path = project.get("objects_stake_asset_path")
    if stake_buildings is None:
        stake_buildings = project.get("objects_stake_buildings", False)
    if waterfall_asset_path is None:
        waterfall_asset_path = project.get("streams_waterfall_asset_path", WATERFALL_DEFAULT_ASSET_PATH)
    if splash_asset_path is None:
        splash_asset_path = project.get("streams_splash_asset_path", WATERSPLASH_DEFAULT_ASSET_PATH)
    return (game_version, theme, tree_variety, tree_theme_config, tree_asset_paths, tree_type_asset_paths,
            stake_asset_path, stake_buildings, waterfall_asset_path, splash_asset_path)


def _build_placed_objects(
    working_dir: Path,
    game_version: str,
    theme: int | None,
    tree_variety: bool,
    tree_theme_config: str | None,
    tree_asset_paths: list[str],
    tree_type_asset_paths: dict[str, str],
    stake_asset_path: str | None,
    stake_buildings: bool,
    waterfall_asset_path: str,
    splash_asset_path: str,
    project: dict,
) -> list[dict]:
    """
    Build the full, merged placedObjects2/3.json group list in memory,
    WITHOUT writing anything to disk -- the exact construction
    step_write_objects used to do inline (trees, cluster fills,
    object-spline fills, collections, imported in-game objects,
    building stakes, stream waterfalls/splashes, then
    merge_object_groups). Factored out so step_write_objects and
    step_import_ingame_edits's "expected" recomputation can never
    silently drift apart -- there is exactly one place this
    construction happens.

    All parameters are assumed ALREADY RESOLVED (see
    _resolve_write_objects_params) -- this never reads project.json
    for a default itself, only for things that aren't step_write_objects
    parameters at all (output_height_shift_m, a stamp-derived
    TerrainModel for elevated collection objects).
    """
    objects_path = working_dir / OBJECTS_FILE
    if not objects_path.exists():
        raise StepError(
            f"No {OBJECTS_FILE} found under {working_dir}. Run --step pack-objects first "
            "(after --step generate-trees)."
        )

    trees, cluster_records, collection_objects, object_spline_fill_records, ingame_object_records = (
        load_objects(objects_path)
    )
    print(f"game_version={game_version}  loaded {len(trees)} tree(s), {len(cluster_records)} cluster "
          f"stamp(s), {len(object_spline_fill_records)} object-spline fill piece(s), "
          f"{len(collection_objects)} collection object(s), {len(ingame_object_records)} imported "
          f"in-game object(s) from {OBJECTS_FILE}")

    # Collection members with a designed elevation (`dy` -- see
    # course_output/collections.py) need an absolute y = target terrain
    # height at (x, z) + output_height_shift_m + dy. Evaluated here, at
    # write time, against the final stamp list -- same "format against
    # the terrain as it stands now" approach as stream waterfalls below.
    elevated = [o for o in collection_objects if o.get("dy") is not None]
    if elevated:
        if _stamps_files(working_dir):
            shift_m = project.get("output_height_shift_m")
            if shift_m is None:
                print("  NOTE: no output_height_shift_m in project.json yet -- run write-terrain so "
                      "elevated collection objects sit at the right height. Using 0 for now.")
                shift_m = 0.0
            stamps_for_height, _ = _load_all_stamps_incl_collections(working_dir)
            model = TerrainModel(stamps_for_height)
            apply_terrain_heights(collection_objects, model.evaluate, shift_m)
            print(f"  resolved terrain height for {len(elevated)} elevated collection object(s)")
        else:
            print(f"  NOTE: {len(elevated)} collection object(s) have a designed elevation but no "
                  "terrain stamps exist yet -- run generate-terrain/write-terrain. They'll "
                  "ground-snap ('-Infinity') for now.")

    placed_objects: list[dict] = []

    if game_version == "2019":
        if trees:
            species = load_tree_theme_species(theme, tree_theme_config)
            if species is not None:
                bucket_sizes = {name: len(ids) for name, ids in species.species.items()}
                tagged = {tg.get(TREE_TYPE_TAG) for _, _, tg in trees if tg.get(TREE_TYPE_TAG)}
                unknown = sorted(t for t in tagged if not species.is_known(t))
                print(f"  theme={theme}  species buckets={bucket_sizes}  "
                      f"default={species.default_species!r} (course_output/tree_themes.json)")
                if unknown:
                    print(f"  NOTE: pga_tree_type {unknown} has no bucket for this theme -- "
                          f"those trees use the {species.default_species!r} bucket")
            else:
                print(f"  theme={theme}  tree_variety={tree_variety}  "
                      "(no tree_themes.json entry -- random-pool trees)")
            placed_objects += build_tree_objects_v2019(
                trees, theme=theme, tree_variety=tree_variety, species=species,
            )

        if cluster_records:
            cluster_groups = cluster_records_to_v2019_groups(cluster_records)
            cluster_count_total = sum(len(g["Value"]["clusters"]) for g in cluster_groups)
            spline_count = len({r["spline_id"] for r in cluster_records})
            print(f"  {cluster_count_total} cluster(s) across {spline_count} tagged spline(s)")
            placed_objects += cluster_groups

        if collection_objects:
            collection_groups = build_collection_objects_v2019(collection_objects)
            placed_count = sum(len(g["Value"]["items"]) for g in collection_groups)
            print(f"  {placed_count} collection object(s) in {len(collection_groups)} group(s)")
            placed_objects += collection_groups

        if ingame_object_records:
            ingame_groups = build_ingame_objects_v2019(ingame_object_records)
            placed_count = sum(len(g["Value"]["items"]) for g in ingame_groups)
            print(f"  {placed_count} imported in-game object(s) in {len(ingame_groups)} group(s)")
            placed_objects += ingame_groups

        if stake_buildings:
            features_path = working_dir / FEATURES_FILE
            if not features_path.exists():
                raise StepError(
                    f"--stake-buildings was given but no {FEATURES_FILE} found under {working_dir} "
                    "(needed for building corners) -- run --step ingest-osm first."
                )
            features = _crop_features_to_course(working_dir, load_features(features_path))
            building_count = sum(1 for f in features if f.kind == "building")
            stakes = build_building_stake_objects_v2019(features)
            stake_count = sum(len(g["Value"]["items"]) for g in stakes)
            print(f"  {stake_count} stake(s) at corners of {building_count} building(s)")
            placed_objects += stakes

        if stake_asset_path:
            print("  NOTE: --stake-asset-path is set but ignored for game_version=2019 -- "
                  "that's a v2021+-only scheme (see objects.py's build_building_stake_objects_v2021 "
                  "docstring). Use --stake-buildings for v2019 instead.")

        if object_spline_fill_records:
            print(f"  NOTE: dropped {len(object_spline_fill_records)} object-spline fill piece(s) -- "
                  "v2019 has no object-spline schema, and objects.json was packed for a different "
                  "version. Re-run pack-objects: mode=auto / mode=spline fills fall back to "
                  "circle-scatter for a v2019 target.")
    else:  # 2021+ (only "2021" itself is in IMPLEMENTED_GAME_VERSIONS right now)
        if trees:
            pool, type_map = tree_asset_paths, tree_type_asset_paths
            if not pool and not type_map:
                pool, type_map = default_tree_asset_paths_v2021(theme, tree_theme_config)
                if pool or type_map:
                    print(f"  no --tree-asset-path given -- using the bundled rustic catalog "
                          f"({len(pool)} general / buckets {sorted(type_map)}), "
                          "see course_output/asset_catalog.json + tree_themes.json")
                else:
                    raise StepError(
                        f"{len(trees)} tree(s) found in {OBJECTS_FILE}, but no tree asset path is "
                        "set. The bundled asset_catalog.json only covers the 'rustic' theme; for "
                        "this project's theme pass --tree-asset-path (repeatable) and/or "
                        "--tree-type-asset-path TAG=path (repeatable). v2021+ placed objects need "
                        "real Unity asset paths -- there's no numeric-id catalog to guess from."
                    )
            print(f"  tree_asset_paths={pool}  tree_type_asset_paths={type_map}")
            placed_objects += build_tree_objects_v2021(trees, pool, type_map)

        if cluster_records:
            cluster_groups = cluster_records_to_v2021_groups(cluster_records)
            cluster_count_total = sum(len(g["Value"]["clusters"]) for g in cluster_groups)
            spline_count = len({r["spline_id"] for r in cluster_records})
            print(f"  {cluster_count_total} cluster(s) across {spline_count} tagged spline(s) "
                  "(mode=stamps)")
            placed_objects += cluster_groups

        if object_spline_fill_records:
            spline_groups = object_spline_fill_records_to_v2021_groups(object_spline_fill_records)
            spline_count_total = sum(len(g["Value"]["splines"]) for g in spline_groups)
            spline_id_count = len({r["spline_id"] for r in object_spline_fill_records})
            print(f"  {spline_count_total} object-spline fill piece(s) across {spline_id_count} "
                  "tagged spline(s) (mode=spline)")
            placed_objects += spline_groups

        if stake_buildings or stake_asset_path:
            resolved_stake_path = stake_asset_path or DEFAULT_STAKE_ASSET_PATH_V2021
            features_path = working_dir / FEATURES_FILE
            if not features_path.exists():
                raise StepError(
                    f"building stakes were requested but no {FEATURES_FILE} found under {working_dir} "
                    "(needed for building corners) -- run --step ingest-osm first."
                )
            features = load_features(features_path)
            features = _crop_features_to_course(working_dir, features)
            building_count = sum(1 for f in features if f.kind == "building")
            stakes = build_building_stake_objects_v2021(features, resolved_stake_path)
            stake_count = sum(len(g["Value"]["items"]) for g in stakes)
            print(f"  {stake_count} stake(s) at corners of {building_count} building(s)")
            placed_objects += stakes

        if collection_objects:
            collection_groups = build_collection_objects_v2021(collection_objects)
            placed_count = sum(len(g["Value"]["items"]) for g in collection_groups)
            print(f"  {placed_count} collection object(s) in {len(collection_groups)} group(s)")
            placed_objects += collection_groups

        if ingame_object_records:
            ingame_groups = build_ingame_objects_v2021(ingame_object_records)
            placed_count = sum(len(g["Value"]["items"]) for g in ingame_groups)
            print(f"  {placed_count} imported in-game object(s) in {len(ingame_groups)} group(s)")
            placed_objects += ingame_groups

    # Waterfall + splash prefabs from streams.json (see step_generate_streams).
    # Their y is re-fit to the real carved terrain here, the same way
    # write-water fits the stream water tiles they hang on -- via a RAW
    # (pre-normalization) TerrainModel plus project.json's
    # output_height_shift_m. streams.json's own frozen levels are only a
    # fallback when no terrain stamps exist yet.
    streams_path = working_dir / STREAMS_FILE
    if streams_path.exists():
        stream_records = load_stream_records(streams_path)
        if sum(len(r.get("waterfalls", [])) for r in stream_records):
            height_shift_m = project.get("output_height_shift_m")
            if height_shift_m is None:
                print("  NOTE: no output_height_shift_m in project.json yet -- run write-terrain "
                      "so stream waterfalls sit at the right elevation. Using 0 for now.")
                height_shift_m = 0.0
            if _stamps_files(working_dir):
                raw_stamps, _ = _load_all_stamps_incl_collections(working_dir, verbose=False)
                raw_model = TerrainModel(raw_stamps)
                stream_records = rebuild_stream_drop_rows(
                    stream_records, raw_model.evaluate_many,
                    water_fill_depth_m=project.get(
                        "streams_water_fill_depth_m", STREAM_WATER_FILL_DEPTH_M),
                    water_base_width_m=project.get(
                        "streams_water_base_width_m", STREAM_WATER_BASE_WIDTH_M),
                    water_widen_per_depth=project.get(
                        "streams_water_widen_per_depth", STREAM_WATER_WIDEN_PER_DEPTH),
                    water_widen_per_descent=project.get(
                        "streams_water_widen_per_descent", STREAM_WATER_WIDEN_PER_DESCENT),
                    level_margin_m=project.get(
                        "streams_water_level_margin_m", STREAM_WATER_LEVEL_MARGIN_M),
                )
            n_falls = sum(len(r.get("waterfalls", [])) for r in stream_records)
            n_splash = sum(len(r.get("splashes", [])) for r in stream_records)
            if game_version == "2019":
                falls = build_waterfall_objects_v2019(stream_records, height_shift_m)
                splashes = build_watersplash_objects_v2019(stream_records, height_shift_m)
            else:
                falls = build_waterfall_objects_v2021(stream_records, waterfall_asset_path, height_shift_m)
                splashes = build_watersplash_objects_v2021(stream_records, splash_asset_path, height_shift_m)
            print(f"  {n_falls} stream waterfall(s), {n_splash} splash(es)")
            placed_objects += falls
            placed_objects += splashes

    placed_objects = merge_object_groups(placed_objects)

    for label, item_count, cluster_count, spline_count in object_counts(placed_objects):
        print(f"    {label}: {item_count} item(s), {cluster_count} cluster(s), {spline_count} spline(s)")

    return placed_objects


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
    hex_spread_ratio: float | None = None,
    method: str | None = None,
    hex_brush: int | None = None,
    hex_tool: int | None = None,
    raster_size: float | None = None,
    raster_spread_ratio: float | None = None,
    raster_center_bias_ratio_x: float | None = None,
    raster_center_bias_ratio_z: float | None = None,
    raster_brush: int | None = None,
    band_spacing_m: float | None = None,
    fill_mode: str | None = None,
    fill_brush: int | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
    radius_step_ratio: float | None = None,
    edge_distance_m: float | None = None,
    rect_brush: int | None = None,
    rect_tolerance_m: float | None = None,
    rect_min_length_m: float | None = None,
    rect_max_search_distance_m: float | None = None,
    rect_width_samples: int | None = None,
    smoothing_brush: int | None = None,
    smoothing_min_radius: float | None = None,
    smooth_ratio: float | None = None,
    smooth_claim_fraction: float | None = None,
    enable_secondary_fill: bool | None = None,
    candidates_per_radius: int | None = None,
    sweet_spot_ratio: float | None = None,
    sweet_spot_sample_bands: int | None = None,
    sweet_spot_seeds: int | None = None,
    sweet_spot_max_candidates: int | None = None,
    sweet_spot_time_budget_s: float | None = None,
    random_seed: int | None = None,
    denoise_px: int | None = None,
    max_stamps: int | None = None,
    use_height_mask: bool | None = None,
    mask_buffer_px: float | None = None,
    remove_covered_stamps: bool | None = None,
    remove_covered_margin_m: float | None = None,
    n_workers: int | None = None,
) -> None:
    """
    pitch (feature-flagged via project.json, same None-means-use-saved
    pattern used throughout this file) is terrain/hexgrid.py's
    HEX_LATTICE_PITCH_M, exposed here rather than hardcoded -- controls
    the spacing (i.e. the lattice CENTERS) of the initial coarse hex-grid
    stamp lattice (smaller pitch = more, smaller, more tightly-packed
    initial stamps). Edge bleed derives from pitch alone (bleed = pitch,
    a center-geometry fact -- see terrain/hexgrid.py's module docstring)
    and is unaffected by hex_spread_ratio below.

    hex_spread_ratio (same None-means-use-saved pattern) independently
    scales stamp RADIUS without moving any lattice center: stamp_radius
    = 2*pitch*hex_spread_ratio. Default 1.0 reproduces the original
    fixed radius = 2*pitch (each stamp reaches exactly to its nearest
    neighbors' centers). generate_hex_grid() itself only defaults
    stamp_radius/bleed to the ORIGINAL fixed pitch's values (Python
    default arguments are evaluated once, not re-derived from whatever
    `pitch`/`hex_spread_ratio` are actually passed), so both are
    computed explicitly here for whatever pitch/spread are in play, not
    left to fall back silently.

    method ("hex", default, "contour", or "raster") picks the initial-
    layout generator: "hex" is the flat lattice above; "contour" is
    terrain/contour_layers.py's two-pass-per-band fill -- see that
    module's docstring for the full design. Briefly: PASS 1 is a fast
    random-candidate poisson pack with a hard, high-plateau fill_brush
    (type 8/73), then PASS 2 is an oversized, heavily-overlapping
    scatter fill with a softer smoothing_brush over whatever pass 1
    leaves as genuine crumbs -- pass 2 IS exhaustive, so overall
    coverage stays complete by construction even though pass 1 trades a
    per-call guarantee for speed. "raster" is terrain/rastergrid.py's
    flat, non-offset square grid of hard type-72 stamps, each valued at
    the nearest heightmap cell to its own center (no fitting, no
    averaging) -- see that module's docstring for the sizing math. Only
    "hex"'s parameters (pitch) apply in "hex" mode, only "raster"'s
    (raster_size) apply in "raster" mode, and vice versa for the
    contour_* parameters below.

    raster_size ("raster" method only) is the raster grid's center-to-
    center spacing (m) -- must be one of terrain/rastergrid.py's
    RASTER_SIZES (256, 64, 32, 16, 8, 4, 2), not a free value. Each
    call places one flat grid at ONE size; layer coarse-to-fine
    yourself by re-running this step at each size in turn (same
    layering this docstring describes below for contour/hex).

    raster_spread_ratio ("raster" method only, same None-means-use-
    saved pattern as hex_spread_ratio) scales each raster stamp's
    radius independently of raster_size -- lattice centers are
    unaffected, only scale_x/scale_z change (see
    terrain/rastergrid.py's SPREAD section). Default 1.0 reproduces
    the exact edge-to-edge tiling radius; mainly useful as an A/B knob
    against that default when investigating stamp-seam artifacts (see
    course_output/userLayers.py's normalize_stamp_heights_by_value_shift).

    raster_center_bias_ratio_x/raster_center_bias_ratio_z ("raster"
    method only, same None-means-use-saved pattern) shift each raster
    stamp's placement (not its sampled value) by raster_size *
    raster_center_bias_ratio_x along x and raster_size *
    raster_center_bias_ratio_z along z, independently -- compensates
    for a resolution-dependent "drop shadow" caused by TerrainModel's
    order-dependent overlap fold always favoring the +x/+z neighbor in
    this lattice's own generation order (see terrain/rastergrid.py's
    module docstring's CENTER BIAS section for the full mechanism).
    Default 0.0 (each) is off; there's no derived "correct" value, so
    like raster_spread_ratio these are meant to be dialed in
    empirically.

    raster_brush ("raster" method only) is the brush every lattice
    stamp uses -- terrain/rastergrid.py's RASTER_BRUSH (type 72,
    "hard square") by default. The grid's own center-to-center spacing
    math (raster_radius above) is derived specifically from type 72's
    measured flat-plateau/instant-edge profile (see
    terrain/rastergrid.py's module docstring's STAMP SIZING section)
    so it tiles edge-to-edge with no gap or overlap; picking any other
    (circular) brush here keeps that same radius but no longer tiles
    exactly -- a circle inscribed in each square cell leaves the
    corners uncovered -- so treat this as a cosmetic/blending choice
    (pair with raster_spread_ratio > 1 to close the resulting gaps),
    not a like-for-like swap.

    enable_secondary_fill (contour method only, default True) turns
    pass 2 off entirely when False -- whatever pass 1 leaves as crumbs
    stays unfilled, so coverage is no longer complete. Trades that
    guarantee for speed, e.g. for a quick look at pass 1's own plateau
    shape.

    fill_mode ("poisson", default, or "rect") only applies within
    method="contour" -- it picks terrain/contour_layers.py's own
    per-band fill algorithm: "poisson" is the two-pass circle fill
    above; "rect" instead traces each band's own real boundary and
    places one type-72 stamp per boundary edge (see generate_contour_layers'
    and contour_layers.py's own docstrings, module section RECT FILL
    MODE, for the full design). Everything else about method="contour"
    -- band_spacing_m, the per-band loop, n_workers parallelization,
    layering behavior -- is identical between the two; only the shape
    of what fills one band's own mask differs. fill_brush/min_radius/
    max_radius/radius_step_ratio/edge_distance_m/candidates_per_radius
    and the sweet-spot auto-tuning knobs apply to fill_mode="poisson"
    only; rect_brush/rect_tolerance_m/rect_min_length_m/
    rect_max_search_distance_m/rect_width_samples apply to
    fill_mode="rect" only.

    Unlike hex mode, contour and raster mode stamps already carry their
    exact final value (contour: the local heightmap mean within each
    stamp's own footprint, computed inside generate_contour_layers
    itself; raster: the nearest heightmap cell to each stamp's center,
    computed inside generate_raster_grid itself) -- so the
    fit_stamp_heights() pass below only runs in "hex" mode. Re-running
    it against contour/raster stamps would be redundant at best.

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

    use_height_mask/mask_buffer_px follow the exact same pattern
    step_refine_terrain uses: use_height_mask restricts this layer to
    inside height_mask.geojson (fairway/green/tee + buffered hole-path
    corridors, see ingest-osm). Unlike step_refine_terrain, restriction
    happens BEFORE generation, not as a filter on the output -- contour
    mode ANDs a rasterized mask into every band's own footprint before
    the poisson-pack/crumb-scatter search runs (see
    generate_contour_layers' region_mask), and hex/raster mode each
    test lattice/grid candidates against the mask polygon before ever
    constructing a Stamp (see generate_hex_grid's/generate_raster_grid's
    own mask_geometry) -- so a masked pass over a small fraction of the
    course does proportionally less work,
    not the same full-course work followed by discarding most of it
    (the course-wide baseline-flatten stamp below, when added at all, is
    never masked -- it's a safety net for the very first contour-mode
    layer only, not part of the masked fill).
    mask_buffer_px is record-keeping only (shows up in preview titles
    and stamp-file metadata) -- the mask itself is already baked into
    height_mask.geojson at ingest-osm time, not rebuilt here. This is
    what makes a targeted pass possible now that generate-terrain is
    layerable: e.g. run a finer contour pass restricted to just the
    green complexes without touching the rest of the course.

    remove_covered_stamps (default off) additionally flags stamps in
    the IMMEDIATELY-PRECEDING stamps_N.json layer as blocked_by this
    new one wherever their whole footprint sits inside this pass's own
    mask, shrunk inward by remove_covered_margin_m (default
    DEFAULT_REMOVE_COVERED_MARGIN_M) -- exact shapely containment (see
    terrain/stamp_containment.py), not a claim this pass repaints every
    pixel, just "this old stamp is about to be entirely underneath
    something new." Requires use_height_mask; a no-op otherwise. See
    _flag_previous_layer_blocked and load_all_stamps for how blocking
    is applied/reversed (undo-transparent, no dedicated undo code).

    This step never overwrites a previous run: each call determines
    the next available stamps_N.json (same auto-increment convention
    refine-terrain's passes already use -- see _stamps_files) and
    writes its own new layer there, on top of whatever layers already
    exist. Layers are composited in order -- later layers take
    precedence wherever they overlap an earlier one (the same
    sequential pull-toward-value compositing every other stamp layer
    already uses), so e.g. a coarse hex pass, then a contour pass,
    then a finer contour pass restricted to one area, is three
    separate, independently addressable/deletable layers, not one
    growing blob. Note this is NOT idempotent -- contour mode's
    poisson sampling has genuine randomness, so re-running identical
    parameters produces a distinct (not byte-identical) layer, by
    design.

    n_workers (contour method only) parallelizes the main per-band loop
    across separate OS processes -- bands never spatially overlap by
    construction, so this is embarrassingly parallel with no
    coordination needed. None (default) auto-detects via os.cpu_count();
    1 forces sequential (e.g. for debugging). Output is BYTE-FOR-BYTE
    IDENTICAL to a sequential run at the same random_seed -- confirmed
    directly, not just assumed -- this only changes how fast it gets
    there, never what it produces. Forced to 1 regardless of what's
    passed whenever max_stamps is set (that flag needs a running total
    checked band-by-band, which is fundamentally sequential).
    """
    if method is None:
        project = load_project(working_dir)
        method = project.get("generate_terrain_method", "hex")
    if method not in ("hex", "contour", "raster"):
        raise StepError(f"method must be 'hex', 'contour', or 'raster', got {method!r}")

    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    project = load_project(working_dir)
    if pitch is None:
        pitch = project.get("generate_terrain_pitch_m", HEX_LATTICE_PITCH_M)
    if hex_spread_ratio is None:
        hex_spread_ratio = project.get(
            "generate_terrain_hex_spread_ratio", HEX_DEFAULT_SPREAD_RATIO
        )
    if hex_brush is None:
        hex_brush = project.get("generate_terrain_hex_brush", HEX_DEFAULT_BRUSH)
    if hex_tool is None:
        hex_tool = project.get("generate_terrain_hex_tool", TOOL_FLATTEN)
    if raster_size is None:
        raster_size = project.get("generate_terrain_raster_size", DEFAULT_RASTER_SIZE)
    if raster_size not in RASTER_SIZES:
        raise StepError(f"raster_size must be one of {RASTER_SIZES}, got {raster_size!r}")
    if raster_spread_ratio is None:
        raster_spread_ratio = project.get(
            "generate_terrain_raster_spread_ratio", DEFAULT_RASTER_SPREAD_RATIO
        )
    if raster_center_bias_ratio_x is None:
        raster_center_bias_ratio_x = project.get(
            "generate_terrain_raster_center_bias_ratio_x", DEFAULT_RASTER_CENTER_BIAS_RATIO
        )
    if raster_center_bias_ratio_z is None:
        raster_center_bias_ratio_z = project.get(
            "generate_terrain_raster_center_bias_ratio_z", DEFAULT_RASTER_CENTER_BIAS_RATIO
        )
    if raster_brush is None:
        raster_brush = project.get("generate_terrain_raster_brush", RASTER_BRUSH)
    if band_spacing_m is None:
        band_spacing_m = project.get("generate_terrain_band_spacing_m", DEFAULT_BAND_SPACING_M)
    if fill_mode is None:
        fill_mode = project.get("generate_terrain_fill_mode", DEFAULT_FILL_MODE)
    if fill_mode not in ("poisson", "rect"):
        raise StepError(f"fill_mode must be 'poisson' or 'rect', got {fill_mode!r}")
    if rect_brush is None:
        rect_brush = project.get("generate_terrain_rect_brush", DEFAULT_RECT_BRUSH)
    if rect_tolerance_m is None:
        rect_tolerance_m = project.get("generate_terrain_rect_tolerance_m", DEFAULT_RECT_TOLERANCE_M)
    if rect_min_length_m is None:
        rect_min_length_m = project.get("generate_terrain_rect_min_length_m", DEFAULT_RECT_MIN_LENGTH_M)
    if rect_max_search_distance_m is None:
        rect_max_search_distance_m = project.get(
            "generate_terrain_rect_max_search_distance_m", DEFAULT_RECT_MAX_SEARCH_DISTANCE_M
        )
    if rect_width_samples is None:
        rect_width_samples = project.get("generate_terrain_rect_width_samples", DEFAULT_RECT_WIDTH_SAMPLES)
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
    if enable_secondary_fill is None:
        enable_secondary_fill = project.get("generate_terrain_enable_secondary_fill", True)
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
    if use_height_mask is None:
        use_height_mask = project.get("generate_terrain_use_height_mask", False)
    if mask_buffer_px is None:
        mask_buffer_px = project.get("generate_terrain_mask_buffer_px", None)
    if remove_covered_stamps is None:
        remove_covered_stamps = project.get("generate_terrain_remove_covered_stamps", False)
    if remove_covered_margin_m is None:
        remove_covered_margin_m = project.get(
            "generate_terrain_remove_covered_margin_m", DEFAULT_REMOVE_COVERED_MARGIN_M
        )
    if n_workers is None:
        n_workers = project.get("generate_terrain_n_workers", DEFAULT_N_WORKERS)

    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first.")
    heightmap, _ = load_heightmap(heightmap_path)

    mask_geometry = None
    region_mask = None
    if use_height_mask:
        mask_path = working_dir / HEIGHT_MASK_FILE
        if not mask_path.exists():
            raise StepError(
                f"use_height_mask is on but no {HEIGHT_MASK_FILE} found under {working_dir}. "
                "Run --step ingest-osm first."
            )
        mask_geometry = load_height_mask(mask_path)
        if mask_geometry is None:
            print(f"  use_height_mask is on but {HEIGHT_MASK_FILE} has no geometry -- nothing to "
                  "mask, generating across the whole course")
        elif method == "contour":
            # Rasterized at the heightmap's own resolution so cells align
            # 1:1 with _band_mask's array inside generate_contour_layers --
            # restricts each band's fill footprint BEFORE the expensive
            # poisson pack/crumb scatter runs, not after.
            region_mask = rasterize_mask(mask_geometry, bounds, resolution=heightmap.shape[0])
            print(f"  height mask restricts this layer to {region_mask.mean():.1%} of the course "
                  "(cropped before filling, not filtered after)")
        else:
            print(f"  height mask restricts this layer's {method} lattice to inside "
                  f"{HEIGHT_MASK_FILE} (stamps outside it are never generated)")

    if method == "contour":
        if fill_mode == "rect":
            print(f"Rect band fill (band_spacing_m={band_spacing_m}, rect_brush={rect_brush}, "
                  f"rect_tolerance_m={rect_tolerance_m}, rect_min_length_m={rect_min_length_m}, "
                  f"rect_max_search_distance_m={rect_max_search_distance_m}, "
                  f"rect_width_samples={rect_width_samples}, "
                  f"pass2_radius={smoothing_min_radius * smooth_ratio} m)...")
        else:
            print(f"Two-pass poisson band fill (band_spacing_m={band_spacing_m}, "
                  f"pass1_radius=[{min_radius}, {max_radius}] m, step_ratio={radius_step_ratio}, "
                  f"pass2_radius={smoothing_min_radius * smooth_ratio} m, "
                  f"candidates_per_radius={'auto-tuning...' if candidates_per_radius is None else candidates_per_radius})...")
        if not enable_secondary_fill:
            print("  enable_secondary_fill is OFF -- pass 2 crumb cleanup is skipped, so coverage "
                  "will NOT be complete (whatever pass 1 leaves as crumbs stays unfilled)")
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

        def _print_auto_tuned_candidates(value: int) -> None:
            print(f"  auto-tuned candidates_per_radius = {value} "
                  f"(pass --candidates-per-radius {value} next time to skip this calibration step)")

        fitted = generate_contour_layers(
            heightmap, bounds,
            band_spacing_m=band_spacing_m,
            fill_mode=fill_mode,
            fill_brush=fill_brush,
            min_radius=min_radius,
            max_radius=max_radius,
            radius_step_ratio=radius_step_ratio,
            edge_distance_m=edge_distance_m,
            rect_brush=rect_brush,
            rect_tolerance_m=rect_tolerance_m,
            rect_min_length_m=rect_min_length_m,
            rect_max_search_distance_m=rect_max_search_distance_m,
            rect_width_samples=rect_width_samples,
            smoothing_brush=smoothing_brush,
            smoothing_min_radius=smoothing_min_radius,
            smooth_ratio=smooth_ratio,
            smooth_claim_fraction=smooth_claim_fraction,
            enable_secondary_fill=enable_secondary_fill,
            candidates_per_radius=candidates_per_radius,
            sweet_spot_ratio=sweet_spot_ratio,
            sweet_spot_sample_bands=sweet_spot_sample_bands,
            sweet_spot_seeds=sweet_spot_seeds,
            sweet_spot_max_candidates=sweet_spot_max_candidates,
            sweet_spot_time_budget_s=sweet_spot_time_budget_s,
            random_seed=random_seed,
            denoise_px=denoise_px,
            max_stamps=max_stamps,
            n_workers=n_workers,
            progress_callback=_print_contour_progress,
            on_candidates_tuned=_print_auto_tuned_candidates,
            region_mask=region_mask,
        )
        if max_stamps is not None and len(fitted) >= max_stamps:
            print(f"  {len(fitted)} stamps placed -- STOPPED at max_stamps={max_stamps}, "
                  "course is only partially filled (see note above)")
        else:
            print(f"  {len(fitted)} stamps placed (tiered band fill + crumb smoothing, all already fitted)")

    elif method == "raster":
        raster_radius = (raster_size / (2.0 * TYPE72_PLATEAU_FRACTION)) * raster_spread_ratio
        print(f"Generating raster grid (size={raster_size} m, spread_ratio={raster_spread_ratio}, "
              f"center_bias_ratio_x={raster_center_bias_ratio_x}, "
              f"center_bias_ratio_z={raster_center_bias_ratio_z}, brush={raster_brush}, "
              f"radius={raster_radius:.3f} m)...")
        if raster_brush != RASTER_BRUSH:
            print(f"  NOTE: brush {raster_brush} isn't type {RASTER_BRUSH} (the square brush this "
                  "grid's spacing math is derived from) -- stamps are placed at the same exact-"
                  "tiling radius regardless, so a circular brush leaves each cell's corners "
                  "uncovered rather than tiling seamlessly.")
        fitted, n_skipped = generate_raster_grid(
            bounds, heightmap, bounds, size=raster_size, spread_ratio=raster_spread_ratio,
            center_bias_ratio_x=raster_center_bias_ratio_x, center_bias_ratio_z=raster_center_bias_ratio_z,
            brush=raster_brush, mask_geometry=mask_geometry,
        )
        print(f"  {len(fitted)} stamps placed"
              + (f", {n_skipped} grid points skipped (no bare-earth heightmap coverage there)"
                 if n_skipped else "") + " (nearest-heightmap-sample, no fitting needed)")

    else:
        stamp_radius = 2.0 * pitch * hex_spread_ratio
        bleed = pitch
        print(f"Generating hex grid (pitch={pitch} m, spread_ratio={hex_spread_ratio}, "
              f"stamp_radius={stamp_radius} m, bleed={bleed} m, "
              f"brush={hex_brush}, tool={'raise' if hex_tool == TOOL_RAISE else 'flatten'})...")
        stamps = generate_hex_grid(
            bounds, pitch=pitch, stamp_radius=stamp_radius, brush=hex_brush, tool=hex_tool,
            bleed=bleed, mask_geometry=mask_geometry,
        )
        print(f"  {len(stamps)} stamps placed")

        print("Fitting stamp heights from the rasterized ground heightmap...")
        fitted = fit_stamp_heights(stamps, heightmap, bounds)
        n_unfitted = sum(1 for s in fitted if s.value == 0.0)
        if n_unfitted:
            print(f"  WARNING: {n_unfitted} stamps had too few nearby heightmap cells and kept "
                  "their placeholder value=0.0")

    next_n = len(_stamps_files(working_dir)) + 1
    if method == "contour" and next_n == 1:
        mean_elevation = float(np.nanmean(heightmap))
        print(f"Prepending a course-wide baseline-flatten stamp at the mean ground "
              f"elevation ({mean_elevation:.2f} m)...")
        baseline_stamp = build_baseline_flatten_stamp(bounds, mean_elevation)
        fitted = [baseline_stamp] + fitted
    elif method in ("hex", "raster"):
        print(f"Skipping the course-wide baseline-flatten stamp -- {method} mode's lattice "
              "already has full coverage on its own.")
    else:
        print(f"Skipping the course-wide baseline-flatten stamp -- this is layer "
              f"{next_n}, not the first, so prior layers already cover the course "
              "(re-adding it would flatten them back to mean elevation).")

    new_layer_id = _new_layer_id()
    if remove_covered_stamps:
        _flag_previous_layer_blocked(working_dir, mask_geometry, remove_covered_margin_m, new_layer_id)

    out_path = _stamps_dir(working_dir) / STAMPS_PATTERN.format(n=next_n)
    save_stamp_file(
        fitted, out_path, step="generate-terrain", layer_id=new_layer_id,
        parameters={"course_size_m": COURSE_SIZE_M, "pitch_m": pitch,
                     "hex_spread_ratio": hex_spread_ratio, "method": method,
                     "hex_brush": hex_brush, "hex_tool": hex_tool,
                     "raster_size": raster_size, "raster_spread_ratio": raster_spread_ratio,
                     "raster_center_bias_ratio_x": raster_center_bias_ratio_x,
                     "raster_center_bias_ratio_z": raster_center_bias_ratio_z,
                     "raster_brush": raster_brush,
                     "band_spacing_m": band_spacing_m, "fill_mode": fill_mode,
                     "rect_brush": rect_brush, "rect_tolerance_m": rect_tolerance_m,
                     "rect_min_length_m": rect_min_length_m,
                     "rect_max_search_distance_m": rect_max_search_distance_m,
                     "rect_width_samples": rect_width_samples,
                     "use_height_mask": use_height_mask,
                     "mask_buffer_px": mask_buffer_px,
                     "remove_covered_stamps": remove_covered_stamps,
                     "remove_covered_margin_m": remove_covered_margin_m},
    )
    print(f"  wrote {out_path}")

    save_project(working_dir, {
        "stamp_count": len(fitted),
        "generate_terrain_pitch_m": pitch,
        "generate_terrain_hex_spread_ratio": hex_spread_ratio,
        "generate_terrain_method": method,
        "generate_terrain_hex_brush": hex_brush,
        "generate_terrain_hex_tool": hex_tool,
        "generate_terrain_raster_size": raster_size,
        "generate_terrain_raster_spread_ratio": raster_spread_ratio,
        "generate_terrain_raster_center_bias_ratio_x": raster_center_bias_ratio_x,
        "generate_terrain_raster_center_bias_ratio_z": raster_center_bias_ratio_z,
        "generate_terrain_raster_brush": raster_brush,
        "generate_terrain_band_spacing_m": band_spacing_m,
        "generate_terrain_fill_mode": fill_mode,
        "generate_terrain_fill_brush": fill_brush,
        "generate_terrain_min_radius_m": min_radius,
        "generate_terrain_max_radius_m": max_radius,
        "generate_terrain_radius_step_ratio": radius_step_ratio,
        "generate_terrain_edge_distance_m": edge_distance_m,
        "generate_terrain_rect_brush": rect_brush,
        "generate_terrain_rect_tolerance_m": rect_tolerance_m,
        "generate_terrain_rect_min_length_m": rect_min_length_m,
        "generate_terrain_rect_max_search_distance_m": rect_max_search_distance_m,
        "generate_terrain_rect_width_samples": rect_width_samples,
        "generate_terrain_smoothing_brush": smoothing_brush,
        "generate_terrain_smoothing_min_radius_m": smoothing_min_radius,
        "generate_terrain_smooth_ratio": smooth_ratio,
        "generate_terrain_smooth_claim_fraction": smooth_claim_fraction,
        "generate_terrain_enable_secondary_fill": enable_secondary_fill,
        "generate_terrain_candidates_per_radius": candidates_per_radius,
        "generate_terrain_sweet_spot_ratio": sweet_spot_ratio,
        "generate_terrain_sweet_spot_sample_bands": sweet_spot_sample_bands,
        "generate_terrain_sweet_spot_seeds": sweet_spot_seeds,
        "generate_terrain_sweet_spot_max_candidates": sweet_spot_max_candidates,
        "generate_terrain_sweet_spot_time_budget_s": sweet_spot_time_budget_s,
        "generate_terrain_random_seed": random_seed,
        "generate_terrain_denoise_px": denoise_px,
        "generate_terrain_use_height_mask": use_height_mask,
        "generate_terrain_mask_buffer_px": mask_buffer_px,
        "generate_terrain_n_workers": n_workers,
    })

    print("Refreshing previews...")
    step_visualize(working_dir)


def step_generate_cart_paths(
    working_dir: Path,
    splines_path: Path | None = None,
    cart_path_surface: int | None = None,
    stamp_radius: float | None = None,
    spacing_m: float | None = None,
    height_avg_radius_m: float | None = None,
) -> None:
    """
    Cart path terrain-flattening stamps -- see terrain/cart_paths.py
    for the real algorithm (pearl necklace, directional rotation,
    real-heightmap height averaging, all built on this pipeline's own
    Stamp/BoundingBox primitives, not raw JSON dicts). This function is
    I/O only: load real inputs, call the algorithm, save real output --
    matching the same separation this project already keeps between
    contour_layers.py's algorithm and step_generate_terrain's I/O.

    ############################################################
    ASSUMPTION, not a confirmed fact: cart path spline data is read
    from splines_path (default: working_dir/"surfaceSplines.json",
    falling back to a couple of other plausible names -- see below),
    parsed as the same bezier waypoint JSON structure the course's own
    exported surfaceSplines.json uses. This project's OWN Feature
    system (features.geojson, loaded via load_features -- already used
    for water via `f.kind == "water"`) is a more "internal" possible
    source, but I have no confirmation cart paths are actually
    OSM-derivable the way water bodies are -- golf cart paths are
    often course-editor-drawn, not present in real-world OSM data for
    a given parcel. If cart paths DO exist as Feature objects with some
    kind (e.g. "cart_path"), that would be the better, more-internal
    source and this function should be pointed at it instead -- ask
    before assuming either way if uncertain.
    ############################################################

    Output is saved using the SAME stamps_{n}.json naming/loading
    convention every other layering step uses (see
    _stamps_files/load_all_stamps) -- not a separate file type
    needing its own plumbing through preview/export machinery that
    already knows how to composite every stamps_N.json in order. The
    metadata's own "step" field still reads "generate-cart-paths" so
    it stays clearly distinguishable from a real adaptive-refine pass
    despite sharing the loading mechanism.
    """
    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first.")
    if not _stamps_files(working_dir):
        raise StepError(f"No stamp layers found under {_stamps_dir(working_dir)}. "
                         "Run --step generate-terrain first -- cart path stamps flatten to the "
                         "average of terrain that should already exist.")

    project = load_project(working_dir)
    if cart_path_surface is None:
        cart_path_surface = project.get("cart_paths_surface_value", None)
    if stamp_radius is None:
        stamp_radius = project.get("cart_paths_stamp_radius_m", CART_PATH_STAMP_RADIUS)
    if spacing_m is None:
        spacing_m = project.get("cart_paths_spacing_m", CART_PATH_SPACING_M)
    if height_avg_radius_m is None:
        height_avg_radius_m = project.get("cart_paths_height_avg_radius_m", CART_PATH_HEIGHT_AVG_RADIUS_M)

    if cart_path_surface is None:
        raise StepError(
            "cart_path_surface is not set (no --cart-path-surface given and none saved in "
            "project.json) -- this is the surface value that identifies a cart path spline in "
            "the source JSON, and there's no safe default to fall back to. Set it explicitly."
        )

    if splines_path is None:
        for candidate in ("surfaceSplines2.json", "surfaceSplines.json"):
            candidate_path = working_dir / candidate
            if candidate_path.exists():
                splines_path = candidate_path
                break
        else:
            raise StepError(
                f"No surfaceSplines2.json or surfaceSplines.json found under {working_dir} -- "
                "pass --splines-path explicitly if cart path spline data lives somewhere else."
            )
    if not splines_path.exists():
        raise StepError(f"{splines_path} does not exist.")

    print(f"Loading heightmap from {heightmap_path}...")
    heights, bounds = load_heightmap(heightmap_path)

    print(f"Loading cart path splines from {splines_path}...")
    with splines_path.open() as f:
        raw_splines = json.load(f)

    def _get_surface_value(spline: dict) -> int | None:
        for container in (spline, spline.get("Value", {}), spline.get("value", {})):
            if not isinstance(container, dict):
                continue
            v = container.get("surface")
            if v is None:
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        return None

    def _get_is_closed(spline: dict) -> bool:
        for container in (spline, spline.get("Value", {}), spline.get("value", {})):
            if not isinstance(container, dict):
                continue
            for key in ("isClosed", "state"):
                v = container.get(key)
                if v is not None:
                    return bool(v)
        return False

    def _get_is_filled(spline: dict) -> bool:
        # ASSUMPTION -- see this function's own docstring header note.
        for container in (spline, spline.get("Value", {}), spline.get("value", {})):
            if not isinstance(container, dict):
                continue
            v = container.get("isFilled")
            if v is not None:
                return bool(v)
        return False

    cart_path_splines: list[CartPathSpline] = []
    n_skipped_wrong_surface = 0
    n_skipped_closed_or_filled = 0
    n_skipped_bad_geometry = 0

    for idx, spline in enumerate(raw_splines):
        surface = _get_surface_value(spline)
        if surface != cart_path_surface:
            n_skipped_wrong_surface += 1
            continue
        if _get_is_closed(spline) or _get_is_filled(spline):
            n_skipped_closed_or_filled += 1
            continue

        line = bezier_waypoints_to_linestring(spline.get("waypoints", []))
        if line is None:
            n_skipped_bad_geometry += 1
            continue

        cart_path_splines.append(CartPathSpline(line=line, source_id=str(idx)))

    print(f"  {len(raw_splines)} total splines: {len(cart_path_splines)} cart paths to process, "
          f"{n_skipped_wrong_surface} different surface, {n_skipped_closed_or_filled} closed/filled, "
          f"{n_skipped_bad_geometry} unusable geometry")

    if not cart_path_splines:
        print("  No cart path splines to process -- nothing to do.")
        return

    print(f"Generating cart path stamps (radius={stamp_radius:.4f}m, spacing={spacing_m:.4f}m, "
          f"height_avg_radius={height_avg_radius_m:.4f}m)...")
    stamps = generate_cart_path_stamps(
        cart_path_splines, heights, bounds,
        spacing_m=spacing_m, stamp_radius=stamp_radius, height_avg_radius_m=height_avg_radius_m,
        brush=CART_PATH_STAMP_TYPE,
    )
    print(f"  {len(stamps)} cart path stamps generated across {len(cart_path_splines)} paths")

    next_n = len(_stamps_files(working_dir)) + 1
    out_path = _stamps_dir(working_dir) / STAMPS_PATTERN.format(n=next_n)
    save_stamp_file(
        stamps, out_path, step="generate-cart-paths",
        parameters={
            "splines_path": str(splines_path), "cart_path_surface": cart_path_surface,
            "stamp_radius_m": stamp_radius, "spacing_m": spacing_m,
            "height_avg_radius_m": height_avg_radius_m,
        },
    )
    print(f"  wrote {out_path}")

    save_project(working_dir, {
        "cart_paths_surface_value": cart_path_surface,
        "cart_paths_stamp_radius_m": stamp_radius,
        "cart_paths_spacing_m": spacing_m,
        "cart_paths_height_avg_radius_m": height_avg_radius_m,
    })

    print("Refreshing previews...")
    step_visualize(working_dir)


def _select_stream_centerlines(features: list[Feature]) -> list[Feature]:
    """OSM water Features that are linear (LineString) streams/ditches --
    the source for step_generate_streams, as opposed to the filled water
    polygons course_output/water.py handles."""
    return [
        f for f in features
        if f.kind == "water"
        and f.geometry.geom_type == "LineString"
        and not f.geometry.is_empty
        and f.tags.get("waterway") in STREAM_WATERWAY_KINDS
    ]


def step_generate_streams(
    working_dir: Path, *,
    depth_m: float | None = None, half_width_m: float | None = None,
    water_fill_depth_m: float | None = None, water_base_width_m: float | None = None,
    water_widen_per_depth: float | None = None, water_widen_per_descent: float | None = None,
    water_level_margin_m: float | None = None, bank_veg_width_m: float | None = None,
) -> None:
    """
    Turn OSM stream/ditch centerlines (features.geojson water Features
    with LineString geometry and waterway=stream|ditch) into a full
    stream, the "compile once, format at write" way:

      1. A downhill-carved streambed, saved as the next stamps_N.json
         layer (step="generate-streams") -- picked up automatically by
         write-terrain AND write-water via _load_normalized_stamps, same
         as generate-cart-paths' output.
      2. streams.json -- a frozen, version-agnostic record of the pearl
         chain, one flow-orientation bearing per stream, and the pearls
         where the bed drops sharply enough for a waterfall. Consumed by
         write-water (flowing water strips, course_output/water.py's
         build_stream_water_objects) and write-objects (waterfall
         prefabs, v2021+ only).
      3. Synthetic stream-bank Features appended to features.geojson --
         the centerline buffered by BANK_VEG_WIDTH_M, tagged with
         PGA_CLUSTER_FILLS_TAG (DEFAULT_STREAM_BANK_VEG_SPECS, or
         project.json's "streams_bank_veg_specs"). These ride the
         existing pack-objects -> write-objects cluster-fill path.

    Re-runnable: a re-run REPLACES its own work -- it deletes any
    generate-streams stamp layer(s) sitting at the top of the stack
    (_clear_trailing_stream_layers) before writing the fresh trench, so
    the carve never stacks, and it strips any stream-bank Features a
    previous run added (STREAM_BANK_MARKER_TAG) before adding fresh ones.
    (A generate-streams layer BURIED under newer terrain layers is left
    alone, with a note.) The bank
    Features are synthetic (SYNTHETIC_MASKED_KIND), so
    a later ingest-osm now carries them over by default (same as the
    GUI's border rings) -- only ingest-osm --no-preserve-synthetic
    drops them, in which case re-run generate-streams afterwards.
    Re-running generate-streams after any ingest-osm stays correct
    regardless: it strips its own prior bank Features before re-adding,
    so the count can't double.
    """
    heightmap_path = working_dir / HEIGHTMAP_FILE
    if not heightmap_path.exists():
        raise StepError(f"No {HEIGHTMAP_FILE} found under {working_dir}. Run --step ingest-laz first.")
    if not _stamps_files(working_dir):
        raise StepError(
            f"No stamp layers found under {_stamps_dir(working_dir)}. Run --step generate-terrain "
            "first -- the streambed is carved relative to terrain that must already exist."
        )
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    project = load_project(working_dir)
    bank_veg_specs = project.get("streams_bank_veg_specs") or DEFAULT_STREAM_BANK_VEG_SPECS

    # CLI arg wins; else whatever a prior run saved in project.json; else
    # the module-constant default. Same idiom as step_generate_cart_paths.
    def _setting(arg, key, const):
        return arg if arg is not None else project.get(key, const)

    depth_m = _setting(depth_m, "streams_depth_m", STREAM_DEPTH_M)
    half_width_m = _setting(half_width_m, "streams_half_width_m", STREAM_HALF_WIDTH_M)
    water_fill_depth_m = _setting(
        water_fill_depth_m, "streams_water_fill_depth_m", STREAM_WATER_FILL_DEPTH_M)
    water_base_width_m = _setting(
        water_base_width_m, "streams_water_base_width_m", STREAM_WATER_BASE_WIDTH_M)
    water_widen_per_depth = _setting(
        water_widen_per_depth, "streams_water_widen_per_depth", STREAM_WATER_WIDEN_PER_DEPTH)
    water_widen_per_descent = _setting(
        water_widen_per_descent, "streams_water_widen_per_descent", STREAM_WATER_WIDEN_PER_DESCENT)
    water_level_margin_m = _setting(
        water_level_margin_m, "streams_water_level_margin_m", STREAM_WATER_LEVEL_MARGIN_M)
    bank_veg_width_m = _setting(bank_veg_width_m, "streams_bank_veg_width_m", BANK_VEG_WIDTH_M)

    full_features = load_features(features_path)
    course_features = _crop_features_to_course(working_dir, full_features)

    course_streams = _select_stream_centerlines(course_features)
    if not course_streams:
        print("No stream/ditch centerlines in features.geojson (water LineString + "
              f"waterway in {STREAM_WATERWAY_KINDS}) -- nothing to do.")
        return

    print(f"Loading heightmap from {heightmap_path}...")
    heights, bounds = load_heightmap(heightmap_path)

    centerlines = [
        StreamCenterline(line=f.geometry, source_id=f.osm_id, waterway=f.tags.get("waterway", "stream"))
        for f in course_streams
    ]
    print(f"  {len(centerlines)} stream/ditch centerline(s)")

    stamps = generate_stream_stamps(
        centerlines, heights, bounds, half_width_m=half_width_m, depth_m=depth_m,
    )
    if not stamps:
        print("  No usable streambed stamps (no finite heightmap data along any centerline) -- nothing written.")
        return

    # Replace, don't stack: drop any generate-streams layer(s) at the top
    # of the stack first (a re-run with different depth/half-width would
    # otherwise leave the old carve showing through at the flanks). A
    # buried stream layer can't be cleanly removed -- warn and append.
    cleared = _clear_trailing_stream_layers(working_dir)
    if not cleared and any(
        load_stamp_file(p)[2].get("step") == "generate-streams"
        for p in _stamps_files(working_dir)
    ):
        print("  NOTE: a previous generate-streams layer is buried under newer terrain layers -- "
              "leaving it (undo those first for a clean replace); the new trench stacks on the old.")

    # Appended as the next stamps_N.json, same as generate-cart-paths --
    # keeps preview-count / stamp-count in lockstep (see step_visualize)
    # and "delete the highest N" a clean undo.
    next_n = len(_stamps_files(working_dir)) + 1
    out_path = _stamps_dir(working_dir) / STAMPS_PATTERN.format(n=next_n)
    save_stamp_file(stamps, out_path, step="generate-streams", parameters={
        "waterway_kinds": list(STREAM_WATERWAY_KINDS), "stream_count": len(centerlines),
        "depth_m": depth_m, "half_width_m": half_width_m,
    })
    print(f"  wrote {out_path} ({len(stamps)} streambed stamp(s))")

    records = build_stream_records(
        centerlines, heights, bounds,
        depth_m=depth_m, half_width_m=half_width_m,
        water_fill_depth_m=water_fill_depth_m, water_base_width_m=water_base_width_m,
        water_widen_per_depth=water_widen_per_depth, water_widen_per_descent=water_widen_per_descent,
    )
    save_stream_records(records, working_dir / STREAMS_FILE)
    print(f"  wrote {working_dir / STREAMS_FILE}")

    # Stream-bank vegetation: buffer the FULL-frame centerlines (bank
    # polygons ride features.geojson's own uncropped frame; pack-objects
    # crops later) and tag each with the cluster-fill specs. Replace any
    # this step added on a previous run first.
    kept = [
        f for f in full_features
        if not (f.kind == SYNTHETIC_MASKED_KIND and f.tags.get(STREAM_BANK_MARKER_TAG))
    ]
    full_streams = _select_stream_centerlines(full_features)
    bank_count = 0
    for f in full_streams:
        poly = f.geometry.buffer(bank_veg_width_m, cap_style=2, join_style=2)
        if poly.is_empty or poly.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        kept.append(Feature(
            geometry=poly, kind=SYNTHETIC_MASKED_KIND,
            tags={
                PGA_CLUSTER_FILLS_TAG: [dict(s) for s in bank_veg_specs],
                STREAM_BANK_MARKER_TAG: True,
            },
            osm_id=next_synthetic_osm_id(kept), mask=True,
        ))
        bank_count += 1

    save_features(kept, features_path)
    print(f"  tagged {bank_count} stream-bank vegetation polygon(s) into {FEATURES_FILE} "
          "(run pack-objects + write-objects to place them)")

    save_project(working_dir, {
        "streams_bank_veg_specs": bank_veg_specs,
        "streams_depth_m": depth_m,
        "streams_half_width_m": half_width_m,
        "streams_water_fill_depth_m": water_fill_depth_m,
        "streams_water_base_width_m": water_base_width_m,
        "streams_water_widen_per_depth": water_widen_per_depth,
        "streams_water_widen_per_descent": water_widen_per_descent,
        "streams_water_level_margin_m": water_level_margin_m,
        "streams_bank_veg_width_m": bank_veg_width_m,
    })

    print("Refreshing previews...")
    step_visualize(working_dir)


def _clear_oob(working_dir: Path) -> None:
    """
    Remove the generated OOB band: delete oob.json + its previews and
    mark the project's outOfBounds layer as deliberately empty
    (oob_enabled=False), so the next write-terrain writes [] rather
    than leaving the last band stale. Also zeroes the array in any
    already-written userLayers.json for immediate effect.
    """
    removed = []
    oob_path = working_dir / OOB_FILE
    if oob_path.exists():
        oob_path.unlink()
        removed.append(OOB_FILE)
    preview_dir = working_dir / PREVIEW_DIR
    if preview_dir.exists():
        for p in preview_dir.glob("preview_oob*"):
            p.unlink()
            removed.append(p.name)

    nodes_dir = working_dir / "course" / "CourseDescription_nodes"
    for name in ("userLayers.json", "userLayers2.json"):
        ul = nodes_dir / name
        if not ul.exists():
            continue
        try:
            with ul.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("outOfBounds"):
            data["outOfBounds"] = []
            with ul.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            removed.append(f"{name}:outOfBounds")

    save_project(working_dir, {"oob_enabled": False})
    if removed:
        print("  cleared OOB: " + ", ".join(removed))
    else:
        print("  no OOB to clear.")


def step_generate_oob(
    working_dir: Path, *,
    inner_buffer_m: float | None = None, band_width_m: float | None = None,
    merge_gap_m: float | None = None, simplify_tol_m: float | None = None,
    cap_scale_ratio: float | None = None, include_caps: bool | None = None,
    clear: bool = False,
) -> None:
    """
    Auto-generate out-of-bounds paint: a constant-width band of brush
    stamps (round type-8 caps + stretched type-15 "smooth square"
    segments) running just outside the playable area.

    The boundary is derived from the same playable-Feature union the
    height mask is built from (fairway/green/tee + hole corridors, see
    ingest.osm.merge_height_mask_features) -- no OSM authoring needed.
    That union is buffered outward by inner_buffer_m (the gap between
    play and OOB), then a curve down the middle of a band_width_m-wide
    ring just outside it is walked: one round cap per simplified vertex,
    one stretched square per edge.

    Output: oob.json -- a frozen, version-agnostic record list
    (course-local frame), overwritten wholesale on every run (like
    parking.json). It is NOT a stamp layer and never reaches
    TerrainModel. step_write_terrain formats it into
    userLayers.json's "outOfBounds" array -- the tool owns that array
    once generate-oob has run for the project (project.json
    "oob_enabled"), so deleting oob.json / passing clear=True and
    re-running write-terrain removes the band.

    clear=True: delete oob.json + previews, mark the layer empty, and
    return (no regeneration).
    """
    if clear:
        _clear_oob(working_dir)
        return

    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    project = load_project(working_dir)

    def _setting(arg, key, const):
        return arg if arg is not None else project.get(key, const)

    inner_buffer_m = _setting(inner_buffer_m, "oob_inner_buffer_m", OOB_INNER_BUFFER_M)
    band_width_m = _setting(band_width_m, "oob_band_width_m", OOB_BAND_WIDTH_M)
    merge_gap_m = _setting(merge_gap_m, "oob_merge_gap_m", OOB_MERGE_GAP_M)
    simplify_tol_m = _setting(simplify_tol_m, "oob_simplify_tol_m", OOB_SIMPLIFY_TOL_M)
    cap_scale_ratio = _setting(cap_scale_ratio, "oob_cap_scale_ratio", OOB_CAP_SCALE_RATIO)
    include_caps = _setting(include_caps, "oob_include_caps", OOB_INCLUDE_CAPS)

    course_features = _crop_features_to_course(working_dir, load_features(features_path))
    playable = merge_height_mask_features(course_features)
    if playable is None or playable.is_empty:
        raise StepError(
            "No playable Features (fairway/green/tee/hole with mask=False) in "
            f"{FEATURES_FILE} -- nothing to base an OOB boundary on. Run --step ingest-osm "
            "and check the course has those features."
        )

    course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    records = build_oob_records(
        playable,
        inner_buffer_m=inner_buffer_m, band_width_m=band_width_m,
        merge_gap_m=merge_gap_m, simplify_tol_m=simplify_tol_m,
        cap_scale_ratio=cap_scale_ratio, include_caps=include_caps,
        course_bounds=course_bounds,
    )
    if not records:
        print("  OOB boundary produced no stamps (empty/degenerate playable area) -- nothing written.")
        return

    out_path = working_dir / OOB_FILE
    save_oob_records(records, out_path)
    n_caps = sum(1 for r in records if r.brush == 8)
    print(f"  wrote {out_path} ({len(records)} OOB stamp(s): {n_caps} caps, "
          f"{len(records) - n_caps} segments)")

    preview_path = working_dir / PREVIEW_DIR / PREVIEW_OOB
    viz.render_oob_preview(records, playable, course_bounds, preview_path)
    print(f"  wrote {preview_path}")
    print("  run --step write-terrain (then repack) to fold it into userLayers.json.")

    save_project(working_dir, {
        "oob_enabled": True,
        "oob_inner_buffer_m": inner_buffer_m,
        "oob_band_width_m": band_width_m,
        "oob_merge_gap_m": merge_gap_m,
        "oob_simplify_tol_m": simplify_tol_m,
        "oob_cap_scale_ratio": cap_scale_ratio,
        "oob_include_caps": bool(include_caps),
    })


def step_generate_collections(working_dir: Path, library_dir: Path | None = None) -> None:
    """
    Resolve every OSM 2-node way tagged pga_collection=<template name>
    (features.geojson "collection" Features -- see ingest/osm.py) against
    the collection library into collections.json, the frozen version-
    agnostic per-project record.

    Node 1 of the line is the anchor; the direction node 1 -> node 2 is
    the group heading. Every template member is rotated about the anchor
    by that heading and translated into place -- see
    course_output/collections.py's resolve_collection. An optional
    pga_parameter tag on the same way is passed through so one template
    can adapt per placement (member "variants" / a "{param}" asset-path
    token -- e.g. a hole sign that swaps by hole number).

    The library is a directory of *.json templates (one per template,
    see course_output/collection_library.py). Order of precedence:
    `library_dir` arg, then project.json's "collections_library_dir",
    then ~/.pga2k/collections/. A missing directory is just an empty
    library (every placement line then logs "unknown template" and is
    skipped -- the step still succeeds).

    "compile once, format at write" split, same as streams.json:
      - the object members ride pack-objects -> write-objects (folded
        into objects.json as kind="collection_object"),
      - the spline members ride write-splines (appended to
        surfaceSplines.json), and
      - the terrain-stamp members (raised flowerbed beds / berms --
        raise-tool only, never flatten) ride write-terrain / write-water
        (folded into the stamp list by _load_normalized_stamps, not
        persisted as a stamps_N.json layer).
    So run this before pack-objects / write-splines / write-terrain.
    Re-runnable; overwrites collections.json wholesale. Re-run after any
    fresh ingest-osm (which rewrites features.geojson) or library edit.
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    project = load_project(working_dir)
    if library_dir is None:
        saved = project.get("collections_library_dir")
        library_dir = Path(saved) if saved else default_library_dir()
    library_dir = Path(library_dir)
    print(f"Collection library: {library_dir}")

    features = _crop_features_to_course(working_dir, load_features(features_path))
    placements = [
        f for f in features
        if f.kind == "collection" and f.geometry.geom_type == "LineString"
        and len(f.geometry.coords) >= 2
    ]
    if not placements:
        print(f"No collection placement lines in {FEATURES_FILE} "
              f"(a 2-node way tagged {PGA_COLLECTION_TAG}=<template name>) -- nothing to do.")
        save_collection_records([], working_dir / COLLECTIONS_FILE)
        return

    library = load_library(library_dir)
    print(f"  {len(placements)} placement line(s), {len(library)} template(s) in the library")

    records: list[dict] = []
    skipped = 0
    for f in placements:
        name = f.tags.get(PGA_COLLECTION_TAG)
        template = library.get(name)
        if template is None:
            print(f"  NOTE: skipping placement (osm_id={f.osm_id}) -- no template named {name!r} "
                  f"in {library_dir}")
            skipped += 1
            continue
        (x1, z1), (x2, z2) = f.geometry.coords[0], f.geometry.coords[1]
        heading = _collection_bearing_deg(x2 - x1, z2 - z1)
        parameter = f.tags.get(PGA_PARAMETER_TAG)
        record = resolve_collection(
            template, x1, z1, heading, source_id=f.osm_id, parameter=parameter,
        )
        records.append(record)
        unresolved = sum(
            1 for o in record["objects"]
            if (isinstance(o.get("path"), str) and "{param}" in o["path"])
            or o.get("options_unresolved")
        )
        if unresolved:
            print(f"  WARNING: {name!r} placement (osm_id={f.osm_id}) has {unresolved} object(s) "
                  f"with an unresolved {{param}} asset path / param_option -- add a "
                  f"{PGA_PARAMETER_TAG} tag to this way, or a \"param_default\" to the template member")
        param_note = f" ({PGA_PARAMETER_TAG}={parameter!r})" if parameter is not None else ""
        print(f"  placed {name!r}{param_note} at ({x1:.1f}, {z1:.1f}) heading {heading:.0f}")

    save_collection_records(records, working_dir / COLLECTIONS_FILE)
    n_obj = sum(len(r["objects"]) for r in records)
    n_spl = sum(len(r["splines"]) for r in records)
    n_stamp = sum(len(r.get("stamps", [])) for r in records)
    print(f"  wrote {working_dir / COLLECTIONS_FILE} ({len(records)} placement(s), {n_obj} object(s), "
          f"{n_spl} spline(s), {n_stamp} terrain stamp(s); {skipped} skipped)")

    save_project(working_dir, {"collections_library_dir": str(library_dir)})

    # Collections only need OSM, so this step can legitimately run before
    # ingest-laz -- don't let a not-yet-possible preview refresh (or a
    # crop whose relief exceeds the in-game ceiling, etc.) fail the step
    # once collections.json is already written.
    try:
        print("Refreshing previews...")
        step_visualize(working_dir)
    except Exception as e:  # noqa: BLE001 -- cosmetic refresh, never fatal
        print(f"  (skipping preview refresh: {type(e).__name__}: {e})")


def step_generate_parking(
    working_dir: Path, *,
    spacing_m: float | None = None, offset_m: float | None = None,
    sides: str | None = None, orientation: str | None = None,
    skip_prob: float | None = None, max_variants: int | None = None,
    color_weights: str | None = None, accent_count: int | None = None,
    seed: int | None = None, no_bank: bool = False,
) -> None:
    """
    Line every OSM way tagged pga_parking=<pool filter> (features.geojson
    "parking" Features -- see ingest/osm.py) with parked-car props from
    course_output/vehicle_catalog.json, frozen into parking.json.

    Vehicle pool: cars only (no vans). Colour is WEIGHTED --
    --parking-color-weights (default black 0.40 / gray 0.15 / white 0.15)
    plus --parking-accent-count accent colours (from red/green/blue/
    yellow, chosen once per course) at 0.10 each. A colour not in the mix
    never spawns. The DISTINCT prefab count across the whole course is
    capped at `max_variants` (--parking-max-variants), >= 1 per active
    colour, so parked cars don't eat the game's placed-object type
    budget. A tag value further restricts one aisle: "yes"/"all"/empty =
    the default weighted mix; otherwise a comma list of colours and/or
    "car"/"van" (the weights still apply among survivors; "van" opts vans
    back in for that aisle), e.g. pga_parking=black,white.

    The aisle line is DIRECTIONAL: cars go on its LEFT side only by
    default (--parking-sides left|right|both -- draw the line the other
    way to flip). They're placed in a row perpendicular nose-in by
    default, spaced `spacing_m` apart, `offset_m` off the line, with a
    fraction `skip_prob` of stalls left empty and small position/heading
    jitter. A `ref` tag on the way (0..1) scales THAT aisle's occupancy:
    populated fraction = (1 - skip_prob) * ref (no tag == ref 1.0).
    Each car banks its pitch/roll to the local ground slope (heightmap),
    clamped to the editor's +-10 deg limit -- pass --no-parking-bank (or
    run before ingest-laz) to keep them flat.

    "compile once, format at write" split, same as collections.json: the
    cars ride pack-objects -> write-objects (folded into objects.json as
    kind="collection_object", carrying pitch/roll). v2021+ only -- the
    vehicle prefabs aren't in asset_catalog.json, so a v2019 write skips
    them with a NOTE.

    Re-runnable; overwrites parking.json wholesale. Run after ingest-osm
    (and ingest-laz for slope banking); re-run after any fresh ingest-osm
    or before pack-objects.
    """
    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(f"No {FEATURES_FILE} found under {working_dir}. Run --step ingest-osm first.")

    project = load_project(working_dir)

    def _setting(arg, key, const):
        return arg if arg is not None else project.get(key, const)

    spacing_m = _setting(spacing_m, "parking_spacing_m", PARKING_SPACING_M)
    offset_m = _setting(offset_m, "parking_offset_m", PARKING_OFFSET_M)
    sides = _setting(sides, "parking_sides", PARKING_SIDES)
    orientation = _setting(orientation, "parking_orientation", PARKING_ORIENTATION)
    skip_prob = _setting(skip_prob, "parking_skip_prob", PARKING_SKIP_PROB)
    max_variants = _setting(max_variants, "parking_max_variants", PARKING_MAX_VARIANTS)
    accent_count = _setting(accent_count, "parking_accent_count", PARKING_ACCENT_COUNT)
    weight_map = parse_color_weights(color_weights) or project.get("parking_color_weights") \
        or dict(PARKING_COLOR_WEIGHTS)
    seed = _setting(seed, "parking_seed", None)
    bank_to_slope = not no_bank

    features = _crop_features_to_course(working_dir, load_features(features_path))
    lines = [
        f for f in features
        if f.kind == "parking" and f.geometry.geom_type == "LineString"
        and len(f.geometry.coords) >= 2
    ]
    if not lines:
        print(f"No parking aisle lines in {FEATURES_FILE} "
              f"(a way tagged {PGA_PARKING_TAG}=<pool filter | yes>) -- nothing to do.")
        save_parking_records([], working_dir / PARKING_FILE)
        return

    heights = bounds = None
    heightmap_path = working_dir / HEIGHTMAP_FILE
    if bank_to_slope and heightmap_path.exists():
        heights, bounds = load_heightmap(heightmap_path)
    elif bank_to_slope:
        print(f"  NOTE: no {HEIGHTMAP_FILE} yet -- cars placed flat (run ingest-laz, then re-run "
              "for ground-slope banking)")

    def _aisle_fill(f) -> float:
        """The way's `ref` tag as an occupancy scale in [0, 1] (default
        1.0 -- see course_output/parking.py ParkingAisle.fill)."""
        raw = f.tags.get(PGA_PARKING_REF_TAG)
        if raw is None:
            return 1.0
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            print(f"  NOTE: aisle osm_id={f.osm_id} has an unparseable {PGA_PARKING_REF_TAG}={raw!r} "
                  "-- treating as fully parked (ref=1.0)")
            return 1.0

    aisles = [
        ParkingAisle(
            line=f.geometry, source_id=f.osm_id,
            pool=resolve_pool(f.tags.get(PGA_PARKING_TAG)),
            fill=_aisle_fill(f),
        )
        for f in lines
    ]
    print(f"  {len(aisles)} parking aisle(s)  spacing={spacing_m} offset={offset_m} "
          f"sides={sides} orientation={orientation}")

    kwargs: dict = dict(
        spacing_m=spacing_m, offset_m=offset_m, sides=sides, orientation=orientation,
        skip_prob=skip_prob, max_variants=int(max_variants),
        color_weights=weight_map, accent_count=int(accent_count),
        bank_to_slope=(heights is not None),
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    records = build_parking_records(
        aisles, heights,
        bounds if bounds is not None
        else BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M),
        **kwargs,
    )
    save_parking_records(records, working_dir / PARKING_FILE)
    print(f"  wrote {working_dir / PARKING_FILE}")

    save_project(working_dir, {
        "parking_spacing_m": spacing_m, "parking_offset_m": offset_m,
        "parking_sides": sides, "parking_orientation": orientation,
        "parking_skip_prob": skip_prob, "parking_max_variants": int(max_variants),
        "parking_color_weights": weight_map, "parking_accent_count": int(accent_count),
    })

    # No preview refresh: parked cars are objects only -- they don't
    # write a stamp layer, so none of the diagnostic PNGs change. The GUI
    # "Show objects" overlay reads parking.json / objects.json live.


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
    remove_covered_stamps: bool | None = None,
    remove_covered_margin_m: float | None = None,
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
    list) are written to the next stamps_N.json. Safe to run
    repeatedly -- each call reconstructs the full current terrain via
    load_all_stamps() (every stamps_N.json under stamps/, in order)
    and builds against that, so "run this a few times, watch coverage
    improve" is the expected way to iterate. Deleting the highest-
    numbered stamps_N.json undoes just that pass.

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

    remove_covered_stamps (default off, requires use_height_mask)
    flags stamps in the IMMEDIATELY-PRECEDING stamps_N.json layer as
    blocked_by this new one wherever their whole footprint sits inside
    this pass's own mask, shrunk inward by remove_covered_margin_m --
    same feature/semantics as step_generate_terrain's option of the
    same name (see terrain/stamp_containment.py, _flag_previous_layer_
    blocked).

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
    if remove_covered_stamps is None:
        remove_covered_stamps = project.get("refine_remove_covered_stamps", False)
    if remove_covered_margin_m is None:
        remove_covered_margin_m = project.get(
            "refine_remove_covered_margin_m", DEFAULT_REMOVE_COVERED_MARGIN_M
        )
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
    n_prior_layers = len(_stamps_files(working_dir))
    print(f"  {len(stamps)} stamps (cumulative across {n_prior_layers} prior layer(s))")
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
    mask_geometry = None
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
        "remove_covered_stamps": remove_covered_stamps,
        "remove_covered_margin_m": remove_covered_margin_m,
        "model_rebuild_interval": model_rebuild_interval,
        "candidate_brushes": list(candidate_brushes) if candidate_brushes is not None else None,
        "max_planar_rms": max_planar_rms,
        "planar_shrink_factor": planar_shrink_factor,
    }

    if new_stamps:
        new_layer_id = _new_layer_id()
        if remove_covered_stamps:
            print("  NOTE: remove_covered_stamps tests against the MASK (where this pass is allowed "
                  "to place hotspots), not against where it actually placed one -- refine only adds "
                  "stamps where error already exceeds tolerance, so parts of the mask can legitimately "
                  "stay untouched this run. A blocked previous-layer stamp there would show through. "
                  "Safer after a few refine passes over the same mask have already filled it in.")
            _flag_previous_layer_blocked(working_dir, mask_geometry, remove_covered_margin_m, new_layer_id)

        next_n = len(_stamps_files(working_dir)) + 1
        out_path = _stamps_dir(working_dir) / STAMPS_PATTERN.format(n=next_n)
        save_stamp_file(
            new_stamps, out_path, step="refine-terrain", parameters=parameters, layer_id=new_layer_id,
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


def _set_json_key_in_file(path: Path, value, key: str, label: str) -> None:
    """
    Set `key` to `value` in the JSON object at `path`, preserving every
    other key already there (same preserve-everything-else pattern
    write_user_layers.py uses for userLayers.json's sibling keys).
    No-ops with a note if `path` doesn't exist yet, rather than creating
    a file whose overall structure we don't actually know.
    """
    if not path.exists():
        print(f"NOTE: {label} is set ({value!r}) but {path} doesn't exist yet "
              "-- run --step ingest-course first if you want it applied there.")
        return

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    data[key] = value
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Set {label} to {value!r} in {path}")


def _apply_course_theme(course_dir: Path, project: dict) -> None:
    """
    Write the selected theme (the File tab's Theme selector / the
    objects_theme project.json field) into the extracted course/ JSON.
    No-ops if no theme has been picked, or if game_version isn't 2019.

    v2019 ONLY: v2019's theme is a numeric id living directly in
    CourseDescription.json/CourseMetadata.json's root, patched here.
    v2021+ has no numeric theme id at all (see objects.py's module
    docstring) -- its course-wide look/theme is whatever the bundled
    template (course_output/course_templates.py) already encodes, so
    there's nothing to patch; live-patching objects_theme's v2019 id
    into a v2021+ course's root would just corrupt whatever real theme
    the template shipped with.

    Called from step_write_terrain AND step_repack -- the latter so a
    theme change made in the Theme selector after the last Write
    Terrain still lands in the packed .course, rather than silently
    shipping a stale theme.

    ASSUMPTION: writing the integer theme id (matching THEMES_V2019 /
    objects_theme's existing representation) into both files' root --
    unconfirmed against a real CourseDescription.json/CourseMetadata.json
    sample. If the game doesn't pick up the theme after this, check
    whether it expects the theme *name* instead (see THEMES_V2019).
    """
    if project.get("game_version", DEFAULT_GAME_VERSION) != "2019":
        return
    theme_id = project.get("objects_theme")
    if theme_id is None:
        return
    _set_json_key_in_file(course_dir / "CourseDescription.json", theme_id, "theme", "theme")
    _set_json_key_in_file(course_dir / "CourseMetadata.json", theme_id, "courseTheme", "theme")


def _apply_course_name(course_dir: Path, project: dict) -> None:
    """
    Write the course name (the File tab's "Course name" field / the
    course_name project.json field) into the extracted course/ JSON.
    No-ops if no name has been set.

    Writes the "name" key into both CourseDescription.json and
    CourseMetadata.json -- which one the game actually reads from
    depends on version: confirmed CourseMetadata.json for 2019, with
    CourseDescription.json believed to be what later versions (2K21+)
    use instead. Writing both covers either case rather than guessing.
    (ASSUMPTION: the "name" key is right for CourseMetadata.json too --
    unconfirmed; if the name doesn't take, that key is the first thing
    to check.)

    Called from step_write_terrain AND step_repack -- the latter so a
    name change (or a course/ baseline reset, which reverts name to the
    template's, as switching to 2021 forces) made after the last Write
    Terrain still lands in the packed .course rather than shipping the
    template's baked-in name.
    """
    course_name = project.get("course_name")
    if not course_name:
        return
    _set_json_key_in_file(course_dir / "CourseDescription.json", course_name, "name", "course name")
    _set_json_key_in_file(course_dir / "CourseMetadata.json", course_name, "name", "course name")


def _load_normalized_stamps(
    working_dir: Path, registration_marks: bool, direct_height_shift: bool,
) -> tuple[list, "BoundingBox", float, float]:
    """
    Shared by step_write_terrain and step_write_water: load every
    stamp layer, optionally add registration marks, then
    height-normalize -- the exact stamp list/normalization both steps
    need to agree on, since water levels
    (course_output/water.py) are fit against the ALREADY-normalized
    stamp list, not the raw one. Returns (stamps, bounds, true_min,
    true_max) -- true_min/true_max are the pre-shift resolved range,
    used by the terrain step to persist output_height_shift_m/
    output_height_range_m.
    """
    print(f"Loading stamps from {working_dir} (all layers)...")
    stamps, added_collection_stamps = _load_all_stamps_incl_collections(working_dir)
    print(f"  {len(stamps)} stamps")

    # The value-shift normalization adds `shift` to EVERY stamp's value,
    # which corrupts a raise stamp's relative delta (it's not an absolute
    # height). Collection terrain stamps are all raise, so force the
    # shim-stamp method when any are present -- it leaves individual
    # values untouched and appends one course-wide raise instead.
    if added_collection_stamps and direct_height_shift:
        print("  NOTE: placed collections contributed raise-tool terrain stamps -- using the "
              "shim-stamp height normalization (not --direct-height-shift) so their relative "
              "deltas survive.")
        direct_height_shift = False

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
        if direct_height_shift:
            print("  Using --direct-height-shift: shifting each stamp's own value directly "
                  "instead of appending a course-wide shim stamp.")
            stamps = normalize_stamp_heights_by_value_shift(stamps, bounds)
        else:
            stamps = normalize_stamp_heights(stamps, bounds)
    except ValueError as e:
        raise StepError(str(e)) from e

    return stamps, bounds, true_min, true_max


def step_write_terrain(
    working_dir: Path, registration_marks: bool = False, direct_height_shift: bool = True,
) -> None:
    course_dir = working_dir / "course"
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)

    stamps, _bounds, true_min, true_max = _load_normalized_stamps(
        working_dir, registration_marks, direct_height_shift,
    )

    _ensure_course_baseline(working_dir)
    nodes_dir = course_dir / "CourseDescription_nodes"

    # Fold in auto-generated out-of-bounds paint. Once generate-oob has
    # run for this project, the tool owns userLayers.json's "outOfBounds"
    # array: oob.json present -> its entries; oob.json gone but the
    # project was OOB-enabled (project.json "oob_enabled" is now False
    # after a clear, or the file was deleted by hand) -> [] so the last
    # band doesn't linger; project never touched OOB -> leave the key
    # exactly as found. Re-run write-terrain after a baseline reset to
    # restore it, same as "height".
    oob_entries = None
    oob_path = working_dir / OOB_FILE
    if oob_path.exists():
        oob_entries = oob_records_to_entries(load_oob_records(oob_path), game_version)
    elif "oob_enabled" in project:
        oob_entries = []

    out_path = nodes_dir / "userLayers.json"
    write_user_layers(out_path, stamps=stamps, oob=oob_entries, game_version=game_version)
    print(f"Wrote {out_path}")
    if oob_entries:
        print(f"  including {len(oob_entries)} out-of-bounds stamp(s) from {OOB_FILE}")
    elif oob_entries == []:
        print("  outOfBounds cleared (no oob.json)")

    # If a course name has been set (see the GUI's "Course name" field /
    # project.json), write it into the course/ JSON. Also re-applied at
    # repack time (see _apply_course_name) so a later name change or
    # baseline reset isn't shipped stale.
    _apply_course_name(course_dir, project)

    # Same idea for the selected theme (see the GUI's Objects tab / the
    # objects_theme project.json field, set by --step write-objects or
    # the GUI dropdown).
    _apply_course_theme(course_dir, project)

    save_project(working_dir, {
        "output_height_shift_m": -true_min,
        "output_height_range_m": true_max - true_min,
    })


def step_write_water(
    working_dir: Path, registration_marks: bool = False, direct_height_shift: bool = True,
    multi_tile_water: bool = False,
    water_fill_mode: str = "edge",
    water_tile_tolerance_m: float = DEFAULT_WATER_TILE_TOLERANCE_M,
    water_tile_min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    water_tile_max_search_m: float = DEFAULT_WATER_TILE_MAX_SEARCH_M,
    water_tile_width_samples: int = DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    water_tile_redundancy_ratio: float = DEFAULT_WATER_TILE_REDUNDANCY_RATIO,
    water_tile_overlap_m: float = DEFAULT_WATER_TILE_OVERLAP_M,
    water_stripe_overlap_m: float = DEFAULT_WATER_STRIPE_OVERLAP_M,
    water_stripe_min_edge_m: float = DEFAULT_WATER_TILE_MIN_EDGE_M,
    water_stripe_tolerance_m: float = DEFAULT_WATER_STRIPE_TOLERANCE_M,
    water_stripe_max_stripes_per_side: int = DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
) -> None:
    """
    Writes only userLayers.json's "water" key, leaving "height" (and
    everything else) exactly as found -- split out from step_write_terrain
    because building water objects means re-running the full stamp
    load/normalize pipeline AND fitting each water polygon
    against a fresh TerrainModel (course_output/water.py), which is
    slow enough that it shouldn't be forced on every terrain-only
    iteration. Water is fit against the CURRENT stamp list here (same
    registration-marks/direct-height-shift settings as
    step_write_terrain), not whatever was last written to
    userLayers.json -- run this again after any terrain change that
    should be reflected in water levels.

    multi_tile_water (default False, byte-identical output to before
    this flag existed) fills each pond with several smaller, possibly-
    overlapping tiles hugging its real boundary instead of one single
    minimum-rotated-rectangle. water_fill_mode picks which multi-tile
    algorithm ("edge" -- the original, default -- or "stripe"; see
    course_output/water.py's fit_water_tiles/fit_water_stripes) and is
    only consulted when multi_tile_water is set. The water_tile_*/
    water_stripe_* args tune their own respective fill mode only.
    """
    course_dir = working_dir / "course"

    features_path = working_dir / FEATURES_FILE
    if not features_path.exists():
        raise StepError(
            f"No {FEATURES_FILE} found under {working_dir} -- run --step ingest-osm first "
            "if this course has water hazards."
        )

    _ensure_course_baseline(working_dir)
    nodes_dir = course_dir / "CourseDescription_nodes"

    stamps, _bounds, _true_min, _true_max = _load_normalized_stamps(
        working_dir, registration_marks, direct_height_shift,
    )

    print("Building water objects from OSM water features (course_output/water.py)...")
    features = load_features(features_path)
    features = _crop_features_to_course(working_dir, features)
    water_features = [f for f in features if f.kind == "water"]
    water_entries = build_water_objects(
        water_features, stamps, multi_tile_water=multi_tile_water, water_fill_mode=water_fill_mode,
        water_tile_tolerance_m=water_tile_tolerance_m, water_tile_min_edge_m=water_tile_min_edge_m,
        water_tile_max_search_m=water_tile_max_search_m, water_tile_width_samples=water_tile_width_samples,
        water_tile_redundancy_ratio=water_tile_redundancy_ratio, water_tile_overlap_m=water_tile_overlap_m,
        water_stripe_overlap_m=water_stripe_overlap_m, water_stripe_min_edge_m=water_stripe_min_edge_m,
        water_stripe_tolerance_m=water_stripe_tolerance_m,
        water_stripe_max_stripes_per_side=water_stripe_max_stripes_per_side,
    )

    streams_path = working_dir / STREAMS_FILE
    if streams_path.exists():
        print("Building stream water tiles (course_output/water.py)...")
        project = load_project(working_dir)
        # Tile width/level are recomputed here (not frozen in streams.json).
        # Level is FIT to the real carved terrain -- the same already-
        # normalized TerrainModel the pond fit uses -- so pass
        # height_shift_m=0 (the model is already in the normalized frame),
        # and it's robust even if write-terrain hasn't re-run since
        # generate-streams. Width knobs re-read from the same keys
        # step_generate_streams persisted.
        stream_model = TerrainModel(stamps)
        water_entries += build_stream_water_objects(
            load_stream_records(streams_path), 0.0,
            bed_sampler=stream_model.evaluate_many,
            level_margin_m=project.get("streams_water_level_margin_m", STREAM_WATER_LEVEL_MARGIN_M),
            water_fill_depth_m=project.get("streams_water_fill_depth_m", STREAM_WATER_FILL_DEPTH_M),
            water_base_width_m=project.get("streams_water_base_width_m", STREAM_WATER_BASE_WIDTH_M),
            water_widen_per_depth=project.get(
                "streams_water_widen_per_depth", STREAM_WATER_WIDEN_PER_DEPTH),
            water_widen_per_descent=project.get(
                "streams_water_widen_per_descent", STREAM_WATER_WIDEN_PER_DESCENT),
        )

    out_path = nodes_dir / "userLayers.json"
    write_user_layers(out_path, water=water_entries)
    print(f"Wrote {out_path} ({len(water_entries)} water object(s))")


def _resolve_project_theme(project: dict) -> str | None:
    """
    The theme NAME to resolve a bundled template with. Prefers the
    version-agnostic "theme" name string, falling back to mapping the
    numeric v2019 "objects_theme" id through THEMES_V2019 -- older
    projects (and any saved before the GUI started persisting "theme")
    only have the numeric id, yet the GUI still displays a theme for
    them, so the CLI should accept one too. Returns None if neither is
    set.
    """
    theme = project.get("theme")
    if theme:
        return theme
    return THEMES_V2019.get(project.get("objects_theme"))


def _extract_course_template(working_dir: Path) -> None:
    """
    Extract the bundled template (course_output/course_templates.py)
    matching project.json's game_version + theme into
    working_dir/course/, OVERWRITING whatever's already there --
    shared by _ensure_course_baseline (only called when course/ is
    missing entirely) and step_ingest_course (an unconditional reset).

    Wipes course/ first rather than extracting on top of it: a stale
    node file from a previous game_version (e.g. placedObjects2.json
    left over after switching from 2019 to 2021) would otherwise still
    get picked up by course_repack.py's blind glob over
    CourseDescription_nodes/*.json alongside the new placedObjects3.json,
    shipping both.
    """
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    theme = _resolve_project_theme(project)
    if not theme:
        raise StepError(
            "No theme set for this project -- pick one on the File tab (GUI) or run "
            "--step ingest-course --course-theme <name> (CLI) before running this step."
        )
    try:
        course_file = resolve_course_template(SCRIPT_DIR, game_version, theme)
    except FileNotFoundError as e:
        raise StepError(str(e)) from e

    script = SCRIPT_DIR / "util" / "course_extract.py"
    if not script.exists():
        raise StepError(f"course_extract.py not found at {script}")

    course_dir = working_dir / "course"
    if course_dir.exists():
        shutil.rmtree(course_dir)
    course_dir.mkdir(parents=True)

    print(f"Extracting template {course_file} -> {course_dir} ...")
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
    save_project(working_dir, {"source_course_template": str(course_file)})


def _ensure_course_baseline(working_dir: Path) -> None:
    """
    Make sure working_dir/course/ exists, auto-provisioning it from the
    bundled template for this project's game_version + theme if it
    doesn't -- called at the top of every writer/repack step so none of
    them require an explicit "ingest" action first. No-op if course/ is
    already populated (from a prior auto-provision, or a manual
    --step ingest-course reset).
    """
    if (working_dir / "course").is_dir():
        return
    _extract_course_template(working_dir)


def step_ingest_course(working_dir: Path, theme: str | None = None) -> None:
    """
    Reset working_dir/course/ from the bundled template matching this
    project's game_version + theme, OVERWRITING whatever's already
    there. Every other writer/repack step provisions course/
    automatically on first need (_ensure_course_baseline) and never
    requires this to be run first -- this is only needed explicitly
    when game_version or theme changes AFTER course/ was already
    populated from the old template. Kept as its own real, visible step
    (not folded silently into a settings change) because it's
    destructive to any accumulated course/ state.

    If `theme` is given, saves it to project.json first (so a fresh
    project's first "ingest" can set the theme in the same action the
    GUI's File tab exposes as one combined control).
    """
    if theme is not None:
        save_project(working_dir, {"theme": theme})
    _extract_course_template(working_dir)


BLANK_TEMPLATE_COURSE_FILE = "blank_template.course"
PUSH_COLLECTION_COURSE_FILE = "pushed_collection.course"


def _new_offline_course_id() -> str:
    """
    A fresh id in the shape the game's own offline saves use --
    "offlineSave" + 12 lowercase base-36-ish chars (real saves use
    [0-9a-z]; hex is a subset, close enough and unambiguous). Used so
    every pushed blank is a DISTINCT course in-game rather than
    overwriting the last one (the game keys on _id/courseId, not the
    display name or the on-disk filename).
    """
    return f"offlineSave{secrets.token_hex(6)}"


def _patch_blank_identity(course_dir: Path, name: str, course_id: str, timestamp_ms: int) -> None:
    """
    Rewrite the blank's display name + course id + timestamp in the
    staged CourseDescription.json / CourseMetadata.json, preserving
    every other key (same pattern as _set_json_key_in_file). CD uses
    "name"/"_id"; CM uses "name"/"courseId" plus its own "_id" of
    "<courseId>-Meta" (see an extracted template for the shapes).
    """
    cd = course_dir / "CourseDescription.json"
    if cd.exists():
        with cd.open(encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = name
        data["_id"] = course_id
        data["timeStamp"] = timestamp_ms
        with cd.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Set name={name!r}, _id={course_id!r} in {cd.name}")

    cm = course_dir / "CourseMetadata.json"
    if cm.exists():
        with cm.open(encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = name
        data["courseId"] = course_id
        data["_id"] = f"{course_id}-Meta"
        data["timeStamp"] = timestamp_ms
        with cm.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Set name={name!r}, courseId={course_id!r} in {cm.name}")


def step_push_blank_template(working_dir: Path, course_name: str | None = None) -> None:
    """
    Build a fresh, ready-to-ship blank .course from the bundled template
    for this project's CURRENT game_version + theme (course_output/
    course_templates.py), stamp it with a unique name + course id so the
    game sees each push as a new course, and write it to
    working_dir/blank_template.course.

    Deliberately does NOT touch working_dir/course/ -- the template is
    staged in a throwaway temp dir, so the pipeline's own accumulated
    course/ state is left exactly as it was. The GUI's "Push Blank to
    Game" button runs this, then copies the result into the game's
    Courses folder (same CLI-builds / GUI-copies split as
    repack -> "Copy to Game Folder").
    """
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    theme = _resolve_project_theme(project)
    if not theme:
        raise StepError(
            "No theme set for this project -- pick one on the File tab (GUI) or run "
            "--step ingest-course --course-theme <name> (CLI) before pushing a blank."
        )
    try:
        template_file = resolve_course_template(SCRIPT_DIR, game_version, theme)
    except FileNotFoundError as e:
        raise StepError(str(e)) from e

    serial = (course_name or "").strip() or f"LIDAR-{game_version}-{time.strftime('%Y%m%d%H%M%S')}"
    course_id = _new_offline_course_id()
    now_ms = int(time.time() * 1000)

    extract_script = SCRIPT_DIR / "util" / "course_extract.py"
    repack_script = SCRIPT_DIR / "util" / "course_repack.py"
    for script in (extract_script, repack_script):
        if not script.exists():
            raise StepError(f"{script.name} not found at {script}")

    stage_dir = Path(tempfile.mkdtemp(prefix="pga2k_blank_"))
    out_path = working_dir / BLANK_TEMPLATE_COURSE_FILE
    try:
        print(f"Staging template {template_file} (game_version={game_version}, theme={theme}) ...")
        result = subprocess.run(
            [sys.executable, str(extract_script), str(template_file), str(stage_dir)],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise StepError(f"course_extract.py failed (exit {result.returncode})")

        _patch_blank_identity(stage_dir, serial, course_id, now_ms)
        # v2019 numeric theme id patch (no-op for v2021+, whose template
        # already encodes its look) -- same call write-terrain/repack use.
        _apply_course_theme(stage_dir, project)

        result = subprocess.run(
            [sys.executable, str(repack_script), str(stage_dir), str(out_path)],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise StepError(f"course_repack.py failed (exit {result.returncode})")
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"Wrote blank course {out_path} (name={serial!r})")
    save_project(working_dir, {"blank_template_serial": serial})


def _inject_collection_into_course(nodes_dir: Path, record: dict, game_version: str) -> tuple[int, int, int]:
    """
    Fold one resolved collection record (see
    course_output/collections.py's resolve_collection -- course-local
    frame, heading already applied) into an EXTRACTED course's
    CourseDescription_nodes/ files, appending to whatever's already
    there rather than overwriting:

      - objects -> placedObjects2.json / placedObjects3.json (per
        game_version), merged with merge_object_groups
      - splines -> surfaceSplines.json
      - raise-tool terrain stamps -> userLayers.json "height" (appended,
        NOT via write_user_layers -- that replaces the whole array and
        would drop the template's own map-wide flatten datum stamp)

    Returns (n_objects, n_splines, n_stamps) actually written.
    """
    # --- objects ---
    objects = list(record.get("objects", []))
    if any(o.get("dy") is not None for o in objects):
        # Re-ground designed-elevation members against the staged
        # course's own flatten datum, so a later capture_from_course of
        # the edited course recovers the same `dy` (= y - datum).
        datum, _found = _collection_flatten_datum(nodes_dir)
        apply_terrain_heights(objects, lambda _x, _z: datum, 0.0)

    if game_version == "2019":
        new_groups = build_collection_objects_v2019(objects)
    else:
        new_groups = build_collection_objects_v2021(objects)

    obj_path = nodes_dir / schema_for(game_version).objects_filename
    n_objects = sum(len(g.get("Value", {}).get("items", [])) for g in new_groups)
    if new_groups:
        existing_groups = load_placed_objects(obj_path) if obj_path.exists() else []
        save_placed_objects(merge_object_groups(existing_groups + new_groups), obj_path)

    # --- splines ---
    new_splines = build_collection_splines([record])
    if new_splines:
        spl_path = nodes_dir / "surfaceSplines.json"
        if spl_path.exists():
            with spl_path.open(encoding="utf-8") as fh:
                existing_splines = json.load(fh)
        else:
            existing_splines = []
        with spl_path.open("w", encoding="utf-8") as fh:
            json.dump(existing_splines + new_splines, fh, indent=2)

    # --- terrain stamps (raise only) ---
    stamps = build_collection_stamps([record])
    n_stamps = 0
    if stamps:
        ul_path = nodes_dir / "userLayers.json"
        if ul_path.exists():
            with ul_path.open(encoding="utf-8") as fh:
                ul = json.load(fh)
        else:
            ul = {"height": []}
        ul.setdefault("height", [])
        ul["height"].extend(stamp_to_entry(s, game_version) for s in stamps)
        with ul_path.open("w", encoding="utf-8") as fh:
            json.dump(ul, fh, indent=2)
        n_stamps = len(stamps)

    return n_objects, len(new_splines), n_stamps


def step_push_collection(
    working_dir: Path, collection_name: str, library_dir: Path | None = None,
    course_name: str | None = None,
) -> None:
    """
    Build a fresh, editable .course containing ONE collection template's
    objects + surface splines + raise-tool terrain stamps, anchored at
    the course centre with heading 0, and write it to
    working_dir/pushed_collection.course. The GUI's "Push to Game for
    Editing" button runs this and then copies the result into the game's
    Courses folder -- open it in the in-game editor, tweak the props,
    then re-run "Capture from .course..." (same template name) to fold
    the edits back into the library.

    The inverse of course_output/collection_library.py's
    capture_from_course. Anchor at (COURSE_SIZE_M/2, COURSE_SIZE_M/2) +
    heading 0 means every member lands at its raw template (dx, dz) in
    the game's origin-centred grid, so a re-capture reproduces the same
    template within rounding.

    Like push-blank-template, this stages the bundled template for the
    project's game_version + theme in a throwaway temp dir and never
    touches working_dir/course/.
    """
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    theme = _resolve_project_theme(project)
    if not theme:
        raise StepError(
            "No theme set for this project -- pick one on the File tab (GUI) or run "
            "--step ingest-course --course-theme <name> (CLI) before pushing a collection."
        )
    try:
        template_file = resolve_course_template(SCRIPT_DIR, game_version, theme)
    except FileNotFoundError as e:
        raise StepError(str(e)) from e

    if library_dir is None:
        saved = project.get("collections_library_dir")
        library_dir = Path(saved) if saved else default_library_dir()
    library_dir = Path(library_dir)
    print(f"Collection library: {library_dir}")

    library = load_library(library_dir)
    template = library.get(collection_name)
    if template is None:
        known = ", ".join(sorted(library)) or "(none)"
        raise StepError(
            f"No collection template named {collection_name!r} in {library_dir}. "
            f"Known templates: {known}"
        )

    serial = (course_name or "").strip() or (
        f"COLL-{_collection_slug(collection_name)}-{time.strftime('%Y%m%d%H%M%S')}"
    )
    course_id = _new_offline_course_id()
    now_ms = int(time.time() * 1000)

    extract_script = SCRIPT_DIR / "util" / "course_extract.py"
    repack_script = SCRIPT_DIR / "util" / "course_repack.py"
    for script in (extract_script, repack_script):
        if not script.exists():
            raise StepError(f"{script.name} not found at {script}")

    stage_dir = Path(tempfile.mkdtemp(prefix="pga2k_collection_push_"))
    out_path = working_dir / PUSH_COLLECTION_COURSE_FILE
    try:
        print(f"Staging template {template_file} (game_version={game_version}, theme={theme}) ...")
        result = subprocess.run(
            [sys.executable, str(extract_script), str(template_file), str(stage_dir)],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise StepError(f"course_extract.py failed (exit {result.returncode})")

        record = resolve_collection(template, COURSE_SIZE_M / 2, COURSE_SIZE_M / 2, 0.0)
        n_obj, n_spl, n_stamp = _inject_collection_into_course(
            stage_dir / "CourseDescription_nodes", record, game_version,
        )
        print(f"  injected {n_obj} object(s), {n_spl} spline(s), {n_stamp} terrain stamp(s) "
              f"from collection {collection_name!r}")

        _patch_blank_identity(stage_dir, serial, course_id, now_ms)
        # v2019 numeric theme id patch (no-op for v2021+) -- same call
        # write-terrain / repack / push-blank-template use.
        _apply_course_theme(stage_dir, project)

        result = subprocess.run(
            [sys.executable, str(repack_script), str(stage_dir), str(out_path)],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise StepError(f"course_repack.py failed (exit {result.returncode})")
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"Wrote collection course {out_path} (name={serial!r})")
    save_project(working_dir, {
        "pushed_collection_serial": serial,
        "collections_library_dir": str(library_dir),
    })


def _stale_version_node_files(course_dir: Path, game_version: str) -> list[str]:
    """
    Node filenames under course/CourseDescription_nodes/ that belong to
    a DIFFERENT game_version's schema than `game_version` -- e.g. a
    leftover placedObjects2.json after switching from 2019 to 2021.

    Nothing ever deletes these: _ensure_course_baseline only extracts
    course/ from the bundled template when it's missing ENTIRELY, so a
    game_version switch on a project that already has a populated
    course/ leaves the old version's node file(s) sitting right next to
    the new version's freshly-written ones. course_repack.py then globs
    EVERY *.json node file (no version filtering) and ships both --
    the game may read the stale one instead of, or alongside, the
    current one. Only an explicit --step ingest-course (GUI: "Reset
    Course Baseline") wipes course/ first and avoids this.
    """
    nodes_dir = course_dir / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        return []
    current = schema_for(game_version)
    current_filenames = {
        current.objects_filename, current.holes_filename,
        current.splines_filename, current.userlayers_filename,
    }
    stale: set[str] = set()
    for version, schema in VERSION_SCHEMAS.items():
        if version == game_version:
            continue
        for filename in (
            schema.objects_filename, schema.holes_filename,
            schema.splines_filename, schema.userlayers_filename,
        ):
            if filename in current_filenames:
                continue  # shared name across versions (e.g. holes.json) -- not stale
            if (nodes_dir / filename).exists():
                stale.add(filename)
    return sorted(stale)


# Export-status categories -- see export_status(). Not a pipeline step
# itself (read-only, cheap: a project.json read + a handful of mtime
# stats), just a status query the GUI's Repack-section indicator lights
# (and, incidentally, a --step could expose from the CLI later) poll.
EXPORT_STATUS_MISSING = "missing"  # never written
EXPORT_STATUS_STALE = "stale"  # written, but an input changed since (or a leftover other-version file exists)
EXPORT_STATUS_FRESH = "fresh"  # written and up to date


def export_status(working_dir: Path) -> dict[str, str]:
    """
    One of EXPORT_STATUS_{MISSING,STALE,FRESH} for each of "height",
    "water", "splines", "holes", "objects" -- whether that output's
    course/CourseDescription_nodes/ file is caught up with the inputs
    it's actually built from. Purely mtime-based, no new project.json
    bookkeeping: FRESH means the output file exists, at least one real
    input for it exists too, and the output is newer than every one of
    those inputs; MISSING means either the output file doesn't exist
    yet, or NONE of its inputs exist yet either (e.g. the bundled blank
    template's own userLayers.json ships real default height/water
    content -- see course_output/course_templates.py -- so "the file
    exists" alone can't distinguish "write-terrain actually ran for
    this project" from "still whatever the template shipped with";
    requiring at least one real input keeps that case honestly MISSING
    instead of a misleading FRESH).

    height and water write into the SAME file (userLayers.json, each
    step only ever replaces its own key -- see userLayers.write_user_
    layers), so this can't perfectly isolate "did water get rewritten
    more recently than height" purely from one shared file mtime -- a
    narrow window (a stamp changes between a terrain write and a LATER
    water write) can show a false FRESH for height. Fine for a status
    light, not meant as a hard gate the way step_repack's
    _stale_version_node_files check is.

    "objects" also goes STALE whenever _stale_version_node_files finds
    a leftover other-version node file sitting in course/ -- the exact
    placedObjects2.json/placedObjects3.json collision that used to ship
    a stale export silently (see the conversation / step_repack).
    """
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    schema = schema_for(game_version)
    nodes_dir = working_dir / "course" / "CourseDescription_nodes"

    def _mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _latest_mtime(paths) -> float | None:
        times = [t for t in (_mtime(Path(p)) for p in paths) if t is not None]
        return max(times) if times else None

    def _status(out_filename: str, input_paths, extra_stale: bool = False) -> str:
        out_mtime = _mtime(nodes_dir / out_filename)
        if out_mtime is None:
            return EXPORT_STATUS_MISSING
        if extra_stale:
            return EXPORT_STATUS_STALE
        newest_input = _latest_mtime(input_paths)
        if newest_input is None:
            # No real input exists at all -- the output file existing is
            # likely just the blank template's own default content, not
            # confirmed output from this project's own write step.
            return EXPORT_STATUS_MISSING
        if newest_input > out_mtime:
            return EXPORT_STATUS_STALE
        return EXPORT_STATUS_FRESH

    stamps_mtime_paths = _stamps_files(working_dir)
    features_path = working_dir / FEATURES_FILE
    collections_path = working_dir / COLLECTIONS_FILE
    streams_path = working_dir / STREAMS_FILE
    oob_path = working_dir / OOB_FILE
    objects_json_path = working_dir / OBJECTS_FILE

    course_dir = working_dir / "course"
    has_stale_version_files = bool(_stale_version_node_files(course_dir, game_version))

    return {
        "height": _status(
            schema.userlayers_filename, [*stamps_mtime_paths, collections_path, oob_path],
        ),
        "water": _status(
            schema.userlayers_filename, [*stamps_mtime_paths, features_path, streams_path],
        ),
        "splines": _status(schema.splines_filename, [features_path, collections_path]),
        "holes": _status(schema.holes_filename, [features_path]),
        "objects": _status(
            schema.objects_filename, [objects_json_path], extra_stale=has_stale_version_files,
        ),
    }


def step_repack(working_dir: Path, filename: str) -> None:
    """
    Repack working_dir/course into a .course file via util/course_repack.py,
    invoked as a subprocess (see step_ingest_course).
    """
    _ensure_course_baseline(working_dir)
    course_dir = working_dir / "course"
    project = load_project(working_dir)
    game_version = project.get("game_version", DEFAULT_GAME_VERSION)

    stale = _stale_version_node_files(course_dir, game_version)
    if stale:
        raise StepError(
            f"course/CourseDescription_nodes/ has leftover file(s) from a different "
            f"game_version, left behind by switching game_version without resetting "
            f"course/: {', '.join(stale)}. These won't be overwritten by any write-* "
            f"step, and course_repack.py would ship them alongside game_version="
            f"{game_version!r}'s own files, which the game may read instead (this is "
            f"exactly what caused an old cluster-fill border to keep showing up after "
            f"switching to a spline fill on a version change -- see the conversation). "
            f"Run --step ingest-course (GUI: \"Reset Course Baseline\") to wipe and "
            f"re-extract course/ for game_version={game_version!r}, then redo "
            f"write-terrain/write-water/write-splines/write-holes/write-objects "
            f"before repacking again."
        )

    script = SCRIPT_DIR / "util" / "course_repack.py"
    if not script.exists():
        raise StepError(f"course_repack.py not found at {script}")

    filename = filename.strip()
    if not filename:
        raise StepError("Repack filename can't be empty.")
    if filename.lower().endswith(".course"):
        filename = filename[: -len(".course")]

    out_path = working_dir / f"{filename}.course"

    # Make sure the theme selected in the Objects tab and the course name
    # from the File tab are baked into the course/ JSON before we pack --
    # either may have changed since the last Write Terrain (the only
    # other place these are applied), and a course/ baseline reset
    # (forced when switching game_version, e.g. to 2021) reverts the name
    # to the template's own.
    _apply_course_theme(course_dir, project)
    _apply_course_name(course_dir, project)

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
# Import in-game edits -- reconcile a saved, hand-edited .course (exported
# by this tool, opened and edited in PGA Tour 2K's own in-game editor, then
# saved) against what this tool currently tracks. See the design notes this
# was scoped from for the full rationale; summary in step_import_ingame_edits's
# own docstring below.
# ---------------------------------------------------------------------------

def _canon_json_value(value):
    """Round every float leaf to this project's usual 3-decimal
    convention and recursively sort dict keys, so two structurally-
    identical dicts that differ only in float precision or key order
    canonicalize to the same value -- exact, rounded value-equality,
    no fuzzy epsilon (see step_import_ingame_edits's docstring)."""
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {k: _canon_json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canon_json_value(v) for v in value]
    return value


def _canon_json_key(value) -> str:
    """A hashable, order/precision-independent key for one item/entry
    dict -- used to build Counter multisets for the new-vs-expected diff."""
    return json.dumps(_canon_json_value(value), sort_keys=True)


def _group_key(key_dict: dict) -> tuple:
    return tuple(sorted(key_dict.items()))


def _group_label(key: dict) -> str:
    """Same label convention as objects.object_counts: the asset path
    for a v2021+ Key, or a category/type/theme string for v2019."""
    if "path" in key:
        return key["path"]
    return f"category={key.get('category')}/type={key.get('type')}/theme={key.get('theme')}"


def _item_match_key(item: dict) -> str:
    """Identity key for one placed-object item in the import diff --
    canonical JSON (3-dp rounded, key-sorted) with `position.y`
    REMOVED.

    y is deliberately excluded: the game resolves it engine-side on
    load (ground-snap for the usual "-Infinity" items, each prefab with
    its own vertical bleed) and bakes the resolved value back on save,
    and this tool re-derives it for collection members with a designed
    `dy` elevation from the current TerrainModel -- neither is
    reproducible to the exact 3 decimals `_canon_json_key` demands. So
    a collection object (hole sign, deck) matched the exported course
    on x/z/rotation/scale/path/options but NOT on y, and every
    `import-ingame-edits` run counted the whole collection as brand-new
    hand-placed edits, which `--commit` then appended to
    ingame_objects.json (compounding run over run) -- write-objects
    then emitted each one twice, once from collections.json and once
    per accumulated phantom copy. Matching on x/z is a sound "is this
    the same placement" test; a purely-vertical nudge in the editor is
    the rare cost."""
    stripped = copy.deepcopy(item)
    pos = stripped.get("position")
    if isinstance(pos, dict):
        pos.pop("y", None)
    return _canon_json_key(stripped)


def _flatten_group_field(
    groups: list[dict], field: str,
) -> tuple[dict[tuple, Counter], dict[tuple, dict], dict[tuple, dict[str, dict]]]:
    """(counters, key_reprs, items_by_match_key) -- one Counter of
    match-key strings (see _item_match_key) per distinct group Key,
    across every group's Value[field] list (field is "items",
    "clusters", or "splines"), each Key's own dict (for labeling
    later), and, per group Key, one representative raw item dict per
    match key (so a "new" item can be emitted with its real
    position.y, which the match key drops)."""
    counters: dict[tuple, Counter] = {}
    key_reprs: dict[tuple, dict] = {}
    items_by_match_key: dict[tuple, dict[str, dict]] = {}
    for g in groups:
        gk = _group_key(g.get("Key", {}))
        key_reprs[gk] = g.get("Key", {})
        counter = counters.setdefault(gk, Counter())
        reps = items_by_match_key.setdefault(gk, {})
        for item in g.get("Value", {}).get(field, []):
            mk = _item_match_key(item)
            counter[mk] += 1
            reps.setdefault(mk, item)
    return counters, key_reprs, items_by_match_key


def _diff_placed_object_items(
    expected_groups: list[dict], actual_groups: list[dict],
) -> tuple[dict[tuple, list[dict]], dict[tuple, int], dict[tuple, dict]]:
    """
    (new_items_by_key, missing_counts_by_key, key_reprs) -- a Counter
    multiset diff (not a plain set: two genuinely-identical placements
    must not collapse into one) of every group's "items" list, keyed by
    the group's own Key (asset identity). Items are matched on
    _item_match_key (everything but position.y -- see its docstring).
    new_items_by_key holds the actual decoded item dicts present in
    `actual_groups` beyond what `expected_groups` accounts for, at the
    same multiplicity; missing_counts_by_key is report-only (this
    tool's own generation is never auto-removed just because it's
    absent from an imported course).
    """
    expected_counters, expected_reprs, _ = _flatten_group_field(expected_groups, "items")
    actual_counters, actual_reprs, actual_items = _flatten_group_field(actual_groups, "items")
    key_reprs = {**expected_reprs, **actual_reprs}

    new_items: dict[tuple, list[dict]] = {}
    missing_counts: dict[tuple, int] = {}
    for gk in set(expected_counters) | set(actual_counters):
        expected_c = expected_counters.get(gk, Counter())
        actual_c = actual_counters.get(gk, Counter())
        extra = actual_c - expected_c
        if extra:
            items = []
            for match_key, count in extra.items():
                items.extend([actual_items[gk][match_key]] * count)
            new_items[gk] = items
        missing = expected_c - actual_c
        if missing:
            missing_counts[gk] = sum(missing.values())
    return new_items, missing_counts, key_reprs


def _diff_stamp_entries(expected_entries: list[dict], actual_entries: list[dict]) -> tuple[list[dict], int]:
    """(new_entries, missing_count) -- same Counter-multiset diff as
    _diff_placed_object_items, applied directly to userLayers.json
    "height" entry dicts. No need to reverse them into Stamp objects
    just to compare: stamp_to_entry(expected_stamp) already produces
    the exact dict shape/rounding a real write-terrain run would, which
    is exactly what's sitting in the imported course's own
    userLayers.json if it's unedited."""
    expected_c = Counter(_canon_json_key(e) for e in expected_entries)
    actual_c = Counter(_canon_json_key(e) for e in actual_entries)
    extra = actual_c - expected_c
    new_entries = []
    for entry_json, count in extra.items():
        new_entries.extend([json.loads(entry_json)] * count)
    missing = expected_c - actual_c
    return new_entries, sum(missing.values())


def _placed_item_to_ingame_object(key: dict, item: dict, group_name: str) -> dict:
    """One ingame_objects.json record from a raw placedObjects2/3.json
    group Key + one of its Value["items"] entries -- inverse of
    objects._placed_item plus the Key's own asset identity. `y` is
    captured VERBATIM (see course_output/ingame_objects.py's module
    docstring) -- None if the source had the literal "-Infinity"."""
    position = item["position"]
    rotation = item.get("rotation", {})
    scale = item.get("scale", {})
    y = position.get("y")
    return {
        "path": key.get("path"),
        "category": key.get("category"),
        "type": key.get("type"),
        "theme": key.get("theme"),
        "x": round(position["x"] + GRID_ORIGIN_OFFSET, 3),
        "z": round(position["z"] + GRID_ORIGIN_OFFSET, 3),
        "rotation_deg": rotation.get("y", 0.0),
        "scale": scale.get("x", 1.0),
        "y": None if isinstance(y, str) else y,
        "group": group_name,
    }


def _ingame_record_identity(rec: dict) -> tuple:
    """Dedup key for an ingame_objects.json record -- asset identity plus
    3-dp position/rotation/scale, ignoring `y` and `group` (same fields,
    same reasons, as _item_match_key). Two records with this identity
    are the same placement and must not both be appended."""
    def _r(v):
        return round(v, 3) if isinstance(v, (int, float)) else v
    return (
        rec.get("path"), rec.get("category"), rec.get("type"), rec.get("theme"),
        _r(rec.get("x")), _r(rec.get("z")), _r(rec.get("rotation_deg")), _r(rec.get("scale")),
    )


def _stamp_entry_to_stamp(entry: dict) -> Stamp:
    """One terrain.stamp.Stamp from a raw userLayers.json "height"
    entry -- inverse of course_output.userLayers.stamp_to_entry: x/z
    from position + GRID_ORIGIN_OFFSET, scale_x/z from scale, value
    from value, brush from type, rotation from rotation.y, tool from
    tool -- every field directly recoverable, nothing inferred."""
    position = entry["position"]
    rotation = entry.get("rotation", {})
    scale = entry.get("scale", {})
    return Stamp(
        x=position["x"] + GRID_ORIGIN_OFFSET,
        z=position["z"] + GRID_ORIGIN_OFFSET,
        scale_x=scale.get("x", 1.0),
        scale_z=scale.get("z", 1.0),
        value=entry.get("value", 0.0),
        brush=entry.get("type"),
        rotation=rotation.get("y", 0.0),
        tool=entry.get("tool", TOOL_FLATTEN),
    )


def step_import_ingame_edits(
    working_dir: Path,
    edited_course_file: Path,
    game_version: str | None = None,
    commit: bool = False,
    group_name: str | None = None,
    registration_marks: bool = False,
    direct_height_shift: bool = True,
) -> None:
    """
    Reconcile a saved, EDITED .course (exported by this tool, then
    hand-edited in PGA Tour 2K's own in-game object/terrain editor and
    saved) against what this tool currently tracks. Never touches
    `working_dir/course/` itself: `edited_course_file` is extracted
    into a scratch temp dir, diffed, and discarded.

    "Expected" -- what this tool's own pipeline currently believes
    should be in the course -- is recomputed in memory via the exact
    same construction a real write would produce (_build_placed_objects
    for objects, _load_normalized_stamps for terrain, at whatever
    settings are currently saved in project.json), NOT a second, hand-
    maintained copy that could silently drift out of sync. "Actual" is
    read straight out of the freshly-extracted edited course's own
    placedObjects2/3.json / userLayers.json.

    Matching is exact, rounded value-equality (this project's usual
    3-decimal convention) via a Counter multiset diff -- two
    legitimately-identical placements (or two identical-valued stamps)
    must not collapse into one under a plain set difference. Anything
    in the saved file with no match (at that multiplicity) in the
    recomputed "expected" set is new -- added in the in-game editor.
    Anything expected but missing from the saved file is reported only,
    never auto-removed -- removing tracked generation automatically
    would be surprising and hard to undo correctly. A small in-game
    nudge to an existing object therefore reads as "new" (the moved
    one) plus a "missing" note for the original, not "moved" -- there's
    no stable id in the game's own JSON to track identity by.

    commit=False (the default) only prints the diff summary -- counts
    per asset/group, new stamp count -- and changes nothing on disk.
    commit=True additionally: appends every new placed-object item as a
    new ingame_objects.json record (group=`group_name`, default
    "Imported <today's date>"), and, if any new terrain stamps were
    found, reverses them back into Stamp objects and writes them as the
    next stamps_N.json layer (same append idiom every other generation
    step uses -- folds into TerrainModel and undo-by-delete for free).

    registration_marks/direct_height_shift default to exactly what
    step_write_terrain/step_write_water default to -- neither step
    persists these to project.json (they're plain CLI/GUI-checkbox
    flags, not project settings), so there's no "saved" value to read
    here either; pass the same flags you used for the write-terrain run
    that actually produced the course you're importing if you
    overrode them there.

    Importing a .course this tool never produced (or an old export)
    isn't specially detected -- "expected" just won't line up with
    anything, and everything reads as new. That's accepted as a
    (deliberately unoptimized) valid way to bulk-import a foreign
    course's objects; only object Keys this project's own asset catalog
    can resolve will actually convert into ingame_objects.json records
    at commit time (unresolvable ones are skipped with a note -- see
    course_output/ingame_objects.py's build_ingame_objects_v20XX).
    """
    edited_course_file = Path(edited_course_file)
    if not edited_course_file.is_file():
        raise StepError(f"{edited_course_file} doesn't exist.")

    project = load_project(working_dir)
    if game_version is None:
        game_version = project.get("game_version", DEFAULT_GAME_VERSION)
    if game_version not in IMPLEMENTED_GAME_VERSIONS:
        raise StepError(
            f"game_version={game_version!r} isn't implemented yet (only {IMPLEMENTED_GAME_VERSIONS} "
            "are) -- see objects.py's module docstring."
        )
    schema = schema_for(game_version)

    script = SCRIPT_DIR / "util" / "course_extract.py"
    if not script.exists():
        raise StepError(f"course_extract.py not found at {script}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="pga2k_import_"))
    try:
        print(f"Extracting {edited_course_file} -> {tmp_dir} "
              f"(scratch -- not touching {working_dir / 'course'}) ...")
        result = subprocess.run(
            [sys.executable, str(script), str(edited_course_file), str(tmp_dir)],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            raise StepError(f"course_extract.py failed (exit {result.returncode})")

        nodes_dir = tmp_dir / "CourseDescription_nodes"
        objects_node_path = nodes_dir / schema.objects_filename
        if not objects_node_path.exists():
            raise StepError(
                f"{edited_course_file} has no {schema.objects_filename} under "
                f"CourseDescription_nodes/ -- this project targets game_version={game_version!r}, "
                "so it expects that file. If the edited course was actually saved from a different "
                "game version, pass --game-version explicitly to match it."
            )
        actual_groups = load_placed_objects(objects_node_path)

        userlayers_path = nodes_dir / "userLayers.json"
        actual_stamp_entries: list[dict] = []
        if userlayers_path.exists():
            with userlayers_path.open(encoding="utf-8") as f:
                actual_stamp_entries = json.load(f).get("height", [])

        # "Expected" -- the exact construction a real write-objects/
        # write-terrain run would produce right now, against whatever
        # this project currently has saved (no override here: this is
        # a diff against the CURRENT tracked state, not a one-off run).
        (r_game_version, r_theme, r_tree_variety, r_tree_theme_config, r_tree_asset_paths,
         r_tree_type_asset_paths, r_stake_asset_path, r_stake_buildings, r_waterfall_asset_path,
         r_splash_asset_path) = _resolve_write_objects_params(
            project, game_version, None, None, None, None, None, None, None, None, None,
        )
        expected_groups = _build_placed_objects(
            working_dir, r_game_version, r_theme, r_tree_variety, r_tree_theme_config,
            r_tree_asset_paths, r_tree_type_asset_paths, r_stake_asset_path, r_stake_buildings,
            r_waterfall_asset_path, r_splash_asset_path, project,
        )

        expected_stamps, _bounds, _true_min, _true_max = _load_normalized_stamps(
            working_dir, registration_marks, direct_height_shift,
        )
        expected_stamp_entries = [stamp_to_entry(s, game_version) for s in expected_stamps]

        new_items_by_key, missing_counts_by_key, key_reprs = _diff_placed_object_items(
            expected_groups, actual_groups,
        )
        new_stamp_entries, missing_stamp_count = _diff_stamp_entries(
            expected_stamp_entries, actual_stamp_entries,
        )

        total_new_items = sum(len(v) for v in new_items_by_key.values())
        total_missing_items = sum(missing_counts_by_key.values())

        print(f"\nDiff vs. {edited_course_file}:")
        if not new_items_by_key and not new_stamp_entries and not total_missing_items and not missing_stamp_count:
            print("  No differences -- the imported course matches what this tool currently tracks exactly.")
        for gk, items in new_items_by_key.items():
            print(f"  {len(items)} new item(s): {_group_label(key_reprs[gk])}")
        for gk, count in missing_counts_by_key.items():
            print(f"  NOTE: {count} tracked item(s) missing from the imported course "
                  f"(left untouched, not removed): {_group_label(key_reprs[gk])}")
        if new_stamp_entries:
            print(f"  {len(new_stamp_entries)} new terrain stamp(s)")
        if missing_stamp_count:
            print(f"  NOTE: {missing_stamp_count} tracked terrain stamp(s) missing from the imported "
                  "course (left untouched, not removed)")

        if not commit:
            print("\n(dry run -- pass --commit to apply. Nothing was written.)")
            return

        if group_name is None:
            group_name = f"Imported {time.strftime('%Y-%m-%d')}"

        if total_new_items:
            ingame_objects_path = working_dir / INGAME_OBJECTS_FILE
            existing_records = load_ingame_objects(ingame_objects_path) if ingame_objects_path.exists() else []
            # Guard against a second --commit of the same edit set landing
            # duplicate records (the match-key diff above already ignores
            # y, so a re-import normally sees no new items -- this is the
            # belt-and-braces case: importing before the first commit's
            # records are picked up, or two courses sharing edits).
            seen = {_ingame_record_identity(r) for r in existing_records}
            new_records = []
            skipped_dupes = 0
            for gk, items in new_items_by_key.items():
                key = key_reprs[gk]
                for item in items:
                    rec = _placed_item_to_ingame_object(key, item, group_name)
                    ident = _ingame_record_identity(rec)
                    if ident in seen:
                        skipped_dupes += 1
                        continue
                    seen.add(ident)
                    new_records.append(rec)
            save_ingame_objects(existing_records + new_records, ingame_objects_path)
            msg = f"\nAppended {len(new_records)} object(s) (group={group_name!r}) to {ingame_objects_path}"
            if skipped_dupes:
                msg += f" ({skipped_dupes} already tracked -- skipped)"
            print(msg)

        if new_stamp_entries:
            # userLayers.json "height" values live in the EXPORTED frame
            # (resolved terrain min shifted to 0 -- see _load_normalized_stamps
            # / normalize_stamp_heights[_by_value_shift]), but a stamps_N.json
            # layer is in the INTERNAL absolute-elevation frame that write
            # re-normalizes from scratch. A flatten stamp painted in the
            # in-game editor is therefore ~`output_height_shift_m` too low if
            # taken verbatim -- undo that shift by adding the pre-shift
            # resolved min back. Raise-tool values are frame-invariant deltas,
            # so they're left alone.
            new_stamps = []
            for e in new_stamp_entries:
                s = _stamp_entry_to_stamp(e)
                if s.tool == TOOL_FLATTEN:
                    s = dataclasses.replace(s, value=s.value + _true_min)
                new_stamps.append(s)
            next_n = len(_stamps_files(working_dir)) + 1
            stamps_path = _stamps_dir(working_dir) / STAMPS_PATTERN.format(n=next_n)
            save_stamp_file(
                new_stamps, stamps_path, step="import-ingame-edits",
                parameters={"source_course": str(edited_course_file), "group": group_name},
            )
            print(f"Wrote {len(new_stamps)} new terrain stamp(s) -> {stamps_path}")

        if not total_new_items and not new_stamp_entries:
            print("\nNothing to commit.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEPS = {
    "init": step_init,
    "ingest-laz": step_ingest_laz,
    "ingest-osm": step_ingest_osm,
    "ingest-course": step_ingest_course,
    "push-blank-template": step_push_blank_template,
    "dig-water": step_dig_water,
    "generate-terrain": step_generate_terrain,
    "generate-cart-paths": step_generate_cart_paths,
    "generate-streams": step_generate_streams,
    "generate-oob": step_generate_oob,
    "generate-collections": step_generate_collections,
    "generate-parking": step_generate_parking,
    "push-collection": step_push_collection,
    "refine-terrain": step_refine_terrain,
    "write-terrain": step_write_terrain,
    "write-water": step_write_water,
    "write-splines": step_write_splines,
    "write-holes": step_write_holes,
    "generate-trees": step_generate_trees,
    "pack-objects": step_pack_objects,
    "write-objects": step_write_objects,
    "repack": step_repack,
    "import-ingame-edits": step_import_ingame_edits,
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
                              "hex-grid stamp lattice CENTERS (terrain/hexgrid.py's "
                              "HEX_LATTICE_PITCH_M) -- smaller pitch means more, more tightly-packed "
                              "lattice points. Edge bleed derives from this automatically (bleed="
                              "pitch); stamp radius instead comes from --hex-spread-ratio. Default: "
                              f"use whatever's saved in project.json, or {HEX_LATTICE_PITCH_M} if "
                              "never set.")
    parser.add_argument("--hex-spread-ratio", type=float, default=None,
                         help="generate-terrain, hex method only: scales stamp radius independently "
                              "of pitch -- stamp_radius = 2*pitch*hex_spread_ratio. 1.0 (default) "
                              "reproduces the original fixed radius = 2*pitch (each stamp reaches "
                              "exactly to its nearest neighbors' centers); >1 grows stamps past their "
                              "neighbors for more overlap/blending, <1 shrinks them, potentially "
                              "opening coverage gaps between lattice centers. Does NOT affect edge "
                              "bleed or lattice center spacing. Default: use whatever's saved in "
                              f"project.json, or {HEX_DEFAULT_SPREAD_RATIO} if never set.")
    parser.add_argument("--generate-terrain-method", type=str, default=None,
                         choices=["hex", "contour", "raster"],
                         help="generate-terrain: 'hex' (default) is the flat hex lattice. 'contour' "
                              "traces elevation-band contours of the real heightmap and places stamps "
                              "along each ring, spaced by local curvature, with a distance-transform "
                              "gap-fill pass for flat interiors ring-tracing can't reach on its own -- "
                              "see terrain/contour_layers.py. 'raster' is a flat, non-offset square "
                              "grid of hard type-72 stamps, each valued at the nearest heightmap cell "
                              "to its own center (no fitting, no averaging) -- see "
                              "terrain/rastergrid.py. Default: use whatever's saved in "
                              "project.json, or 'hex' if never set.")
    parser.add_argument("--hex-brush", type=int, default=None,
                         help="generate-terrain, hex method only: brush every lattice stamp uses. "
                              "Default: use whatever's saved in project.json, or "
                              f"{HEX_DEFAULT_BRUSH} (terrain/hexgrid.py's DEFAULT_BRUSH) if never set.")
    parser.add_argument("--hex-tool", type=int, default=None, choices=[0, 1],
                         help="generate-terrain, hex method only: tool every lattice stamp uses -- "
                              "0=flatten (pulls terrain toward an absolute target height, the default) "
                              "or 1=raise (adds a delta, preserving existing relief -- see "
                              "terrain/stamp.py). Most useful for a masked pass that should build up "
                              "an area without flattening it. Default: use whatever's saved in "
                              "project.json, or 0 (flatten) if never set.")
    parser.add_argument("--raster-size", type=int, default=None, choices=list(RASTER_SIZES),
                         help="generate-terrain, raster method only: center-to-center spacing (m) of "
                              "the flat square-stamp grid (terrain/rastergrid.py's RASTER_SIZES) -- "
                              "each call places ONE flat grid at this size; layer coarse-to-fine "
                              "yourself by re-running generate-terrain at each size in turn. Default: "
                              f"use whatever's saved in project.json, or {DEFAULT_RASTER_SIZE} if "
                              "never set.")
    parser.add_argument("--raster-spread-ratio", type=float, default=None,
                         help="generate-terrain, raster method only: scales stamp radius (x/z scale) "
                              "independently of raster_size -- lattice centers are unaffected, only "
                              "scale_x/scale_z change (terrain/rastergrid.py's SPREAD section). 1.0 "
                              "(default) reproduces the exact edge-to-edge tiling radius; >1 grows "
                              "stamps past their own cell so neighbors overlap; <1 leaves a gap between "
                              "cells. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RASTER_SPREAD_RATIO} if never set.")
    parser.add_argument("--raster-center-bias-ratio-x", type=float, default=None,
                         help="generate-terrain, raster method only: shifts each raster stamp's "
                              "placement (not its sampled value) along x by raster_size * this ratio "
                              "-- compensates for a resolution-dependent 'drop shadow' caused by "
                              "TerrainModel's order-dependent overlap fold always favoring the +x/+z "
                              "neighbor in this lattice's own generation order (terrain/rastergrid.py's "
                              "CENTER BIAS section). No derived 'correct' value -- dial in empirically, "
                              "same as raster_spread_ratio. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_RASTER_CENTER_BIAS_RATIO} (off) if never set.")
    parser.add_argument("--raster-center-bias-ratio-z", type=float, default=None,
                         help="generate-terrain, raster method only: same as "
                              "--raster-center-bias-ratio-x but along z. Default: use whatever's saved "
                              f"in project.json, or {DEFAULT_RASTER_CENTER_BIAS_RATIO} (off) if never "
                              "set.")
    parser.add_argument("--raster-brush", type=int, default=None,
                         help="generate-terrain, raster method only: brush every lattice stamp uses. "
                              f"The grid's spacing math is derived from type {RASTER_BRUSH}'s (the "
                              "default) measured square flat-plateau/instant-edge profile, so it tiles "
                              "edge-to-edge exactly only for that brush -- any other (circular) brush "
                              "keeps the same radius but leaves each cell's corners uncovered instead "
                              "of tiling seamlessly. Default: use whatever's saved in project.json, or "
                              f"{RASTER_BRUSH} if never set.")
    parser.add_argument("--band-spacing-m", type=float, default=None,
                         help="generate-terrain, contour method only: elevation spacing (m) defining "
                              "each band -- smaller means more, narrower bands. Default: use whatever's "
                              f"saved in project.json, or {DEFAULT_BAND_SPACING_M} if never set.")
    parser.add_argument("--fill-mode", type=str, default=None, choices=["poisson", "rect"],
                         help="generate-terrain, contour method only: per-band fill algorithm. "
                              "'poisson' (default) is the two-pass circle fill (--fill-brush tiered "
                              "pack + --smoothing-brush crumb scatter). 'rect' instead traces each "
                              "band's own real boundary and places one type-72 stamp per boundary edge "
                              "(--rect-brush/--rect-tolerance-m/--rect-min-length-m/"
                              "--rect-max-search-distance-m/--rect-width-samples), falling through to "
                              "the same --smoothing-brush crumb scatter for whatever it can't reach -- "
                              "see terrain/contour_layers.py's RECT FILL MODE docstring section. Default: "
                              f"use whatever's saved in project.json, or {DEFAULT_FILL_MODE!r} if never set.")
    parser.add_argument("--rect-brush", type=int, default=None,
                         help="generate-terrain, contour rect fill_mode only: brush for the main "
                              "per-boundary-edge pass -- must be a square-shaped brush (type 72 is the "
                              "only one today). Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RECT_BRUSH} if never set.")
    parser.add_argument("--rect-tolerance-m", type=float, default=None,
                         help="generate-terrain, contour rect fill_mode only: Douglas-Peucker "
                              "boundary-simplify tolerance (m) when tracing each band's own shape into "
                              "vector polygons before placing per-edge stamps. Default: use whatever's "
                              f"saved in project.json, or {DEFAULT_RECT_TOLERANCE_M} if never set.")
    parser.add_argument("--rect-min-length-m", type=float, default=None,
                         help="generate-terrain, contour rect fill_mode only: floor length (m) for a "
                              "per-edge stamp when no opposite wall is found within "
                              "--rect-max-search-distance-m. Default: use whatever's saved in "
                              f"project.json, or {DEFAULT_RECT_MIN_LENGTH_M} if never set.")
    parser.add_argument("--rect-max-search-distance-m", type=float, default=None,
                         help="generate-terrain, contour rect fill_mode only: cap (m) on the ray-cast "
                              "search for the wall opposite each boundary edge. Default: use whatever's "
                              f"saved in project.json, or {DEFAULT_RECT_MAX_SEARCH_DISTANCE_M} if never set.")
    parser.add_argument("--rect-width-samples", type=int, default=None,
                         help="generate-terrain, contour rect fill_mode only: points sampled across each "
                              "boundary edge's own width (not just its center) when ray-casting for the "
                              "opposite wall. Default: use whatever's saved in project.json, or "
                              f"{DEFAULT_RECT_WIDTH_SAMPLES} if never set.")
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
    parser.add_argument("--generate-terrain-secondary-fill", action=argparse.BooleanOptionalAction,
                         default=None,
                         help="generate-terrain, contour method only: whether pass 2's crumb-scatter "
                              "cleanup runs after pass 1. Off trades pass 2's completeness guarantee "
                              "for speed -- whatever pass 1 leaves as crumbs stays unfilled. Default: "
                              "use whatever's saved in project.json, or ON if never set.")
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
                              "candidates_per_radius as long as each doubling reduces a sample "
                              "band's own uncovered-area fraction by at least this much, RELATIVE "
                              "to the previous doubling's own uncovered fraction (0.10 = stop once "
                              "another doubling buys less than a 10%% relative improvement). NOT an "
                              "absolute area target -- pass 1 has a genuine structural floor (real "
                              "area smaller than --min-radius, which no candidate count can ever "
                              "close), so a fixed target can be unreachable regardless of true "
                              "coverage quality. Default: use whatever's saved in project.json, or "
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
    parser.add_argument("--generate-terrain-use-height-mask", action=argparse.BooleanOptionalAction, default=None,
                         help="generate-terrain: restrict this layer to inside height_mask.geojson "
                              "(fairway/green, see ingest-osm) -- stamps whose center falls outside it "
                              "are dropped from this layer before it's written (the course-wide "
                              "baseline-flatten stamp is never masked). Default: use whatever's saved "
                              "in project.json, or off if never set.")
    parser.add_argument("--generate-terrain-mask-buffer-px", type=float, default=None,
                         help="generate-terrain: record-keeping only -- the buffer distance (m) the "
                              "current height_mask.geojson was built with, saved alongside this layer's "
                              "other parameters so it shows up in preview titles and stamp-file "
                              "metadata. Doesn't affect the mask itself (already baked into "
                              "height_mask.geojson) or any computation here.")
    parser.add_argument("--generate-terrain-remove-covered-stamps", action=argparse.BooleanOptionalAction,
                         default=None,
                         help="generate-terrain: flag stamps in the immediately-preceding stamps_N.json "
                              "layer as blocked_by this new one wherever their whole footprint sits "
                              "inside this pass's own mask (exact shapely containment, terrain/"
                              "stamp_containment.py) -- requires --generate-terrain-use-height-mask. "
                              "Default: use whatever's saved in project.json, or off if never set.")
    parser.add_argument("--generate-terrain-remove-covered-margin-m", type=float, default=None,
                         help="generate-terrain: inward safety margin (m) the mask is shrunk by before "
                              "the remove-covered-stamps containment test -- a stamp must be fully "
                              f"inside mask minus this margin to be flagged. Default: "
                              f"{DEFAULT_REMOVE_COVERED_MARGIN_M} m, or whatever's saved in project.json.")
    parser.add_argument("--n-workers", type=int, default=None,
                         help="generate-terrain, contour method only: parallelize the main per-band "
                              "loop across this many OS processes. Bands never spatially overlap, so "
                              "this is embarrassingly parallel -- output is byte-for-byte identical to "
                              "a sequential run at the same --random-seed, confirmed directly, only "
                              "faster. 1 forces sequential (e.g. for debugging). Forced to 1 regardless "
                              "of what's given whenever --max-stamps is set (that flag needs a running "
                              "total checked band-by-band, which is fundamentally sequential). Default: "
                              "use whatever's saved in project.json, or auto-detect via CPU count if "
                              "never set.")
    parser.add_argument("--splines-path", type=Path, default=None,
                         help="generate-cart-paths: path to the exported spline JSON (bezier waypoint "
                              "format) containing cart path data. Default: looks for "
                              "surfaceSplines2.json then surfaceSplines.json directly under the "
                              "working directory.")
    parser.add_argument("--cart-path-surface", type=int, default=None,
                         help="generate-cart-paths: the surface value identifying a cart path spline "
                              "in the source JSON. No safe default -- must be set (CLI or saved in "
                              "project.json from a prior run) before this step will run.")
    parser.add_argument("--cart-path-stamp-radius", type=float, default=None,
                         help="generate-cart-paths: type 15 stamp radius (m), a true center-to-edge "
                              "distance -- controls the flattened path's real width via the brush's "
                              "own measured plateau geometry. Default: use whatever's saved in "
                              f"project.json, or {CART_PATH_STAMP_RADIUS:.4f} if never set (gives "
                              "exactly a 1.7m plateau width).")
    parser.add_argument("--cart-path-spacing", type=float, default=None,
                         help="generate-cart-paths: pearl spacing (m) along each cart path -- how far "
                              "apart consecutive stamps are placed. Default: use whatever's saved in "
                              f"project.json, or {CART_PATH_SPACING_M:.4f} if never set (85%% of the "
                              "1.7m plateau width, for genuine along-path overlap).")
    parser.add_argument("--cart-path-height-avg-radius", type=float, default=None,
                         help="generate-cart-paths: radius (m) to average real heightmap data over at "
                              "each stamp. Default: use whatever's saved in project.json, or "
                              f"{CART_PATH_HEIGHT_AVG_RADIUS_M:.4f} if never set (half the stamp's own "
                              "active/nonzero footprint).")
    parser.add_argument("--stream-depth", type=float, default=None,
                         help="generate-streams: streambed carve depth (m) below original grade. "
                              "Default: use whatever's saved in project.json, or "
                              f"{STREAM_DEPTH_M} if never set.")
    parser.add_argument("--stream-half-width", type=float, default=None,
                         help="generate-streams: trench stamp radius (m, center-to-edge); the carved "
                              "channel reads about twice this wide. Also the heightmap probe radius. "
                              "Default: use whatever's saved in project.json, or "
                              f"{STREAM_HALF_WIDTH_M} if never set.")
    parser.add_argument("--stream-water-fill-depth", type=float, default=None,
                         help="generate-streams: water film depth (m) at a water tile's shallow "
                              "(upstream) end. Default: use whatever's saved in project.json, or "
                              f"{STREAM_WATER_FILL_DEPTH_M} if never set.")
    parser.add_argument("--stream-water-base-width", type=float, default=None,
                         help="generate-streams: flowing-water cross-flow span (m) at zero extra "
                              "depth. Default: use whatever's saved in project.json, or "
                              f"{STREAM_WATER_BASE_WIDTH_M} if never set.")
    parser.add_argument("--stream-water-widen-per-depth", type=float, default=None,
                         help="generate-streams: extra water width (m) per m of water depth at a "
                              "tile's deep end. Default: use whatever's saved in project.json, or "
                              f"{STREAM_WATER_WIDEN_PER_DEPTH} if never set.")
    parser.add_argument("--stream-water-widen-per-descent", type=float, default=None,
                         help="generate-streams: extra water width (m) per m of total bed descent "
                              "from the stream source. Default: use whatever's saved in project.json, "
                              f"or {STREAM_WATER_WIDEN_PER_DESCENT} if never set.")
    parser.add_argument("--stream-water-level-margin", type=float, default=None,
                         help="generate-streams: how far (m) below the fitted carved streambed the "
                              "flowing-water surface sits (write-water/write-objects fit each tile's "
                              "level to the real terrain, like a pond, then subtract this). Default: "
                              f"use whatever's saved in project.json, or {STREAM_WATER_LEVEL_MARGIN_M} "
                              "if never set.")
    parser.add_argument("--stream-bank-veg-width", type=float, default=None,
                         help="generate-streams: half-width (m) of the stream-bank vegetation buffer "
                              "polygon tagged for cluster fills. Default: use whatever's saved in "
                              f"project.json, or {BANK_VEG_WIDTH_M} if never set.")
    parser.add_argument("--oob-inner-buffer", type=float, default=None,
                         help="generate-oob: gap (m) between the playable area and where the OOB band "
                              f"starts. Default: project.json, or {OOB_INNER_BUFFER_M}.")
    parser.add_argument("--oob-band-width", type=float, default=None,
                         help="generate-oob: painted width (m) of the OOB band across the boundary. "
                              f"Default: project.json, or {OOB_BAND_WIDTH_M}.")
    parser.add_argument("--oob-merge-gap", type=float, default=None,
                         help="generate-oob: morphological-closing radius (m) -- bridges gaps up to "
                              "~2x this between playfield pieces so the OOB line follows the outer "
                              f"course boundary, not every fragment. 0 disables. Default: "
                              f"project.json, or {OOB_MERGE_GAP_M}.")
    parser.add_argument("--oob-simplify-tol", type=float, default=None,
                         help="generate-oob: Douglas-Peucker tolerance (m) on the boundary curve "
                              "before it's walked -- higher = fewer, longer square stamps, coarser "
                              f"OOB line. Default: project.json, or {OOB_SIMPLIFY_TOL_M}.")
    parser.add_argument("--oob-cap-ratio", type=float, default=None,
                         help="generate-oob: round-cap (type 8) half-extent as a fraction of the "
                              f"square's across-path half-extent. Default: project.json, or "
                              f"{OOB_CAP_SCALE_RATIO}.")
    parser.add_argument("--oob-no-caps", action="store_true",
                         help="generate-oob: skip the round type-8 caps at boundary vertices "
                              "(segments only).")
    parser.add_argument("--oob-clear", action="store_true",
                         help="generate-oob: delete the generated OOB band (oob.json + previews) "
                              "and mark the layer empty instead of regenerating -- the next "
                              "write-terrain then writes an empty outOfBounds array.")
    parser.add_argument("--parking-spacing", type=float, default=None,
                         help="generate-parking: along-row centre-to-centre car spacing (m). "
                              f"Default: project.json, or {PARKING_SPACING_M}.")
    parser.add_argument("--parking-offset", type=float, default=None,
                         help="generate-parking: perpendicular distance (m) from the aisle line to "
                              f"the car row. Default: project.json, or {PARKING_OFFSET_M}.")
    parser.add_argument("--parking-sides", type=str, default=None,
                         choices=["both", "left", "right"],
                         help="generate-parking: which side(s) of the line to fill, relative to its "
                              f"direction. Default: project.json, or {PARKING_SIDES}.")
    parser.add_argument("--parking-orientation", type=str, default=None,
                         choices=["perpendicular", "parallel"],
                         help="generate-parking: 'perpendicular' nose-in stalls or 'parallel' "
                              f"kerbside parking. Default: project.json, or {PARKING_ORIENTATION}.")
    parser.add_argument("--parking-skip-prob", type=float, default=None,
                         help="generate-parking: fraction of stalls left empty (0-1). "
                              f"Default: project.json, or {PARKING_SKIP_PROB}.")
    parser.add_argument("--parking-max-variants", type=int, default=None,
                         help="generate-parking: cap on the number of DISTINCT car prefabs placed "
                              "across the whole course (keeps parked cars off the placed-object type "
                              f"budget). 0 = no cap. Default: project.json, or {PARKING_MAX_VARIANTS}.")
    parser.add_argument("--parking-color-weights", type=str, default=None,
                         help="generate-parking: car colour mix as 'colour=weight,...' (weights "
                              "normalised; percentages fine), e.g. 'black=40,gray=15,white=15'. A "
                              "colour not listed (and not an accent) never spawns. Default: "
                              f"project.json, or {PARKING_COLOR_WEIGHTS}.")
    parser.add_argument("--parking-accent-count", type=int, default=None,
                         help="generate-parking: how many accent colours (from red/green/blue/yellow) "
                              "to add to the mix, chosen once per course at 0.10 weight each. "
                              f"Default: project.json, or {PARKING_ACCENT_COUNT}.")
    parser.add_argument("--parking-seed", type=int, default=None,
                         help="generate-parking: RNG seed (car choice / empty stalls / jitter). "
                              "Default: a fixed built-in seed.")
    parser.add_argument("--no-parking-bank", action="store_true",
                         help="generate-parking: keep cars flat instead of banking pitch/roll to "
                              "the ground slope.")
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
    parser.add_argument("--remove-covered-stamps", action=argparse.BooleanOptionalAction, default=None,
                         help="refine-terrain: flag stamps in the immediately-preceding stamps_N.json "
                              "layer as blocked_by this new one wherever their whole footprint sits "
                              "inside this pass's own mask (exact shapely containment, terrain/"
                              "stamp_containment.py) -- requires --use-height-mask. Default: use "
                              "whatever's saved in project.json, or off if never set.")
    parser.add_argument("--remove-covered-margin-m", type=float, default=None,
                         help="refine-terrain: inward safety margin (m) the mask is shrunk by before "
                              "the remove-covered-stamps containment test -- a stamp must be fully "
                              f"inside mask minus this margin to be flagged. Default: "
                              f"{DEFAULT_REMOVE_COVERED_MARGIN_M} m, or whatever's saved in project.json.")
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
                         help="write-terrain/write-water/write-splines: add a small type-73 (circle) "
                              "raise stamp and a matching 5m circle spline (cart path surface) at "
                              "each of the 4 course corners (5m inset from each edge) -- for visually "
                              "confirming in-game that terrain and splines land exactly where "
                              "expected, and that the game isn't scaling/repositioning either one "
                              "unexpectedly. Opt-in; off by default.")
    parser.add_argument("--direct-height-shift", action=argparse.BooleanOptionalAction, default=True,
                         help="write-terrain/write-water: normalize final heights by shifting every "
                              "stamp's own value directly (normalize_stamp_heights_by_value_shift) "
                              "instead of appending a course-wide shim stamp (normalize_stamp_heights, "
                              "--no-direct-height-shift) -- not a mathematically equivalent shift (see "
                              "that function's docstring). Default: on.")
    parser.add_argument("--multi-tile-water", action="store_true",
                         help="write-water: fill each water hazard polygon with several smaller, "
                              "possibly-overlapping rectangular tiles hugging its real boundary, "
                              "instead of one single minimum-rotated-rectangle -- reduces (does not "
                              "guarantee zero) visible water-over-land overshoot on non-rectangular "
                              "ponds. Overlap between tiles is fine (renders with a clean seam in-"
                              "game); only overshoot past the polygon boundary is minimized. Opt-in; "
                              "off by default -- output is byte-identical to the single-rectangle path "
                              "when unset. See course_output/water.py's fit_water_tiles.")
    parser.add_argument("--water-tile-tolerance-m", type=float, default=DEFAULT_WATER_TILE_TOLERANCE_M,
                         help="write-water, --multi-tile-water only: Douglas-Peucker boundary-simplify "
                              "tolerance (m) applied to a pond polygon before per-edge tile placement "
                              f"-- collapses small boundary wiggles into fewer, longer edges. Default: "
                              f"{DEFAULT_WATER_TILE_TOLERANCE_M}.")
    parser.add_argument("--water-tile-min-edge-m", type=float, default=DEFAULT_WATER_TILE_MIN_EDGE_M,
                         help="write-water, --multi-tile-water only: floor depth (m) for a per-edge "
                              "tile when no opposite wall is found within --water-tile-max-search-m. "
                              f"Default: {DEFAULT_WATER_TILE_MIN_EDGE_M}.")
    parser.add_argument("--water-tile-max-search-m", type=float, default=DEFAULT_WATER_TILE_MAX_SEARCH_M,
                         help="write-water, --multi-tile-water only: cap (m) on the ray-cast search for "
                              f"the wall opposite each boundary edge. Default: {DEFAULT_WATER_TILE_MAX_SEARCH_M}.")
    parser.add_argument("--water-tile-width-samples", type=int, default=DEFAULT_WATER_TILE_WIDTH_SAMPLES,
                         help="write-water, --multi-tile-water only: points sampled across each "
                              "boundary edge's own width (not just its center) when ray-casting for "
                              f"the opposite wall. Default: {DEFAULT_WATER_TILE_WIDTH_SAMPLES}.")
    parser.add_argument("--water-tile-redundancy-ratio", type=float, default=DEFAULT_WATER_TILE_REDUNDANCY_RATIO,
                         help="write-water, --multi-tile-water only: a candidate tile contributing "
                              "less than this fraction of its own area as genuinely new coverage (not "
                              "already covered by tiles already placed) is dropped as redundant. Lower "
                              f"= more, more-overlapping tiles kept. Default: {DEFAULT_WATER_TILE_REDUNDANCY_RATIO}.")
    parser.add_argument("--water-tile-overlap-m", type=float, default=DEFAULT_WATER_TILE_OVERLAP_M,
                         help="write-water, --multi-tile-water only: extend each per-edge tile's width "
                              "by this much (m) at BOTH ends, centered the same as before -- without "
                              "it, adjacent tiles meet exactly corner-to-corner and can look visibly "
                              f"inset once real rounding is involved. Default: {DEFAULT_WATER_TILE_OVERLAP_M}.")
    parser.add_argument("--water-fill-mode", choices=["edge", "stripe"], default="edge",
                         help="write-water, --multi-tile-water only: 'edge' (default) fits one "
                              "rectangle per simplified boundary edge with a greedy redundancy-dedup "
                              "pass (course_output/water.py's fit_water_tiles); 'stripe' seeds from the "
                              "outer fitted rectangle's center and walks outward in independently-sized "
                              "stripes (fit_water_stripes) -- avoids edge-fill's dedup pass "
                              "unpredictably leaving gaps on complex pond shapes, at the cost of "
                              "possibly more overshoot on very irregular boundaries. Ignored unless "
                              "--multi-tile-water is set.")
    parser.add_argument("--water-stripe-overlap-m", type=float, default=DEFAULT_WATER_STRIPE_OVERLAP_M,
                         help="write-water, --water-fill-mode stripe only: the single knob controlling "
                              "both how far a stripe may overshoot the true polygon boundary and how "
                              f"much consecutive stripes overlap along the stacking axis. Default: "
                              f"{DEFAULT_WATER_STRIPE_OVERLAP_M}.")
    parser.add_argument("--water-stripe-min-edge-m", type=float, default=DEFAULT_WATER_TILE_MIN_EDGE_M,
                         help="write-water, --water-fill-mode stripe only: floor (m) for a stripe's "
                              f"found width/depth. Default: {DEFAULT_WATER_TILE_MIN_EDGE_M}.")
    parser.add_argument("--water-stripe-tolerance-m", type=float, default=DEFAULT_WATER_STRIPE_TOLERANCE_M,
                         help="write-water, --water-fill-mode stripe only: optional boundary-simplify "
                              "(m) before probing -- perf/noise-reduction only, not structural (unlike "
                              f"--water-tile-tolerance-m). Default: {DEFAULT_WATER_STRIPE_TOLERANCE_M}.")
    parser.add_argument("--water-stripe-max-stripes-per-side", type=int,
                         default=DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
                         help="write-water, --water-fill-mode stripe only: safety cap on stripes walked "
                              "outward in each of the +/- stacking directions. Default: "
                              f"{DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE}.")
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
    parser.add_argument("--no-preserve-synthetic", action="store_true",
                         help="ingest-osm: rebuild features.geojson purely from map.osm, discarding "
                              "GUI-authored cluster-fill border rings / 'Use mask' fills / per-feature "
                              "mask & cluster-fill edits and generate-streams bank vegetation (the "
                              "pre-existing behavior). Default: preserve them across the re-parse.")
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
    parser.add_argument("--course-theme", type=str, default=None,
                         help="ingest-course: theme name for template resolution (e.g. 'rustic') -- "
                              "saved to project.json's \"theme\" field. Optional: only needed on the "
                              "first --step ingest-course for a project, or to change theme afterward.")
    parser.add_argument("--blank-course-name", type=str, default=None,
                         help="push-blank-template / push-collection: in-game name to stamp into the "
                              "built .course (its course id is also regenerated so the game sees a new "
                              "course). Optional -- defaults to a generated serial "
                              "('LIDAR-<version>-<timestamp>' / 'COLL-<name>-<timestamp>').")
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
                              "Ignored when a tree_themes.json species table covers the theme (the "
                              "buckets are the pools). Default: saved value, or ON if never set.")
    parser.add_argument("--tree-theme-config", type=str, default=None, metavar="PATH",
                         help="write-objects (game_version=2019 only): override the default "
                              "course_output/tree_themes.json (per-theme species buckets that route "
                              "trees by their pga_tree_type tag -- e.g. leaf_type=needleleaved -> the "
                              "theme's 'pine' ids). Default: saved value, else the bundled file.")
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
                         help="write-objects (game_version=2021+ only): Unity asset path for the stake "
                              "placed at every corner of every 'building' feature, overriding the "
                              "default objects.DEFAULT_STAKE_ASSET_PATH_V2021. Setting this also "
                              "implies --stake-buildings. Default: use whatever's saved in "
                              "project.json, or the built-in default if never set.")
    parser.add_argument("--stake-buildings", action=argparse.BooleanOptionalAction, default=None,
                         help="write-objects: place a stake at every corner of every 'building' "
                              "feature. v2019 uses the fence-post prop of the cart-path debug marker "
                              "(category=objects.BUILDING_STAKE_CATEGORY_V2019/type="
                              "objects.BUILDING_STAKE_TYPE_V2019); v2021+ uses "
                              "objects.DEFAULT_STAKE_ASSET_PATH_V2021 (or --stake-asset-path). "
                              "Both at a subtle objects.BUILDING_STAKE_SCALE_V2019 scale. "
                              "Default: use whatever's saved in project.json, or OFF if never set.")
    parser.add_argument("--waterfall-asset-path", type=str, default=None,
                         help="write-objects (game_version=2021+ only): Unity asset path for the "
                              "waterfall prefab placed at each stream drop (see --step generate-streams "
                              "/ streams.json). Default: use whatever's saved in project.json, or "
                              f"'{WATERFALL_DEFAULT_ASSET_PATH}' if never set.")
    parser.add_argument("--splash-asset-path", type=str, default=None,
                         help="write-objects (game_version=2021+ only): Unity asset path for the "
                              "'low splash' prefab placed on the tile each stream drop lands in. "
                              "Default: use whatever's saved in project.json, or "
                              f"'{WATERSPLASH_DEFAULT_ASSET_PATH}' (unverified) if never set.")
    parser.add_argument("--collection-library", type=Path, default=None,
                         help="generate-collections / push-collection: directory of *.json collection "
                              "templates (see course_output/collection_library.py). Default: "
                              "project.json's 'collections_library_dir', or ~/.pga2k/collections/.")
    parser.add_argument("--collection-name", type=str, default=None,
                         help="push-collection: name of the library template to build into an editable "
                              ".course (working_dir/pushed_collection.course). Required for that step.")
    parser.add_argument("--detect-lidar-trees", action=argparse.BooleanOptionalAction, default=None,
                         help="generate-trees: also detect individual trees directly from LIDAR canopy "
                              "points (ingest/tree_detection.py), added on top of any OSM natural=tree "
                              "nodes. Confined to height_mask.geojson's core-play-area polygon if one "
                              "exists (the game's own procedural vegetation fill is expected to handle "
                              "everywhere else). Needs heightmap.npz and pointcloud.npz (--step "
                              "ingest-laz). Default: use whatever's saved in project.json, or ON if "
                              "never set (OSM alone typically finds few or no trees on a real course).")
    parser.add_argument("--mark-cartpath-trees", action=argparse.BooleanOptionalAction, default=None,
                         help="generate-trees: DEBUG mode -- trees detected sitting on a cart path are "
                              "left in place (not relocated) and tagged so write-objects "
                              "(game_version=2019 only) swaps them for an oversized, obvious marker "
                              "object (see objects.py's CARTPATH_DEBUG_MARKER_CATEGORY_V2019/TYPE/SCALE) "
                              "instead of a real tree, so you can see in-game exactly which trees are "
                              "being flagged. NOT sticky -- always OFF by default even if a previous run "
                              "used it; pass this flag explicitly every time you want it.")
    parser.add_argument("--repack-filename", type=str, default=None,
                         help="repack: output filename (without .course extension)")
    parser.add_argument("--edited-course", type=Path, default=None,
                         help="import-ingame-edits: path to a saved .course file (exported by this "
                              "tool, then hand-edited in PGA Tour 2K's own in-game editor and saved) "
                              "to reconcile against what this tool currently tracks. Required for "
                              "this step.")
    parser.add_argument("--commit", action="store_true",
                         help="import-ingame-edits: apply the diff (append new objects to "
                              "ingame_objects.json, write a new stamps_N.json layer for new terrain "
                              "stamps) instead of just printing a summary (the default, dry-run).")
    parser.add_argument("--import-group", type=str, default=None,
                         help="import-ingame-edits, --commit only: label stored on every new "
                              "ingame_objects.json record from this run (see the Objects tab / "
                              "course_output/ingame_objects.py). Default: 'Imported <today's date>'.")
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
            step_ingest_osm(working_dir, args.height_mask_buffer_px, args.hole_corridor_buffer_px,
                            preserve_synthetic=not args.no_preserve_synthetic)
        elif args.step == "ingest-course":
            step_ingest_course(working_dir, args.course_theme)
        elif args.step == "push-blank-template":
            step_push_blank_template(working_dir, args.blank_course_name)
        elif args.step == "dig-water":
            step_dig_water(working_dir, args.dig_depth, args.dig_buffer)
        elif args.step == "generate-terrain":
            step_generate_terrain(
                working_dir,
                pitch=args.pitch,
                hex_spread_ratio=args.hex_spread_ratio,
                method=args.generate_terrain_method,
                hex_brush=args.hex_brush,
                hex_tool=args.hex_tool,
                raster_size=args.raster_size,
                raster_spread_ratio=args.raster_spread_ratio,
                raster_center_bias_ratio_x=args.raster_center_bias_ratio_x,
                raster_center_bias_ratio_z=args.raster_center_bias_ratio_z,
                raster_brush=args.raster_brush,
                band_spacing_m=args.band_spacing_m,
                fill_mode=args.fill_mode,
                fill_brush=args.fill_brush,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                radius_step_ratio=args.radius_step_ratio,
                edge_distance_m=args.edge_distance_m,
                rect_brush=args.rect_brush,
                rect_tolerance_m=args.rect_tolerance_m,
                rect_min_length_m=args.rect_min_length_m,
                rect_max_search_distance_m=args.rect_max_search_distance_m,
                rect_width_samples=args.rect_width_samples,
                smoothing_brush=args.smoothing_brush,
                smoothing_min_radius=args.smoothing_min_radius,
                smooth_ratio=args.smooth_ratio,
                smooth_claim_fraction=args.smooth_claim_fraction,
                enable_secondary_fill=args.generate_terrain_secondary_fill,
                candidates_per_radius=args.candidates_per_radius,
                sweet_spot_ratio=args.sweet_spot_ratio,
                sweet_spot_sample_bands=args.sweet_spot_sample_bands,
                sweet_spot_seeds=args.sweet_spot_seeds,
                sweet_spot_max_candidates=args.sweet_spot_max_candidates,
                sweet_spot_time_budget_s=args.sweet_spot_time_budget_s,
                random_seed=args.random_seed,
                denoise_px=args.denoise_px,
                max_stamps=args.max_stamps,
                use_height_mask=args.generate_terrain_use_height_mask,
                mask_buffer_px=args.generate_terrain_mask_buffer_px,
                remove_covered_stamps=args.generate_terrain_remove_covered_stamps,
                remove_covered_margin_m=args.generate_terrain_remove_covered_margin_m,
                n_workers=args.n_workers,
            )
        elif args.step == "generate-cart-paths":
            step_generate_cart_paths(
                working_dir,
                splines_path=args.splines_path,
                cart_path_surface=args.cart_path_surface,
                stamp_radius=args.cart_path_stamp_radius,
                spacing_m=args.cart_path_spacing,
                height_avg_radius_m=args.cart_path_height_avg_radius,
            )
        elif args.step == "generate-streams":
            step_generate_streams(
                working_dir,
                depth_m=args.stream_depth,
                half_width_m=args.stream_half_width,
                water_fill_depth_m=args.stream_water_fill_depth,
                water_base_width_m=args.stream_water_base_width,
                water_widen_per_depth=args.stream_water_widen_per_depth,
                water_widen_per_descent=args.stream_water_widen_per_descent,
                water_level_margin_m=args.stream_water_level_margin,
                bank_veg_width_m=args.stream_bank_veg_width,
            )
        elif args.step == "generate-oob":
            step_generate_oob(
                working_dir,
                inner_buffer_m=args.oob_inner_buffer,
                band_width_m=args.oob_band_width,
                merge_gap_m=args.oob_merge_gap,
                simplify_tol_m=args.oob_simplify_tol,
                cap_scale_ratio=args.oob_cap_ratio,
                include_caps=(False if args.oob_no_caps else None),
                clear=args.oob_clear,
            )
        elif args.step == "generate-collections":
            step_generate_collections(working_dir, args.collection_library)
        elif args.step == "generate-parking":
            step_generate_parking(
                working_dir,
                spacing_m=args.parking_spacing,
                offset_m=args.parking_offset,
                sides=args.parking_sides,
                orientation=args.parking_orientation,
                skip_prob=args.parking_skip_prob,
                max_variants=args.parking_max_variants,
                color_weights=args.parking_color_weights,
                accent_count=args.parking_accent_count,
                seed=args.parking_seed,
                no_bank=args.no_parking_bank,
            )
        elif args.step == "push-collection":
            if not args.collection_name:
                print("error: --step push-collection requires --collection-name", file=sys.stderr)
                return 1
            step_push_collection(
                working_dir, args.collection_name, args.collection_library, args.blank_course_name,
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
                                 args.remove_covered_stamps, args.remove_covered_margin_m,
                                 args.model_rebuild_interval, parsed_candidate_brushes,
                                 args.max_planar_rms, args.planar_shrink_factor, args.rad_m,
                                 args.use_slope_radius, args.use_variation_radius,
                                 args.variation_contrast_gamma, args.density_weighted,
                                 args.subpixel_jitter_fraction)
        elif args.step == "write-terrain":
            step_write_terrain(working_dir, registration_marks=args.registration_marks,
                                direct_height_shift=args.direct_height_shift)
        elif args.step == "write-water":
            step_write_water(working_dir, registration_marks=args.registration_marks,
                              direct_height_shift=args.direct_height_shift,
                              multi_tile_water=args.multi_tile_water,
                              water_tile_tolerance_m=args.water_tile_tolerance_m,
                              water_tile_min_edge_m=args.water_tile_min_edge_m,
                              water_tile_max_search_m=args.water_tile_max_search_m,
                              water_tile_width_samples=args.water_tile_width_samples,
                              water_tile_redundancy_ratio=args.water_tile_redundancy_ratio,
                              water_tile_overlap_m=args.water_tile_overlap_m,
                              water_fill_mode=args.water_fill_mode,
                              water_stripe_overlap_m=args.water_stripe_overlap_m,
                              water_stripe_min_edge_m=args.water_stripe_min_edge_m,
                              water_stripe_tolerance_m=args.water_stripe_tolerance_m,
                              water_stripe_max_stripes_per_side=args.water_stripe_max_stripes_per_side)
        elif args.step == "write-splines":
            step_write_splines(working_dir, registration_marks=args.registration_marks)
        elif args.step == "write-holes":
            step_write_holes(working_dir)
        elif args.step == "generate-trees":
            step_generate_trees(working_dir, args.detect_lidar_trees, args.mark_cartpath_trees)
        elif args.step == "pack-objects":
            step_pack_objects(working_dir)
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
                args.stake_buildings, args.waterfall_asset_path, args.splash_asset_path,
                args.tree_theme_config,
            )
        elif args.step == "repack":
            if not args.repack_filename:
                print("error: --step repack requires --repack-filename <name>", file=sys.stderr)
                return 1
            step_repack(working_dir, args.repack_filename)
        elif args.step == "import-ingame-edits":
            if not args.edited_course:
                print("error: --step import-ingame-edits requires --edited-course <path>", file=sys.stderr)
                return 1
            step_import_ingame_edits(
                working_dir, args.edited_course, game_version=args.game_version, commit=args.commit,
                group_name=args.import_group, registration_marks=args.registration_marks,
                direct_height_shift=args.direct_height_shift,
            )
        elif args.step == "visualize":
            step_visualize(working_dir, overwrite_current_version=True, error_resolution=args.error_resolution)
    except StepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
