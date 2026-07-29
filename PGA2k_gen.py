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
from pathlib import Path

import pyproj

SCRIPT_DIR = Path(__file__).resolve().parent

from constants import (
    COURSE_SIZE_M, PREVIEW_ERROR, PREVIEW_HEIGHT, PREVIEW_HEX,
    PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_STAMPS,
    POINTCLOUD_FILE, PROJECT_FILE,
)
import visualize as viz
from ingest.laz_reader import LazReadError, PointCloud, load_point_cloud, recentered_crop
from osm import parse_osm_features, save_features
from terrain.adaptive_refine import DEFAULT_MIN_HOTSPOT_RADIUS_CELLS, DEFAULT_RESOLUTION, refine_stamps
from terrain.bounding_box import BoundingBox
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import generate_hex_grid
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel
from writer import normalize_stamp_heights, write_user_layers

INITIAL_STAMPS_FILE = "initial_stamps.json"
FEATURES_FILE = "features.geojson"
REFINE_STAMPS_PATTERN = "refine_stamps_{n}.json"


def _refine_stamps_files(working_dir: Path) -> list[Path]:
    """Every refine_stamps_N.json present, in order (N=1, 2, 3, ...)."""
    files = []
    n = 1
    while (working_dir / REFINE_STAMPS_PATTERN.format(n=n)).exists():
        files.append(working_dir / REFINE_STAMPS_PATTERN.format(n=n))
        n += 1
    return files


def load_all_stamps(working_dir: Path) -> list[Stamp]:
    """
    Reconstruct the full, current stamp list: initial_stamps.json plus
    every refine_stamps_N.json in order.

    Each refine-terrain pass writes only the stamps *it* added, not a
    cumulative snapshot -- so deleting the highest-numbered
    refine_stamps_N.json is a natural undo of just the most recent
    pass, and every earlier pass's file stays exactly as it was
    (nothing gets rewritten/renumbered by later passes).
    """
    initial_path = working_dir / INITIAL_STAMPS_FILE
    if not initial_path.exists():
        raise StepError(
            f"No {INITIAL_STAMPS_FILE} found under {working_dir}. Run --step generate-terrain first."
        )

    stamps = load_stamps(initial_path)
    for path in _refine_stamps_files(working_dir):
        stamps.extend(load_stamps(path))
    return stamps


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
# userLayers.json -- this is our own working representation, not PGA's)
# ---------------------------------------------------------------------------

def save_stamps(stamps: list[Stamp], path: Path) -> None:
    with path.open("w") as f:
        json.dump([dataclasses.asdict(s) for s in stamps], f, indent=2)


def load_stamps(path: Path) -> list[Stamp]:
    with path.open() as f:
        raw = json.load(f)
    return [Stamp(**entry) for entry in raw]


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

    full_cloud = PointCloud.load(pointcloud_path)
    print(f"Loaded {pointcloud_path} ({full_cloud.count:,} points)")

    lidar_preview_paths = [working_dir / PREVIEW_LIDAR, working_dir / PREVIEW_LIDAR_HEIGHTMAP]
    pointcloud_mtime = pointcloud_path.stat().st_mtime
    lidar_previews_stale = any(
        not p.exists() or p.stat().st_mtime < pointcloud_mtime for p in lidar_preview_paths
    )

    if lidar_previews_stale:
        print(f"Writing {PREVIEW_LIDAR} and {PREVIEW_LIDAR_HEIGHTMAP} "
              "(full merged point cloud, not just the course crop)...")
        viz.render_lidar_preview(full_cloud, working_dir / PREVIEW_LIDAR)
        viz.render_lidar_heightmap(full_cloud, full_cloud.bounds, working_dir / PREVIEW_LIDAR_HEIGHTMAP)
    else:
        print(f"{PREVIEW_LIDAR} / {PREVIEW_LIDAR_HEIGHTMAP} already up to date with "
              f"{POINTCLOUD_FILE} -- skipping (re-run --step ingest-laz to force a refresh)")

    if not (working_dir / INITIAL_STAMPS_FILE).exists():
        print(f"No {INITIAL_STAMPS_FILE} yet -- run --step generate-terrain for the "
              "hex/stamps/height/error previews. Stopping after the LIDAR previews.")
        return

    stamps = load_all_stamps(working_dir)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
    model = TerrainModel(stamps)

    print(f"Writing {PREVIEW_HEX}...")
    viz.render_hex_preview(stamps, bounds, working_dir / PREVIEW_HEX)
    print(f"Writing {PREVIEW_STAMPS}...")
    viz.render_stamps_preview(stamps, bounds, working_dir / PREVIEW_STAMPS)
    print(f"Writing {PREVIEW_HEIGHT}...")
    viz.render_height_preview(model, bounds, working_dir / PREVIEW_HEIGHT)

    print(f"Writing {PREVIEW_ERROR} (course-cropped point cloud vs. TerrainModel)...")
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    viz.render_error_preview(model, course_cloud, bounds, working_dir / PREVIEW_ERROR)

    print(f"All previews written to {working_dir}")


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


def step_ingest_osm(working_dir: Path) -> None:
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

    save_project(working_dir, {"osm_feature_count": len(features), "osm_feature_kinds": counts})


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

    print("Generating hex grid...")
    stamps = generate_hex_grid(bounds)
    print(f"  {len(stamps)} stamps placed")

    print("Fitting stamp heights from nearby LIDAR points...")
    fitted = fit_stamp_heights(stamps, course_cloud)
    n_unfitted = sum(1 for s in fitted if s.value == 0.0)
    if n_unfitted:
        print(f"  WARNING: {n_unfitted} stamps had too few nearby points and kept "
              "their placeholder value=0.0")

    out_path = working_dir / INITIAL_STAMPS_FILE
    save_stamps(fitted, out_path)
    print(f"  wrote {out_path}")

    save_project(working_dir, {
        "course_origin_x": course_cloud.origin_x,
        "course_origin_y": course_cloud.origin_y,
        "stamp_count": len(fitted),
    })


def step_refine_terrain(
    working_dir: Path,
    tolerance: float,
    resolution: int,
    min_hotspot_radius_cells: float,
    max_new_stamps: int | None,
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
    """
    pointcloud_path = working_dir / POINTCLOUD_FILE
    if not pointcloud_path.exists():
        raise StepError(
            f"No {POINTCLOUD_FILE} found under {working_dir}. Run --step ingest-laz first."
        )

    stamps = load_all_stamps(working_dir)
    print(f"  {len(stamps)} stamps (cumulative: initial + {len(_refine_stamps_files(working_dir))} "
          "prior refine pass(es))")

    full_cloud = PointCloud.load(pointcloud_path)
    course_cloud = recentered_crop(full_cloud, size_m=COURSE_SIZE_M)
    bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)

    print(f"Scanning the error grid ({resolution}x{resolution}, tolerance={tolerance} m)...")
    refined, hotspots = refine_stamps(
        stamps, course_cloud, bounds, tolerance=tolerance,
        resolution=resolution, min_hotspot_radius_cells=min_hotspot_radius_cells,
        max_new_stamps=max_new_stamps,
    )

    new_stamps = refined[len(stamps):]
    if hotspots:
        worst = hotspots[0]
        print(f"  {len(hotspots)} hotspots over tolerance (worst: {worst.peak_error:.3f} m "
              f"at ({worst.x:.1f}, {worst.z:.1f}), {worst.n_cells} cells)")
    else:
        print("  0 hotspots over tolerance -- nothing to refine, terrain already meets it")

    if new_stamps:
        next_n = len(_refine_stamps_files(working_dir)) + 1
        out_path = working_dir / REFINE_STAMPS_PATTERN.format(n=next_n)
        save_stamps(new_stamps, out_path)
        print(f"  wrote {out_path} ({len(new_stamps)} new stamps; "
              f"{len(stamps)} -> {len(refined)} total)")
    else:
        print(f"  nothing written ({len(stamps)} total, unchanged)")

    save_project(working_dir, {
        "last_refine_tolerance_m": tolerance,
        "last_refine_hotspot_count": len(hotspots),
        "last_refine_added_count": len(new_stamps),
        "total_stamp_count": len(refined),
    })


def step_output_terrain(working_dir: Path) -> None:
    course_dir = working_dir / "course"

    print(f"Loading stamps from {working_dir} (initial + all refine passes)...")
    stamps = load_all_stamps(working_dir)
    print(f"  {len(stamps)} stamps")

    min_value = min(s.value for s in stamps)
    max_value = max(s.value for s in stamps)
    print(f"Normalizing heights: raw range [{min_value:.3f}, {max_value:.3f}] m "
          f"-> shifting by {-min_value:.3f} m so minimum lands at 0")
    try:
        stamps = normalize_stamp_heights(stamps)
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

    # If a course name has been set (see the GUI's "Course name" field / project.json),
    # write it into CourseDescription.json's root-level "name" key, preserving every
    # other key already there -- same preserve-everything-else pattern as
    # write_user_layers uses for userLayers.json's sibling keys.
    project = load_project(working_dir)
    course_name = project.get("course_name")
    course_desc_path = course_dir / "CourseDescription.json"
    if course_name:
        if course_desc_path.exists():
            with course_desc_path.open(encoding="utf-8") as f:
                desc = json.load(f)
            desc["name"] = course_name
            with course_desc_path.open("w", encoding="utf-8") as f:
                json.dump(desc, f, indent=2)
            print(f"Set course name to '{course_name}' in {course_desc_path}")
        else:
            print(f"NOTE: course_name is set ('{course_name}') but {course_desc_path} "
                  "doesn't exist yet -- run --step ingest-course first if you want the "
                  "name applied.")

    save_project(working_dir, {
        "output_height_shift_m": -min_value,
        "output_height_range_m": max_value - min_value,
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
            step_ingest_osm(working_dir)
        elif args.step == "ingest-course":
            if args.course_file is None:
                print("error: --step ingest-course requires --course-file <path>", file=sys.stderr)
                return 1
            step_ingest_course(working_dir, args.course_file)
        elif args.step == "generate-terrain":
            step_generate_terrain(working_dir)
        elif args.step == "refine-terrain":
            step_refine_terrain(working_dir, args.error_tolerance, args.resolution,
                                 args.min_hotspot_radius_cells, args.max_new_stamps)
        elif args.step == "output-terrain":
            step_output_terrain(working_dir)
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
