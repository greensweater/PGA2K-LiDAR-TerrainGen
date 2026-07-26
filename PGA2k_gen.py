#!/usr/bin/env python3
"""
PGA2k_gen.py

CLI orchestrator for the terrain compiler. Operates on a working
directory, running one pipeline step at a time:

    PGA2k_gen.py <working_dir>                       (same as --step init)
    PGA2k_gen.py <working_dir> --step init
    PGA2k_gen.py <working_dir> --step ingest-laz --projection <EPSG>
    PGA2k_gen.py <working_dir> --step ingest-osm
    PGA2k_gen.py <working_dir> --step generate-terrain
    PGA2k_gen.py <working_dir> --step output-terrain [--course-dir DIR]

Each step reads/writes plain-file artifacts in <working_dir> instead of
holding state in memory across invocations -- this is a CLI today, a
GUI eventually (per the architecture doc), so every step needs to be
independently resumable and inspectable, never a black box.

<working_dir> layout:
    laz/                    input LAZ/LAS tiles
    map.osm                 input OSM export (user-downloaded, using
                             the lat/lon bbox ingest-laz prints)
    project.json             small state manifest (projection, merged
                             bounds, course origin) carried between
                             steps so they don't need re-specifying
    pointcloud.npz            ingest-laz output (ingest.laz_reader.PointCloud)
    initial_stamps.json      generate-terrain output (Stamp list)
    course/                  extracted blank .course to write into
                             (see output-terrain); override with
                             --course-dir

Step ordering is enforced with clear errors (e.g. generate-terrain
without a pointcloud.npz on disk yet) rather than letting a later step
fail on a confusing missing-file exception.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pyproj

from constants import COURSE_SIZE_M, POINTCLOUD_FILE, PROJECT_FILE
from ingest.laz_reader import LazReadError, PointCloud, load_point_cloud, recentered_crop
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import generate_hex_grid
from terrain.stamp import Stamp
from writer import write_user_layers

INITIAL_STAMPS_FILE = "initial_stamps.json"
OPTIMIZED_STAMPS_FILE = "optimized_stamps.json"  # not produced yet; see output-terrain


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


def step_ingest_laz(working_dir: Path, projection: int) -> None:
    laz_dir = working_dir / "laz"
    if not laz_dir.is_dir():
        raise StepError(f"No laz/ folder found under {working_dir} -- expected {laz_dir}")

    try:
        force_crs = pyproj.CRS.from_epsg(projection)
    except pyproj.exceptions.CRSError as e:
        raise StepError(f"--projection {projection} is not a valid EPSG code: {e}") from e

    print(f"Reading LAZ tiles from {laz_dir} (forcing CRS EPSG:{projection})...")
    cloud = load_point_cloud(laz_dir, force_crs=force_crs)
    print(f"  {cloud.count} points loaded, local bounds {cloud.bounds}")

    cloud.save(working_dir / POINTCLOUD_FILE)
    print(f"  wrote {POINTCLOUD_FILE}")

    # Report the merged extent as a lat/lon bbox, since that's what's
    # needed to manually pull an OSM export before the next step.
    proj_min_x = cloud.origin_x + cloud.bounds.min_x
    proj_max_x = cloud.origin_x + cloud.bounds.max_x
    proj_min_z = cloud.origin_y + cloud.bounds.min_z
    proj_max_z = cloud.origin_y + cloud.bounds.max_z
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
        "projection_epsg": projection,
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

    print(f"Found {osm_path} ({osm_path.stat().st_size:,} bytes).")
    print("OSM parsing is not implemented yet (osm.py TBD per the architecture doc's "
          "Milestone 5) -- this step currently just confirms the file is in place.")

    save_project(working_dir, {"osm_file_present": True})


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

    from terrain.bounding_box import BoundingBox
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


def step_output_terrain(working_dir: Path, course_dir: Path) -> None:
    stamps_path = working_dir / OPTIMIZED_STAMPS_FILE
    if not stamps_path.exists():
        stamps_path = working_dir / INITIAL_STAMPS_FILE
    if not stamps_path.exists():
        raise StepError(
            f"Neither {OPTIMIZED_STAMPS_FILE} nor {INITIAL_STAMPS_FILE} found under "
            f"{working_dir}. Run --step generate-terrain first."
        )

    print(f"Loading stamps from {stamps_path}...")
    stamps = load_stamps(stamps_path)
    print(f"  {len(stamps)} stamps")

    nodes_dir = course_dir / "CourseDescription_nodes"
    if not nodes_dir.is_dir():
        raise StepError(
            f"{nodes_dir} doesn't exist. Extract the blank starting .course into "
            f"{course_dir} first (see course_extract.py), or pass --course-dir."
        )

    out_path = nodes_dir / "userLayers.json"
    write_user_layers(stamps, out_path)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEPS = {
    "init": step_init,
    "ingest-laz": step_ingest_laz,
    "ingest-osm": step_ingest_osm,
    "generate-terrain": step_generate_terrain,
    "output-terrain": step_output_terrain,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PGA2K terrain compiler CLI")
    parser.add_argument("working_dir", type=Path, help="Project working directory")
    parser.add_argument("--step", default="init", choices=sorted(STEPS.keys()),
                         help="Pipeline step to run (default: init)")
    parser.add_argument("--projection", type=int, help="EPSG code (required for ingest-laz)")
    parser.add_argument("--course-dir", type=Path, default=None,
                         help="Extracted blank .course folder (default: <working_dir>/course)")
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
            if args.projection is None:
                print("error: --step ingest-laz requires --projection <EPSG code>", file=sys.stderr)
                return 1
            step_ingest_laz(working_dir, args.projection)
        elif args.step == "ingest-osm":
            step_ingest_osm(working_dir)
        elif args.step == "generate-terrain":
            step_generate_terrain(working_dir)
        elif args.step == "output-terrain":
            course_dir = args.course_dir or (working_dir / "course")
            step_output_terrain(working_dir, course_dir)
    except StepError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
