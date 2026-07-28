"""
terrain/adaptive_refine.py

Milestone 4's initial adaptive refinement pass: find stamps whose
local fit is worst against real LIDAR, and add smaller detail stamps
there -- rather than uniformly subdividing everywhere (see the
architecture doc's "Adaptive Refinement": "Only subdivide stamps whose
local error exceeds tolerance").

No masks exist yet (Milestone 5), so this uses a single global error
tolerance rather than the mask-driven per-region tolerances the doc
describes as the eventual design -- this is the "for now" version,
same spirit as height_fit.py's naive averaging.

Designed to run repeatedly: each call only touches stamps still over
tolerance, including previously-added detail stamps from an earlier
call that still aren't fine enough -- "largest error -> split ->
repeat" naturally falls out of just re-running refine_stamps() against
its own prior output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ingest.laz_reader import PointCloud
from terrain.bounding_box import BoundingBox
from terrain.height_fit import fit_stamp_heights
from terrain.hexgrid import DEFAULT_BRUSH, generate_hex_grid
from terrain.stamp import Stamp
from terrain.terrain_model import TerrainModel

DEFAULT_SUBDIVISION_FACTOR = 2.0
DEFAULT_MIN_POINTS = 3


@dataclass(slots=True)
class StampError:
    stamp: Stamp
    rms_error: float
    n_points: int


def compute_stamp_errors(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
) -> list[StampError]:
    """
    Local RMS error per stamp: predicted TerrainModel height (from the
    *full* current stamp list, so every stamp's neighbors' contributions
    are accounted for) vs. actual LIDAR elevation, at that stamp's own
    nearby bare-earth points.

    Stamps with fewer than `min_points` nearby points are skipped
    (excluded from the result, not scored as zero-error) -- there's
    no data to judge them by, so they shouldn't look artificially fine.
    """
    model = TerrainModel(stamps)
    results: list[StampError] = []

    for stamp in stamps:
        idx = cloud.query_radius(stamp.x, stamp.z, stamp.radius)
        if bare_earth_only and idx.size > 0:
            idx = idx[cloud.bare_earth_mask()[idx]]

        if idx.size < min_points:
            continue

        points = np.column_stack((cloud.x[idx], cloud.z[idx]))
        predicted = model.evaluate_many(points)
        actual = cloud.elevation[idx]
        rms = float(np.sqrt(np.mean(np.square(predicted - actual))))
        results.append(StampError(stamp=stamp, rms_error=rms, n_points=int(idx.size)))

    return results


def generate_detail_stamps(
    parent: Stamp,
    subdivision_factor: float = DEFAULT_SUBDIVISION_FACTOR,
    brush: int = DEFAULT_BRUSH,
) -> list[Stamp]:
    """
    A small local hex lattice covering `parent`'s footprint, at
    parent.radius / subdivision_factor -- same 2:1 radius:pitch ratio
    used for the whole-course grid (see hexgrid.py), just applied
    locally at a finer scale. No bleed: this is a local patch, not a
    course-edge concern.
    """
    child_radius = parent.radius / subdivision_factor
    child_pitch = child_radius / 2.0
    local_bounds = BoundingBox(
        min_x=parent.x - parent.radius, max_x=parent.x + parent.radius,
        min_z=parent.z - parent.radius, max_z=parent.z + parent.radius,
    )
    return generate_hex_grid(local_bounds, pitch=child_pitch, stamp_radius=child_radius,
                              brush=brush, bleed=0.0)


def refine_stamps(
    stamps: Sequence[Stamp],
    cloud: PointCloud,
    tolerance: float,
    subdivision_factor: float = DEFAULT_SUBDIVISION_FACTOR,
    bare_earth_only: bool = True,
    min_points: int = DEFAULT_MIN_POINTS,
    max_new_stamps: int | None = None,
) -> tuple[list[Stamp], list[StampError]]:
    """
    One adaptive refinement pass: flag every stamp whose local RMS
    error exceeds `tolerance`, generate detail stamps for each, fit
    their heights against the existing (unchanged) coarse baseline,
    and return the combined stamp list.

    Flagged (unchanged) stamps are NOT removed -- their detail stamps
    are appended after them, so under pull-toward-value semantics the
    detail stamps refine on top of the existing baseline rather than
    replacing it outright (see terrain_model.py). This also means
    re-running refine_stamps() on its own output is safe and does the
    right thing: previously-added detail stamps get scored again, and
    only the ones still over tolerance get subdivided further.

    Returns (new_stamp_list, flagged_errors) -- flagged_errors is
    useful for reporting/diagnostics even though the stamps themselves
    are already folded into the returned list.
    """
    errors = compute_stamp_errors(stamps, cloud, bare_earth_only=bare_earth_only, min_points=min_points)
    flagged = sorted((e for e in errors if e.rms_error > tolerance),
                      key=lambda e: e.rms_error, reverse=True)

    detail_positions: list[Stamp] = []
    for stamp_error in flagged:
        detail_positions.extend(generate_detail_stamps(stamp_error.stamp, subdivision_factor))
        if max_new_stamps is not None and len(detail_positions) >= max_new_stamps:
            detail_positions = detail_positions[:max_new_stamps]
            break

    if not detail_positions:
        return list(stamps), flagged

    fitted_details = fit_stamp_heights(
        detail_positions, cloud,
        bare_earth_only=bare_earth_only, min_points=min_points,
        existing_stamps=stamps,
    )

    return list(stamps) + fitted_details, flagged
