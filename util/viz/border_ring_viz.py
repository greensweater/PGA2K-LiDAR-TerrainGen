"""
util/viz/border_ring_viz.py

Diagnostic for the "border fill" cluster-object feature (see
course_output/object_clusters.py's build_border_ring_geometry,
_pack_circles, _pack_ring_walk): renders the real border-ring geometry
for a real project's height mask, marks every concave (reflex) vertex
on the mask's own boundary, and scatters the actual circles the chosen
--algo places into that ring, sized/colored by tier/radius.

--algo walk (the default, and what production write-objects now uses
for every border fill) exists to replace --algo dart, which is kept
here only for side-by-side comparison. dart's original hypothesis check
-- that its small-tier circles cluster near concave corners of the mask
boundary, because build_border_ring_geometry's symmetric stroke can
locally widen (self-overlap) on the inward side there -- was confirmed
against real project geometry (it could not be reproduced against a
synthetic rectangle/circle mask during triage), which is what motivated
walk in the first place: it places the largest circle that locally fits
at each step along the ring's own centerline, so it never depends on a
fixed tier ladder happening to match the ring's actual local width.

THIN WRAPPER, not a reimplementation, same convention as
util/viz/poisson_viz.py: every geometric result here comes from calling
the real production functions directly (merge_height_mask_features,
build_border_ring_geometry, _pack_circles/_pack_ring_walk), never
reimplemented, so there's zero risk of this tool's own logic drifting
out of sync with the real algorithm.

Run from the project root (or with it on PYTHONPATH), same
expectation every other script in this project already has.

Usage:
    python util/viz/border_ring_viz.py <working_dir> --category-id N [options]

--category-id must name one of asset_catalog.json's clusterable
("nature") categories -- see course_output/asset_catalog.py's
NATURE_CATEGORY_IDS for which ids qualify.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never opens a window, just writes a file
import matplotlib.pyplot as plt

try:
    from PGA2k_gen import FEATURES_FILE
    from ingest.osm import load_features, merge_height_mask_features
    from course_output.object_clusters import build_border_ring_geometry, _pack_circles, _pack_ring_walk
    from course_output.asset_catalog import ASSET_CATEGORIES
except ImportError as e:
    print(f"Couldn't import project modules ({e}).\n"
          "Run this from the project root, or put the project root on PYTHONPATH -- "
          "same expectation every other script in this project already has.")
    sys.exit(1)


def _concave_vertices(linear_ring) -> list[tuple[float, float]]:
    """
    Coords on `linear_ring` (a shapely LinearRing -- Polygon.exterior or
    one of Polygon.interiors) whose interior angle is reflex (concave,
    bending INTO the polygon's own material rather than into open
    space) -- the corners suspected of locally widening
    build_border_ring_geometry's symmetric stroke on the inward side
    (see module docstring). `linear_ring.is_ccw` fixes the turn-sign
    convention for both winding directions (exterior rings and
    interior/hole rings wind oppositely in a valid shapely Polygon).
    """
    coords = list(linear_ring.coords)[:-1]  # last coord duplicates the first
    n = len(coords)
    if n < 3:
        return []
    sign = 1.0 if linear_ring.is_ccw else -1.0
    concave = []
    for i in range(n):
        x0, z0 = coords[i - 1]
        x1, z1 = coords[i]
        x2, z2 = coords[(i + 1) % n]
        ax, az = x1 - x0, z1 - z0
        bx, bz = x2 - x1, z2 - z1
        cross = ax * bz - az * bx
        if cross * sign < 0:
            concave.append((x1, z1))
    return concave


def _polygon_parts(geometry):
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    return [p for p in parts if p.geom_type == "Polygon"]


def _draw_polygon(ax, geometry, facecolor, alpha=1.0, zorder=1):
    for part in _polygon_parts(geometry):
        xs, zs = part.exterior.xy
        ax.fill(xs, zs, facecolor=facecolor, edgecolor="none", alpha=alpha, zorder=zorder)
        for interior in part.interiors:
            ixs, izs = interior.xy
            ax.fill(ixs, izs, facecolor="white", edgecolor="none", alpha=1.0, zorder=zorder + 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("working_dir", type=Path, help="project working dir containing features.geojson")
    parser.add_argument("--border-width", type=float, default=10.0, help="ring width in meters (GUI default: 10.0)")
    parser.add_argument("--buffer-px", type=float, default=50.0,
                         help="mask buffer in meters, matches the GUI's mask_buffer_preview_var (default: 50.0)")
    parser.add_argument("--category-id", type=int, required=True,
                         help="clusterable asset category id to pull cluster_radius from (see asset_catalog.json)")
    parser.add_argument("--ratio", type=float, default=1.0, help="GUI's 'Raster ratio' knob (default: 1.0)")
    parser.add_argument("--algo", choices=["walk", "dart"], default="walk",
                         help="packing algorithm: 'walk' (default, production) is the deterministic "
                              "centerline walk; 'dart' is the old tiered dart-throw, kept for comparison")
    parser.add_argument("--seed", type=int, default=0, help="only used by --algo dart (walk is deterministic)")
    parser.add_argument("--output", type=Path, default=Path("border_ring_viz.png"))
    args = parser.parse_args()

    category = ASSET_CATEGORIES.get(args.category_id)
    if category is None or category.cluster_radius is None:
        raise SystemExit(
            f"category {args.category_id} isn't a clusterable ('nature') category -- "
            "see asset_catalog.json's cluster_radius field / NATURE_CATEGORY_IDS"
        )

    features_path = args.working_dir / FEATURES_FILE
    features = load_features(features_path)
    merged = merge_height_mask_features(features)
    if merged is None or merged.is_empty:
        raise SystemExit(f"no unmasked height-mask features found in {features_path}")
    mask_geom = merged.buffer(args.buffer_px)

    ring = build_border_ring_geometry(mask_geom, args.border_width)
    if ring.is_empty:
        raise SystemExit("border ring is empty -- border_width must be > 0 and mask_geom non-empty")

    if args.algo == "walk":
        circles = _pack_ring_walk(mask_geom.boundary, ring, category.cluster_radius, args.ratio)
    else:
        rng = random.Random(args.seed)
        circles = _pack_circles(ring, category.cluster_radius, args.ratio, rng)

    concave_pts: list[tuple[float, float]] = []
    for part in _polygon_parts(mask_geom):
        concave_pts.extend(_concave_vertices(part.exterior))
        for interior in part.interiors:
            concave_pts.extend(_concave_vertices(interior))

    fig, ax = plt.subplots(figsize=(12, 12), dpi=120)
    ax.set_aspect("equal")

    minx, minz, maxx, maxz = ring.bounds
    pad = max(args.border_width * 2, 5.0)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(minz - pad, maxz + pad)

    _draw_polygon(ax, mask_geom, facecolor="#eeeeee", zorder=1)
    _draw_polygon(ax, ring, facecolor="#a8d5ff", alpha=0.75, zorder=2)

    if concave_pts:
        cx, cz = zip(*concave_pts)
        ax.scatter(cx, cz, color="red", s=25, zorder=5, label=f"concave boundary vertex ({len(concave_pts)})")

    if circles:
        # --algo dart's radii come from a fixed tier ladder (a handful of exact
        # values) -- --algo walk's vary continuously with the ring's local width,
        # so grouping to 1 decimal (not dart's exact-match) keeps the legend/color
        # buckets sane for walk instead of exploding into one entry per circle.
        tier_radii = sorted({round(r, 1) for _, _, r in circles}, reverse=True)
        cmap = plt.get_cmap("plasma")
        for i, tier_radius in enumerate(tier_radii):
            tier_pts = [(x, z, r) for x, z, r in circles if round(r, 1) == tier_radius]
            color = cmap(i / max(1, len(tier_radii) - 1))
            for x, z, r in tier_pts:
                ax.add_patch(plt.Circle((x, z), r, facecolor="none",
                                         edgecolor=color, linewidth=1.2, zorder=4))
            ax.scatter([], [], color=color, label=f"r~{tier_radius:.1f}m ({len(tier_pts)})")
    else:
        print("NOTE: no circles placed at all -- border_width/category combination doesn't fit anywhere")

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"algo={args.algo}  border_width={args.border_width}m  category={category.description}  "
        f"cluster_radius={category.cluster_radius}m  total circles={len(circles)}"
    )
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"wrote {args.output}  ({len(circles)} circles across {len(set(round(r, 1) for _, _, r in circles))} "
          f"radius bucket(s), {len(concave_pts)} concave boundary vertices)")


if __name__ == "__main__":
    main()
