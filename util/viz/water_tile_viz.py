#!/usr/bin/env python3
"""
water_tile_viz.py

Standalone diagnostic tool -- not part of the pipeline's runtime call
graph, imports real production functions to validate the multi-tile
water fill modes in isolation (see course_output/water.py's
fit_water_tiles/fit_water_stripes). Same "prototype/debug via a real
matplotlib plot, against real project data" idea as boundary_trace_
viz.py/fallline_fill_viz.py/rect_fill_viz.py at the repo root.

For every "water" Polygon Feature in a working directory's
features.geojson (course-cropped), computes fit_water_rectangle
(single-rectangle baseline), fit_water_tiles ("edge" multi-tile fill),
and fit_water_stripes ("stripe" multi-tile fill). Prints, per pond and
totalled, both the OVERSHOOT area (tile coverage outside the real
polygon) and the GAP area (real polygon left uncovered) for each of
edge/stripe -- gap % is the metric that actually matters for comparing
the two modes, since edge-fill's greedy dedup pass is what can leave
real, unpredictable holes on complex shapes, which is the whole reason
fit_water_stripes exists. Renders a 2-panel plot per pond: real outline
+ old single rectangle (reference) on both panels, edge-fill tiles on
the left, stripe-fill tiles on the right.

Usage:
    python water_tile_viz.py <working_dir> [--tolerance-m M] [--min-edge-m M]
                              [--max-search-m M] [--width-samples N]
                              [--redundancy-ratio R] [--overlap-m M]
                              [--stripe-overlap-m M] [--stripe-min-edge-m M]
                              [--stripe-tolerance-m M] [--stripe-max-stripes N]
                              [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ingest.osm import load_features
from PGA2k_gen import _crop_features_to_course
from course_output.water import (
    fit_water_rectangle, fit_water_tiles, fit_water_stripes, _water_tile_corners,
    DEFAULT_WATER_TILE_TOLERANCE_M, DEFAULT_WATER_TILE_MIN_EDGE_M,
    DEFAULT_WATER_TILE_MAX_SEARCH_M, DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    DEFAULT_WATER_TILE_REDUNDANCY_RATIO, DEFAULT_WATER_TILE_OVERLAP_M,
    DEFAULT_WATER_STRIPE_OVERLAP_M, DEFAULT_WATER_STRIPE_TOLERANCE_M,
    DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
)


def _plot_panel(ax, poly, old_rect, tiles, color, label, title):
    px, pz = poly.exterior.xy
    ax.plot(px, pz, "k-", linewidth=2, label="real pond outline")
    ox, oz = old_rect.exterior.xy
    ax.plot(ox, oz, "r--", linewidth=1.5, label="old: single rectangle")
    for i, t in enumerate(tiles):
        corners = _water_tile_corners(*t) + [_water_tile_corners(*t)[0]]
        tx, tz = zip(*corners)
        ax.plot(tx, tz, color=color, linewidth=1, alpha=0.7, label=label if i == 0 else None)
        ax.fill(tx, tz, color=color, alpha=0.1)
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("working_dir", type=Path)
    parser.add_argument("--tolerance-m", type=float, default=DEFAULT_WATER_TILE_TOLERANCE_M)
    parser.add_argument("--min-edge-m", type=float, default=DEFAULT_WATER_TILE_MIN_EDGE_M)
    parser.add_argument("--max-search-m", type=float, default=DEFAULT_WATER_TILE_MAX_SEARCH_M)
    parser.add_argument("--width-samples", type=int, default=DEFAULT_WATER_TILE_WIDTH_SAMPLES)
    parser.add_argument("--redundancy-ratio", type=float, default=DEFAULT_WATER_TILE_REDUNDANCY_RATIO)
    parser.add_argument("--overlap-m", type=float, default=DEFAULT_WATER_TILE_OVERLAP_M)
    parser.add_argument("--stripe-overlap-m", type=float, default=DEFAULT_WATER_STRIPE_OVERLAP_M)
    parser.add_argument("--stripe-min-edge-m", type=float, default=DEFAULT_WATER_TILE_MIN_EDGE_M)
    parser.add_argument("--stripe-tolerance-m", type=float, default=DEFAULT_WATER_STRIPE_TOLERANCE_M)
    parser.add_argument("--stripe-max-stripes", type=int, default=DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE)
    parser.add_argument("--out", type=Path, default=None,
                         help="Directory to save per-pond PNGs into (default: <working_dir>/preview)")
    args = parser.parse_args()

    features_path = args.working_dir / "features.geojson"
    if not features_path.exists():
        print(f"No features.geojson found under {args.working_dir}")
        return 1

    features = load_features(features_path)
    features = _crop_features_to_course(args.working_dir, features)
    water = [f for f in features if f.kind == "water" and f.geometry.geom_type == "Polygon"]
    if not water:
        print("No water polygons found.")
        return 0

    out_dir = args.out or (args.working_dir / "preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(water)} water polygon(s).\n"
          f"  edge:   tolerance_m={args.tolerance_m} min_edge_m={args.min_edge_m} "
          f"max_search_m={args.max_search_m} width_samples={args.width_samples} "
          f"redundancy_ratio={args.redundancy_ratio} overlap_m={args.overlap_m}\n"
          f"  stripe: overlap_m={args.stripe_overlap_m} min_edge_m={args.stripe_min_edge_m} "
          f"tolerance_m={args.stripe_tolerance_m} max_stripes_per_side={args.stripe_max_stripes}\n")

    totals = {"edge_overshoot": 0.0, "edge_gap": 0.0, "stripe_overshoot": 0.0, "stripe_gap": 0.0, "area": 0.0}

    for f in water:
        poly = f.geometry
        old = fit_water_rectangle(poly)
        edge = fit_water_tiles(
            poly, tolerance_m=args.tolerance_m, min_edge_m=args.min_edge_m,
            max_search_m=args.max_search_m, width_samples=args.width_samples,
            redundancy_ratio=args.redundancy_ratio, overlap_m=args.overlap_m,
        )
        stripe = fit_water_stripes(
            poly, overlap_m=args.stripe_overlap_m, min_edge_m=args.stripe_min_edge_m,
            tolerance_m=args.stripe_tolerance_m, max_stripes_per_side=args.stripe_max_stripes,
        )
        if old is None or edge is None or stripe is None:
            print(f"osm_id={f.osm_id}: fit failed (old={old is not None} edge={edge is not None} "
                  f"stripe={stripe is not None})")
            continue

        old_rect = Polygon(_water_tile_corners(*old))
        edge_union = unary_union([Polygon(_water_tile_corners(*t)) for t in edge])
        stripe_union = unary_union([Polygon(_water_tile_corners(*t)) for t in stripe])
        pond_area = poly.area

        edge_overshoot = edge_union.difference(poly).area
        edge_gap = poly.difference(edge_union).area
        stripe_overshoot = stripe_union.difference(poly).area
        stripe_gap = poly.difference(stripe_union).area

        totals["edge_overshoot"] += edge_overshoot
        totals["edge_gap"] += edge_gap
        totals["stripe_overshoot"] += stripe_overshoot
        totals["stripe_gap"] += stripe_gap
        totals["area"] += pond_area

        print(f"osm_id={f.osm_id!s:>12} pond_area={pond_area:8.1f}  "
              f"edge: tiles={len(edge):2d} overshoot={100*edge_overshoot/pond_area:5.1f}% "
              f"gap={100*edge_gap/pond_area:5.1f}%   "
              f"stripe: tiles={len(stripe):2d} overshoot={100*stripe_overshoot/pond_area:5.1f}% "
              f"gap={100*stripe_gap/pond_area:5.1f}%")

        fig, (ax_edge, ax_stripe) = plt.subplots(1, 2, figsize=(12, 6))
        _plot_panel(ax_edge, poly, old_rect, edge, "b", "edge: tiles",
                    f"EDGE  osm_id={f.osm_id}\novershoot={100*edge_overshoot/pond_area:.1f}% "
                    f"gap={100*edge_gap/pond_area:.1f}%")
        _plot_panel(ax_stripe, poly, old_rect, stripe, "g", "stripe: tiles",
                    f"STRIPE  osm_id={f.osm_id}\novershoot={100*stripe_overshoot/pond_area:.1f}% "
                    f"gap={100*stripe_gap/pond_area:.1f}%")
        out_path = out_dir / f"water_tile_check_{f.osm_id}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)

    if totals["area"] > 0:
        a = totals["area"]
        print(f"\nTOTAL over {a:.1f} sqm of pond:\n"
              f"  edge:   overshoot={100*totals['edge_overshoot']/a:.1f}%  gap={100*totals['edge_gap']/a:.1f}%\n"
              f"  stripe: overshoot={100*totals['stripe_overshoot']/a:.1f}%  gap={100*totals['stripe_gap']/a:.1f}%")
    print(f"\nSaved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
