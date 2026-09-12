"""
util/extract_object_spline_fills.py

Dump every object-spline fill (placedObjects3.json Value.splines[]) from a
hand-built / in-game-saved .course so we can reverse-engineer how the
designer's foliage-density slider maps to the stored `fillPct` -- which
is NOT a simple percent and appears to be per-asset (a maxed tree spline
came out at fillPct 0.03, a maxed grass spline at 1.0).

HOW TO COLLECT DATA (in-game):
  1. Open any course in the designer.
  2. For each asset you care about, paint a few CLOSED object-spline
     fills (the foliage "spline" tool), each a decent size, and set each
     one's density slider to a round position -- e.g. 0%, 25%, 50%,
     75%, 100%. One patch per (asset, slider %) so they don't merge.
     Jot down which patch you set to what.
  3. Save the course, then extract it:
        python util/course_extract.py "<that>.course" <out_dir>
  4. python util/extract_object_spline_fills.py <out_dir>

The table gives, per spline: asset path, stored fillPct, isFilled/state
flags, polygon area (m^2) and centroid -- match centroid/area back to the
patch you painted to pair "slider %" with "fillPct". --catalog adds each
asset's asset_catalog.json spacing / native canopy radius next to it so
you can eyeball whether either predicts the per-asset fillPct ceiling.

Also prints the course-wide foliage knobs (CourseDescription.json's
grassDensity / detailDensity / rockDensity / plantDensity / treeDensity
and treeOptions.json's density) -- those are the global procedural
scatter, separate from per-spline fillPct, but worth recording.

Read-only: never writes anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _polygon_area_centroid(pts: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Shoelace area (abs) + centroid for a closed ring (first==last ok)."""
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    a = cx = cz = 0.0
    for (x0, z0), (x1, z1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * z1 - x1 * z0
        a += cross
        cx += (x0 + x1) * cross
        cz += (z0 + z1) * cross
    if abs(a) < 1e-9:
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        return 0.0, sum(xs) / len(xs), sum(zs) / len(zs)
    return abs(a) / 2.0, cx / (3.0 * a), cz / (3.0 * a)


def _waypoint_xy(wp: dict) -> tuple[float, float]:
    p = wp.get("waypoint", wp)
    return float(p["x"]), float(p["y"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("course_dir", help="extracted .course folder (contains CourseDescription_nodes/)")
    ap.add_argument("--catalog", action="store_true",
                    help="annotate each asset with its asset_catalog.json spacing / native canopy radius")
    ap.add_argument("--min-area", type=float, default=0.0,
                    help="hide spline pieces smaller than this many m^2 (subdivision slivers)")
    args = ap.parse_args()

    nodes = Path(args.course_dir) / "CourseDescription_nodes"
    placed_path = nodes / "placedObjects3.json"
    if not placed_path.exists():
        print(f"No {placed_path} -- is this a v2021+ extracted course?")
        return 1
    placed = json.loads(placed_path.read_text(encoding="utf-8"))

    cat_by_path: dict[str, str] = {}
    if args.catalog:
        from course_output.asset_catalog import ASSET_ENTRIES  # noqa: E402
        for e in ASSET_ENTRIES:
            bits = []
            if getattr(e, "spacing", None) is not None:
                bits.append(f"spacing={e.spacing:.3f}")
            r = getattr(e, "native_canopy_radius_m", None)
            if r is not None:
                bits.append(f"canopyR={r:.1f}")
            cat_by_path[e.path] = " ".join(bits) if bits else "(no catalog metrics)"

    # --- course-wide foliage knobs ---
    cd_path = Path(args.course_dir) / "CourseDescription.json"
    if cd_path.exists():
        cd = json.loads(cd_path.read_text(encoding="utf-8"))
        knobs = {k: cd.get(k) for k in (
            "grassDensity", "detailDensity", "rockDensity", "plantDensity", "treeDensity"
        )}
        print("course-wide foliage knobs (CourseDescription.json):")
        for k, v in knobs.items():
            print(f"  {k:>14} = {v!r}")
    to_path = nodes / "treeOptions.json"
    if to_path.exists():
        to = json.loads(to_path.read_text(encoding="utf-8"))
        print(f"  treeOptions.density = {to.get('density')!r}")
    print()

    rows: list[tuple] = []
    per_asset: dict[str, list[float]] = {}
    for group in placed:
        path = group.get("Key", {}).get("path", "")
        for sp in group.get("Value", {}).get("splines", []) or []:
            pathobj = sp.get("path", {})
            wps = [_waypoint_xy(w) for w in pathobj.get("waypoints", [])]
            area, cx, cz = _polygon_area_centroid(wps)
            if area < args.min_area:
                continue
            fill_pct = sp.get("fillPct")
            rows.append((
                path, fill_pct, pathobj.get("isFilled"), pathobj.get("state"),
                pathobj.get("ClosedPath"), pathobj.get("isClosed"),
                len(wps), area, cx, cz,
            ))
            per_asset.setdefault(path, []).append(fill_pct if isinstance(fill_pct, (int, float)) else math.nan)

    if not rows:
        print("No object-spline fills (Value.splines[]) found.")
        return 0

    print(f"{'fillPct':>13}  {'isFill':>6} {'state':>5} {'closed':>6} {'nwp':>4} "
          f"{'area_m2':>10}  {'cx':>9} {'cz':>9}  asset")
    for path, fp, isf, st, cp, ic, nwp, area, cx, cz in sorted(rows, key=lambda r: (r[0], r[1] or 0)):
        fps = f"{fp:.10g}" if isinstance(fp, (int, float)) else str(fp)
        closed = "Y" if cp and ic else ("cp" if cp else ("ic" if ic else "n"))
        print(f"{fps:>13}  {str(isf):>6} {str(st):>5} {closed:>6} {nwp:>4} "
              f"{area:>10.1f}  {cx:>9.1f} {cz:>9.1f}  {path}")

    print("\nper-asset distinct fillPct (min .. max):")
    for path in sorted(per_asset):
        vals = sorted(v for v in per_asset[path] if not math.isnan(v))
        if not vals:
            continue
        uniq = sorted(set(round(v, 10) for v in vals))
        extra = f"   [{cat_by_path.get(path, '')}]" if args.catalog else ""
        shown = ", ".join(f"{v:.6g}" for v in uniq[:8]) + (" ..." if len(uniq) > 8 else "")
        print(f"  {min(vals):.6g} .. {max(vals):.6g}  ({len(vals)} splines)  {path}{extra}")
        print(f"      values: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
