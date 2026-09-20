"""
course_output/range_nets.py

Tiling range-net modules (course_output/range_net_module.py) along OSM
splines tagged barrier=range_nets.

The tiling is a "string of pearls": every 8 m along a range-net way, a
RivieraCC post is placed (one per node -- shared at the seams, never
doubled), and between every pair of adjacent nodes the 4 stacked wire
panels + the buried brick anchor span the 8 m interval (anchored at the
interval start, the panels' 8 m extent comes from the prefab model's
long axis). See range_net_module.py for the module frame.

CHAINS, NOT PER-WAY: range-net ways in an OSM edit often connect into a
polyline (a box of nets with one side missing = 4 points, 3 lines, the
inner points shared). Tiling per-way would double the posts at every
shared inner point AND leave the inner points at arbitrary distances
(not multiples of 8 m), so no post would fall on a corner. So this
module:

  1. groups connected range-net ways into CHAINS (ways that share an
     endpoint within RANGE_NET_ENDPOINT_TOL_M),
  2. orders each chain into a single polyline,
  3. SOLVES for the inner-corner positions so every segment length is
     exactly a multiple of 8 m (RANGE_NET_MODULE_LENGTH_M) -- the
     user's "reposition the two inner points so the distance between
     them is a multiple of 8" requirement. The endpoints of the chain
     stay fixed; the inner corners slide to the nearest 8 m grid
     positions (least-squares, closed-form per corner -- see
     _solve_corners). This is what makes a post fall exactly on each
     corner, not mid-corner.
  4. tiles the solved polyline: one post per node (0, 8, 16, ...), one
     4-panel span + brick anchor per interval.

Each placed object carries `dy` (the module member's height above the
template course's 1 m ground datum -- see range_net_module.py). At
write-objects time, course_output/collections.py:apply_terrain_heights
resolves `dy` to an absolute y = target terrain height at (x, z) +
output_height_shift_m + dy, so the net re-grounds itself on whatever
terrain it lands on. No heightmap is needed in this generate step --
the same "compile once, format at write" split as course_output/collections.py
and course_output/parking.py:

  step_generate_range_nets:  features.geojson barrier=range_nets lines
                             -> range_nets.json   (frozen, course-local frame)
  step_pack_objects:         range_nets.json .objects -> objects.json
                             (kind="collection_object")
  step_write_objects:        -> placedObjects2/3.json groups (v2021
                             path-keyed; v2019 skips -- the fence prefabs
                             aren't in asset_catalog.json)

A range_nets.json record:

    {"source_ids": [int, ...],          # OSM way ids of the chain
     "length_m": float,                 # solved chain length (multiple of 8)
     "corners_moved_m": [float, ...],   # per inner corner, how far it moved
     "objects": [{"x","z","rotation_deg","scale","path","dy"}, ...]}
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from course_output.collections import _bearing_deg, _rotate
from course_output.range_net_module import (
    MODULE_LENGTH_M,
    far_post,
    post_anchor,
    span_members,
    start_post,
)

_DECIMALS = 3

# Defaults. Overridable per project (project.json "range_net_*" keys)
# and per CLI flag -- see PGA2k_gen.py step_generate_range_nets.
RANGE_NET_ENDPOINT_TOL_M = 0.5     # two range-net ways within this distance share an endpoint
RANGE_NET_SNAP = True              # solve the inner corners to the 8 m grid (the user's "reposition corners")
RANGE_NET_MAX_SOLVE_ITERS = 50
RANGE_NET_SOLVE_TOL_M = 1e-4
# A single (non-chain) way shorter than one module (8 m) can't hold even
# one full 8 m span -- skip it rather than emit a lone post.
RANGE_NET_MIN_LENGTH_M = MODULE_LENGTH_M


def _round(v: float) -> float:
    return round(float(v), _DECIMALS)


@dataclass(slots=True)
class RangeNetLine:
    """One barrier=range_nets OSM way. `line` is a course-local
    LineString; `source_id` is the OSM way id for back-reference."""
    line: LineString
    source_id: Optional[int] = None


@dataclass(slots=True)
class RangeNetChain:
    """A connected group of range-net ways, ordered into one polyline.
    `nodes` are the polyline's (x, z) points in order (the first and
    last are the chain's fixed endpoints; the inner ones are the solved
    corners). `source_ids` are the OSM way ids of the ways in the chain,
    in polyline order (None where the way carried no OSM id)."""
    nodes: list[tuple[float, float]] = field(default_factory=list)
    source_ids: list[Optional[int]] = field(default_factory=list)

    @property
    def line(self) -> LineString:
        return LineString(self.nodes)

    @property
    def length_m(self) -> float:
        return self.line.length


# ---------------------------------------------------------------------------
# chain grouping
# ---------------------------------------------------------------------------

def _chain_endpoints(line: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    return (line.coords[0], line.coords[-1])


def group_range_net_chains(lines: list[RangeNetLine],
                          tol_m: float = RANGE_NET_ENDPOINT_TOL_M) -> list[RangeNetChain]:
    """Group range-net ways into connected chains.

    Two ways belong to the same chain when an endpoint of one is within
    `tol_m` of an endpoint of the other. The chain is ordered by walking
    from one free endpoint to the other (the box case: 3 ways, 4 points,
    2 free ends, 2 shared inner points -> a 4-node polyline).

    A way that shares both of its endpoints with already-grouped ways
    (a closed loop) is rejected -- range-nets are open chains, not
    polygons, and a closed loop has no well-defined "endpoints" to
    snap. A way that can't be walked (branching / non-path topology) is
    also rejected with a clear error."""
    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    ends: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for ln in lines:
        ends.append(_chain_endpoints(ln.line))

    def _share_endpoint(ea: tuple, eb: tuple) -> bool:
        return any(math.hypot(a[0] - b[0], a[1] - b[1]) <= tol_m
                   for a in ea for b in eb)

    for i in range(n):
        for j in range(i + 1, n):
            if _share_endpoint(ends[i], ends[j]):
                union(i, j)

    by_group: dict[int, list[int]] = {}
    for i in range(n):
        by_group.setdefault(find(i), []).append(i)

    chains: list[RangeNetChain] = []
    for group in by_group.values():
        chain = _order_chain([lines[i] for i in group], tol_m)
        if chain is not None:
            chains.append(chain)
    return chains


def _order_chain(ways: list[RangeNetLine],
                 tol_m: float) -> Optional[RangeNetChain]:
    """Order one connected group of ways into a single open polyline.

    Greedy walk: start at the first way's first endpoint, then keep
    extending by the next way that shares the current endpoint (within
    `tol_m`) but isn't the one we just came from. A clean path (the box
    case) terminates at the other free end having used every way.

    Returns None when the group isn't a simple open path (a closed loop
    or a branch) -- the caller then reports the group as unsolvable."""
    if not ways:
        return RangeNetChain()

    def _way_coords(ln: RangeNetLine) -> list[tuple[float, float]]:
        # The way's FULL vertex list, not just its two endpoints -- a way
        # tagged barrier=range_nets can (and typically does) have interior
        # nodes (a bend drawn as one way), and those must stay in the
        # chain's polyline or tiling degenerates to a straight line
        # between the way's first and last node.
        return [(float(x), float(y)) for x, y in ln.line.coords]

    if len(ways) == 1:
        return RangeNetChain(nodes=_way_coords(ways[0]), source_ids=[ways[0].source_id])

    # Walk from the first way's first endpoint to the other free end,
    # extending at each step by the one unused way that shares the
    # current node. Each way contributes its full vertex list (reversed
    # if it's being walked from its "far" end), with the shared joint
    # point deduplicated.
    first = ways[0]
    nodes: list[tuple[float, float]] = _way_coords(first)
    source_ids: list[Optional[int]] = [first.source_id]
    used = {0}
    cur = nodes[-1]
    prev_idx = 0
    for _ in range(len(ways) - 1):
        nxt_idx: Optional[int] = None
        nxt_coords: Optional[list[tuple[float, float]]] = None
        for j in range(len(ways)):
            if j in used or j == prev_idx:
                continue
            coords = _way_coords(ways[j])
            ja, jb = coords[0], coords[-1]
            if math.hypot(ja[0] - cur[0], ja[1] - cur[1]) <= tol_m:
                nxt_idx, nxt_coords = j, coords
                break
            if math.hypot(jb[0] - cur[0], jb[1] - cur[1]) <= tol_m:
                nxt_idx, nxt_coords = j, list(reversed(coords))
                break
        if nxt_idx is None or nxt_coords is None:
            return None  # dead end before using every way -> not a path
        used.add(nxt_idx)
        nodes.extend(nxt_coords[1:])  # [0] is the shared joint, already in nodes
        source_ids.append(ways[nxt_idx].source_id)
        cur = nodes[-1]
        prev_idx = nxt_idx
    if len(used) != len(ways):
        return None
    return RangeNetChain(nodes=nodes, source_ids=source_ids)


# ---------------------------------------------------------------------------
# corner solver
# ---------------------------------------------------------------------------

def _circle_intersection(
    ax: float, az: float, r1: float,
    bx: float, bz: float, r2: float,
    hint: tuple[float, float],
) -> Optional[tuple[float, float]]:
    """The intersection point of circle(A, r1) and circle(B, r2) nearest
    to `hint`. None when the circles don't intersect (within tolerance).

    Standard two-circle formula: the intersection lies on the line
    A->B at distance d1 from A (d1 = (r1^2 - r2^2 + |AB|^2) / 2|AB|),
    offset perpendicularly by h = sqrt(r1^2 - d1^2)."""
    abx, abz = bx - ax, bz - az
    d = math.hypot(abx, abz)
    if d < 1e-9:
        return None
    if d > r1 + r2 + 1e-6 or d < abs(r1 - r2) - 1e-6:
        return None  # no intersection (or tangent handled by the h=0 case below)
    d1 = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - d1 * d1
    if h2 < 0:
        h2 = 0.0
    h = math.sqrt(h2)
    ux, uz = abx / d, abz / d
    mx, mz = ax + d1 * ux, az + d1 * uz
    c1 = (mx + h * uz, mz - h * ux)
    c2 = (mx - h * uz, mz + h * ux)
    return c1 if math.hypot(c1[0] - hint[0], c1[1] - hint[1]) <= math.hypot(c2[0] - hint[0], c2[1] - hint[1]) else c2


def _solve_corners(nodes: list[tuple[float, float]],
                   module_len: float = MODULE_LENGTH_M,
                   max_iters: int = RANGE_NET_MAX_SOLVE_ITERS,
                   tol_m: float = RANGE_NET_SOLVE_TOL_M,
                   ) -> tuple[list[tuple[float, float]], list[float]]:
    """Reposition the INNER points of a polyline so every segment length
    is an exact multiple of `module_len`. The endpoints stay fixed.

    Closed-form per iteration: with the current positions, each inner
    corner i picks the nearest segment multiples n_i = round(|p_{i-1} ->
    p_i| / L) and m_i = round(|p_i -> p_{i+1}| / L), then is moved to
    the circle-circle intersection of circle(p_{i-1}, n_i*L) and
    circle(p_{i+1}, m_i*L) that's nearest to its drawn position. Repeat
    until the max movement drops below `tol_m` (or `max_iters`).

    Returns (solved nodes, per-inner-corner movement in metres). A
    single-segment polyline (no inner corners) is returned unchanged,
    but its length is snapped by the caller (the chain's endpoints are
    fixed, so a lone short way is just too short -- see
    RANGE_NET_MIN_LENGTH_M)."""
    n = len(nodes)
    if n < 3:
        return list(nodes), []
    pts = [np.array(p, dtype=float) for p in nodes]
    drawn = [p.copy() for p in pts]
    for _ in range(max_iters):
        moved_max = 0.0
        for i in range(1, n - 1):
            d_prev = float(np.linalg.norm(pts[i] - pts[i - 1]))
            d_next = float(np.linalg.norm(pts[i + 1] - pts[i]))
            ni = max(1, int(round(d_prev / module_len)))
            mi = max(1, int(round(d_next / module_len)))
            sol = _circle_intersection(
                float(pts[i - 1][0]), float(pts[i - 1][1]), ni * module_len,
                float(pts[i + 1][0]), float(pts[i + 1][1]), mi * module_len,
                tuple(drawn[i]),
            )
            if sol is None:
                raise ValueError(
                    f"range-net corner {i} can't be snapped to the "
                    f"{module_len:g} m grid (segments {ni} and {mi} modules don't reach "
                    f"each other from the current endpoints) -- the chain's endpoints "
                    f"are fixed, so this layout has no exact 8-multiple solution. "
                    f"Move the chain's endpoints or draw it on a 8 m grid in the OSM editor."
                )
            new = np.array(sol, dtype=float)
            moved = float(np.linalg.norm(new - pts[i]))
            moved_max = max(moved_max, moved)
            pts[i] = new
        if moved_max < tol_m:
            break
    movements = [float(np.linalg.norm(pts[i] - drawn[i])) for i in range(1, n - 1)]
    return [(float(p[0]), float(p[1])) for p in pts], movements


# ---------------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------------

def _cumulative_lengths(nodes: list[tuple[float, float]]) -> list[float]:
    """Cumulative arc length at each node of a polyline, `cum[0] == 0.0`."""
    cum = [0.0]
    for i in range(1, len(nodes)):
        ax, az = nodes[i - 1]
        bx, bz = nodes[i]
        cum.append(cum[-1] + math.hypot(bx - ax, bz - az))
    return cum


def _segment_bearing(nodes: list[tuple[float, float]], i: int) -> float:
    ax, az = nodes[i]
    bx, bz = nodes[i + 1]
    return _bearing_deg(bx - ax, bz - az)


def _point_and_heading_at(nodes: list[tuple[float, float]], cum: list[float],
                          d: float) -> tuple[float, float, float]:
    """(x, z, bearing_deg) at arc-length `d` along the polyline `nodes`.

    The bearing is the FORWARD direction of the segment that starts at or
    just before `d` -- when `d` lands exactly on an interior node (every
    post/span anchor does, since posts sit on nodes and spans are
    anchored at their interval's start node), that's the segment leaving
    the node, never an average with the segment arriving at it. Averaging
    the two (as an earlier version of this function did, by sampling a
    small epsilon ahead of and behind `d` and taking the tangent) makes a
    90 degree turn in the OSM way come out as a 45 degree rotation on the
    tiled post/span -- see tile_range_net_chain's closed-loop seam post
    for the one case that legitimately wants that average."""
    n = len(nodes)
    d = max(0.0, min(d, cum[-1]))
    i = 0
    for j in range(n - 1):
        if cum[j] <= d + 1e-9:
            i = j
        else:
            break
    ax, az = nodes[i]
    bx, bz = nodes[i + 1]
    seg_len = cum[i + 1] - cum[i]
    t = 0.0 if seg_len < 1e-9 else (d - cum[i]) / seg_len
    x = ax + t * (bx - ax)
    z = az + t * (bz - az)
    return (x, z, _segment_bearing(nodes, i))


def _bisector_bearing_deg(bearing_in: float, bearing_out: float) -> float:
    """The bearing bisecting a turn from `bearing_in` to `bearing_out`
    (sum of the two heading unit vectors) -- used only for a closed
    loop's shared seam post, which has no single well-defined "next"
    segment to face."""
    rad_in, rad_out = math.radians(bearing_in), math.radians(bearing_out)
    dx = math.sin(rad_in) + math.sin(rad_out)
    dz = math.cos(rad_in) + math.cos(rad_out)
    return _bearing_deg(dx, dz) if math.hypot(dx, dz) > 1e-9 else bearing_out


def _resolve_members_at(members: list[dict], ax: float, az: float, heading_deg: float) -> list[dict]:
    """Rotate+translate a set of module members (each with dx=0, dz,
    rotation_deg, scale, dy, path) to world (ax, az) facing heading_deg --
    the same transform collections.resolve_collection applies to its
    template members (see that function)."""
    out: list[dict] = []
    for m in members:
        rx, rz = _rotate(m.get("dx", 0.0), m.get("dz", 0.0), heading_deg)
        obj = {
            "x": _round(ax + rx),
            "z": _round(az + rz),
            "rotation_deg": _round((m.get("rotation_deg", 0.0) + heading_deg) % 360.0),
            "scale": _round(m.get("scale", 1.0)),
            "dy": _round(m["dy"]),
            "path": m["path"],
        }
        out.append(obj)
    return out


def _dedup_same_node_posts(objects: list[dict]) -> list[dict]:
    """Collapse duplicate posts that land on the same node -- happens at a
    closed-loop seam node (node0 == nodeN) where the start post and the
    end post both resolve to the same (x, z, path). Keeps the first
    occurrence (the start post, which carries the bisector heading)."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for obj in objects:
        # full object identity -- the 4 stacked wire panels share (x, z,
        # path) but differ in dy (the 4-high stack), so dy MUST be in the
        # key or the dedup collapses the stack into a single panel.
        key = (obj["x"], obj["z"], obj["path"], obj.get("scale"), obj.get("rotation_deg"), obj.get("dy"))
        if key in seen:
            continue
        seen.add(key)
        out.append(obj)
    return out


def tile_range_net_chain(chain: RangeNetChain,
                         module_len: float = MODULE_LENGTH_M,
                         snap: bool = RANGE_NET_SNAP) -> dict:
    """One range_nets.json record for a solved chain: one post per node
    (0, module_len, 2*module_len, ...) and one 4-panel span + brick
    anchor per interval (anchored at the interval start). Posts are
    SHARED at the seams -- a run of N intervals emits N+1 posts, never
    two at one node. A closed loop (first node == last node) emits the
    seam post once, oriented by the bisector of its two incident
    segments."""
    nodes, movements = (
        _solve_corners(chain.nodes, module_len) if snap else (list(chain.nodes), [])
    )
    chain.nodes = nodes
    cum = _cumulative_lengths(nodes)
    total = cum[-1]
    n_segments = max(0, int(round(total / module_len)))
    # Guard: the solved length should be an exact multiple of the module
    # length; if the solver left a residual (it shouldn't -- the
    # circle-intersection snaps exactly), round the node count to the
    # nearest multiple so we don't emit a partial tail span.
    residual = abs(total - n_segments * module_len)
    if residual > 0.05:
        warnings.warn(
            f"range-net chain (source_ids={chain.source_ids}) solved length {total:.3f} m "
            f"isn't an exact multiple of {module_len:g} m (residual {residual:.3f} m) -- "
            f"snapping to {n_segments * module_len:g} m",
            RuntimeWarning,
        )

    is_closed = n_segments >= 2 and nodes[0] == nodes[-1]
    objects: list[dict] = []
    # Posts: one per node (0, L, 2L, ...). For a closed loop the end post
    # (k=n_segments) is the twin of the start post (k=0) at the same seam
    # node -- skip it; _dedup_same_node_posts is the safety net for any
    # remaining same-node twins.
    for k in range(n_segments + 1):
        if is_closed and k == n_segments:
            continue  # the seam node's post is the start post (k=0)
        x, z, bearing = _point_and_heading_at(nodes, cum, k * module_len)
        if is_closed and k == 0:
            # the seam post faces two different segments (the loop's
            # last and first) -- bisect, rather than arbitrarily facing
            # only the outgoing one.
            bearing = _bisector_bearing_deg(_segment_bearing(nodes, len(nodes) - 2),
                                            _segment_bearing(nodes, 0))
        post = start_post()
        objects += _resolve_members_at([post], x, z, bearing)
    # Spans: one per interval (anchored at the interval start).
    for k in range(n_segments):
        x, z, bearing = _point_and_heading_at(nodes, cum, k * module_len)
        objects += _resolve_members_at(span_members(), x, z, bearing)
    # The chain's terminal post (open chains only -- a closed loop's seam
    # post is the start of interval 0, so it already got one above) isn't
    # the start of any interval, so span_members() never anchors one
    # there -- give it its own post_anchor(), same as the metal post is
    # explicitly placed at every node above.
    if n_segments > 0 and not is_closed:
        x, z, bearing = _point_and_heading_at(nodes, cum, n_segments * module_len)
        objects += _resolve_members_at([post_anchor()], x, z, bearing)

    objects = _dedup_same_node_posts(objects)

    return {
        "source_ids": list(chain.source_ids),
        "length_m": _round(total),
        "segments": n_segments,
        "corners_moved_m": [_round(m) for m in movements],
        "objects": objects,
    }


def build_range_net_records(
    lines: list[RangeNetLine],
    *,
    module_len: float = MODULE_LENGTH_M,
    endpoint_tol_m: float = RANGE_NET_ENDPOINT_TOL_M,
    snap: bool = RANGE_NET_SNAP,
    min_length_m: float = RANGE_NET_MIN_LENGTH_M,
    printf=print,
) -> list[dict]:
    """Frozen range_nets.json payload -- one record per connected chain
    (see the module docstring). Chains shorter than one module are
    skipped (a lone post with no span is pointless)."""
    chains = group_range_net_chains(lines, endpoint_tol_m)
    if not chains:
        printf("  no range-net chains found (no connected barrier=range_nets lines) -- nothing to do")
        return []
    records: list[dict] = []
    skipped = 0
    for chain in chains:
        if chain.length_m < min_length_m:
            printf(f"  NOTE: skipping range-net chain (source_ids={chain.source_ids}) -- "
                   f"{chain.length_m:.1f} m is shorter than one {module_len:g} m module")
            skipped += 1
            continue
        record = tile_range_net_chain(chain, module_len, snap)
        records.append(record)
        moved_note = ""
        if record["corners_moved_m"]:
            moved_note = (f", {len(record['corners_moved_m'])} inner corner(s) repositioned "
                          f"({max(record['corners_moved_m']):.2f} m max)")
        printf(f"  chain (source_ids={record['source_ids']}): {record['segments']} segment(s) "
               f"along {record['length_m']:.1f} m, {len(record['objects'])} object(s){moved_note}")
    printf(f"  {len(records)} range-net chain(s) ({skipped} skipped)")
    return records


def iter_range_net_objects(records: list[dict]):
    """Every placed object across every range-net chain record, shaped as
    a collections.py-style object dict (x, z, rotation_deg, scale, path,
    dy) plus its chain's source_ids -- the flat list
    step_pack_objects folds into objects.json as kind="collection_object"."""
    for record in records:
        for obj in record.get("objects", []):
            yield {
                "x": obj["x"], "z": obj["z"],
                "rotation_deg": obj.get("rotation_deg", 0.0),
                "scale": obj.get("scale", 1.0),
                "dy": obj.get("dy"),
                "path": obj["path"],
                "category": None, "type": None, "theme": False,
                "source_ids": record.get("source_ids"),
            }


def save_range_net_records(records: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def load_range_net_records(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)
