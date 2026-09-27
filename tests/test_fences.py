"""
Tests for course_output/fences.py (v2023 task 3.2) -- stdlib unittest,
run with:  python -m unittest tests.test_fences -v

Structure is checked against the real objectPaths rows in
templates/2023_fences.course (extracted to a temp dir), and the handle
rule is checked by rebuilding two sample rows from their waypoints.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import LineString

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "util"))

from course_extract import extract_course_file  # noqa: E402
from course_output.fences import (  # noqa: E402
    BRICK_LOW_RAILS_ASSET, BRICK_WALL_ASSET, CANVAS_BLACK_ASSET, FENCE_TYPES, FenceRecord,
    LOW_METAL_RAILING_ASSET, PICKET_PICKETS_ASSET, PICKET_POSTS_ASSET, RETAINING_WALL_ASSET,
    STONE_WALL_ASSET, UNI_FENCE_ASSET, WIRE_FENCE_ASSET, WOOD_FENCE_ASSET, apply_leveled_heights,
    build_fence_records,
    fence_record_to_object_path_v2023, fence_records_to_groups_v2023, load_fence_records,
    resolve_fence_style, save_fence_records,
)
from course_output.userLayers import GRID_ORIGIN_OFFSET  # noqa: E402
from ingest.osm import Feature  # noqa: E402


def _sample_object_paths() -> list[tuple[str, dict]]:
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        extract_course_file(str(REPO / "templates" / "2023_fences.course"), tmp)
        groups = json.loads((Path(tmp) / "CourseDescription_nodes" / "placedObjects3.json").read_text())
    return [(g["Key"]["path"], op) for g in groups for op in g["Value"].get("objectPaths") or []]


SAMPLE = _sample_object_paths()


def _shape(obj):
    """Key structure + value types, recursively (lists by their first element)."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shape(obj[0])] if obj else []
    return type(obj).__name__


def _way(coords, tags, osm_id=1, kind="fence") -> Feature:
    return Feature(geometry=LineString(coords), kind=kind, tags=tags, osm_id=osm_id)


def _local(p: dict) -> tuple[float, float]:
    return (p["x"] + GRID_ORIGIN_OFFSET, p["y"] + GRID_ORIGIN_OFFSET)


class HandleRuleTest(unittest.TestCase):
    """Rebuild sample rows from their waypoints alone."""

    def assertPointClose(self, got: dict, want: dict, tol: float):
        self.assertLess(math.hypot(got["x"] - want["x"], got["y"] - want["y"]), tol, (got, want))

    def _rebuild(self, asset, op, corner_angle_deg):
        rules = {"width": op["path"]["width"], "height": op["height"], "spacing": op["spacing"],
                 "spacingRule": op["spacingRule"], "flexibilityRule": op["flexibilityRule"],
                 "heightRule": op["heightRule"], "hasCurves": op["path"]["hasCurves"]}
        rec = FenceRecord(asset=asset, points=[_local(w["waypoint"]) for w in op["path"]["waypoints"]],
                          closed=op["path"]["state"] == 1, rules=rules, corner_angle_deg=corner_angle_deg)
        return fence_record_to_object_path_v2023(rec)

    def test_open_two_waypoint_rows(self):
        # Every open row except WoodFences (its end was moved after drawing;
        # handles weren't re-derived).
        rows = [(a, op) for a, op in SAMPLE if op["path"]["state"] == 0 and a != WOOD_FENCE_ASSET]
        self.assertGreaterEqual(len(rows), 10)
        for asset, op in rows:
            got = self._rebuild(asset, op, corner_angle_deg=None)
            for g, w in zip(got["path"]["waypoints"], op["path"]["waypoints"]):
                self.assertPointClose(g["pointOne"], w["pointOne"], 2e-3)
                self.assertPointClose(g["pointTwo"], w["pointTwo"], 2e-3)

    def test_closed_retaining_wall_row(self):
        (asset, op), = [(a, op) for a, op in SAMPLE if op["path"]["state"] == 1]
        self.assertEqual(asset, RETAINING_WALL_ASSET)
        got = self._rebuild(asset, op, corner_angle_deg=None)
        self.assertEqual(got["path"]["state"], 1)
        for g, w in zip(got["path"]["waypoints"], op["path"]["waypoints"]):
            self.assertPointClose(g["pointOne"], w["pointOne"], 2e-3)
            self.assertPointClose(g["pointTwo"], w["pointTwo"], 2e-3)


class BuilderTest(unittest.TestCase):
    STRAIGHT = [(100.0, 100.0), (130.0, 100.0)]
    CURVED = [(200 + 20 * math.cos(t / 10), 200 + 20 * math.sin(t / 10)) for t in range(0, 16)]  # ~90 deg arc
    SQUARE = [(300.0, 300.0), (320.0, 300.0), (320.0, 320.0), (300.0, 320.0), (300.0, 300.0)]

    def _build(self, features):
        records, warnings = build_fence_records(features)
        return records, warnings, [fence_record_to_object_path_v2023(r) for r in records]

    def test_structure_matches_sample(self):
        _, _, paths = self._build([
            _way(self.STRAIGHT, {"barrier": "fence"}, 1),
            _way(self.CURVED, {"barrier": "hedge"}, 2, "hedge"),
            _way(self.SQUARE, {"barrier": "wall"}, 3, "wall"),
        ])
        want = _shape(SAMPLE[0][1])
        for p in paths:
            self.assertEqual(_shape(p), want)
            self.assertEqual(list(p), list(SAMPLE[0][1]))            # key order
            self.assertEqual(list(p["path"]), list(SAMPLE[0][1]["path"]))
            for w in p["path"]["waypoints"]:
                self.assertEqual(set(w["waypoint"]), {"x", "y"})   # 2D

    def test_straight_open(self):
        recs, _, (p,) = self._build([_way(self.STRAIGHT, {"barrier": "fence"})])
        self.assertEqual(recs[0].asset, WOOD_FENCE_ASSET)
        wps = p["path"]["waypoints"]
        self.assertEqual(p["path"]["state"], 0)
        self.assertEqual(wps[0]["pointOne"], wps[0]["waypoint"])
        self.assertEqual(wps[-1]["pointTwo"], wps[-1]["waypoint"])
        self.assertAlmostEqual(wps[0]["pointTwo"]["x"] - wps[0]["waypoint"]["x"], 7.5, places=3)
        self.assertEqual(wps[0]["waypoint"], {"x": 100.0 - GRID_ORIGIN_OFFSET, "y": 100.0 - GRID_ORIGIN_OFFSET})

    def test_curved_open_has_smooth_handles(self):
        _, _, (p,) = self._build([_way(self.CURVED, {"barrier": "hedge"}, kind="hedge")])
        wps = p["path"]["waypoints"]
        self.assertGreater(len(wps), 2)
        self.assertTrue(p["path"]["hasCurves"])
        for w in wps[1:-1]:
            self.assertNotEqual(w["pointOne"], w["waypoint"])
            # smooth: handles and waypoint collinear (tangent line)
            a, b, c = w["pointOne"], w["waypoint"], w["pointTwo"]
            cross = (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])
            self.assertAlmostEqual(cross, 0.0, places=2)

    def test_closed_square_keeps_corners(self):
        recs, _, (p,) = self._build([_way(self.SQUARE, {"barrier": "wall"}, kind="wall")])
        self.assertTrue(recs[0].closed)
        self.assertEqual(p["path"]["state"], 1)
        wps = p["path"]["waypoints"]
        self.assertEqual(len(wps), 4)  # closing point not repeated
        for w in wps:  # corner handles lie on their own segments -> axis-aligned offsets
            for h in (w["pointOne"], w["pointTwo"]):
                self.assertTrue(math.isclose(h["x"], w["waypoint"]["x"], abs_tol=1e-6)
                                or math.isclose(h["y"], w["waypoint"]["y"], abs_tol=1e-6))

    def test_shared_corner_joins_same_style(self):
        # An L drawn as two ways (second reversed) -> one run, corner node once.
        recs, _, _ = self._build([
            _way([(0, 0), (10, 0)], {"barrier": "fence"}, 1),
            _way([(10, 10), (10, 0.2)], {"barrier": "fence"}, 2),
        ])
        self.assertEqual(len(recs), 1)
        self.assertEqual(len(recs[0].points), 3)
        self.assertEqual(recs[0].source_ids, [1, 2])

    def test_ways_forming_a_loop_close(self):
        recs, _, _ = self._build([
            _way(self.SQUARE[:3], {"barrier": "fence"}, 1),
            _way(self.SQUARE[2:], {"barrier": "fence"}, 2),
        ])
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0].closed)
        self.assertEqual(len(recs[0].points), 4)

    def test_t_junction_and_style_change_stay_separate(self):
        recs, _, _ = self._build([
            _way([(0, 0), (10, 0)], {"barrier": "fence"}, 1),
            _way([(10, 0), (20, 0)], {"barrier": "fence"}, 2),
            _way([(10, 0), (10, 10)], {"barrier": "fence"}, 3),
        ])
        self.assertEqual(len(recs), 3)
        recs, _, _ = self._build([
            _way([(0, 0), (10, 0)], {"barrier": "fence"}, 1),
            _way([(10, 0), (20, 0)], {"barrier": "fence", "fence_type": "chain_link"}, 2),
        ])
        self.assertEqual(sorted(r.asset for r in recs), sorted([WOOD_FENCE_ASSET, WIRE_FENCE_ASSET]))

    def test_multi_part_type_one_record_per_part(self):
        # Picket = pickets + posts over the same run; a 2-way L still joins first.
        recs, w, paths = self._build([
            _way([(0, 0), (10, 0)], {"barrier": "fence", "fence_type": "picket"}, 1),
            _way([(10, 0), (10, 10)], {"barrier": "fence", "fence_type": "picket"}, 2),
        ])
        self.assertEqual(w, [])
        self.assertEqual({r.asset for r in recs}, {PICKET_PICKETS_ASSET, PICKET_POSTS_ASSET})
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].points, recs[1].points)
        self.assertEqual({r.fence_type for r in recs}, {"picket"})
        self.assertEqual(paths[0]["path"]["waypoints"], paths[1]["path"]["waypoints"])
        self.assertEqual(sorted(p["spacing"] for p in paths), [0.1, 4.0])
        groups = fence_records_to_groups_v2023(recs)
        self.assertEqual(len(groups), 2)

    def test_groups_and_roundtrip(self):
        recs, _, _ = self._build([
            _way(self.STRAIGHT, {"barrier": "fence"}, 1),
            _way([(0, 50), (20, 50)], {"barrier": "fence"}, 2),
            _way(self.SQUARE, {"barrier": "wall"}, 3, "wall"),
        ])
        groups = fence_records_to_groups_v2023(recs)
        self.assertEqual({g["Key"]["path"]: len(g["Value"]["objectPaths"]) for g in groups},
                         {WOOD_FENCE_ASSET: 2, STONE_WALL_ASSET: 1})
        with tempfile.TemporaryDirectory() as tmp:
            save_fence_records(recs, Path(tmp) / "fences.json")
            again = load_fence_records(Path(tmp) / "fences.json")
        self.assertEqual(fence_records_to_groups_v2023(again), groups)


class LeveledHeightTest(unittest.TestCase):
    """heightRule=1: one height per run = lowest ground + shift + offset."""

    @staticmethod
    def _rec(points, heightRule=1, height=0.0, closed=False):
        rules = {"width": 4.0, "height": height, "spacing": 4.0, "spacingRule": 2,
                 "flexibilityRule": 1, "heightRule": heightRule, "hasCurves": True}
        return FenceRecord(asset=WOOD_FENCE_ASSET, points=list(points), closed=closed, rules=rules)

    def test_slope_takes_minimum_plus_shift_and_offset(self):
        slope = lambda x, z: 10.0 + 0.5 * x  # noqa: E731
        (r,) = apply_leveled_heights([self._rec([(0, 0), (20, 0)], height=0.25)], slope, 3.0)
        self.assertAlmostEqual(r.rules["height"], 10.0 + 3.0 + 0.25)

    def test_dip_between_waypoints_counts(self):
        dip = lambda x, z: 5.0 - (2.0 if 9.5 <= x <= 10.5 else 0.0)  # noqa: E731
        (r,) = apply_leveled_heights([self._rec([(0, 0), (20, 0)])], dip)
        self.assertAlmostEqual(r.rules["height"], 3.0)

    def test_closed_run_includes_wrap_segment(self):
        # Low ground only on the closing edge (0,10)->(0,0), x == 0, 0 < z < 10.
        ground = lambda x, z: 1.0 if (x == 0 and 0 < z < 10) else 8.0  # noqa: E731
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        (closed,) = apply_leveled_heights([self._rec(square, closed=True)], ground)
        (opened,) = apply_leveled_heights([self._rec(square)], ground)
        self.assertEqual((closed.rules["height"], opened.rules["height"]), (1.0, 8.0))

    def test_contoured_untouched_and_input_not_mutated(self):
        rec = self._rec([(0, 0), (5, 0)], heightRule=0, height=-1.5)
        leveled = self._rec([(0, 0), (5, 0)], height=0.5)
        out = apply_leveled_heights([rec, leveled], lambda x, z: 7.0, 1.0)
        self.assertEqual(out[0].rules["height"], -1.5)
        self.assertEqual(leveled.rules["height"], 0.5)

    def test_multi_part_parts_share_height(self):
        recs, _ = build_fence_records([_way([(0, 0), (10, 5)], {
            "fence_type": "picket", "pga_fence_heightRule": "1"})])
        out = apply_leveled_heights(recs, lambda x, z: 2.0 + x)
        self.assertEqual(len({r.rules["height"] for r in out}), 1)
        self.assertEqual(out[0].rules["height"], 2.0)


class StyleTest(unittest.TestCase):
    @staticmethod
    def _one(kind, tags):
        """(asset, rules, warnings) for a single-part style."""
        style, w = resolve_fence_style(kind, tags)
        (asset, rules), = style.parts
        return asset, rules, w

    def test_asset_defaults_match_sample(self):
        _, rules, w = self._one("wall", {"barrier": "wall", "pga_fence_asset": "asian_green_cap"})
        self.assertEqual((rules["spacingRule"], rules["flexibilityRule"], rules["hasCurves"]), (0, 0, False))
        self.assertEqual(w, [])

    def test_presets(self):
        for preset, asset, height in (("curb", STONE_WALL_ASSET, -1.498),
                                      ("railroad", CANVAS_BLACK_ASSET, -2.688),
                                      ("retaining_wall", RETAINING_WALL_ASSET, -1.015)):
            got_asset, rules, w = self._one("fence", {"barrier": "fence", "pga_fence_preset": preset})
            self.assertEqual((got_asset, rules["height"]), (asset, height))
            self.assertEqual(w, [])
        self.assertEqual(self._one("wall", {"pga_fence_preset": "retaining_wall"})[1]["width"], 2.5)

    def test_field_overrides_and_asset_override(self):
        # pga_fence_asset takes a type name, a menu label, or a part asset label.
        for requested in ("asian_green_cap", "Brick wall - Asian green cap", "Asia_BrickWalls_PostPrefab"):
            asset, rules, w = self._one("fence", {
                "pga_fence_asset": requested, "pga_fence_spacingRule": "3",
                "pga_fence_hasCurves": "yes", "pga_fence_height": "3.23", "pga_fence_heightRule": "1"})
            self.assertEqual(asset, BRICK_WALL_ASSET)
            self.assertEqual((rules["spacingRule"], rules["hasCurves"], rules["height"], rules["heightRule"]),
                             (3, True, 3.23, 1))
            self.assertEqual(w, [])

    def test_multi_part_overrides(self):
        # Shared fields (height) hit every part; the rest only the first.
        style, w = resolve_fence_style("fence", {"pga_fence_asset": "brick_with_railings",
                                                 "pga_fence_height": "-0.3", "pga_fence_spacing": "6"})
        self.assertEqual(w, [])
        (a1, r1), (a2, r2) = style.parts
        self.assertEqual((a1, a2), (BRICK_LOW_RAILS_ASSET, LOW_METAL_RAILING_ASSET))
        self.assertEqual((r1["height"], r2["height"]), (-0.3, -0.3))
        self.assertEqual((r1["spacing"], r2["spacing"]), (6.0, 0.1))
        # Defaults: both parts at the sampler's lowered railing height.
        style, _ = resolve_fence_style("fence", {"material": "brick_with_railings"})
        self.assertEqual({r["height"] for _, r in style.parts}, {-0.594})

    def test_every_menu_type_resolves_clean(self):
        self.assertEqual(len(FENCE_TYPES), 28)
        for name in FENCE_TYPES:
            style, w = resolve_fence_style("fence", {"pga_fence_asset": name})
            self.assertEqual((style.type_name, w), (name, []))

    def test_matrix_validation(self):
        # Complete set -> invalid value reverts to the default.
        _, rules, w = self._one("fence", {"pga_fence_asset": "Asia_BrickWalls_PostPrefab",
                                          "pga_fence_spacingRule": "7"})
        self.assertEqual(rules["spacingRule"], 0)
        self.assertIn("not a valid option", w[0])
        # Incomplete set -> unseen value kept, flagged unverified (UniFence;
        # WoodFences' spacingRule set is complete since the 2026-09-26 test).
        asset, rules, w = self._one("fence", {"barrier": "chain", "pga_fence_spacingRule": "3"})
        self.assertEqual(asset, UNI_FENCE_ASSET)
        self.assertEqual(rules["spacingRule"], 3)
        self.assertIn("unverified", w[0])
        # Bad input -> warned and ignored.
        _, rules, w = self._one("fence", {"pga_fence_preset": "moat", "pga_fence_spacing": "wide",
                                          "pga_fence_asset": "Nope"})
        self.assertEqual(len(w), 3)
        self.assertEqual(rules["spacing"], 4.0)

    def test_warnings_carry_way_id(self):
        _, warnings = build_fence_records([_way([(0, 0), (5, 0)], {"pga_fence_preset": "moat"}, 42)])
        self.assertTrue(warnings[0].startswith("way 42:"))


if __name__ == "__main__":
    unittest.main()
