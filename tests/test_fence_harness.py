"""
Tests for the push-fence-test layout (course_output/fences.py
build_fence_test_layout) and the game-safe course filename rule
(PGA2k_gen.game_safe_course_stem, V2023_TASKS.md 3.3a) -- stdlib unittest:
    python -m unittest tests.test_fence_harness -v
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "util"))

from course_extract import extract_course_file  # noqa: E402
from course_output.fences import (  # noqa: E402
    ASSET_DEFAULT_RULES, FENCE_PRESETS, FENCE_TEST_ORIGIN, build_fence_test_layout,
    fence_records_to_groups_v2023,
)
from course_output.userLayers import GRID_ORIGIN_OFFSET  # noqa: E402
from PGA2k_gen import game_safe_course_stem  # noqa: E402


def _sample_groups() -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        extract_course_file(str(REPO / "templates" / "2023_fences.course"), tmp)
        return json.loads((Path(tmp) / "CourseDescription_nodes" / "placedObjects3.json").read_text())


SAMPLE = _sample_groups()


def _waypoints(groups):
    return [(g["Key"]["path"], wp) for g in groups for op in g["Value"]["objectPaths"]
            for wp in op["path"]["waypoints"]]


class FenceTestLayoutTests(unittest.TestCase):
    def setUp(self):
        self.layout = build_fence_test_layout(SAMPLE, datum=3.23)
        self.ours = fence_records_to_groups_v2023(self.layout.records)

    def row(self, label):
        return self.layout.records[self.layout.labels.index(label)]

    def test_follow_up_rows(self):
        e1, e2 = self.row("E1"), self.row("E2")
        self.assertEqual((e1.rules["heightRule"], e1.rules["height"]), (1, 3.23))
        self.assertEqual((e2.rules["heightRule"], e2.rules["height"]), (1, 6.23))
        e3, e4 = self.row("E3"), self.row("E4")
        self.assertEqual((e3.rules["spacingRule"], e4.rules["spacingRule"]), (1, 3))
        self.assertEqual(e3.points, [(x, y + 15.0) for x, y in e4.points])  # same zigzag, one row apart
        self.assertEqual(len(e3.points), 4)
        self.assertEqual((self.row("E5").rules["hasCurves"], self.row("E6").rules["hasCurves"]), (True, False))

    def test_every_asset_and_preset_has_a_row(self):
        assets = {r.asset for r in self.layout.records}
        self.assertEqual(assets, set(ASSET_DEFAULT_RULES))
        legend = "\n".join(self.layout.legend)
        for name in FENCE_PRESETS:
            self.assertIn(f"preset {name}", legend)

    def test_everything_near_origin_and_inside_grid(self):
        x0, y0 = FENCE_TEST_ORIGIN
        for _, wp in _waypoints(self.ours + self.layout.control_groups):
            for key in ("pointOne", "pointTwo", "waypoint"):
                x, y = wp[key]["x"], wp[key]["y"]
                self.assertLess(math.hypot(x - x0, y - y0), 500.0)
                self.assertLess(max(abs(x), abs(y)), GRID_ORIGIN_OFFSET)
        (px, py), h = self.layout.pad_center, self.layout.pad_half
        self.assertLess(max(abs(px) + h, abs(py) + h), GRID_ORIGIN_OFFSET)

    def test_off_map_origin_raises(self):
        with self.assertRaises(ValueError):
            build_fence_test_layout(SAMPLE, (-1900.0, 1900.0))

    def test_control_rows_are_sample_rows_translated(self):
        orig = _waypoints([g for g in SAMPLE if (g.get("Value") or {}).get("objectPaths")])
        moved = _waypoints(self.layout.control_groups)
        self.assertEqual(len(orig), len(moved))
        (_, a0), (_, b0) = orig[0], moved[0]
        dx, dy = b0["waypoint"]["x"] - a0["waypoint"]["x"], b0["waypoint"]["y"] - a0["waypoint"]["y"]
        for (asset_a, a), (asset_b, b) in zip(orig, moved):
            self.assertEqual(asset_a, asset_b)
            for key in ("pointOne", "pointTwo", "waypoint"):
                self.assertAlmostEqual(b[key]["x"], a[key]["x"] + dx, places=3)
                self.assertAlmostEqual(b[key]["y"], a[key]["y"] + dy, places=3)
        # Rule fields are untouched (no builder in the loop).
        ops_a = [op for g in SAMPLE for op in (g.get("Value") or {}).get("objectPaths") or []]
        ops_b = [op for g in self.layout.control_groups for op in g["Value"]["objectPaths"]]
        strip = lambda op: {k: v for k, v in op.items() if k != "path"}  # noqa: E731
        self.assertEqual([strip(o) for o in ops_a], [strip(o) for o in ops_b])

    def test_burial_rows_sit_on_and_across_the_pad(self):
        (px, py), h = self.layout.pad_center, self.layout.pad_half
        on_pad, lifted, crossing = (self.row(k) for k in ("D1", "D2", "D3"))
        for rec in (on_pad, lifted, crossing):
            xs = [x - GRID_ORIGIN_OFFSET for x, _ in rec.points]
            ys = [z - GRID_ORIGIN_OFFSET for _, z in rec.points]
            self.assertTrue(all(abs(y - py) < h for y in ys))
            if rec is crossing:
                self.assertTrue(min(xs) < px - h and max(xs) > px + h)
            else:
                self.assertTrue(all(abs(x - px) < h for x in xs))
        self.assertEqual(lifted.rules["height"], self.layout.pad_lift)
        self.assertEqual(on_pad.rules["height"], 0.0)

    def test_orientation_marker_arms(self):
        (ax, ay), (cx, cy), (bx, by) = self.row("C").points
        self.assertEqual(ax, cx)
        self.assertGreater(ay, cy)          # short arm toward +y
        self.assertEqual(by, cy)
        self.assertGreater(bx - cx, ay - cy)  # long arm toward +x


class GameSafeCourseStemTests(unittest.TestCase):
    # Every filename the in-game bisection proved fails / loads (3.3a).
    FAILED = ["FT2-1-S-compact", "FT3-1-gamefmt-fileid", "FenceTest-A-bc-nofences",
              "LIDAR-2023-bouldercreek-fences", "BY1-metadata-spaced", "BZ1-shawnee-good",
              "BZ4-shawnee-name"]
    LOADED = ["BX1-shawnee-level1", "BZ2-shawnee-good2", "BZ3-shawnee-3",
              "LIDAR-2023-20260925163935", "2023_fences", "hinckleyhills", "hhills3"]
    BAD_TAIL = re.compile(r"-[A-Za-z]+$")

    def test_failing_names_become_safe(self):
        for stem in self.FAILED:
            self.assertRegex(stem, self.BAD_TAIL)
            self.assertNotRegex(game_safe_course_stem(stem), self.BAD_TAIL, stem)

    def test_hyphens_replaced_rest_kept(self):
        self.assertEqual(game_safe_course_stem("LIDAR-2023-bouldercreek-fences"),
                         "LIDAR_2023_bouldercreek_fences")
        for stem in ("2023_fences", "hinckleyhills", "hhills3"):
            self.assertEqual(game_safe_course_stem(stem), stem)

    def test_loaded_names_stay_loadable(self):
        for stem in self.LOADED:
            self.assertNotRegex(game_safe_course_stem(stem), self.BAD_TAIL, stem)


if __name__ == "__main__":
    unittest.main()
