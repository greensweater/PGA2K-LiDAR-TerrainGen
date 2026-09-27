"""
Tests for course_output/splines.py V-handle smoothing -- stdlib
unittest, run with:  python -m unittest tests.test_splines -v

Green/fairway waypoints whose handles form a little "V" (renders as a
sharp corner in-game) must come out smooth; handles that are already
collinear, or deliberately sharpened along the path, are left alone.
"""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from course_output.splines import (  # noqa: E402
    SMOOTH_HANDLE_RATIO, _STATIC_SPLINE_PARAMS, _build_spline, _smooth_v_handles, feature_to_spline,
)
from course_output.userLayers import GRID_ORIGIN_OFFSET  # noqa: E402
from ingest.osm import Feature  # noqa: E402


def _dense_blob() -> Polygon:
    # Irregularly-noded OSM-like outline: ~40 m ellipse, 1-4 m spacing.
    pts = []
    a = 0.0
    step = 0
    while a < 2 * math.pi - 0.05:
        pts.append((500 + 40 * math.cos(a), 500 + 25 * math.sin(a)))
        a += 0.03 if step % 3 else 0.09
        step += 1
    return Polygon(pts)


def _vec(a, b):
    return b["x"] - a["x"], b["y"] - a["y"]


def _cos(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))


def _wp(c, h1, h2):
    return {"waypoint": {"x": c[0], "y": c[1]},
            "pointOne": {"x": h1[0], "y": h1[1]},
            "pointTwo": {"x": h2[0], "y": h2[1]}}


def _unsmoothed(kind: str) -> dict:
    params = dict(_STATIC_SPLINE_PARAMS[kind])
    params.pop("smooth_v_handles", None)
    pts = [(x - GRID_ORIGIN_OFFSET, z - GRID_ORIGIN_OFFSET)
           for x, z in _dense_blob().exterior.coords[:-1]]
    return _build_spline(pts, **params)


class SmoothVHandleTests(unittest.TestCase):
    def test_green_v_handles_smoothed(self):
        # Chad's tight green handles are a V at every node -> all fixed.
        self.assertGreater(_cos(*(
            _vec(w["waypoint"], w[k]) for w in [_unsmoothed("green")["waypoints"][5]]
            for k in ("pointOne", "pointTwo"))), 0.5)
        wps = feature_to_spline(Feature(geometry=_dense_blob(), kind="green", tags={}))["waypoints"]
        n = len(wps)
        for i, wp in enumerate(wps):
            c = wp["waypoint"]
            b, f = _vec(c, wp["pointOne"]), _vec(c, wp["pointTwo"])
            self.assertLess(_cos(b, f), -0.999, f"waypoint {i} still a V")
            prev_len = math.hypot(*_vec(wps[i - 1]["waypoint"], c))
            next_len = math.hypot(*_vec(c, wps[(i + 1) % n]["waypoint"]))
            self.assertAlmostEqual(math.hypot(*b), prev_len * SMOOTH_HANDLE_RATIO, delta=0.01)
            self.assertAlmostEqual(math.hypot(*f), next_len * SMOOTH_HANDLE_RATIO, delta=0.01)

    def test_fairway_collinear_left_alone(self):
        # Loose fairway handles are already collinear -> untouched.
        smoothed = feature_to_spline(Feature(geometry=_dense_blob(), kind="fairway", tags={}))
        self.assertEqual(smoothed["waypoints"], _unsmoothed("fairway")["waypoints"])

    def test_sharpened_left_alone(self):
        # Right-angle corner, each handle pointing along its own segment.
        wps = [
            _wp((0, 0), (0, 0), (1, 0)),
            _wp((10, 0), (9, 0), (10, 1)),
            _wp((10, 10), (10, 9), (10, 10)),
        ]
        before = copy.deepcopy(wps)
        self.assertEqual(_smooth_v_handles(wps, is_closed=False), 0)
        self.assertEqual(wps, before)

    def test_single_v_fixed(self):
        # Middle point has both handles pointing "down" -> a V.
        wps = [
            _wp((0, 0), (0, 0), (0, 0)),
            _wp((10, 0), (9.9, -0.5), (10.1, -0.5)),
            _wp((20, 0), (20, 0), (20, 0)),
        ]
        self.assertEqual(_smooth_v_handles(wps, is_closed=False), 1)
        self.assertEqual(wps[1]["pointOne"], {"x": 6.25, "y": 0.0})
        self.assertEqual(wps[1]["pointTwo"], {"x": 13.75, "y": 0.0})


if __name__ == "__main__":
    unittest.main()
