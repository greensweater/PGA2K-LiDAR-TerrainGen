"""
constants.py

Project-wide constants for the PGA2K Terrain Compiler.

All distances are in meters unless otherwise noted.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Course geometry
# ---------------------------------------------------------------------------

COURSE_SIZE_M = 2000.0
DEFAULT_HEX_RADIUS_M = 100.0
MIN_STAMP_RADIUS_M = 1.0

# Initial optimization parameters
INITIAL_BRUSH = 10

# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

LIDAR_QUERY_RADIUS_M = 125.0

# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------

DEBUG_IMAGE_SIZE = 2000  # pixels
DEBUG_BACKGROUND = (0, 0, 0)

# ---------------------------------------------------------------------------
# PGA terrain
# ---------------------------------------------------------------------------

DEFAULT_ROTATION_DEG = 0.0

# Internal units are always meters.
# The writer module is solely responsible for converting to PGA JSON.
HEIGHT_SCALE = 1.0

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

EPSILON = 1.0e-9

# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------

PROJECT_FILE = "project.json"
POINTCLOUD_FILE = "pointcloud.npz"

PREVIEW_LIDAR = "preview_lidar.png"
PREVIEW_HEX = "preview_hex.png"
PREVIEW_HEIGHT = "preview_height.png"
PREVIEW_ERROR = "preview_error.png"
PREVIEW_STAMPS = "preview_stamps.png"
