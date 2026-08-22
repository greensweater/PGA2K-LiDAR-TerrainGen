#!/usr/bin/env python3
"""
PGA2k_gen_gui.py

Minimal desktop GUI wrapping PGA2k_gen.py's steps as buttons: set a
working directory, click a step, watch its output stream in, see
whatever diagnostic preview it produced.

Deliberately thin: every pipeline step runs PGA2k_gen.py as a
subprocess with the exact same arguments the CLI takes, rather than
re-implementing or calling into its internals directly. That means
there's exactly one place pipeline behavior lives -- this GUI can't
drift out of sync with the CLI, and anything that works from the
command line works here. (The one exception is "Copy to Game Folder",
which is a plain file copy with no pipeline logic of its own -- see
that section below.)

Requires: tkinter (stdlib) + Pillow (for preview images -- the GUI
still works without Pillow, previews just won't render).
"""

from __future__ import annotations

import json
import os
import platform
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = SCRIPT_DIR / "PGA2k_gen.py"

# Reused directly rather than duplicated -- these are plain, side-effect-free
# JSON helpers already tested as part of the CLI (see PGA2k_gen.py).
sys.path.insert(0, str(SCRIPT_DIR))
from constants import (  # noqa: E402
    COURSE_SIZE_M, PREVIEW_COMPOSITE, PREVIEW_DIR, PREVIEW_ERROR, PREVIEW_HEIGHT, PREVIEW_HEX,
    PREVIEW_LIDAR, PREVIEW_LIDAR_GROUND, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
    PREVIEW_STAMPS, STAMPS_DIR,
)
from PGA2k_gen import (  # noqa: E402
    DEFAULT_DIG_WATER_BUFFER_M, DEFAULT_DIG_WATER_DEPTH_M, FEATURES_FILE, HEIGHT_MASK_FILE,
    HEIGHTMAP_FILE, OBJECT_LIST_FILE, PLACED_OBJECTS_FILE, load_all_stamps, load_project, save_project,
)
from ingest.heightmap import load_heightmap  # noqa: E402
from course_output.objects import (  # noqa: E402
    DEFAULT_GAME_VERSION, GAME_VERSIONS, IMPLEMENTED_GAME_VERSIONS, THEMES_V2019, TREE_HEIGHT_TAG,
    TREE_RADIUS_TAG, TREE_TYPE_TAG, load_object_list, load_placed_objects, save_object_list,
)
from course_output.asset_catalog import (  # noqa: E402
    ASSET_CATEGORIES, ASSET_ENTRIES, CLUSTERABLE_ENTRIES, NATURE_CATEGORY_IDS,
)
from course_output.object_clusters import (  # noqa: E402
    CLUSTER_FILL_SOURCE_BORDER, CLUSTER_FILL_SOURCE_MANUAL, DEFAULT_RASTER_RATIO,
    PGA_CLUSTER_FILLS_TAG, SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND, build_border_ring_geometry,
)
from course_output.userLayers import GRID_ORIGIN_OFFSET  # noqa: E402
from ingest.osm import (  # noqa: E402
    DEFAULT_HOLE_CORRIDOR_BUFFER_PX, Feature, build_height_mask, crop_features,
    merge_height_mask_features, load_features,
    rasterize_mask_rgba, save_features, save_height_mask, shift_features,
)
from terrain.bounding_box import BoundingBox  # noqa: E402
from terrain.hexgrid import (  # noqa: E402
    DEFAULT_BRUSH as HEX_DEFAULT_BRUSH,
    HEX_DEFAULT_SPREAD_RATIO,
    HEX_LATTICE_PITCH_M,
)
from terrain.rastergrid import (  # noqa: E402
    RASTER_SIZES,
    RASTER_BRUSH as DEFAULT_RASTER_BRUSH,
    DEFAULT_RASTER_SIZE,
    DEFAULT_RASTER_SPREAD_RATIO,
    DEFAULT_RASTER_CENTER_BIAS_RATIO,
)
from terrain.cart_paths import (  # noqa: E402
    CART_PATH_STAMP_RADIUS, CART_PATH_SPACING_M, CART_PATH_HEIGHT_AVG_RADIUS_M,
)
from shapely.ops import unary_union  # noqa: E402
import viz.visualize as viz  # noqa: E402

try:
    # Same interface adaptive_refine.py already relies on for real
    # scoring (TerrainKernel(BRUSH_PROFILES[brush]).sample_many(r_norm)) --
    # reused here, not reimplemented, so the influence overlay reflects
    # the SAME kernel the actual pipeline evaluates terrain with. Wrapped
    # defensively since this GUI file hasn't independently verified
    # terrain_kernel.py's exact API beyond that established call pattern;
    # a mismatch here should degrade to the plain coverage overlay, not
    # crash preview rendering entirely.
    from terrain.brush_profiles import BRUSH_PROFILES, SHAPE_SQUARE  # noqa: E402
    from terrain.terrain_kernel import TerrainKernel  # noqa: E402
    from terrain.stamp import local_square_offsets  # noqa: E402
    _HAVE_TERRAIN_KERNEL = True
except ImportError:
    _HAVE_TERRAIN_KERNEL = False

PREVIEW_FILES = [
    "preview_lidar_heightmap.png",
    "preview_lidar.png",
    "preview_hex.png",
    "preview_stamps.png",
    "preview_error.png",
    "preview_lidar_ground.png",
    "preview_height.png",
    "preview_composite.png",
]

# Game version -> Courses folder name under .../AppData/LocalLow/2K/.
# Windows-specific path (AppData/LocalLow only exists on Windows, which is
# also the only platform The Golf Club / PGA 2K actually runs on) -- only
# 2019 is wired up for now, per the request to add 2K21/23/25 later. Keyed
# by the SAME canonical version strings as objects.py's GAME_VERSIONS
# ("2019", "2021", ...), not a display name -- this is looked up directly
# from the single elevated Game version selector (self.game_version) at
# the top of the window, same value write-objects targets, so "write" and
# "move" (Copy to Game Folder) always agree on which version they mean.
GAME_VERSION_FOLDERS = {
    "2019": "The Golf Club 2019",
}


def _spline_tag_detail(f: Feature) -> str:
    """
    Extra OSM tag value to show alongside a spline's kind in the
    Splines tab list, e.g. distinguishing a broadleaved wood from a
    needleleaved one, or a stream from a ditch -- info classify_way
    collapses into one `kind` but is still on the raw `tags`.
    """
    if f.kind == "wood":
        leaf_type = f.tags.get("leaf_type")
        if leaf_type:
            return leaf_type
    if f.kind == "water":
        waterway = f.tags.get("waterway")
        if waterway:
            return waterway
    return f.tags.get("natural", "")


_ASSET_LABEL_BY_KEY = {(e.category, e.type): e.label for e in ASSET_ENTRIES}

# v2021+ placedObjects2 groups are keyed by {"path": ...} rather than
# v2019's {"category", "type", "theme"} (see course_output/objects.py's
# module docstring) -- this resolves a group's asset path back to its
# asset_catalog.json category, so the "Show objects" preview overlay
# (below) can color either schema's groups the same way.
_ASSET_CATEGORY_BY_PATH = {e.path: e.category for e in ASSET_ENTRIES}

# Category id -> (legend label, RGB) for the Objects tab's "Show objects"
# preview overlay. Only the "nature"/scatterable categories (course_output.
# asset_catalog.NATURE_CATEGORY_IDS) get a color -- building stakes,
# cart-path debug markers, walls/signs/bridges/vehicles etc. have no
# useful preview color and are silently skipped (see
# _get_cached_object_preview_layer). Deliberately dark, desaturated
# tones -- bright colors would be hard to pick out against the OSM
# overlay/heightmap this draws on top of.
_OBJECT_LAYER_STYLE = {
    0: ("Trees", (8, 36, 8)),
    3: ("Ground cover", (30, 74, 30)),
    2: ("Grass", (94, 90, 45)),
    12: ("Display plants", (110, 24, 24)),
    1: ("Rocks", (68, 68, 68)),
}

# Draw order for the "Show objects" overlay -- LOW to HIGH, i.e. this
# index is the position in the stack (later entries drawn on top of
# earlier ones), not visual/z-height. Modeled on real-world stacking by
# how tall each category typically stands off the ground: low ground
# cover/rocks/grass first, trees (by far the tallest) drawn dead last so
# a tree marker is never buried under a shorter category's circle.
# Buildings (category 5) aren't in _OBJECT_LAYER_STYLE yet -- no building
# placement data currently feeds this overlay -- but would slot in just
# before trees (second-to-last) if that ever changes.
_OBJECT_LAYER_DRAW_ORDER = (1, 2, 3, 12, 0)  # rocks, grass, ground cover, display plants, trees


def _spline_cluster_detail(f: Feature) -> str:
    """
    Comma-joined asset labels for whatever's currently in
    f.tags[PGA_CLUSTER_FILLS_TAG] (see course_output/object_clusters.py),
    shown in the Splines tab's "Clusters" column -- "" if untagged. A
    spec that no longer resolves (stale tag after asset_catalog.json
    changed) shows as "?" rather than being silently dropped, so it's
    still visible that *something* is tagged there.
    """
    specs = f.tags.get(PGA_CLUSTER_FILLS_TAG)
    if not specs:
        return ""
    return ", ".join(_ASSET_LABEL_BY_KEY.get((s.get("category"), s.get("type")), "?") for s in specs)


class _Tooltip:
    """Minimal hover tooltip: shows `text` near the widget on mouse-enter."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, font=("TkDefaultFont", 8), wraplength=240,
        )
        label.pack(ipadx=4, ipady=2)

    def _hide(self, _event=None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class PGAGenGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PGA2K Terrain Compiler")
        root.geometry("1200x760")

        self.working_dir = tk.StringVar()
        self.course_name = tk.StringVar()
        self.game_version = tk.StringVar(value=DEFAULT_GAME_VERSION)
        self.log_queue: queue.Queue = queue.Queue()
        self.running = False
        self._current_proc = None  # see _stop_current_step
        self._stop_requested = False
        self._step_start_time = 0.0
        self._preview_imgtk = None  # keep a reference so tkinter doesn't GC it
        self._cached_mask_merged_geom = None  # see _get_cached_mask_merged_geometry
        self._cached_mask_geom_key = None
        self._cached_composited_base = None  # see _show_preview's static-part cache (zoom-independent)
        self._cached_composited_base_key = None
        self._cached_geo_overlay = None  # see _show_preview's geo-overlay cache (mask buffer/spline highlight, also zoom-independent)
        self._cached_geo_overlay_key = None
        self._splines_features = []  # loaded features.geojson content, for the Splines tab
        self._splines_features_mtime = None  # see _ensure_splines_features_fresh
        self._objects_tree_list = []  # loaded object_list.json content, for the Objects tab
        self._cluster_fill_rows = []  # (spline_osm_ids_str, asset_label, ratio, source) -- see _build_cluster_fill_rows
        self._highlighted_feature_osm_ids = set()  # currently-selected spline(s), if any, to highlight on the preview
        self._highlighted_object_points = []  # (x, z) of currently-selected individual object/tree row(s), for a preview ring
        self._highlighted_object_group_spline_ids = set()  # currently-selected cluster-fill row(s)' source spline osm_ids, unioned into the spline highlight overlay
        self._selection_preview_job = None  # see _on_spline_selected's debounce
        self._splines_selection_memory: set[str] = set()  # see _splines_memory_store/_recall
        self._suppress_course_name_save = False
        self._suppress_repack_filename_save = False
        self._suppress_game_version_save = False
        self._suppress_objects_theme_save = False

        self._build_layout()
        self._poll_log_queue()

        self.working_dir.trace_add("write", lambda *a: self._on_working_dir_changed())
        self.course_name.trace_add("write", lambda *a: self._on_course_name_changed())
        self.game_version.trace_add("write", lambda *a: self._on_game_version_changed())
        self.objects_theme_var.trace_add("write", lambda *a: self._on_objects_theme_changed())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        footer = ttk.Frame(self.root, padding=(8, 4))
        footer.pack(side="bottom", fill="x")
        self._build_footer(footer)

        style = ttk.Style()
        style.configure("Thin.Vertical.TScrollbar", width=10)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        # Outer horizontal split: sidebar (Notebook) on the left, the
        # preview/log split on the right -- a draggable sash between
        # them, same resizable-pane pattern as the preview/log split
        # itself (see `right` below), instead of the sidebar's old
        # fixed natural width.
        outer = ttk.PanedWindow(main, orient="horizontal")
        outer.pack(fill="both", expand=True)

        left = ttk.Notebook(outer)

        file_tab = ttk.Frame(left, padding=4)
        terrain_tab = ttk.Frame(left, padding=4)
        splines_tab = ttk.Frame(left, padding=4)
        objects_tab = ttk.Frame(left, padding=4)
        options_tab = ttk.Frame(left, padding=4)
        left.add(file_tab, text="File")
        left.add(terrain_tab, text="Terrain")
        left.add(splines_tab, text="Splines")
        left.add(objects_tab, text="Objects")
        left.add(options_tab, text="Options")

        # Horizontal split: preview (more room, per request) on the left,
        # log on the right; the sash between them resizes width, not height.
        right = ttk.PanedWindow(outer, orient="horizontal")

        outer.add(left, weight=0)
        outer.add(right, weight=1)

        self._build_file_tab(self._make_scrollable_tab(file_tab))
        self._build_terrain_tab(self._make_scrollable_tab(terrain_tab))
        self._build_splines_tab(self._make_scrollable_tab(splines_tab))
        self._build_objects_tab(self._make_scrollable_tab(objects_tab))
        self._build_options_tab(self._make_scrollable_tab(options_tab))
        self._build_preview_panel(right)
        self._build_log_panel(right)

        # Belt-and-suspenders alongside the working-dir-change trace
        # (_on_working_dir_changed already calls _refresh_refine_stats):
        # also refresh whenever the Terrain tab itself becomes the
        # selected tab, so the panel is never stale if something else
        # changed project.json's refine-terrain values while a
        # different tab was showing.
        self._terrain_tab_widget = terrain_tab
        left.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

    def _on_notebook_tab_changed(self, event) -> None:
        notebook = event.widget
        try:
            current = notebook.nametowidget(notebook.select())
        except (tk.TclError, KeyError):
            return
        if current is self._terrain_tab_widget:
            self._refresh_refine_stats()

    def _make_scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        """
        Wrap one Notebook tab in a vertically scrollable canvas, so a
        tab's fields can exceed the visible pane height without
        forcing the whole window to grow. Returns the inner frame
        callers should actually build into.

        Deliberately no mousewheel binding: an earlier version bound
        the wheel globally while the pointer was over the canvas, but
        that's too broad -- it intercepts wheel events meant for a
        widget nested inside the tab (a Treeview's own scrollbar, a
        Combobox's wheel-to-cycle-values behavior) instead of letting
        them reach it. Scrolling is drag-the-scrollbar (or resize the
        pane -- see the outer PanedWindow in _build_layout) only.

        The canvas gets an explicit, deliberately small starting
        width (well under any tab's actual content) rather than
        letting it size itself from that content -- ttk.PanedWindow
        has no "minsize" option on add()/pane() (unlike the old
        tk.PanedWindow), so the sidebar pane's minimum drag width is
        just whatever its content's own natural requested width comes
        out to. Without this, that came out to ~389px (driven by
        whichever tab's fields are widest), so the sash couldn't be
        dragged narrower than that at all. The canvas already crops/
        scrolls its content at any width, so this doesn't lose
        anything -- narrower than the content just means more of it
        needs the (vertical) scrollbar to reach, same as always.
        """
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0, width=200)
        vscroll = ttk.Scrollbar(
            container, orient="vertical", command=canvas.yview, style="Thin.Vertical.TScrollbar",
        )

        def _on_vscroll_set(lo: str, hi: str) -> None:
            # Autohide: yscrollcommand fires with fractions spanning
            # the full 0.0-1.0 range whenever all content already fits
            # in the visible canvas height -- nothing to scroll, so
            # there's nothing useful the scrollbar can do. Packed/
            # forgotten here instead of always-visible; canvas still
            # packs with expand=True, so it reclaims the freed strip
            # automatically when the scrollbar disappears.
            lo_f, hi_f = float(lo), float(hi)
            if lo_f <= 0.0 and hi_f >= 1.0:
                vscroll.pack_forget()
            elif not vscroll.winfo_ismapped():
                vscroll.pack(side="right", fill="y")
            vscroll.set(lo, hi)

        canvas.configure(yscrollcommand=_on_vscroll_set)
        canvas.pack(side="left", fill="both", expand=True)
        # Not packed here -- _on_vscroll_set packs it only once tk
        # reports there's actually something to scroll.

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            # Stretch the inner frame to the canvas's own width, so
            # widgets packed with fill="x" actually fill the visible
            # width instead of just their own natural content width.
            canvas.itemconfig(inner_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        return inner

    def _build_footer(self, parent: ttk.Frame) -> None:
        """
        Full-width, one-line status bar at the bottom of the window,
        outside every tab -- always visible regardless of which tab is
        selected or how far it's scrolled, unlike its previous home
        inside the Terrain tab's own scrolling stack.
        """
        self.status_label = ttk.Label(parent, text="Idle", foreground="gray")
        self.status_label.pack(side="left")
        self.play_sound_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            parent, text="\U0001F514 Sound when done", variable=self.play_sound_var,
        ).pack(side="left", padx=(12, 0))
        self.restart_button = ttk.Button(parent, text="Restart", command=self._restart_app)
        self.restart_button.pack(side="right")
        self.stop_button = ttk.Button(parent, text="Stop", command=self._stop_current_step, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 4))

    def _build_file_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Working directory:").pack(anchor="w")
        wd_row = ttk.Frame(parent)
        wd_row.pack(fill="x", pady=(2, 8))
        ttk.Entry(wd_row, textvariable=self.working_dir, width=26).pack(side="left", fill="x", expand=True)
        ttk.Button(wd_row, text="Browse...", command=self._browse_working_dir).pack(side="left")

        ttk.Label(parent, text="Game version:").pack(anchor="w")
        game_version_box = ttk.Combobox(
            parent, textvariable=self.game_version, state="readonly", width=8, values=list(GAME_VERSIONS),
        )
        game_version_box.pack(anchor="w", pady=(2, 8))
        _Tooltip(game_version_box, "PGA 2K's .course schema diverges across versions -- currently "
                 f"only {IMPLEMENTED_GAME_VERSIONS} are actually implemented (see objects.py's "
                 "module docstring); the others can be selected and saved, but write/repack steps "
                 "will raise a clear error until their schema is confirmed. Project-level, saved "
                 "immediately, used by write-objects and (eventually) write-splines/write-terrain/"
                 "repack.")

        ttk.Label(parent, text="Course name:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.course_name, width=26).pack(anchor="w", fill="x", pady=(2, 8))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Initialize", self._run_init)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.projection_var = tk.StringVar()
        ttk.Label(parent, text="Projection EPSG (optional):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.projection_var, width=14).pack(anchor="w")
        self.fill_heightmap_gaps_var = tk.BooleanVar(value=True)
        fill_gaps_checkbox = ttk.Checkbutton(
            parent, text="Fill heightmap gaps", variable=self.fill_heightmap_gaps_var
        )
        fill_gaps_checkbox.pack(anchor="w")
        _Tooltip(fill_gaps_checkbox, "Fill NaN heightmap gaps (water, buildings, other no-ground-"
                 "point areas) via harmonic inpainting -- iterative neighbor-average relaxation, "
                 "not a single-pass flood-fill. On by default; uncheck to leave gaps as NaN, "
                 "excluded from error scoring/fitting downstream (old behavior).")
        self._add_step_button(parent, "Ingest LAZ", self._run_ingest_laz)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Ingest OSM", self._run_ingest_osm)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        dig_row = ttk.Frame(parent)
        dig_row.pack(fill="x", pady=(0, 4))
        depth_col = ttk.Frame(dig_row)
        depth_col.pack(side="left")
        ttk.Label(depth_col, text="Depth (m):").pack(anchor="w")
        self.dig_water_depth_var = tk.StringVar(value=str(DEFAULT_DIG_WATER_DEPTH_M))
        depth_entry = ttk.Entry(depth_col, textvariable=self.dig_water_depth_var, width=6)
        depth_entry.pack(anchor="w")
        _Tooltip(depth_entry, "How much (m) to lower heightmap.npz under each water polygon.")

        buffer_col = ttk.Frame(dig_row)
        buffer_col.pack(side="left", padx=(12, 0))
        ttk.Label(buffer_col, text="Buffer (m):").pack(anchor="w")
        self.dig_water_buffer_var = tk.StringVar(value=str(DEFAULT_DIG_WATER_BUFFER_M))
        buffer_entry = ttk.Entry(buffer_col, textvariable=self.dig_water_buffer_var, width=6)
        buffer_entry.pack(anchor="w")
        _Tooltip(buffer_entry, "Inward (negative) buffer applied to each water polygon before "
                 "determining which cells to lower -- lets the water plane object (sized from the "
                 "ORIGINAL un-buffered polygon; see Write Water) clip slightly into the "
                 "surrounding terrain at the edges, instead of floating exactly at the rim of a "
                 "perfectly-matching recess with a visible seam.")

        dig_water_btn = self._add_step_button(parent, "Dig Water", self._run_dig_water)
        _Tooltip(dig_water_btn, "Lowers heightmap.npz under every OSM water polygon (needs Ingest LAZ "
                 "and Ingest OSM to have both run). Modifies heightmap.npz IN PLACE -- running this "
                 "twice compounds the dig rather than re-digging to a fixed level; re-run Ingest LAZ "
                 "first for a clean slate if you want to change the depth/buffer after already "
                 "digging once. Meant to run before Generate/Refine Terrain, so the recessed shape is "
                 "already baked into the heightmap those steps fit against -- no water-specific "
                 "terrain logic needed downstream.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.course_file_var = tk.StringVar()
        ttk.Label(parent, text="Course file (.course):").pack(anchor="w")
        course_file_row = ttk.Frame(parent)
        course_file_row.pack(anchor="w", fill="x")
        ttk.Entry(course_file_row, textvariable=self.course_file_var, width=18).pack(side="left")
        ttk.Button(course_file_row, text="...", width=3, command=self._browse_course_file).pack(side="left")
        self._add_step_button(parent, "Ingest Course", self._run_ingest_course)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.repack_filename_var = tk.StringVar()
        self.repack_filename_var.trace_add("write", lambda *a: self._on_repack_filename_changed())
        ttk.Label(parent, text="Repack filename:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.repack_filename_var, width=20).pack(anchor="w")
        self._add_step_button(parent, "Repack", self._run_repack)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Copy to Game Folder", self._run_copy_to_game)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        error_res_row = ttk.Frame(parent)
        error_res_row.pack(fill="x")
        ttk.Label(error_res_row, text="Error resolution:").pack(side="left")
        self.error_resolution_var = tk.StringVar(value="")
        error_res_entry = ttk.Entry(error_res_row, textvariable=self.error_resolution_var, width=6)
        error_res_entry.pack(side="left", padx=4)
        _Tooltip(error_res_entry, "preview_error.png's own grid resolution. Left blank, it inherits "
                 "whatever --resolution refine-terrain last used, or a hardcoded 200 if refine-terrain "
                 "hasn't run at all yet -- far too coarse to localize a specific small feature (at "
                 "RES~1000, a 200x200 error grid averages ~5x5 real cells into one). Set this directly "
                 "to decouple it from refine-terrain entirely. Not saved to project.json -- costs more "
                 "at higher values (evaluated at every cell), so start moderate (e.g. 500) before "
                 "jumping to 1000+.")
        self._add_step_button(parent, "Visualize", self._run_visualize)

    def _build_terrain_tab(self, parent: ttk.Frame) -> None:
        self._build_generate_section(parent)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._build_refine_section(parent)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._build_terrain_preview_overlays(parent)

    # ------------------------------------------------------------------
    # Live preview overlays -- elevation contour / stamp influence. Read
    # heightmap.npz / stamps directly and composite purely in-memory (see
    # _show_preview's compositing block), so they live here next to the
    # Generate/Refine settings they're meant to be eyeballed against.
    # ------------------------------------------------------------------

    def _build_terrain_preview_overlays(self, parent: ttk.Frame) -> None:
        # Live elevation-band overlay -- purely in-memory (see
        # _show_preview's compositing block below and _get_cached_
        # heightmap): reads heightmap.npz directly, thresholds it into
        # a 1-bit [elevation, elevation+width) band, and composites
        # that on top of whatever preview is currently showing. Never
        # writes anything to disk. Useful for eyeballing real channel
        # width against generate-terrain's BAND m / MAX RING m settings
        # (Terrain tab) before committing to a run.
        elev_row = ttk.Frame(parent)
        elev_row.pack(fill="x", pady=(4, 0))
        self.show_elevation_contour_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            elev_row, text="Elevation contour", variable=self.show_elevation_contour_var,
            command=self._on_elevation_contour_toggle,
        ).pack(side="left")
        ttk.Label(elev_row, text="Elev (m):").pack(side="left", padx=(8, 0))
        self.elevation_contour_var = tk.DoubleVar(value=0.0)
        self.elevation_contour_scale = ttk.Scale(
            elev_row, from_=0.0, to=1.0, orient="horizontal",
            variable=self.elevation_contour_var, command=self._on_elevation_slider_drag,
        )
        self.elevation_contour_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.elevation_contour_label = ttk.Label(elev_row, text="0.0", width=8)
        self.elevation_contour_label.pack(side="left")
        ttk.Label(elev_row, text="Width (m):").pack(side="left", padx=(8, 0))
        self.elevation_contour_width_var = tk.StringVar(value="1.0")
        width_entry = ttk.Entry(elev_row, textvariable=self.elevation_contour_width_var, width=6)
        width_entry.pack(side="left", padx=(2, 0))
        width_entry.bind("<Return>", lambda e: self._on_elevation_width_changed())
        width_entry.bind("<FocusOut>", lambda e: self._on_elevation_width_changed())
        _Tooltip(width_entry, "Band thickness (m): shows heights in [Elev, Elev+Width) as solid "
                 "white, everything else transparent -- 1-bit, no antialiasing, purely an in-"
                 "memory visual (nothing written to disk). Narrow this toward your BAND m setting "
                 "to see one traced level at a time; widen it to see a whole channel at once. The "
                 "slider snaps to multiples of this value, matching the same band boundaries a real "
                 "generate-terrain run at that BAND m would actually produce.")

        stamp_cov_row = ttk.Frame(parent)
        stamp_cov_row.pack(fill="x", pady=(4, 0))
        self.show_stamp_coverage_var = tk.BooleanVar(value=False)
        stamp_cov_checkbox = ttk.Checkbutton(
            stamp_cov_row, text="Show stamp influence (Elevation contour)",
            variable=self.show_stamp_coverage_var, command=self._show_preview,
        )
        stamp_cov_checkbox.pack(side="left")
        _Tooltip(stamp_cov_checkbox, "Only has an effect while Elevation contour is also checked. "
                 "Filters the real current stamp list (every stamps_N.json layer under stamps/) "
                 "to only stamps whose own VALUE falls within the current [Elev, Elev+Width) band -- "
                 "not the whole course -- then evaluates the REAL kernel weight (via terrain."
                 "terrain_kernel.TerrainKernel -- the same kernel adaptive_refine.py scores candidates "
                 "with) from whichever of THOSE stamps pulls hardest at each cell, not just whether "
                 "some stamp's nominal radius geometrically reaches it. Pure RED = untouched by this "
                 "band's own stamps (not just 'touched by a neighboring band's stamp instead' -- those "
                 "are excluded), pure GREEN = strongly pulled, anything in between (orange/yellow) is "
                 "geometrically 'covered' but only weakly influenced. If a region reads solid green "
                 "under the old binary check but still looks unfilled visually, this is what actually "
                 "explains it. Recomputed per elevation (filtering changes which stamps are relevant), "
                 "but only over that band's own stamp subset, so it stays fast in practice.")

    # ------------------------------------------------------------------
    # Generate -- hex / contour / raster each live in their own tab with
    # only the fields that method actually reads (see PGA2k_gen.py's
    # step_generate_terrain). Restrict to mask is shared above the tabs:
    # every method respects it, just via a different mechanism (contour
    # crops each band's fill, hex/raster skip lattice points outside it
    # -- see generate_hex_grid/generate_raster_grid's mask_geometry arg).
    # ------------------------------------------------------------------

    def _build_generate_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Generate", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        self.generate_terrain_method_var = tk.StringVar(value="hex")

        self.generate_terrain_use_height_mask_var = tk.BooleanVar(value=False)
        generate_terrain_mask_checkbox = ttk.Checkbutton(
            parent, text="Restrict to mask", variable=self.generate_terrain_use_height_mask_var,
        )
        generate_terrain_mask_checkbox.pack(anchor="w", pady=(0, 4))
        _Tooltip(generate_terrain_mask_checkbox, "Restrict this layer to inside height_mask.geojson "
                 "(fairway/green/tee + buffered hole-path corridors, from Ingest OSM) -- restriction "
                 "happens BEFORE generation, not as a filter afterward: contour mode crops each "
                 "band's fill to the mask before the poisson-pack/crumb-scatter search runs, and hex "
                 "mode never places a lattice stamp outside it, so a small masked region does "
                 "proportionally less work, not full-course work followed by discarding most of it "
                 "(the course-wide baseline-flatten stamp is never masked). Uses the same "
                 "Buffer (px) slider as Refine's Mask option, in the Splines tab.")

        gen_notebook = ttk.Notebook(parent)
        gen_notebook.pack(fill="x", pady=(0, 4))
        hex_tab = ttk.Frame(gen_notebook, padding=4)
        contour_tab = ttk.Frame(gen_notebook, padding=4)
        raster_tab = ttk.Frame(gen_notebook, padding=4)
        gen_notebook.add(hex_tab, text="Hex")
        gen_notebook.add(contour_tab, text="Contour")
        gen_notebook.add(raster_tab, text="Raster")

        self._build_generate_hex_tab(hex_tab)
        self._build_generate_contour_tab(contour_tab)
        self._build_generate_raster_tab(raster_tab)

        gen_notebook.bind("<<NotebookTabChanged>>", lambda _e: self._autosize_notebook(gen_notebook))
        self._autosize_notebook(gen_notebook)

    def _run_generate_terrain_as(self, method: str) -> None:
        self.generate_terrain_method_var.set(method)
        self._run_generate_terrain()

    def _autosize_notebook(self, notebook: ttk.Notebook) -> None:
        """Resize a nested Notebook's pane to fit only its currently
        selected tab, instead of ttk's default of sizing to the tallest
        tab and never shrinking for shorter ones."""
        notebook.update_idletasks()
        tab_id = notebook.select()
        if not tab_id:
            return
        child = notebook.nametowidget(tab_id)
        notebook.configure(height=child.winfo_reqheight())

    def _add_grid_field(self, target, row, col, var, label, tooltip,
                         combobox_values: Optional[list[str]] = None, width: int = 8) -> ttk.Widget:
        """Shared field builder for the 3-col-max Generate grids -- label above widget,
        so BAND m/BRUSH/SIZE/etc. all line up the same way regardless of tab."""
        cell = ttk.Frame(target)
        cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
        lbl = ttk.Label(cell, text=label)
        lbl.pack(anchor="w")
        if combobox_values is not None:
            entry: ttk.Widget = ttk.Combobox(
                cell, textvariable=var, state="readonly", width=width, values=combobox_values,
            )
        else:
            entry = ttk.Entry(cell, textvariable=var, width=width)
        entry.pack(anchor="w")
        _Tooltip(lbl, tooltip)
        _Tooltip(entry, tooltip)
        return entry

    def _add_brush_field(self, target, row, col, var, tooltip, choices=None, labels=None) -> ttk.Widget:
        """Single-select BRUSH field, styled like Refine's Brush Type selector (label
        above, dropdown menu of 'id: name') but bound directly to one StringVar holding
        just the numeric brush id -- CLI args read that var's raw value unchanged."""
        choices = choices if choices is not None else self.BRUSH_TYPE_ORDER
        labels = labels if labels is not None else self.BRUSH_TYPE_LABELS
        cell = ttk.Frame(target)
        cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
        lbl = ttk.Label(cell, text="BRUSH")
        lbl.pack(anchor="w")
        menubutton = ttk.Menubutton(cell, textvariable=var, direction="below", width=6)
        menu = tk.Menu(menubutton, tearoff=False)
        for b in choices:
            menu.add_radiobutton(label=f"{b}: {labels.get(b, b)}", variable=var, value=str(b))
        menubutton["menu"] = menu
        menubutton.pack(anchor="w")
        _Tooltip(lbl, tooltip)
        _Tooltip(menubutton, tooltip)
        return menubutton

    def _build_generate_hex_tab(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(0, 4))

        self.pitch_var = tk.StringVar(value=str(HEX_LATTICE_PITCH_M))
        self._add_grid_field(grid, 0, 0, self.pitch_var, "SIZE",
                 "hex method only: spacing (m) of the initial coarse hex-grid stamp lattice "
                 "CENTERS (terrain/hexgrid.py's HEX_LATTICE_PITCH_M) -- smaller pitch means "
                 "more, more tightly-packed lattice points. Edge bleed derives from this "
                 "automatically (bleed=pitch); stamp radius instead comes from SPR%.")

        self.hex_spread_ratio_var = tk.StringVar(value=str(HEX_DEFAULT_SPREAD_RATIO))
        self._add_grid_field(grid, 0, 1, self.hex_spread_ratio_var, "SPR%",
                 "hex method only: scales stamp radius independently of pitch -- "
                 "stamp_radius = 2*pitch*SPR%. 1 (default) reproduces the original fixed radius = "
                 "2*pitch (each stamp reaches exactly to its nearest neighbors' centers); >1 grows "
                 "stamps past their neighbors for more overlap/blending, <1 shrinks them, "
                 "potentially opening coverage gaps between lattice centers. Does NOT move any "
                 "lattice center or affect edge bleed (bleed always = pitch).")

        self.hex_brush_var = tk.StringVar(value=str(HEX_DEFAULT_BRUSH))
        self._add_brush_field(grid, 0, 2, self.hex_brush_var,
                 "hex method only: brush every lattice stamp uses. Default: "
                 f"{HEX_DEFAULT_BRUSH} (terrain/hexgrid.py's DEFAULT_BRUSH).")

        self.hex_tool_var = tk.StringVar(value="flatten")
        self._add_grid_field(grid, 1, 0, self.hex_tool_var, "TOOL",
                 "hex method only: tool every lattice stamp uses -- flatten (0) pulls "
                 "terrain toward an absolute target height; raise (1) adds a delta instead, "
                 "preserving whatever relief already exists (see terrain/stamp.py). Most useful "
                 "for a masked pass that should build up an area without flattening it.",
                 combobox_values=["flatten", "raise"])

        self._add_step_button(parent, "Generate Terrain", lambda: self._run_generate_terrain_as("hex"))

    def _build_generate_raster_tab(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(fill="x", pady=(0, 4))

        self.raster_size_var = tk.StringVar(value=str(DEFAULT_RASTER_SIZE))
        self._add_grid_field(grid, 0, 0, self.raster_size_var, "SIZE",
                 "raster method only: center-to-center spacing (m) of the flat square-stamp grid "
                 "(terrain/rastergrid.py's RASTER_SIZES). Each run places ONE flat grid at this size -- "
                 "layer coarse-to-fine yourself by re-running Generate Terrain at each size in turn, "
                 "largest first.", combobox_values=[str(s) for s in RASTER_SIZES])

        self.raster_spread_ratio_var = tk.StringVar(value=str(DEFAULT_RASTER_SPREAD_RATIO))
        self._add_grid_field(grid, 0, 1, self.raster_spread_ratio_var, "SPR%",
                 "raster method only: scales stamp radius (x/z scale) independently of raster size -- "
                 "lattice centers are unaffected. 1 (default) reproduces the exact edge-to-edge tiling "
                 "radius; >1 grows stamps past their own cell for more overlap/blending, <1 shrinks "
                 "them, opening a gap between cells. Does NOT move any lattice center, only how big "
                 "each stamp is.")

        self.raster_brush_var = tk.StringVar(value=str(DEFAULT_RASTER_BRUSH))
        self._add_brush_field(
            grid, 0, 2, self.raster_brush_var,
            "raster method only: brush every lattice stamp uses. Default: "
            f"{DEFAULT_RASTER_BRUSH} (terrain/rastergrid.py's RASTER_BRUSH, the square 'hard "
            "square' brush) -- the grid's own center-to-center spacing is derived specifically "
            f"from type {DEFAULT_RASTER_BRUSH}'s measured flat-plateau/instant-edge profile, so "
            "it tiles edge-to-edge with no gap or overlap. Picking any other (circular) brush "
            "keeps that same radius but no longer tiles exactly -- a circle inscribed in each "
            "square cell leaves the corners uncovered -- so treat this as a cosmetic/blending "
            "choice (pair with a wider SPR% to close the resulting gaps), not a like-for-like swap.",
            choices=(DEFAULT_RASTER_BRUSH,) + self.BRUSH_TYPE_ORDER,
            labels={DEFAULT_RASTER_BRUSH: "square", **self.BRUSH_TYPE_LABELS},
        )

        center_bias_tooltip = (
            "raster method only: shifts each stamp's placement (not its sampled value) along this "
            "axis by raster size * this ratio. Compensates for a resolution-dependent 'drop shadow' "
            "at mask edges, caused by TerrainModel always favoring the +x/+z neighbor when stamps "
            "overlap (see terrain/rastergrid.py's CENTER BIAS section). 0 (default) is off -- no "
            "derived 'correct' value, dial in independently per axis, empirically, same as SPR%."
        )
        self.raster_center_bias_ratio_x_var = tk.StringVar(value=str(DEFAULT_RASTER_CENTER_BIAS_RATIO))
        self._add_grid_field(grid, 1, 0, self.raster_center_bias_ratio_x_var, "BIAS x", center_bias_tooltip)

        self.raster_center_bias_ratio_z_var = tk.StringVar(value=str(DEFAULT_RASTER_CENTER_BIAS_RATIO))
        self._add_grid_field(grid, 1, 1, self.raster_center_bias_ratio_z_var, "BIAS z", center_bias_tooltip)

        self._add_step_button(parent, "Generate Terrain", lambda: self._run_generate_terrain_as("raster"))

    # ------------------------------------------------------------------
    # Contour -- universal band/edge/cutoff settings up top, then Primary
    # Fill (Poisson / Fall Line, each its own tab and its own Generate
    # Terrain button that also pins fill_mode_var), then a shared
    # Secondary Fill section (the crumb-scatter cleanup pass, which runs
    # after either Primary Fill mode). Poisson's auto-tune calibration
    # knobs live behind their own popup instead of cluttering the grid.
    # ------------------------------------------------------------------

    def _build_generate_contour_tab(self, parent: ttk.Frame) -> None:
        self.band_spacing_var = tk.StringVar(value="5")
        self.edge_distance_var = tk.StringVar(value="0")
        self.max_stamps_var = tk.StringVar(value="")

        universal_grid = ttk.Frame(parent)
        universal_grid.pack(anchor="w", fill="x", pady=(0, 6))
        self._add_grid_field(universal_grid, 0, 0, self.band_spacing_var, "BAND m",
                 "contour method only: elevation spacing (m) defining each band -- smaller "
                 "means more, narrower bands. Tweak and re-run to see how it behaves.")
        self._add_grid_field(universal_grid, 0, 1, self.edge_distance_var, "EDGE m",
                 "contour method only, Poisson fill only: buffer (m) past the true band "
                 "boundary that every candidate's plateau must additionally clear, on top of "
                 "just fitting within it. Leaves a strip along every band edge for Secondary "
                 "Fill's finer crumb fill to handle instead of Primary Fill's large hard "
                 "stamps. Ignored in Fall Line mode. 0 disables.")
        self._add_grid_field(universal_grid, 0, 2, self.max_stamps_var, "MAX stamps",
                 "contour method only: stop once this many stamps have been placed in total -- "
                 "a quick way to sanity-check a parameter combination before committing to the "
                 "full run. Bands fill ascending by elevation, so the cutoff always lands on "
                 "the low-elevation end and most of the course will be genuinely unfilled, not "
                 "just coarser -- this is a partial-preview tool, not a real generation mode. "
                 "Leave blank for a real run. Not saved to project.json -- set explicitly each "
                 "time.")

        ttk.Label(parent, text="Primary Fill", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.fill_mode_var = tk.StringVar(value="poisson")
        self.fill_notebook = ttk.Notebook(parent)
        self.fill_notebook.pack(fill="x", pady=(0, 4))
        poisson_tab = ttk.Frame(self.fill_notebook, padding=4)
        fall_line_tab = ttk.Frame(self.fill_notebook, padding=4)
        self.fill_notebook.add(poisson_tab, text="Poisson")
        self.fill_notebook.add(fall_line_tab, text="Fall Line")
        self._build_contour_poisson_tab(poisson_tab)
        self._build_contour_fall_line_tab(fall_line_tab)
        self._autosize_notebook(self.fill_notebook)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._build_contour_secondary_fill(parent)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        # One button for both Primary Fill modes -- its label/command morph
        # to match whichever of Poisson/Fall Line is the active tab, rather
        # than keeping two separate near-identical buttons in sync by hand.
        self.contour_generate_btn_text_var = tk.StringVar(value="Generate Terrain (Poisson)")
        ttk.Button(
            parent, textvariable=self.contour_generate_btn_text_var, width=24,
            command=self._run_generate_contour,
        ).pack(anchor="w", pady=2)
        self.fill_notebook.bind("<<NotebookTabChanged>>", self._on_fill_mode_tab_changed)

    def _on_fill_mode_tab_changed(self, _event=None) -> None:
        idx = self.fill_notebook.index(self.fill_notebook.select())
        label = self.fill_notebook.tab(idx, "text")
        self.fill_mode_var.set("rect" if idx == 1 else "poisson")
        self.contour_generate_btn_text_var.set(f"Generate Terrain ({label})")
        self._autosize_notebook(self.fill_notebook)

    def _run_generate_contour(self) -> None:
        idx = self.fill_notebook.index(self.fill_notebook.select())
        self._run_generate_contour_as("rect" if idx == 1 else "poisson")

    def _run_generate_contour_as(self, fill_mode: str) -> None:
        self.fill_mode_var.set(fill_mode)
        self._run_generate_terrain_as("contour")

    def _build_contour_poisson_tab(self, parent: ttk.Frame) -> None:
        # Consolidated down from the earlier ring/rough/interior/residual
        # design's 16 separate fields -- that split no longer exists:
        # every elevation band gets identical two-pass treatment now,
        # regardless of shape or connectivity. The fit-tolerance knob (an
        # earlier version's EDGE SOFT) is gone too -- derived directly
        # from each brush's own real measured plateau, not a
        # separately-guessed ratio (see terrain/brush_profiles.py).
        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x")

        self.fill_brush_var = tk.StringVar(value="8")
        self._add_brush_field(grid, 0, 0, self.fill_brush_var,
                 "Poisson fill only: the fast poisson pack that does the bulk of the work. Type "
                 "8 (wide flat plateau) recommended: best real measured plateau fraction of the "
                 "four brush types, so it packs most efficiently. A brush with no real plateau "
                 "(10/54) can't do this job at all and is refused outright.")
        self.min_radius_var = tk.StringVar(value="10")
        self._add_grid_field(grid, 0, 1, self.min_radius_var, "MIN m",
                 "Poisson fill only: smallest tier (m).")
        self.max_radius_var = tk.StringVar(value="50")
        self._add_grid_field(grid, 0, 2, self.max_radius_var, "MAX m",
                 "Poisson fill only: largest tier (m) -- the main level-of-detail knob: how big "
                 "the biggest stamps in a band are allowed to be.")
        self.radius_step_var = tk.StringVar(value="0.85")
        self._add_grid_field(grid, 1, 0, self.radius_step_var, "STEP ratio",
                 "Poisson fill only: geometric (multiplicative) step between tiers, scanning "
                 "from MAX m down to MIN m -- each tier's radius is the previous tier's radius "
                 "times this ratio (0-1, NOT a fixed meters step). Scales automatically with "
                 "whatever MIN m/MAX m range you pick.")
        self.candidates_per_radius_var = tk.StringVar(value="")  # blank = auto-tune
        self._add_grid_field(grid, 1, 1, self.candidates_per_radius_var, "CAP",
                 "Poisson fill only: random-candidate cap per tier. Left blank, this is "
                 "AUTO-TUNED once at the start of the run (see Auto-tune... below) by searching "
                 "a handful of sample bands for the point of diminishing returns. Set this "
                 "explicitly, once you've seen a good auto-tuned value, to skip re-running that "
                 "search on every subsequent run -- it is NOT auto-persisted from an auto-tuned "
                 "run.")
        self.denoise_px_var = tk.StringVar(value="1")
        self._add_grid_field(grid, 1, 2, self.denoise_px_var, "DENOISE",
                 "contour method only (applies to both Primary Fill modes): morphological "
                 "open+close radius (heightmap pixels) applied to each band's mask before "
                 "filling -- trims isolated few-pixel bumps and fills isolated few-pixel gaps, "
                 "simplifying the boundary before it fragments the fill into unnecessary tiny "
                 "stamps. 0 disables.")

        # Auto-tune calibration -- only matters when CAP above is left
        # blank. Tucked behind a popup instead of its own grid: these tune
        # HOW the auto-tune search itself runs, not the actual fill, so
        # most runs shouldn't need to touch them.
        self.sweet_spot_ratio_var = tk.StringVar(value="0.10")
        self.sweet_spot_sample_bands_var = tk.StringVar(value="3")
        self.sweet_spot_seeds_var = tk.StringVar(value="2")

        ttk.Button(
            parent, text="Auto-tune...", command=self._open_auto_tune_dialog,
        ).pack(anchor="w", pady=(6, 0))

    def _open_auto_tune_dialog(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Poisson auto-tune calibration")
        win.transient(self.root)
        win.resizable(False, False)
        ttk.Label(
            win, text="Only used when CAP (above) is left blank.", padding=(8, 8, 8, 0),
        ).pack(anchor="w")
        grid = ttk.Frame(win, padding=8)
        grid.pack(fill="both", expand=True)
        self._add_grid_field(grid, 0, 0, self.sweet_spot_ratio_var, "RATIO",
                 "Keep doubling candidates-per-tier as long as each doubling reduces a sample "
                 "band's own uncovered-area fraction by at least this much, RELATIVE to the "
                 "previous doubling's own fraction (0.10 = stop once another doubling buys less "
                 "than a 10% relative improvement). NOT an absolute area target -- Poisson fill "
                 "has a genuine structural floor (real area smaller than MIN m, which no "
                 "candidate count can ever close), so a fixed target can be unreachable "
                 "regardless of true coverage quality.")
        self._add_grid_field(grid, 0, 1, self.sweet_spot_sample_bands_var, "SAMPLE",
                 "How many regularly-spaced bands to calibrate against, not every band -- bands "
                 "are similar enough that per-band tuning would mostly repeat the same search "
                 "for no benefit.")
        self._add_grid_field(grid, 0, 2, self.sweet_spot_seeds_var, "SEEDS",
                 "Random seeds per sampled band, so one lucky/unlucky seed doesn't skew the "
                 "calibration -- the MAX candidate count found across every band/seed "
                 "combination is what actually gets used. MAX candidates and the calibration "
                 "time budget are no longer separately tunable -- fixed at 50000 / 60s.")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def _build_contour_fall_line_tab(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x")

        self.rect_brush_var = tk.StringVar(value="72")
        self._add_brush_field(grid, 0, 0, self.rect_brush_var,
                 "Fall Line fill only: brush for the main per-boundary-edge pass -- must be a "
                 "square-shaped brush (type 72 is the only one today).",
                 choices=(72,), labels={72: "square"})
        self.rect_tolerance_var = tk.StringVar(value="1.5")
        self._add_grid_field(grid, 0, 1, self.rect_tolerance_var, "TOL m",
                 "Fall Line fill only: Douglas-Peucker boundary-simplify tolerance (m) when "
                 "tracing each band's own shape into vector polygons before placing per-edge "
                 "stamps.")
        self.rect_min_length_var = tk.StringVar(value="1")
        self._add_grid_field(grid, 0, 2, self.rect_min_length_var, "MIN m",
                 "Fall Line fill only: floor length (m) for a per-edge stamp when no opposite "
                 "wall is found within MAX search m.")
        self.rect_max_search_distance_var = tk.StringVar(value="400")
        self._add_grid_field(grid, 1, 0, self.rect_max_search_distance_var, "ROLL m",
                 "Fall Line fill only: cap (m) on the ray-cast search for the wall opposite each "
                 "boundary edge.")
        self.rect_width_samples_var = tk.StringVar(value="5")
        self._add_grid_field(grid, 1, 1, self.rect_width_samples_var, "SAMP",
                 "Fall Line fill only: points sampled across each boundary edge's own width (not "
                 "just its center) when ray-casting for the opposite wall.")

    def _build_contour_secondary_fill(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Secondary Fill", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.enable_secondary_fill_var = tk.BooleanVar(value=True)
        secondary_fill_checkbox = ttk.Checkbutton(
            parent, text="Enable", variable=self.enable_secondary_fill_var,
        )
        secondary_fill_checkbox.pack(anchor="w", pady=(0, 4))
        _Tooltip(secondary_fill_checkbox, "contour method only: whether the crumb-scatter cleanup "
                 "pass runs after Primary Fill. Unchecked skips it entirely -- whatever Primary "
                 "Fill leaves as crumbs stays unfilled, so coverage is no longer complete. Trades "
                 "that guarantee for speed, e.g. for a quick look at Primary Fill's own plateau "
                 "shape.")

        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x")

        self.smoothing_brush_var = tk.StringVar(value="10")
        self._add_brush_field(grid, 0, 0, self.smoothing_brush_var,
                 "Secondary Fill only: the crumb-scatter fill over whatever Primary Fill leaves "
                 "as genuine crumbs. A softer brush (type 10 default) so small scattered crumbs "
                 "blend rather than showing a hard-edged patch.")
        self.smoothing_min_radius_var = tk.StringVar(value="4")
        self._add_grid_field(grid, 0, 1, self.smoothing_min_radius_var, "MIN m",
                 "Secondary Fill only: its OWN radius floor, independent of Primary Fill's MIN m "
                 "-- the crumb stage's scale is a property of how it does its own job, not of "
                 "how finely Primary Fill happened to be tiered.")
        self.smooth_ratio_var = tk.StringVar(value="4")
        self._add_grid_field(grid, 0, 2, self.smooth_ratio_var, "RATIO",
                 "Secondary Fill only: scatter radius as a multiple of its own MIN m (default 4 "
                 "-- a 16m scatter radius at the 4m default floor). Deliberately a ratio: the "
                 "crumb stage's scale should track its own floor directly rather than needing "
                 "separate re-tuning.")
        self.smooth_claim_fraction_var = tk.StringVar(value="0.25")
        self._add_grid_field(grid, 1, 0, self.smooth_claim_fraction_var, "EAT",
                 "Secondary Fill only: \"eat\" -- how much of each crumb-scatter stamp's placed "
                 "radius gets claimed. Deliberately much heavier overlap (claim less) than "
                 "Primary Fill's real-plateau-derived claim, since Secondary Fill's whole job is "
                 "blanket-covering whatever Primary Fill couldn't reach, not precise packing.")

    # ------------------------------------------------------------------
    # Cart Paths
    # ------------------------------------------------------------------

    def _build_cart_paths_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Cart Paths", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        splines_row = ttk.Frame(parent)
        splines_row.pack(fill="x", pady=2)
        ttk.Label(splines_row, text="Splines path:").pack(side="left")
        self.cart_path_splines_path_var = tk.StringVar(value="")
        splines_entry = ttk.Entry(splines_row, textvariable=self.cart_path_splines_path_var, width=28)
        splines_entry.pack(side="left", padx=4)
        ttk.Button(splines_row, text="Browse...", command=self._browse_cart_path_splines).pack(side="left")
        _Tooltip(splines_entry, "Path to the exported spline JSON (bezier waypoint format) containing "
                 "cart path data. Blank = auto-detect surfaceSplines2.json then surfaceSplines.json "
                 "directly under the working directory.")

        cart_path_grid = ttk.Frame(parent)
        cart_path_grid.pack(fill="x", pady=2)

        def add_cart_path_field(row, col, var, label, tooltip):
            cell = ttk.Frame(cart_path_grid)
            cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
            lbl = ttk.Label(cell, text=label)
            lbl.pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=var, width=10)
            entry.pack(anchor="w")
            _Tooltip(lbl, tooltip)
            _Tooltip(entry, tooltip)

        self.cart_path_surface_var = tk.StringVar(value="")
        self.cart_path_stamp_radius_var = tk.StringVar(value=f"{CART_PATH_STAMP_RADIUS:.4f}")
        self.cart_path_spacing_var = tk.StringVar(value=f"{CART_PATH_SPACING_M:.4f}")
        self.cart_path_height_avg_radius_var = tk.StringVar(value=f"{CART_PATH_HEIGHT_AVG_RADIUS_M:.4f}")

        add_cart_path_field(0, 0, self.cart_path_surface_var, "SURFACE",
                             "The surface value identifying a cart path spline in the source JSON. "
                             "No safe default -- must be set (here, or saved in project.json from a "
                             "prior run) before this step will run.")
        add_cart_path_field(0, 1, self.cart_path_stamp_radius_var, "RADIUS m",
                             "Type 15 stamp radius (m), a true center-to-edge distance -- controls "
                             "the flattened path's real width via the brush's own measured plateau "
                             f"geometry. Default {CART_PATH_STAMP_RADIUS:.4f} gives exactly a 1.7m "
                             "plateau width.")
        add_cart_path_field(0, 2, self.cart_path_spacing_var, "SPACING m",
                             "Pearl spacing (m) along each cart path -- how far apart consecutive "
                             f"stamps are placed. Default {CART_PATH_SPACING_M:.4f} is 85% of the "
                             "1.7m plateau width, for genuine along-path overlap.")
        add_cart_path_field(1, 0, self.cart_path_height_avg_radius_var, "HEIGHT AVG m",
                             "Radius (m) to average real heightmap data over at each stamp. Default "
                             f"{CART_PATH_HEIGHT_AVG_RADIUS_M:.4f} is half the stamp's own active/"
                             "nonzero footprint.")

        self._add_step_button(parent, "Generate Cart Paths", self._run_generate_cart_paths)

    # ------------------------------------------------------------------
    # Refine -- adaptive / scatter each live in their own tab with only
    # the fields that method actually reads (see PGA2k_gen.py's
    # step_refine_terrain). Resolution/Mask/brush-type/RAD m/EAT %/
    # SPR %/SHR % are shared above the tabs: both methods take the same
    # rad_m / claim_radius_fraction / brush_radius_spread_ratio /
    # planar_shrink_factor params -- SHR % does double duty (RMS-shrink
    # step in adaptive, radius jitter magnitude in scatter), same knob,
    # method-appropriate meaning, not two separate fields.
    # ------------------------------------------------------------------

    def _add_refine_field(self, target, row, col, key, abbrev, tooltip, variable, required,
                           combobox_values=None):
        cell = ttk.Frame(target)
        cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
        label = ttk.Label(cell, text=abbrev)
        label.pack(anchor="w")
        if combobox_values:
            entry = ttk.Combobox(cell, textvariable=variable, values=combobox_values, width=7, state="normal")
        else:
            entry = ttk.Entry(cell, textvariable=variable, width=8)
        entry.pack(anchor="w")
        full_tooltip = tooltip + ("" if required else " (optional)")
        _Tooltip(label, full_tooltip)
        _Tooltip(entry, full_tooltip)
        if required:
            self.refine_labels[key] = label

    def _build_refine_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Refine", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        self.refine_method_var = tk.StringVar(value="adaptive")

        brush_type_row = ttk.Frame(parent)
        brush_type_row.pack(fill="x", pady=(0, 6))
        ttk.Label(brush_type_row, text="Brush type:").pack(side="left")
        self.brush_type_vars: dict[int, tk.BooleanVar] = {
            b: tk.BooleanVar(value=True) for b in self.BRUSH_TYPE_ORDER
        }
        self.brush_types_menubutton = ttk.Menubutton(brush_type_row, text="", direction="below")
        brush_menu = tk.Menu(self.brush_types_menubutton, tearoff=False)
        for b in self.BRUSH_TYPE_ORDER:
            brush_menu.add_checkbutton(
                label=f"{b}: {self.BRUSH_TYPE_LABELS[b]}",
                variable=self.brush_type_vars[b],
                command=lambda b=b: self._on_brush_type_toggled(b),
            )
        self.brush_types_menubutton["menu"] = brush_menu
        self.brush_types_menubutton.pack(side="left", padx=4)
        self._update_brush_types_label()
        _Tooltip(self.brush_types_menubutton, "Which brush shapes refine-terrain is allowed to place "
                 "at a hotspot (candidate_brushes -- every checked type is scored, the best fit wins). "
                 "type 10 alone approximates Chad's smooth 2m-grid raster result (uniform small "
                 "stamps, no flat plateau); mixing in 8/9 trades some of that smoothness for better "
                 "fit against sharp features. At least one type must stay checked.")

        self.tolerance_var = tk.StringVar(value="2")
        self.resolution_var = tk.StringVar(value="200")
        self.min_hotspot_radius_cells_var = tk.StringVar(value="1.0")
        self.max_new_var = tk.StringVar()
        self.model_rebuild_interval_var = tk.StringVar()
        self.spread_ratio_var = tk.StringVar(value="1")
        self.claim_fraction_var = tk.StringVar(value="1")
        self.rad_var = tk.StringVar(value="25")
        self.max_planar_rms_var = tk.StringVar(value="")  # blank = off (old behavior)
        self.planar_shrink_var = tk.StringVar(value="0.75")
        self.variation_contrast_gamma_var = tk.StringVar(value="2")
        self.subpixel_jitter_var = tk.StringVar(value="0.5")
        self.refine_labels: dict[str, ttk.Label] = {}

        shared_grid = ttk.Frame(parent)
        shared_grid.pack(anchor="w", fill="x")

        _RESOLUTION_PRESETS = ["25", "50", "100", "125", "200", "250", "400", "500", "1000", "2000"]
        self._add_refine_field(shared_grid, 0, 0, "resolution", "RES px", "Error grid resolution "
                  "(cells per side) -- same grid preview_error.png uses. Presets are exact divisors "
                  "of the 2000 m course, so every cell lands on a whole-meter boundary matching the "
                  "ground heightmap's own 1 px = 1 m grid -- other values still work, they just "
                  "won't align as cleanly (e.g. 1600 gives 1.25 m cells, straddling heightmap pixel "
                  "boundaries).", self.resolution_var, required=True, combobox_values=_RESOLUTION_PRESETS)

        self.use_height_mask_var = tk.BooleanVar(value=False)
        mask_checkbox = ttk.Checkbutton(shared_grid, text="Mask", variable=self.use_height_mask_var)
        mask_checkbox.grid(row=0, column=1, sticky="w", padx=3, pady=2)
        _Tooltip(mask_checkbox, "Restrict hotspot placement to inside height_mask.geojson "
                 "(fairway/green/tee + buffered hole-path corridors, from Ingest OSM). Everything "
                 "outside is treated like no-data -- never becomes a hotspot.")

        self._add_refine_field(shared_grid, 1, 0, "max_new", "MAX n", "Cap on new stamps this pass. "
                  "Leave blank for no cap.", self.max_new_var, required=False)
        self._add_refine_field(shared_grid, 1, 1, "spread_ratio", "SPR %", "Brush radius spread ratio: "
                  "every brush is scored at the same base radius (a fair comparison of which brush "
                  "shape fits best), but the winning brush is PLACED at radius scaled by "
                  "spread_ratio ** rank (ranks 0..3 for types 8/9/10/54) -- higher-rank (smoother) "
                  "brushes end up placed wider, forcing more overlap as the falloff gets gentler. "
                  "1 disables it.", self.spread_ratio_var, required=True)
        self._add_refine_field(shared_grid, 1, 2, "claim_fraction", "EAT %", "Claimed radius "
                  "fraction: how much of the placed radius gets marked done. Below 1 lets "
                  "neighboring stamps overlap. adaptive: 1 disables it (old behavior). scatter: "
                  "RAD * EAT is the minimum center-to-center spacing between stamps (the actual "
                  "Poisson-disc constraint) AND the radius within which real LIDAR points are "
                  "averaged for each stamp's flatten target.", self.claim_fraction_var, required=True)
        self._add_refine_field(shared_grid, 2, 0, "rad", "RAD m", "Literal target stamp radius (m) "
                  "for this pass. adaptive: becomes max_radius (min_radius derives from the same "
                  "fixed 0.5 ratio as before). scatter: the literal per-stamp placement radius "
                  "before jitter. Replaces the old DEC % (radius_decay_per_pass) -- the implied "
                  "decay vs. the last run is now shown as a computed, read-only value below instead "
                  "of being something you type in.", self.rad_var, required=True)
        self._add_refine_field(shared_grid, 2, 1, "planar_shrink", "SHR %", "adaptive: how much to "
                  "shrink a hotspot's radius each time it fails the max planar-fit RMS check (only "
                  "used when PLN m is set). scatter: repurposed as radius jitter (when Slope is "
                  "off) -- each stamp's radius is randomized within [RAD * SHR%, RAD], so centers "
                  "arrange themselves organically instead of a visibly uniform lattice. When Slope "
                  "is on, this instead becomes the floor for how small a stamp can shrink to on the "
                  "steepest ground. 1.0 disables shrinking either way (every stamp is exactly RAD).",
                  self.planar_shrink_var, required=False)

        refine_notebook = ttk.Notebook(parent)
        refine_notebook.pack(fill="x", pady=(4, 0))
        adaptive_tab = ttk.Frame(refine_notebook, padding=4)
        scatter_tab = ttk.Frame(refine_notebook, padding=4)
        refine_notebook.add(adaptive_tab, text="Adaptive")
        refine_notebook.add(scatter_tab, text="Scatter")

        self._build_refine_adaptive_tab(adaptive_tab)
        self._build_refine_scatter_tab(scatter_tab)

        refine_notebook.bind("<<NotebookTabChanged>>", lambda _e: self._autosize_notebook(refine_notebook))
        self._autosize_notebook(refine_notebook)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.registration_marks_var = tk.BooleanVar(value=False)
        reg_marks_checkbox = ttk.Checkbutton(
            parent, text="Registration marks", variable=self.registration_marks_var,
        )
        reg_marks_checkbox.pack(anchor="w")
        _Tooltip(reg_marks_checkbox, "Add a small type-73 (circle) raise stamp and a matching 5m "
                 "circle spline (cart path surface) at each of the 4 course corners -- for visually "
                 "confirming in-game that terrain and splines land exactly where expected. Shared "
                 "with the same checkbox in the Splines tab (one setting, both places).")
        self.prune_overlapped_stamps_var = tk.BooleanVar(value=True)
        prune_checkbox = ttk.Checkbutton(
            parent, text="Delete overlapped stamps", variable=self.prune_overlapped_stamps_var,
        )
        prune_checkbox.pack(anchor="w")
        _Tooltip(prune_checkbox, "Drop stamps fully overwritten by later stamps before writing/fitting "
                 "water (terrain/stamp_pruning.py) -- pure cleanup with no effect on the resolved "
                 "terrain, just fewer stamps in the output. On by default; uncheck to keep every stamp "
                 "as-is (e.g. while debugging pruning itself).")
        write_terrain_btn = self._add_step_button(parent, "Write Terrain", self._run_write_terrain)
        _Tooltip(write_terrain_btn, "Writes userLayers.json's \"height\" key (the terrain stamps) -- "
                 "leaves \"water\" exactly as it was found (see the separate Write Water button below, "
                 "which is slower and only needs re-running when water levels should track a terrain "
                 "change).")
        write_water_btn = self._add_step_button(parent, "Write Water", self._run_write_water)
        _Tooltip(write_water_btn, "Writes userLayers.json's \"water\" key -- leaves \"height\" exactly "
                 "as it was found (see course_output/water.py). Water objects are built from "
                 "features.geojson's water polygons (natural=water, golf=water_hazard, waterway=* "
                 "areas -- run Ingest OSM first if none show up) fitted to the CURRENT stamp list's "
                 "low points (re-loaded/pruned/normalized the same way as Write Terrain, using the "
                 "same checkboxes above), so this can be slow -- only run it when water levels need "
                 "to catch up with a terrain change, not after every terrain iteration.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Refinement values:").pack(anchor="w")
        self.refine_stats_text = tk.Text(
            parent, height=11, width=28, wrap="none", state="disabled",
            font=("TkFixedFont", 9), background="#f0f0f0",
        )
        self.refine_stats_text.pack(anchor="w", fill="x", pady=(2, 0))
        self._refresh_refine_stats()

    def _run_refine_terrain_as(self, method: str) -> None:
        self.refine_method_var.set(method)
        self._run_refine_terrain()

    def _build_refine_adaptive_tab(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x")

        self._add_refine_field(grid, 0, 0, "tolerance", "TOL m", "Error tolerance (m): "
                  "|predicted - actual| above this counts as a hotspot.", self.tolerance_var,
                  required=True)
        self._add_refine_field(grid, 0, 1, "min_hotspot", "HOT px", "Min hotspot radius, in cells "
                  "at the CURRENT resolution (not meters -- a cell is 2000/RES m, so this floor "
                  "scales with resolution, not tied to a fixed real-world size). Smaller regions "
                  "are treated as noise, not a real feature.", self.min_hotspot_radius_cells_var,
                  required=True)
        self._add_refine_field(grid, 1, 0, "model_rebuild_interval", "INT n", "Model rebuild "
                  "interval: how many new stamps accumulate before the error grid gets fully "
                  "re-rendered against everything placed so far this pass (see "
                  "terrain/adaptive_refine.py). Higher = fewer, more expensive-per-stamp-count "
                  "rebuilds -- the single biggest lever on how long a pass takes at high "
                  "resolution/stamp counts, at the cost of later candidates in the same pass being "
                  "fit against a slightly staler model (bounded by this many stamps' worth of "
                  "staleness). Leave blank to use the default (25).",
                  self.model_rebuild_interval_var, required=False)
        self._add_refine_field(grid, 1, 1, "max_planar_rms", "PLN m", "Max planar-fit RMS (m): "
                  "shrinks a hotspot's radius until the region's actual LIDAR heights fit a single "
                  "tilted plane within this tolerance -- catches valleys/ridges/creases an "
                  "error-sign-only region never stops growing across (e.g. a V-shaped valley "
                  "cross-section stays one sign of error from floor to rim, so without this it "
                  "gets averaged into one stamp that pulls the floor up and the rim down). Leave "
                  "blank to disable (old behavior).", self.max_planar_rms_var, required=False)

        self._add_step_button(parent, "Refine Terrain", lambda: self._run_refine_terrain_as("adaptive"))

    def _build_refine_scatter_tab(self, parent: ttk.Frame) -> None:
        grid = ttk.Frame(parent)
        grid.pack(anchor="w", fill="x")

        self.use_slope_radius_var = tk.BooleanVar(value=False)
        slope_checkbox = ttk.Checkbutton(grid, text="Slope", variable=self.use_slope_radius_var)
        slope_checkbox.grid(row=0, column=0, sticky="w", padx=3, pady=2)
        _Tooltip(slope_checkbox, "scatter only: drive each stamp's radius from real local terrain "
                 "slope (raw gradient magnitude, computed once over the whole grid) instead of random "
                 "jitter. Superseded by Variation below, which fixes a real blind spot this has (a "
                 "gentle, multi-km fairway grade reads as 'steep' everywhere, even where a wide flat "
                 "stamp would represent it fine) -- kept for back-compat/comparison. Variation wins if "
                 "both are checked. SHR % is reused as the 'how small can it shrink' floor either way.")

        self.use_variation_radius_var = tk.BooleanVar(value=False)
        variation_checkbox = ttk.Checkbutton(
            grid, text="Variation", variable=self.use_variation_radius_var,
        )
        variation_checkbox.grid(row=0, column=1, sticky="w", padx=3, pady=2)
        _Tooltip(variation_checkbox, "scatter only: drive each stamp's radius from RMS-from-local-mean "
                 "at lag=RAD (computed once, over the whole grid) instead of random jitter -- unlike "
                 "Slope, this scales with BOTH real curvature (bumps/greens) and macro-scale grade "
                 "carried across the window, since a single flat stamp can't represent a tilted plane "
                 "any better than a bumpy one. Flat/uniformly-graded ground still gets large stamps; "
                 "genuinely detailed ground gets small ones. Wins over Slope if both are checked. SHR%% "
                 "is the shrink floor, GAM sharpens the map toward the extremes.")

        self._add_refine_field(grid, 1, 0, "variation_contrast_gamma", "GAM", "Variation contrast "
                  "gamma: exponent applied to the normalized variation field before mapping into "
                  "[RAD * SHR%, RAD] -- >1 sharpens toward the extremes (only genuinely "
                  "high-variation cells shrink much), 1.0 is a plain linear map. Only used when "
                  "Variation is checked.", self.variation_contrast_gamma_var, required=False)

        self.density_weighted_var = tk.BooleanVar(value=False)
        density_checkbox = ttk.Checkbutton(
            grid, text="Density", variable=self.density_weighted_var,
        )
        density_checkbox.grid(row=1, column=1, sticky="w", padx=3, pady=2)
        _Tooltip(density_checkbox, "scatter only: draw candidate sites from a ~1/radius^2-weighted "
                 "density field instead of uniform-random over the whole course. Fixes what shrinking "
                 "radius alone can't -- a smaller target radius only changes how big an accepted dart "
                 "is, never how often darts land there, so small high-detail regions were getting "
                 "isolated small stamps that read as random bumps instead of a tightly-packed cluster "
                 "resolving the actual detail. Has no effect unless Slope or Variation is also checked "
                 "-- with neither, every site wants the same radius already.")

        self._add_refine_field(grid, 2, 0, "subpixel_jitter_fraction", "JIT %", "Subpixel jitter "
                  "fraction: fraction of a heightmap cell's width to jitter density-weighted draws "
                  "by, off the exact cell center -- dither only, to avoid visibly grid-aligned "
                  "stamp centers (the course never needs sub-cell precision on its own). Only used "
                  "when Density is checked.", self.subpixel_jitter_var, required=False)

        self._add_step_button(parent, "Refine Terrain", lambda: self._run_refine_terrain_as("scatter"))

    def _on_brush_type_toggled(self, changed_brush: int) -> None:
        if not any(v.get() for v in self.brush_type_vars.values()):
            # Keep at least one brush type checked -- an empty
            # --candidate-brushes would leave refine-terrain with
            # nothing to score/place at any hotspot. Revert whichever
            # checkbox was just unchecked to cause this.
            self.brush_type_vars[changed_brush].set(True)
            return
        self._update_brush_types_label()

    def _update_brush_types_label(self) -> None:
        selected = [b for b in self.BRUSH_TYPE_ORDER if self.brush_type_vars[b].get()]
        if len(selected) == len(self.BRUSH_TYPE_ORDER):
            text = "all"
        else:
            text = ", ".join(str(b) for b in selected)
        self.brush_types_menubutton.config(text=text)

    def _build_log_panel(self, paned: ttk.PanedWindow) -> None:
        frame = ttk.Frame(paned)
        paned.add(frame, weight=1)
        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text="Output log").pack(side="left")
        ttk.Button(header, text="Clear", command=self._clear_log).pack(side="right")

        text_row = ttk.Frame(frame)
        text_row.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            text_row, width=40, wrap="word", state="disabled",
            bg="#1e1e1e", fg="#dddddd", insertbackground="white",
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(text_row, command=self.log_text.yview)
        scroll.pack(side="left", fill="y")
        self.log_text["yscrollcommand"] = scroll.set

    def _build_preview_panel(self, paned: ttk.PanedWindow) -> None:
        frame = ttk.Frame(paned)
        paned.add(frame, weight=2)

        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text="Preview:").pack(side="left")
        self.preview_choice = tk.StringVar(value=PREVIEW_FILES[0])
        dropdown = ttk.Combobox(
            header, textvariable=self.preview_choice, values=PREVIEW_FILES,
            state="readonly", width=30,
        )
        dropdown.pack(side="left", padx=4)
        dropdown.bind("<<ComboboxSelected>>", lambda e: self._on_preview_choice_changed())
        self.undo_button = ttk.Button(header, text="Undo", command=self._run_undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(header, text="Redo", command=self._run_redo)
        self.redo_button.pack(side="left")

        ttk.Label(header, text="Version:").pack(side="left", padx=(8, 0))
        self.preview_version = tk.IntVar(value=0)
        self.preview_version_scale = ttk.Scale(
            header, from_=0, to=0, orient="horizontal",
            variable=self.preview_version, command=self._on_preview_version_changed,
        )
        # This is the widget that should take up whatever space is left
        # in the row, unlike the fixed-size dropdown/button/label beside it.
        self.preview_version_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.preview_version_label = ttk.Label(header, text="current", width=10)
        self.preview_version_label.pack(side="left")

        # Scroll wheel over either the slider or the image itself steps
        # through versions -- Windows/Mac send <MouseWheel> with event.delta;
        # Linux sends <Button-4>/<Button-5> instead. Shift+scroll instead
        # cycles the preview *type* dropdown (same cross-platform split).
        for widget in (self.preview_version_scale,):
            widget.bind("<MouseWheel>", self._on_preview_scroll)
            widget.bind("<Button-4>", self._on_preview_scroll)
            widget.bind("<Button-5>", self._on_preview_scroll)
            widget.bind("<Shift-MouseWheel>", self._on_preview_type_scroll)
            widget.bind("<Shift-Button-4>", self._on_preview_type_scroll)
            widget.bind("<Shift-Button-5>", self._on_preview_type_scroll)

        ttk.Label(header, text="Zoom:").pack(side="left", padx=(8, 0))
        self.preview_zoom_var = tk.DoubleVar(value=1.0)
        zoom_scale = ttk.Scale(
            header, from_=0.25, to=3.0, orient="horizontal",
            variable=self.preview_zoom_var, command=lambda _v: self._show_preview(),
        )
        zoom_scale.pack(side="left", padx=(2, 0))
        self.preview_zoom_label = ttk.Label(header, text="100%", width=5)
        self.preview_zoom_label.pack(side="left")
        ttk.Button(header, text="Reset", width=6, command=self._reset_preview_zoom).pack(side="left", padx=(2, 0))

        canvas_frame = ttk.Frame(frame)
        canvas_frame.pack(fill="both", expand=True)
        self.preview_canvas = tk.Canvas(canvas_frame, background="gray85", highlightthickness=0)
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.preview_canvas.xview)
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        self.preview_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self._preview_canvas_image_id = None

        # Plain scroll still cycles versions, Shift+scroll still cycles
        # preview type (both unchanged) -- Ctrl+scroll is new, and zooms
        # instead, so none of the existing scroll behavior is disturbed.
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_scroll)
        self.preview_canvas.bind("<Button-4>", self._on_preview_scroll)
        self.preview_canvas.bind("<Button-5>", self._on_preview_scroll)
        self.preview_canvas.bind("<Shift-MouseWheel>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Shift-Button-4>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Shift-Button-5>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_preview_zoom_scroll)
        self.preview_canvas.bind("<Control-Button-4>", self._on_preview_zoom_scroll)
        self.preview_canvas.bind("<Control-Button-5>", self._on_preview_zoom_scroll)
        # Middle-click-drag pans the view -- scan_mark/scan_dragto are
        # tkinter Canvas's own built-in support for exactly this, so no
        # manual scroll-position math is needed here.
        self.preview_canvas.bind("<Button-2>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B2-Motion>", lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))
        self.preview_canvas.bind("<Configure>", self._center_preview_image)

    _SPLINE_KIND_FILTERS = (
        "All", "green", "tee", "fairway", "rough", "heavyrough", "bunker",
        "water", "cartpath", "service_road", "roadway", "driveway", "path",
        "building", "wood", "pavement", "mulch", "hole",
        SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND,
    )

    BRUSH_TYPE_ORDER = (8, 9, 10, 54)
    BRUSH_TYPE_LABELS = {8: "hard", 9: "med", 10: "soft", 54: "smooth"}

    _OBJECT_SOURCE_FILTERS = ("All", "OSM", "LIDAR", "Manual", "Border")

    def _build_splines_tab(self, parent: ttk.Frame) -> None:

        hole_width_row = ttk.Frame(parent)
        hole_width_row.pack(fill="x", pady=(4, 0))
        ttk.Label(hole_width_row, text="Hole width (m):").pack(side="left")
        self.hole_corridor_buffer_var = tk.StringVar(value=str(DEFAULT_HOLE_CORRIDOR_BUFFER_PX))
        hole_width_entry = ttk.Entry(hole_width_row, textvariable=self.hole_corridor_buffer_var, width=6)
        hole_width_entry.pack(side="left", padx=4)
        _Tooltip(hole_width_entry, "Buffer applied to each hole routing centerline (tee-to-green line) "
                 "before it contributes to the height mask -- an unbuffered centerline alone would leave "
                 "most of the actual playing corridor outside the mask. This is a BUFFER distance, "
                 "roughly HALF the resulting corridor width, not the total width. Takes effect the next "
                 "time Ingest OSM runs (File tab).")

        overlay_row = ttk.Frame(parent)
        overlay_row.pack(fill="x", pady=(4, 0))
        self.overlay_osm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            overlay_row, text="Overlay OSM", variable=self.overlay_osm_var,
            command=self._show_preview,
        ).pack(side="left")
        ttk.Label(overlay_row, text="Opacity:").pack(side="left", padx=(8, 0))
        self.overlay_opacity_var = tk.DoubleVar(value=0.6)
        ttk.Scale(
            overlay_row, from_=0.0, to=1.0, orient="horizontal",
            variable=self.overlay_opacity_var, command=lambda _v: self._show_preview(),
        ).pack(side="left", fill="x", expand=True, padx=4)

        # Separate, independent overlay from the static OSM one above:
        # a live-redrawn highlight of the fairway/green mask buffer, so
        # dragging the slider shows exactly how far the buffer currently
        # reaches without needing to re-run ingest-osm each time (see
        # _get_cached_mask_merged_geometry / ingest.osm.rasterize_mask_rgba).
        # Also doubles as the base geometry for the "Border" object-fill
        # mode below (see _mask_geometry_full_frame) -- Mask/Source/Buffer
        # (px) are shared by both the preview and that fill mode; Border
        # and Border width (m) are new, only meaningful to the fill mode.
        mask_section = ttk.Frame(parent)
        mask_section.pack(fill="x", pady=(4, 0))

        mask_row1 = ttk.Frame(mask_section)
        mask_row1.pack(fill="x")
        self.show_mask_buffer_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_row1, text="Mask", variable=self.show_mask_buffer_var,
            command=self._show_preview,
        ).pack(side="left")
        self.mask_border_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            mask_row1, text="Border", variable=self.mask_border_var,
            command=self._show_preview,
        ).pack(side="left", padx=(8, 0))
        ttk.Label(mask_row1, text="Source:").pack(side="left", padx=(8, 0))
        self.mask_source_var = tk.StringVar(value="marked")
        mask_source_combo = ttk.Combobox(
            mask_row1, textvariable=self.mask_source_var, state="readonly", width=8,
            values=("marked", "selected"),
        )
        mask_source_combo.pack(side="left", padx=(2, 0))
        mask_source_combo.bind("<<ComboboxSelected>>", lambda e: self._show_preview())
        _Tooltip(mask_source_combo, "What defines the mask region: 'marked' uses whatever Features "
                 "are flagged in the Splines tab (Toggle Mask/Toggle All); 'selected' uses whichever "
                 "spline(s) are currently selected in the Splines tab instead, regardless of their "
                 "mask flag. Both still go through the same Buffer (px) outward buffer above -- and, "
                 "when 'Border' is checked, feed the border ring's base outline too.")

        mask_row2 = ttk.Frame(mask_section)
        mask_row2.pack(fill="x", pady=(2, 0))
        ttk.Label(mask_row2, text="Buffer:").pack(side="left")
        self.mask_buffer_preview_var = tk.DoubleVar(value=50.0)
        ttk.Scale(
            mask_row2, from_=0.0, to=400.0, orient="horizontal",
            variable=self.mask_buffer_preview_var, command=lambda _v: self._show_preview(),
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.mask_buffer_preview_label = ttk.Label(mask_row2, text="50", width=4)
        self.mask_buffer_preview_label.pack(side="left")

        mask_row3 = ttk.Frame(mask_section)
        mask_row3.pack(fill="x", pady=(2, 0))
        ttk.Label(mask_row3, text="Border:").pack(side="left")
        self.border_width_var = tk.DoubleVar(value=10.0)
        ttk.Scale(
            mask_row3, from_=0.0, to=400.0, orient="horizontal",
            variable=self.border_width_var, command=lambda _v: self._show_preview(),
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.border_width_preview_label = ttk.Label(mask_row3, text="10", width=4)
        self.border_width_preview_label.pack(side="left")

        filter_row = ttk.Frame(parent)
        filter_row.pack(fill="x")
        ttk.Label(filter_row, text="Filter:").pack(side="left")
        self.splines_kind_filter_var = tk.StringVar(value="All")
        filter_box = ttk.Combobox(
            filter_row, textvariable=self.splines_kind_filter_var, state="readonly", width=12,
            values=self._SPLINE_KIND_FILTERS,
        )
        filter_box.pack(side="left", padx=4)
        filter_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_splines_list())
        ttk.Button(filter_row, text="Refresh", command=self._refresh_splines_list).pack(side="left")
        
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.splines_tree = ttk.Treeview(
            tree_frame, columns=("kind", "tag", "mask", "clusters"), show="headings", height=18,
            selectmode="extended",
        )
        self.splines_tree.heading("kind", text="Kind")
        self.splines_tree.heading("tag", text="Tag")
        self.splines_tree.heading("mask", text="Mask")
        self.splines_tree.heading("clusters", text="Clusters")
        self.splines_tree.column("kind", width=90)
        self.splines_tree.column("tag", width=90)
        self.splines_tree.column("mask", width=24, anchor="center")
        self.splines_tree.column("clusters", width=140)
        self.splines_tree.pack(side="left", fill="both", expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.splines_tree.yview)
        tree_scroll.pack(side="left", fill="y")
        self.splines_tree["yscrollcommand"] = tree_scroll.set
        self.splines_tree.bind("<<TreeviewSelect>>", lambda e: self._on_spline_selected())

        button_row = ttk.Frame(parent)
        button_row.pack(fill="x", pady=(6, 0))
        ttk.Button(button_row, text="Mask", command=self._toggle_selected_mask).pack(side="left")
        toggle_all_btn = ttk.Button(button_row, text="Mask All", command=self._toggle_all_mask)
        toggle_all_btn.pack(side="left", padx=(4, 0))
        _Tooltip(toggle_all_btn, "Toggle mask for every feature currently visible (i.e. matching the "
                 "active Filter) -- any kind, not just golf objects. If any are currently unmasked, "
                 "masks all of them in; otherwise masks all of them out.")

        ms_btn = ttk.Button(button_row, text="MS", width=4, command=self._splines_memory_store)
        ms_btn.pack(side="left", padx=(12, 0))
        _Tooltip(ms_btn, "Memory Store: remember the current row selection, regardless of the active "
                 "Filter. Replaces whatever was remembered before.")
        mr_btn = ttk.Button(button_row, text="MR", width=4, command=self._splines_memory_recall)
        mr_btn.pack(side="left", padx=(4, 0))
        _Tooltip(mr_btn, "Memory Recall: re-select whatever MS last remembered. Only restores rows "
                 "that are currently visible under the active Filter -- switch the Filter back to "
                 "'All' first to recall a selection that spans multiple kinds.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Object Clusters", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        cluster_grid = ttk.Frame(parent)
        cluster_grid.pack(anchor="w", fill="x")
        fill_clusters_btn = ttk.Button(
            cluster_grid, text="Fill", command=self._open_cluster_fill_dialog,
        )
        fill_clusters_btn.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        _Tooltip(fill_clusters_btn, "Fill the selected spline(s) with object clusters -- pick "
                 "one or more nature assets (trees/bushes, rocks, grass, ground-cover/detail plants) "
                 "from asset_catalog.json and pack circular stamps across each selected area (largest "
                 "first, then progressively smaller passes fill in what's left, sized from that asset's "
                 "own measured planting density). Only Polygon/MultiPolygon splines can be filled.")
        clear_clusters_btn = ttk.Button(
            cluster_grid, text="Clear", command=self._clear_selected_cluster_fills,
        )
        clear_clusters_btn.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(clear_clusters_btn, "Remove every cluster-fill asset from the selected spline(s), "
                 "so Write Objects stops generating clusters for them.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        reg_marks_checkbox2 = ttk.Checkbutton(
            parent, text="Registration marks", variable=self.registration_marks_var,
        )
        reg_marks_checkbox2.pack(anchor="w")
        _Tooltip(reg_marks_checkbox2, "Add a small type-73 (circle) raise stamp and a matching 5m "
                 "circle spline (cart path surface) at each of the 4 course corners -- for visually "
                 "confirming in-game that terrain and splines land exactly where expected. Shared "
                 "with the same checkbox in the Terrain tab (one setting, both places).")
        self._add_step_button(parent, "Write Splines", self._run_write_splines)
        self._add_step_button(parent, "Write Holes", self._run_write_holes)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._build_cart_paths_section(parent)

    def _run_write_splines(self) -> None:
        wd = self._require_working_dir()
        if wd:
            args = ["--step", "write-splines"]
            if self.registration_marks_var.get():
                args.append("--registration-marks")
            self._run_step(args, wd)

    def _run_write_holes(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "write-holes"], wd)

    def _build_objects_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Theme:").pack(anchor="w")
        self._theme_name_to_id = {"(not set)": None}
        self._theme_name_to_id.update({name: theme_id for theme_id, name in THEMES_V2019.items()})
        self.objects_theme_var = tk.StringVar(value="(not set)")
        theme_box = ttk.Combobox(
            parent, textvariable=self.objects_theme_var, state="readonly", width=14,
            values=list(self._theme_name_to_id.keys()),
        )
        theme_box.pack(anchor="w", pady=(0, 8))
        _Tooltip(theme_box, "From the ingested .course (CourseDescription.json's theme / "
                 "CourseMetadata.json's courseTheme) -- controls which of the game's tree types are "
                 "available, same set for every game version. Leave as '(not set)' to use a single "
                 "generic tree type.")

        """
        Commenting this out for now.
        ttk.Label(parent, text="Asset List (.json):").pack(anchor="w")
        self.objects_asset_list_var = tk.StringVar(value="")
        asset_list_row = ttk.Frame(parent)
        asset_list_row.pack(anchor="w", fill="x", pady=(2, 8))
        asset_list_entry = ttk.Entry(asset_list_row, textvariable=self.objects_asset_list_var, width=24)
        asset_list_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(asset_list_row, text="...", width=3, command=self._browse_objects_asset_list).pack(side="left")
        _Tooltip(asset_list_entry, "Not wired up yet -- placeholder for a future 2021+ asset-path "
                 "list (will also cover building stakes; replaces hand-typing individual asset "
                 "paths one at a time).")
        """

        show_objects_row = ttk.Frame(parent)
        show_objects_row.pack(fill="x")
        self.show_objects_var = tk.BooleanVar(value=False)
        show_objects_checkbox = ttk.Checkbutton(
            show_objects_row, text="Show objects", variable=self.show_objects_var,
            command=self._show_preview,
        )
        show_objects_checkbox.pack(side="left")
        _Tooltip(show_objects_checkbox, "Overlay placed objects on the preview, similar to Overlay OSM "
                 "(Preview tab) -- individually placed trees (object_list.json, after Generate Trees) "
                 "as a 2px dot, or a circle sized to its measured LIDAR canopy radius if it has one; "
                 "rock/grass/ground-cover/display-plant cluster-fill stamps (placedObjects2.json, after "
                 "Write Objects) as a circle at that stamp's own packed radius. Colored by category, "
                 "fully opaque -- see the key drawn on the preview. Categories with no placed objects "
                 "yet just don't appear.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        obj_filter_row = ttk.Frame(parent)
        obj_filter_row.pack(fill="x")
        ttk.Label(obj_filter_row, text="Filter:").pack(side="left")
        self.objects_filter_var = tk.StringVar(value="All")
        self.objects_filter_box = ttk.Combobox(
            obj_filter_row, textvariable=self.objects_filter_var, state="readonly", width=14,
            values=self._OBJECT_SOURCE_FILTERS,
        )
        self.objects_filter_box.pack(side="left", padx=4)
        self.objects_filter_box.bind("<<ComboboxSelected>>", lambda e: self._refresh_objects_list())
        ttk.Button(obj_filter_row, text="Refresh", command=self._refresh_objects_list).pack(side="left")

        obj_tree_frame = ttk.Frame(parent)
        obj_tree_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.objects_tree = ttk.Treeview(
            obj_tree_frame, columns=("x", "z", "source", "detail"), show="headings",
            height=18, selectmode="extended",
        )
        self.objects_tree.heading("x", text="X")
        self.objects_tree.heading("z", text="Z")
        self.objects_tree.heading("source", text="Source")
        self.objects_tree.heading("detail", text="Detail")
        self.objects_tree.column("x", width=60, anchor="center")
        self.objects_tree.column("z", width=60, anchor="center")
        self.objects_tree.column("source", width=55, anchor="center")
        self.objects_tree.column("detail", width=110)
        self.objects_tree.pack(side="left", fill="both", expand=True)
        self.objects_tree.bind("<<TreeviewSelect>>", lambda e: self._on_object_selected())
        obj_tree_scroll = ttk.Scrollbar(obj_tree_frame, orient="vertical", command=self.objects_tree.yview)
        obj_tree_scroll.pack(side="left", fill="y")
        self.objects_tree["yscrollcommand"] = obj_tree_scroll.set

        obj_button_row = ttk.Frame(parent)
        obj_button_row.pack(fill="x", pady=(6, 0))
        clear_all_btn = ttk.Button(obj_button_row, text="Clear All", command=self._clear_filtered_objects)
        clear_all_btn.pack(side="left")
        _Tooltip(clear_all_btn, "Delete every tree currently shown (i.e. matching the active Filter) "
                 "from object_list.json permanently, then re-run Write Objects to pick up the change. "
                 "Set Filter to 'LIDAR' first to dump only auto-detected trees, keeping any hand-tagged "
                 "OSM ones -- useful for clearing out a bad detection run without losing curated data. "
                 "'Manual' (cluster fill) rows are never affected -- use Clear Cluster Fills in the "
                 "Splines tab for those.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Generate Trees", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        self.detect_lidar_trees_var = tk.BooleanVar(value=True)
        lidar_trees_checkbox = ttk.Checkbutton(
            parent, text="Detect trees from LIDAR canopy", variable=self.detect_lidar_trees_var,
        )
        lidar_trees_checkbox.pack(anchor="w", pady=(0, 4))
        _Tooltip(lidar_trees_checkbox, "Also detect individual trees directly from LIDAR canopy "
                 "points (ingest/tree_detection.py), on top of any OSM natural=tree nodes. Confined "
                 "to height_mask.geojson's core-play-area polygon if one exists -- the game's own "
                 "procedural vegetation fill is expected to handle everywhere else. Needs "
                 "heightmap.npz and pointcloud.npz (Ingest LAZ). On by default: OSM alone typically "
                 "finds few or no individually-tagged trees on a real course.")
        self._add_step_button(parent, "Generate Trees", self._run_generate_trees)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Stake Buildings", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        stake_grid = ttk.Frame(parent)
        stake_grid.pack(anchor="w", fill="x")
        stake_btn = ttk.Button(stake_grid, text="Place", command=self._run_stake_buildings)
        stake_btn.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        _Tooltip(stake_btn, "Places a subtle 0.5x-scale stake at every corner of every 'building' "
                 "feature (features.geojson) and re-writes placedObjects2.json -- same fence-post prop "
                 "used oversized for the cart-path debug marker. Needs Write Objects to have been run "
                 "at least once first (object_list.json). Persists as the default for future Write "
                 "Objects runs too -- use Clear Building Stakes to turn it back off.")
        clear_stakes_btn = ttk.Button(stake_grid, text="Clear", command=self._run_clear_building_stakes)
        clear_stakes_btn.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(clear_stakes_btn, "Re-writes placedObjects2.json without building stakes. Not "
                 "destructive -- building corners are recomputed from features.geojson each time, so "
                 "Stake Buildings can always bring them back.")
        
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Write Objects", self._run_write_objects)

    def _build_options_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Options", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        n_workers_row = ttk.Frame(parent)
        n_workers_row.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(n_workers_row, text="Workers:").pack(side="left")
        self.n_workers_var = tk.StringVar(value="")
        n_workers_entry = ttk.Entry(n_workers_row, textvariable=self.n_workers_var, width=6)
        n_workers_entry.pack(side="left", padx=4)
        _Tooltip(n_workers_entry, "contour method only (Terrain tab): parallelize the main "
                 "per-band loop across this many OS processes. Bands never spatially overlap, so "
                 "this is embarrassingly parallel -- output is byte-for-byte identical to a "
                 "sequential run given the same random seed, confirmed directly, only faster. "
                 "Blank = auto-detect via CPU count. 1 forces sequential (e.g. for debugging). "
                 "Forced to 1 regardless of this setting whenever max_stamps is set (that flag "
                 "needs a running total checked band-by-band, which is fundamentally sequential).")

        self.direct_height_shift_var = tk.BooleanVar(value=True)
        direct_shift_checkbox = ttk.Checkbutton(
            parent, text="Direct height shift", variable=self.direct_height_shift_var,
        )
        direct_shift_checkbox.pack(anchor="w", pady=(8, 0))
        _Tooltip(direct_shift_checkbox, "Write Terrain / Write Water (Refine tab): normalize final "
                 "heights by shifting every stamp's own value directly instead of appending a "
                 "course-wide shim stamp. Not a mathematically equivalent shift -- fixes the 'cheese "
                 "grater' stamp-seam artifact the shim-stamp method can produce. On by default; "
                 "uncheck to fall back to the shim-stamp method.")

    def _browse_objects_asset_list(self) -> None:
        f = filedialog.askopenfilename(
            title="Select an asset list (.json)", filetypes=[(".json files", "*.json"), ("All files", "*.*")]
        )
        if f:
            self.objects_asset_list_var.set(f)

    def _run_generate_trees(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return

        # Same snapshot-before-run fix as _run_refine_terrain, and for
        # the same reason: step_generate_trees loads whatever's
        # currently saved in height_mask.geojson (only consulted when
        # LIDAR detection is on -- that's the only thing the mask
        # confines), which otherwise reflects whichever buffer was set
        # the last time Ingest OSM (or a refine-terrain run) touched
        # that file, not necessarily what's actually on screen here.
        if self.detect_lidar_trees_var.get():
            merged_geom = self._get_cached_mask_merged_geometry(Path(wd))
            if merged_geom is not None:
                buffer_px = self.mask_buffer_preview_var.get()
                buffered = merged_geom.buffer(buffer_px)
                save_height_mask(buffered, Path(wd) / HEIGHT_MASK_FILE)
                self._append_log(
                    f"\n[snapshotting height_mask.geojson at buffer={buffer_px:.0f} px "
                    "before generate-trees]\n"
                )

        args = [
            "--step", "generate-trees",
            "--detect-lidar-trees" if self.detect_lidar_trees_var.get() else "--no-detect-lidar-trees",
        ]
        self._run_step(args, wd)
        self._refresh_objects_list()

    def _run_write_objects(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "write-objects", "--tree-variety"]
        theme_id = self._theme_name_to_id.get(self.objects_theme_var.get())
        if theme_id is not None:
            args += ["--theme", str(theme_id)]
        self._run_step(args, wd)
        self._refresh_objects_list()

    def _run_stake_buildings(self) -> None:
        self._run_write_objects_with_stakes(True)

    def _run_clear_building_stakes(self) -> None:
        self._run_write_objects_with_stakes(False)

    def _run_write_objects_with_stakes(self, stake_buildings: bool) -> None:
        """Shared by the Stake Buildings / Clear Building Stakes buttons --
        both are just a Write Objects run with --stake-buildings /
        --no-stake-buildings tacked on (see objects.py's
        build_building_stake_objects_v2019 / PGA2k_gen.py's
        step_write_objects), so building corners stay in lockstep with
        whatever's currently in features.geojson rather than needing
        separate placedObjects2.json surgery."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = [
            "--step", "write-objects", "--tree-variety",
            "--stake-buildings" if stake_buildings else "--no-stake-buildings",
        ]
        theme_id = self._theme_name_to_id.get(self.objects_theme_var.get())
        if theme_id is not None:
            args += ["--theme", str(theme_id)]
        self._run_step(args, wd)
        self._refresh_objects_list()

    @staticmethod
    def _object_source(tags: dict) -> str:
        """LIDAR-detected trees always carry TREE_RADIUS_TAG/TREE_HEIGHT_TAG
        (see objects.py's lidar_trees_to_tagged); anything else came from
        an OSM natural=tree node."""
        return "LIDAR" if TREE_RADIUS_TAG in tags or TREE_HEIGHT_TAG in tags else "OSM"

    @staticmethod
    def _object_detail(tags: dict) -> str:
        tree_type = tags.get(TREE_TYPE_TAG)
        height = tags.get(TREE_HEIGHT_TAG)
        radius = tags.get(TREE_RADIUS_TAG)
        parts = []
        if tree_type:
            parts.append(str(tree_type))
        if height:
            parts.append(f"h={float(height):.1f}m")
        if radius:
            parts.append(f"r={float(radius):.1f}m")
        return " ".join(parts)

    @staticmethod
    def _build_cluster_fill_rows(features: list) -> list[tuple[str, str, float, str]]:
        """
        (spline_osm_ids, asset_label, ratio, source) -- one row per
        distinct {"category","type","ratio","source"} fill spec seen
        across `features` (see course_output/object_clusters.py's
        PGA_CLUSTER_FILLS_TAG), with every contributing spline's osm_id
        collected together -- so a single Fill action (which can target
        many splines and several assets at once) shows as one row per
        asset/ratio/source combo, not one row per spline. `source`
        defaults to "manual" for specs saved before that field existed.
        """
        groups: dict[tuple, list[int]] = {}
        for f in features:
            if f.osm_id is None:
                continue
            for spec in f.tags.get(PGA_CLUSTER_FILLS_TAG) or []:
                source = spec.get("source", CLUSTER_FILL_SOURCE_MANUAL)
                key = (spec.get("category"), spec.get("type"), spec.get("ratio", DEFAULT_RASTER_RATIO), source)
                groups.setdefault(key, []).append(f.osm_id)

        rows = []
        for (category, type_, ratio, source), osm_ids in groups.items():
            label = _ASSET_LABEL_BY_KEY.get((category, type_), f"category={category}/type={type_}")
            ids_str = ",".join(str(i) for i in sorted(set(osm_ids)))
            rows.append((ids_str, label, ratio, source))
        return rows

    def _refresh_objects_list(self) -> None:
        """
        Populates the Objects tab list from two independent sources:
        object_list.json (individually placed trees -- iid is a plain
        int index into self._objects_tree_list, source OSM/LIDAR) and
        features.geojson's cluster-fill tags (area fills -- iid is
        "c"+index into self._cluster_fill_rows, source "manual" or
        "border"). Kept as two disjoint iid namespaces so
        _clear_filtered_objects (tree-only) can never mistake one for
        the other.
        """
        wd = self.working_dir.get().strip()
        self.objects_tree.delete(*self.objects_tree.get_children())
        self._objects_tree_list = []
        self._cluster_fill_rows = []
        # Old iids/indices are gone along with the rows above -- drop any
        # stale preview highlight state rather than let it dangle.
        self._highlighted_object_points = []
        self._highlighted_object_group_spline_ids = set()
        if not wd or not Path(wd).is_dir():
            return

        object_list_path = Path(wd) / OBJECT_LIST_FILE
        if object_list_path.exists():
            try:
                self._objects_tree_list = load_object_list(object_list_path)
            except (json.JSONDecodeError, OSError, KeyError):
                self._objects_tree_list = []

        features_path = Path(wd) / FEATURES_FILE
        if features_path.exists():
            try:
                self._cluster_fill_rows = self._build_cluster_fill_rows(load_features(features_path))
            except (json.JSONDecodeError, OSError, KeyError):
                self._cluster_fill_rows = []

        filter_val = self.objects_filter_var.get()
        for i, (x, z, tags) in enumerate(self._objects_tree_list):
            source = self._object_source(tags)
            if filter_val != "All" and source != filter_val:
                continue
            self.objects_tree.insert(
                "", "end", iid=str(i), values=(f"{x:.1f}", f"{z:.1f}", source, self._object_detail(tags)),
            )

        for j, (spline_ids, asset_label, ratio, source) in enumerate(self._cluster_fill_rows):
            if filter_val not in ("All", source.capitalize()):
                continue
            detail = f"splines={spline_ids} fill={asset_label} ratio={ratio:g}"
            self.objects_tree.insert("", "end", iid=f"c{j}", values=("", "", source, detail))

    def _clear_filtered_objects(self) -> None:
        """
        Deletes trees only -- cluster fills (source "manual" or
        "border", iid prefixed "c") are managed from the Splines tab's
        Clear Cluster Fills instead, since they live in features.geojson's
        tags, not object_list.json.
        """
        wd = self._require_working_dir()
        if not wd:
            return
        visible_iids = [iid for iid in self.objects_tree.get_children() if iid.isdigit()]
        if not visible_iids:
            messagebox.showinfo(
                "Nothing to clear", "No trees are currently shown for the active filter. (Manual cluster "
                "fills aren't cleared here -- use Clear Cluster Fills in the Splines tab.)",
            )
            return
        if not messagebox.askyesno(
            "Clear filtered trees",
            f"Permanently delete {len(visible_iids)} tree(s) currently shown (filter="
            f"{self.objects_filter_var.get()!r}) from object_list.json? This can't be undone -- "
            "re-run Generate Trees to get them back.",
        ):
            return
        remove_indices = {int(iid) for iid in visible_iids}
        self._objects_tree_list = [
            tree for i, tree in enumerate(self._objects_tree_list) if i not in remove_indices
        ]
        save_object_list(self._objects_tree_list, Path(wd) / OBJECT_LIST_FILE)
        self._append_log(f"\n[cleared {len(remove_indices)} tree(s) from {OBJECT_LIST_FILE}]\n")
        self._refresh_objects_list()

    def _refresh_splines_list(self) -> None:
        wd = self.working_dir.get().strip()
        self.splines_tree.delete(*self.splines_tree.get_children())
        if not wd or not Path(wd).is_dir():
            self._splines_features = []
            self._splines_features_mtime = None
            return

        features_path = Path(wd) / FEATURES_FILE
        if not features_path.exists():
            self._splines_features = []
            self._splines_features_mtime = None
            return

        self._splines_features = load_features(features_path)
        self._splines_features_mtime = features_path.stat().st_mtime
        kind_filter = self.splines_kind_filter_var.get()
        for f in self._splines_features:
            if kind_filter != "All" and f.kind != kind_filter:
                continue
            if f.osm_id is None:
                continue  # nothing stable to select/highlight/toggle by
            self.splines_tree.insert(
                "", "end", iid=str(f.osm_id),
                values=(f.kind, _spline_tag_detail(f), "✔" if f.mask else "", _spline_cluster_detail(f)),
            )

    def _on_spline_selected(self) -> None:
        """
        <<TreeviewSelect>> fires once PER CLICK during a multi-select
        gesture (ctrl-click, shift-click extending a range) -- not
        once for the whole final selection. Calling _show_preview()
        (a real image redraw) synchronously here would redraw once per
        click while building up a multi-select, before the mask toggle
        the user's actually going for even runs. Debounced instead:
        each call cancels any still-pending redraw and schedules a new
        one a short delay out, so a rapid run of clicks collapses into
        a single redraw once selection actually settles.
        """
        selection = self.splines_tree.selection()
        self._highlighted_feature_osm_ids = {int(s) for s in selection}
        if self._selection_preview_job is not None:
            self.root.after_cancel(self._selection_preview_job)
        self._selection_preview_job = self.root.after(150, self._show_preview_after_selection)

    def _show_preview_after_selection(self) -> None:
        self._selection_preview_job = None
        self._show_preview()

    def _on_object_selected(self) -> None:
        """
        Objects-tab counterpart to _on_spline_selected -- same debounce
        (and same shared self._selection_preview_job/_show_preview_after_selection),
        since only one of the two trees is ever the one the user just
        clicked in. A plain object/tree row (iid is a digit, indexing
        self._objects_tree_list) has a concrete (x, z) to ring-highlight
        directly in _composite_objects_layer. A cluster-fill "group" row
        (iid "c"+index into self._cluster_fill_rows) has no such fixed
        points -- object_clusters.py's packed positions carry no
        back-reference to the spline(s) that produced them -- so instead
        its source spline osm_ids feed into the SAME spline-highlight
        overlay _on_spline_selected drives, unioned rather than
        overwritten so a Splines-tab selection isn't clobbered.
        """
        selection = self.objects_tree.selection()
        points: list[tuple[float, float]] = []
        spline_ids: set[int] = set()
        for iid in selection:
            if iid.isdigit():
                idx = int(iid)
                if 0 <= idx < len(self._objects_tree_list):
                    x, z, _tags = self._objects_tree_list[idx]
                    points.append((x, z))
            elif iid.startswith("c"):
                try:
                    j = int(iid[1:])
                except ValueError:
                    continue
                if 0 <= j < len(self._cluster_fill_rows):
                    ids_str = self._cluster_fill_rows[j][0]
                    spline_ids.update(int(s) for s in ids_str.split(",") if s)
        self._highlighted_object_points = points
        self._highlighted_object_group_spline_ids = spline_ids
        if self._selection_preview_job is not None:
            self.root.after_cancel(self._selection_preview_job)
        self._selection_preview_job = self.root.after(150, self._show_preview_after_selection)

    def _regenerate_height_mask(self, working_dir: Path) -> None:
        """
        Rebuild height_mask.geojson from the currently in-memory
        feature list (self._splines_features) right away, rather than
        waiting for the next Ingest OSM run -- so a mask toggle in the
        Splines tab is reflected immediately, matching how the live
        buffer-preview slider already behaves. Also invalidates the
        slider's own cached merged geometry, since the mask membership
        it was built from just changed.
        """
        mask_geometry = build_height_mask(
            self._splines_features, buffer_px=self.mask_buffer_preview_var.get(),
        )
        save_height_mask(mask_geometry, working_dir / HEIGHT_MASK_FILE)
        self._cached_mask_merged_geom = None
        self._cached_mask_geom_key = None

    def _toggle_selected_mask(self) -> None:
        """
        Toggle mask for every currently-selected row (multi-select, so
        this can be several at once) -- unlike Toggle All, not scoped
        to golf-object kinds only, since the user explicitly selected
        these themselves. Same select-all/deselect-all pattern: if any
        selected feature is currently unmasked, masks all of them in;
        otherwise masks all of them out.
        """
        wd = self.working_dir.get().strip()
        selected_ids = {int(s) for s in self.splines_tree.selection()}
        if not wd or not selected_ids:
            return
        targets = [f for f in self._splines_features if f.osm_id in selected_ids]
        if not targets:
            return
        new_state = any(not f.mask for f in targets)
        for f in targets:
            f.mask = new_state
        save_features(self._splines_features, Path(wd) / FEATURES_FILE)
        self._regenerate_height_mask(Path(wd))
        self._refresh_splines_list()
        # One batched call, not one .selection_add() per id: each
        # individual call fires its own separate <<TreeviewSelect>>
        # event (Tkinter doesn't coalesce these), so restoring a
        # selection of hundreds of splines one at a time queued
        # hundreds of events -- each triggering a selection readback
        # that itself grows more expensive as the loop progressed
        # (O(n) per read, O(n^2) total) -- which is what was actually
        # locking up the UI on a large multi-select, not the toggle
        # logic itself (already single-pass) or even _show_preview
        # (already debounced, see _on_spline_selected).
        restorable_ids = [str(i) for i in selected_ids if self.splines_tree.exists(str(i))]
        if restorable_ids:
            self.splines_tree.selection_set(restorable_ids)
        self._show_preview()

    def _toggle_all_mask(self) -> None:
        """
        Toggle mask for every feature currently visible in the tree
        (i.e. matching the active kind filter) -- same select-all/
        deselect-all pattern as _toggle_selected_mask: if any visible
        feature is currently unmasked, masks all of them in; otherwise
        masks all of them out.

        Previously restricted to golf-object kinds only (fairway/
        green/tee/hole), deliberately asymmetric with
        _toggle_selected_mask (which never had that restriction).
        Removed: with the "All" filter active, that asymmetry meant
        most of what's actually visible (bunker/rough/water/cartpath/
        etc.) silently never got touched, which reads as "Toggle All
        doesn't work" rather than as the intentional restriction it
        was.
        """
        wd = self.working_dir.get().strip()
        if not wd:
            return
        visible_ids = {int(iid) for iid in self.splines_tree.get_children()}
        targets = [f for f in self._splines_features if f.osm_id in visible_ids]
        if not targets:
            return
        new_state = any(not f.mask for f in targets)
        for f in targets:
            f.mask = new_state
        save_features(self._splines_features, Path(wd) / FEATURES_FILE)
        self._regenerate_height_mask(Path(wd))
        self._refresh_splines_list()
        self._show_preview()

    _SYNTHETIC_CLUSTER_KINDS = (SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND)

    def _clear_selected_cluster_fills(self) -> None:
        """
        Remove PGA_CLUSTER_FILLS_TAG from every currently-selected row
        (multi-select). For a real spline (kind not one of
        _SYNTHETIC_CLUSTER_KINDS) that just strips the tag, same as
        always. For a border/masked-manual row, the underlying Feature
        exists ONLY to carry that tag (see _open_cluster_fill_dialog) --
        stripping the tag would leave a permanent, purposeless row, so
        the whole Feature is deleted instead.
        """
        wd = self.working_dir.get().strip()
        selected_ids = {int(s) for s in self.splines_tree.selection()}
        if not wd or not selected_ids:
            return
        targets = [f for f in self._splines_features if f.osm_id in selected_ids]
        if not targets:
            return
        changed = False
        remove_ids = set()
        for f in targets:
            if f.kind in self._SYNTHETIC_CLUSTER_KINDS:
                remove_ids.add(f.osm_id)
                changed = True
            elif f.tags.pop(PGA_CLUSTER_FILLS_TAG, None) is not None:
                changed = True
        if not changed:
            return
        if remove_ids:
            self._splines_features = [f for f in self._splines_features if f.osm_id not in remove_ids]
        save_features(self._splines_features, Path(wd) / FEATURES_FILE)
        self._refresh_splines_list()
        restorable_ids = [
            str(i) for i in selected_ids if i not in remove_ids and self.splines_tree.exists(str(i))
        ]
        if restorable_ids:
            self.splines_tree.selection_set(restorable_ids)

    def _open_cluster_fill_dialog(self) -> None:
        """
        "Fill with Clusters...": pick one or more nature assets (see
        course_output/asset_catalog.py) and a raster ratio, then create
        a {"category","type","ratio","source"} spec per chosen asset
        (see course_output/object_clusters.py for what actually
        consumes these tags at write-objects time -- it treats any
        tagged Feature identically regardless of "source" or how its
        geometry was built).

        Two modes, chosen by the Splines tab's "Border" checkbox
        (self.mask_border_var), decided once here before the dialog
        opens:

        - Manual (default): specs are appended directly to every
          currently-selected spline's own tags, same as always -- unless
          "Use mask" is checked in the dialog, in which case each
          selected spline instead gets a NEW synthetic Feature (kind
          SYNTHETIC_MASKED_KIND) carrying its geometry intersected with
          the current mask region, so the real spline's own geometry
          (shared with splines/holes/water output) is never mutated.

        - Border: the current Mask/Source/Buffer(px)/Border-width
          controls build a ring polygon ONCE, right now (while marked/
          selected state is still available), wrapped in a single new
          synthetic Feature (kind SYNTHETIC_BORDER_KIND) carrying the
          fill specs -- see _mask_geometry_full_frame's docstring for
          why this has to happen now rather than being deferred to
          write-objects time.

        Splines/mask state are captured up front, before the dialog
        steals focus -- the dialog's own asset Treeview otherwise makes
        it easy to lose track of what was selected in the main list.
        """
        wd = self.working_dir.get().strip()
        if not wd:
            messagebox.showwarning("No working directory", "Set a working directory first.")
            return

        border_mode = self.mask_border_var.get()
        selected_ids = {int(s) for s in self.splines_tree.selection()}
        targets: list[Feature] = []
        area_targets: list[Feature] = []
        mask_geom = None

        if border_mode:
            mask_geom = self._mask_geometry_full_frame(Path(wd))
            if mask_geom is None or mask_geom.is_empty:
                messagebox.showwarning(
                    "No mask geometry",
                    f"No '{self.mask_source_var.get()}' splines found to build a border from -- "
                    "mark or select some splines first (see the Source dropdown above the Splines list).",
                )
                return
        else:
            if not selected_ids:
                messagebox.showwarning("No splines selected", "Select one or more splines in the list first.")
                return
            targets = [f for f in self._splines_features if f.osm_id in selected_ids]
            area_targets = [f for f in targets if f.geometry.geom_type in ("Polygon", "MultiPolygon")]
            if not area_targets:
                messagebox.showwarning(
                    "No fillable splines", "None of the selected splines have polygon area to fill "
                    "(cluster fill needs a Polygon/MultiPolygon spline, not a bare line).",
                )
                return
        if not CLUSTERABLE_ENTRIES:
            messagebox.showinfo(
                "No clusterable assets", "course_output/asset_catalog.json has no entries with both "
                "a category cluster_radius and a measured spacing yet -- nothing to fill with.",
            )
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Fill with Border Clusters" if border_mode else "Fill with Clusters")
        dialog.transient(self.root)
        dialog.grab_set()

        filter_row = ttk.Frame(dialog)
        filter_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(filter_row, text="Category:").pack(side="left")
        category_names = ["All"] + [ASSET_CATEGORIES[cid].description for cid in sorted(NATURE_CATEGORY_IDS)]
        category_var = tk.StringVar(value="All")
        category_box = ttk.Combobox(
            filter_row, textvariable=category_var, state="readonly", width=20, values=category_names,
        )
        category_box.pack(side="left", padx=4)

        tree_frame = ttk.Frame(dialog)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=4)
        asset_tree = ttk.Treeview(
            tree_frame, columns=("category", "asset"), show="headings", height=14, selectmode="extended",
        )
        asset_tree.heading("category", text="Category")
        asset_tree.heading("asset", text="Asset")
        asset_tree.column("category", width=140)
        asset_tree.column("asset", width=260)
        asset_tree.pack(side="left", fill="both", expand=True)
        asset_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=asset_tree.yview)
        asset_scroll.pack(side="left", fill="y")
        asset_tree["yscrollcommand"] = asset_scroll.set

        def refresh_asset_tree() -> None:
            asset_tree.delete(*asset_tree.get_children())
            chosen = category_var.get()
            for i, entry in enumerate(CLUSTERABLE_ENTRIES):
                description = ASSET_CATEGORIES[entry.category].description
                if chosen != "All" and description != chosen:
                    continue
                asset_tree.insert("", "end", iid=str(i), values=(description, entry.label))

        category_box.bind("<<ComboboxSelected>>", lambda e: refresh_asset_tree())
        refresh_asset_tree()

        ratio_row = ttk.Frame(dialog)
        ratio_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(ratio_row, text="Raster ratio:").pack(side="left")
        ratio_var = tk.StringVar(value=str(DEFAULT_RASTER_RATIO))
        ratio_entry = ttk.Entry(ratio_row, textvariable=ratio_var, width=6)
        ratio_entry.pack(side="left", padx=4)
        _Tooltip(ratio_entry, "How much placed stamps are allowed to overlap each other during packing "
                 "-- minimum center-to-center separation is (r1+r2) x ratio. 1.0 means stamps may only "
                 "just touch; <1 lets them overlap more (denser fill); >1 spaces them further apart. "
                 "Applies to every asset picked in this dialog, across all 3 packing passes.")

        use_mask_var = tk.BooleanVar(value=False)
        if border_mode:
            ttk.Label(
                dialog, text=f"Border ring from '{self.mask_source_var.get()}' mask "
                f"(Buffer (px)={self.mask_buffer_preview_var.get():g}, "
                f"Border width (m)={self.border_width_var.get()})",
            ).pack(anchor="w", padx=8, pady=(0, 4))
        else:
            use_mask_row = ttk.Frame(dialog)
            use_mask_row.pack(fill="x", padx=8, pady=(0, 4))
            use_mask_checkbox = ttk.Checkbutton(use_mask_row, text="Use mask", variable=use_mask_var)
            use_mask_checkbox.pack(side="left")
            _Tooltip(use_mask_checkbox, "Restrict each selected spline's fill area to its intersection "
                     "with the current mask region (Source/Buffer (px) above), instead of filling the "
                     "whole spline. The real spline geometry is never modified -- a new tagged copy is "
                     "created for the clipped area, the same synthetic-Feature mechanism 'Border' fills "
                     "use.")

        def do_fill() -> None:
            selected_rows = asset_tree.selection()
            if not selected_rows:
                messagebox.showwarning("No asset selected", "Select one or more assets to fill with.", parent=dialog)
                return
            try:
                ratio = float(ratio_var.get())
                if ratio <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid ratio", "Raster ratio must be a positive number.", parent=dialog)
                return

            chosen_entries = [CLUSTERABLE_ENTRIES[int(i)] for i in selected_rows]

            if border_mode:
                try:
                    border_width = float(self.border_width_var.get())
                    if border_width <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showwarning(
                        "Invalid border width", "Border width (m) must be a positive number.", parent=dialog,
                    )
                    return
                ring = build_border_ring_geometry(mask_geom, border_width)
                if ring.is_empty:
                    messagebox.showwarning(
                        "Empty ring", "The border ring came out empty -- try a larger Border width (m) "
                        "or a wider Buffer (px), or check that the mask region isn't degenerate.",
                        parent=dialog,
                    )
                    return
                specs = [
                    {"category": e.category, "type": e.type, "ratio": ratio, "source": CLUSTER_FILL_SOURCE_BORDER}
                    for e in chosen_entries
                ]
                border_feature = Feature(
                    geometry=ring, kind=SYNTHETIC_BORDER_KIND, tags={PGA_CLUSTER_FILLS_TAG: specs},
                    osm_id=self._next_synthetic_osm_id(), mask=True,
                )
                self._splines_features.append(border_feature)
                save_features(self._splines_features, Path(wd) / FEATURES_FILE)
                self._append_log(
                    f"\n[created border ring (source={self.mask_source_var.get()}, width={border_width}) "
                    f"filled with {len(chosen_entries)} asset(s) at ratio={ratio}]\n"
                )
            else:
                specs = [
                    {"category": e.category, "type": e.type, "ratio": ratio, "source": CLUSTER_FILL_SOURCE_MANUAL}
                    for e in chosen_entries
                ]
                if use_mask_var.get():
                    live_mask_geom = self._mask_geometry_full_frame(Path(wd))
                    if live_mask_geom is None or live_mask_geom.is_empty:
                        messagebox.showwarning(
                            "No mask geometry",
                            f"No '{self.mask_source_var.get()}' splines found to mask against -- mark "
                            "or select some splines first (see the Source dropdown above the Splines "
                            "list), or uncheck 'Use mask'.", parent=dialog,
                        )
                        return
                    created = 0
                    for f in area_targets:
                        clipped = f.geometry.intersection(live_mask_geom)
                        if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
                            continue
                        synth = Feature(
                            geometry=clipped, kind=SYNTHETIC_MASKED_KIND,
                            tags={PGA_CLUSTER_FILLS_TAG: [dict(s) for s in specs]},
                            osm_id=self._next_synthetic_osm_id(), mask=True,
                        )
                        self._splines_features.append(synth)
                        created += 1
                    save_features(self._splines_features, Path(wd) / FEATURES_FILE)
                    self._append_log(
                        f"\n[masked-filled {created}/{len(area_targets)} spline(s) with "
                        f"{len(chosen_entries)} asset(s) at ratio={ratio}]\n"
                    )
                else:
                    for f in area_targets:
                        fills = f.tags.setdefault(PGA_CLUSTER_FILLS_TAG, [])
                        fills.extend(dict(s) for s in specs)
                    save_features(self._splines_features, Path(wd) / FEATURES_FILE)
                    skipped = len(targets) - len(area_targets)
                    note = f" ({skipped} non-area spline(s) skipped)" if skipped else ""
                    self._append_log(
                        f"\n[filled {len(area_targets)} spline(s) with {len(chosen_entries)} asset(s) "
                        f"at ratio={ratio}{note}]\n"
                    )

            self._refresh_splines_list()
            restorable_ids = [str(i) for i in selected_ids if self.splines_tree.exists(str(i))]
            if restorable_ids:
                self.splines_tree.selection_set(restorable_ids)
            dialog.destroy()

        button_row = ttk.Frame(dialog)
        button_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(button_row, text="Fill", command=do_fill).pack(side="left")
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side="left", padx=(4, 0))

    def _splines_memory_store(self) -> None:
        """MS: remember the current row selection (by osm_id), independent of the active Filter."""
        self._splines_selection_memory = set(self.splines_tree.selection())
        self._append_log(f"\n[MS: remembered {len(self._splines_selection_memory)} row(s)]\n")

    def _splines_memory_recall(self) -> None:
        """
        MR: re-select whatever MS last remembered. Only ever restores
        rows that currently exist in the tree (i.e. match the active
        Filter) -- a remembered row hidden by the current filter is
        silently skipped, not an error, same as _toggle_selected_mask's
        own restore-selection step. One batched selection_set() call,
        not one .selection_add() per row (see that same method's
        docstring for why: each individual call fires its own separate
        <<TreeviewSelect>> event).
        """
        if not self._splines_selection_memory:
            return
        restorable = [iid for iid in self._splines_selection_memory if self.splines_tree.exists(iid)]
        if restorable:
            self.splines_tree.selection_set(restorable)
        self._append_log(
            f"\n[MR: recalled {len(restorable)} of {len(self._splines_selection_memory)} "
            "remembered row(s) (rest hidden by the current filter)]\n"
        )

    def _add_step_button(self, parent: ttk.Frame, label: str, command) -> ttk.Button:
        btn = ttk.Button(parent, text=label, command=command, width=22)
        btn.pack(anchor="w", pady=2)
        return btn

    # ------------------------------------------------------------------
    # Working directory / course name: load & persist via project.json
    # ------------------------------------------------------------------

    def _on_working_dir_changed(self) -> None:
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        self._cached_mask_merged_geom = None
        self._cached_mask_geom_key = None
        project = load_project(Path(wd))
        self._suppress_course_name_save = True
        try:
            self.course_name.set(project.get("course_name", ""))
        finally:
            self._suppress_course_name_save = False
        self._suppress_repack_filename_save = True
        try:
            self.repack_filename_var.set(project.get("repack_filename", ""))
        finally:
            self._suppress_repack_filename_save = False
        self._suppress_game_version_save = True
        try:
            self.game_version.set(project.get("game_version", DEFAULT_GAME_VERSION))
        finally:
            self._suppress_game_version_save = False
        self._suppress_objects_theme_save = True
        try:
            theme_name = THEMES_V2019.get(project.get("objects_theme"), "(not set)")
            self.objects_theme_var.set(theme_name)
        finally:
            self._suppress_objects_theme_save = False
        self._refresh_refine_stats()
        self._refresh_preview_and_slider()

    def _on_repack_filename_changed(self) -> None:
        if self._suppress_repack_filename_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        save_project(Path(wd), {"repack_filename": self.repack_filename_var.get()})

    def _on_course_name_changed(self) -> None:
        if self._suppress_course_name_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        save_project(Path(wd), {"course_name": self.course_name.get()})

    def _on_game_version_changed(self) -> None:
        """
        game_version is a project-level setting, same tier as
        course_name -- not tied to any one step, since (per the
        conversation that established this) userLayers.py/splines.py/
        objects.py will all eventually need to target it for "write"
        and "move"/repack steps alike. Saved immediately on change,
        same pattern as course_name, rather than only being passed as
        a per-step CLI flag.
        """
        if self._suppress_game_version_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        save_project(Path(wd), {"game_version": self.game_version.get()})

    def _on_objects_theme_changed(self) -> None:
        if self._suppress_objects_theme_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        theme_id = self._theme_name_to_id.get(self.objects_theme_var.get())
        save_project(Path(wd), {"objects_theme": theme_id})

    # ------------------------------------------------------------------
    # Folder / file pickers
    # ------------------------------------------------------------------

    def _browse_working_dir(self) -> None:
        d = filedialog.askdirectory(title="Select working directory")
        if d:
            self.working_dir.set(d)

    def _browse_course_file(self) -> None:
        f = filedialog.askopenfilename(
            title="Select a .course file", filetypes=[(".course files", "*.course"), ("All files", "*.*")]
        )
        if f:
            self.course_file_var.set(f)

    # ------------------------------------------------------------------
    # Step commands -- each just builds a CLI arg list and hands off to
    # _run_step, which does the actual subprocess work.
    # ------------------------------------------------------------------

    def _require_working_dir(self):
        wd = self.working_dir.get().strip()
        if not wd:
            messagebox.showwarning("No working directory", "Set a working directory first.")
            return None
        return Path(wd)

    def _run_init(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "init"], wd)

    def _run_ingest_laz(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "ingest-laz"]
        proj = self.projection_var.get().strip()
        if proj:
            args += ["--projection", proj]
        if not self.fill_heightmap_gaps_var.get():
            args += ["--no-fill-heightmap-gaps"]
        self._run_step(args, wd)

    def _run_ingest_osm(self) -> None:
        wd = self._require_working_dir()
        if wd:
            args = [
                "--step", "ingest-osm",
                "--height-mask-buffer-px", f"{self.mask_buffer_preview_var.get():.0f}",
            ]
            hole_corridor_buffer = self.hole_corridor_buffer_var.get().strip()
            if hole_corridor_buffer:
                args += ["--hole-corridor-buffer-px", hole_corridor_buffer]
            self._run_step(args, wd)

    def _run_dig_water(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        depth = self.dig_water_depth_var.get().strip()
        buffer = self.dig_water_buffer_var.get().strip()
        args = ["--step", "dig-water"]
        if depth:
            args += ["--dig-depth", depth]
        if buffer:
            args += ["--dig-buffer", buffer]
        self._run_step(args, wd)

    def _run_ingest_course(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        course_file = self.course_file_var.get().strip()
        if not course_file:
            messagebox.showwarning("No course file", "Choose a .course file to ingest first.")
            return
        self._run_step(["--step", "ingest-course", "--course-file", course_file], wd)

    def _run_generate_terrain(self) -> None:
        wd = self._require_working_dir()
        if wd:
            # Same snapshot-before-run fix as _run_refine_terrain, and for
            # the same reason: step_generate_terrain loads whatever's
            # currently saved in height_mask.geojson, which otherwise
            # reflects whichever buffer was set the last time Ingest OSM
            # (or a refine-terrain run) touched that file, not necessarily
            # what's actually on screen here.
            if self.generate_terrain_use_height_mask_var.get():
                merged_geom = self._get_cached_mask_merged_geometry(Path(wd))
                if merged_geom is not None:
                    buffer_px = self.mask_buffer_preview_var.get()
                    buffered = merged_geom.buffer(buffer_px)
                    save_height_mask(buffered, Path(wd) / HEIGHT_MASK_FILE)
                    self._append_log(
                        f"\n[snapshotting height_mask.geojson at buffer={buffer_px:.0f} px "
                        "before generate-terrain]\n"
                    )

            args = ["--step", "generate-terrain",
                     "--generate-terrain-method", self.generate_terrain_method_var.get()]
            pitch = self.pitch_var.get().strip()
            if pitch:
                args += ["--pitch", pitch]
            hex_spread_ratio = self.hex_spread_ratio_var.get().strip()
            if hex_spread_ratio:
                args += ["--hex-spread-ratio", hex_spread_ratio]
            hex_brush = self.hex_brush_var.get().strip()
            if hex_brush:
                args += ["--hex-brush", hex_brush]
            args += ["--hex-tool", "1" if self.hex_tool_var.get() == "raise" else "0"]
            raster_size = self.raster_size_var.get().strip()
            if raster_size:
                args += ["--raster-size", raster_size]
            raster_spread_ratio = self.raster_spread_ratio_var.get().strip()
            if raster_spread_ratio:
                args += ["--raster-spread-ratio", raster_spread_ratio]
            raster_center_bias_ratio_x = self.raster_center_bias_ratio_x_var.get().strip()
            if raster_center_bias_ratio_x:
                args += ["--raster-center-bias-ratio-x", raster_center_bias_ratio_x]
            raster_center_bias_ratio_z = self.raster_center_bias_ratio_z_var.get().strip()
            if raster_center_bias_ratio_z:
                args += ["--raster-center-bias-ratio-z", raster_center_bias_ratio_z]
            raster_brush = self.raster_brush_var.get().strip()
            if raster_brush:
                args += ["--raster-brush", raster_brush]
            band_spacing = self.band_spacing_var.get().strip()
            if band_spacing:
                args += ["--band-spacing-m", band_spacing]
            fill_mode = self.fill_mode_var.get().strip()
            if fill_mode:
                args += ["--fill-mode", fill_mode]
            fill_brush = self.fill_brush_var.get().strip()
            if fill_brush:
                args += ["--fill-brush", fill_brush]
            min_radius = self.min_radius_var.get().strip()
            if min_radius:
                args += ["--min-radius", min_radius]
            max_radius = self.max_radius_var.get().strip()
            if max_radius:
                args += ["--max-radius", max_radius]
            radius_step = self.radius_step_var.get().strip()
            if radius_step:
                args += ["--radius-step-ratio", radius_step]
            edge_distance = self.edge_distance_var.get().strip()
            if edge_distance:
                args += ["--edge-distance-m", edge_distance]
            rect_brush = self.rect_brush_var.get().strip()
            if rect_brush:
                args += ["--rect-brush", rect_brush]
            rect_tolerance = self.rect_tolerance_var.get().strip()
            if rect_tolerance:
                args += ["--rect-tolerance-m", rect_tolerance]
            rect_min_length = self.rect_min_length_var.get().strip()
            if rect_min_length:
                args += ["--rect-min-length-m", rect_min_length]
            rect_max_search_distance = self.rect_max_search_distance_var.get().strip()
            if rect_max_search_distance:
                args += ["--rect-max-search-distance-m", rect_max_search_distance]
            rect_width_samples = self.rect_width_samples_var.get().strip()
            if rect_width_samples:
                args += ["--rect-width-samples", rect_width_samples]
            smoothing_brush = self.smoothing_brush_var.get().strip()
            if smoothing_brush:
                args += ["--smoothing-brush", smoothing_brush]
            smoothing_min_radius = self.smoothing_min_radius_var.get().strip()
            if smoothing_min_radius:
                args += ["--smoothing-min-radius", smoothing_min_radius]
            smooth_ratio = self.smooth_ratio_var.get().strip()
            if smooth_ratio:
                args += ["--smooth-ratio", smooth_ratio]
            smooth_claim_fraction = self.smooth_claim_fraction_var.get().strip()
            if smooth_claim_fraction:
                args += ["--smooth-claim-fraction", smooth_claim_fraction]
            args.append(
                "--generate-terrain-secondary-fill" if self.enable_secondary_fill_var.get()
                else "--no-generate-terrain-secondary-fill"
            )
            candidates_per_radius = self.candidates_per_radius_var.get().strip()
            if candidates_per_radius:
                args += ["--candidates-per-radius", candidates_per_radius]
            sweet_spot_ratio = self.sweet_spot_ratio_var.get().strip()
            if sweet_spot_ratio:
                args += ["--sweet-spot-ratio", sweet_spot_ratio]
            sweet_spot_sample_bands = self.sweet_spot_sample_bands_var.get().strip()
            if sweet_spot_sample_bands:
                args += ["--sweet-spot-sample-bands", sweet_spot_sample_bands]
            sweet_spot_seeds = self.sweet_spot_seeds_var.get().strip()
            if sweet_spot_seeds:
                args += ["--sweet-spot-seeds", sweet_spot_seeds]
            # MAX candidates / TIME budget are no longer user-tunable -- the
            # GUI's old defaults (50000 / 60s) matched terrain/contour_layers.py's
            # own DEFAULT_SWEET_SPOT_MAX_CANDIDATES/_TIME_BUDGET_S exactly, so
            # simply not passing them lets the backend defaults apply. SEED is
            # gone the same way: no reproducible fixed seed to set, just a
            # fresh random one every run.
            args += ["--random-seed", str(random.randint(1, 2**31 - 1))]
            denoise_px = self.denoise_px_var.get().strip()
            if denoise_px:
                args += ["--denoise-px", denoise_px]
            max_stamps = self.max_stamps_var.get().strip()
            if max_stamps:
                args += ["--max-stamps", max_stamps]
            args.append(
                "--generate-terrain-use-height-mask" if self.generate_terrain_use_height_mask_var.get()
                else "--no-generate-terrain-use-height-mask"
            )
            if self.generate_terrain_use_height_mask_var.get():
                args += ["--generate-terrain-mask-buffer-px", f"{self.mask_buffer_preview_var.get():.0f}"]
            n_workers = self.n_workers_var.get().strip()
            if n_workers:
                args += ["--n-workers", n_workers]
            self._run_step(args, wd)

    def _browse_cart_path_splines(self) -> None:
        path = filedialog.askopenfilename(
            title="Select cart path splines JSON", filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.cart_path_splines_path_var.set(path)

    def _run_generate_cart_paths(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return

        surface = self.cart_path_surface_var.get().strip()
        if not surface:
            messagebox.showerror(
                "Missing SURFACE", "SURFACE must be set -- it's the value identifying a cart path "
                "spline in the source JSON, and there's no safe default to fall back to.",
            )
            return

        args = ["--step", "generate-cart-paths", "--cart-path-surface", surface]

        splines_path = self.cart_path_splines_path_var.get().strip()
        if splines_path:
            args += ["--splines-path", splines_path]
        stamp_radius = self.cart_path_stamp_radius_var.get().strip()
        if stamp_radius:
            args += ["--cart-path-stamp-radius", stamp_radius]
        spacing = self.cart_path_spacing_var.get().strip()
        if spacing:
            args += ["--cart-path-spacing", spacing]
        height_avg_radius = self.cart_path_height_avg_radius_var.get().strip()
        if height_avg_radius:
            args += ["--cart-path-height-avg-radius", height_avg_radius]

        self._run_step(args, wd)

    def _validate_refine_fields(self) -> bool:
        """Highlight (in red) any required Refine Terrain field left empty; returns True if all are filled."""
        field_vars = {
            "tolerance": self.tolerance_var,
            "resolution": self.resolution_var,
            "min_hotspot": self.min_hotspot_radius_cells_var,
            "spread_ratio": self.spread_ratio_var,
            "claim_fraction": self.claim_fraction_var,
            "rad": self.rad_var,
        }
        all_valid = True
        for key, label in self.refine_labels.items():
            if field_vars[key].get().strip():
                label.configure(foreground="black")
            else:
                label.configure(foreground="red")
                all_valid = False
        return all_valid

    def _run_refine_terrain(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        if not self._validate_refine_fields():
            messagebox.showwarning(
                "Missing required fields",
                "Fill in all required Refine Terrain fields (shown in red) before running.",
            )
            return

        # If the mask buffer preview has been used, snapshot its current
        # value into height_mask.geojson before running -- otherwise
        # --use-height-mask would silently use whatever buffer was set
        # the last time Ingest OSM ran (likely stale/different from
        # whatever was just previewed), not what's actually on screen.
        if self.use_height_mask_var.get():
            merged_geom = self._get_cached_mask_merged_geometry(Path(wd))
            if merged_geom is not None:
                buffer_px = self.mask_buffer_preview_var.get()
                buffered = merged_geom.buffer(buffer_px)
                save_height_mask(buffered, Path(wd) / HEIGHT_MASK_FILE)
                self._append_log(
                    f"\n[snapshotting height_mask.geojson at buffer={buffer_px:.0f} px "
                    "before refine-terrain]\n"
                )

        args = [
            "--step", "refine-terrain",
            "--method", self.refine_method_var.get(),
            "--error-tolerance", self.tolerance_var.get().strip(),
            "--resolution", self.resolution_var.get().strip(),
            "--min-hotspot-radius-cells", self.min_hotspot_radius_cells_var.get().strip(),
            "--brush-radius-spread-ratio", self.spread_ratio_var.get().strip(),
            "--claim-radius-fraction", self.claim_fraction_var.get().strip(),
            "--rad-m", self.rad_var.get().strip(),
            "--use-height-mask" if self.use_height_mask_var.get() else "--no-use-height-mask",
        ]
        if self.use_height_mask_var.get():
            args += ["--mask-buffer-px", f"{self.mask_buffer_preview_var.get():.0f}"]
        selected_brushes = [b for b in self.BRUSH_TYPE_ORDER if self.brush_type_vars[b].get()]
        args += ["--candidate-brushes", ",".join(str(b) for b in selected_brushes)]
        max_new = self.max_new_var.get().strip()
        if max_new:
            args += ["--max-new-stamps", max_new]
        model_rebuild_interval = self.model_rebuild_interval_var.get().strip()
        if model_rebuild_interval:
            args += ["--model-rebuild-interval", model_rebuild_interval]
        # Always sent, not just when max_planar_rms is set: SHR% also
        # controls scatter mode's radius jitter/slope-shrink floor
        # (see the "Slope" checkbox), which has nothing to do with
        # max_planar_rms at all -- gating it behind that meant scatter
        # mode never actually received whatever was typed into this
        # field, silently using the backend default instead regardless
        # of what the GUI showed.
        args += ["--planar-shrink-factor", self.planar_shrink_var.get().strip()]
        max_planar_rms = self.max_planar_rms_var.get().strip()
        if max_planar_rms:
            args += ["--max-planar-rms", max_planar_rms]
        args.append("--use-slope-radius" if self.use_slope_radius_var.get() else "--no-use-slope-radius")
        args.append(
            "--use-variation-radius" if self.use_variation_radius_var.get()
            else "--no-use-variation-radius"
        )
        args += ["--variation-contrast-gamma", self.variation_contrast_gamma_var.get().strip()]
        args.append("--density-weighted" if self.density_weighted_var.get() else "--no-density-weighted")
        args += ["--subpixel-jitter-fraction", self.subpixel_jitter_var.get().strip()]
        self._run_step(args, wd)
        self._refresh_refine_stats()

    def _refresh_refine_stats(self) -> None:
        """
        Populate the read-only Refinement values panel from
        project.json -- implied decay (last_refine_rad_m / this run's
        RAD, see PGA2k_gen.py's step_refine_terrain), current vs.
        last-run stamp counts, last-run params, and RMS fit quality.
        Called after every Refine Terrain run, and on working-dir
        change, so the panel reflects whatever's actually on disk
        rather than only updating within this GUI session.
        """
        self.refine_stats_text.config(state="normal")
        self.refine_stats_text.delete("1.0", "end")

        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            self.refine_stats_text.insert("1.0", "(no working directory)")
            self.refine_stats_text.config(state="disabled")
            return
        project = load_project(Path(wd))
        if "last_refine_rad_m" not in project:
            self.refine_stats_text.insert("1.0", "(no refine-terrain run yet)")
            self.refine_stats_text.config(state="disabled")
            return

        last_rad = project.get("last_refine_rad_m")
        implied_decay = project.get("last_refine_implied_decay")
        decay_line = f"{implied_decay:.3f}x" if implied_decay is not None else "n/a (first run)"
        lines = [
            f"Last method:      {project.get('last_refine_method', 'adaptive')}",
            f"Last RAD (m):     {last_rad}",
            f"Implied decay:    {decay_line}",
            f"Last tolerance:   {project.get('last_refine_tolerance_m')}",
            "",
            f"Stamps this run:  {project.get('last_refine_added_count', 0)}",
            f"Total stamps:     {project.get('total_stamp_count', 0)}",
            f"Hotspots/sites:   {project.get('last_refine_hotspot_count', 0)}",
            "",
            f"Fit RMS (mean):   {project.get('last_refine_mean_fit_rms')}",
            f"Fit RMS (max):    {project.get('last_refine_max_fit_rms')}",
        ]
        self.refine_stats_text.insert("1.0", "\n".join(lines))
        self.refine_stats_text.config(state="disabled")

    def _write_terrain_water_args(self, step: str) -> list[str]:
        args = ["--step", step]
        if self.registration_marks_var.get():
            args.append("--registration-marks")
        args.append("--direct-height-shift" if self.direct_height_shift_var.get()
                     else "--no-direct-height-shift")
        args.append("--prune-overlapped-stamps" if self.prune_overlapped_stamps_var.get()
                     else "--no-prune-overlapped-stamps")
        return args

    def _run_write_terrain(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(self._write_terrain_water_args("write-terrain"), wd)

    def _run_write_water(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(self._write_terrain_water_args("write-water"), wd)

    def _run_repack(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        filename = self.repack_filename_var.get().strip()
        if not filename:
            messagebox.showwarning("No filename", "Enter a repack filename first.")
            return
        self._run_step(["--step", "repack", "--repack-filename", filename], wd)

    def _run_visualize(self) -> None:
        wd = self._require_working_dir()
        if wd:
            args = ["--step", "visualize"]
            error_resolution = self.error_resolution_var.get().strip()
            if error_resolution:
                args += ["--error-resolution", error_resolution]
            self._run_step(args, wd)

    # ------------------------------------------------------------------
    # Copy to Game Folder -- a plain file copy, not a pipeline step, so
    # this doesn't go through PGA2k_gen.py at all (there's no terrain
    # logic involved, just moving a finished .course file into place).
    # ------------------------------------------------------------------

    def _run_copy_to_game(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return

        project = load_project(wd)
        filename = project.get("repack_filename")
        if not filename:
            messagebox.showwarning(
                "No repacked course",
                "Run Repack first so there's a .course file to copy.",
            )
            return

        source = wd / f"{filename}.course"
        if not source.exists():
            messagebox.showerror("File not found", f"Expected {source} but it doesn't exist.")
            return

        version = self.game_version.get()
        folder_name = GAME_VERSION_FOLDERS.get(version)
        if folder_name is None:
            messagebox.showerror(
                "Unknown game folder for this version",
                f"No Courses-folder mapping is known yet for game_version={version!r} "
                f"(only {list(GAME_VERSION_FOLDERS)} are wired up). Set Game version (top of "
                "window) to one of those, or add this version's folder name to "
                "GAME_VERSION_FOLDERS once it's confirmed.",
            )
            return
        dest_dir = Path.home() / "AppData" / "LocalLow" / "2K" / folder_name / "Courses"
        dest_path = dest_dir / source.name

        if dest_path.exists():
            if not messagebox.askyesno(
                "File already exists",
                f"{dest_path} already exists.\n\nReplace it?",
            ):
                self._append_log(f"Copy to game folder cancelled ({dest_path} already exists).\n")
                return

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_path)
            self._append_log(f"Copied {source} -> {dest_path}\n")
            self.status_label.config(text="Copied to game folder", foreground="green")
        except OSError as e:
            messagebox.showerror("Copy failed", str(e))
            self.status_label.config(text="Copy failed", foreground="red")

    # ------------------------------------------------------------------
    # Subprocess execution (pipeline steps only -- see Copy to Game above)
    # ------------------------------------------------------------------

    def _run_step(self, extra_args: list[str], working_dir: Path) -> None:
        if self.running:
            messagebox.showinfo("Busy", "A step is already running -- wait for it to finish.")
            return

        self.running = True
        self._stop_requested = False
        self.stop_button.config(state="normal")
        step_name = extra_args[1]
        self.status_label.config(text=f"Running {step_name}...", foreground="orange")
        self._step_start_time = time.time()

        # -u forces the child's stdout/stderr to be unbuffered -- without
        # it, Python fully buffers stdout whenever it isn't a terminal
        # (exactly the case here, piped to this GUI), so print() calls
        # (e.g. adaptive_refine.py's periodic progress updates) would sit
        # in the child's own internal buffer and never actually reach
        # this process's read loop below until that buffer filled or the
        # subprocess exited -- regardless of how often _poll_log_queue
        # itself polls (already every 100ms, so that was never the
        # bottleneck).
        cmd = [sys.executable, "-u", str(CLI_SCRIPT), str(working_dir)] + extra_args
        started_at = time.strftime("%H:%M:%S")
        # Log output accumulates across steps (not cleared each run) so
        # earlier results -- stamp counts, hotspot counts, etc. -- stay
        # visible/scrollable; use the Clear button for a fresh view.
        self._append_log(f"\n{'-' * 70}\n[{started_at}] $ {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        thread.start()

    def _run_subprocess(self, cmd: list[str]) -> None:
        try:
            # start_new_session=True (POSIX only -- ignored on Windows,
            # where _stop_current_step instead uses `taskkill /T` to
            # reach the same goal) puts the child in its own process
            # group rather than the GUI's own, so _stop_current_step can
            # signal that whole group without also hitting this GUI
            # process. See _stop_current_step's docstring for why
            # killing only the immediate child isn't enough here.
            popen_kwargs = {} if platform.system() == "Windows" else {"start_new_session": True}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, **popen_kwargs,
            )
            self._current_proc = proc
            for line in proc.stdout:
                self.log_queue.put(("line", line))
            proc.wait()
            self.log_queue.put(("done", proc.returncode))
        except Exception as e:
            self.log_queue.put(("error", str(e)))
        finally:
            self._current_proc = None

    def _stop_current_step(self) -> None:
        """
        Kill the running subprocess AND every process it spawned, not
        just the immediate child. Plain terminate()/SIGTERM on only
        the immediate child (the old behavior) is enough for most
        steps, but generate-terrain's contour method with n_workers>1
        runs a ProcessPoolExecutor -- separate OS worker processes
        that inherit this Popen's stdout PIPE handle (on both Windows
        and POSIX, a spawned/forked child inherits its parent's open
        handles/fds unless something explicitly closes them, and
        neither multiprocessing nor this code does). Killing only the
        immediate child leaves those workers alive, each still holding
        that handle open -- so the pipe's write end never fully
        closes, _run_subprocess's `for line in proc.stdout` read loop
        never sees EOF, and it hangs forever waiting for output that
        will never come. That's what looked like Stop "doing nothing"
        and the whole run "freezing": the background thread wedged,
        so "done" never reached the log queue and the UI never
        reflected the stop (the Tk main loop itself was never
        actually blocked, but nothing looked like it was happening).

        Fixed by reaching the whole tree/group instead of just the
        one process: `taskkill /T` walks and kills the full process
        tree by recorded parent PID on Windows; os.killpg signals
        every process in the child's process group on POSIX (which is
        why _run_subprocess launches it with start_new_session=True --
        without a group of its own, killpg here would hit this GUI
        process too). Either way, once every handle-holder actually
        exits, the pipe closes for real and the read loop unblocks
        exactly like a normal exit.
        """
        if self._current_proc is None:
            return
        self._stop_requested = True
        pid = self._current_proc.pid
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            self._current_proc.terminate()  # belt-and-suspenders in case the above missed it
        except OSError:
            pass
        self.stop_button.config(state="disabled")

    def _restart_app(self) -> None:
        """
        Relaunch this script as a fresh process with the current
        working directory passed on argv (see main()), then close this
        one -- a full interpreter restart rather than e.g. rebuilding
        the Tk widget tree in place, so it also picks up any edits to
        this file itself without the user having to relaunch by hand.
        """
        if self.running:
            if not messagebox.askyesno(
                "Restart", "A step is currently running -- restarting will stop it. Restart anyway?"
            ):
                return
            self._stop_current_step()

        wd = self.working_dir.get().strip()
        cmd = [sys.executable, str(Path(__file__).resolve())]
        if wd:
            cmd.append(wd)
        subprocess.Popen(cmd, cwd=str(SCRIPT_DIR))
        self.root.destroy()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self._append_log(payload)
                elif kind == "done":
                    self.running = False
                    self.stop_button.config(state="disabled")
                    elapsed = time.time() - self._step_start_time
                    if self._stop_requested:
                        self.status_label.config(text=f"Stopped ({elapsed:.1f}s)", foreground="gray")
                        self._append_log(f"\n[stopped by user after {elapsed:.1f}s]\n")
                    elif payload == 0:
                        self.status_label.config(text=f"Done ({elapsed:.1f}s)", foreground="green")
                        self._append_log(f"\n[finished in {elapsed:.1f}s]\n")
                    else:
                        self.status_label.config(
                            text=f"Failed (exit {payload}, {elapsed:.1f}s)", foreground="red"
                        )
                        self._append_log(f"\n[finished in {elapsed:.1f}s]\n")
                    self._refresh_preview_and_slider()
                    self._ring_bell()
                elif kind == "error":
                    self.running = False
                    self.stop_button.config(state="disabled")
                    elapsed = time.time() - self._step_start_time
                    self.status_label.config(text=f"Error ({elapsed:.1f}s)", foreground="red")
                    self._append_log(f"\n[GUI error] {payload}\n[finished in {elapsed:.1f}s]\n")
                    self._ring_bell()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _play_completion_sound(self) -> str:
        """
        Try platform-specific, actually-audible methods in order,
        falling back to root.bell() only if none work. Returns a short
        description of what was actually tried/used.
        """
        system = platform.system()
        try:
            if system == "Windows":
                import winsound
                winsound.MessageBeep()
                return "winsound.MessageBeep() (Windows)"
            elif system == "Darwin":
                path = "/System/Library/Sounds/Glass.aiff"
                subprocess.run(["afplay", path], timeout=2, check=False)
                return f"afplay {path} (macOS)"
            elif system == "Linux":
                # Common freedesktop sound-theme paths; paplay
                # (PulseAudio/PipeWire) covers most modern desktops,
                # aplay (plain ALSA) as a second try.
                candidates = [
                    ("paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"),
                    ("paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"),
                    ("aplay", "/usr/share/sounds/alsa/Front_Center.wav"),
                ]
                for player, sound_path in candidates:
                    if shutil.which(player) and Path(sound_path).exists():
                        subprocess.run([player, sound_path], timeout=2, check=False)
                        return f"{player} {sound_path} (Linux)"
        except Exception as e:
            self.root.bell()
            return f"root.bell() fallback (exception trying platform method: {e})"

        self.root.bell()
        return "root.bell() fallback (no platform-specific method matched or found)"

    def _ring_bell(self) -> None:
        """
        root.bell() alone is unreliable: on Windows it depends on the
        "Default Beep" system sound not being set to None, on macOS it
        can silently just flash the screen instead of making noise
        depending on Accessibility settings, and on Linux it depends
        on X11 bell / PC-speaker support that's disabled by default on
        many modern distros -- none of that is something code can
        force. See _play_completion_sound for the platform-specific
        methods tried first.
        """
        if self.play_sound_var.get():
            self._play_completion_sound()

    def _append_log(self, text: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Preview panel
    # ------------------------------------------------------------------

    def _versioned_preview_path(self, working_dir: Path, version: int) -> Path:
        """
        ui_version=0 is the latest (highest-numbered) file on disk;
        ui_version=1 is the next-highest, and so on -- see
        visualize.py's find_all_preview_versions/_next_version_path:
        every preview is written as preview_error_0.png,
        preview_error_1.png, ... (never unsuffixed), so "latest" is
        just whichever N is highest, with nothing to rename either way.
        """
        preview_dir = working_dir / PREVIEW_DIR
        name = self.preview_choice.get()
        versions = viz.find_all_preview_versions(preview_dir, name)
        if version < len(versions):
            return versions[version]
        # Nothing at this slot -- return a path that can't exist, so
        # callers correctly show "no preview yet" rather than erroring.
        stem, suffix = Path(name).stem, Path(name).suffix
        return preview_dir / f"{stem}_no_such_version{suffix}"

    def _max_preview_version(self, working_dir: Path) -> int:
        """Number of scrollable versions available for the currently chosen preview, 0 if only one (or none) exist."""
        preview_dir = working_dir / PREVIEW_DIR
        versions = viz.find_all_preview_versions(preview_dir, self.preview_choice.get())
        return max(0, len(versions) - 1)

    _UNDO_SUFFIX_RE = re.compile(r"^(.*)\.(\d{14})\.undo$")
    _TERRAIN_PREVIEW_KINDS = (
        PREVIEW_HEX, PREVIEW_STAMPS, PREVIEW_HEIGHT, PREVIEW_LIDAR_GROUND, PREVIEW_COMPOSITE, PREVIEW_ERROR,
    )

    def _find_latest_stamps_file(self, working_dir: Path) -> Path | None:
        stamps_dir = working_dir / STAMPS_DIR
        n = 1
        latest = None
        while (stamps_dir / f"stamps_{n}.json").exists():
            latest = stamps_dir / f"stamps_{n}.json"
            n += 1
        return latest

    def _find_undo_group(self, working_dir: Path) -> list[Path]:
        """
        Files that make up "the last layering pass": the latest
        stamps_N.json (whichever step -- generate-terrain, refine-
        terrain, generate-cart-paths -- wrote it), plus the latest
        version of each terrain-related preview (hex/stamps/height/
        lidar_ground/error) -- the same set every layering step's
        auto-visualize always regenerates together.

        Every layer is a peer here, including the very first one --
        unlike the old initial_stamps.json/refine_stamps_N.json split,
        there's no more fundamentally-non-undoable "initial" file, since
        generate-terrain is itself now a repeatable, addressable layering
        step. Undoing down to zero layers is a valid (if unusual) result.
        """
        files = []
        latest_stamps = self._find_latest_stamps_file(working_dir)
        if latest_stamps is not None:
            files.append(latest_stamps)
        preview_dir = working_dir / PREVIEW_DIR
        for kind in self._TERRAIN_PREVIEW_KINDS:
            latest_preview = viz.find_latest_preview(preview_dir, kind)
            if latest_preview is not None:
                files.append(latest_preview)
        return files

    def _find_redo_group(self, working_dir: Path) -> tuple[str | None, list[Path]]:
        """Most recent group of .undo files (sharing the same undo timestamp), across stamps/ and preview/."""
        candidates = []
        for directory in (working_dir / STAMPS_DIR, working_dir / PREVIEW_DIR):
            if not directory.is_dir():
                continue
            for f in directory.iterdir():
                m = self._UNDO_SUFFIX_RE.match(f.name)
                if m:
                    candidates.append((m.group(2), f))
        if not candidates:
            return None, []
        latest_ts = max(ts for ts, _ in candidates)
        group = [f for ts, f in candidates if ts == latest_ts]
        return latest_ts, group

    def _run_undo(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        wd = Path(wd)
        files = self._find_undo_group(wd)
        if not files:
            messagebox.showinfo("Nothing to undo", "No stamp layer found to undo.")
            return

        timestamp = time.strftime("%Y%m%d%H%M%S")
        for f in files:
            f.rename(f.with_name(f.name + f".{timestamp}.undo"))
        self._append_log(
            f"\n[undo] moved {len(files)} file(s) aside with suffix .{timestamp}.undo "
            f"(click Redo to bring them back)\n"
        )
        self._cached_composited_base_key = None  # the on-disk "latest" just changed underneath it
        self._cached_geo_overlay_key = None
        self._refresh_preview_and_slider()

    def _run_redo(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        wd = Path(wd)
        timestamp, files = self._find_redo_group(wd)
        if not files:
            messagebox.showinfo("Nothing to redo", "No undone files found to restore.")
            return

        restored = 0
        for f in files:
            m = self._UNDO_SUFFIX_RE.match(f.name)
            target = f.with_name(m.group(1))
            if target.exists():
                self._append_log(f"\n[redo] skipped {f.name}: {target.name} already exists\n")
                continue
            f.rename(target)
            restored += 1
        self._append_log(f"\n[redo] restored {restored} file(s) from {timestamp}\n")
        self._cached_composited_base_key = None
        self._cached_geo_overlay_key = None
        self._refresh_preview_and_slider()

    def _refresh_preview_and_slider(self) -> None:
        """
        Recompute how many archived versions exist for the current
        preview choice, update the slider's range accordingly, and jump
        to version 0 (current) -- called whenever the working directory
        changes, the preview type changes, or a step just finished (so
        the newest result is what's shown by default; older versions
        are still one scroll away).
        """
        wd = self.working_dir.get().strip()
        max_version = self._max_preview_version(Path(wd)) if wd and Path(wd).is_dir() else 0
        self.preview_version_scale.configure(to=max_version)
        self.preview_version.set(0)
        self._update_version_label()
        self._show_preview()

    def _on_preview_choice_changed(self) -> None:
        self._refresh_preview_and_slider()

    def _on_preview_version_changed(self, _value: str) -> None:
        self._update_version_label()
        self._show_preview()

    def _update_version_label(self) -> None:
        v = int(round(self.preview_version.get()))
        self.preview_version_label.config(text="current" if v == 0 else f"-{v}")

    def _on_preview_zoom_scroll(self, event) -> None:
        """Ctrl+scroll zooms the preview (see _on_preview_scroll for the plain-scroll version control)."""
        if event.num == 4:
            step = 0.1
        elif event.num == 5:
            step = -0.1
        else:
            step = 0.1 if event.delta > 0 else -0.1

        new_zoom = max(0.25, min(3.0, self.preview_zoom_var.get() + step))
        self.preview_zoom_var.set(new_zoom)
        self._show_preview()

    def _reset_preview_zoom(self) -> None:
        self.preview_zoom_var.set(1.0)
        self._show_preview()

    def _on_preview_scroll(self, event) -> None:
        """
        Cross-platform scroll handling: Windows/Mac send <MouseWheel>
        with event.delta (positive = scroll up, magnitude varies by
        platform); Linux sends <Button-4> (up) / <Button-5> (down)
        instead, with no delta. Scrolling up moves toward current
        (version 0); scrolling down moves back through history.
        """
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1

        current = int(round(self.preview_version.get()))
        max_version = int(round(float(self.preview_version_scale.cget("to"))))
        new_version = max(0, min(max_version, current + step))
        if new_version != current:
            self.preview_version.set(new_version)
            self._update_version_label()
            self._show_preview()

    def _on_preview_type_scroll(self, event) -> None:
        """Shift+scroll cycles the preview *type* dropdown (see _on_preview_scroll for the plain-scroll version control)."""
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1

        current = self.preview_choice.get()
        idx = PREVIEW_FILES.index(current) if current in PREVIEW_FILES else 0
        new_idx = max(0, min(len(PREVIEW_FILES) - 1, idx + step))
        if new_idx != idx:
            self.preview_choice.set(PREVIEW_FILES[new_idx])
            self._on_preview_choice_changed()

    def _shift_and_crop_to_course(self, working_dir: Path, features: list):
        """
        Shift features (as stored in features.geojson -- the full
        point cloud's frame, uncropped, see ingest/osm.py's
        parse_osm_features) into the course crop's own
        [0, COURSE_SIZE_M] frame, then crop to it -- mirrors
        PGA2k_gen.py's _crop_features_to_course, needed here too since
        the mask preview and Splines-tab highlighting both render
        against course-cropped previews.
        """
        project = load_project(working_dir)
        shift_x = project.get("course_crop_origin_in_full_frame_x")
        shift_z = project.get("course_crop_origin_in_full_frame_z")
        if shift_x is None or shift_z is None:
            return features  # pre-dates this being saved; best effort, treat as already course-frame
        course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
        shifted = shift_features(features, dx=-shift_x, dz=-shift_z)
        return crop_features(shifted, course_bounds)

    def _get_cached_mask_merged_geometry(self, working_dir: Path):
        """
        Lazily load features.geojson and cache the merged (pre-buffer)
        fairway/green geometry -- the relatively expensive part
        (parsing + shapely unary_union) -- so the buffer slider can
        redraw on every tick by just re-buffering this cached shape
        (cheap) and rasterizing it (also cheap, see
        ingest.osm.rasterize_mask_rgba), not re-parsing/re-unioning
        from scratch each time.

        Cache key includes the file's mtime, not just working_dir --
        without that, re-running Ingest OSM (e.g. to pick up a
        corrected map.osm) while pointed at the same directory would
        never invalidate this cache, silently serving stale geometry
        indefinitely (confirmed as a real cause of the mask/highlight
        going out of register after a re-ingest).

        Returns None if there's no features.geojson yet, or it has no
        fairway/green features to mask.

        Dispatches to _get_selected_mask_merged_geometry instead when
        mask_source_var is "selected" -- every caller of this method
        (the live preview overlay, and the snapshot-before-run blocks
        in Generate Trees/Generate Terrain/Refine Terrain) already just
        buffers and rasterizes/saves whatever geometry comes back, so
        branching here is enough to make the mode switch apply
        everywhere without touching those call sites.
        """
        if self.mask_source_var.get() == "selected":
            return self._get_selected_mask_merged_geometry(working_dir)

        features_path = working_dir / FEATURES_FILE
        mtime = features_path.stat().st_mtime if features_path.exists() else None
        cache_key = (working_dir, mtime)
        if getattr(self, "_cached_mask_geom_key", None) == cache_key:
            return self._cached_mask_merged_geom

        merged = None
        if features_path.exists():
            features = load_features(features_path)
            features = self._shift_and_crop_to_course(working_dir, features)
            merged = merge_height_mask_features(features)

        self._cached_mask_merged_geom = merged
        self._cached_mask_geom_key = cache_key
        return merged

    def _get_selected_mask_merged_geometry(self, working_dir: Path):
        """
        Merge the geometries of whatever spline(s) are currently
        selected in the Splines tab into one shape -- same "filter
        Features, then unary_union" shape as merge_height_mask_
        features, but selecting by the live Treeview selection instead
        of the mask flag. Not cached (unlike the "marked" path above):
        selection changes far more often than features.geojson and the
        union itself is cheap over a typically-small selected subset.

        No buffering of bare LineStrings here (unlike the Splines-tab
        highlight overlay's cosmetic .buffer(5.0) -- that's just to
        make a zero-area line visible on screen): the outer
        .buffer(buffer_px) every caller of _get_cached_mask_merged_
        geometry already applies turns a zero-width selected line into
        a real corridor on its own, the same way an unbuffered "hole"
        centerline does in merge_height_mask_features.

        Returns None if nothing is currently selected.
        """
        self._ensure_splines_features_fresh(working_dir)
        selected_ids = {int(s) for s in self.splines_tree.selection()}
        if not selected_ids:
            return None
        features = self._shift_and_crop_to_course(working_dir, self._splines_features)
        relevant = [f.geometry for f in features if f.osm_id in selected_ids]
        if not relevant:
            return None
        return unary_union(relevant)

    def _mask_geometry_full_frame(self, working_dir: Path):
        """
        Same "marked union" (merge_height_mask_features) / "selected
        union" semantics as _get_cached_mask_merged_geometry /
        _get_selected_mask_merged_geometry above, buffered by the same
        mask_buffer_preview_var -- but computed directly against
        self._splines_features WITHOUT _shift_and_crop_to_course, so the
        result stays in features.geojson's own full, uncropped frame
        (see PGA2k_gen.py's _crop_features_to_course docstring). The two
        _get_*_mask_merged_geometry methods above are course-cropped/
        shifted -- correct for preview rasterization, but WRONG to
        persist directly into a new Feature: it would silently double-
        shift on next load and truncate the ring to the current course
        crop rectangle. Used only for building geometry meant to be
        SAVED (border rings, masked-manual clips), never for preview.

        Returns None if there's nothing to union.
        """
        self._ensure_splines_features_fresh(working_dir)
        if self.mask_source_var.get() == "selected":
            selected_ids = {int(s) for s in self.splines_tree.selection()}
            if not selected_ids:
                return None
            relevant = [f.geometry for f in self._splines_features if f.osm_id in selected_ids]
            merged = unary_union(relevant) if relevant else None
        else:
            merged = merge_height_mask_features(self._splines_features)
        if merged is None or merged.is_empty:
            return None
        return merged.buffer(self.mask_buffer_preview_var.get())

    def _next_synthetic_osm_id(self) -> int:
        """
        First unused negative int -- real OSM way ids are always
        non-negative, so a negative synthetic id can never collide with
        one. osm_id is otherwise only used for GUI selection/display
        (see _spline_cluster_detail, _build_cluster_fill_rows), never
        for cross-referencing back to real OSM data.
        """
        existing = {f.osm_id for f in self._splines_features if f.osm_id is not None}
        candidate = -1
        while candidate in existing:
            candidate -= 1
        return candidate

    _OBJECT_DOT_RADIUS_PX = 1  # -> a 2px-diameter dot, per the Objects tab's "Show objects" spec
    _OBJECT_LAYER_FILL_ALPHA = round(255 * 0.4)  # circle interior only -- center dot/outer stroke stay 100%

    def _get_cached_object_preview_layer(self, working_dir: Path):
        """
        Lazily build the "Show objects" preview overlay's point list --
        (x, z, radius_or_None, category_id) triples in the local
        [0, COURSE_SIZE_M] frame -- mtime-keyed the same idiom as
        _get_cached_mask_merged_geometry/_get_cached_heightmap.

        Trees always come from object_list.json, never placedObjects2.
        json's "items": it's the same data the Objects tab list itself
        already shows, available as soon as Generate Trees has run
        (rather than needing a full Write Objects first), and keeps a
        per-tree LIDAR canopy radius (TREE_RADIUS_TAG) around for the
        circle-vs-dot choice below -- placedObjects2.json's tree items
        have already baked that into an opaque scale factor by the time
        they're written (see objects.py's build_tree_objects_v2019),
        losing it. Rocks/grass/ground-cover/display-plants have no pre-
        Write-Objects equivalent at all -- a cluster fill spec
        (features.geojson's PGA_CLUSTER_FILLS_TAG) has no concrete
        positions until course_output/object_clusters.py actually packs
        it -- so those come from placedObjects2.json's "clusters" only.
        Splitting the two sources this way (trees from one file, every
        other nature category from the other, items vs. clusters never
        both read for the same category) means nothing is ever drawn
        twice.

        A tree renders as a circle sized to TREE_RADIUS_TAG when present
        (LIDAR-detected), otherwise a plain dot -- OSM-sourced trees
        carry no per-tree size data. A cluster stamp always renders as a
        circle at its own packed radius (see object_clusters.py's
        _cluster_entry) -- the closest thing this pipeline has to a
        "spacing"-derived on-the-ground size for a scatter-fill area.
        """
        object_list_path = working_dir / OBJECT_LIST_FILE
        placed_objects_path = working_dir / "course" / "CourseDescription_nodes" / PLACED_OBJECTS_FILE
        ol_mtime = object_list_path.stat().st_mtime if object_list_path.exists() else None
        po_mtime = placed_objects_path.stat().st_mtime if placed_objects_path.exists() else None
        cache_key = (str(working_dir), ol_mtime, po_mtime)
        if getattr(self, "_cached_object_layer_key", None) == cache_key:
            return self._cached_object_layer

        points: list[tuple[float, float, Optional[float], int]] = []

        if object_list_path.exists():
            try:
                for x, z, tags in load_object_list(object_list_path):
                    try:
                        radius = float(tags[TREE_RADIUS_TAG]) if TREE_RADIUS_TAG in tags else None
                    except (TypeError, ValueError):
                        radius = None
                    points.append((x, z, radius, 0))
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        if placed_objects_path.exists():
            try:
                for group in load_placed_objects(placed_objects_path):
                    key = group.get("Key", {})
                    category = _ASSET_CATEGORY_BY_PATH.get(key["path"]) if "path" in key else key.get("category")
                    if category not in _OBJECT_LAYER_STYLE:
                        continue
                    for cluster in group.get("Value", {}).get("clusters", []):
                        pos = cluster.get("position", {})
                        x = pos.get("x", 0.0) + GRID_ORIGIN_OFFSET
                        z = pos.get("z", 0.0) + GRID_ORIGIN_OFFSET
                        points.append((x, z, cluster.get("radius"), category))
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        self._cached_object_layer = points
        self._cached_object_layer_key = cache_key
        return points

    def _composite_objects_layer(self, img: "Image.Image", working_dir: Path) -> "Image.Image":
        """
        Composite the "Show objects" overlay onto `img` (already at its
        current zoomed size) -- one filled circle per point from
        _get_cached_object_preview_layer, colored/sized per _OBJECT_
        LAYER_STYLE, plus a legend key in the lower-left corner for
        whichever categories actually appeared. Same _PLOT_RECT-aware
        data-area positioning as the mask buffer/elevation contour/
        highlight overlays above -- course-cropped previews only (see
        their shared caller-side exclusion in _show_preview).

        Dot radius (_OBJECT_DOT_RADIUS_PX) is a fixed SCREEN size,
        deliberately not scaled with zoom -- it's a location marker, not
        a to-scale measurement, so it should stay visible/legible at any
        zoom level. A cluster/tree circle's radius, by contrast, IS a
        real ground measurement (a packed stamp's radius, or a tree's
        measured canopy radius) and is converted through the SAME
        data_width/COURSE_SIZE_M scale used for point positions, so it
        shrinks and grows with zoom exactly like the geometry it
        represents.

        Points are drawn in _OBJECT_LAYER_DRAW_ORDER, not the order
        _get_cached_object_preview_layer happened to return them --
        plain overwrite onto `layer` (PIL's ImageDraw doesn't alpha-
        blend between separate draw calls), so whatever's drawn last
        visually sits on top wherever markers overlap. A circle draws
        as three passes -- semi-transparent fill, then an opaque outer
        stroke, then an opaque center dot on top -- so the exact center
        stays pinpoint-legible even where a large, mutually-overlapping
        cluster of circles would otherwise blur it into the fill.
        """
        points = self._get_cached_object_preview_layer(working_dir)
        if not points:
            return img

        left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
        data_left = round(img.width * left_frac)
        data_top = round(img.height * (1 - bottom_frac - height_frac))
        data_width = max(1, round(img.width * width_frac))
        data_height = max(1, round(img.height * height_frac))

        order = {cat: i for i, cat in enumerate(_OBJECT_LAYER_DRAW_ORDER)}
        points = sorted(points, key=lambda p: order.get(p[3], -1))

        layer = Image.new("RGBA", (data_width, data_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        used_categories: set[int] = set()
        for x, z, radius, category in points:
            style = _OBJECT_LAYER_STYLE.get(category)
            if style is None:
                continue
            used_categories.add(category)
            color = style[1]
            px = (x / COURSE_SIZE_M) * data_width
            py = (1.0 - z / COURSE_SIZE_M) * data_height  # row 0 = max z, same flip as the other overlays
            if radius:
                radius_px = max(self._OBJECT_DOT_RADIUS_PX, (radius / COURSE_SIZE_M) * data_width)
                bbox = (px - radius_px, py - radius_px, px + radius_px, py + radius_px)
                draw.ellipse(bbox, fill=(*color, self._OBJECT_LAYER_FILL_ALPHA))
                draw.ellipse(bbox, outline=(*color, 255))
            draw.ellipse(
                (
                    px - self._OBJECT_DOT_RADIUS_PX, py - self._OBJECT_DOT_RADIUS_PX,
                    px + self._OBJECT_DOT_RADIUS_PX, py + self._OBJECT_DOT_RADIUS_PX,
                ),
                fill=(*color, 255),
            )

        # Objects-tab selection ring -- same cyan as the Splines tab's
        # highlight overlay, drawn as an outlined ring (not a fill) so it
        # doesn't hide the marker/color it's pointing at underneath.
        for hx, hz in self._highlighted_object_points:
            hpx = (hx / COURSE_SIZE_M) * data_width
            hpy = (1.0 - hz / COURSE_SIZE_M) * data_height
            ring_radius = self._OBJECT_DOT_RADIUS_PX + 5
            draw.ellipse(
                (hpx - ring_radius, hpy - ring_radius, hpx + ring_radius, hpy + ring_radius),
                outline=(0, 255, 255, 255), width=2,
            )

        full = Image.new("RGBA", (img.width, img.height), (0, 0, 0, 0))
        full.paste(layer, (data_left, data_top), layer)
        img = Image.alpha_composite(img, full)

        if used_categories:
            img = self._draw_object_layer_key(img, used_categories)
        return img

    @staticmethod
    def _draw_object_layer_key(img: "Image.Image", used_categories: set) -> "Image.Image":
        """Small swatch+label legend, lower-left corner of `img`, one row
        per category actually present in the current overlay -- so a
        course with e.g. no cluster-filled rocks yet just shows Trees,
        not a 5-row key advertising categories with nothing on screen."""
        entries = [style for cat, style in _OBJECT_LAYER_STYLE.items() if cat in used_categories]
        if not entries:
            return img

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        swatch = 10
        pad = 6
        line_h = swatch + 5
        try:
            text_w = max(draw.textlength(label, font=font) for label, _ in entries)
        except AttributeError:  # older Pillow without textlength
            text_w = max(len(label) for label, _ in entries) * 6
        box_w = int(pad * 2 + swatch + 6 + text_w)
        box_h = int(pad * 2 + line_h * len(entries))
        x0, y0 = 8, img.height - box_h - 8

        draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(255, 255, 255, 210), outline=(0, 0, 0, 255))
        for i, (label, color) in enumerate(entries):
            sy = y0 + pad + i * line_h
            draw.ellipse(
                (x0 + pad, sy, x0 + pad + swatch, sy + swatch), fill=(*color, 255), outline=(0, 0, 0, 255),
            )
            draw.text((x0 + pad + swatch + 6, sy - 1), label, fill=(0, 0, 0, 255), font=font)
        return img

    def _get_cached_heightmap(self, working_dir: Path):
        """
        Lazily load heightmap.npz and cache it (mtime-keyed, same
        idiom as _get_cached_mask_merged_geometry) so the elevation-
        contour slider/width box can redraw on every tick by just
        re-thresholding this already-in-memory array (cheap numpy
        comparison + resize), not re-reading the file from disk each
        time. Returns None if there's no heightmap.npz yet.
        """
        path = working_dir / HEIGHTMAP_FILE
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cache_key = (str(path), mtime)
        if getattr(self, "_cached_heightmap_key", None) == cache_key:
            return self._cached_heightmap

        heights, _ = load_heightmap(path)
        self._cached_heightmap = heights
        self._cached_heightmap_key = cache_key
        return heights

    def _get_cached_stamps(self, working_dir: Path):
        """
        Lazily load the full current stamp list (every stamps_N.json
        layer under stamps/, via PGA2k_gen.load_all_stamps) and cache
        it, keyed on the mtime of every *.json file present under
        stamps/ -- so running another layering pass (or undoing one)
        properly invalidates this the next time the overlay redraws.
        Returns None if there's no stamp layer yet (load_all_stamps
        raises in that case; caught here so a missing stamp file just
        means "no overlay" rather than an error swallowing the whole
        preview).
        """
        stamps_dir = working_dir / STAMPS_DIR
        if not stamps_dir.exists():
            return None
        stamp_files = sorted(stamps_dir.glob("*.json"))
        if not stamp_files:
            return None
        cache_key = tuple((str(p), p.stat().st_mtime) for p in stamp_files)
        if getattr(self, "_cached_stamps_key", None) == cache_key:
            return self._cached_stamps
        try:
            stamps = load_all_stamps(working_dir)
        except Exception:
            return None
        self._cached_stamps = stamps
        self._cached_stamps_key = cache_key
        return stamps

    def _get_cached_stamp_influence(
        self, working_dir: Path, shape: tuple[int, int], elevation: float, width: float,
        band_mask: np.ndarray,
    ):
        """
        Real weighted influence, not binary coverage: for every cell,
        the MAX kernel weight (0..1, via terrain.terrain_kernel.
        TerrainKernel -- the same kernel adaptive_refine.py already
        scores candidates with) among every stamp *whose own CENTER
        POSITION falls within band_mask* (the current [elevation,
        elevation+width) band's real spatial footprint, same mask the
        Elevation Contour overlay itself already computed) whose radius
        reaches it -- i.e. only stamps actually placed ON this band's
        terrain, not just whichever stamps happen to be nearby.

        Filtering by CENTER POSITION, not fitted VALUE: an earlier
        version filtered by whether a stamp's own fitted value (the
        local heightmap mean over ITS OWN radius, which can be 20-50m
        for a pass-1 stamp) fell within [elevation, elevation+width).
        Confirmed imprecise on request: that value is a blurred average
        that can easily drift outside a stamp's own nominal band range
        near a real edge or slope, excluding stamps that were genuinely
        placed for this band while including stamps that weren't.
        Checking the stamp's CENTER against the band's own real spatial
        mask instead directly answers "was this stamp actually placed
        on this band's terrain," independent of what its fitted value
        (a separate, downstream computation) ended up being.

        A point can sit well inside a soft-falloff brush's (type
        10/9/54) nominal radius while its real weight there is close to
        0 -- confirmed as the actual issue behind "100% covered, still
        looks unfilled": binary radius-only coverage can't distinguish
        "geometrically reached" from "meaningfully pulled," and this
        can.

        One TerrainKernel per brush type actually in use (not per
        stamp) -- construction cost, whatever it is, shouldn't be paid
        hundreds of thousands of times over. Shape-aware per stamp: a
        SHAPE_SQUARE brush (type 72, e.g. contour rect fill_mode's own
        stamps) gets the same per-axis Chebyshev rectangle test
        terrain_model.py itself uses (rotated into the stamp's own
        local frame via local_square_offsets when rotation != 0, same
        as terrain_model.py's own evaluate*/render -- rect fill_mode's
        stamps are genuinely rotated per boundary edge, not just
        axis-aligned), not a circular approximation -- so a rectangular
        stamp reads as a rectangle here too, not a bounding circle.

        Cached and keyed on (stamps version, shape, elevation, width) --
        filtering by band means the relevant stamp subset changes with
        the slider, unlike an even-earlier whole-course version, so the
        cache can't be elevation-independent. Kept fast in practice
        because the FILTERED subset (one band's worth of stamps, not
        the whole course) is typically a small fraction of the total,
        so each slider tick's recompute is proportionally cheap even
        though it's no longer a pure cache hit.

        Returns None if terrain.terrain_kernel/terrain.brush_profiles
        aren't importable as expected (see the try/except at module
        level) or there are no stamps yet.
        """
        if not _HAVE_TERRAIN_KERNEL:
            return None
        stamps = self._get_cached_stamps(working_dir)
        if stamps is None:
            return None
        cache_key = (self._cached_stamps_key, shape, elevation, width)
        if getattr(self, "_cached_stamp_influence_key", None) == cache_key:
            return self._cached_stamp_influence

        n_rows, n_cols = shape
        cell_x = COURSE_SIZE_M / n_cols
        cell_z = COURSE_SIZE_M / n_rows

        band_stamps = []
        for s in stamps:
            col = int(s.x / cell_x)
            row = int(s.z / cell_z)
            if 0 <= row < n_rows and 0 <= col < n_cols and band_mask[row, col]:
                band_stamps.append(s)

        kernels: dict[int, "TerrainKernel"] = {}

        def _kernel_for(brush: int):
            if brush not in kernels:
                kernels[brush] = TerrainKernel(BRUSH_PROFILES[brush])
            return kernels[brush]

        x_centers = (np.arange(n_cols) + 0.5) * cell_x
        z_centers = (np.arange(n_rows) + 0.5) * cell_z
        influence = np.zeros(shape, dtype=np.float32)

        for s in band_stamps:
            if s.scale_x <= 0 or s.scale_z <= 0:
                continue
            profile = BRUSH_PROFILES.get(s.brush)
            is_square = profile is not None and profile.shape == SHAPE_SQUARE
            # Bounding box: tight per-axis (scale_x, scale_z) for an
            # AXIS-ALIGNED rectangle (rotation==0) -- correct as-is, not
            # just a safe superset, same as terrain_model.py's own
            # render(). A ROTATED square stamp (rect fill_mode's own
            # stamps) needs the same loose-but-safe hypot(scale_x,
            # scale_z) widening terrain_model.py's render() uses, since
            # its true axis-aligned reach is otherwise tighter than a
            # plain per-axis box would assume.
            if is_square and s.rotation != 0.0:
                reach_x = reach_z = (s.scale_x ** 2 + s.scale_z ** 2) ** 0.5
            else:
                reach_x = s.scale_x
                reach_z = s.scale_z if is_square else s.scale_x
            col_min = max(0, int((s.x - reach_x) / cell_x))
            col_max = min(n_cols, int((s.x + reach_x) / cell_x) + 1)
            row_min = max(0, int((s.z - reach_z) / cell_z))
            row_max = min(n_rows, int((s.z + reach_z) / cell_z) + 1)
            if col_min >= col_max or row_min >= row_max:
                continue
            sub_x = x_centers[col_min:col_max]
            sub_z = z_centers[row_min:row_max]
            xx, zz = np.meshgrid(sub_x, sub_z)
            # Same per-axis normalized distance terrain_model.py's own
            # evaluate_many()/render() use: Chebyshev-per-axis (a true
            # rectangle test, not a bounding circle) for SHAPE_SQUARE
            # brushes -- rotated into the stamp's own local frame first
            # (identity when rotation==0) -- plain Euclidean/scale_x for
            # circular ones.
            if is_square:
                ax, az = local_square_offsets(s, xx - s.x, zz - s.z)
                r_full = np.maximum(np.abs(ax) / s.scale_x, np.abs(az) / s.scale_z)
            else:
                r_full = np.hypot(xx - s.x, zz - s.z) / s.scale_x
            within = r_full <= 1.0
            if not within.any():
                continue
            r_norm = r_full[within]
            weight = _kernel_for(s.brush).sample_many(r_norm)

            sub_view = influence[row_min:row_max, col_min:col_max]
            sub_view[within] = np.maximum(sub_view[within], weight)

        self._cached_stamp_influence = influence
        self._cached_stamp_influence_key = cache_key
        return influence

    def _snap_elevation(self, value: float) -> float:
        """
        Round `value` to the nearest multiple of the current Width,
        counted from the slider's own lower bound (the heightmap's real
        min elevation) -- so snapped positions land exactly on the same
        band boundaries a real generate-terrain run at that BAND m would
        actually produce, not an arbitrary sub-meter drag position.
        Clamped to the slider's [from, to] range.
        """
        try:
            width = float(self.elevation_contour_width_var.get())
        except ValueError:
            width = 1.0
        if width <= 0:
            width = 1.0
        lo = float(self.elevation_contour_scale.cget("from"))
        hi = float(self.elevation_contour_scale.cget("to"))
        steps = round((value - lo) / width)
        snapped = lo + steps * width
        return max(lo, min(hi, snapped))

    def _on_elevation_slider_drag(self, raw_value) -> None:
        """
        Scale's own command callback -- fires continuously while
        dragging, with the raw (unsnapped) value as a string. Snapping
        here (rather than leaving the slider continuous and only
        snapping what gets displayed) means the thumb itself visibly
        jumps to each real band boundary as you drag across it, instead
        of scrubbing smoothly through positions that don't correspond
        to any band a real run would produce.
        """
        try:
            raw = float(raw_value)
        except (TypeError, ValueError):
            raw = self.elevation_contour_var.get()
        self.elevation_contour_var.set(self._snap_elevation(raw))
        self._show_preview()

    def _on_elevation_width_changed(self) -> None:
        """
        Re-snaps the current elevation to the new Width's own grid --
        without this, changing Width after already dragging the slider
        would leave the displayed elevation sitting off-grid until the
        next drag, silently showing a band boundary that wouldn't
        actually exist at the new Width.
        """
        self.elevation_contour_var.set(self._snap_elevation(self.elevation_contour_var.get()))
        self._show_preview()

    def _on_elevation_contour_toggle(self) -> None:
        """
        First time the checkbox goes on, size the slider to the real
        heightmap's own [min, max] -- it starts at a 0..1 placeholder
        range since that range isn't known until a heightmap actually
        exists. Re-checking an already-sized slider is a no-op here
        (harmless -- just re-applies the same range).
        """
        wd = self.working_dir.get().strip()
        if self.show_elevation_contour_var.get() and wd:
            heights = self._get_cached_heightmap(Path(wd))
            if heights is not None:
                finite = heights[np.isfinite(heights)]
                if finite.size:
                    lo, hi = float(finite.min()), float(finite.max())
                    if hi > lo:
                        self.elevation_contour_scale.configure(from_=lo, to=hi)
                        current = self.elevation_contour_var.get()
                        if not (lo <= current <= hi):
                            self.elevation_contour_var.set(self._snap_elevation((lo + hi) / 2.0))
                        else:
                            self.elevation_contour_var.set(self._snap_elevation(current))
        self._show_preview()

    def _set_preview_text(self, text: str) -> None:
        self.preview_canvas.delete("all")
        self.preview_canvas.create_text(10, 10, anchor="nw", text=text, fill="black")
        self.preview_canvas.configure(scrollregion=(0, 0, 400, 60))
        self._preview_imgtk = None
        self._preview_canvas_image_id = None

    def _set_preview_image(self, pil_image) -> None:
        self._preview_imgtk = ImageTk.PhotoImage(pil_image)
        self.preview_canvas.delete("all")
        self._preview_canvas_image_id = self.preview_canvas.create_image(
            0, 0, anchor="nw", image=self._preview_imgtk,
        )
        self._center_preview_image()

    def _center_preview_image(self, event=None) -> None:
        """
        Center the current preview image within the canvas's visible
        viewport when the image is smaller than the viewport, instead
        of leaving it glued to the canvas's (0,0) origin -- previously
        the image always rendered in the upper-left corner of the
        preview pane whenever the pane was larger than the image
        itself. Only repositions the image's existing canvas item (via
        coords(), not a delete+recreate), so this is cheap enough to
        call on every pane resize (bound to <Configure> below) without
        flicker, and never touches zoom/pan (Ctrl+scroll, middle-drag)
        -- those already work by scrolling/scaling within whatever
        scrollregion is set here, independent of this offset.

        event is the <Configure> event when bound as a resize handler
        (its width/height are the new canvas size, already current at
        the time the event fires); called with no event (from
        _set_preview_image) it falls back to querying the canvas
        directly.
        """
        if self._preview_canvas_image_id is None or self._preview_imgtk is None:
            return
        canvas_w = event.width if event is not None else self.preview_canvas.winfo_width()
        canvas_h = event.height if event is not None else self.preview_canvas.winfo_height()
        img_w = self._preview_imgtk.width()
        img_h = self._preview_imgtk.height()
        offset_x = max((canvas_w - img_w) // 2, 0)
        offset_y = max((canvas_h - img_h) // 2, 0)
        self.preview_canvas.coords(self._preview_canvas_image_id, offset_x, offset_y)
        self.preview_canvas.configure(scrollregion=(0, 0, max(canvas_w, img_w), max(canvas_h, img_h)))

    def _ensure_splines_features_fresh(self, working_dir: Path) -> None:
        """
        Auto-refresh self._splines_features (used for Splines-tab
        highlighting) if features.geojson has changed on disk since it
        was last loaded -- e.g. after re-running Ingest OSM from a step
        button, which doesn't otherwise touch the Splines tab at all.
        Without this, highlighting would keep using stale geometry
        (confirmed as a real cause of the highlight going out of
        register relative to a freshly re-ingested/cropped feature)
        until the user happened to click Refresh or change directories.
        """
        features_path = working_dir / FEATURES_FILE
        if not features_path.exists():
            return
        mtime = features_path.stat().st_mtime
        if mtime != self._splines_features_mtime:
            self._refresh_splines_list()

    def _show_preview(self) -> None:
        wd = self.working_dir.get().strip()
        if not wd:
            return
        if not _HAVE_PIL:
            self._set_preview_text("(Pillow not installed -- pip install pillow for image previews)")
            return

        version = int(round(self.preview_version.get()))
        path = self._versioned_preview_path(Path(wd), version)
        if not path.exists():
            self._set_preview_text(f"(no {path.name} yet)")
            self._preview_imgtk = None
            return

        try:
            # path.name is always versioned now (e.g. "preview_hex_3.png",
            # never bare "preview_hex.png"), so compare against the
            # stripped kind, not the exact name -- see visualize.py's
            # strip_preview_version. Hoisted above the composite cache so
            # the geo-overlay cache below (mask buffer / spline highlight)
            # can use it regardless of whether that cache hits or misses.
            base_kind = viz.strip_preview_version(path.name)

            # Cache the "static" part -- base image + OSM overlay, at
            # FULL resolution, BEFORE the zoom-dependent resize -- keyed
            # on everything that would change IT specifically. zoom is
            # deliberately NOT part of this key: only the resize below
            # depends on zoom, and that's a cheap, already-in-memory PIL
            # operation, not the disk I/O + full-resolution alpha-
            # compositing this cache exists to avoid repeating. zoom
            # WAS part of this key before -- meaning every single zoom
            # tick (Ctrl+scroll) forced a full cache miss, silently
            # re-opening the base PNG from disk and re-compositing the
            # OSM overlay on EVERY zoom step, not just when the
            # underlying image/overlay/opacity actually changed. That's
            # what made zooming feel slow: it was redoing the expensive
            # part on every tick, not doing genuinely new work.
            overlay_on = self.overlay_osm_var.get()
            zoom = self.preview_zoom_var.get()
            self.preview_zoom_label.config(text=f"{zoom*100:.0f}%")
            composite_key = (
                str(path), path.stat().st_mtime, overlay_on,
                self.overlay_opacity_var.get() if overlay_on else None,
            )
            if getattr(self, "_cached_composited_base_key", None) == composite_key:
                composited = self._cached_composited_base
            else:
                img = Image.open(path).convert("RGBA")

                # The LIDAR previews render the *full* merged point cloud
                # in its own local frame, not the course crop's
                # [0, COURSE_SIZE_M] frame every other preview uses --
                # they need the separately shifted overlay
                # (preview_osm_full.png), not the course-crop one, or
                # features land in the wrong relative position (see
                # ingest/osm.py's shift_features / step_ingest_osm).
                if overlay_on and base_kind not in (PREVIEW_OSM, PREVIEW_OSM_FULL):
                    overlay_name = (
                        PREVIEW_OSM_FULL if base_kind in (PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP)
                        else PREVIEW_OSM
                    )
                    overlay_path = viz.find_latest_preview(Path(wd) / PREVIEW_DIR, overlay_name)
                    if overlay_path is not None:
                        overlay = Image.open(overlay_path).convert("RGBA")
                        if overlay.size == img.size:
                            opacity = self.overlay_opacity_var.get()
                            r, g, b, a = overlay.split()
                            a = a.point(lambda v: int(v * opacity))
                            overlay = Image.merge("RGBA", (r, g, b, a))
                            img = Image.alpha_composite(img, overlay)
                        else:
                            # A previous version of this check failed
                            # silently on a size mismatch -- no error,
                            # overlay just didn't appear, with nothing
                            # to indicate why (confirmed as the actual
                            # cause of exactly this: a stale
                            # preview_osm.png from before the preview
                            # resolution change, mismatching newer base
                            # previews). Surfacing it now instead.
                            print(f"NOTE: skipped OSM overlay -- {overlay_path.name} is "
                                  f"{overlay.size[0]}x{overlay.size[1]}, but the base preview is "
                                  f"{img.size[0]}x{img.size[1]}. Re-run Ingest OSM to regenerate it "
                                  "at the current size.")

                composited = img
                self._cached_composited_base = composited
                self._cached_composited_base_key = composite_key

            # Geometry-based overlays (mask-buffer highlight, selected-
            # spline highlight) that rasterize a shapely geometry via
            # shapely.vectorized.contains -- cached at composited's
            # native resolution, by their own inputs, same reasoning as
            # the composite cache above. These used to rasterize at
            # whatever the CURRENT ZOOMED size was (up to ~3x native
            # pixel count at max zoom), re-running shapely over millions
            # of points on every single zoom tick even though neither
            # shape had changed -- that's what made zooming laggy again,
            # the same bug the composite cache above already fixed once,
            # just in code added afterward that didn't inherit it.
            # Rasterizing once at native resolution and letting the
            # existing cheap zoom-resize below scale the result (like
            # the base composite already does) fixes it the same way.
            mask_buffer_on = self.show_mask_buffer_var.get() and base_kind not in (
                PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
            )
            buffer_px = self.mask_buffer_preview_var.get() if mask_buffer_on else None
            border_on = self.mask_border_var.get() if mask_buffer_on else None
            border_width_raw = self.border_width_var.get() if mask_buffer_on else None
            if mask_buffer_on:
                self.mask_buffer_preview_label.config(text=f"{buffer_px:.0f}")
                self.border_width_preview_label.config(text=f"{border_width_raw:.0f}")
                # _get_cached_mask_merged_geometry can source the mask
                # geometry from features.geojson's marked fairway/green
                # features OR (mask_source_var == "selected") from
                # whatever's currently selected in the Splines tree --
                # neither of those is reflected in composite_key (which
                # only tracks the *preview PNG's* path/mtime), so without
                # this fingerprint a re-ingest or a changed tree
                # selection would leave this cache silently serving a
                # stale mask overlay.
                features_path = Path(wd) / FEATURES_FILE
                mask_geom_fingerprint = (
                    self.mask_source_var.get(),
                    features_path.stat().st_mtime if features_path.exists() else None,
                    tuple(sorted(self.splines_tree.selection())),
                )
            else:
                mask_geom_fingerprint = None

            # Objects tab's selected cluster-fill "group" rows unioned in
            # here too (see _on_object_selected) -- they highlight via
            # their source spline(s)' own geometry, the same overlay a
            # Splines-tab selection drives, rather than a separate
            # highlight pass.
            combined_highlight_ids = self._highlighted_feature_osm_ids | self._highlighted_object_group_spline_ids
            highlight_on = bool(combined_highlight_ids) and base_kind not in (
                PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
            )
            if highlight_on:
                self._ensure_splines_features_fresh(Path(wd))

            geo_overlay_key = (
                composite_key, composited.size,
                mask_buffer_on, buffer_px, border_on, border_width_raw, mask_geom_fingerprint,
                highlight_on,
                tuple(sorted(combined_highlight_ids)) if highlight_on else None,
                self._splines_features_mtime if highlight_on else None,
            )
            if getattr(self, "_cached_geo_overlay_key", None) == geo_overlay_key:
                geo_img = self._cached_geo_overlay
            else:
                geo_img = composited

                if mask_buffer_on:
                    merged_geom = self._get_cached_mask_merged_geometry(Path(wd))
                    if merged_geom is not None:
                        buffered = merged_geom.buffer(buffer_px)
                        if border_on:
                            # Preview only -- draws the same ring shape Fill
                            # would persist, but built from the course-cropped
                            # preview geometry, not _mask_geometry_full_frame's
                            # save-safe one (see that method's docstring for
                            # why the two must stay separate).
                            try:
                                border_width = float(border_width_raw)
                            except (TypeError, ValueError):
                                border_width = 0.0
                            ring = build_border_ring_geometry(buffered, border_width)
                            if not ring.is_empty:
                                buffered = ring
                        course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
                        # The course's 2000x2000 data only occupies the
                        # _PLOT_RECT sub-region of the image (margins around
                        # it hold axis labels/title/colorbar) -- rasterizing
                        # at the full geo_img.width/height, as done before,
                        # stretched the mask across the *entire* image
                        # instead of just that data area. Compute the same
                        # sub-region in pixel terms (matplotlib's _PLOT_RECT
                        # is figure-fraction, origin bottom-left; image
                        # pixels are top-left) and rasterize/paste only
                        # there, leaving the margin fully transparent.
                        left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
                        data_left = round(geo_img.width * left_frac)
                        data_top = round(geo_img.height * (1 - bottom_frac - height_frac))
                        data_width = max(1, round(geo_img.width * width_frac))
                        data_height = max(1, round(geo_img.height * height_frac))

                        mask_rgba = rasterize_mask_rgba(buffered, course_bounds, data_width, data_height)
                        mask_data_img = Image.fromarray(mask_rgba, mode="RGBA")
                        mask_full = Image.new("RGBA", (geo_img.width, geo_img.height), (0, 0, 0, 0))
                        mask_full.paste(mask_data_img, (data_left, data_top), mask_data_img)
                        geo_img = Image.alpha_composite(geo_img, mask_full)

                if highlight_on:
                    course_features = self._shift_and_crop_to_course(Path(wd), self._splines_features)
                    selected_features = [
                        f for f in course_features if f.osm_id in combined_highlight_ids
                    ]
                    if selected_features:
                        geoms = []
                        for f in selected_features:
                            g = f.geometry
                            if g.geom_type == "LineString":
                                # A zero-area line has nothing for
                                # shapely.vectorized.contains to find "inside" --
                                # buffer it into a thin ribbon so it's visible.
                                g = g.buffer(5.0)
                            geoms.append(g)
                        highlight_geom = geoms[0] if len(geoms) == 1 else unary_union(geoms)
                        course_bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=COURSE_SIZE_M, max_z=COURSE_SIZE_M)
                        left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
                        data_left = round(geo_img.width * left_frac)
                        data_top = round(geo_img.height * (1 - bottom_frac - height_frac))
                        data_width = max(1, round(geo_img.width * width_frac))
                        data_height = max(1, round(geo_img.height * height_frac))
                        highlight_rgba = rasterize_mask_rgba(
                            highlight_geom, course_bounds, data_width, data_height,
                            color=(0, 255, 255), opacity=0.6, invert=False,
                        )
                        highlight_img = Image.fromarray(highlight_rgba, mode="RGBA")
                        highlight_full = Image.new("RGBA", (geo_img.width, geo_img.height), (0, 0, 0, 0))
                        highlight_full.paste(highlight_img, (data_left, data_top), highlight_img)
                        geo_img = Image.alpha_composite(geo_img, highlight_full)

                self._cached_geo_overlay = geo_img
                self._cached_geo_overlay_key = geo_overlay_key

            # zoom=1.0 shows the image at its actual native resolution
            # (1959x1780) rather than the old fixed 900x900 cap --
            # nearly 80% of the real pixel area was being thrown away
            # before the user ever saw it. Scrollbars (see
            # _build_preview_panel) handle the case where the zoomed
            # image no longer fits the visible area. This resize runs
            # fresh every call (cheap, in-memory) -- only the composite
            # and geo-overlay stages above are cached.
            target_w = max(1, round(geo_img.width * zoom))
            target_h = max(1, round(geo_img.height * zoom))
            base_thumb = geo_img.resize((target_w, target_h), Image.LANCZOS)

            img = base_thumb

            # Live elevation-band overlay -- see _build_preview_panel's
            # comment and _get_cached_heightmap. Reads heightmap.npz
            # directly (not any preview PNG), so it's exact regardless
            # of which preview is currently selected. Same _PLOT_RECT-
            # aware positioning as the mask buffer above.
            if self.show_elevation_contour_var.get() and viz.strip_preview_version(path.name) not in (
                PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
            ):
                heights = self._get_cached_heightmap(Path(wd))
                if heights is not None:
                    elevation = self.elevation_contour_var.get()
                    self.elevation_contour_label.config(text=f"{elevation:.1f}")
                    try:
                        band_width = float(self.elevation_contour_width_var.get())
                    except ValueError:
                        band_width = 1.0

                    # NaN comparisons are False either way in numpy, so
                    # gaps in the heightmap fall out of the band with no
                    # special-casing needed.
                    band = (heights >= elevation) & (heights < elevation + band_width)

                    band_rgba = np.zeros((*band.shape, 4), dtype=np.uint8)
                    if self.show_stamp_coverage_var.get():
                        influence = self._get_cached_stamp_influence(
                            Path(wd), heights.shape, elevation, band_width, band,
                        )
                        if influence is not None:
                            in_band_weight = influence[band]
                            # Continuous RED (weight=0, untouched) -> GREEN
                            # (weight=1, strongly pulled) -- a solidly green
                            # region under the OLD binary check that still
                            # reads orange/yellow here is exactly a soft-
                            # brush "geometrically covered, weakly pulled"
                            # zone, not a rendering artifact.
                            rows_idx, cols_idx = np.nonzero(band)
                            band_rgba[rows_idx, cols_idx, 0] = np.clip((1.0 - in_band_weight) * 255, 0, 255).astype(np.uint8)
                            band_rgba[rows_idx, cols_idx, 1] = np.clip(in_band_weight * 255, 0, 255).astype(np.uint8)
                            band_rgba[rows_idx, cols_idx, 3] = 255
                        elif not _HAVE_TERRAIN_KERNEL:
                            # terrain.terrain_kernel didn't import as
                            # expected -- degrade to plain band rendering
                            # rather than silently showing nothing, per
                            # this checkbox's tooltip.
                            band_rgba[band, :3] = 255
                            band_rgba[band, 3] = 255
                        else:
                            band_rgba[band, :3] = 255
                            band_rgba[band, 3] = 255
                    else:
                        # 1-bit: solid white where the band matches, fully
                        # transparent elsewhere -- no antialiasing/blending.
                        band_rgba[band, :3] = 255
                        band_rgba[band, 3] = 255
                    # imshow(heights, origin="lower") in visualize.py puts
                    # heights' row 0 at the BOTTOM of the rendered preview,
                    # but PIL images put row 0 at the TOP -- flip vertically
                    # or this overlay renders upside-down relative to the
                    # base image underneath it.
                    band_img = Image.fromarray(band_rgba[::-1, :, :], mode="RGBA")

                    left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
                    data_left = round(img.width * left_frac)
                    data_top = round(img.height * (1 - bottom_frac - height_frac))
                    data_width = max(1, round(img.width * width_frac))
                    data_height = max(1, round(img.height * height_frac))
                    # NEAREST, not LANCZOS -- keeps the band's edges hard
                    # (genuinely 1-bit) rather than antialiased/blurred.
                    band_resized = band_img.resize((data_width, data_height), Image.NEAREST)
                    band_full = Image.new("RGBA", (img.width, img.height), (0, 0, 0, 0))
                    band_full.paste(band_resized, (data_left, data_top), band_resized)
                    img = Image.alpha_composite(img, band_full)

            # "Show objects" (Objects tab) -- drawn last, on top of
            # everything else, so placed-object markers are never hidden
            # under the mask/elevation/highlight overlays above. Same
            # course-cropped-previews-only restriction as those.
            if self.show_objects_var.get() and viz.strip_preview_version(path.name) not in (
                PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
            ):
                img = self._composite_objects_layer(img, Path(wd))

            self._set_preview_image(img)
        except Exception as e:
            self._set_preview_text(f"(couldn't load {path.name}: {e})")


def main() -> int:
    if not CLI_SCRIPT.exists():
        print(
            f"error: {CLI_SCRIPT} not found -- this GUI must sit in the same folder as PGA2k_gen.py",
            file=sys.stderr,
        )
        return 1

    root = tk.Tk()
    gui = PGAGenGUI(root)
    if len(sys.argv) > 1:
        gui.working_dir.set(sys.argv[1])
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
