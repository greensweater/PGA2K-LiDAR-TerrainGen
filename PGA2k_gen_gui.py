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
import math
import os
import platform
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

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
    PREVIEW_STAMPS, PROJECT_FILE, STAMPS_DIR,
)
from PGA2k_gen import (  # noqa: E402
    BLANK_TEMPLATE_COURSE_FILE, PUSH_COLLECTION_COURSE_FILE,
    COLLECTIONS_FILE, DEFAULT_DIG_WATER_BUFFER_M, DEFAULT_DIG_WATER_DEPTH_M,
    DEFAULT_REMOVE_COVERED_MARGIN_M, EXPORT_STATUS_FRESH, EXPORT_STATUS_MISSING, EXPORT_STATUS_STALE,
    FEATURES_FILE, HEIGHT_MASK_FILE, HEIGHTMAP_FILE, INGAME_OBJECTS_FILE, OBJECT_LIST_FILE, OBJECTS_FILE,
    OOB_FILE, PARKING_FILE, PGA_COLLECTION_TAG, RANGE_NETS_FILE, export_status, load_all_stamps, load_project, save_project,
    DEFAULT_LIDAR_TREE_MIN_HEIGHT_M,
)
from course_output.out_of_bounds import (  # noqa: E402
    OOB_BAND_WIDTH_M, OOB_CAP_SCALE_RATIO, OOB_INNER_BUFFER_M, OOB_MERGE_GAP_M, OOB_SIMPLIFY_TOL_M,
)
from course_output.parking import (  # noqa: E402
    PARKING_ACCENT_COUNT, PARKING_COLOR_WEIGHTS, PARKING_MAX_VARIANTS, PARKING_OFFSET_M,
    PARKING_ORIENTATION, PARKING_SIDES, PARKING_SKIP_PROB, PARKING_SPACING_M,
    iter_parking_cars, load_parking_records,
)
from course_output.range_nets import (  # noqa: E402
    iter_range_net_objects, load_range_net_records,
)
from course_output.ingame_objects import (  # noqa: E402
    load_ingame_objects, remove_ingame_object_groups, save_ingame_objects, summarize_ingame_object_groups,
)
from course_output.collection_library import (  # noqa: E402
    _slug as _collection_slug,
    capture_from_course, default_library_dir, load_library, save_collection,
)
from course_output.collections import (  # noqa: E402
    iter_collection_objects, load_collection_records, save_collection_records,
)
from ingest.heightmap import load_heightmap  # noqa: E402
from course_output.objects import (  # noqa: E402
    DEFAULT_GAME_VERSION, GAME_VERSIONS, IMPLEMENTED_GAME_VERSIONS, THEMES_V2019, TREE_HEIGHT_TAG,
    TREE_RADIUS_TAG, TREE_TYPE_TAG, load_object_list, load_objects, save_object_list, save_objects,
)
from course_output.asset_catalog import (  # noqa: E402
    ASSET_CATEGORIES, ASSET_ENTRIES, CLUSTERABLE_ENTRIES, NATURE_CATEGORY_IDS,
)
from course_output.object_clusters import (  # noqa: E402
    CLUSTER_FILL_MODE_AUTO, CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_STAMPS, CLUSTER_FILL_SOURCE_BORDER,
    CLUSTER_FILL_SOURCE_MANUAL, DEFAULT_FILL_DENSITY, DEFAULT_RASTER_RATIO, DEFAULT_SPLINE_FILL_DENSITY,
    DEFAULT_SPLINE_FILL_MAX_PCT, SPLINE_FILL_MAX_PCT_BY_CATEGORY,
    PGA_CLUSTER_CENTERLINE_REF_TAG, PGA_CLUSTER_FILLS_TAG, SYNTHETIC_BORDER_CENTERLINE_KIND,
    SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND, build_border_ring_geometry, next_synthetic_osm_id,
    pack_cluster_records, pack_spline_records,
)
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
from terrain.streams import (  # noqa: E402
    BANK_VEG_WIDTH_M, STREAM_DEPTH_M, STREAM_HALF_WIDTH_M, STREAM_WATER_BASE_WIDTH_M,
    STREAM_WATER_FILL_DEPTH_M, STREAM_WATER_LEVEL_MARGIN_M, STREAM_WATER_WIDEN_PER_DEPTH,
    STREAM_WATER_WIDEN_PER_DESCENT,
)
from course_output.water import (  # noqa: E402
    fit_water_rectangle, fit_water_tiles, fit_water_stripes,
    DEFAULT_WATER_TILE_TOLERANCE_M, DEFAULT_WATER_TILE_MIN_EDGE_M,
    DEFAULT_WATER_TILE_MAX_SEARCH_M, DEFAULT_WATER_TILE_WIDTH_SAMPLES,
    DEFAULT_WATER_TILE_REDUNDANCY_RATIO, DEFAULT_WATER_TILE_OVERLAP_M,
    DEFAULT_WATER_STRIPE_OVERLAP_M, DEFAULT_WATER_STRIPE_TOLERANCE_M,
    DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE, DEFAULT_WATER_STRIPE_BUFFER_M,
)
from shapely.geometry import Point, box as shapely_box  # noqa: E402
from shapely.ops import linemerge, unary_union  # noqa: E402
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
    "preview_oob.png",
]

# Game version -> Courses folder name under .../AppData/LocalLow/2K/.
# Windows-specific path (AppData/LocalLow only exists on Windows, which is
# also the only platform The Golf Club / PGA 2K actually runs on). Keyed
# by the SAME canonical version strings as objects.py's GAME_VERSIONS
# ("2019", "2021", ...), not a display name -- this is looked up directly
# from the single elevated Game version selector (self.game_version) at
# the top of the window, same value write-objects targets, so "write" and
# "move" (Copy to Game Folder) always agree on which version they mean.
# 2023/2025 stay unmapped until those versions are actually implemented
# (see objects.IMPLEMENTED_GAME_VERSIONS).
GAME_VERSION_FOLDERS = {
    "2019": "The Golf Club 2019",
    "2021": "PGA TOUR 2K21",
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
    if f.kind == "roadway":
        railway = f.tags.get("railway")
        if railway:
            return railway
    return f.tags.get("natural", "")


_ASSET_LABEL_BY_KEY = {(e.category, e.type): e.display for e in ASSET_ENTRIES}
_ASSET_LABEL_BY_PATH = {e.path: e.display for e in ASSET_ENTRIES}

# Category id -> (legend label, RGB) for the Objects tab's "Show objects"
# preview overlay. Only the "nature"/scatterable categories (course_output.
# asset_catalog.NATURE_CATEGORY_IDS) get a color -- building stakes,
# cart-path debug markers, walls/signs/bridges/vehicles etc. have no
# useful preview color and are silently skipped (see
# _get_cached_object_preview_layer). Deliberately dark, desaturated
# tones -- bright colors would be hard to pick out against the OSM
# overlay/heightmap this draws on top of.
# The sentinel "category" every placed object-collection member
# (course_output/collections.py) is drawn under in the overlay,
# regardless of its real asset category -- collection members are
# benches/signs/vehicles/etc. with no per-category meaning worth
# splitting out here, unlike the nature categories above/below.
_COLLECTION_LAYER_CATEGORY = -1

_OBJECT_LAYER_STYLE = {
    0: ("Trees", (8, 36, 8)),
    3: ("Ground cover", (30, 74, 30)),
    2: ("Grass", (94, 90, 45)),
    12: ("Display plants", (110, 24, 24)),
    1: ("Rocks", (68, 68, 68)),
    _COLLECTION_LAYER_CATEGORY: ("Collections", (120, 80, 140)),
}

# Fitted water-rectangle preview color -- same steel-blue as the OSM
# overlay's own water color (PGA2k_gen.py's FEATURE_COLORS "water"
# entry, "#4682B4"), so the fitted rectangle reads as "the same feature"
# rather than an unrelated new color. Drawn as an outlined rotated
# rectangle, not a circle, so it's never confused with a tree/cluster
# marker from _OBJECT_LAYER_STYLE below.
_WATER_LAYER_LABEL = "Water (fitted rectangle)"
_WATER_LAYER_COLOR = (70, 130, 180)
_WATER_LAYER_FILL_ALPHA = round(255 * 0.25)

# Draw order for the "Show objects" overlay -- LOW to HIGH, i.e. this
# index is the position in the stack (later entries drawn on top of
# earlier ones), not visual/z-height. Modeled on real-world stacking by
# how tall each category typically stands off the ground: low ground
# cover/rocks/grass first, trees (by far the tallest) drawn dead last so
# a tree marker is never buried under a shorter category's circle.
# Buildings (category 5) aren't in _OBJECT_LAYER_STYLE yet -- no building
# placement data currently feeds this overlay -- but would slot in just
# before trees (second-to-last) if that ever changes.
_OBJECT_LAYER_DRAW_ORDER = (1, 2, 3, 12, _COLLECTION_LAYER_CATEGORY, 0)  # rocks, grass, ground cover, display plants, collections, trees


def _spline_object_detail(f: Feature) -> str:
    """
    What to show in the Splines tab's "Objects" column for this feature.

    For a pga_collection placement line (kind "collection") -- the OSM
    2-node marker way that places a reusable object/spline collection --
    the template name it resolves against, so the row reads as an object
    source too (its resolved objects live in collections.json, cleared
    from this tab via the Spline Objects "Clear" button).

    Otherwise: comma-joined asset labels for whatever's currently in
    f.tags[PGA_CLUSTER_FILLS_TAG] (see course_output/object_clusters.py)
    -- "" if untagged. A spec that no longer resolves (stale tag after
    asset_catalog.json changed) shows as "?" rather than being silently
    dropped, so it's still visible that *something* is tagged there.
    """
    if f.kind == "collection":
        return f.tags.get(PGA_COLLECTION_TAG, "")
    specs = f.tags.get(PGA_CLUSTER_FILLS_TAG)
    if not specs:
        return ""
    return ", ".join(_ASSET_LABEL_BY_KEY.get((s.get("category"), s.get("type")), "?") for s in specs)


def _hand_tuned_member_paths(template_path: Path) -> list[str]:
    """Asset paths of every object member in `template_path` (a collection
    library JSON file) that carries a hand-added variant field -- see
    course_output/collections.py's PARAMETER section: "param_option",
    "variants", or a "{param}" token in "path". [] if the file doesn't
    exist, doesn't parse, or has no such members.

    Re-capturing a collection (_capture_collection_dialog) always rebuilds
    the template from scratch via capture_from_course, which has no notion
    of these fields -- a fresh capture silently overwrites (and loses) any
    hand-tuning a previous capture + hand-edit pass added. This is used to
    warn before that overwrite happens, not to block or merge it."""
    try:
        data = json.loads(template_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tuned = []
    for member in data.get("objects", []):
        if not isinstance(member, dict):
            continue
        path = member.get("path")
        if (member.get("param_option") or member.get("variants")
                or (isinstance(path, str) and "{param}" in path)):
            tuned.append(path or "(no path)")
    return tuned


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
        self._step_name = ""
        self._step_on_done: Optional[Callable[[], None]] = None  # see _run_step's on_done param
        self._last_step_ok = False  # set in _poll_log_queue before on_done fires; on_done runs win/lose
        self._preview_imgtk = None  # keep a reference so tkinter doesn't GC it
        self._preview_photo_cached = None  # see _show_preview's PhotoImage cache (identity-keyed)
        self._preview_photo_src = None  # the PIL image _preview_photo_cached was built from
        self._cached_mask_merged_geom = None  # see _get_cached_mask_merged_geometry
        self._cached_mask_geom_key = None
        self._cached_composited_base = None  # see _show_preview's static-part cache (zoom-independent)
        self._cached_composited_base_key = None
        self._cached_geo_overlay = None  # see _show_preview's geo-overlay cache (mask buffer/spline highlight, also zoom-independent)
        self._cached_geo_overlay_key = None
        self._cached_water_preview_rects = None  # see _get_water_preview_rects
        self._cached_water_preview_rects_key = None
        self._cluster_pack_cache = None  # see _regenerate_packed_objects
        self._cluster_pack_cache_key = None
        self._splines_features = []  # loaded features.geojson content, for the Splines tab
        self._splines_features_mtime = None  # see _ensure_splines_features_fresh
        self._objects_tree_list = []  # loaded object_list.json content, for the Objects tab
        self.objects_tree_asset_paths: list[str] = []  # v2021+ placed-tree asset pool; [] = use bundled rustic catalog default (see _open_tree_assets_dialog)
        self._cluster_fill_rows = []  # (spline_osm_ids_str, asset_label, ratio, density, source, category, type_) -- see _build_cluster_fill_rows
        self._ingame_object_groups = []  # (group_name, count) -- see summarize_ingame_object_groups
        self._ingame_objects_records = []  # raw ingame_objects.json content, parallel to _ingame_object_groups below
        self._ingame_object_group_record_indices = []  # per group in _ingame_object_groups: list of indices into _ingame_objects_records
        self._highlighted_feature_osm_ids = set()  # currently-selected spline(s), if any, to highlight on the preview
        self._highlighted_object_points = []  # (x, z) of currently-selected individual object/tree row(s), for a preview ring
        self._highlighted_object_group_spline_ids = set()  # currently-selected cluster-fill row(s)' source spline osm_ids, unioned into the spline highlight overlay
        self._selection_preview_job = None  # see _on_spline_selected's debounce
        # Viewport picking / marquee-select (see _build_preview_panel's
        # <Button-1> bindings and _on_preview_pick_*). _preview_render_meta
        # records the last _show_preview render's final image size + base
        # kind so a canvas click can be mapped back to course metres.
        self._preview_render_meta = None
        self._marquee_anchor = None  # (canvasx, canvasy) at Button-1 press, else None
        self._marquee_rect_id = None  # canvas rectangle item id while dragging
        self._marquee_subtract_mode = False  # latched between an <Alt-Button-1> press and its release
        self._splines_selection_memory: set[str] = set()  # see _splines_memory_store/_recall
        self._suppress_course_name_save = False
        self._suppress_repack_filename_save = False
        self._suppress_game_version_save = False
        self._suppress_objects_theme_save = False
        self._suppress_tree_assets_save = False
        self._ui_ready = False

        self._build_layout()
        self._ui_ready = True
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

        version_row = ttk.Frame(parent)
        version_row.pack(fill="x", pady=(0, 8))
        version_col = ttk.Frame(version_row)
        version_col.pack(side="left")
        ttk.Label(version_col, text="Game version:").pack(anchor="w")
        game_version_box = ttk.Combobox(
            version_col, textvariable=self.game_version, state="readonly", width=8, values=list(GAME_VERSIONS),
        )
        game_version_box.pack(anchor="w", pady=(2, 0))
        _Tooltip(game_version_box, "PGA 2K's .course schema diverges across versions -- currently "
                 f"only {IMPLEMENTED_GAME_VERSIONS} are actually implemented (see objects.py's "
                 "module docstring); the others can be selected and saved, but write/repack steps "
                 "will raise a clear error until their schema is confirmed. Project-level, saved "
                 "immediately, used by write-objects and (eventually) write-splines/write-terrain/"
                 "repack.")
        reset_baseline_btn = ttk.Button(
            version_row, text="Reset Course Baseline", command=self._run_ingest_course, width=22,
        )
        reset_baseline_btn.pack(side="left", padx=(12, 0), pady=(0, 2), anchor="s")
        _Tooltip(reset_baseline_btn, "Optional. course/ is auto-provisioned from the bundled "
                 "templates/{game_version}_{theme}.course the first time any write/repack step needs "
                 "it -- no manual ingest required. Only use this to explicitly re-extract course/ "
                 "from scratch, e.g. after changing Game version or Theme once course/ already "
                 "exists (which auto-provisioning won't do by itself, since it only fills in a "
                 "missing course/, never overwrites an existing one).")

        ttk.Label(parent, text="Theme:").pack(anchor="w")
        self._theme_name_to_id = {"(not set)": None}
        self._theme_name_to_id.update({name: theme_id for theme_id, name in THEMES_V2019.items()})
        self.objects_theme_var = tk.StringVar(value="(not set)")
        theme_row = ttk.Frame(parent)
        theme_row.pack(fill="x", pady=(2, 8))
        theme_box = ttk.Combobox(
            theme_row, textvariable=self.objects_theme_var, state="readonly", width=14,
            values=list(self._theme_name_to_id.keys()),
        )
        theme_box.pack(side="left")
        push_blank_btn = ttk.Button(
            theme_row, text="Push Blank to Game", command=self._run_push_blank_to_game,
        )
        push_blank_btn.pack(side="left", padx=(6, 0))
        _Tooltip(push_blank_btn, "Build a fresh blank .course from the bundled template for the "
                 "CURRENTLY selected Game version + Theme, give it a unique auto-generated name "
                 "(a 'LIDAR-<version>-<timestamp>' serial) and course id, and copy it straight into "
                 "this version's in-game Courses folder -- ready to open in the game's editor. "
                 "Independent of the pipeline: it does NOT touch course/ or any Write/Repack "
                 "output -- it just needs a working directory (for staging) plus Game version + "
                 "Theme. The pushed course opens in-game under an auto serial name; push again "
                 "for another fresh copy.")
        _Tooltip(theme_box, "Picked up front (not just an Objects-tab v2019 asset-id concern any "
                 "more) -- selects which bundled blank template (templates/{game_version}_{theme}."
                 "course) course/ is auto-provisioned from, so it has to be set before Ingest/Write "
                 "Terrain/etc. run. Also still controls which of v2019's tree types are available at "
                 "write-objects time (v2021+ has no numeric theme id, so there it's template "
                 "selection + display only). Leave as '(not set)' only if you're not ready to pick a "
                 "template yet -- write/repack steps will error clearly asking for one.")

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
        self.preserve_synthetic_var = tk.BooleanVar(value=True)
        preserve_cb = ttk.Checkbutton(
            parent, text="Preserve edits",
            variable=self.preserve_synthetic_var,
        )
        preserve_cb.pack(anchor="w")
        _Tooltip(preserve_cb, "Keep GUI-authored cluster-fill border rings, 'Use mask' fills, "
                 "per-feature mask overrides, cluster-fill specs, and generate-streams bank "
                 "vegetation when re-running Ingest OSM. Uncheck to rebuild features.geojson "
                 "purely from map.osm (old behavior).")
        self._add_step_button(parent, "Ingest OSM", self._run_ingest_osm)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Import in-game edits:").pack(anchor="w")
        import_course_row = ttk.Frame(parent)
        import_course_row.pack(fill="x", pady=(2, 0))
        self.import_ingame_course_var = tk.StringVar()
        import_course_entry = ttk.Entry(import_course_row, textvariable=self.import_ingame_course_var)
        import_course_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(
            import_course_row, text="Browse...", command=self._browse_import_ingame_course,
        ).pack(side="left", padx=(4, 0))
        _Tooltip(import_course_entry, "A .course file exported by this tool, then hand-edited in "
                 "PGA Tour 2K's own in-game object/terrain editor and saved -- reconciled against "
                 "what this tool currently tracks (see course_output/ingame_objects.py). Never "
                 "touches this project's own course/ folder.")
        ttk.Label(parent, text="Group label:").pack(anchor="w", pady=(4, 0))
        self.import_ingame_group_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.import_ingame_group_var, width=24).pack(anchor="w")
        import_buttons_row = ttk.Frame(parent)
        import_buttons_row.pack(fill="x", pady=(4, 0))
        preview_import_btn = ttk.Button(
            import_buttons_row, text="Preview Import", command=self._run_import_ingame_preview,
        )
        preview_import_btn.pack(side="left")
        _Tooltip(preview_import_btn, "Dry run -- prints a summary (new objects/stamps found, plus "
                 "anything this tool tracks that's missing from the imported course, reported only, "
                 "never removed) without changing anything on disk.")
        commit_import_btn = ttk.Button(
            import_buttons_row, text="Commit Import", command=self._run_import_ingame_commit,
        )
        commit_import_btn.pack(side="left", padx=(4, 0))
        _Tooltip(commit_import_btn, "Applies the same diff Preview Import shows: appends every new "
                 "in-game object to ingame_objects.json (under the Group label above) and writes "
                 "every new terrain stamp as the next stamps_N.json layer. Run Pack Objects / Write "
                 "Objects / Write Terrain afterward to fold the result into placedObjects/userLayers.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.repack_filename_var = tk.StringVar()
        self.repack_filename_var.trace_add("write", lambda *a: self._on_repack_filename_changed())
        ttk.Label(parent, text="Repack filename:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.repack_filename_var, width=20).pack(anchor="w")

        status_row = ttk.Frame(parent)
        status_row.pack(fill="x", pady=(0, 4))
        _EXPORT_STATUS_EMOJI = {
            "height": "⛰️",  # mountain
            "water": "💦",  # sweat droplets
            "splines": "🧩",  # puzzle piece
            "holes": "⛳",  # flag in hole
            "objects": "🌳",  # wood
        }
        self._export_status_lights: dict[str, tk.Label] = {}
        for category in ("height", "water", "splines", "holes", "objects"):
            cell = ttk.Frame(status_row)
            cell.pack(side="left", padx=(6, 0))
            light = tk.Label(cell, text="●", fg="gray", font=("TkDefaultFont", 10))
            light.pack(side="top")
            emoji_label = ttk.Label(cell, text=_EXPORT_STATUS_EMOJI[category], font=("TkDefaultFont", 11))
            emoji_label.pack(side="top")
            _Tooltip(emoji_label, category.capitalize())
            self._export_status_lights[category] = light
        _Tooltip(status_row, "Whether each course/CourseDescription_nodes/ output is caught up "
                    "with what it's built from. Green = written and up to date. Red = an input "
                    "(features.geojson, a stamp layer, objects.json) changed since the last write, "
                    "or a leftover different-game_version file is sitting in course/ (Objects only "
                    "-- see Reset Course Baseline) -- re-run that write step (or Reset Course "
                    "Baseline) before repacking. Gray = never written yet. Refreshes after every "
                    "step. Height/Water share one file (userLayers.json) so this can occasionally "
                    "show a false green for one of the pair right after writing the other -- not a "
                    "hard gate, just a status hint (Repack itself still refuses on the "
                    "different-version-file case regardless of what these show).")

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

        self.generate_terrain_remove_covered_stamps_var = tk.BooleanVar(value=False)
        remove_covered_checkbox = ttk.Checkbutton(
            parent, text="Remove covered stamps from previous layer",
            variable=self.generate_terrain_remove_covered_stamps_var,
        )
        remove_covered_checkbox.pack(anchor="w")
        _Tooltip(remove_covered_checkbox, "Flags stamps in the immediately-preceding stamps_N.json "
                 "layer as blocked_by this new one wherever their whole footprint sits entirely inside "
                 "this pass's own MASK, shrunk inward by the margin below (exact shapely polygon "
                 "containment -- terrain/stamp_containment.py). Tests against the mask -- the area this "
                 "pass INTENDS to cover -- not against verified brush coverage, so it's only as safe as "
                 "this pass's own fill actually is solid there: a generous margin and a "
                 "fully-tiling/high-density fill (e.g. raster + brush 72 + spread%>=1) leave no real "
                 "gap; a sparse or gappy fill (hex mode, or raster with a soft/low-plateau brush) could "
                 "leave a small hole where a blocked stamp used to show through, since nothing else "
                 "painted over it. Requires 'Restrict to mask' above. Blocked stamps aren't deleted -- "
                 "the older layer file gains one annotation field, undo/redo of this new layer toggles "
                 "it automatically.")
        margin_frame = ttk.Frame(parent)
        margin_frame.pack(anchor="w", pady=(0, 4))
        ttk.Label(margin_frame, text="    Margin (m):").pack(side="left")
        self.generate_terrain_remove_covered_margin_var = tk.StringVar(
            value=str(DEFAULT_REMOVE_COVERED_MARGIN_M)
        )
        margin_entry = ttk.Entry(
            margin_frame, textvariable=self.generate_terrain_remove_covered_margin_var, width=6,
        )
        margin_entry.pack(side="left")
        _Tooltip(margin_entry, "Inward safety margin (m): a previous-layer stamp must be fully inside "
                 "the mask minus this distance from its boundary to get flagged. Larger = more "
                 "conservative (fewer stamps flagged).")

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

        self.refine_remove_covered_stamps_var = tk.BooleanVar(value=False)
        refine_remove_covered_checkbox = ttk.Checkbutton(
            parent, text="Remove covered stamps from previous layer",
            variable=self.refine_remove_covered_stamps_var,
        )
        refine_remove_covered_checkbox.pack(anchor="w", pady=(4, 0))
        _Tooltip(refine_remove_covered_checkbox, "Flags stamps in the immediately-preceding "
                 "stamps_N.json layer as blocked_by this new one wherever their whole footprint sits "
                 "entirely inside this pass's own MASK, shrunk inward by the margin below (exact "
                 "shapely polygon containment -- terrain/stamp_containment.py). Tests against the mask "
                 "-- the area this pass is allowed to place hotspots in -- not against verified brush "
                 "coverage there, so it's only as safe as this pass's own placements actually turn out "
                 "to be dense/solid: refine's hotspot placement can legitimately leave gaps inside the "
                 "mask (it only refines where error already exceeds tolerance), so a blocked stamp "
                 "could show through a spot nothing else painted over. Use a generous margin, and "
                 "expect to re-run refine a few times over the same mask before trusting this. "
                 "Requires 'Mask' above. Blocked stamps aren't deleted -- the older layer file gains "
                 "one annotation field, undo/redo of this new layer toggles it automatically.")
        refine_margin_frame = ttk.Frame(parent)
        refine_margin_frame.pack(anchor="w", pady=(0, 4))
        ttk.Label(refine_margin_frame, text="    Margin (m):").pack(side="left")
        self.refine_remove_covered_margin_var = tk.StringVar(value=str(DEFAULT_REMOVE_COVERED_MARGIN_M))
        refine_margin_entry = ttk.Entry(
            refine_margin_frame, textvariable=self.refine_remove_covered_margin_var, width=6,
        )
        refine_margin_entry.pack(side="left")
        _Tooltip(refine_margin_entry, "Inward safety margin (m): a previous-layer stamp must be fully "
                 "inside the mask minus this distance from its boundary to get flagged. Larger = more "
                 "conservative (fewer stamps flagged).")

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
        write_terrain_btn = self._add_step_button(parent, "Write Terrain", self._run_write_terrain)
        _Tooltip(write_terrain_btn, "Writes userLayers.json's \"height\" key (the terrain stamps) -- "
                 "leaves \"water\" exactly as it was found (see the separate Write Water button below, "
                 "which is slower and only needs re-running when water levels should track a terrain "
                 "change).")

        self.multi_tile_water_var = tk.BooleanVar(value=True)
        multi_tile_checkbox = ttk.Checkbutton(
            parent, text="Multi-tile water fill", variable=self.multi_tile_water_var,
            command=self._show_preview,
        )
        multi_tile_checkbox.pack(anchor="w")
        _Tooltip(multi_tile_checkbox, "Write Water only: fill each pond with several smaller, "
                 "possibly-overlapping rectangular tiles hugging its real boundary, instead of one "
                 "single minimum-rotated-rectangle -- reduces visible water-over-land overshoot on "
                 "non-rectangular ponds (overlap between tiles is fine, renders with a clean seam "
                 "in-game). Also affects the \"Show objects\" water preview immediately. On by "
                 "default; uncheck for single-rectangle behavior.")

        # Fill mode is a tabbed interface -- the visible tab IS the selected
        # mode (water_fill_mode_var); Stripe first / default.
        self.water_fill_mode_var = tk.StringVar(value="stripe")
        water_mode_nb = ttk.Notebook(parent)
        water_mode_nb.pack(anchor="w", fill="x", pady=(4, 0))
        stripe_tab = ttk.Frame(water_mode_nb, padding=4)
        edge_tab = ttk.Frame(water_mode_nb, padding=4)
        water_mode_nb.add(stripe_tab, text="Stripe")
        water_mode_nb.add(edge_tab, text="Edge")

        def _on_water_mode_tab(_e=None):
            if not getattr(self, "_ui_ready", False):
                return
            self.water_fill_mode_var.set(
                "edge" if water_mode_nb.index("current") == 1 else "stripe"
            )
            self._show_preview()
        water_mode_nb.bind("<<NotebookTabChanged>>", _on_water_mode_tab)

        # --- Stripe fill (first tab / default) ---
        # Layout: row 1 = TOL, OVERLAP, BUF ; row 2 = MIN w, MAX L
        water_stripe_grid = ttk.Frame(stripe_tab)
        water_stripe_grid.pack(anchor="w", fill="x")
        self.water_stripe_tolerance_var = tk.StringVar(value=str(DEFAULT_WATER_STRIPE_TOLERANCE_M))
        self._add_grid_field(water_stripe_grid, 0, 0, self.water_stripe_tolerance_var, "TOL",
                 "Stripe water fill only: optional boundary-simplify tolerance (m) before probing -- "
                 "perf/noise-reduction only, not structural (unlike the edge-fill TOL field). "
                 "Defaults off (0).")
        self.water_stripe_overlap_var = tk.StringVar(value=str(DEFAULT_WATER_STRIPE_OVERLAP_M))
        self._add_grid_field(water_stripe_grid, 0, 1, self.water_stripe_overlap_var, "OVERLAP",
                 "Stripe water fill only: the single knob controlling both how far a stripe may "
                 "overshoot the true polygon boundary and how much consecutive stripes overlap along "
                 "the stacking axis.")
        self.water_stripe_buffer_var = tk.StringVar(value=str(DEFAULT_WATER_STRIPE_BUFFER_M))
        self._add_grid_field(water_stripe_grid, 0, 2, self.water_stripe_buffer_var, "BUF m",
                 "Stripe water fill only: grow the pond's own polygon by this many meters (all "
                 "directions) BEFORE stripe fitting -- a fudge factor for when the OSM way and the "
                 "real LIDAR-derived terrain don't quite line up, leaving stripes ending short of "
                 "where the ground actually starts sloping into the water. Defaults off (0).")
        self.water_stripe_min_edge_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_MIN_EDGE_M))
        self._add_grid_field(water_stripe_grid, 1, 0, self.water_stripe_min_edge_var, "MIN w",
                 "Stripe water fill only: floor (m) for a stripe's found width/depth.")
        self.water_stripe_max_stripes_var = tk.StringVar(value=str(DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE))
        self._add_grid_field(water_stripe_grid, 1, 1, self.water_stripe_max_stripes_var, "MAX L",
                 "Stripe water fill only: safety cap on stripes walked outward in each of the +/- "
                 "stacking directions -- defensive only, not expected to bind on a real pond.")

        # --- Edge fill ---
        water_tile_grid = ttk.Frame(edge_tab)
        water_tile_grid.pack(anchor="w", fill="x")
        self.water_tile_tolerance_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_TOLERANCE_M))
        self._add_grid_field(water_tile_grid, 0, 0, self.water_tile_tolerance_var, "TOL m",
                 "Multi-tile water fill only: Douglas-Peucker boundary-simplify tolerance (m) applied "
                 "to a pond polygon before per-edge tile placement -- collapses small boundary "
                 "wiggles into fewer, longer edges.")
        self.water_tile_min_edge_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_MIN_EDGE_M))
        self._add_grid_field(water_tile_grid, 0, 1, self.water_tile_min_edge_var, "MIN m",
                 "Multi-tile water fill only: floor depth (m) for a per-edge tile when no opposite "
                 "wall is found within SEARCH m.")
        self.water_tile_max_search_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_MAX_SEARCH_M))
        self._add_grid_field(water_tile_grid, 0, 2, self.water_tile_max_search_var, "SEARCH m",
                 "Multi-tile water fill only: cap (m) on the ray-cast search for the wall opposite "
                 "each boundary edge.")
        self.water_tile_width_samples_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_WIDTH_SAMPLES))
        self._add_grid_field(water_tile_grid, 1, 0, self.water_tile_width_samples_var, "SAMPLES",
                 "Multi-tile water fill only: points sampled across each boundary edge's own width "
                 "when ray-casting for the opposite wall.")
        self.water_tile_redundancy_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_REDUNDANCY_RATIO))
        self._add_grid_field(water_tile_grid, 1, 1, self.water_tile_redundancy_var, "REDUN",
                 "Multi-tile water fill only: a candidate tile contributing less than this fraction "
                 "of its own area as new coverage is dropped as redundant. Lower = more, "
                 "more-overlapping tiles kept.")
        self.water_tile_overlap_var = tk.StringVar(value=str(DEFAULT_WATER_TILE_OVERLAP_M))
        self._add_grid_field(water_tile_grid, 1, 2, self.water_tile_overlap_var, "OVERLAP m",
                 "Multi-tile water fill only: extend each per-edge tile's width by this much (m) at "
                 "BOTH ends, centered the same as before -- without it, adjacent tiles meet exactly "
                 "corner-to-corner and can look visibly inset once real rounding is involved.")

        write_water_btn = self._add_step_button(parent, "Write Water", self._run_write_water)
        _Tooltip(write_water_btn, "Writes userLayers.json's \"water\" key -- leaves \"height\" exactly "
                 "as it was found (see course_output/water.py). Water objects are built from "
                 "features.geojson's water polygons (natural=water, golf=water_hazard, waterway=* "
                 "areas -- run Ingest OSM first if none show up) fitted to the CURRENT stamp list's "
                 "low points (re-loaded/normalized the same way as Write Terrain, using the "
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
        # The console/log lives in its own pane of the horizontal
        # preview|log PanedWindow. "Window shade" collapse: we drop the
        # whole frame out of the PanedWindow (paned.forget) rather than
        # just hiding the text, so the preview gets the full width back;
        # re-opening re-adds it and restores the remembered sash width.
        # Collapsed by default -- most steps don't need the console, and
        # the "◂ Console" button in the preview header brings it back.
        self._log_paned = paned
        frame = ttk.Frame(paned)
        self._log_frame = frame
        self._log_collapsed = False
        self._log_pane_width = 340
        self._log_unread = False
        paned.add(frame, weight=1)
        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text="Output log").pack(side="left")
        ttk.Button(header, text="Hide ▸", width=7,
                   command=self._toggle_log_panel).pack(side="right")
        ttk.Button(header, text="Clear", command=self._clear_log).pack(side="right", padx=(0, 4))

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

        # Start collapsed. Defer one tick so the PanedWindow is realized
        # first (a forget() before the initial layout can leave the sash
        # in a weird spot on some Tk builds).
        self.root.after(0, lambda: self._toggle_log_panel(force_collapse=True))

    def _toggle_log_panel(self, force_collapse: bool = False) -> None:
        if not self._log_collapsed or force_collapse:
            # Collapse: remember the current width, then drop the pane.
            if not self._log_collapsed:
                try:
                    total = self._log_paned.winfo_width()
                    pos = self._log_paned.sashpos(0)
                    if total > 1 and pos > 0:
                        self._log_pane_width = max(160, total - pos)
                except Exception:
                    pass
            try:
                self._log_paned.forget(self._log_frame)
            except Exception:
                pass
            self._log_collapsed = True
            self._show_log_button.pack(side="right", padx=(6, 0))
            self._refresh_log_button_label()
        else:
            # Expand: re-add the pane and push the sash left so it gets
            # back roughly the width it had.
            try:
                self._log_paned.add(self._log_frame, weight=1)
                self._log_paned.update_idletasks()
                total = self._log_paned.winfo_width()
                if total > 1:
                    self._log_paned.sashpos(0, max(0, total - self._log_pane_width))
            except Exception:
                pass
            self._log_collapsed = False
            self._log_unread = False
            self._show_log_button.pack_forget()

    def _refresh_log_button_label(self) -> None:
        self._show_log_button.config(
            text="◂ Console ●" if self._log_unread else "◂ Console"
        )

    def _build_preview_panel(self, paned: ttk.PanedWindow) -> None:
        frame = ttk.Frame(paned)
        paned.add(frame, weight=2)

        header = ttk.Frame(frame)
        header.pack(fill="x")
        preview_label = ttk.Label(header, text="Preview:")
        preview_label.pack(side="left")
        _Tooltip(preview_label, "Left-click a spline/object on the image to select its row in the "
                 "Splines/Objects list (Shift/Ctrl-click adds to the selection); left-drag a box to "
                 "marquee-select everything inside it; click empty space to clear. Alt-drag a box "
                 "to cut that region out of a selected GUI-authored border ring or 'Use mask' fill "
                 "(e.g. leave one edge of a border unfilled) -- real OSM features are left alone. "
                 "Splines are "
                 "pickable only while 'Overlay OSM' is on, objects only while 'Show objects' is on. "
                 "Right-drag pans; scroll zooms, Ctrl+scroll steps versions, Shift+scroll changes "
                 "preview type. Picking is off on the LIDAR / full-frame OSM previews.")
        self.preview_choice = tk.StringVar(value="preview_height.png")
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

        # Scroll wheel over this slider steps through versions -- Windows/Mac
        # send <MouseWheel> with event.delta; Linux sends <Button-4>/<Button-5>
        # instead. Shift+scroll instead cycles the preview *type* dropdown
        # (same cross-platform split). Over the image itself, plain scroll
        # zooms and Ctrl+scroll does this version stepping -- see the canvas
        # bindings below.
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

        # Re-opens the collapsed console/log pane (see _build_log_panel).
        # Packed on the right so it stays clear of the left-packed row of
        # preview controls; hidden whenever the log pane is expanded.
        self._show_log_button = ttk.Button(
            header, text="◂ Console", width=10, command=self._toggle_log_panel,
        )
        # Packed/unpacked by _toggle_log_panel -- visible only while the
        # console pane is collapsed (which is the default).

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

        # Over the image: plain scroll zooms (the common gesture),
        # Ctrl+scroll steps through versions, Shift+scroll cycles the
        # preview type. The Version slider itself still takes plain
        # scroll for versions -- see its own bindings above.
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_zoom_scroll)
        self.preview_canvas.bind("<Button-4>", self._on_preview_zoom_scroll)
        self.preview_canvas.bind("<Button-5>", self._on_preview_zoom_scroll)
        self.preview_canvas.bind("<Shift-MouseWheel>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Shift-Button-4>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Shift-Button-5>", self._on_preview_type_scroll)
        self.preview_canvas.bind("<Control-MouseWheel>", self._on_preview_scroll)
        self.preview_canvas.bind("<Control-Button-4>", self._on_preview_scroll)
        self.preview_canvas.bind("<Control-Button-5>", self._on_preview_scroll)
        # Right-click-hold-and-drag pans the view -- scan_mark/scan_dragto are
        # tkinter Canvas's own built-in support for exactly this, so no
        # manual scroll-position math is needed here.
        self.preview_canvas.bind("<Button-3>", lambda e: self.preview_canvas.scan_mark(e.x, e.y))
        self.preview_canvas.bind("<B3-Motion>", lambda e: self.preview_canvas.scan_dragto(e.x, e.y, gain=1))

        # Left button: click a spline/object to select its row (in place,
        # no tab switch); click+drag a box to marquee-select every
        # spline/object inside it. Right button stays pan, so these don't
        # collide. See _on_preview_pick_press/_motion/_release.
        #
        # Alt+left-drag a box instead SUBTRACTS that region from the
        # geometry of the selected Splines row(s) (see _marquee_subtract).
        # Only <Alt-Button-1> is bound -- the generic <B1-Motion>/
        # <ButtonRelease-1> below still fire for the Alt-drag (Tk falls
        # back to them when no <Alt-...> variant is bound), so a latched
        # mode flag set at press is all that's needed and it survives the
        # user letting go of Alt mid-drag.
        self.preview_canvas.bind("<Button-1>", self._on_preview_pick_press)
        self.preview_canvas.bind("<Alt-Button-1>", self._on_preview_subtract_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_pick_motion)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_pick_release)
        self.preview_canvas.bind("<Configure>", self._center_preview_image)

    _SPLINE_KIND_FILTERS = (
        "All", "green", "tee", "fairway", "rough", "heavyrough", "bunker",
        "water", "cartpath", "service_road", "roadway", "driveway", "path",
        "building", "wood", "pavement", "mulch", "hole", "collection",
        SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND,
    )

    BRUSH_TYPE_ORDER = (8, 9, 10, 54)
    BRUSH_TYPE_LABELS = {8: "hard", 9: "med", 10: "soft", 54: "smooth"}

    _OBJECT_SOURCE_FILTERS = ("All", "OSM", "LIDAR", "Manual", "Border", "Imported")

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
        self.overlay_osm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            overlay_row, text="Overlay OSM", variable=self.overlay_osm_var,
            command=self._on_overlay_osm_toggled,
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
        self.mask_source_var = tk.StringVar(value="selected")
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
        self.mask_buffer_preview_text = tk.StringVar(value="50")
        self._add_slider_entry(mask_row2, self.mask_buffer_preview_var, self.mask_buffer_preview_text)

        mask_row3 = ttk.Frame(mask_section)
        mask_row3.pack(fill="x", pady=(2, 0))
        ttk.Label(mask_row3, text="Border:").pack(side="left")
        self.border_width_var = tk.DoubleVar(value=10.0)
        ttk.Scale(
            mask_row3, from_=0.0, to=400.0, orient="horizontal",
            variable=self.border_width_var, command=lambda _v: self._show_preview(),
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.border_width_preview_text = tk.StringVar(value="10")
        self._add_slider_entry(mask_row3, self.border_width_var, self.border_width_preview_text)

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
            tree_frame, columns=("kind", "tag", "mask", "objects"), show="headings", height=10,
            selectmode="extended",
        )
        self.splines_tree.heading("kind", text="Kind")
        self.splines_tree.heading("tag", text="Tag")
        self.splines_tree.heading("mask", text="Mask")
        self.splines_tree.heading("objects", text="Objects")
        self.splines_tree.column("kind", width=90)
        self.splines_tree.column("tag", width=90)
        self.splines_tree.column("mask", width=24, anchor="center")
        self.splines_tree.column("objects", width=140)
        self.splines_tree.pack(side="left", fill="both", expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.splines_tree.yview)
        tree_scroll.pack(side="left", fill="y")
        self.splines_tree["yscrollcommand"] = tree_scroll.set
        self.splines_tree.bind("<<TreeviewSelect>>", lambda e: self._on_spline_selected())
        self.splines_tree.bind("<Double-1>", self._on_spline_row_double_click)

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
        ttk.Label(parent, text="Spline Objects", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
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
                 "so Write Objects stops generating clusters for them. For a selected pga_collection "
                 "marker, drops that placement's resolved objects from collections.json instead -- "
                 "iteration-scoped: re-running Generate rebuilds it from the OSM markers. For "
                 "permanent removal, delete the pga_collection way in your OSM editor and re-ingest.")

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

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._build_oob_section(parent)

    def _build_oob_section(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Out of Bounds", font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 2))
        ttk.Label(
            parent,
            text="Auto-paint an OOB band (brush stamps -- round caps + smooth squares) just "
                 "outside the playable area.",
            wraplength=360, foreground="#555",
        ).pack(anchor="w", pady=(0, 2))

        oob_grid = ttk.Frame(parent)
        oob_grid.pack(fill="x", pady=2)

        def add_oob_field(row, col, var, label, tooltip):
            cell = ttk.Frame(oob_grid)
            cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
            lbl = ttk.Label(cell, text=label)
            lbl.pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=var, width=10)
            entry.pack(anchor="w")
            _Tooltip(lbl, tooltip)
            _Tooltip(entry, tooltip)

        self.oob_inner_buffer_var = tk.StringVar(value=f"{OOB_INNER_BUFFER_M}")
        self.oob_band_width_var = tk.StringVar(value=f"{OOB_BAND_WIDTH_M}")
        self.oob_merge_gap_var = tk.StringVar(value=f"{OOB_MERGE_GAP_M}")
        self.oob_simplify_tol_var = tk.StringVar(value=f"{OOB_SIMPLIFY_TOL_M}")
        self.oob_cap_ratio_var = tk.StringVar(value=f"{OOB_CAP_SCALE_RATIO}")
        self.oob_include_caps_var = tk.BooleanVar(value=True)

        add_oob_field(0, 0, self.oob_inner_buffer_var, "INNER BUFFER m",
                       "Gap (m) between the playable area and where the OOB band starts.")
        add_oob_field(0, 1, self.oob_band_width_var, "BAND WIDTH m",
                       "Painted width (m) of the OOB band across the boundary.")
        add_oob_field(1, 0, self.oob_merge_gap_var, "MERGE GAP m",
                       "Morphological-closing radius (m) -- bridges gaps up to ~2x this between "
                       "playfield pieces so the OOB line follows the outer course boundary, not "
                       "every fairway fragment. 0 disables.")
        add_oob_field(1, 1, self.oob_simplify_tol_var, "SIMPLIFY m",
                       "Douglas-Peucker tolerance (m) on the boundary curve -- higher = fewer, "
                       "longer square stamps and a coarser OOB line that cuts corners.")
        add_oob_field(2, 0, self.oob_cap_ratio_var, "CAP RATIO",
                       "Round-cap (type 8) half-extent as a fraction of the square's across-path "
                       "half-extent. ~0.8 matches the reference course.")

        caps_cb = ttk.Checkbutton(parent, text="Include round caps at boundary vertices",
                                   variable=self.oob_include_caps_var)
        caps_cb.pack(anchor="w", pady=2)
        _Tooltip(caps_cb, "Place a round type-8 stamp at each boundary vertex to round the "
                 "corners between straight square segments.")

        oob_btn_row = ttk.Frame(parent)
        oob_btn_row.pack(anchor="w", pady=2)
        ttk.Button(oob_btn_row, text="Generate OOB", width=22,
                    command=self._run_generate_oob).pack(side="left")
        clear_oob_btn = ttk.Button(oob_btn_row, text="Clear OOB", width=12,
                                    command=self._run_clear_oob)
        clear_oob_btn.pack(side="left", padx=(4, 0))
        _Tooltip(clear_oob_btn, "Delete the generated OOB band (oob.json + previews) and mark "
                 "the layer empty. Re-run Write Terrain + Repack to apply.")
        ttk.Label(
            parent,
            text="Then run Write Terrain (Terrain tab) + Repack to ship it.",
            foreground="#555",
        ).pack(anchor="w")

    def _run_generate_oob(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "generate-oob"]
        for flag, var in (
            ("--oob-inner-buffer", self.oob_inner_buffer_var),
            ("--oob-band-width", self.oob_band_width_var),
            ("--oob-merge-gap", self.oob_merge_gap_var),
            ("--oob-simplify-tol", self.oob_simplify_tol_var),
            ("--oob-cap-ratio", self.oob_cap_ratio_var),
        ):
            val = var.get().strip()
            if val:
                args += [flag, val]
        if not self.oob_include_caps_var.get():
            args.append("--oob-no-caps")
        self._run_step(args, wd)

    def _run_clear_oob(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "generate-oob", "--oob-clear"], wd)

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
        self.show_objects_var = tk.BooleanVar(value=True)
        show_objects_checkbox = ttk.Checkbutton(
            show_objects_row, text="Show objects", variable=self.show_objects_var,
            command=self._on_show_objects_toggled,
        )
        show_objects_checkbox.pack(side="left")
        _Tooltip(show_objects_checkbox, "Overlay placed objects on the preview, similar to Overlay OSM "
                 "(Preview tab) -- individually placed trees (object_list.json, after Generate Trees) "
                 "as a 2px dot, or a circle sized to its measured LIDAR canopy radius if it has one; "
                 "rock/grass/ground-cover/display-plant cluster-fill stamps (placedObjects2.json, after "
                 "Write Objects) as a circle at that stamp's own packed radius; each OSM water body as "
                 "the outlined, rotated rectangle its water object will actually be sized/placed to "
                 "(live from features.geojson -- no Write Water run needed). Colored by category, "
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
            height=10, selectmode="extended",
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
        self.objects_tree.bind("<Double-1>", self._on_object_row_double_click)
        obj_tree_scroll = ttk.Scrollbar(obj_tree_frame, orient="vertical", command=self.objects_tree.yview)
        obj_tree_scroll.pack(side="left", fill="y")
        self.objects_tree["yscrollcommand"] = obj_tree_scroll.set

        obj_button_row = ttk.Frame(parent)
        obj_button_row.pack(fill="x", pady=(6, 0))
        delete_btn = ttk.Button(obj_button_row, text="Delete", command=self._delete_selected_objects)
        delete_btn.pack(side="left")
        _Tooltip(delete_btn, "Permanently delete the currently selected row(s) -- individual trees "
                 "are removed from object_list.json; cluster-fill groups have just their own "
                 "{category, type, ratio, source} spec stripped from the spline(s) that carry it "
                 "(other specs on the same spline are left alone), or the whole synthetic border/"
                 "masked Feature deleted if that spec was the only thing on it. Re-run Generate "
                 "Trees / Fill with Clusters to get them back.")
        delete_all_btn = ttk.Button(obj_button_row, text="Delete All", command=self._delete_all_filtered_objects)
        delete_all_btn.pack(side="left", padx=(4, 0))
        _Tooltip(delete_all_btn, "Delete every tree and cluster-fill group currently shown (i.e. "
                 "matching the active Filter), same as Delete but scoped to the whole filtered list "
                 "rather than just the selection. Set Filter to 'LIDAR' first to dump only "
                 "auto-detected trees, keeping any hand-tagged OSM ones -- useful for clearing out a "
                 "bad detection run without losing curated data.")

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

        # Minimum detected canopy height (m above ground) for a crown to
        # count as a tree -- the single biggest lever on how many trees
        # come out (see ingest/tree_detection.py's DEFAULT_MIN_HEIGHT_M).
        # Persists to project.json (objects_tree_min_height_m) on every
        # Generate Trees run; loaded back when the working directory
        # changes (see _on_working_dir_changed).
        min_height_row = ttk.Frame(parent)
        min_height_row.pack(anchor="w", fill="x", pady=(0, 4))
        ttk.Label(min_height_row, text="Min tree height (m):").pack(side="left")
        self.tree_min_height_var = tk.DoubleVar(value=float(DEFAULT_LIDAR_TREE_MIN_HEIGHT_M))
        ttk.Scale(
            min_height_row, from_=0.0, to=20.0, orient="horizontal",
            variable=self.tree_min_height_var,
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.tree_min_height_text = tk.StringVar(value=f"{DEFAULT_LIDAR_TREE_MIN_HEIGHT_M:.0f}")

        def _commit_tree_min_height(_evt=None):
            try:
                val = float(self.tree_min_height_text.get())
            except (TypeError, ValueError):
                val = self.tree_min_height_var.get()
            val = max(0.0, min(20.0, val))
            self.tree_min_height_var.set(val)
            self.tree_min_height_text.set(f"{val:.1f}")

        entry = ttk.Entry(min_height_row, textvariable=self.tree_min_height_text, width=5)
        entry.pack(side="left")
        entry.bind("<Return>", _commit_tree_min_height)
        entry.bind("<FocusOut>", _commit_tree_min_height)
        _Tooltip(entry, "Minimum detected canopy height (meters above ground) for a LIDAR-detected "
                 "crown to count as a tree. Higher = fewer, bigger trees (cuts low understory/hedges); "
                 "lower = more detections, including small young trees that render tiny in-game "
                 "(scale = detected height / asset native height). Applies to 'Detect trees from LIDAR "
                 "canopy' only; OSM natural=tree nodes are unaffected. Persists as the default for "
                 "this working directory.")
        self._add_step_button(parent, "Generate Trees", self._run_generate_trees)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Stake Buildings", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        stake_grid = ttk.Frame(parent)
        stake_grid.pack(anchor="w", fill="x")
        stake_btn = ttk.Button(stake_grid, text="Place", command=self._run_stake_buildings)
        stake_btn.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        _Tooltip(stake_btn, "Places a subtle 0.5x-scale stake at every corner of every 'building' "
                 "feature (features.geojson) and re-writes placedObjects2.json -- the fence-post prop "
                 "used oversized for the cart-path debug marker (v2019) or the matching WoodFences "
                 "post (v2021+). Needs Write Objects to have been run at least "
                 "once first (object_list.json). Persists as the default for future Write Objects "
                 "runs too -- use Clear Building Stakes to turn it back off.")
        clear_stakes_btn = ttk.Button(stake_grid, text="Clear", command=self._run_clear_building_stakes)
        clear_stakes_btn.grid(row=0, column=1, sticky="w", padx=(4, 0), pady=2)
        _Tooltip(clear_stakes_btn, "Re-writes placedObjects2.json without building stakes. Not "
                 "destructive -- building corners are recomputed from features.geojson each time, so "
                 "Stake Buildings can always bring them back.")
        
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Streams", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        streams_grid = ttk.Frame(parent)
        streams_grid.pack(fill="x", pady=2)

        def add_stream_field(row, col, var, label, tooltip):
            cell = ttk.Frame(streams_grid)
            cell.grid(row=row, column=col, sticky="w", padx=3, pady=2)
            lbl = ttk.Label(cell, text=label)
            lbl.pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=var, width=10)
            entry.pack(anchor="w")
            _Tooltip(lbl, tooltip)
            _Tooltip(entry, tooltip)

        self.stream_depth_var = tk.StringVar(value=f"{STREAM_DEPTH_M}")
        self.stream_half_width_var = tk.StringVar(value=f"{STREAM_HALF_WIDTH_M}")
        self.stream_water_fill_depth_var = tk.StringVar(value=f"{STREAM_WATER_FILL_DEPTH_M}")
        self.stream_water_base_width_var = tk.StringVar(value=f"{STREAM_WATER_BASE_WIDTH_M}")
        self.stream_water_widen_per_depth_var = tk.StringVar(value=f"{STREAM_WATER_WIDEN_PER_DEPTH}")
        self.stream_water_widen_per_descent_var = tk.StringVar(value=f"{STREAM_WATER_WIDEN_PER_DESCENT}")
        self.stream_water_level_margin_var = tk.StringVar(value=f"{STREAM_WATER_LEVEL_MARGIN_M}")
        self.stream_bank_veg_width_var = tk.StringVar(value=f"{BANK_VEG_WIDTH_M}")

        add_stream_field(0, 0, self.stream_depth_var, "DEPTH m",
                         "Streambed carve depth (m) below original grade. "
                         f"Default {STREAM_DEPTH_M}.")
        add_stream_field(0, 1, self.stream_half_width_var, "HALF-WIDTH m",
                         "Trench stamp radius (m, center-to-edge); carved channel reads about "
                         f"twice this wide. Also the heightmap probe radius. Default {STREAM_HALF_WIDTH_M}.")
        add_stream_field(0, 2, self.stream_bank_veg_width_var, "BANK VEG m",
                         "Half-width (m) of the stream-bank vegetation buffer polygon tagged for "
                         f"cluster fills. Default {BANK_VEG_WIDTH_M}.")
        add_stream_field(1, 0, self.stream_water_fill_depth_var, "WATER FILL m",
                         "Water film depth (m) at a water tile's shallow (upstream) end. "
                         f"Default {STREAM_WATER_FILL_DEPTH_M}.")
        add_stream_field(1, 1, self.stream_water_base_width_var, "WATER WIDTH m",
                         "Flowing-water cross-flow span (m) at zero extra depth. "
                         f"Default {STREAM_WATER_BASE_WIDTH_M}.")
        add_stream_field(1, 2, self.stream_water_widen_per_depth_var, "WIDEN / DEPTH",
                         "Extra water width (m) per m of water depth at a tile's deep end. "
                         f"Default {STREAM_WATER_WIDEN_PER_DEPTH}.")
        add_stream_field(2, 0, self.stream_water_widen_per_descent_var, "WIDEN / DESCENT",
                         "Extra water width (m) per m of total bed descent from the stream source. "
                         f"Default {STREAM_WATER_WIDEN_PER_DESCENT}.")
        add_stream_field(2, 1, self.stream_water_level_margin_var, "WATER LEVEL MARGIN m",
                         "How far (m) below the fitted carved streambed the flowing-water surface "
                         "sits. Write Water / Write Objects fit each tile's level to the real "
                         f"terrain (like a pond), then subtract this. Default {STREAM_WATER_LEVEL_MARGIN_M}.")

        streams_btn = ttk.Button(parent, text="Generate", command=self._run_generate_streams)
        streams_btn.pack(anchor="w", pady=2)
        _Tooltip(streams_btn, "Turns every OSM waterway=stream / waterway=ditch line (a water "
                 "LineString in features.geojson) into a stream: a downhill-carved streambed added "
                 "as a new terrain stamp layer, flowing-water strips + waterfall records frozen into "
                 "streams.json, and a buffered stream-bank vegetation polygon tagged for cluster "
                 "fill. Needs Ingest OSM + a terrain layer already. Re-run Write Terrain, Write "
                 "Water AND Write Objects afterwards to render it. Re-runnable; re-run after any "
                 "fresh Ingest OSM (that drops the bank tags).")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Parking", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        parking_grid = ttk.Frame(parent)
        parking_grid.pack(fill="x", pady=2)
        self.parking_spacing_var = tk.StringVar(value=f"{PARKING_SPACING_M}")
        self.parking_offset_var = tk.StringVar(value=f"{PARKING_OFFSET_M}")
        self.parking_skip_prob_var = tk.StringVar(value=f"{PARKING_SKIP_PROB}")
        self.parking_max_variants_var = tk.StringVar(value=f"{PARKING_MAX_VARIANTS}")
        self.parking_accent_count_var = tk.StringVar(value=f"{PARKING_ACCENT_COUNT}")
        self.parking_color_weights_var = tk.StringVar(
            value=",".join(f"{k}={v:g}" for k, v in PARKING_COLOR_WEIGHTS.items()))
        self.parking_sides_var = tk.StringVar(value=PARKING_SIDES)
        self.parking_orientation_var = tk.StringVar(value=PARKING_ORIENTATION)

        def _pk_field(col, var, label, tip, values=None):
            cell = ttk.Frame(parking_grid)
            cell.grid(row=col // 3, column=col % 3, sticky="w", padx=3, pady=2)
            lbl = ttk.Label(cell, text=label)
            lbl.pack(anchor="w")
            w = (ttk.Combobox(cell, textvariable=var, values=values, width=12, state="readonly")
                 if values else ttk.Entry(cell, textvariable=var, width=10))
            w.pack(anchor="w")
            _Tooltip(lbl, tip)
            _Tooltip(w, tip)

        _pk_field(0, self.parking_spacing_var, "SPACING m",
                  f"Along-row centre-to-centre car spacing. Default {PARKING_SPACING_M}.")
        _pk_field(1, self.parking_offset_var, "OFFSET m",
                  f"Perpendicular distance from the aisle line to the car row. Default {PARKING_OFFSET_M}.")
        _pk_field(2, self.parking_skip_prob_var, "SKIP PROB",
                  f"Fraction of stalls left empty (0-1). Default {PARKING_SKIP_PROB}.")
        _pk_field(3, self.parking_max_variants_var, "MAX MODELS",
                  "Cap on the number of distinct car prefabs used across the whole course -- "
                  "keeps parked cars off the game's placed-object type budget. 0 = no cap. "
                  f"Default {PARKING_MAX_VARIANTS}. >= 1 prefab is kept per active colour. Vans "
                  "are excluded unless a pga_parking tag explicitly asks for 'van'.")
        _pk_field(4, self.parking_accent_count_var, "ACCENTS",
                  "How many accent colours (from red/green/blue/yellow) to add to the base mix, "
                  f"chosen once per course at 0.10 weight each. Default {PARKING_ACCENT_COUNT}.")
        _pk_field(5, self.parking_sides_var, "SIDES",
                  "Which side of the DIRECTIONAL aisle line to fill. Default 'left' -- draw the "
                  "line the other way to put cars on the other side.",
                  values=["left", "right", "both"])
        _pk_field(6, self.parking_orientation_var, "ORIENTATION",
                  "'perpendicular' nose-in stalls, or 'parallel' kerbside parking.",
                  values=["perpendicular", "parallel"])

        cw_row = ttk.Frame(parent)
        cw_row.pack(anchor="w", fill="x", pady=(2, 0))
        cw_lbl = ttk.Label(cw_row, text="COLOUR MIX")
        cw_lbl.pack(side="left")
        cw_entry = ttk.Entry(cw_row, textvariable=self.parking_color_weights_var)
        cw_entry.pack(side="left", fill="x", expand=True, padx=4)
        _cw_tip = ("Car colour weights as 'colour=weight,...' (normalised; percentages fine). A "
                   "colour not listed here and not an accent never spawns. Catalogue colours: "
                   "black, gray, white, red, green, blue, yellow, plus a few rare ones. "
                   f"Default '{self.parking_color_weights_var.get()}' + {PARKING_ACCENT_COUNT} accents.")
        _Tooltip(cw_lbl, _cw_tip)
        _Tooltip(cw_entry, _cw_tip)

        parking_btn = ttk.Button(parent, text="Generate", command=self._run_generate_parking)
        parking_btn.pack(anchor="w", pady=2)
        _Tooltip(parking_btn, "Lines every OSM way tagged pga_parking=<pool> (an aisle / service "
                 "lane) with parked-car props from course_output/vehicle_catalog.json, frozen into "
                 "parking.json. The line is DIRECTIONAL -- cars go on its LEFT side by default "
                 "(draw it the other way, or set SIDES, to flip). Tag value picks the pool: "
                 "'yes'/'all' = whole catalogue, or a comma list of colours and/or 'car'/'van' "
                 "(e.g. pga_parking=black,white,van). A 'ref' tag on the way (0..1) scales that "
                 "aisle's occupancy: populated = (1 - SKIP PROB) * ref. Cars bank pitch/roll to "
                 "the ground slope (needs Ingest LAZ). Needs Ingest OSM. v2021+ only. Re-run after "
                 "any fresh Ingest OSM; re-pack objects afterwards (this does it).")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Range Nets", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        range_nets_btn = ttk.Button(parent, text="Generate", command=self._run_generate_range_nets)
        range_nets_btn.pack(anchor="w", pady=2)
        _Tooltip(range_nets_btn, "Tiles the built-in range-net module (course_output/"
                  "range_net_module.py -- 4 stacked wire panels + a post each end + the buried "
                  "brick anchor, one 8 m section) along every OSM way tagged barrier=range_nets, "
                  "in a string of posts at exact 8 m increments with shared seam posts (no "
                  "doubled posts). Connected ways are grouped into chains and the inner corner "
                  "points are repositioned so every side is an exact multiple of 8 m (your "
                  "box-shaped, one-side-missing layout). Each object carries a dy (y = terrain "
                  "height + 1 m stamp datum, resolved at Write Objects), not a frozen y. Needs "
                  "Ingest OSM. v2021+ only. Re-run after any fresh Ingest OSM; re-pack objects "
                  "afterwards (this does it).")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Collections", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))

        lib_row = ttk.Frame(parent)
        lib_row.pack(anchor="w", fill="x", pady=(0, 2))
        ttk.Label(lib_row, text="Library:").pack(side="left")
        self.collections_library_var = tk.StringVar(value=str(default_library_dir()))
        lib_entry = ttk.Entry(lib_row, textvariable=self.collections_library_var, width=22)
        lib_entry.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(lib_row, text="...", width=3, command=self._browse_collection_library).pack(side="left")
        _Tooltip(lib_entry, "Directory of *.json collection templates (course_output/collection_library.py). "
                 "Default ~/.pga2k/collections/. Persisted per project as 'collections_library_dir'.")

        cap_gen_row = ttk.Frame(parent)
        cap_gen_row.pack(anchor="w", fill="x", pady=2)
        cap_btn = ttk.Button(cap_gen_row, text="Capture from .course...", command=self._capture_collection_dialog)
        cap_btn.pack(side="left")
        _Tooltip(cap_btn, "Pick a .course file and snapshot EVERY placed object + surface spline in it "
                 "into one collection template (anchor = their centroid, heading 0). Edit the saved "
                 "JSON afterward to trim/tune members.")
        gen_btn = ttk.Button(cap_gen_row, text="Generate", command=self._run_generate_collections)
        gen_btn.pack(side="left", padx=(4, 0))
        _Tooltip(gen_btn, "Resolve every OSM 2-node way tagged pga_collection=<template name> "
                 "(features.geojson) against the library into collections.json. Needs Ingest OSM. "
                 "Re-run Write Objects AND Write Splines afterwards to render. Re-run after any fresh "
                 "Ingest OSM.")

        self.collections_templates_tree = ttk.Treeview(
            parent, columns=("objects", "splines"), show="tree headings", height=5, selectmode="browse",
        )
        self.collections_templates_tree.heading("#0", text="Template")
        self.collections_templates_tree.heading("objects", text="Obj")
        self.collections_templates_tree.heading("splines", text="Spl")
        self.collections_templates_tree.column("#0", width=160)
        self.collections_templates_tree.column("objects", width=44, anchor="center")
        self.collections_templates_tree.column("splines", width=44, anchor="center")
        self.collections_templates_tree.pack(fill="x", pady=(4, 0))
        self.collections_templates_tree.bind(
            "<<TreeviewSelect>>", lambda _e: self._sync_push_collection_btn()
        )

        self._push_collection_btn = ttk.Button(
            parent, text="Push to Game for Editing", state="disabled",
            command=self._run_push_collection_to_game,
        )
        self._push_collection_btn.pack(anchor="w", pady=(4, 0))
        _Tooltip(self._push_collection_btn,
                 "Build a fresh .course holding the SELECTED template's objects, surface splines and "
                 "raise-tool terrain stamps at the course centre (heading 0), give it a unique "
                 "'COLL-<name>-<timestamp>' serial + course id, and copy it into this Game version's "
                 "in-game Courses folder -- open it in the game's editor to tweak the collection, then "
                 "re-run 'Capture from .course...' with the same template name to fold the edits back "
                 "in. The inverse of Capture. Independent of the pipeline: does NOT touch course/. "
                 "Needs a working directory (for staging) + Game version + Theme.")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(parent, text="Tree assets (v2021+)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 2))
        tree_assets_row = ttk.Frame(parent)
        tree_assets_row.pack(anchor="w", fill="x", pady=(0, 2))
        self._tree_assets_summary = ttk.Label(tree_assets_row, text="")
        self._tree_assets_summary.pack(side="left")
        self._tree_assets_edit_btn = ttk.Button(
            tree_assets_row, text="Edit…", command=self._open_tree_assets_dialog,
        )
        self._tree_assets_edit_btn.pack(side="right")
        _Tooltip(self._tree_assets_edit_btn,
                 "Which Unity tree prefabs v2021+ Write Objects plants (v2019 uses numeric theme ids "
                 "instead, so this is ignored there). Leave empty to use the bundled rustic catalog "
                 "(course_output/asset_catalog.json) -- it also routes pine/deciduous by the "
                 "pga_tree_type hints, matching v2019. Pick assets here to override with your own "
                 "pool. Persisted as 'objects_tree_asset_paths'.")
        self._refresh_tree_assets_summary()
        self._sync_tree_assets_enabled()

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Write Objects", self._run_write_objects)

    def _refresh_tree_assets_summary(self) -> None:
        n = len(self.objects_tree_asset_paths)
        self._tree_assets_summary.configure(
            text=f"{n} custom asset(s) selected" if n else "Using bundled rustic catalog default"
        )

    def _sync_tree_assets_enabled(self) -> None:
        """v2021+ only -- grey the picker on a v2019 project (the CLI flag
        is silently ignored there, see step_write_objects)."""
        is_v2021 = self.game_version.get() != "2019"
        state = "normal" if is_v2021 else "disabled"
        self._tree_assets_edit_btn.configure(state=state)
        self._tree_assets_summary.configure(foreground="" if is_v2021 else "grey")

    def _open_tree_assets_dialog(self) -> None:
        """Multi-select picker over the catalog's category-0 tree prefabs
        (mirrors _open_cluster_fill_dialog). Empty selection => fall back
        to the bundled rustic catalog default at write-objects time."""
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            messagebox.showwarning("No working directory", "Set a working directory first.")
            return
        entries = sorted(
            (e for e in ASSET_ENTRIES if e.category == 0 and "/Trees/" in e.path),
            key=lambda e: e.display,
        )
        dlg = tk.Toplevel(self.root)
        dlg.title("Tree assets")
        dlg.transient(self.root)
        ttk.Label(dlg, text="Prefabs v2021+ Write Objects may plant. Select none to use the "
                  "bundled rustic catalog default.", wraplength=420).pack(anchor="w", padx=8, pady=(8, 4))
        tree = ttk.Treeview(dlg, columns=("native_h",), show="tree headings", height=16, selectmode="extended")
        tree.heading("#0", text="Asset")
        tree.heading("native_h", text="Native height (m)")
        tree.column("#0", width=280)
        tree.column("native_h", width=120, anchor="center")
        tree.pack(fill="both", expand=True, padx=8)
        iid_to_path = {}
        for e in entries:
            iid = tree.insert("", "end", text=e.display,
                              values=(f"{e.native_height_m:.1f}" if e.native_height_m else "-",))
            iid_to_path[iid] = e.path
            if e.path in self.objects_tree_asset_paths:
                tree.selection_add(iid)

        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill="x", padx=8, pady=8)

        def _save() -> None:
            self.objects_tree_asset_paths = [iid_to_path[i] for i in tree.selection()]
            self._suppress_tree_assets_save = True
            try:
                save_project(Path(wd), {"objects_tree_asset_paths": self.objects_tree_asset_paths})
            finally:
                self._suppress_tree_assets_save = False
            self._refresh_tree_assets_summary()
            dlg.destroy()

        ttk.Button(btn_row, text="OK", command=_save).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 4))

    def _collection_library_dir(self) -> Path:
        text = self.collections_library_var.get().strip()
        return Path(text) if text else default_library_dir()

    def _browse_collection_library(self) -> None:
        d = filedialog.askdirectory(title="Select the collection template library directory")
        if d:
            self.collections_library_var.set(d)
            wd = self.working_dir.get().strip()
            if wd and Path(wd).is_dir():
                save_project(Path(wd), {"collections_library_dir": d})
            self._refresh_collection_templates()

    def _refresh_collection_templates(self) -> None:
        tree = getattr(self, "collections_templates_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        try:
            library = load_library(self._collection_library_dir())
        except OSError:
            library = {}
        for name, collection in sorted(library.items()):
            tree.insert("", "end", text=name, values=(len(collection.objects), len(collection.splines)))
        self._sync_push_collection_btn()

    def _selected_collection_name(self) -> Optional[str]:
        """The template name of the highlighted Collections-tree row, or None."""
        tree = getattr(self, "collections_templates_tree", None)
        if tree is None:
            return None
        sel = tree.selection()
        return tree.item(sel[0], "text") if sel else None

    def _sync_push_collection_btn(self) -> None:
        """Enable 'Push to Game for Editing' only while a template row is selected."""
        btn = getattr(self, "_push_collection_btn", None)
        if btn is not None:
            btn.config(state="normal" if self._selected_collection_name() else "disabled")

    def _capture_collection_dialog(self) -> None:
        course_file = filedialog.askopenfilename(
            title="Select a .course file to capture a collection from",
            filetypes=[(".course files", "*.course"), ("All files", "*.*")],
            **self._courses_dir_dialog_kwarg(),
        )
        if not course_file:
            return
        default_name = Path(course_file).stem
        name = simpledialog.askstring(
            "Collection name", "Name for this collection template:", initialvalue=default_name, parent=self.root,
        )
        if not name:
            return

        libdir = self._collection_library_dir()
        existing_path = Path(libdir) / f"{_collection_slug(name)}.json"
        tuned_paths = _hand_tuned_member_paths(existing_path)
        if tuned_paths:
            listing = "\n".join(f"  - {p}" for p in tuned_paths)
            proceed = messagebox.askyesno(
                "Overwrite hand-tuned template?",
                f"{existing_path.name} already has hand-added variant fields "
                f"(param_option / variants / a \"{{param}}\" path token) on:\n\n{listing}\n\n"
                "Re-capturing from a .course file always rebuilds the template from scratch "
                "and does NOT carry these over -- they will be lost unless you re-add them "
                "afterward. Continue anyway?",
            )
            self._append_log(
                f"[NOTE: {existing_path.name} has hand-tuned variant field(s) on "
                f"{len(tuned_paths)} member(s) -- {'overwriting anyway' if proceed else 'capture cancelled'}: "
                + ", ".join(tuned_paths) + "]\n"
            )
            if not proceed:
                return

        tmp_dir = Path(tempfile.mkdtemp(prefix="pga2k_collection_"))
        try:
            self._append_log(f"\n[capturing collection {name!r} from {course_file}]\n")
            script = Path(__file__).resolve().parent / "util" / "course_extract.py"
            result = subprocess.run(
                [sys.executable, str(script), str(course_file), str(tmp_dir)],
                capture_output=True, text=True,
            )
            if result.stdout:
                self._append_log(result.stdout)
            if result.returncode != 0:
                self._append_log(result.stderr or "course_extract.py failed\n")
                messagebox.showerror("Capture failed", "Could not extract the .course file -- see the log.")
                return
            collection = capture_from_course(
                tmp_dir / "CourseDescription_nodes", name,
                printf=lambda s: self._append_log(s + "\n"),
            )
            out_path = save_collection(collection, libdir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        wd = self.working_dir.get().strip()
        if wd and Path(wd).is_dir():
            save_project(Path(wd), {"collections_library_dir": str(libdir)})
        self._append_log(
            f"[captured {len(collection.objects)} object(s), {len(collection.splines)} spline(s) "
            f"-> {out_path}]\n"
        )
        self._refresh_collection_templates()

    def _run_generate_collections(self) -> None:
        """Objects / Collections / Generate -- runs the generate-collections
        CLI step (features.geojson pga_collection lines -> collections.json).
        Touches objects AND splines, so the user still has to re-run Write
        Objects and Write Splines. on_done re-packs objects.json and
        refreshes the Objects list."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "generate-collections", "--collection-library", str(self._collection_library_dir())]
        self._run_step(args, wd, on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True))

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

    def _on_objects_step_done(self, wd: str, regenerate_packed: bool) -> None:
        """
        Shared _run_step on_done callback for Generate Trees / Write
        Objects / Stake Buildings -- refreshes the Objects tab list
        (rebuilt from whatever object_list.json/features.geojson now
        hold) once the subprocess has actually finished. regenerate_
        packed=True (Generate Trees only -- the other two only ever
        FORMAT objects.json, never change what trees/cluster fills
        exist, so they instead pre-regenerate it BEFORE launching, see
        _run_write_objects) re-packs objects.json first, so the "Show
        objects" preview and this list both pick up the new tree set
        immediately, without a separate Splines-tab action needed.
        """
        working_dir = Path(wd)
        if regenerate_packed:
            self._regenerate_packed_objects(working_dir)
        self._refresh_objects_list()

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

        # Persist the current min-height slider as this working
        # directory's sticky default (the CLI reads the same
        # project.json key), so a later CLI run matches the GUI.
        save_project(Path(wd), {"objects_tree_min_height_m": float(self.tree_min_height_var.get())})
        args = [
            "--step", "generate-trees",
            "--detect-lidar-trees" if self.detect_lidar_trees_var.get() else "--no-detect-lidar-trees",
            f"--tree-min-height={self.tree_min_height_var.get()}",
        ]
        # regenerate objects.json (which embeds a COPY of whatever
        # object_list.json holds -- see _regenerate_packed_objects) as
        # part of on_done, not right away -- generate-trees runs as a
        # background subprocess (see _run_step), so object_list.json
        # doesn't actually have this run's new trees in it until the
        # "done" queue message arrives; regenerating any sooner would
        # just re-pack the PREVIOUS run's tree set.
        self._run_step(args, wd, on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True))

    def _run_write_objects(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        # write-objects (unlike generate-trees) only ever FORMATS
        # objects.json -- it never changes what trees/cluster fills
        # exist -- so it's safe (and necessary, now that write-objects
        # requires the file to exist) to make sure it's current
        # BEFORE launching the step, rather than racing it via on_done.
        self._regenerate_packed_objects(Path(wd))
        args = ["--step", "write-objects", "--tree-variety"]
        theme_id = self._theme_name_to_id.get(self.objects_theme_var.get())
        if theme_id is not None:
            args += ["--theme", str(theme_id)]
        args += self._tree_asset_path_args()
        self._run_step(args, wd, on_done=lambda: self._refresh_objects_list())

    def _tree_asset_path_args(self) -> list[str]:
        """--tree-asset-path flags for the v2021+ picker (see
        _open_tree_assets_dialog). Empty when nothing's selected -- the
        CLI then falls back to the bundled rustic catalog, or errors for
        a non-rustic theme."""
        args: list[str] = []
        for p in self.objects_tree_asset_paths:
            args += ["--tree-asset-path", p]
        return args

    def _run_stake_buildings(self) -> None:
        self._run_write_objects_with_stakes(True)

    def _run_clear_building_stakes(self) -> None:
        self._run_write_objects_with_stakes(False)

    def _run_generate_streams(self) -> None:
        """Objects / Generate Streams -- runs the generate-streams CLI
        step (streambed stamp layer + streams.json + stream-bank
        vegetation tags from OSM waterway=stream/ditch lines). Touches
        terrain, water AND objects, so it can't finish the job on its
        own: the user still has to re-run Write Terrain, Write Water and
        Write Objects. on_done re-packs objects.json (features.geojson
        just changed) and refreshes the Objects list."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "generate-streams"]
        for flag, var in (
            ("--stream-depth", self.stream_depth_var),
            ("--stream-half-width", self.stream_half_width_var),
            ("--stream-water-fill-depth", self.stream_water_fill_depth_var),
            ("--stream-water-base-width", self.stream_water_base_width_var),
            ("--stream-water-widen-per-depth", self.stream_water_widen_per_depth_var),
            ("--stream-water-widen-per-descent", self.stream_water_widen_per_descent_var),
            ("--stream-water-level-margin", self.stream_water_level_margin_var),
            ("--stream-bank-veg-width", self.stream_bank_veg_width_var),
        ):
            val = var.get().strip()
            if val:
                args += [flag, val]
        self._run_step(
            args, wd,
            on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True),
        )

    def _run_generate_parking(self) -> None:
        """Objects / Parking / Generate -- runs the generate-parking CLI
        step (parking.json from OSM pga_parking lines + vehicle_catalog).
        on_done re-packs objects.json so the cars show in the preview and
        refreshes the Objects list; still needs a Write Objects + Repack
        to reach the game."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "generate-parking"]
        for flag, var in (
            ("--parking-spacing", self.parking_spacing_var),
            ("--parking-offset", self.parking_offset_var),
            ("--parking-skip-prob", self.parking_skip_prob_var),
            ("--parking-max-variants", self.parking_max_variants_var),
            ("--parking-accent-count", self.parking_accent_count_var),
            ("--parking-color-weights", self.parking_color_weights_var),
            ("--parking-sides", self.parking_sides_var),
            ("--parking-orientation", self.parking_orientation_var),
        ):
            val = var.get().strip()
            if val:
                args += [flag, val]
        self._run_step(
            args, wd,
            on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True),
        )

    def _run_generate_range_nets(self) -> None:
        """Objects / Range Nets / Generate -- runs the generate-range-nets
        CLI step (range_nets.json from OSM barrier=range_nets ways + the
        built-in module). on_done re-packs objects.json so the nets show
        in the preview and refreshes the Objects list; still needs a
        Write Objects + Repack to reach the game."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "generate-range-nets"]
        self._run_step(
            args, wd,
            on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True),
        )

    def _run_write_objects_with_stakes(self, stake_buildings: bool) -> None:
        """Shared by the Stake Buildings / Clear Building Stakes buttons --
        both are just a Write Objects run with --stake-buildings /
        --no-stake-buildings tacked on (see objects.py's
        build_building_stake_objects_v2019 /
        build_building_stake_objects_v2021 / PGA2k_gen.py's
        step_write_objects), so building corners stay in lockstep with
        whatever's currently in features.geojson rather than needing
        separate placedObjects2.json surgery. Works for both game
        versions -- v2021+ uses objects.DEFAULT_STAKE_ASSET_PATH_V2021."""
        wd = self._require_working_dir()
        if not wd:
            return
        self._regenerate_packed_objects(Path(wd))  # see _run_write_objects's identical call
        args = [
            "--step", "write-objects", "--tree-variety",
            "--stake-buildings" if stake_buildings else "--no-stake-buildings",
        ]
        theme_id = self._theme_name_to_id.get(self.objects_theme_var.get())
        if theme_id is not None:
            args += ["--theme", str(theme_id)]
        args += self._tree_asset_path_args()
        self._run_step(args, wd, on_done=lambda: self._refresh_objects_list())

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
    def _ingame_object_detail(record: dict) -> str:
        """One-line asset label + rotation/scale for an ingame_objects.json
        record's child row -- path (v2021+) resolves through
        _ASSET_LABEL_BY_PATH, category/type (v2019) through
        _ASSET_LABEL_BY_KEY, same fallback-to-raw-key style as
        _build_cluster_fill_rows uses for cluster-fill asset labels."""
        if record.get("path"):
            label = _ASSET_LABEL_BY_PATH.get(record["path"], record["path"])
        else:
            label = _ASSET_LABEL_BY_KEY.get(
                (record.get("category"), record.get("type")),
                f"category={record.get('category')}/type={record.get('type')}",
            )
        rotation = float(record.get("rotation_deg") or 0.0)
        scale = float(record.get("scale") or 1.0)
        return f"{label} rot={rotation:.1f}° scale={scale:.2f}"

    @staticmethod
    def _build_cluster_fill_rows(features: list) -> list[tuple[str, str, float, float, str, int, int, str]]:
        """
        (spline_osm_ids, asset_label, ratio, density, source, category,
        type_, mode) -- one row per distinct
        {"category","type","ratio","density","source","mode"} fill spec
        seen across `features` (see course_output/object_clusters.py's
        PGA_CLUSTER_FILLS_TAG), with every contributing spline's osm_id
        collected together -- so a single Fill action (which can target
        many splines and several assets at once) shows as one row per
        asset/ratio/density/source/mode combo, not one row per spline.
        `source` defaults to "manual", `density` to DEFAULT_FILL_DENSITY
        (or DEFAULT_SPLINE_FILL_DENSITY for a spline-mode spec), and
        `mode` to CLUSTER_FILL_MODE_STAMPS for specs saved before
        those fields existed -- mode is part of the grouping key (not
        just display) so a stamps-mode and spline-mode spec that
        otherwise match never merge into one row (they're functionally
        different fills, and _remove_cluster_fill_groups needs to be
        able to remove just one without touching the other).
        category/type_ are carried alongside the display label so
        _on_object_selected can resolve which of placedObjects2.json's
        already-packed clusters belong to this row's asset (see there).
        """
        groups: dict[tuple, list[int]] = {}
        for f in features:
            if f.osm_id is None:
                continue
            for spec in f.tags.get(PGA_CLUSTER_FILLS_TAG) or []:
                source = spec.get("source", CLUSTER_FILL_SOURCE_MANUAL)
                mode = spec.get("mode", CLUSTER_FILL_MODE_STAMPS)
                default_density = (
                    DEFAULT_SPLINE_FILL_DENSITY
                    if mode in (CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_AUTO)
                    else DEFAULT_FILL_DENSITY
                )
                key = (
                    spec.get("category"), spec.get("type"), spec.get("ratio", DEFAULT_RASTER_RATIO),
                    spec.get("density", default_density), source, mode,
                )
                groups.setdefault(key, []).append(f.osm_id)

        rows = []
        for (category, type_, ratio, density, source, mode), osm_ids in groups.items():
            label = _ASSET_LABEL_BY_KEY.get((category, type_), f"category={category}/type={type_}")
            ids_str = ",".join(str(i) for i in sorted(set(osm_ids)))
            rows.append((ids_str, label, ratio, density, source, category, type_, mode))
        return rows

    def _refresh_objects_list(self) -> None:
        """
        Populates the Objects tab list from three independent sources:
        object_list.json (individually placed trees -- iid is a plain
        int index into self._objects_tree_list, source OSM/LIDAR),
        features.geojson's cluster-fill tags (area fills -- iid is
        "c"+index into self._cluster_fill_rows, source "manual" or
        "border"), and ingame_objects.json's `group` labels (objects
        imported from a hand-edited .course, see
        step_import_ingame_edits -- the group itself is iid "g"+index
        into self._ingame_object_groups, source "Imported"; each
        individual object in that group is a CHILD row nested under
        it, iid f"g{{j}}i{{idx}}" where idx indexes
        self._ingame_objects_records, so it can be double-clicked to
        edit that one object's position/rotation/scale -- see
        _on_object_row_double_click). Kept as three disjoint top-level
        iid namespaces so _delete_selected_objects/
        _delete_all_filtered_objects can always tell which kind of row
        a given iid is (the nested "g{j}i{idx}" child iids are never
        returned by objects_tree.get_children() with no args, and fail
        the int(iid[1:]) group-index parse those two use, so they're
        harmlessly skipped there -- individual-object delete isn't
        wired up yet, only edit).
        """
        wd = self.working_dir.get().strip()
        self.objects_tree.delete(*self.objects_tree.get_children())
        self._objects_tree_list = []
        self._cluster_fill_rows = []
        self._ingame_object_groups = []
        self._ingame_objects_records = []
        self._ingame_object_group_record_indices = []
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

        ingame_objects_path = Path(wd) / INGAME_OBJECTS_FILE
        if ingame_objects_path.exists():
            try:
                self._ingame_objects_records = load_ingame_objects(ingame_objects_path)
            except (json.JSONDecodeError, OSError, KeyError):
                self._ingame_objects_records = []
            # Group by `group` label, first-seen order, same grouping
            # summarize_ingame_object_groups does -- but keeping each
            # group's member record indices too (not just its count), so
            # every individual object can get its own child row below.
            group_indices: dict[str, list[int]] = {}
            group_order: list[str] = []
            for idx, record in enumerate(self._ingame_objects_records):
                name = record.get("group", "")
                if name not in group_indices:
                    group_indices[name] = []
                    group_order.append(name)
                group_indices[name].append(idx)
            self._ingame_object_groups = [(name, len(group_indices[name])) for name in group_order]
            self._ingame_object_group_record_indices = [group_indices[name] for name in group_order]

        filter_val = self.objects_filter_var.get()
        for i, (x, z, tags) in enumerate(self._objects_tree_list):
            source = self._object_source(tags)
            if filter_val != "All" and source != filter_val:
                continue
            self.objects_tree.insert(
                "", "end", iid=str(i), values=(f"{x:.1f}", f"{z:.1f}", source, self._object_detail(tags)),
            )

        for j, (spline_ids, asset_label, ratio, density, source, _category, _type, mode) in enumerate(
            self._cluster_fill_rows
        ):
            if filter_val not in ("All", source.capitalize()):
                continue
            if mode == CLUSTER_FILL_MODE_SPLINE:
                ceiling = SPLINE_FILL_MAX_PCT_BY_CATEGORY.get(_category, DEFAULT_SPLINE_FILL_MAX_PCT)
                detail = (f"splines={spline_ids} fill={asset_label} [spline] "
                          f"{density:g}% of max -> fillPct={ceiling * density / 100:g}")
            else:
                detail = f"splines={spline_ids} fill={asset_label} [stamps] ratio={ratio:g} density={density:g}%"
            self.objects_tree.insert("", "end", iid=f"c{j}", values=("", "", source, detail))

        if filter_val in ("All", "Imported"):
            for j, (group_name, count) in enumerate(self._ingame_object_groups):
                # open=True: the tree is built with show="headings" (no
                # tree/disclosure column, matching every other row in this
                # list), so there's no expand triangle for the user to click
                # -- children need to start open to be reachable at all.
                self.objects_tree.insert(
                    "", "end", iid=f"g{j}", open=True,
                    values=("", "", "Imported", f"group={group_name!r} ({count} object(s))"),
                )
                for idx in self._ingame_object_group_record_indices[j]:
                    record = self._ingame_objects_records[idx]
                    x, z = float(record.get("x", 0.0)), float(record.get("z", 0.0))
                    self.objects_tree.insert(
                        f"g{j}", "end", iid=f"g{j}i{idx}",
                        values=(f"{x:.1f}", f"{z:.1f}", "Imported", self._ingame_object_detail(record)),
                    )

    def _remove_ingame_object_groups_by_name(self, working_dir: Path, group_names: set[str]) -> None:
        """
        Removes every ingame_objects.json record whose `group` is in
        `group_names` -- Objects-tab counterpart of
        _remove_cluster_fill_groups, but surgical by group label
        instead of by feature/spec, since imported objects have no
        Feature/OSM tie-in at all (see course_output/ingame_objects.py's
        module docstring). No-op if the file doesn't exist. Caller is
        responsible for re-running self._refresh_objects_list() and
        self._regenerate_packed_objects() afterward.
        """
        path = working_dir / INGAME_OBJECTS_FILE
        if not path.exists():
            return
        try:
            records = load_ingame_objects(path)
        except (json.JSONDecodeError, OSError, KeyError):
            return
        kept = remove_ingame_object_groups(records, group_names)
        if len(kept) != len(records):
            save_ingame_objects(kept, path)

    def _remove_cluster_fill_groups(self, working_dir: Path, group_rows: list[tuple]) -> None:
        """
        Removes each given Objects-tab cluster-fill "group" row's own
        {category, type, ratio, density, source} spec from every Feature
        it was built from, surgically -- unlike _clear_selected_cluster_fills
        (Splines tab), which wipes ALL specs off a selected Feature at
        once, a single group row here is only one spec among
        potentially several sharing that Feature (see
        _open_cluster_fill_dialog: choosing multiple assets in one Fill
        action tags them all onto the same synthetic Feature). A
        synthetic border/masked Feature left with no specs afterward is
        deleted outright, same rationale as _clear_selected_cluster_fills.
        Caller is responsible for re-running self._refresh_objects_list()
        and self._regenerate_packed_objects() afterward, if the "Show
        objects" preview/highlight should reflect this -- not done here
        since a caller that's also touching object_list.json in the
        same action (_delete_selected_objects/_delete_all_filtered_objects)
        only needs one repack covering both, not one per helper call.
        """
        self._ensure_splines_features_fresh(working_dir)
        target_osm_ids: set[int] = set()
        match_keys: set[tuple] = set()
        for ids_str, _label, ratio, density, source, category, type_, mode in group_rows:
            target_osm_ids.update(int(s) for s in ids_str.split(",") if s)
            match_keys.add((category, type_, ratio, density, source, mode))

        remove_feature_ids: set[int] = set()
        changed = False
        for f in self._splines_features:
            if f.osm_id not in target_osm_ids:
                continue
            specs = f.tags.get(PGA_CLUSTER_FILLS_TAG)
            if not specs:
                continue
            def _spec_key(spec):
                mode = spec.get("mode", CLUSTER_FILL_MODE_STAMPS)
                default_density = (
                    DEFAULT_SPLINE_FILL_DENSITY
                    if mode in (CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_AUTO)
                    else DEFAULT_FILL_DENSITY
                )
                return (spec.get("category"), spec.get("type"),
                        spec.get("ratio", DEFAULT_RASTER_RATIO),
                        spec.get("density", default_density),
                        spec.get("source", CLUSTER_FILL_SOURCE_MANUAL), mode)

            kept = [spec for spec in specs if _spec_key(spec) not in match_keys]
            if len(kept) == len(specs):
                continue
            changed = True
            if kept:
                f.tags[PGA_CLUSTER_FILLS_TAG] = kept
            elif f.kind in self._SYNTHETIC_CLUSTER_KINDS:
                remove_feature_ids.add(f.osm_id)
                centerline_id = f.tags.get(PGA_CLUSTER_CENTERLINE_REF_TAG)
                if centerline_id is not None:
                    remove_feature_ids.add(centerline_id)  # its SYNTHETIC_BORDER_CENTERLINE_KIND companion
            else:
                f.tags.pop(PGA_CLUSTER_FILLS_TAG, None)

        if not changed:
            return
        if remove_feature_ids:
            self._splines_features = [f for f in self._splines_features if f.osm_id not in remove_feature_ids]
        save_features(self._splines_features, working_dir / FEATURES_FILE)
        self._refresh_splines_list()

    def _delete_selected_objects(self) -> None:
        """
        Deletes exactly the currently-selected row(s) in the Objects
        list -- individual trees (iid is a digit), cluster-fill groups
        (iid prefixed "c"), and/or imported in-game object groups (iid
        prefixed "g") together, so all three kinds can be cleared from
        this tab without a trip to the Splines tab. Objects-tab
        counterpart of the Splines tab's Mask button.
        """
        wd = self._require_working_dir()
        if not wd:
            return
        selection = self.objects_tree.selection()
        tree_indices = {int(iid) for iid in selection if iid.isdigit()}
        group_rows = []
        ingame_group_names = set()
        for iid in selection:
            if iid.startswith("c"):
                try:
                    j = int(iid[1:])
                except ValueError:
                    continue
                if 0 <= j < len(self._cluster_fill_rows):
                    group_rows.append(self._cluster_fill_rows[j])
            elif iid.startswith("g"):
                try:
                    j = int(iid[1:])
                except ValueError:
                    continue
                if 0 <= j < len(self._ingame_object_groups):
                    ingame_group_names.add(self._ingame_object_groups[j][0])
        if not tree_indices and not group_rows and not ingame_group_names:
            return
        if not messagebox.askyesno(
            "Delete selected objects",
            f"Permanently delete {len(tree_indices)} tree(s), {len(group_rows)} cluster-fill group(s), "
            f"and {len(ingame_group_names)} imported group(s)? This can't be undone.",
        ):
            return

        if tree_indices:
            self._objects_tree_list = [
                tree for i, tree in enumerate(self._objects_tree_list) if i not in tree_indices
            ]
            save_object_list(self._objects_tree_list, Path(wd) / OBJECT_LIST_FILE)
        if group_rows:
            self._remove_cluster_fill_groups(Path(wd), group_rows)
        if ingame_group_names:
            self._remove_ingame_object_groups_by_name(Path(wd), ingame_group_names)

        self._append_log(
            f"\n[deleted {len(tree_indices)} tree(s), {len(group_rows)} cluster-fill group(s), "
            f"{len(ingame_group_names)} imported group(s)]\n"
        )
        self._regenerate_packed_objects(Path(wd))
        self._refresh_objects_list()
        self._show_preview()

    def _delete_all_filtered_objects(self) -> None:
        """
        Deletes everything currently shown in the Objects list (i.e.
        matching the active Filter) -- trees, cluster-fill groups, and
        imported in-game object groups alike, no longer punting cluster
        fills to the Splines tab's Clear Cluster Fills. Objects-tab
        counterpart of the Splines tab's Mask All button.
        """
        wd = self._require_working_dir()
        if not wd:
            return
        visible_tree_iids = [iid for iid in self.objects_tree.get_children() if iid.isdigit()]
        visible_group_iids = [iid for iid in self.objects_tree.get_children() if iid.startswith("c")]
        visible_ingame_iids = [iid for iid in self.objects_tree.get_children() if iid.startswith("g")]
        if not visible_tree_iids and not visible_group_iids and not visible_ingame_iids:
            messagebox.showinfo(
                "Nothing to delete", "No objects are currently shown for the active filter.",
            )
            return
        if not messagebox.askyesno(
            "Delete filtered objects",
            f"Permanently delete {len(visible_tree_iids)} tree(s), {len(visible_group_iids)} "
            f"cluster-fill group(s), and {len(visible_ingame_iids)} imported group(s) currently shown "
            f"(filter={self.objects_filter_var.get()!r})? This can't be undone -- re-run Generate Trees "
            "/ Fill with Clusters to get generated ones back (imported ones can't be regenerated).",
        ):
            return

        remove_indices = {int(iid) for iid in visible_tree_iids}
        if remove_indices:
            self._objects_tree_list = [
                tree for i, tree in enumerate(self._objects_tree_list) if i not in remove_indices
            ]
            save_object_list(self._objects_tree_list, Path(wd) / OBJECT_LIST_FILE)

        group_rows = []
        for iid in visible_group_iids:
            j = int(iid[1:])
            if 0 <= j < len(self._cluster_fill_rows):
                group_rows.append(self._cluster_fill_rows[j])
        if group_rows:
            self._remove_cluster_fill_groups(Path(wd), group_rows)

        ingame_group_names = {
            self._ingame_object_groups[int(iid[1:])][0]
            for iid in visible_ingame_iids
            if 0 <= int(iid[1:]) < len(self._ingame_object_groups)
        }
        if ingame_group_names:
            self._remove_ingame_object_groups_by_name(Path(wd), ingame_group_names)

        self._append_log(
            f"\n[deleted {len(remove_indices)} tree(s), {len(group_rows)} cluster-fill group(s), "
            f"{len(ingame_group_names)} imported group(s)]\n"
        )
        self._regenerate_packed_objects(Path(wd))
        self._refresh_objects_list()
        self._show_preview()

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
        seen_iids: set[str] = set()
        for f in self._splines_features:
            if f.kind == SYNTHETIC_BORDER_CENTERLINE_KIND:
                continue  # implementation detail of a border fill, not a user-facing row -- see _open_cluster_fill_dialog
            if kind_filter != "All" and f.kind != kind_filter:
                continue
            if f.osm_id is None:
                continue  # nothing stable to select/highlight/toggle by
            iid = str(f.osm_id)
            if iid in seen_iids:
                continue  # a corrupted features.geojson with duplicate osm_ids -- one row per id, Treeview iids must be unique
            seen_iids.add(iid)
            self.splines_tree.insert(
                "", "end", iid=iid,
                values=(f.kind, _spline_tag_detail(f), "✔" if f.mask else "", _spline_object_detail(f)),
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
        points of its own until packed -- so its source spline osm_ids
        feed into the SAME spline-highlight overlay _on_spline_selected
        drives (unioned rather than overwritten so a Splines-tab
        selection isn't clobbered), AND, once objects.json exists (see
        _regenerate_packed_objects), every packed cluster record whose
        own "spline_id" is one of this row's source splines and whose
        category matches gets ring-highlighted too -- a plain field
        lookup against objects.json's cluster records (each carries its
        source spline's osm_id directly -- see course_output/
        object_clusters.py's pack_cluster_records), not the point-in-
        polygon geometry cross-reference an earlier version needed back
        when the only available packed-position source (placedObjects2.
        json) had already lost that back-reference by merging every
        spline's clusters into shared per-asset groups.
        """
        selection = self.objects_tree.selection()
        points: list[tuple[float, float]] = []
        spline_ids: set[int] = set()
        group_specs: list[tuple[set[int], int]] = []  # (this row's spline osm_ids, asset category)
        for iid in selection:
            if iid.isdigit():
                idx = int(iid)
                if 0 <= idx < len(self._objects_tree_list):
                    x, z, _tags = self._objects_tree_list[idx]
                    points.append((x, z))
            elif re.fullmatch(r"g\d+i\d+", iid):
                idx = int(iid.split("i", 1)[1])
                if 0 <= idx < len(self._ingame_objects_records):
                    record = self._ingame_objects_records[idx]
                    points.append((float(record.get("x", 0.0)), float(record.get("z", 0.0))))
            elif iid.startswith("c"):
                try:
                    j = int(iid[1:])
                except ValueError:
                    continue
                if 0 <= j < len(self._cluster_fill_rows):
                    ids_str, _label, _ratio, _density, _source, category, _type, _mode = self._cluster_fill_rows[j]
                    row_spline_ids = {int(s) for s in ids_str.split(",") if s}
                    spline_ids.update(row_spline_ids)
                    group_specs.append((row_spline_ids, category))

        wd = self.working_dir.get().strip()
        objects_path = Path(wd) / OBJECTS_FILE if wd else None
        if group_specs and objects_path is not None and objects_path.exists():
            try:
                _, cluster_records, _, _, _ = load_objects(objects_path)
            except (json.JSONDecodeError, OSError, KeyError):
                cluster_records = []
            for row_spline_ids, category in group_specs:
                for record in cluster_records:
                    if record.get("category") == category and record.get("spline_id") in row_spline_ids:
                        points.append((record["x"], record["z"]))

        self._highlighted_object_points = points
        self._highlighted_object_group_spline_ids = spline_ids
        if self._selection_preview_job is not None:
            self.root.after_cancel(self._selection_preview_job)
        self._selection_preview_job = self.root.after(150, self._show_preview_after_selection)

    def _on_object_row_double_click(self, event) -> None:
        """
        Double-click an individual imported in-game object's child row
        (iid f"g{{j}}i{{idx}}", see _refresh_objects_list) to open
        _open_ingame_object_edit_dialog for it. Every other row kind in
        this tree (a tree, a cluster-fill group, or an imported-group
        parent row) has no single position/rotation/scale to edit, so
        double-clicking one is a no-op here -- for the group parent row
        that just leaves Tk's own default double-click (open/close
        toggle) as the only effect, same as any other Treeview.
        """
        iid = self.objects_tree.identify_row(event.y)
        if not iid:
            return
        m = re.fullmatch(r"g\d+i(\d+)", iid)
        if not m:
            return
        idx = int(m.group(1))
        if not 0 <= idx < len(self._ingame_objects_records):
            return
        wd = self.working_dir.get().strip()
        if not wd:
            return
        self._open_ingame_object_edit_dialog(Path(wd), idx)

    def _open_ingame_object_edit_dialog(self, working_dir: Path, record_index: int) -> None:
        """
        Edit one imported in-game object's position/rotation/scale in
        place. `record_index` indexes self._ingame_objects_records
        (the live in-memory copy of ingame_objects.json -- see
        _refresh_objects_list); Save writes the whole list straight
        back to disk, same "mutate in place, persist the whole file"
        approach _open_spline_fill_pct_dialog uses for features.geojson.

        `y` is edited as an optional field (blank = ground-snap, i.e.
        the record's saved-verbatim `y` is None) rather than always a
        number, since that's the actual tri-state the record supports
        (see course_output/ingame_objects.py's module docstring) --
        leaving it blank on an already-ground-snapped object keeps it
        ground-snapped instead of accidentally pinning it at whatever
        elevation the terrain happened to resolve to right now.
        """
        record = self._ingame_objects_records[record_index]

        dialog = tk.Toplevel(self.root)
        dialog.title("Edit imported object")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog, text=self._ingame_object_detail(record), wraplength=320, justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 6))

        form = ttk.Frame(dialog)
        form.pack(fill="x", padx=8, pady=2)

        x_var = tk.StringVar(value=f"{float(record.get('x', 0.0)):g}")
        z_var = tk.StringVar(value=f"{float(record.get('z', 0.0)):g}")
        y_raw = record.get("y")
        y_var = tk.StringVar(value="" if y_raw is None else f"{float(y_raw):g}")
        rotation_var = tk.StringVar(value=f"{float(record.get('rotation_deg', 0.0) or 0.0):g}")
        scale_var = tk.StringVar(value=f"{float(record.get('scale', 1.0) or 1.0):g}")

        for row, (label, var, hint) in enumerate([
            ("X (m)", x_var, ""),
            ("Z (m)", z_var, ""),
            ("Y (m)", y_var, "blank = ground-snap"),
            ("Rotation (deg)", rotation_var, ""),
            ("Scale", scale_var, ""),
        ]):
            ttk.Label(form, text=label, width=16).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(form, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=2)
            if hint:
                ttk.Label(form, text=hint, foreground="gray").grid(row=row, column=2, sticky="w", padx=(6, 0))

        def do_save() -> None:
            try:
                x = float(x_var.get())
                z = float(z_var.get())
                rotation = float(rotation_var.get())
                scale = float(scale_var.get())
                y_text = y_var.get().strip()
                y = None if not y_text else float(y_text)
            except ValueError:
                messagebox.showwarning(
                    "Invalid value", "X, Z, Rotation, and Scale must be numbers "
                    "(Y may be left blank for ground-snap).", parent=dialog,
                )
                return
            record["x"] = x
            record["z"] = z
            record["y"] = y
            record["rotation_deg"] = rotation
            record["scale"] = scale
            save_ingame_objects(self._ingame_objects_records, working_dir / INGAME_OBJECTS_FILE)
            dialog.destroy()
            self._regenerate_packed_objects(working_dir)
            self._refresh_objects_list()
            self._show_preview()
            self._append_log(
                f"\n[edited imported object {record.get('group')!r}: "
                f"x={x:g} z={z:g} y={'ground-snap' if y is None else f'{y:g}'} "
                f"rot={rotation:g} scale={scale:g}]\n"
            )

        btn_row = ttk.Frame(dialog)
        btn_row.pack(fill="x", padx=8, pady=(8, 8))
        ttk.Button(btn_row, text="Save", command=do_save).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right", padx=(0, 4))

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

    def _regenerate_packed_objects(self, working_dir: Path) -> None:
        """
        Rebuild objects.json in-process, right away, from the currently
        in-memory feature list (self._splines_features) plus whatever
        object_list.json currently holds -- same "regenerate immediately
        rather than wait for the next explicit pipeline step" idea
        _regenerate_height_mask already uses for the mask, so a Splines/
        Objects-tab Fill/Clear/Delete (or a fresh Generate Trees run)
        shows up in the "Show objects" preview and the Objects-tab
        cluster-fill selection highlight (_on_object_selected) right
        away -- no `course/` extraction or a full Write Objects run
        needed (see _get_cached_object_preview_layer's docstring).

        trees come from object_list.json if it exists yet (an empty
        list otherwise -- packing cluster fills doesn't actually need
        trees to have been generated first); cluster records come from
        packing self._splines_features, course-cropped, through
        object_clusters.pack_cluster_records -- the exact same packer
        PGA2k_gen.py's step_pack_objects uses, just invoked directly
        here instead of as a subprocess step (mtime-cached -- see below
        -- so repeated calls that don't touch fill splines skip the
        actual repacking).
        """
        trees = []
        object_list_path = working_dir / OBJECT_LIST_FILE
        if object_list_path.exists():
            try:
                trees = load_object_list(object_list_path)
            except (json.JSONDecodeError, OSError, KeyError):
                trees = []

        self._ensure_splines_features_fresh(working_dir)
        # "auto"/"spline" fill specs resolve against game_version at pack
        # time (object_clusters.resolve_fill_mode), so re-pack on a version
        # switch too (see _on_game_version_changed).
        gv = self.game_version.get()

        # pack_cluster_records/pack_spline_records are the RNG/shapely
        # dart-throw circle-packing work -- not cheap on a course with many
        # tagged fill splines, and this method is called unconditionally
        # before every Write Objects run (see _run_write_objects) even when
        # the edit that prompted it (a tree tweak, an ingame-object edit, a
        # theme change, ...) never touched a cluster/spline fill at all.
        # Both packers are pure functions of (course_features, game_version)
        # -- same "compile once" reasoning as the pack-objects/write-objects
        # CLI split itself (see step_pack_objects's docstring) -- so this
        # skips the repack whenever nothing that could change their result
        # has changed, same mtime-keyed cache idiom as
        # _get_cached_water_preview_rects/_get_cached_object_preview_layer.
        project_path = working_dir / PROJECT_FILE
        pack_cache_key = (
            str(working_dir), self._splines_features_mtime,
            project_path.stat().st_mtime if project_path.exists() else None, gv,
        )
        cached = getattr(self, "_cluster_pack_cache", None)
        if getattr(self, "_cluster_pack_cache_key", None) == pack_cache_key and cached is not None:
            cluster_records, object_spline_fill_records = cached
        else:
            course_features = self._shift_and_crop_to_course(working_dir, self._splines_features)
            cluster_records = pack_cluster_records(course_features, game_version=gv)
            object_spline_fill_records = pack_spline_records(course_features, game_version=gv)
            self._cluster_pack_cache_key = pack_cache_key
            self._cluster_pack_cache = (cluster_records, object_spline_fill_records)

        # Preserve resolved collection placements -- they're owned by
        # collections.json (written by the generate-collections step),
        # not by this in-process cluster re-pack, so fold them straight
        # back in rather than dropping them until the next CLI run.
        collection_objects = []
        collections_path = working_dir / COLLECTIONS_FILE
        if collections_path.exists():
            try:
                collection_objects = list(iter_collection_objects(load_collection_records(collections_path)))
            except (json.JSONDecodeError, OSError, KeyError):
                collection_objects = []

        # Parked cars (parking.json, owned by the generate-parking step) --
        # same "preserve, don't drop until the next CLI run" idea; folded in
        # as collection_object-shaped records (see course_output/parking.py).
        parking_path = working_dir / PARKING_FILE
        if parking_path.exists():
            try:
                collection_objects += list(iter_parking_cars(load_parking_records(parking_path)))
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        # Range-net objects (range_nets.json, owned by the
        # generate-range-nets step) -- same "preserve, don't drop until the
        # next CLI run" idea; folded in as collection_object-shaped records
        # (see course_output/range_nets.py).
        range_nets_path = working_dir / RANGE_NETS_FILE
        if range_nets_path.exists():
            try:
                collection_objects += list(iter_range_net_objects(load_range_net_records(range_nets_path)))
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        # Same "preserve, don't drop until the next CLI run" idea as
        # collection_objects above -- ingame_objects.json is owned by
        # the import-ingame-edits step, not this in-process re-pack.
        ingame_object_records = []
        ingame_objects_path = working_dir / INGAME_OBJECTS_FILE
        if ingame_objects_path.exists():
            try:
                ingame_object_records = load_ingame_objects(ingame_objects_path)
            except (json.JSONDecodeError, OSError, KeyError):
                ingame_object_records = []

        save_objects(
            trees, cluster_records, working_dir / OBJECTS_FILE, collection_objects,
            object_spline_fill_records, ingame_object_records,
        )

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

    def _finalize_empty_fills(self, feature) -> set:
        """
        Called once `feature`'s PGA_CLUSTER_FILLS_TAG list has become
        empty (its last asset was zeroed out). For a generated spline
        (_SYNTHETIC_CLUSTER_KINDS), which exists only to carry fill specs,
        returns {feature.osm_id, its centerline companion if any} for the
        caller to drop from self._splines_features -- same rationale as
        _clear_selected_cluster_fills. For a real OSM spline, just pops
        the now-empty tag in place (the spline itself is real geometry
        and must survive) and returns an empty set.
        """
        if feature.kind in self._SYNTHETIC_CLUSTER_KINDS:
            remove_ids = {feature.osm_id}
            centerline_id = feature.tags.get(PGA_CLUSTER_CENTERLINE_REF_TAG)
            if centerline_id is not None:
                remove_ids.add(centerline_id)
            return remove_ids
        feature.tags.pop(PGA_CLUSTER_FILLS_TAG, None)
        return set()

    def _clear_selected_cluster_fills(self) -> None:
        """
        Remove PGA_CLUSTER_FILLS_TAG from every currently-selected row
        (multi-select). For a real spline (kind not one of
        _SYNTHETIC_CLUSTER_KINDS) that just strips the tag, same as
        always. For a border/masked-manual row, the underlying Feature
        exists ONLY to carry that tag (see _open_cluster_fill_dialog) --
        stripping the tag would leave a permanent, purposeless row, so
        the whole Feature is deleted instead.

        For a pga_collection marker (kind "collection") there's no tag to
        strip -- its objects are the resolved records in collections.json
        keyed by the marker way's osm_id as `source_id`. "Clear" drops
        those records (Write Objects / the live preview then stop
        emitting them); the marker row itself stays, since it's real OSM
        data -- re-run Generate to resolve it again.
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
            if f.kind == "collection":
                continue  # handled below against collections.json
            if f.kind in self._SYNTHETIC_CLUSTER_KINDS:
                remove_ids.add(f.osm_id)
                centerline_id = f.tags.get(PGA_CLUSTER_CENTERLINE_REF_TAG)
                if centerline_id is not None:
                    remove_ids.add(centerline_id)  # its SYNTHETIC_BORDER_CENTERLINE_KIND companion (see _open_cluster_fill_dialog)
                changed = True
            elif f.tags.pop(PGA_CLUSTER_FILLS_TAG, None) is not None:
                changed = True

        collections_changed = False
        collection_ids = {f.osm_id for f in targets if f.kind == "collection"}
        collections_path = Path(wd) / COLLECTIONS_FILE
        if collection_ids and collections_path.exists():
            try:
                records = load_collection_records(collections_path)
            except (json.JSONDecodeError, OSError):
                records = []
            kept = [r for r in records if r.get("source_id") not in collection_ids]
            if len(kept) != len(records):
                save_collection_records(kept, collections_path)
                collections_changed = True

        if not changed and not collections_changed:
            return
        if remove_ids:
            self._splines_features = [f for f in self._splines_features if f.osm_id not in remove_ids]
        if changed:
            save_features(self._splines_features, Path(wd) / FEATURES_FILE)
        self._regenerate_packed_objects(Path(wd))
        self._refresh_splines_list()
        self._refresh_objects_list()
        restorable_ids = [
            str(i) for i in selected_ids if i not in remove_ids and self.splines_tree.exists(str(i))
        ]
        if restorable_ids:
            self.splines_tree.selection_set(restorable_ids)
        self._show_preview()

    def _on_spline_row_double_click(self, event) -> None:
        """
        Double-click a Splines-tab row that carries one or more object-
        spline fills (mode="spline" specs in PGA_CLUSTER_FILLS_TAG) to
        adjust each fill's "Fill % of max" in place -- the same knob the
        Fill dialog sets at creation time, without having to Clear and
        re-Fill. One entry per spline-mode spec on that feature (an asset
        shouldn't appear twice going forward -- re-Filling with an
        already-present asset now overwrites its spec in place -- but
        legacy data or specs from a separate border/masked Feature can
        still show the same asset more than once). Setting an entry to 0
        deletes that asset's fill; zeroing the last one deletes the whole
        spline too, but only for a generated (pga_cluster_border/
        pga_cluster_masked) spline -- an OSM spline just loses the tag.

        A row with nothing spline-mode to edit (untagged, stamps-mode
        fills only, or a pga_collection marker) is a no-op.
        """
        wd = self.working_dir.get().strip()
        iid = self.splines_tree.identify_row(event.y)
        if not wd or not iid:
            return
        try:
            osm_id = int(iid)
        except ValueError:
            return
        feature = next((f for f in self._splines_features if f.osm_id == osm_id), None)
        if feature is None:
            return
        spline_specs = [
            s for s in (feature.tags.get(PGA_CLUSTER_FILLS_TAG) or [])
            if s.get("mode", CLUSTER_FILL_MODE_STAMPS)
            in (CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_AUTO)
        ]
        if not spline_specs:
            return
        self._open_spline_fill_pct_dialog(Path(wd), feature, spline_specs)

    def _open_spline_fill_pct_dialog(self, working_dir: Path, feature, spline_specs: list) -> None:
        """
        Popup from _on_spline_row_double_click: one Fill %-of-max entry
        per mode="spline" spec on `feature`, seeded from the spec's
        current `density` (see course_output/object_clusters.py's
        pack_spline_records -- fillPct = category ceiling x density/100).
        Save writes the new percents straight back into the same spec
        dicts, persists features.geojson, and re-packs objects.json so
        the preview updates without a Write Objects run.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Object-spline fill %")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text=f"Spline {feature.osm_id} ({feature.kind}) -- percent (0 <= % <= 100, linear) of "
                 "the asset category's max renderable fillPct. Trees & bushes cap at fillPct 0.03, "
                 "grass / ground-cover are 1:1. Setting an entry to 0 deletes that asset's fill; if "
                 "it's the last one left and this is a generated (pga_cluster_border/pga_cluster_"
                 "masked) spline, the spline itself is deleted too -- an OSM spline just loses the tag.",
            wraplength=380, justify="left",
        ).pack(anchor="w", padx=8, pady=(8, 6))

        rows = []
        for spec in spline_specs:
            row = ttk.Frame(dialog)
            row.pack(fill="x", padx=8, pady=2)
            label = _ASSET_LABEL_BY_KEY.get((spec.get("category"), spec.get("type")), "?")
            ttk.Label(row, text=label, width=24).pack(side="left")
            var = tk.StringVar(
                value=f"{float(spec.get('density', DEFAULT_SPLINE_FILL_DENSITY)):g}"
            )
            ttk.Entry(row, textvariable=var, width=8).pack(side="left", padx=4)
            ttk.Label(row, text="%").pack(side="left")
            ceiling = SPLINE_FILL_MAX_PCT_BY_CATEGORY.get(spec.get("category"), DEFAULT_SPLINE_FILL_MAX_PCT)
            hint = ttk.Label(row, text="")
            hint.pack(side="left", padx=(8, 0))

            def _update_hint(*_a, v=var, h=hint, c=ceiling):
                try:
                    d = float(v.get())
                except ValueError:
                    h.configure(text="→ fillPct ?")
                    return
                h.configure(text=f"→ fillPct {c * d / 100:g}")

            var.trace_add("write", _update_hint)
            _update_hint()
            rows.append((spec, var))

        def do_save() -> None:
            updated = []
            zeroed = []
            for spec, var in rows:
                try:
                    d = float(var.get())
                    if not 0.0 <= d <= 100.0:
                        raise ValueError
                except ValueError:
                    messagebox.showwarning(
                        "Invalid fill %",
                        "Every Fill % must be a number from 0 to 100 (0 deletes that asset's fill).",
                        parent=dialog,
                    )
                    return
                if d == 0.0:
                    zeroed.append(spec)
                else:
                    updated.append((spec, d))
            for spec, d in updated:
                spec["density"] = d
            spline_deleted = False
            if zeroed:
                fills = feature.tags.get(PGA_CLUSTER_FILLS_TAG) or []
                remaining = [s for s in fills if not any(s is z for z in zeroed)]
                if remaining:
                    feature.tags[PGA_CLUSTER_FILLS_TAG] = remaining
                else:
                    remove_ids = self._finalize_empty_fills(feature)
                    if remove_ids:
                        spline_deleted = True
                        self._splines_features = [
                            f for f in self._splines_features if f.osm_id not in remove_ids
                        ]
            save_features(self._splines_features, working_dir / FEATURES_FILE)
            dialog.destroy()
            self._regenerate_packed_objects(working_dir)
            self._refresh_splines_list()
            self._refresh_objects_list()
            if self.splines_tree.exists(str(feature.osm_id)):
                self.splines_tree.selection_set(str(feature.osm_id))
            self._show_preview()
            log_parts = [
                f"{_ASSET_LABEL_BY_KEY.get((s.get('category'), s.get('type')), '?')}={d:g}%"
                for s, d in updated
            ]
            log_parts += [
                f"{_ASSET_LABEL_BY_KEY.get((s.get('category'), s.get('type')), '?')}=deleted"
                for s in zeroed
            ]
            suffix = " [spline deleted]" if spline_deleted else ""
            self._append_log(
                f"\n[updated object-spline fill % on spline {feature.osm_id}: "
                + ", ".join(log_parts) + suffix + "]\n"
            )

        btn_row = ttk.Frame(dialog)
        btn_row.pack(fill="x", padx=8, pady=(8, 8))
        ttk.Button(btn_row, text="Save", command=do_save).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right", padx=(0, 4))

    def _open_cluster_fill_dialog(self) -> None:
        """
        "Fill with Clusters...": pick one or more nature assets (see
        course_output/asset_catalog.py), a raster ratio, and a fill
        density (%), then create a
        {"category","type","ratio","density","source"} spec per chosen
        asset (see course_output/object_clusters.py for what actually
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
            tree_frame, columns=("category", "asset", "description"), show="headings", height=14,
            selectmode="extended",
        )
        asset_tree.heading("category", text="Category")
        asset_tree.heading("asset", text="Asset")
        asset_tree.heading("description", text="Description")
        asset_tree.column("category", width=140)
        asset_tree.column("asset", width=220)
        asset_tree.column("description", width=220)
        asset_tree.pack(side="left", fill="both", expand=True)
        asset_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=asset_tree.yview)
        asset_scroll.pack(side="left", fill="y")
        asset_tree["yscrollcommand"] = asset_scroll.set

        def refresh_asset_tree() -> None:
            asset_tree.delete(*asset_tree.get_children())
            chosen = category_var.get()
            for i, entry in enumerate(CLUSTERABLE_ENTRIES):
                cat_desc = ASSET_CATEGORIES[entry.category].description
                if chosen != "All" and cat_desc != chosen:
                    continue
                asset_tree.insert("", "end", iid=str(i), values=(cat_desc, entry.label, entry.description or ""))

        category_box.bind("<<ComboboxSelected>>", lambda e: refresh_asset_tree())
        refresh_asset_tree()

        # Auto (default) -> object-spline fill on v2021+ for a vegetation
        # category, circle-scatter otherwise / on v2019 (see
        # object_clusters.resolve_fill_mode). Stamps / Spline force one.
        mode_row = ttk.Frame(dialog)
        mode_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(mode_row, text="Fill mode:").pack(side="left")
        default_mode = CLUSTER_FILL_MODE_AUTO
        mode_var = tk.StringVar(value=default_mode)
        auto_radio = ttk.Radiobutton(
            mode_row, text="Auto", value=CLUSTER_FILL_MODE_AUTO, variable=mode_var,
        )
        auto_radio.pack(side="left", padx=(4, 0))
        stamps_radio = ttk.Radiobutton(
            mode_row, text="Stamps", value=CLUSTER_FILL_MODE_STAMPS, variable=mode_var,
        )
        stamps_radio.pack(side="left", padx=(4, 0))
        spline_radio = ttk.Radiobutton(
            mode_row, text="Spline", value=CLUSTER_FILL_MODE_SPLINE, variable=mode_var,
        )
        spline_radio.pack(side="left")
        _Tooltip(auto_radio, "Recommended: an object-spline fill on v2021+ for vegetation "
                 "(trees/bushes, grass, ground-cover, detail plants), circle-scatter for rocks/"
                 "hardscape and on v2019. Resolved at pack time against the project's game version.")
        _Tooltip(stamps_radio, "Force client-side circle-packed scatter stamps (Value.clusters) -- works "
                 "for every game version, but density/placement is this tool's own approximation.")
        _Tooltip(spline_radio, "Force an object-spline fill region (Value.splines) -- the game engine "
                 "itself auto-scatters the asset at the given density, no packing here. Falls back to "
                 "circle-scatter if this project's game_version is 2019.")

        ratio_row = ttk.Frame(dialog)
        ratio_row.pack(fill="x", padx=8, pady=(0, 4))
        ratio_label = ttk.Label(ratio_row, text="Raster ratio:")
        ratio_label.pack(side="left")
        ratio_var = tk.StringVar(value=str(DEFAULT_RASTER_RATIO))
        ratio_entry = ttk.Entry(ratio_row, textvariable=ratio_var, width=6)
        ratio_entry.pack(side="left", padx=4)
        _Tooltip(ratio_entry, "Stamps mode only: how much placed stamps are allowed to overlap each "
                 "other during packing -- minimum center-to-center separation is (r1+r2) x ratio. 1.0 "
                 "means stamps may only just touch; <1 lets them overlap more (denser fill); >1 spaces "
                 "them further apart. Applies to every asset picked in this dialog, across all 3 "
                 "packing passes. Ignored in spline mode.")

        # Stamps mode: a percent-of-catalog-density knob (default 100).
        # Spline mode: "% of the max fill this asset CATEGORY renders at",
        # linear (default 50). The game's density slider maps linearly to
        # fillPct, but trees & bushes cap at fillPct 0.03 however hard you
        # push -- pack_spline_records applies SPLINE_FILL_MAX_PCT_BY_
        # CATEGORY, so 100% here means "as dense as that category goes".
        _SPLINE_FILL_PCT_DEFAULT = str(DEFAULT_SPLINE_FILL_DENSITY)  # "50.0"
        _STAMPS_DENSITY_DEFAULT = str(DEFAULT_FILL_DENSITY)  # "100.0"
        density_label = ttk.Label(ratio_row, text="Fill density (%):")
        density_label.pack(side="left", padx=(12, 0))
        density_var = tk.StringVar(value=(
            _STAMPS_DENSITY_DEFAULT if default_mode == CLUSTER_FILL_MODE_STAMPS
            else _SPLINE_FILL_PCT_DEFAULT  # auto + spline both read density as "% of max"
        ))
        density_entry = ttk.Entry(ratio_row, textvariable=density_var, width=6)
        density_entry.pack(side="left", padx=4)
        _Tooltip(density_entry, "Stamps mode: how many instances render inside each placed stamp circle, "
                 "as a percent of the asset's own measured planting density -- 100 is the catalog's real "
                 "density, 50 is half as many instances at the same stamp size/positions, 200 is double. "
                 "Doesn't change stamp size, count, or placement (see Raster ratio for that). Spline "
                 "mode: percent (0-100) of the maximum fill this asset's CATEGORY renders at -- linear. "
                 "Trees & bushes top out at fillPct 0.03, so 100% on a tree spline = 0.03; grass / "
                 "ground-cover are 1:1, so 100% = fillPct 1.0.")

        def _update_mode_widgets(*_args) -> None:
            mode = mode_var.get()
            pct_of_max = mode in (CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_AUTO)
            # Ratio only bites for a real stamp pack; Auto might still fall
            # back to scatter, so keep it editable there.
            ratio_entry.configure(state="disabled" if mode == CLUSTER_FILL_MODE_SPLINE else "normal")
            ratio_label.configure(
                text="Raster ratio (stamps only):" if pct_of_max else "Raster ratio:")
            density_label.configure(text="Fill % of max:" if pct_of_max else "Fill density (%):")
            # Re-seed to the new mode's default, but only when the field
            # still holds the OTHER mode's default (user hasn't typed
            # their own): the two families want different numbers.
            cur = density_var.get().strip()
            if pct_of_max and cur in ("", _STAMPS_DENSITY_DEFAULT):
                density_var.set(_SPLINE_FILL_PCT_DEFAULT)
            elif not pct_of_max and cur in ("", _SPLINE_FILL_PCT_DEFAULT):
                density_var.set(_STAMPS_DENSITY_DEFAULT)

        mode_var.trace_add("write", _update_mode_widgets)
        _update_mode_widgets()

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
            mode = mode_var.get()
            ratio = DEFAULT_RASTER_RATIO
            if mode == CLUSTER_FILL_MODE_STAMPS:
                try:
                    ratio = float(ratio_var.get())
                    if ratio <= 0:
                        raise ValueError
                except ValueError:
                    messagebox.showwarning(
                        "Invalid ratio", "Raster ratio must be a positive number.", parent=dialog,
                    )
                    return
            # 0 is only meaningful against an already-tagged, existing
            # spline (it deletes that asset's fill there -- see below) --
            # Border/"Use mask" always build a brand-new, empty-tagged
            # synthetic Feature, so there's nothing existing to delete.
            allow_zero = not border_mode and not use_mask_var.get()
            # Both modes store `density` as a percent. Spline (and Auto,
            # which becomes spline on v2021+) read it as "% of the asset
            # category's max fillPct" (0-100, see pack_spline_records);
            # stamps mode as "% of catalog planting density" (can exceed 100).
            if mode in (CLUSTER_FILL_MODE_SPLINE, CLUSTER_FILL_MODE_AUTO):
                try:
                    density = float(density_var.get())
                    if density < 0.0 or density > 100.0 or (density == 0.0 and not allow_zero):
                        raise ValueError
                except ValueError:
                    messagebox.showwarning(
                        "Invalid fill %",
                        "Fill % of max must be a number from 0 to 100."
                        if allow_zero else
                        "Fill % of max must be a number greater than 0 and at most 100 (0 isn't "
                        "allowed here -- there's no existing fill on a new border/masked spline to "
                        "delete).",
                        parent=dialog,
                    )
                    return
            else:
                try:
                    density = float(density_var.get())
                    if density < 0.0 or (density == 0.0 and not allow_zero):
                        raise ValueError
                except ValueError:
                    messagebox.showwarning(
                        "Invalid density",
                        "Fill density (%) must be a positive number."
                        if not allow_zero else
                        "Fill density (%) must be a non-negative number (0 deletes that asset's fill).",
                        parent=dialog,
                    )
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
                    {
                        "category": e.category, "type": e.type, "ratio": ratio, "density": density,
                        "source": CLUSTER_FILL_SOURCE_BORDER, "mode": mode,
                    }
                    for e in chosen_entries
                ]
                # Companion Feature carrying the ORIGINAL boundary line (pre-buffer)
                # as its own .geometry -- object_clusters.py's new ring-walk packer
                # needs that path, which the buffered `ring` polygon itself doesn't
                # retain. A separate Feature (not a tag on border_feature) so it
                # rides the normal shift_features/_crop_features_to_course pipeline
                # like every other Feature's geometry, rather than needing a
                # shift-aware special case for a geometry value buried in tags.
                centerline_feature = Feature(
                    geometry=mask_geom.boundary, kind=SYNTHETIC_BORDER_CENTERLINE_KIND, tags={},
                    osm_id=self._next_synthetic_osm_id(), mask=True,
                )
                # Appended before the next _next_synthetic_osm_id() call so that
                # call sees centerline_feature's own id in `existing` and can't
                # hand border_feature the same one back.
                self._splines_features.append(centerline_feature)
                border_feature = Feature(
                    geometry=ring, kind=SYNTHETIC_BORDER_KIND,
                    tags={PGA_CLUSTER_FILLS_TAG: specs, PGA_CLUSTER_CENTERLINE_REF_TAG: centerline_feature.osm_id},
                    osm_id=self._next_synthetic_osm_id(), mask=True,
                )
                self._splines_features.append(border_feature)
                save_features(self._splines_features, Path(wd) / FEATURES_FILE)
                self._append_log(
                    f"\n[created border ring (source={self.mask_source_var.get()}, width={border_width}) "
                    f"filled with {len(chosen_entries)} asset(s) at ratio={ratio} density={density}%]\n"
                )
            else:
                specs = [
                    {
                        "category": e.category, "type": e.type, "ratio": ratio, "density": density,
                        "source": CLUSTER_FILL_SOURCE_MANUAL, "mode": mode,
                    }
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
                        f"{len(chosen_entries)} asset(s) at ratio={ratio} density={density}%]\n"
                    )
                else:
                    # Re-Filling an asset already present on the spline
                    # overwrites its spec in place rather than duplicating
                    # it (identity is category+type only, mode-agnostic).
                    # density==0 deletes that asset's spec instead; if
                    # that empties the spline's fills entirely, a
                    # generated (border/masked) spline is deleted outright
                    # -- an OSM spline just loses the tag. See
                    # _finalize_empty_fills.
                    added = overwritten = removed = 0
                    deleted_feature_ids: set = set()
                    for f in area_targets:
                        existing = f.tags.get(PGA_CLUSTER_FILLS_TAG) or []
                        had_tag = PGA_CLUSTER_FILLS_TAG in f.tags
                        for spec in specs:
                            match = next(
                                (s for s in existing
                                 if s.get("category") == spec["category"] and s.get("type") == spec["type"]),
                                None,
                            )
                            if density == 0.0:
                                if match is not None:
                                    existing[:] = [s for s in existing if s is not match]
                                    removed += 1
                            elif match is not None:
                                match.update(spec)
                                overwritten += 1
                            else:
                                existing.append(dict(spec))
                                added += 1
                        if existing:
                            f.tags[PGA_CLUSTER_FILLS_TAG] = existing
                        elif had_tag:
                            deleted_feature_ids |= self._finalize_empty_fills(f)
                    if deleted_feature_ids:
                        self._splines_features = [
                            f for f in self._splines_features if f.osm_id not in deleted_feature_ids
                        ]
                    save_features(self._splines_features, Path(wd) / FEATURES_FILE)
                    skipped = len(targets) - len(area_targets)
                    note = f" ({skipped} non-area spline(s) skipped)" if skipped else ""
                    self._append_log(
                        f"\n[fill on {len(area_targets)} spline(s): {added} added, {overwritten} "
                        f"overwritten, {removed} deleted (ratio={ratio} density={density}%)"
                        f"{f', {len(deleted_feature_ids)} spline(s) deleted' if deleted_feature_ids else ''}"
                        f"{note}]\n"
                    )

            self._regenerate_packed_objects(Path(wd))
            self._refresh_splines_list()
            self._refresh_objects_list()
            restorable_ids = [str(i) for i in selected_ids if self.splines_tree.exists(str(i))]
            if restorable_ids:
                self.splines_tree.selection_set(restorable_ids)
            self._show_preview()
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

    def _add_slider_entry(self, parent, double_var: tk.DoubleVar, text_var: tk.StringVar,
                          lo: float = 0.0, hi: float = 400.0) -> ttk.Entry:
        """A small editable text box that mirrors a slider's DoubleVar both ways:
        the slider keeps text_var in sync (see _show_preview), and committing the
        entry (Return / focus-out) parses + clamps the value back into double_var
        and refreshes the preview."""
        entry = ttk.Entry(parent, textvariable=text_var, width=5)
        entry.pack(side="left")

        def _commit(_evt=None):
            try:
                val = float(text_var.get())
            except (TypeError, ValueError):
                text_var.set(f"{double_var.get():.0f}")
                return
            val = max(lo, min(hi, val))
            double_var.set(val)
            text_var.set(f"{val:.0f}")
            self._show_preview()

        entry.bind("<Return>", _commit)
        entry.bind("<FocusOut>", _commit)
        return entry

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
        # Restore the last-used LIDAR tree minimum height for this working
        # directory (see the slider's comment in the Objects tab's Generate
        # Trees section), so a GUI run matches the CLI's sticky default.
        saved_min_height = project.get("objects_tree_min_height_m", DEFAULT_LIDAR_TREE_MIN_HEIGHT_M)
        self.tree_min_height_var.set(float(saved_min_height))
        self.tree_min_height_text.set(f"{float(saved_min_height):.1f}")
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
            # Prefer the name-string "theme" field (works for any game
            # version); fall back to the numeric v2019 "objects_theme" id
            # for older projects saved before "theme" existed.
            saved_theme = project.get("theme")
            if saved_theme:
                theme_name = next(
                    (name for name in self._theme_name_to_id if name.lower() == saved_theme.lower()),
                    "(not set)",
                )
            else:
                theme_name = THEMES_V2019.get(project.get("objects_theme"), "(not set)")
            self.objects_theme_var.set(theme_name)
        finally:
            self._suppress_objects_theme_save = False
        self.objects_tree_asset_paths = list(project.get("objects_tree_asset_paths") or [])
        self._refresh_tree_assets_summary()
        self._sync_tree_assets_enabled()
        saved_libdir = project.get("collections_library_dir")
        self.collections_library_var.set(saved_libdir or str(default_library_dir()))
        self._refresh_collection_templates()
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
        self._sync_tree_assets_enabled()
        if self._suppress_game_version_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        save_project(Path(wd), {"game_version": self.game_version.get()})
        # mode="auto" / mode="spline" cluster fills resolve differently per
        # version (object_clusters.resolve_fill_mode), so objects.json + the
        # "Show objects" preview need a re-pack now, not just at the next
        # Fill/Clear.
        try:
            self._regenerate_packed_objects(Path(wd))
        except Exception as exc:  # preview refresh must never break a version switch
            self._append_log(f"[warn] objects re-pack after version change failed: {exc}")

    def _on_objects_theme_changed(self) -> None:
        """
        objects_theme_var holds a theme NAME (e.g. "rustic"); this saves
        it two ways: "objects_theme" as the numeric v2019 id (existing
        use -- write-objects' --theme, v2019 asset resolution only), and
        "theme" as the lowercase name itself (new -- course_templates.py's
        resolve_course_template needs the name, not the id, and v2021+
        has no numeric id at all).
        """
        if self._suppress_objects_theme_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        theme_name = self.objects_theme_var.get()
        theme_id = self._theme_name_to_id.get(theme_name)
        save_project(Path(wd), {
            "objects_theme": theme_id,
            "theme": theme_name.lower() if theme_id is not None else None,
        })

    # ------------------------------------------------------------------
    # Folder / file pickers
    # ------------------------------------------------------------------

    def _browse_working_dir(self) -> None:
        d = filedialog.askdirectory(title="Select working directory")
        if d:
            self.working_dir.set(d)

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
            if not self.preserve_synthetic_var.get():
                args += ["--no-preserve-synthetic"]
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
        """
        Explicit "reset course/ from the bundled template" action -- see
        the button's tooltip. game_version/theme are already persisted
        in project.json (set immediately on change), so this step just
        re-reads them; no args needed here.
        """
        wd = self._require_working_dir()
        if not wd:
            return
        self._run_step(["--step", "ingest-course"], wd)

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
            args.append(
                "--generate-terrain-remove-covered-stamps"
                if self.generate_terrain_remove_covered_stamps_var.get()
                else "--no-generate-terrain-remove-covered-stamps"
            )
            if self.generate_terrain_remove_covered_stamps_var.get():
                margin = self.generate_terrain_remove_covered_margin_var.get().strip()
                if margin:
                    args += ["--generate-terrain-remove-covered-margin-m", margin]
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
        args.append(
            "--remove-covered-stamps" if self.refine_remove_covered_stamps_var.get()
            else "--no-remove-covered-stamps"
        )
        if self.refine_remove_covered_stamps_var.get():
            refine_margin = self.refine_remove_covered_margin_var.get().strip()
            if refine_margin:
                args += ["--remove-covered-margin-m", refine_margin]
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
        if step == "write-water":
            if self.multi_tile_water_var.get():
                args.append("--multi-tile-water")
            args += ["--water-fill-mode", self.water_fill_mode_var.get()]
            for flag, var in (
                ("--water-tile-tolerance-m", self.water_tile_tolerance_var),
                ("--water-tile-min-edge-m", self.water_tile_min_edge_var),
                ("--water-tile-max-search-m", self.water_tile_max_search_var),
                ("--water-tile-width-samples", self.water_tile_width_samples_var),
                ("--water-tile-redundancy-ratio", self.water_tile_redundancy_var),
                ("--water-tile-overlap-m", self.water_tile_overlap_var),
                ("--water-stripe-overlap-m", self.water_stripe_overlap_var),
                ("--water-stripe-min-edge-m", self.water_stripe_min_edge_var),
                ("--water-stripe-tolerance-m", self.water_stripe_tolerance_var),
                ("--water-stripe-max-stripes-per-side", self.water_stripe_max_stripes_var),
                ("--water-stripe-buffer-m", self.water_stripe_buffer_var),
            ):
                value = var.get().strip()
                if value:  # blank field (mid-edit) -- let the CLI fall back to its own default
                    args += [flag, value]
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

    def _browse_import_ingame_course(self) -> None:
        course_file = filedialog.askopenfilename(
            title="Select a saved, hand-edited .course file to import",
            filetypes=[(".course files", "*.course"), ("All files", "*.*")],
            **self._courses_dir_dialog_kwarg(),
        )
        if course_file:
            self.import_ingame_course_var.set(course_file)

    def _import_ingame_args(self) -> Optional[list[str]]:
        course_file = self.import_ingame_course_var.get().strip()
        if not course_file:
            messagebox.showwarning("No course file", "Browse to a saved, hand-edited .course file first.")
            return None
        if not Path(course_file).is_file():
            messagebox.showerror("File not found", f"{course_file} doesn't exist.")
            return None
        args = ["--step", "import-ingame-edits", "--edited-course", course_file]
        if self.registration_marks_var.get():
            args.append("--registration-marks")
        args.append("--direct-height-shift" if self.direct_height_shift_var.get()
                     else "--no-direct-height-shift")
        return args

    def _run_import_ingame_preview(self) -> None:
        """Objects tab / File tab -- dry-run diff only, see
        step_import_ingame_edits. Prints its summary to the log; changes
        nothing on disk."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = self._import_ingame_args()
        if args is None:
            return
        self._run_step(args, wd)

    def _run_import_ingame_commit(self) -> None:
        """Objects tab / File tab -- applies the diff (new objects ->
        ingame_objects.json, new stamps -> the next stamps_N.json
        layer), then re-packs objects.json in-process and refreshes the
        Objects list so the import shows up immediately, same as any
        other Objects-tab mutation."""
        wd = self._require_working_dir()
        if not wd:
            return
        args = self._import_ingame_args()
        if args is None:
            return
        args.append("--commit")
        group = self.import_ingame_group_var.get().strip()
        if group:
            args += ["--import-group", group]
        if not messagebox.askyesno(
            "Commit import",
            "This appends any new in-game objects to ingame_objects.json and writes any new terrain "
            "stamps as a new stamps_N.json layer. Run Preview Import first if you haven't already, to "
            "see counts. Continue?",
        ):
            return
        self._run_step(args, wd, on_done=lambda: self._on_objects_step_done(wd, regenerate_packed=True))

    def _run_visualize(self) -> None:
        wd = self._require_working_dir()
        if wd:
            args = ["--step", "visualize"]
            error_resolution = self.error_resolution_var.get().strip()
            if error_resolution:
                args += ["--error-resolution", error_resolution]
            self._run_step(args, wd)

    # ------------------------------------------------------------------
    # Push Blank to Game -- the "push-blank-template" CLI step builds a
    # fresh named blank .course (from the current Game version + Theme
    # template) into working_dir/blank_template.course; this then copies
    # that straight into the game's Courses folder. Same CLI-builds /
    # GUI-copies split as Repack -> "Copy to Game Folder" below.
    # ------------------------------------------------------------------

    def _game_courses_dir(self, version: Optional[str] = None) -> Optional[Path]:
        """
        The in-game Courses folder for `version` (default: the selected
        Game version), or None if that version has no known folder
        mapping (see GAME_VERSION_FOLDERS). Doesn't check existence --
        callers that just want a dialog's starting point should.
        """
        folder_name = GAME_VERSION_FOLDERS.get(version or self.game_version.get())
        if folder_name is None:
            return None
        return Path.home() / "AppData" / "LocalLow" / "2K" / folder_name / "Courses"

    def _courses_dir_dialog_kwarg(self) -> dict:
        """
        {"initialdir": <selected version's in-game Courses folder>} if
        that folder exists, else {} -- so a .course open dialog starts
        where the game keeps its saves. Empty (dialog's own default)
        when the version is unmapped or the folder isn't there.
        """
        courses_dir = self._game_courses_dir()
        if courses_dir is not None and courses_dir.is_dir():
            return {"initialdir": str(courses_dir)}
        return {}

    def _run_push_blank_to_game(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return

        version = self.game_version.get()
        if version not in GAME_VERSION_FOLDERS:
            messagebox.showerror(
                "Unknown game folder for this version",
                f"No Courses-folder mapping is known yet for game_version={version!r} "
                f"(only {list(GAME_VERSION_FOLDERS)} are wired up). Set Game version to one "
                "of those first.",
            )
            return
        if self.objects_theme_var.get() == "(not set)":
            messagebox.showwarning(
                "No theme selected",
                "Pick a Theme first -- it selects which bundled blank template gets pushed.",
            )
            return

        # Serial name is generated GUI-side so the on-disk filename in the
        # game folder can match the in-game name without parsing step
        # output; the step stamps this same string inside the .course.
        serial = f"LIDAR-{version}-{time.strftime('%Y%m%d%H%M%S')}"
        self._run_step(
            ["--step", "push-blank-template", "--blank-course-name", serial], wd,
            on_done=lambda: self._copy_blank_to_game(wd, version, serial),
        )

    def _copy_blank_to_game(self, wd: Path, version: str, serial: str) -> None:
        if not self._last_step_ok:
            return  # the build step failed/was stopped -- nothing to push
        source = wd / BLANK_TEMPLATE_COURSE_FILE
        if not source.exists():
            self._append_log(f"Push Blank: expected {source}, but it doesn't exist.\n")
            return

        dest_dir = self._game_courses_dir(version)
        dest_path = dest_dir / f"{serial}.course"
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_path)
            self._append_log(f"Pushed blank template -> {dest_path}\n")
            self.status_label.config(text=f"Blank '{serial}' pushed to game", foreground="green")
        except OSError as e:
            messagebox.showerror("Copy failed", str(e))
            self.status_label.config(text="Push blank failed", foreground="red")

    # ------------------------------------------------------------------
    # Push to Game for Editing -- the inverse of "Capture from .course..."
    # The "push-collection" CLI step builds a fresh .course holding one
    # library template's objects/splines/stamps into
    # working_dir/pushed_collection.course; this copies it into the game's
    # Courses folder. Same CLI-builds / GUI-copies split as Push Blank.
    # ------------------------------------------------------------------

    def _run_push_collection_to_game(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return

        name = self._selected_collection_name()
        if not name:
            messagebox.showwarning(
                "No template selected",
                "Select a collection template in the list first.",
            )
            return

        version = self.game_version.get()
        if version not in GAME_VERSION_FOLDERS:
            messagebox.showerror(
                "Unknown game folder for this version",
                f"No Courses-folder mapping is known yet for game_version={version!r} "
                f"(only {list(GAME_VERSION_FOLDERS)} are wired up). Set Game version to one "
                "of those first.",
            )
            return
        if self.objects_theme_var.get() == "(not set)":
            messagebox.showwarning(
                "No theme selected",
                "Pick a Theme first -- it selects which bundled blank template the collection "
                "is built into.",
            )
            return

        serial = f"COLL-{_collection_slug(name)}-{time.strftime('%Y%m%d%H%M%S')}"
        self._run_step(
            ["--step", "push-collection", "--collection-name", name,
             "--collection-library", str(self._collection_library_dir()),
             "--blank-course-name", serial], wd,
            on_done=lambda: self._copy_pushed_collection_to_game(wd, version, serial),
        )

    def _copy_pushed_collection_to_game(self, wd: Path, version: str, serial: str) -> None:
        if not self._last_step_ok:
            return  # the build step failed/was stopped -- nothing to push
        source = wd / PUSH_COLLECTION_COURSE_FILE
        if not source.exists():
            self._append_log(f"Push Collection: expected {source}, but it doesn't exist.\n")
            return

        dest_dir = self._game_courses_dir(version)
        dest_path = dest_dir / f"{serial}.course"
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_path)
            self._append_log(f"Pushed collection -> {dest_path}\n")
            self.status_label.config(text=f"Collection '{serial}' pushed to game", foreground="green")
        except OSError as e:
            messagebox.showerror("Copy failed", str(e))
            self.status_label.config(text="Push collection failed", foreground="red")

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

    def _run_step(
        self, extra_args: list[str], working_dir: Path, on_done: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        on_done, if given, runs once the subprocess actually finishes
        (from _poll_log_queue's "done" handling, same completion signal
        _refresh_preview_and_slider already uses) -- regardless of
        success/failure/stop, same as that refresh, and BEFORE it, so
        anything on_done writes (e.g. regenerating objects.json) is
        already in place by the time the preview redraws. NOT a
        substitute for checking the step's own exit status; just a hook
        for GUI state (e.g. re-reading a file the step just wrote) that
        would be wrong if run immediately after this call returns,
        since _run_subprocess does its actual work on a background
        thread -- this method itself only launches that thread and
        returns right away.
        """
        if self.running:
            messagebox.showinfo("Busy", "A step is already running -- wait for it to finish.")
            return

        self.running = True
        self._stop_requested = False
        self._step_on_done = on_done
        self.stop_button.config(state="normal")
        step_name = extra_args[1]
        self._step_name = step_name
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
                        self.status_label.config(
                            text=f"Stopped ({elapsed:.1f}s): {self._step_name}", foreground="gray"
                        )
                        self._append_log(f"\n[stopped by user after {elapsed:.1f}s]\n")
                    elif payload == 0:
                        self.status_label.config(
                            text=f"Done ({elapsed:.1f}s): {self._step_name}", foreground="green"
                        )
                        self._append_log(f"\n[finished in {elapsed:.1f}s]\n")
                    else:
                        self.status_label.config(
                            text=f"Failed (exit {payload}, {elapsed:.1f}s): {self._step_name}", foreground="red"
                        )
                        self._append_log(f"\n[finished in {elapsed:.1f}s]\n")
                    self._last_step_ok = (not self._stop_requested) and payload == 0
                    on_done, self._step_on_done = self._step_on_done, None
                    if on_done is not None:
                        on_done()
                    self._refresh_preview_and_slider()
                    self._ring_bell()
                elif kind == "error":
                    self.running = False
                    self.stop_button.config(state="disabled")
                    elapsed = time.time() - self._step_start_time
                    self.status_label.config(text=f"Error ({elapsed:.1f}s)", foreground="red")
                    self._append_log(f"\n[GUI error] {payload}\n[finished in {elapsed:.1f}s]\n")
                    self._step_on_done = None  # never fired -- the subprocess itself never started
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
        # Flag unread output on the "◂ Console" button while collapsed so
        # a running step still signals activity without popping the pane open.
        if getattr(self, "_log_collapsed", False) and text.strip() and not self._log_unread:
            self._log_unread = True
            self._refresh_log_button_label()

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
        self._refresh_export_status_lights()

    _EXPORT_STATUS_COLORS = {
        EXPORT_STATUS_FRESH: "#2ea043",  # green
        EXPORT_STATUS_STALE: "#d1242f",  # red
        EXPORT_STATUS_MISSING: "gray",
    }

    def _refresh_export_status_lights(self) -> None:
        """
        Repaint the Repack section's export-status lights (see
        _build_file_tab) from PGA2k_gen.export_status -- called from
        _refresh_preview_and_slider, i.e. on working-dir change and
        after every step finishes, same as the preview redraw. Cheap
        (a project.json read + a handful of file stats), safe to call
        this often.
        """
        wd = self.working_dir.get().strip()
        statuses = export_status(Path(wd)) if wd and Path(wd).is_dir() else {}
        for category, light in self._export_status_lights.items():
            status = statuses.get(category, EXPORT_STATUS_MISSING)
            light.configure(fg=self._EXPORT_STATUS_COLORS.get(status, "gray"))

    def _on_preview_choice_changed(self) -> None:
        self._refresh_preview_and_slider()

    def _on_preview_version_changed(self, _value: str) -> None:
        self._update_version_label()
        self._show_preview()

    def _update_version_label(self) -> None:
        v = int(round(self.preview_version.get()))
        self.preview_version_label.config(text="current" if v == 0 else f"-{v}")

    def _on_preview_zoom_scroll(self, event) -> None:
        """Plain scroll over the preview image zooms it (see _on_preview_scroll for version stepping, now Ctrl+scroll over the image)."""
        if event.num == 4:
            step = 0.1
        elif event.num == 5:
            step = -0.1
        else:
            step = 0.1 if event.delta > 0 else -0.1

        new_zoom = max(0.25, min(3.0, self.preview_zoom_var.get() + step))
        if new_zoom == self.preview_zoom_var.get():
            return
        self._zoom_preview_keeping_point_under_cursor(event.x, event.y, new_zoom)

    def _zoom_preview_keeping_point_under_cursor(self, pointer_x: int, pointer_y: int, new_zoom: float) -> None:
        """
        Re-anchor the scroll position so whatever data point was under
        the pointer before the zoom step is still under it afterward --
        the usual "zoom toward cursor" behavior of image editors/maps.
        Without this, a zoom step re-centers the image (or leaves the
        scroll position untouched) and the point the user was actually
        looking at drifts out from under the pointer on every scroll
        tick.
        """
        canvas = self.preview_canvas
        frac_x = frac_y = 0.5
        if self._preview_canvas_image_id is not None and self._preview_imgtk is not None:
            canvas_x = canvas.canvasx(pointer_x)
            canvas_y = canvas.canvasy(pointer_y)
            offset_x, offset_y = canvas.coords(self._preview_canvas_image_id)
            img_w = self._preview_imgtk.width()
            img_h = self._preview_imgtk.height()
            if img_w and img_h:
                frac_x = (canvas_x - offset_x) / img_w
                frac_y = (canvas_y - offset_y) / img_h

        self.preview_zoom_var.set(new_zoom)
        self._show_preview()

        if self._preview_canvas_image_id is None or self._preview_imgtk is None:
            return
        new_offset_x, new_offset_y = canvas.coords(self._preview_canvas_image_id)
        new_img_w = self._preview_imgtk.width()
        new_img_h = self._preview_imgtk.height()
        target_canvas_x = new_offset_x + frac_x * new_img_w
        target_canvas_y = new_offset_y + frac_y * new_img_h

        region = canvas.cget("scrollregion").split()
        if len(region) != 4:
            return
        region_w = float(region[2]) - float(region[0])
        region_h = float(region[3]) - float(region[1])
        if region_w > 0:
            canvas.xview_moveto(max(0.0, min(1.0, (target_canvas_x - pointer_x) / region_w)))
        if region_h > 0:
            canvas.yview_moveto(max(0.0, min(1.0, (target_canvas_y - pointer_y) / region_h)))

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
        (see _spline_object_detail, _build_cluster_fill_rows), never
        for cross-referencing back to real OSM data.
        """
        return next_synthetic_osm_id(self._splines_features)

    _OBJECT_DOT_RADIUS_PX = 1  # -> a 2px-diameter dot, per the Objects tab's "Show objects" spec
    _OBJECT_LAYER_FILL_ALPHA = round(255 * 0.4)  # circle interior only -- center dot/outer stroke stay 100%
    _PICK_TOLERANCE_PX = 12  # screen-space hit radius for a single viewport click (see _pick_single)

    def _get_cached_object_preview_layer(self, working_dir: Path):
        """
        Lazily build the "Show objects" preview overlay's point list --
        (x, z, radius_or_None, category_id) triples in the local
        [0, COURSE_SIZE_M] frame -- mtime-keyed the same idiom as
        _get_cached_mask_merged_geometry/_get_cached_heightmap.

        Trees always come from object_list.json (available as soon as
        Generate Trees has run), keeping a per-tree LIDAR canopy radius
        (TREE_RADIUS_TAG) around for the circle-vs-dot choice below --
        placedObjects2.json's tree items would have already baked that
        into an opaque scale factor by the time they're written (see
        objects.py's build_tree_objects_v2019), losing it.

        Rocks/grass/ground-cover/display-plants come from objects.json's
        already-packed cluster records (course_output/object_clusters.py's
        pack_cluster_records -- run by the pack-objects CLI step, or, in
        this GUI, synchronously in-process by _regenerate_packed_objects
        right after a Splines/Objects-tab Fill/Clear/Delete or Generate
        Trees -- see that method). Reading objects.json directly here
        rather than placedObjects2.json means this overlay reflects a
        cluster fill immediately, with no `course/` extraction or a full
        Write Objects run needed first, and no version-specific Key
        shape (Key.path vs. Key.category/type) to translate back through
        -- objects.json's cluster records already carry a plain
        version-agnostic category id.

        Splitting the two sources this way (trees from one file, every
        other nature category from the other) means nothing is ever
        drawn twice.

        A tree renders as a circle sized to TREE_RADIUS_TAG when present
        (LIDAR-detected), otherwise a plain dot -- OSM-sourced trees
        carry no per-tree size data. A cluster stamp always renders as a
        circle at its own packed radius -- the closest thing this
        pipeline has to a "spacing"-derived on-the-ground size for a
        scatter-fill area.
        """
        object_list_path = working_dir / OBJECT_LIST_FILE
        objects_path = working_dir / OBJECTS_FILE
        ol_mtime = object_list_path.stat().st_mtime if object_list_path.exists() else None
        obj_mtime = objects_path.stat().st_mtime if objects_path.exists() else None
        cache_key = (str(working_dir), ol_mtime, obj_mtime)
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

        if objects_path.exists():
            try:
                _, cluster_records, collection_objects, _, _ = load_objects(objects_path)
                for record in cluster_records:
                    category = record.get("category")
                    if category not in _OBJECT_LAYER_STYLE:
                        continue
                    points.append((record["x"], record["z"], record.get("radius"), category))
                for obj in collection_objects:
                    points.append((obj["x"], obj["z"], None, _COLLECTION_LAYER_CATEGORY))
            except (json.JSONDecodeError, OSError, KeyError):
                pass

        self._cached_object_layer = points
        self._cached_object_layer_key = cache_key
        return points

    def _get_cached_object_spline_fill_layer(self, working_dir: Path):
        """
        Lazily built (waypoints, category) list -- one entry per object-
        spline fill piece currently in objects.json (course_output/
        object_clusters.py's pack_spline_records output, mode="spline"
        fills only) -- same mtime-keyed caching idiom as
        _get_cached_object_preview_layer, kept separate from it since an
        object-spline fill covers an AREA (no single meaningful radius,
        unlike a cluster stamp) and _composite_objects_layer renders it
        as a filled polygon outline instead of a circle.
        """
        objects_path = working_dir / OBJECTS_FILE
        obj_mtime = objects_path.stat().st_mtime if objects_path.exists() else None
        cache_key = (str(working_dir), obj_mtime)
        if getattr(self, "_cached_object_spline_layer_key", None) == cache_key:
            return self._cached_object_spline_layer

        pieces: list[tuple[list[tuple[float, float]], int]] = []
        if objects_path.exists():
            try:
                _, _, _, spline_fill_records, _ = load_objects(objects_path)
                for record in spline_fill_records:
                    category = record.get("category")
                    if category not in _OBJECT_LAYER_STYLE:
                        continue
                    waypoints = [(float(x), float(z)) for x, z in record.get("waypoints", [])]
                    if len(waypoints) >= 3:
                        pieces.append((waypoints, category))
            except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
                pass

        self._cached_object_spline_layer = pieces
        self._cached_object_spline_layer_key = cache_key
        return pieces

    @staticmethod
    def _float_field(var: "tk.StringVar", default: float) -> float:
        """Parse a numeric tolerance Entry's current text, falling back to `default`
        on a blank/invalid value (e.g. mid-edit) -- same idiom as _snap_elevation."""
        try:
            return float(var.get())
        except ValueError:
            return default

    def _get_water_preview_rects(self, working_dir: Path) -> list[tuple[float, float, float, float, float]]:
        """
        (center_x, center_z, width_m, depth_m, rotation_deg) for every
        "water" Feature currently in self._splines_features, course-
        cropped/shifted the same way the Splines-tab highlight overlay
        is (_shift_and_crop_to_course) -- so this lines up with the
        base preview image without needing course/ extraction or a
        Write Water run first. Mirrors the Refine tab's "Multi-tile
        water fill" checkbox and FILL MODE selector: when checked,
        calls course_output.water.fit_water_tiles ("edge" mode) or
        fit_water_stripes ("stripe" mode) per pond -- using the same
        tolerance fields that feed the actual Write Water run, see
        _write_terrain_water_args -- and returns every tile/stripe;
        otherwise calls fit_water_rectangle, exactly as before. Either
        way this is the exact geometry fit build_water_objects itself
        uses, not a re-derived approximation, and skips the water-level
        lookup entirely -- this overlay only needs the 2D footprint,
        not an elevation, so it never needs normalized stamps and never
        skips a pond just because refine-terrain hasn't reached it yet.

        Cached (mtime + params keyed, same idiom as
        _get_cached_object_preview_layer): the stripe/edge boundary
        probes fit_water_stripes/fit_water_tiles run per pond are NOT
        cheap on a course with many water bodies, and _show_preview
        re-runs this on every zoom tick / every water-param change (via
        _composite_objects_layer). Without the cache that made zooming
        and slider drags with "Show objects" on visibly stutter, and
        looked from the console like Write Water was re-running. The key
        covers features.geojson + project.json mtime (crop origin) and
        every water fill field that feeds the fit -- the only inputs the
        fit has -- so any real change still invalidates it. (Nothing a
        Write Water run produces feeds back into this footprint fit, so
        that run needs no explicit invalidation.)
        """
        self._ensure_splines_features_fresh(working_dir)
        multi_tile = self.multi_tile_water_var.get()
        fill_mode = self.water_fill_mode_var.get()

        project_path = working_dir / PROJECT_FILE
        cache_key = (
            str(working_dir),
            self._splines_features_mtime,
            project_path.stat().st_mtime if project_path.exists() else None,
            multi_tile, fill_mode,
            self.water_stripe_overlap_var.get(), self.water_stripe_min_edge_var.get(),
            self.water_stripe_tolerance_var.get(), self.water_stripe_max_stripes_var.get(),
            self.water_stripe_buffer_var.get(),
            self.water_tile_tolerance_var.get(), self.water_tile_min_edge_var.get(),
            self.water_tile_max_search_var.get(), self.water_tile_width_samples_var.get(),
            self.water_tile_redundancy_var.get(), self.water_tile_overlap_var.get(),
        )
        if getattr(self, "_cached_water_preview_rects_key", None) == cache_key:
            return self._cached_water_preview_rects

        features = self._shift_and_crop_to_course(working_dir, self._splines_features)
        rects = []
        for f in features:
            if f.kind != "water" or f.geometry.geom_type != "Polygon":
                continue
            if multi_tile and fill_mode == "stripe":
                fits = fit_water_stripes(
                    f.geometry,
                    overlap_m=self._float_field(self.water_stripe_overlap_var, DEFAULT_WATER_STRIPE_OVERLAP_M),
                    min_edge_m=self._float_field(self.water_stripe_min_edge_var, DEFAULT_WATER_TILE_MIN_EDGE_M),
                    tolerance_m=self._float_field(self.water_stripe_tolerance_var, DEFAULT_WATER_STRIPE_TOLERANCE_M),
                    max_stripes_per_side=int(self._float_field(
                        self.water_stripe_max_stripes_var, DEFAULT_WATER_STRIPE_MAX_STRIPES_PER_SIDE,
                    )),
                    buffer_m=self._float_field(self.water_stripe_buffer_var, DEFAULT_WATER_STRIPE_BUFFER_M),
                )
                if fits:
                    rects.extend(fits)
            elif multi_tile:
                fits = fit_water_tiles(
                    f.geometry,
                    tolerance_m=self._float_field(self.water_tile_tolerance_var, DEFAULT_WATER_TILE_TOLERANCE_M),
                    min_edge_m=self._float_field(self.water_tile_min_edge_var, DEFAULT_WATER_TILE_MIN_EDGE_M),
                    max_search_m=self._float_field(self.water_tile_max_search_var, DEFAULT_WATER_TILE_MAX_SEARCH_M),
                    width_samples=int(self._float_field(
                        self.water_tile_width_samples_var, DEFAULT_WATER_TILE_WIDTH_SAMPLES,
                    )),
                    redundancy_ratio=self._float_field(
                        self.water_tile_redundancy_var, DEFAULT_WATER_TILE_REDUNDANCY_RATIO,
                    ),
                    overlap_m=self._float_field(self.water_tile_overlap_var, DEFAULT_WATER_TILE_OVERLAP_M),
                )
                if fits:
                    rects.extend(fits)
            else:
                fit = fit_water_rectangle(f.geometry)
                if fit is not None:
                    rects.append(fit)

        self._cached_water_preview_rects = rects
        self._cached_water_preview_rects_key = cache_key
        return rects

    def _object_overlay_fingerprint(self, working_dir: Path):
        """
        Everything _composite_objects_layer's three data sources
        (_get_cached_object_preview_layer, _get_water_preview_rects,
        _get_cached_object_spline_fill_layer) actually read, collapsed
        into one tuple for _show_preview's outer per-zoom-level render
        cache (see render_key there). Each of those three has its own
        internal mtime-keyed cache already, but that only saves the
        WORK of rebuilding their point/rect lists -- it does nothing
        for the render cache sitting in front of them, which used to be
        keyed only on show_objects_var.get() (on/off), not on whether
        the underlying files actually changed. A step that rewrites
        objects.json/object_list.json without touching the base preview
        PNG (generate-parking, generate-range-nets, generate-streams,
        generate-collections -- see PGA2k_gen.py) left geo_overlay_key
        identical to the pre-run value, so the render cache kept
        returning the stale pre-run composite forever, even though the
        three inner caches above were already correctly holding the new
        data. Cheap to compute every call: a handful of stat()s plus
        already-materialized StringVar reads, no file parsing.
        """
        object_list_path = working_dir / OBJECT_LIST_FILE
        objects_path = working_dir / OBJECTS_FILE
        features_path = working_dir / FEATURES_FILE
        project_path = working_dir / PROJECT_FILE
        return (
            object_list_path.stat().st_mtime if object_list_path.exists() else None,
            objects_path.stat().st_mtime if objects_path.exists() else None,
            features_path.stat().st_mtime if features_path.exists() else None,
            project_path.stat().st_mtime if project_path.exists() else None,
            self.multi_tile_water_var.get(), self.water_fill_mode_var.get(),
            self.water_stripe_overlap_var.get(), self.water_stripe_min_edge_var.get(),
            self.water_stripe_tolerance_var.get(), self.water_stripe_max_stripes_var.get(),
            self.water_stripe_buffer_var.get(),
            self.water_tile_tolerance_var.get(), self.water_tile_min_edge_var.get(),
            self.water_tile_max_search_var.get(), self.water_tile_width_samples_var.get(),
            self.water_tile_redundancy_var.get(), self.water_tile_overlap_var.get(),
        )

    def _composite_objects_layer(self, img: "Image.Image", working_dir: Path) -> "Image.Image":
        """
        Composite the "Show objects" overlay onto `img` (already at its
        current zoomed size) -- one filled circle per point from
        _get_cached_object_preview_layer, colored/sized per _OBJECT_
        LAYER_STYLE, one outlined rotated rectangle per water body from
        _get_water_preview_rects, plus a legend key in the lower-left
        corner for whichever categories actually appeared. Same
        _PLOT_RECT-aware data-area positioning as the mask buffer/
        elevation contour/highlight overlays above -- course-cropped
        previews only (see their shared caller-side exclusion in
        _show_preview).

        Water rectangles are drawn first, underneath every tree/cluster
        circle -- a pond is a large background feature, and a tree/rock
        marker sitting on its bank should never be hidden under the
        water fill.

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
        water_rects = self._get_water_preview_rects(working_dir)
        spline_fill_pieces = self._get_cached_object_spline_fill_layer(working_dir)
        if not points and not water_rects and not spline_fill_pieces:
            return img

        left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
        data_left = round(img.width * left_frac)
        data_top = round(img.height * (1 - bottom_frac - height_frac))
        data_width = max(1, round(img.width * width_frac))
        data_height = max(1, round(img.height * height_frac))

        def _to_px(x: float, z: float) -> tuple[float, float]:
            return (
                (x / COURSE_SIZE_M) * data_width,
                (1.0 - z / COURSE_SIZE_M) * data_height,  # row 0 = max z, same flip as the other overlays
            )

        order = {cat: i for i, cat in enumerate(_OBJECT_LAYER_DRAW_ORDER)}
        points = sorted(points, key=lambda p: order.get(p[3], -1))

        layer = Image.new("RGBA", (data_width, data_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        for cx, cz, width_m, depth_m, rotation_deg in water_rects:
            angle = math.radians(rotation_deg)
            width_dir = (math.cos(angle), math.sin(angle))
            depth_dir = (-math.sin(angle), math.cos(angle))
            half_w, half_d = width_m / 2.0, depth_m / 2.0
            corners = [
                (
                    cx + sw * half_w * width_dir[0] + sd * half_d * depth_dir[0],
                    cz + sw * half_w * width_dir[1] + sd * half_d * depth_dir[1],
                )
                for sw, sd in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            corners_px = [_to_px(x, z) for x, z in corners]
            draw.polygon(corners_px, fill=(*_WATER_LAYER_COLOR, _WATER_LAYER_FILL_ALPHA))
            draw.line(corners_px + [corners_px[0]], fill=(*_WATER_LAYER_COLOR, 255), width=2)

        used_categories: set[int] = set()
        for waypoints, category in spline_fill_pieces:
            style = _OBJECT_LAYER_STYLE.get(category)
            if style is None:
                continue
            used_categories.add(category)
            color = style[1]
            ring_px = [_to_px(x, z) for x, z in waypoints]
            draw.polygon(ring_px, fill=(*color, self._OBJECT_LAYER_FILL_ALPHA))
            draw.line(ring_px + [ring_px[0]], fill=(*color, 255), width=2)

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

        if used_categories or water_rects:
            img = self._draw_object_layer_key(img, used_categories, show_water=bool(water_rects))
        return img

    @staticmethod
    def _draw_object_layer_key(img: "Image.Image", used_categories: set, show_water: bool = False) -> "Image.Image":
        """Small swatch+label legend, lower-left corner of `img`, one row
        per category actually present in the current overlay -- so a
        course with e.g. no cluster-filled rocks yet just shows Trees,
        not a 5-row key advertising categories with nothing on screen.
        Water gets a rectangle swatch (not an ellipse, like every other
        row) since it's drawn on the map as a rotated rectangle, not a
        circle -- the legend shape should match what's actually on screen."""
        entries = [(*style, False) for cat, style in _OBJECT_LAYER_STYLE.items() if cat in used_categories]
        if show_water:
            entries.append((_WATER_LAYER_LABEL, _WATER_LAYER_COLOR, True))
        if not entries:
            return img

        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        swatch = 10
        pad = 6
        line_h = swatch + 5
        try:
            text_w = max(draw.textlength(label, font=font) for label, _, _ in entries)
        except AttributeError:  # older Pillow without textlength
            text_w = max(len(label) for label, _, _ in entries) * 6
        box_w = int(pad * 2 + swatch + 6 + text_w)
        box_h = int(pad * 2 + line_h * len(entries))
        x0, y0 = 8, img.height - box_h - 8

        draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(255, 255, 255, 210), outline=(0, 0, 0, 255))
        for i, (label, color, is_rect) in enumerate(entries):
            sy = y0 + pad + i * line_h
            swatch_box = (x0 + pad, sy, x0 + pad + swatch, sy + swatch)
            if is_rect:
                draw.rectangle(swatch_box, fill=(*color, 255), outline=(0, 0, 0, 255))
            else:
                draw.ellipse(swatch_box, fill=(*color, 255), outline=(0, 0, 0, 255))
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
        # PhotoImage creation (the full PIL->X pixel upload) is the
        # second most expensive per-tick cost after the resize. When
        # the caller hands back the exact PIL object we already
        # encoded (see _show_preview's per-zoom-level render cache --
        # a zoom tick inside one level reuses the cached final image
        # untouched), reuse the PhotoImage: the tick then costs only
        # a canvas item swap, no re-encode. Identity-keyed: any
        # re-render that composites fresh (overlay change, new zoom
        # level) yields a new PIL object and gets a new PhotoImage.
        # _preview_imgtk always references the live PhotoImage.
        cached = self._preview_photo_cached
        if cached is not None and self._preview_photo_src is pil_image:
            self._preview_imgtk = cached
        else:
            self._preview_imgtk = ImageTk.PhotoImage(pil_image)
            self._preview_photo_cached = self._preview_imgtk
            self._preview_photo_src = pil_image
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

    # ------------------------------------------------------------------
    # Viewport picking / marquee-select
    #
    # Left-click a spline or object marker on the preview to select its
    # row in the Splines/Objects tree (in place -- no tab switch); those
    # <<TreeviewSelect>> handlers already drive the cyan highlight and
    # every Mask/Clear/Delete action. Left-drag a box to marquee-select
    # everything inside it (Shift/Ctrl to add to the current selection).
    # Alt+left-drag a box instead cuts that region out of a selected
    # GUI-authored border ring / "Use mask" fill (never a real OSM
    # feature) -- see _marquee_subtract. Right-drag stays pan.
    #
    # Only *visible* layers take part: splines are pickable only while
    # "Overlay OSM" (Splines tab) is on, objects only while "Show
    # objects" (Objects tab) is on -- you can't click what you can't see.
    # Turning either toggle off also drops that layer's current selection
    # (see _on_overlay_osm_toggled / _on_show_objects_toggled).
    # ------------------------------------------------------------------

    def _canvas_xy_to_course(self, cx: float, cy: float):
        """
        Map a canvas point (already through canvasx()/canvasy(), i.e.
        scroll-adjusted) to (x_m, z_m, m_per_px) in the course
        [0, COURSE_SIZE_M] frame, or None when there's no pickable
        preview or the point is outside the plot's data area.

        The PhotoImage is drawn 1:1 on the canvas and every preview
        reserves the identical viz._PLOT_RECT sub-rectangle (figure
        fractions) for its 2000x2000 data area at any zoom, so inverting
        is just: undo the image-item offset, then undo _PLOT_RECT.
        """
        meta = self._preview_render_meta
        if meta is None or self._preview_canvas_image_id is None:
            return None
        if meta["base_kind"] in (
            PREVIEW_LIDAR, PREVIEW_LIDAR_HEIGHTMAP, PREVIEW_OSM, PREVIEW_OSM_FULL,
        ):
            return None  # not in the course crop frame -- same exclusion the overlays use
        try:
            ox, oy = self.preview_canvas.coords(self._preview_canvas_image_id)
        except (ValueError, tk.TclError):
            return None
        img_w, img_h = meta["img_w"], meta["img_h"]
        left_frac, bottom_frac, width_frac, height_frac = viz._PLOT_RECT
        data_left = img_w * left_frac
        data_top = img_h * (1.0 - bottom_frac - height_frac)
        data_w = img_w * width_frac
        data_h = img_h * height_frac
        if data_w <= 0 or data_h <= 0:
            return None
        fx = (cx - ox - data_left) / data_w
        fy = (cy - oy - data_top) / data_h
        if not (-0.02 <= fx <= 1.02 and -0.02 <= fy <= 1.02):
            return None
        x_m = min(max(fx, 0.0), 1.0) * COURSE_SIZE_M
        z_m = (1.0 - min(max(fy, 0.0), 1.0)) * COURSE_SIZE_M  # row 0 = max z, same flip as the overlays
        return (x_m, z_m, COURSE_SIZE_M / data_w)

    def _on_preview_pick_press(self, event) -> None:
        self._marquee_subtract_mode = False  # a plain left-press is always "select"
        if self._preview_canvas_image_id is None:
            self._marquee_anchor = None
            return
        self._marquee_anchor = (
            self.preview_canvas.canvasx(event.x), self.preview_canvas.canvasy(event.y),
        )
        if self._marquee_rect_id is not None:
            self.preview_canvas.delete(self._marquee_rect_id)
            self._marquee_rect_id = None

    def _on_preview_subtract_press(self, event) -> None:
        # <Alt-Button-1>: same as a pick press, but latch subtract mode so
        # the drag box (drawn by the generic <B1-Motion> handler) is
        # applied as a geometry subtraction on release instead of a
        # selection. More-specific binding, so the generic <Button-1>
        # handler does not also fire for this press.
        if self._preview_canvas_image_id is None:
            self._marquee_anchor = None
            self._marquee_subtract_mode = False
            return
        self._marquee_subtract_mode = True
        self._marquee_anchor = (
            self.preview_canvas.canvasx(event.x), self.preview_canvas.canvasy(event.y),
        )
        if self._marquee_rect_id is not None:
            self.preview_canvas.delete(self._marquee_rect_id)
            self._marquee_rect_id = None

    def _on_preview_pick_motion(self, event) -> None:
        if self._marquee_anchor is None:
            return
        ax, ay = self._marquee_anchor
        cx = self.preview_canvas.canvasx(event.x)
        cy = self.preview_canvas.canvasy(event.y)
        if self._marquee_rect_id is None:
            if abs(cx - ax) < 4 and abs(cy - ay) < 4:
                return  # still within click slop -- don't start a box yet
            outline = "#ff6a00" if self._marquee_subtract_mode else "#00b0ff"
            self._marquee_rect_id = self.preview_canvas.create_rectangle(
                ax, ay, cx, cy, outline=outline, width=1, dash=(3, 3),
            )
        else:
            self.preview_canvas.coords(self._marquee_rect_id, ax, ay, cx, cy)

    def _on_preview_pick_release(self, event) -> None:
        anchor = self._marquee_anchor
        self._marquee_anchor = None
        subtract = self._marquee_subtract_mode
        self._marquee_subtract_mode = False
        was_drag = self._marquee_rect_id is not None
        if self._marquee_rect_id is not None:
            self.preview_canvas.delete(self._marquee_rect_id)
            self._marquee_rect_id = None
        if anchor is None:
            return
        release_pt = (self.preview_canvas.canvasx(event.x), self.preview_canvas.canvasy(event.y))
        additive = bool(event.state & 0x0005)  # Shift (0x0001) or Control (0x0004) held
        if was_drag:
            if subtract:
                self._marquee_subtract(anchor, release_pt)
            else:
                self._marquee_select(anchor, release_pt, additive)
        elif not subtract:
            self._pick_single(event, additive)

    def _clear_tree_selection(self, tree: "ttk.Treeview") -> None:
        sel = tree.selection()
        if sel:
            tree.selection_remove(*sel)  # fires <<TreeviewSelect>> -> highlight state clears

    def _on_overlay_osm_toggled(self) -> None:
        # Splines are only pickable while visible; turning the overlay
        # off drops any spline selection so a now-hidden highlight
        # doesn't linger.
        if not self.overlay_osm_var.get():
            self._clear_tree_selection(self.splines_tree)
        self._show_preview()

    def _on_show_objects_toggled(self) -> None:
        if not self.show_objects_var.get():
            self._clear_tree_selection(self.objects_tree)
        self._show_preview()

    def _select_tree_row(self, tree: "ttk.Treeview", iid: str, additive: bool) -> None:
        if additive:
            if iid in tree.selection():
                tree.selection_remove(iid)
            else:
                tree.selection_add(iid)
        else:
            tree.selection_set(iid)
        if tree.exists(iid):
            tree.see(iid)
            tree.focus(iid)

    def _pick_single(self, event, additive: bool) -> None:
        hit = self._canvas_xy_to_course(
            self.preview_canvas.canvasx(event.x), self.preview_canvas.canvasy(event.y),
        )
        if hit is None:
            return
        x_m, z_m, m_per_px = hit
        tol_m = self._PICK_TOLERANCE_PX * m_per_px
        wd = self.working_dir.get().strip()
        working_dir = Path(wd) if wd else None
        objects_visible = self.show_objects_var.get()
        splines_visible = self.overlay_osm_var.get()

        # Objects first: small point targets, and a marker sitting on top
        # of a large fairway polygon should win the click.
        if objects_visible:
            obj_iid = self._nearest_object_iid(x_m, z_m, tol_m, working_dir)
            if obj_iid is not None:
                self._select_tree_row(self.objects_tree, obj_iid, additive)
                return

        if splines_visible:
            spline_iid = self._nearest_spline_iid(x_m, z_m, tol_m, working_dir)
            if spline_iid is not None:
                self._select_tree_row(self.splines_tree, spline_iid, additive)
                return

        # Plain click on empty space clears the selection -- but only for
        # layers currently visible (you can't deselect what you can't see).
        if not additive:
            for tree, vis in (
                (self.splines_tree, splines_visible), (self.objects_tree, objects_visible),
            ):
                if vis and tree.selection():
                    tree.selection_remove(*tree.selection())

    def _nearest_object_iid(self, x_m: float, z_m: float, tol_m: float, working_dir):
        """Nearest currently-listed Objects-tab row to (x_m, z_m) within
        tol_m, or None. Plain tree rows hit-test against their own point;
        cluster-fill "group" rows hit-test against their packed scatter
        points from objects.json (matched back by spline_id + category,
        same cross-reference _on_object_selected uses)."""
        children = self.objects_tree.get_children()
        best_iid, best_d = None, tol_m
        for iid in children:
            if not iid.isdigit():
                continue
            idx = int(iid)
            if 0 <= idx < len(self._objects_tree_list):
                ox, oz, _tags = self._objects_tree_list[idx]
                d = math.hypot(ox - x_m, oz - z_m)
                if d < best_d:
                    best_iid, best_d = iid, d

        group_iids = [iid for iid in children if iid.startswith("c")]
        if group_iids and working_dir is not None:
            objects_path = working_dir / OBJECTS_FILE
            records = []
            if objects_path.exists():
                try:
                    _, records, _, _, _ = load_objects(objects_path)
                except (json.JSONDecodeError, OSError, KeyError):
                    records = []
            for iid in group_iids:
                try:
                    j = int(iid[1:])
                except ValueError:
                    continue
                if not (0 <= j < len(self._cluster_fill_rows)):
                    continue
                ids_str, _lbl, _r, _dns, _src, category, _type, _mode = self._cluster_fill_rows[j]
                row_spline_ids = {int(s) for s in ids_str.split(",") if s}
                for rec in records:
                    if rec.get("category") != category or rec.get("spline_id") not in row_spline_ids:
                        continue
                    d = math.hypot(rec["x"] - x_m, rec["z"] - z_m)
                    if d < best_d:
                        best_iid, best_d = iid, d
        return best_iid

    def _nearest_spline_iid(self, x_m: float, z_m: float, tol_m: float, working_dir):
        """Nearest currently-listed Splines-tab feature to (x_m, z_m)
        within tol_m, or None. A polygon the point falls inside scores
        distance 0; ties there break toward the smallest-area polygon so
        a click inside overlapping fills picks the tightest one."""
        if not self._splines_features or working_dir is None:
            return None
        visible = set(self.splines_tree.get_children())
        if not visible:
            return None
        pt = Point(x_m, z_m)
        best_iid, best_d, best_area = None, tol_m, float("inf")
        for f in self._shift_and_crop_to_course(working_dir, self._splines_features):
            if f.osm_id is None:
                continue
            iid = str(f.osm_id)
            if iid not in visible:
                continue
            try:
                d = f.geometry.distance(pt)
                area = f.geometry.area
            except Exception:
                continue
            if d < best_d or (d <= 0.0 and best_d <= 0.0 and area < best_area):
                best_iid, best_d, best_area = iid, d, area
        return best_iid

    def _marquee_select(self, p0, p1, additive: bool) -> None:
        c0 = self._canvas_xy_to_course(*p0)
        c1 = self._canvas_xy_to_course(*p1)
        if c0 is None or c1 is None:
            return
        x0, z0, _ = c0
        x1, z1, _ = c1
        region = shapely_box(min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
        if region.area <= 0:
            return
        wd = self.working_dir.get().strip()
        working_dir = Path(wd) if wd else None
        objects_visible = self.show_objects_var.get()
        splines_visible = self.overlay_osm_var.get()

        if objects_visible:
            obj_iids = []
            for iid in self.objects_tree.get_children():
                if not iid.isdigit():
                    continue
                idx = int(iid)
                if 0 <= idx < len(self._objects_tree_list):
                    ox, oz, _tags = self._objects_tree_list[idx]
                    if region.contains(Point(ox, oz)):
                        obj_iids.append(iid)
            self._apply_marquee_selection(self.objects_tree, obj_iids, additive)

        if splines_visible and self._splines_features and working_dir is not None:
            spline_iids = []
            listed = set(self.splines_tree.get_children())
            for f in self._shift_and_crop_to_course(working_dir, self._splines_features):
                if f.osm_id is None:
                    continue
                iid = str(f.osm_id)
                if iid not in listed:
                    continue
                try:
                    if region.intersects(f.geometry):
                        spline_iids.append(iid)
                except Exception:
                    continue
            self._apply_marquee_selection(self.splines_tree, spline_iids, additive)

    @staticmethod
    def _apply_marquee_selection(tree: "ttk.Treeview", iids: list, additive: bool) -> None:
        if additive:
            for iid in iids:
                tree.selection_add(iid)
        else:
            tree.selection_set(*iids)  # no args clears the selection

    _SUBTRACT_AREA_TYPES = ("Polygon", "MultiPolygon")
    _SUBTRACT_LINE_TYPES = ("LineString", "MultiLineString", "LinearRing")
    # Alt-subtract only ever edits GUI-authored cluster-fill geometry
    # (border rings + their centerlines, "Use mask" clips). It deliberately
    # will NOT cut a real OSM feature's geometry: that edit is destructive,
    # persists to features.geojson, silently reshapes the height mask for a
    # mask=False feature, and is only recoverable by re-running Ingest OSM.
    _SUBTRACT_KINDS = (SYNTHETIC_BORDER_KIND, SYNTHETIC_MASKED_KIND)

    @staticmethod
    def _lines_only(geom):
        """Line parts of `geom` (drop stray Points from a GeometryCollection
        that shapely's difference can leave where a cut edge just grazes a
        vertex). Returns the geometry unchanged when it's already a line
        type or has no line parts to salvage."""
        if geom.is_empty or geom.geom_type in PGAGenGUI._SUBTRACT_LINE_TYPES:
            return geom
        if geom.geom_type == "GeometryCollection":
            parts = [
                g for g in geom.geoms
                if g.geom_type in PGAGenGUI._SUBTRACT_LINE_TYPES and not g.is_empty
            ]
            if parts:
                return parts[0] if len(parts) == 1 else unary_union(parts)
        return geom

    @staticmethod
    def _estimate_ring_half_width(ring_polygon, centerline) -> float:
        """Half the band width of a border ring (build_border_ring_geometry
        stroked its centerline by border_width, half each side). A point
        sitting ON the centerline is at the band's middle, so its distance
        to the band edge is that half-width -- median a few such samples,
        falling back to area/(2*length) for a degenerate centerline."""
        boundary = ring_polygon.boundary
        dists = []
        for t in (0.15, 0.35, 0.5, 0.65, 0.85):
            try:
                p = centerline.interpolate(t, normalized=True)
            except Exception:
                continue
            if ring_polygon.contains(p):
                dists.append(boundary.distance(p))
        if dists:
            dists.sort()
            return dists[len(dists) // 2]
        length = centerline.length
        return ring_polygon.area / (2.0 * length) if length > 0 else 0.0

    def _marquee_subtract(self, p0, p1) -> None:
        """
        Alt+left-drag box -> cut the box region out of every currently-
        selected GUI-authored cluster-fill row. Real OSM features are
        skipped (with a log note): cutting their geometry is destructive,
        persists to features.geojson, silently reshapes the height mask,
        and only Ingest OSM can undo it -- not worth the footgun.

        For a border ring the cut is centerline-driven: the box trims the
        ring's own centerline (an "o" becomes a "c"), then the ring
        polygon is rebuilt as that trimmed line re-stroked to the band's
        original width and clipped back to the original outline -- so the
        removed span disappears cleanly instead of the band just detouring
        inward at full width. A "Use mask" clip (SYNTHETIC_MASKED_KIND)
        is a straight geometry difference against the box.

        Mutates self._splines_features in place, persists to
        features.geojson, and refreshes exactly as a Fill would. Reports
        through the log pane -- no dialog.
        """
        wd = self.working_dir.get().strip()
        if not wd:
            self._append_log("\n[Alt-subtract: set a working directory first]\n")
            return
        working_dir = Path(wd)

        c0 = self._canvas_xy_to_course(*p0)
        c1 = self._canvas_xy_to_course(*p1)
        if c0 is None or c1 is None:
            self._append_log(
                "\n[Alt-subtract: box was outside the course data area "
                "(or this preview isn't in the course frame)]\n"
            )
            return
        x0, z0, _ = c0
        x1, z1, _ = c1
        box_course = shapely_box(min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1))
        if box_course.area <= 0:
            return

        # course frame -> features.geojson's full/uncropped frame.
        # _shift_and_crop_to_course goes full->course with dx=-shift_x,
        # dz=-shift_z; the inverse is dx=+shift_x, dz=+shift_z (pure
        # translation, no Z flip). Reuse shift_features via a throwaway
        # Feature so the translation matches that path exactly.
        project = load_project(working_dir)
        shift_x = project.get("course_crop_origin_in_full_frame_x")
        shift_z = project.get("course_crop_origin_in_full_frame_z")
        if shift_x is None or shift_z is None:
            box_full = box_course  # pre-dates the saved origin; treat as course frame
        else:
            box_full = shift_features(
                [Feature(geometry=box_course, kind="__tmp__", tags={}, osm_id=None, mask=False)],
                dx=shift_x, dz=shift_z,
            )[0].geometry

        self._ensure_splines_features_fresh(working_dir)
        selected_ids = {int(s) for s in self.splines_tree.selection()}
        if not selected_ids:
            self._append_log(
                "\n[Alt-subtract: no Splines rows selected -- select the feature(s) to "
                "cut from first (Overlay OSM must be on to pick them)]\n"
            )
            return

        by_id = {f.osm_id: f for f in self._splines_features if f.osm_id is not None}
        changed = 0
        skipped_real = []
        for osm_id in selected_ids:
            f = by_id.get(osm_id)
            if f is None:
                continue
            if f.kind not in self._SUBTRACT_KINDS:
                skipped_real.append(f)
                continue

            centerline_feat = None
            if f.kind == SYNTHETIC_BORDER_KIND:
                ref = f.tags.get(PGA_CLUSTER_CENTERLINE_REF_TAG)
                if ref is not None:
                    try:
                        centerline_feat = by_id.get(ref) or by_id.get(int(ref))
                    except (TypeError, ValueError):
                        centerline_feat = by_id.get(ref)

            if (
                centerline_feat is not None
                and not centerline_feat.geometry.is_empty
                and f.geometry.geom_type in self._SUBTRACT_AREA_TYPES
            ):
                if self._subtract_from_border(f, centerline_feat, box_full):
                    changed += 1
                continue

            # Non-border: straight difference against the box.
            try:
                new_geom = f.geometry.difference(box_full)
            except Exception as exc:  # pragma: no cover -- defensive against odd geometries
                self._append_log(f"\n[Alt-subtract: skipped #{f.osm_id} ({f.kind}): {exc}]\n")
                continue
            accept = self._SUBTRACT_AREA_TYPES + self._SUBTRACT_LINE_TYPES
            if new_geom.is_empty:
                self._append_log(
                    f"\n[Alt-subtract: box covers all of #{f.osm_id} ({f.kind}) -- skipped "
                    f"(use Clear/Delete to remove a whole feature)]\n"
                )
                continue
            if not new_geom.is_valid:
                new_geom = new_geom.buffer(0)
            if new_geom.is_empty or new_geom.geom_type not in accept:
                self._append_log(
                    f"\n[Alt-subtract: cut of #{f.osm_id} ({f.kind}) produced "
                    f"{new_geom.geom_type or 'nothing usable'} -- skipped]\n"
                )
                continue
            f.geometry = new_geom
            changed += 1

        if skipped_real:
            ids = ", ".join(f"#{f.osm_id} ({f.kind})" for f in skipped_real)
            self._append_log(
                f"\n[Alt-subtract: won't cut real OSM feature(s) {ids} -- it only trims "
                f"GUI-authored border rings / 'Use mask' fills. Nothing was changed on those.]\n"
            )

        if changed == 0:
            if not skipped_real:
                self._append_log(
                    "\n[Alt-subtract: nothing changed -- drag the box across the part of the "
                    "selected border ring / mask fill you want removed]\n"
                )
            return

        save_features(self._splines_features, working_dir / FEATURES_FILE)
        self._cached_mask_merged_geom = None
        self._cached_mask_geom_key = None
        self._cached_geo_overlay = None
        self._cached_geo_overlay_key = None
        self._regenerate_height_mask(working_dir)
        self._regenerate_packed_objects(working_dir)
        self._refresh_splines_list()
        self._refresh_objects_list()
        restorable = [str(i) for i in selected_ids if self.splines_tree.exists(str(i))]
        if restorable:
            self.splines_tree.selection_set(restorable)
        self._show_preview()
        self._append_log(f"\n[Alt-subtract: cut box region from {changed} feature(s)]\n")

    def _subtract_from_border(self, border_feat, centerline_feat, box_full) -> bool:
        """
        Trim a border ring by its centerline: cut `box_full` out of the
        centerline, then rebuild the ring polygon as the trimmed line
        re-stroked to the band's original half-width and clipped back to
        the original outline. Mutates both Features in place. Returns True
        if the geometry actually changed, False (with a log note) if the
        box missed the centerline or swallowed the whole ring.
        """
        cl_geom = centerline_feat.geometry
        try:
            trimmed = self._lines_only(cl_geom.difference(box_full))
            # difference() splits a closed ring at its own seam vertex, so a
            # single cut yields a MultiLineString of two pieces meeting at
            # that vertex. Buffering that gives each piece a (flat) end cap
            # there instead of a proper join -> a notch bitten out of the
            # band's outer corner well away from the cut. linemerge stitches
            # the contiguous pieces back into one LineString so the seam
            # vertex is an interior corner again; genuinely disjoint pieces
            # (two cuts, or several rings) correctly stay separate.
            if trimmed.geom_type == "MultiLineString":
                trimmed = linemerge(trimmed)
        except Exception as exc:  # pragma: no cover -- defensive
            self._append_log(f"\n[Alt-subtract: skipped #{border_feat.osm_id} (border): {exc}]\n")
            return False

        if trimmed.is_empty:
            self._append_log(
                f"\n[Alt-subtract: box covers the whole border ring #{border_feat.osm_id} -- "
                f"use Clear/Delete to remove it]\n"
            )
            return False
        if trimmed.geom_type not in self._SUBTRACT_LINE_TYPES:
            self._append_log(
                f"\n[Alt-subtract: border #{border_feat.osm_id} centerline cut produced "
                f"{trimmed.geom_type} -- skipped]\n"
            )
            return False
        if trimmed.length >= cl_geom.length - 1e-6:
            self._append_log(
                f"\n[Alt-subtract: box didn't reach border #{border_feat.osm_id}'s centerline -- "
                f"drag the box across the middle of the band]\n"
            )
            return False

        half_w = self._estimate_ring_half_width(border_feat.geometry, cl_geom)
        new_ring = border_feat.geometry.difference(box_full)  # sane fallback
        if half_w > 0:
            # cap_style=2 (flat) ends the band square at the cut rather
            # than bulging a rounded cap half_w past it.
            rebuilt = trimmed.buffer(half_w, cap_style=2).intersection(border_feat.geometry)
            if not rebuilt.is_valid:
                rebuilt = rebuilt.buffer(0)
            if not rebuilt.is_empty and rebuilt.geom_type in self._SUBTRACT_AREA_TYPES:
                new_ring = rebuilt

        if new_ring.is_empty or new_ring.geom_type not in self._SUBTRACT_AREA_TYPES:
            self._append_log(
                f"\n[Alt-subtract: border #{border_feat.osm_id} rebuild came out empty -- skipped]\n"
            )
            return False

        centerline_feat.geometry = trimmed
        border_feat.geometry = new_ring
        return True

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
                # keep the editable boxes in step with the sliders
                self.mask_buffer_preview_text.set(f"{buffer_px:.0f}")
                self.border_width_preview_text.set(f"{border_width_raw:.0f}")
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

                        # Bright cyan outline traced over the translucent
                        # fill -- the same crisp "selected" cue the Objects
                        # tab's marker gets from its selection ring, so a
                        # picked spline reads as selected at a glance
                        # rather than just faintly tinted.
                        def _course_to_px(x: float, z: float) -> tuple[float, float]:
                            return (
                                (x / COURSE_SIZE_M) * data_width,
                                (1.0 - z / COURSE_SIZE_M) * data_height,
                            )

                        def _iter_boundary_rings(geom):
                            if geom.is_empty:
                                return
                            gt = geom.geom_type
                            if gt == "Polygon":
                                yield list(geom.exterior.coords)
                                for interior in geom.interiors:
                                    yield list(interior.coords)
                            elif gt in ("LineString", "LinearRing"):
                                yield list(geom.coords)
                            elif gt in ("MultiPolygon", "MultiLineString", "GeometryCollection"):
                                for part in geom.geoms:
                                    yield from _iter_boundary_rings(part)

                        outline_layer = Image.new("RGBA", (data_width, data_height), (0, 0, 0, 0))
                        outline_draw = ImageDraw.Draw(outline_layer)
                        for ring in _iter_boundary_rings(highlight_geom):
                            pts = [_course_to_px(x, z) for x, z in ring]
                            if len(pts) >= 2:
                                outline_draw.line(pts, fill=(0, 255, 255, 255), width=3)
                        outline_full = Image.new("RGBA", (geo_img.width, geo_img.height), (0, 0, 0, 0))
                        outline_full.paste(outline_layer, (data_left, data_top), outline_layer)
                        geo_img = Image.alpha_composite(geo_img, outline_full)

                self._cached_geo_overlay = geo_img
                self._cached_geo_overlay_key = geo_overlay_key

            # Per-zoom-level render cache. Everything from here down
            # (resize + elevation band + objects layer + PhotoImage)
            # used to run on every single zoom tick at the zoomed
            # pixel size -- LANCZOS alone was ~170ms at 100%, ~490ms at
            # 200%, ~930ms at 300% on a 1959x1780 composite, all on
            # the Tk mainloop, which is what made scroll-zooming feel
            # frozen. Zooming in adds no information, so: (a) the zoom
            # fed to the renderer is quantized to 1/4 steps -- the
            # cursor-anchored scroll reposition below keeps the point
            # under the pointer exact regardless, and (b) the finished
            # image for a (content, zoom-level) pair is cached, so a
            # tick that stays within one level costs only a canvas
            # item swap (and _set_preview_image reuses the PhotoImage
            # when it gets the same PIL object back). Content changes
            # bust the cache through geo_overlay_key (which already
            # covers the base PNG mtime, OSM overlay, mask buffer and
            # spline highlight); the elevation-band and objects-toggle
            # inputs applied AFTER the cache check are added to the
            # key explicitly. NOTE: show_objects_var.get() alone only
            # says whether the overlay is ON, not whether the DATA it
            # draws changed -- a step like generate-parking/generate-
            # range-nets/generate-streams/generate-collections rewrites
            # objects.json (or object_list.json) without touching the
            # base preview PNG at all (see PGA2k_gen.py's steps that
            # skip the full preview refresh), so without an explicit
            # fingerprint here this cache kept serving the pre-run
            # image forever -- confirmed as the cause of "Show objects"
            # looking frozen after those steps. _object_overlay_fingerprint
            # covers every input _composite_objects_layer actually reads.
            qz = max(0.25, round(zoom * 4) / 4)
            render_key = (
                geo_overlay_key, qz,
                self.show_elevation_contour_var.get(),
                self.elevation_contour_var.get(), self.elevation_contour_width_var.get(),
                self.show_stamp_coverage_var.get(),
                self.show_objects_var.get(),
                self._object_overlay_fingerprint(Path(wd)) if self.show_objects_var.get() else None,
                tuple(self._highlighted_object_points),
            )
            render_cache = getattr(self, "_preview_render_cache", None)
            if render_cache is None:
                render_cache = self._preview_render_cache = {}
            cached_img = render_cache.pop(render_key, None)
            if cached_img is not None:
                # LRU: re-insert at the end (most recent).
                render_cache[render_key] = cached_img
                self._preview_render_meta = {
                    "img_w": cached_img.width,
                    "img_h": cached_img.height,
                    "base_kind": base_kind,
                }
                self._set_preview_image(cached_img)
                return

            # LANCZOS only for downsampling (the one case where the
            # resampling is quality-relevant); BICUBIC for upscaling --
            # LANCZOS is ~1.5x slower there and magnifies nothing.
            target_w = max(1, round(geo_img.width * qz))
            target_h = max(1, round(geo_img.height * qz))
            resample = Image.LANCZOS if qz < 1.0 else Image.BICUBIC
            base_thumb = geo_img.resize((target_w, target_h), resample)

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

            # Record what a viewport click needs to invert the transform:
            # the final displayed image size (PhotoImage is drawn 1:1 on
            # the canvas, so course->pixel uses viz._PLOT_RECT fractions
            # of exactly this size) and the base kind (LIDAR/full-frame
            # OSM previews aren't in the course [0, COURSE_SIZE_M] frame,
            # so picking is disabled on them -- same exclusion the
            # highlight/objects overlays already use).
            self._preview_render_meta = {
                "img_w": img.width,
                "img_h": img.height,
                "base_kind": base_kind,
            }

            render_cache[render_key] = img
            # LRU cap: at most 3 entries AND at most ~45MP of cached
            # RGBA (~180MB) -- a single 300%-zoom render is ~31MP, so
            # the pixel cap lets the expensive top levels stay cached
            # alone while bounding worst-case memory (an uncapped 3x3
            # entry LRU could hold ~380MB of 300%-zoom renders).
            def _cache_pixels() -> int:
                return sum(e.width * e.height for e in render_cache.values())
            while len(render_cache) > 3 or _cache_pixels() > 45_000_000:
                render_cache.pop(next(iter(render_cache)))

            self._set_preview_image(img)
        except Exception as e:
            self._preview_render_meta = None
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
