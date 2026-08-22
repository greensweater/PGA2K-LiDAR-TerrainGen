"""
rect_fill_viz.py

Standalone Tkinter tool for prototyping a tiered greedy rectangular fill
against an arbitrary 1-bit PNG target -- e.g. a hand-built "punched
circle" or an irregular blob, to see how the algorithm handles thin
tendrils, concave corners, etc. before it's lifted into
terrain/contour_layers.py as the real per-band fill.

UNLIKE poisson_viz.py, this is NOT a thin wrapper around an existing
real function -- there is no greedy-rectangle fill in contour_layers.py
yet. The algorithm lives in this file (see `greedy_rect_fill` below),
written so it can be copy-pasted into contour_layers.py largely as-is
once it's validated here. It constructs real `terrain.stamp.Stamp`
objects so brush/scale/value semantics match production exactly.

ALGORITHM (multi-pass, deterministic, no randomness):
    Runs a sequence of size tiers derived from `target_dimension_m` by
    repeated halving down to 1 (e.g. 16 -> [16, 8, 4, 2, 1]). Within
    each tier, repeatedly find the single largest axis-aligned
    all-target rectangle in the "uncovered" mask whose shorter side is
    >= the tier's size (classic largest-rectangle-in-binary-matrix,
    via per-row histogram + monotonic stack, with the size floor built
    into candidate generation itself -- see `_largest_rectangle`),
    place a type72 stamp sized so its PLATEAU exactly covers that
    rectangle, and repeat until no rectangle at that tier remains.
    Then drop to the next smaller tier and continue on the same
    shrinking mask. This prioritizes big, cheap rectangles for large
    flat regions first, rather than pure global-largest-by-area (which
    can interleave a big region's coverage with unrelated thin strips
    just because they happen to have similar area). It also tends to
    keep neighboring rectangle sizes within roughly 2x of each other,
    which incidentally bounds worst-case seam step size at boundaries
    (see the no-falloff / hard-edge note below).

    Overlap between stamps IS allowed (a placed stamp can cover ground
    an earlier stamp already covered); only the *uncovered* mask
    shrinks each iteration, never the target mask itself. `max_stamps`
    is a safety valve (across all tiers combined) for pathological
    masks with a huge number of tiny leftover holes.

TYPE72 GEOMETRY (measured, not derived here):
    512x512 texture, flat weight=1.0 plateau, hard 0->255 step at a 6px
    black border -- NOT a soft falloff. Plateau is exactly 500/512 of
    nominal stamp scale. To make a stamp's plateau land exactly on a
    found rectangle of world-unit size (w, h), the stamp's scale must
    be set to (w/2)/PLATEAU_FRACTION x (h/2)/PLATEAU_FRACTION -- i.e.
    slightly larger than the rectangle itself, so the black border
    falls just outside it and the flat plateau exactly fills it.

1 pixel = 1 meter -- same convention as poisson_viz.py. No real
elevation data here (this tests placement, not fitting), so stamp
VALUE is a meaningless placeholder; only position/scale/brush matter.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:
    from terrain.stamp import Stamp, TOOL_FLATTEN
except ImportError as e:
    print(f"Couldn't import terrain.stamp ({e}).\n"
          "Run this from the project root, or put the project root on PYTHONPATH -- "
          "same expectation every other script in this project already has.")
    sys.exit(1)


DEFAULT_FILL_BRUSH = 72
DEFAULT_TARGET_DIMENSION_M = 16
DEFAULT_MAX_STAMPS = 20_000

# Measured, hard 0->255 step at a 6px border on a 512px texture -- not a
# falloff ramp. See module docstring.
PLATEAU_FRACTION = 500.0 / 512.0


def _largest_rectangle(mask: np.ndarray, min_dimension: float = 1.0) -> tuple[int, int, int, int]:
    """
    Find the largest-AREA all-True axis-aligned rectangle in a 2D bool
    array, subject to both sides being >= min_dimension. Classic
    per-row histogram + monotonic stack, with the constraint applied
    AT candidate-generation time (every stack pop is a candidate; only
    ones meeting the size floor are tracked) rather than after the
    fact -- rejecting the single unconstrained global max post hoc and
    excluding its whole bounding box from further search is far too
    destructive: that box is chosen to maximize area, not to be
    minimal, so it can sweep through columns that also support a
    genuinely valid smaller rectangle a row or two away and wipe it
    out as collateral damage before it's ever considered.

    Returns (top, left, height, width) in pixel/cell units, or
    (0, 0, 0, 0) if no qualifying rectangle exists.
    """
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int32)

    best_area = 0
    best = (0, 0, 0, 0)

    for r in range(rows):
        row = mask[r]
        heights = np.where(row, heights + 1, 0)

        stack = []  # list of (start_col, height)
        for c in range(cols + 1):
            h = int(heights[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] > h:
                idx, sh = stack.pop()
                width = c - idx
                if sh >= min_dimension and width >= min_dimension:
                    area = sh * width
                    if area > best_area:
                        best_area = area
                        best = (r - sh + 1, idx, sh, width)
                start = idx
            stack.append((start, h))

    return best


def _dimension_tiers(target_dimension_m: int) -> list[int]:
    """
    16 -> [16, 8, 4, 2, 1]. Halves down to 1 regardless of whether
    target_dimension_m is a power of two (e.g. 10 -> [10, 5, 2, 1]).
    """
    tiers = []
    t = max(1, int(target_dimension_m))
    while t >= 1:
        tiers.append(t)
        if t == 1:
            break
        t //= 2
    return tiers


def greedy_rect_fill(
    target: np.ndarray,
    heights: np.ndarray,
    bounds_min_x: float,
    bounds_min_z: float,
    fill_brush: int = DEFAULT_FILL_BRUSH,
    target_dimension_m: int = DEFAULT_TARGET_DIMENSION_M,
    max_stamps: int = DEFAULT_MAX_STAMPS,
) -> list[tuple[int, Stamp]]:
    """
    Multi-pass greedy rectangular fill. Runs a sequence of tiers
    (target_dimension_m, halved down to 1 -- e.g. 16 -> [16,8,4,2,1]);
    within each tier, repeatedly place the largest all-target
    rectangle with both sides >= the tier's size until none remain at
    that size, THEN drop to the next smaller tier and continue on the
    same shrinking `remaining` mask. This prioritizes big, cheap
    rectangles for large flat regions before spending stamp budget on
    detail, rather than pure global-largest-by-area (which can
    interleave a big flat region's coverage with unrelated thin
    strips just because they happen to have comparable area).

    `heights` is used only to compute a placeholder stamp value at
    each rectangle's center (same meaningless-placeholder convention
    as poisson_viz.py -- this tool tests placement, not fitting).

    Returns a flat list of (tier, Stamp) pairs, largest tier first --
    the tier tag is purely so this visualizer can color-code passes;
    a production lift into contour_layers.py that doesn't need that
    can drop the tier and keep just the Stamp.
    """
    remaining = target.copy()
    stamps: list[tuple[int, Stamp]] = []

    for tier in _dimension_tiers(target_dimension_m):
        while len(stamps) < max_stamps:
            top, left, h, w = _largest_rectangle(remaining, tier)
            if h == 0 or w == 0:
                # Nothing left at this tier -- drop to the next
                # smaller one rather than stopping the whole fill.
                break

            remaining[top:top + h, left:left + w] = False

            cx = bounds_min_x + left + w / 2.0
            cz = bounds_min_z + top + h / 2.0
            scale_x = (w / 2.0) / PLATEAU_FRACTION
            scale_z = (h / 2.0) / PLATEAU_FRACTION

            row = int(min(max(cz - bounds_min_z, 0), heights.shape[0] - 1))
            col = int(min(max(cx - bounds_min_x, 0), heights.shape[1] - 1))
            value = float(heights[row, col])

            stamps.append((tier, Stamp(
                x=cx, z=cz,
                scale_x=scale_x, scale_z=scale_z,
                value=value, brush=fill_brush,
                rotation=0.0, tool=TOOL_FLATTEN,
            )))

    return stamps


def tier_colors(tiers: list[int]) -> dict[int, tuple[int, int, int]]:
    """
    One distinct, stable color per tier size, largest tier warm (red)
    through smallest tier cool (violet) -- ordered by position in the
    tiers list (largest first) rather than raw tier value, so this
    stays evenly spaced regardless of how many tiers there are.
    """
    n = max(1, len(tiers))
    colors = {}
    for i, tier in enumerate(tiers):
        hue = 0.0 if n == 1 else (i / (n - 1)) * 0.8  # red -> violet
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        colors[tier] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


class RectFillApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rectangular Fill Visualizer (prototype -- tiered greedy)")
        self.root.geometry("1350x900")
        self.root.minsize(1050, 700)

        self.target = None  # loaded 1-bit mask, (rows, cols) bool
        self.target_image = None
        self.stamps: list[tuple[int, Stamp]] = []
        self.tier_colors: dict[int, tuple[int, int, int]] = {}
        self.tk_image = None
        self.running = False

        self.fill_brush = tk.IntVar(value=DEFAULT_FILL_BRUSH)
        self.target_dimension_m = tk.IntVar(value=DEFAULT_TARGET_DIMENSION_M)
        self.max_stamps = tk.IntVar(value=DEFAULT_MAX_STAMPS)

        self.show_stamps = tk.BooleanVar(value=True)

        self.stamp_count = tk.StringVar(value="Stamps: 0")
        self.coverage = tk.StringVar(value="Coverage: 0.0%")
        self.file_name = tk.StringVar(value="No image loaded")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=300)
        controls.pack(side="left", fill="y", padx=(0, 8))
        controls.pack_propagate(False)

        ttk.Label(
            controls, text="Rectangular Fill", font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            controls, text="Tiered greedy largest-rectangle, type72 -- 1px = 1m",
            font=("TkDefaultFont", 9), foreground="#555",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Open 1-bit PNG...", command=self.open_image).pack(fill="x")
        ttk.Label(controls, textvariable=self.file_name, wraplength=280).pack(anchor="w", pady=(6, 14))

        self._separator(controls)
        ttk.Label(controls, text="FILL", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Fill brush", self.fill_brush, 1, 200)
        self._add_spinbox(controls, "Target dimension (px)", self.target_dimension_m, 1, 512)
        self._add_spinbox(controls, "Max stamps (safety cap)", self.max_stamps, 100, 200_000, increment=100)

        self._separator(controls)
        ttk.Checkbutton(
            controls, text="Show stamps (solid fill)", variable=self.show_stamps, command=self.redraw,
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(controls, text="Run fill", command=self.run_fill).pack(fill="x", pady=(4, 4))
        ttk.Button(controls, text="Clear stamps", command=self.clear_stamps).pack(fill="x")

        self._separator(controls)
        ttk.Label(controls, textvariable=self.stamp_count, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 2))
        ttk.Label(controls, textvariable=self.coverage, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        self._separator(controls)
        ttk.Label(controls, text="TIER LEGEND", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.legend_frame = ttk.Frame(controls)
        self.legend_frame.pack(fill="x", pady=(2, 0))

        ttk.Label(
            controls,
            text=("The target is white/nonzero pixels. Black pixels are outside the fill area. "
                  "Rectangles are drawn as solid overlapping fills -- darker patches show where "
                  "stamps overlap, which is expected and allowed."),
            wraplength=280, justify="left",
        ).pack(anchor="w", pady=(18, 0))

        view_frame = ttk.Frame(outer, relief="sunken", borderwidth=1)
        view_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(view_frame, background="#303030", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def _separator(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

    def _add_spinbox(self, parent, label, variable, low, high, increment=1):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label).pack(side="left")
        spin = ttk.Spinbox(
            frame, from_=low, to=high, textvariable=variable, width=10, increment=increment,
        )
        spin.pack(side="right")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Open 1-bit target PNG",
            filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("L")
            arr = np.asarray(img) > 0
            if not arr.any():
                raise ValueError("The image contains no nonzero pixels.")
            self.target = arr
            self.target_image = img
            self.stamps = []
            self.tier_colors = {}
            self.file_name.set(f"{path}  ({img.width} x {img.height})")
            self._update_stats()
            self._update_legend()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def clear_stamps(self):
        self.stamps = []
        self.tier_colors = {}
        self._update_stats()
        self._update_legend()
        self.redraw()

    def run_fill(self):
        if self.target is None:
            messagebox.showwarning("No target", "Open a 1-bit PNG first.")
            return
        if self.running:
            return

        try:
            fill_brush = int(self.fill_brush.get())
            target_dimension_m = int(self.target_dimension_m.get())
            max_stamps = int(self.max_stamps.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid settings", "One or more fields aren't valid numbers.")
            return

        self.running = True
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            # 1px = 1m: the target mask's own shape IS the heightmap shape,
            # bounds start at (0, 0). No real elevation data in this test,
            # so stamp VALUE is a meaningless placeholder -- only
            # position/scale/brush matter for what this tool is checking.
            h, w = self.target.shape
            heights = np.zeros((h, w), dtype=np.float64)

            self.stamps = greedy_rect_fill(
                self.target, heights, 0.0, 0.0,
                fill_brush=fill_brush,
                target_dimension_m=target_dimension_m,
                max_stamps=max_stamps,
            )
            self.tier_colors = tier_colors(_dimension_tiers(target_dimension_m))
            self._update_stats()
            self._update_legend()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Fill failed", str(exc))
        finally:
            self.running = False
            self.root.config(cursor="")

    def _update_stats(self):
        if self.target is None:
            self.stamp_count.set("Stamps: 0")
            self.coverage.set("Coverage: 0.0%")
            return

        target_pixels = int(self.target.sum())
        if not self.stamps:
            covered = 0
        else:
            h, w = self.target.shape
            coverage_mask = np.zeros((h, w), dtype=bool)
            for _tier, s in self.stamps:
                plateau_hw = s.scale_x * PLATEAU_FRACTION
                plateau_hh = s.scale_z * PLATEAU_FRACTION
                x0 = max(0, int(round(s.x - plateau_hw)))
                x1 = min(w, int(round(s.x + plateau_hw)))
                y0 = max(0, int(round(s.z - plateau_hh)))
                y1 = min(h, int(round(s.z + plateau_hh)))
                if x0 >= x1 or y0 >= y1:
                    continue
                coverage_mask[y0:y1, x0:x1] = True
            covered = int(np.logical_and(coverage_mask, self.target).sum())

        percent = 100.0 * covered / target_pixels if target_pixels else 0.0
        self.stamp_count.set(f"Stamps: {len(self.stamps):,}")
        self.coverage.set(f"Coverage: {percent:.2f}%")

    def _update_legend(self):
        for child in self.legend_frame.winfo_children():
            child.destroy()

        counts: dict[int, int] = {}
        for tier, _s in self.stamps:
            counts[tier] = counts.get(tier, 0) + 1

        for tier in sorted(counts, reverse=True):
            color = self.tier_colors.get(tier, (150, 150, 150))
            row = ttk.Frame(self.legend_frame)
            row.pack(fill="x", pady=1)
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=1,
                                highlightbackground="#888")
            swatch.pack(side="left", padx=(0, 6))
            swatch.create_rectangle(0, 0, 14, 14, fill="#%02x%02x%02x" % color, outline="")
            ttk.Label(row, text=f"{tier}px  ({counts[tier]:,} stamps)").pack(side="left")

    def redraw(self):
        self.canvas.delete("all")

        if self.target_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
                text="Open a 1-bit PNG", fill="white", font=("TkDefaultFont", 16),
            )
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        h, w = self.target.shape

        base = Image.new("RGB", (w, h), (30, 30, 30))
        mask = self.target_image.convert("L")
        white = Image.new("RGB", (w, h), (245, 245, 245))
        base.paste(white, mask=mask)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        if self.show_stamps.get():
            for tier, s in self.stamps:
                plateau_hw = s.scale_x * PLATEAU_FRACTION
                plateau_hh = s.scale_z * PLATEAU_FRACTION
                color = self.tier_colors.get(tier, (150, 150, 150))
                # Solid fill, no outline -- overlapping stamps compound
                # into visibly darker/more-opaque patches by design.
                #
                # PIL treats rectangle() bounds as pixel-inclusive on
                # both ends, but plateau_hw/plateau_hh were computed to
                # match numpy's half-open slicing convention (the same
                # coordinates used to clear `remaining`) -- passed
                # straight through, that draws one extra column/row on
                # the right and bottom of every stamp. Pull the far
                # edge in by 1px to match the actual covered cells.
                draw.rectangle(
                    (s.x - plateau_hw, s.z - plateau_hh,
                     s.x + plateau_hw - 1, s.z + plateau_hh - 1),
                    fill=(*color, 150),
                )

        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

        scale = min(cw / w, ch / h)
        display_w = max(1, int(w * scale))
        display_h = max(1, int(h * scale))
        display = base.resize((display_w, display_h), Image.Resampling.NEAREST) if (display_w, display_h) != (w, h) else base

        self.tk_image = ImageTk.PhotoImage(display)
        x = (cw - display_w) // 2
        y = (ch - display_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = RectFillApp(root)
    root.mainloop()