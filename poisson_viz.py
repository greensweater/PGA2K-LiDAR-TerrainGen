"""
circle_fill_visualizer.py

Standalone Tkinter tool for testing terrain/contour_layers.py's real
two-pass fill algorithm (_poisson_pack_band + _scatter_fill_remaining)
against an arbitrary 1-bit PNG target -- e.g. a hand-built "punched
circle" (a smaller circle removed from a larger one, aligned to one
side) to isolate tendril-tip/edge behavior without needing a real
heightmap or running the full generate-terrain pipeline.

This is a THIN WRAPPER, not a reimplementation: it imports the actual
functions from terrain/contour_layers.py directly, so results here are
exactly what a real run would do at the same parameters, with zero risk
of the visualizer's own logic drifting out of sync with the real
algorithm. Run this from the project root (or with the project root on
PYTHONPATH) so `terrain.contour_layers` is importable, the same way
every other script in this project already expects.

1 pixel = 1 meter -- the loaded PNG's own pixel grid is used directly
as both the heightmap's shape and the world coordinate system (bounds
= [0, width] x [0, height]). There's no real elevation data here (this
tool tests PLACEMENT, not terrain fitting), so a flat dummy heights
array is used -- stamp VALUES computed during this test are meaningless
placeholders, only stamp POSITION/RADIUS/BRUSH matter for what this
tool is actually checking.
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# Run from the project root (same expectation every other script here
# already has) -- if terrain.contour_layers isn't importable, fail
# loudly and immediately rather than deep inside a button callback.
try:
    from terrain.bounding_box import BoundingBox
    from terrain.contour_layers import (
        DEFAULT_FILL_BRUSH, DEFAULT_MIN_RADIUS_M, DEFAULT_MAX_RADIUS_M, DEFAULT_RADIUS_STEP_RATIO,
        DEFAULT_EDGE_DISTANCE_M, DEFAULT_SMOOTHING_BRUSH, DEFAULT_SMOOTHING_MIN_RADIUS_M,
        DEFAULT_CRUMB_SCATTER_MULTIPLIER, DEFAULT_SMOOTH_CLAIM_FRACTION, DEFAULT_CANDIDATES_PER_RADIUS,
        DEFAULT_RANDOM_SEED, _poisson_pack_band, _scatter_fill_remaining,
    )
except ImportError as e:
    print(f"Couldn't import terrain.contour_layers ({e}).\n"
          "Run this from the project root, or put the project root on PYTHONPATH -- "
          "same expectation every other script in this project already has.")
    sys.exit(1)


class CircleFillApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Contour Layers Fill Visualizer (real algorithm)")
        self.root.geometry("1350x900")
        self.root.minsize(1050, 700)

        self.target = None  # loaded 1-bit mask, (rows, cols) bool
        self.target_image = None
        self.pass1_stamps = []
        self.pass2_stamps = []
        self.tk_image = None
        self.running = False

        # PASS 1 -- fast poisson pack (see _poisson_pack_band)
        self.fill_brush = tk.IntVar(value=DEFAULT_FILL_BRUSH)
        self.max_radius = tk.DoubleVar(value=DEFAULT_MAX_RADIUS_M)
        self.min_radius = tk.DoubleVar(value=DEFAULT_MIN_RADIUS_M)
        self.radius_step_ratio = tk.DoubleVar(value=DEFAULT_RADIUS_STEP_RATIO)
        self.edge_distance_m = tk.DoubleVar(value=DEFAULT_EDGE_DISTANCE_M)
        self.candidates_per_radius = tk.IntVar(value=DEFAULT_CANDIDATES_PER_RADIUS)
        self.seed = tk.IntVar(value=DEFAULT_RANDOM_SEED)

        # PASS 2 -- crumb scatter (see _scatter_fill_remaining)
        self.run_pass2 = tk.BooleanVar(value=True)
        self.smoothing_brush = tk.IntVar(value=DEFAULT_SMOOTHING_BRUSH)
        self.smoothing_min_radius = tk.DoubleVar(value=DEFAULT_SMOOTHING_MIN_RADIUS_M)
        self.smooth_ratio = tk.DoubleVar(value=DEFAULT_CRUMB_SCATTER_MULTIPLIER)
        self.smooth_claim_fraction = tk.DoubleVar(value=DEFAULT_SMOOTH_CLAIM_FRACTION)

        self.show_pass1 = tk.BooleanVar(value=True)
        self.show_pass2 = tk.BooleanVar(value=True)

        self.stamp_count = tk.StringVar(value="Pass 1: 0   Pass 2: 0   Total: 0")
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
            controls, text="Contour Layers Fill", font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            controls, text="Real terrain.contour_layers algorithm -- 1px = 1m",
            font=("TkDefaultFont", 9), foreground="#555",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Open 1-bit PNG...", command=self.open_image).pack(fill="x")
        ttk.Label(controls, textvariable=self.file_name, wraplength=280).pack(anchor="w", pady=(6, 14))

        self._separator(controls)
        ttk.Label(controls, text="PASS 1 -- poisson pack", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Fill brush", self.fill_brush, 1, 200)
        self._add_spinbox(controls, "Max radius (m)", self.max_radius, 1, 500)
        self._add_spinbox(controls, "Min radius (m)", self.min_radius, 1, 500)
        self._add_spinbox(controls, "Radius step ratio", self.radius_step_ratio, 0.01, 0.99, increment=0.05)
        self._add_spinbox(controls, "Edge distance (m)", self.edge_distance_m, 0, 100)
        self._add_spinbox(controls, "Candidates / tier", self.candidates_per_radius, 100, 200000)
        self._add_spinbox(controls, "Random seed", self.seed, 0, 2_147_483_647)

        self._separator(controls)
        ttk.Checkbutton(
            controls, text="Run PASS 2 (crumb scatter)", variable=self.run_pass2,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(controls, text="PASS 2 -- crumb scatter", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Smoothing brush", self.smoothing_brush, 1, 200)
        self._add_spinbox(controls, "Smoothing min radius (m)", self.smoothing_min_radius, 0.5, 100)
        self._add_spinbox(controls, "Smooth ratio (x min)", self.smooth_ratio, 1, 20, increment=0.5)
        self._add_spinbox(controls, "Smooth eat (claim frac)", self.smooth_claim_fraction, 0.01, 1.0, increment=0.05)

        self._separator(controls)
        ttk.Checkbutton(
            controls, text="Show pass 1 stamps (red)", variable=self.show_pass1, command=self.redraw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls, text="Show pass 2 stamps (blue)", variable=self.show_pass2, command=self.redraw,
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(controls, text="Run fill", command=self.run_fill).pack(fill="x", pady=(4, 4))
        ttk.Button(controls, text="Clear stamps", command=self.clear_stamps).pack(fill="x")

        self._separator(controls)
        ttk.Label(controls, textvariable=self.stamp_count, font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(4, 2))
        ttk.Label(controls, textvariable=self.coverage, font=("TkDefaultFont", 11, "bold")).pack(anchor="w")

        ttk.Label(
            controls,
            text=("The target is white/nonzero pixels. Black pixels are outside the fill area. "
                  "Try a smaller circle punched out of a larger one, aligned to one side, to see "
                  "how the algorithm handles a thin tendril tip."),
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
            title="Open target PNG", filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
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
            self.pass1_stamps = []
            self.pass2_stamps = []
            self.file_name.set(f"{path}  ({img.width} x {img.height})")
            self._update_stats()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def clear_stamps(self):
        self.pass1_stamps = []
        self.pass2_stamps = []
        self._update_stats()
        self.redraw()

    def run_fill(self):
        if self.target is None:
            messagebox.showwarning("No target", "Open a 1-bit PNG first.")
            return
        if self.running:
            return

        try:
            fill_brush = int(self.fill_brush.get())
            max_r = float(self.max_radius.get())
            min_r = float(self.min_radius.get())
            step_ratio = float(self.radius_step_ratio.get())
            edge_distance = float(self.edge_distance_m.get())
            candidates = int(self.candidates_per_radius.get())
            seed = int(self.seed.get())
            smoothing_brush = int(self.smoothing_brush.get())
            smoothing_min_radius = float(self.smoothing_min_radius.get())
            smooth_ratio = float(self.smooth_ratio.get())
            smooth_claim_fraction = float(self.smooth_claim_fraction.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid settings", "One or more fields aren't valid numbers.")
            return

        if min_r > max_r:
            messagebox.showerror("Invalid settings", "Min radius must not exceed max radius.")
            return
        if not (0.0 < step_ratio < 1.0):
            messagebox.showerror("Invalid settings", "Radius step ratio must be between 0 and 1.")
            return

        self.running = True
        self.root.config(cursor="watch")
        self.root.update_idletasks()

        try:
            # 1px = 1m: the target mask's own shape IS the heightmap shape,
            # and bounds span exactly [0, width] x [0, height] in the same
            # units. No real elevation data exists in this test, so a flat
            # dummy heights array is used -- fitted VALUES are meaningless
            # here (this tool tests placement, not terrain fitting), only
            # position/radius/brush matter for what it's actually checking.
            h, w = self.target.shape
            heights = np.zeros((h, w), dtype=np.float64)
            bounds = BoundingBox(min_x=0.0, min_z=0.0, max_x=float(w), max_z=float(h))

            rng = np.random.default_rng(seed)
            pass1_stamps, crumbs = _poisson_pack_band(
                self.target, heights, bounds, fill_brush, max_r, min_r, step_ratio,
                edge_distance, candidates, rng,
            )

            pass2_stamps = []
            if self.run_pass2.get() and crumbs.any():
                crumb_radius = smoothing_min_radius * smooth_ratio
                pass2_stamps = _scatter_fill_remaining(
                    crumbs, heights, bounds, smoothing_brush, crumb_radius,
                    claim_radius_fraction=smooth_claim_fraction,
                )

            self.pass1_stamps = pass1_stamps
            self.pass2_stamps = pass2_stamps
            self._update_stats()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Fill failed", str(exc))
        finally:
            self.running = False
            self.root.config(cursor="")

    def _update_stats(self):
        if self.target is None:
            self.stamp_count.set("Pass 1: 0   Pass 2: 0   Total: 0")
            self.coverage.set("Coverage: 0.0%")
            return

        target_pixels = int(self.target.sum())
        all_stamps = self.pass1_stamps + self.pass2_stamps
        if not all_stamps:
            covered = 0
        else:
            h, w = self.target.shape
            coverage_mask = np.zeros((h, w), dtype=bool)
            for s in all_stamps:
                x, z, r = s.x, s.z, s.radius
                x0 = max(0, int(x - r))
                x1 = min(w, int(x + r) + 1)
                y0 = max(0, int(z - r))
                y1 = min(h, int(z + r) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                region_y, region_x = np.ogrid[y0:y1, x0:x1]
                circle = (region_x - x) ** 2 + (region_y - z) ** 2 <= r * r
                coverage_mask[y0:y1, x0:x1] |= circle
            covered = int(np.logical_and(coverage_mask, self.target).sum())

        percent = 100.0 * covered / target_pixels if target_pixels else 0.0
        self.stamp_count.set(
            f"Pass 1: {len(self.pass1_stamps):,}   Pass 2: {len(self.pass2_stamps):,}   "
            f"Total: {len(self.pass1_stamps) + len(self.pass2_stamps):,}"
        )
        self.coverage.set(f"Coverage: {percent:.2f}%")

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

        if self.show_pass1.get():
            for s in self.pass1_stamps:
                r = s.radius
                draw.ellipse(
                    (s.x - r, s.z - r, s.x + r, s.z + r),
                    outline=(220, 40, 40, 210), width=max(1, min(3, int(r) // 8 or 1)),
                )
        if self.show_pass2.get():
            for s in self.pass2_stamps:
                r = s.radius
                draw.ellipse(
                    (s.x - r, s.z - r, s.x + r, s.z + r),
                    outline=(60, 120, 230, 190), width=max(1, min(3, int(r) // 8 or 1)),
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
    app = CircleFillApp(root)
    root.mainloop()
