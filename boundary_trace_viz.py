"""
boundary_trace_viz.py

Standalone Tkinter tool for prototyping raster-to-vector boundary
tracing of a 1-bit target mask -- e.g. Illustrator's Image Trace, but
straight-line-segment only (no beziers): find the fewest points that
represent the mask's boundary shape within a tolerance.

Traces the EXTERIOR boundary of each connected foreground region, plus
the boundary of any fully-enclosed HOLE (a background region that
doesn't touch the image border). For a contour-band mask, a hole
corresponds to the band above (nested higher elevation), so exterior
ring = low boundary, hole ring(s) = high boundary -- but this tool
only does the tracing/simplification itself; it doesn't assume or
require that interpretation, and doesn't yet feed the result into the
fall-line feather placement rule (see feather_fill_viz.py) -- that's
the natural next step once this is validated on its own.

TWO STAGES, deliberately kept separate and both testable:
    1. RAW TRACE (`_moore_boundary_trace`): classic Moore-neighbor
       boundary-following on the raster, giving an exact, pixel-
       accurate ordered polygon (one point per boundary pixel -- lots
       of points, zero shape error).
    2. SIMPLIFY (`shapely.Polygon.simplify`, Douglas-Peucker): collapse
       the raw trace down to the fewest straight-line vertices whose
       shape stays within `tolerance` of the original. Already a
       project dependency (see terrain/cart_paths.py) -- no new
       library needed.

NOTE ON PRIOR ART IN THIS CODEBASE: terrain/contour_layers.py's own
docstring records that an earlier version of THIS PROJECT traced
explicit contour lines via skimage.measure.find_contours and later
deliberately dropped it (both the ring-tracing approach and the
skimage dependency) in favor of direct raster-mask filling, after a
confirmed compositing bias. That bias was specific to blending BETWEEN
two traced rings sharing identical geometry -- this tool traces a
SINGLE mask's own boundary for vectorization/visualization, not for
blending, so the same failure mode doesn't apply here. Still worth
knowing this project tried ring-tracing before and walked away from it
once already.

1 pixel = 1 world unit, matching the other visualizers in this
project.
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
from scipy import ndimage

try:
    from shapely.geometry import Point, Polygon
except ImportError as e:
    print(f"Couldn't import shapely ({e}). It's already a project dependency "
          "(see terrain/cart_paths.py) -- install it in this environment.")
    sys.exit(1)


DEFAULT_TOLERANCE = 1.5

# (row, col) offsets, clockwise starting from North -- standard 8-connectivity
# ordering for Moore-neighbor boundary tracing.
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

EXTERIOR_COLOR = (60, 140, 240)  # matches LOW_COLOR convention from feather_fill_viz.py
HOLE_COLOR = (240, 160, 60)       # matches HIGH_COLOR convention from feather_fill_viz.py
RAW_TRACE_COLOR = (120, 120, 120)


def _first_pixel(mask: np.ndarray) -> tuple[int, int] | None:
    """First True pixel in raster-scan (row-major) order, or None."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    i = np.lexsort((xs, ys))[0]
    return (int(ys[i]), int(xs[i]))


def _moore_boundary_trace(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Classic Moore-neighbor boundary trace. `start` MUST be a
    raster-scan-first pixel of the region being traced (guarantees the
    pixel immediately to its West is not part of the region, which
    fixes the initial backtrack direction). Returns an ordered,
    non-repeating list of (row, col) boundary pixels forming one
    closed loop.

    Known simplification (documented, not solved here -- consistent
    with how this project already flags tendril-tip/edge behavior
    elsewhere rather than over-engineering a fix): stops as soon as
    the traced position returns to `start`, without also checking
    entry direction. A region with a pixel-width-1 self-touching pinch
    could in principle close the loop early. Not handled specially.
    """
    r0, c0 = start
    boundary = [(r0, c0)]
    b_dir = DIRECTIONS.index((0, -1))  # came from the West
    current = (r0, c0)
    rows, cols = mask.shape

    for _ in range(mask.size * 4):  # safety cap, not expected to be hit
        found = False
        for i in range(1, 9):
            idx = (b_dir + i) % 8
            dr, dc = DIRECTIONS[idx]
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc]:
                b_dir = (idx + 4) % 8  # opposite direction -- new backtrack
                current = (nr, nc)
                found = True
                break
        if not found:
            break  # isolated single pixel, no neighbors
        if current == (r0, c0):
            break
        boundary.append(current)

    return boundary


def trace_and_simplify(
    target: np.ndarray, tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict]:
    """
    Trace every connected foreground region in `target`, plus any
    fully-enclosed hole within each, then simplify each ring with
    shapely's Douglas-Peucker `simplify`.

    Returns a list of dicts, one per foreground region:
        {
            "exterior_raw": [(x, z), ...],       pixel-accurate
            "exterior_simplified": [(x, z), ...], decimated
            "holes_raw": [[(x, z), ...], ...],
            "holes_simplified": [[(x, z), ...], ...],
        }
    Coordinates are (x, z) = (col, row), matching this project's other
    visualizers' 1px = 1m convention.
    """
    fg_labels, n_fg = ndimage.label(target)

    bg_mask = ~target
    bg_labels, n_bg = ndimage.label(bg_mask)
    border_labels = (
        set(np.unique(bg_labels[0, :])) | set(np.unique(bg_labels[-1, :])) |
        set(np.unique(bg_labels[:, 0])) | set(np.unique(bg_labels[:, -1]))
    )
    border_labels.discard(0)
    hole_label_ids = [l for l in range(1, n_bg + 1) if l not in border_labels]

    regions = []
    for fg_id in range(1, n_fg + 1):
        region_mask = fg_labels == fg_id
        start = _first_pixel(region_mask)
        if start is None:
            continue
        exterior_raw = _moore_boundary_trace(region_mask, start)
        exterior_xy = [(c, r) for r, c in exterior_raw]

        if len(exterior_xy) < 3:
            continue  # degenerate (single/double pixel) -- not a real polygon

        # Which holes actually belong to THIS foreground region -- test
        # containment of each hole's own start pixel against this
        # region's raw exterior polygon (already have it, no need to
        # simplify first just to test containment).
        exterior_poly_raw = Polygon(exterior_xy)
        holes_raw = []
        for hole_id in hole_label_ids:
            hole_mask = bg_labels == hole_id
            hole_start = _first_pixel(hole_mask)
            if hole_start is None:
                continue
            if not exterior_poly_raw.contains(Point(hole_start[1], hole_start[0])):
                continue
            hole_trace = _moore_boundary_trace(hole_mask, hole_start)
            hole_xy = [(c, r) for r, c in hole_trace]
            if len(hole_xy) >= 3:
                holes_raw.append(hole_xy)

        full_poly = Polygon(exterior_xy, holes_raw)
        simplified = full_poly.simplify(tolerance, preserve_topology=True)

        # simplify() on a Polygon-with-holes can in rare degenerate
        # cases (tolerance wiping out a hole entirely) return a bare
        # Polygon with no interiors -- guard rather than assume shape.
        ext_simplified = list(simplified.exterior.coords)[:-1]
        holes_simplified = [list(ring.coords)[:-1] for ring in simplified.interiors]

        regions.append({
            "exterior_raw": exterior_xy,
            "exterior_simplified": ext_simplified,
            "holes_raw": holes_raw,
            "holes_simplified": holes_simplified,
        })

    return regions


class BoundaryTraceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Boundary Trace Visualizer (raster-to-vector, prototype)")
        self.root.geometry("1350x900")
        self.root.minsize(1050, 700)

        self.target = None
        self.target_image = None
        self.regions: list[dict] = []
        self.tk_image = None

        self.tolerance = tk.DoubleVar(value=DEFAULT_TOLERANCE)
        self.show_raw = tk.BooleanVar(value=False)
        self.show_simplified = tk.BooleanVar(value=True)
        self.show_vertices = tk.BooleanVar(value=True)

        self.stats = tk.StringVar(value="No trace run yet")
        self.file_name = tk.StringVar(value="No image loaded")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=300)
        controls.pack(side="left", fill="y", padx=(0, 8))
        controls.pack_propagate(False)

        ttk.Label(
            controls, text="Boundary Trace", font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            controls, text="Raster-to-vector, straight segments only -- 1px = 1m",
            font=("TkDefaultFont", 9), foreground="#555",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Open 1-bit PNG...", command=self.open_image).pack(fill="x")
        ttk.Label(controls, textvariable=self.file_name, wraplength=280).pack(anchor="w", pady=(6, 14))

        self._separator(controls)
        ttk.Label(controls, text="SIMPLIFY", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Tolerance (m)", self.tolerance, 0.1, 100, increment=0.1)

        self._separator(controls)
        ttk.Checkbutton(
            controls, text="Show raw pixel trace (gray)", variable=self.show_raw, command=self.redraw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls, text="Show simplified boundary", variable=self.show_simplified, command=self.redraw,
        ).pack(anchor="w")
        ttk.Checkbutton(
            controls, text="Show vertices", variable=self.show_vertices, command=self.redraw,
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(controls, text="Trace + simplify", command=self.run_trace).pack(fill="x", pady=(4, 4))

        self._separator(controls)
        ttk.Label(controls, textvariable=self.stats, wraplength=280, justify="left").pack(anchor="w")

        ttk.Label(
            controls,
            text=("Blue = exterior boundary. Orange = hole boundary (a fully-"
                  "enclosed background region -- the shape you'd get from an "
                  "annular contour band around a higher region). Doesn't feed "
                  "into feather placement yet -- this only validates the trace."),
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
            self.regions = []
            self.file_name.set(f"{path}  ({img.width} x {img.height})")
            self.stats.set("No trace run yet")
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def run_trace(self):
        if self.target is None:
            messagebox.showwarning("No target", "Open a 1-bit PNG first.")
            return
        try:
            tolerance = float(self.tolerance.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid settings", "Tolerance isn't a valid number.")
            return

        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.regions = trace_and_simplify(self.target, tolerance)
            self._update_stats()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Trace failed", str(exc))
        finally:
            self.root.config(cursor="")

    def _update_stats(self):
        if not self.regions:
            self.stats.set("No regions found.")
            return
        lines = [f"Regions: {len(self.regions)}"]
        for i, r in enumerate(self.regions):
            raw_n = len(r["exterior_raw"])
            simp_n = len(r["exterior_simplified"])
            hole_summary = ", ".join(
                f"{len(hr)}->{len(hs)}" for hr, hs in zip(r["holes_raw"], r["holes_simplified"])
            )
            line = f"Region {i}: exterior {raw_n}->{simp_n} pts"
            if hole_summary:
                line += f"; holes {hole_summary}"
            lines.append(line)
        self.stats.set("\n".join(lines))

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
        mask_img = self.target_image.convert("L")
        white = Image.new("RGB", (w, h), (70, 70, 70))
        base.paste(white, mask=mask_img)

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        def draw_ring(points, color, closed=True):
            if len(points) < 2:
                return
            pts = points + [points[0]] if closed else points
            draw.line(pts, fill=(*color, 220), width=1)
            if self.show_vertices.get():
                for x, z in points:
                    draw.ellipse((x - 1.5, z - 1.5, x + 1.5, z + 1.5), fill=(*color, 255))

        for region in self.regions:
            if self.show_raw.get():
                draw_ring(region["exterior_raw"], RAW_TRACE_COLOR)
                for hole in region["holes_raw"]:
                    draw_ring(hole, RAW_TRACE_COLOR)
            if self.show_simplified.get():
                draw_ring(region["exterior_simplified"], EXTERIOR_COLOR)
                for hole in region["holes_simplified"]:
                    draw_ring(hole, HOLE_COLOR)

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
    app = BoundaryTraceApp(root)
    root.mainloop()