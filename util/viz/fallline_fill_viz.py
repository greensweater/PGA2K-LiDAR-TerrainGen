"""
fallline_fill_viz.py

Standalone Tkinter tool: fill the INTERIOR of a real traced boundary
(see boundary_trace_viz.trace_and_simplify) with real type72 solid
rotated Stamp objects, one per boundary edge -- no elevation semantics
("high"/"low") involved at all. Every ring the shape has (the outer
boundary, and any holes) is just a wall; for each edge of every ring,
shoot a stamp straight into the shape's interior until it hits
whatever wall is opposite, wherever that happens to be. A plain blob
with no hole works exactly the same way as an annulus -- each edge's
ray just travels across to the opposite side of the SAME ring instead
of to a different one.

EVERY REAL EDGE GETS A STAMP -- no fallback bucket for "too short" or
"no wall found" anymore. Width is always exactly |p1 - p0|; length
extends toward the opposite wall where possible, floored at
`min_length` where it can't (a wall too close, or nothing found at
all within max_search_distance). Earlier versions of this tool skipped
placing anything in those cases and flagged it instead -- but a real
boundary edge with literally nothing covering it is invisible ground
truth, which is worse than a minimum-length stamp. The only thing that
still gets flagged (not placed) is a genuinely degenerate zero-width
segment (duplicate consecutive points) -- not a tunable threshold,
just numerically meaningless to build a stamp from.

PER EDGE (p0, p1) on ANY ring, all rings walked as CLOSED loops
(wraparound edge included):
    width  = |p1 - p0|
    dir    = normalize(p1 - p0)
    perp   = dir rotated by a FIXED sign based on which ring this is --
             +1 for the exterior ring, -1 for every hole ring. Not a
             per-segment raster probe: verified that the exterior ring
             and every hole ring, though independently traced (each as
             its own "positive" Moore-traced region) and therefore
             sharing the SAME winding sign under a plain shoelace
             computation, need OPPOSITE signs to get the correct
             "which side is real material" answer -- because a hole's
             self-interior IS the hole, so real material is on the far
             side. Checked against every previously hand-verified
             interior_perp (all 4 rectangle-exterior edges, the annulus
             hole edge) and it reproduces all of them exactly. This
             also removes the corner-adjacent fragility a raster probe
             had near tight/concave features.
    length = the minimum safe distance across `width_samples` points
             spanning the segment (not just its midpoint) -- a single
             centerline ray only constrains the stamp's own centerline,
             not its actual rectangular footprint, and a nearby angled
             wall can be much closer to the stamp's real corner than
             its center is. Samples that find no wall are excluded
             rather than aborting the segment; if none find one, or
             the result is under min_length, length is min_length.

    Placement: center = midpoint + perp * (length / 2) -- translated
    off the edge by half the stamp's own length so the PLATEAU's near
    edge lands exactly on the boundary edge. rotation = degrees(atan2(
    perp_x, perp_z)) -- the same formula/axis convention
    terrain/cart_paths.py already uses in production.

1 pixel = 1 world unit. No real elevation data -- placement geometry
only, same convention as every other visualizer in this project.
"""

import math
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:
    from terrain.stamp import Stamp, TOOL_FLATTEN
except ImportError as e:
    print(f"Couldn't import terrain.stamp ({e}).\n"
          "Run this from the project root, or put the project root on PYTHONPATH.")
    sys.exit(1)

try:
    from util.viz.boundary_trace_viz import trace_and_simplify, DEFAULT_TOLERANCE
except ImportError as e:
    print(f"Couldn't import boundary_trace_viz ({e}).\n"
          "This tool reuses trace_and_simplify rather than re-deriving it -- "
          "run from the same directory as boundary_trace_viz.py.")
    sys.exit(1)


DEFAULT_FILL_BRUSH = 72
DEFAULT_MAX_SEARCH_DISTANCE = 400.0
DEFAULT_WIDTH_SAMPLES = 5

# Measured, hard 0->255 step at a 6px border on a 512px texture -- not
# a falloff ramp. Same constant as rect_fill_viz.py; redefined here
# rather than imported since rect_fill_viz.py isn't otherwise a
# dependency of this tool and duplicating one float is cheaper than
# coupling to an unrelated file for it.
PLATEAU_FRACTION = 500.0 / 512.0

OUTER_RING_COLOR = (60, 140, 240)
INNER_RING_COLOR = (240, 160, 60)
FALLBACK_COLOR = (220, 60, 60)
STAMP_COLOR = (90, 200, 140)


@dataclass
class EdgeResult:
    """One entry per boundary edge, in strict iteration order -- always
    present whether or not a stamp was placed, so a UI can index
    directly into 'the Nth edge walked' regardless of outcome."""
    ring_index: int      # 0 = exterior ring, 1+ = hole rings
    ring_kind: str        # "exterior" or "hole N"
    edge_index: int       # index within that ring
    p0: tuple[float, float]
    p1: tuple[float, float]
    stamp: "Stamp | None"
    fallback_reason: "str | None"


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _scale(a, s):
    return (a[0] * s, a[1] * s)


def _length(a):
    return math.hypot(a[0], a[1])


def _normalize(a):
    ln = _length(a)
    if ln < 1e-9:
        return (0.0, 0.0)
    return (a[0] / ln, a[1] / ln)


def _rotate90(a, sign):
    return (-sign * a[1], sign * a[0])


def _ray_edges_intersection(origin, direction, edges, max_distance):
    """
    Nearest intersection of the ray (origin + t*direction, t > 0) with
    any segment in `edges` (a flat list of (p0, p1) pairs -- may span
    multiple rings), within max_distance. Returns t or None.
    """
    best_t = None
    ox, oz = origin
    dx, dz = direction

    for p0, p1 in edges:
        ex, ez = p1[0] - p0[0], p1[1] - p0[1]

        denom = dx * ez - dz * ex
        if abs(denom) < 1e-9:
            continue

        qx, qz = p0[0] - ox, p0[1] - oz
        t = (qx * ez - qz * ex) / denom
        u = (qx * dz - qz * dx) / denom
        if t > 1e-6 and 0.0 <= u <= 1.0 and t <= max_distance:
            if best_t is None or t < best_t:
                best_t = t

    return best_t


def _ring_edges(ring: list[tuple[float, float]]) -> list[tuple[tuple, tuple]]:
    n = len(ring)
    return [(ring[i], ring[(i + 1) % n]) for i in range(n)]


DEFAULT_MIN_LENGTH = 1.0


def build_interior_fill_stamps(
    region: dict,
    target_mask: np.ndarray,
    fill_brush: int = DEFAULT_FILL_BRUSH,
    max_search_distance: float = DEFAULT_MAX_SEARCH_DISTANCE,
    width_samples: int = DEFAULT_WIDTH_SAMPLES,
    min_length: float = DEFAULT_MIN_LENGTH,
) -> list[EdgeResult]:
    """
    One EdgeResult for EVERY boundary edge of EVERY ring in `region`
    (exterior + all holes), in strict iteration order -- no elevation
    semantics, see module docstring. Every real segment gets a stamp:
    width is exactly |p1 - p0|, length is however far it can extend
    toward the opposite wall, floored at `min_length` if the ray-cast
    comes up short or finds nothing at all. The only EdgeResult with
    stamp=None is a genuinely degenerate (duplicate-point) segment --
    not a tunable threshold, just numerically meaningless to build a
    stamp from.

    INTERIOR DIRECTION comes from the ring's WINDING ORDER, not a
    per-segment raster probe. Both the exterior ring and every hole
    ring, when traced independently via Moore-neighbor tracing (each
    as its own "positive" region -- a hole is traced as the boundary
    of its own True-region on the inverted mask, not as a reversed
    view of the true polygon), come out with the SAME winding sign
    (verified: shoelace-positive/CCW for both). But the correct
    "which side is real material" rule is OPPOSITE between them --
    for holes, the traced self-interior IS the hole, so real material
    is on the far side. This was confirmed against every previously
    hand-verified interior_perp (all 4 rectangle-exterior edges plus
    the annulus hole edge): a FIXED sign, +1 for the exterior ring and
    -1 for every hole ring, reproduces all of them exactly -- no
    raster lookup needed, and nothing left that can fail near a tight
    corner the way probing could.

    LENGTH is the minimum safe distance across `width_samples` points
    spanning the segment (not just its midpoint) -- a single centerline
    ray only constrains the stamp's own centerline, not its actual
    rectangular footprint; see the width-sampling note in git history
    for the wedge case this was built to catch. Samples that find no
    wall are simply excluded rather than aborting the whole segment;
    if literally none find one, length falls back to min_length.
    """
    rings = [region["exterior_simplified"], *region["holes_simplified"]]
    all_edges = [e for ring in rings for e in _ring_edges(ring)]

    results: list[EdgeResult] = []

    for ring_index, ring in enumerate(rings):
        winding_sign = 1 if ring_index == 0 else -1  # 0 = exterior, rest = holes
        ring_kind = "exterior" if ring_index == 0 else f"hole {ring_index - 1}"

        for edge_index, (p0, p1) in enumerate(_ring_edges(ring)):
            seg = _sub(p1, p0)
            width = _length(seg)

            if width < 1e-9:
                # A genuinely degenerate (duplicate-point) segment --
                # not a tunable "too small to bother with" judgment,
                # just numerically meaningless to build a stamp from.
                results.append(EdgeResult(ring_index, ring_kind, edge_index, p0, p1,
                                           None, "degenerate zero-width segment"))
                continue

            direction = _normalize(seg)
            midpoint = _scale(_add(p0, p1), 0.5)
            interior_perp = _rotate90(direction, winding_sign)

            sample_ts = np.linspace(0.05, 0.95, width_samples)
            sample_distances = []
            for t in sample_ts:
                sample_origin = _add(p0, _scale(seg, float(t)))
                d = _ray_edges_intersection(sample_origin, interior_perp, all_edges, max_search_distance)
                if d is not None:
                    sample_distances.append(d)

            length = max(min_length, min(sample_distances)) if sample_distances else min_length

            center = _add(midpoint, _scale(interior_perp, length / 2.0))

            # Same formula/axis convention as terrain/cart_paths.py's
            # real production rotation -- scale_z is the
            # along-rotation "forward" dimension, scale_x is across.
            rotation_deg = math.degrees(math.atan2(interior_perp[0], interior_perp[1]))

            scale_x = (width / 2.0) / PLATEAU_FRACTION
            scale_z = (length / 2.0) / PLATEAU_FRACTION

            stamp = Stamp(
                x=center[0], z=center[1],
                scale_x=scale_x, scale_z=scale_z,
                value=0.0,  # placeholder -- no real elevation data in this test
                brush=fill_brush, rotation=rotation_deg, tool=TOOL_FLATTEN,
            )
            results.append(EdgeResult(ring_index, ring_kind, edge_index, p0, p1, stamp, None))

    return results


class FallLineFillApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fall-Line Fill Visualizer (real boundary + real type72 stamps)")
        self.root.geometry("1400x920")
        self.root.minsize(1100, 720)

        self.target = None
        self.target_image = None
        self.regions: list[dict] = []
        self.edge_results: list[EdgeResult] = []
        self.tk_image = None

        self.tolerance = tk.DoubleVar(value=DEFAULT_TOLERANCE)
        self.fill_brush = tk.IntVar(value=DEFAULT_FILL_BRUSH)
        self.min_length = tk.DoubleVar(value=DEFAULT_MIN_LENGTH)
        self.max_search_distance = tk.DoubleVar(value=DEFAULT_MAX_SEARCH_DISTANCE)

        self.show_rings = tk.BooleanVar(value=True)
        self.show_stamps = tk.BooleanVar(value=True)
        self.show_fallbacks = tk.BooleanVar(value=True)

        self.selected_index = tk.IntVar(value=0)
        self.selected_index.trace_add("write", lambda *args: self._on_selection_change())
        self.selected_info = tk.StringVar(value="No edges yet")

        self.stats = tk.StringVar(value="No run yet")
        self.file_name = tk.StringVar(value="No image loaded")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer, width=310)
        controls.pack(side="left", fill="y", padx=(0, 8))
        controls.pack_propagate(False)

        ttk.Label(
            controls, text="Fall-Line Fill", font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            controls,
            text=("Real traced boundary -> real type72 solid rotated stamps, "
                  "one per boundary edge, shot inward to the opposite wall."),
            wraplength=290, justify="left", font=("TkDefaultFont", 9), foreground="#555",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Button(controls, text="Open 1-bit PNG...", command=self.open_image).pack(fill="x")
        ttk.Label(controls, textvariable=self.file_name, wraplength=290).pack(anchor="w", pady=(6, 14))

        self._separator(controls)
        ttk.Label(controls, text="TRACE", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Tolerance (m)", self.tolerance, 0.1, 100, increment=0.1)

        self._separator(controls)
        ttk.Label(controls, text="FALL-LINE RULE", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self._add_spinbox(controls, "Fill brush", self.fill_brush, 1, 200)
        self._add_spinbox(controls, "Min length (always placed)", self.min_length, 0.1, 50, increment=0.5)
        self._add_spinbox(controls, "Max search distance", self.max_search_distance, 10, 5000, increment=10)

        self._separator(controls)
        ttk.Checkbutton(controls, text="Show boundary rings", variable=self.show_rings, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(controls, text="Show stamps (solid fill)", variable=self.show_stamps, command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(controls, text="Show fallback segments (red)", variable=self.show_fallbacks, command=self.redraw).pack(anchor="w", pady=(0, 10))

        ttk.Button(controls, text="Trace + generate stamps", command=self.run_fill).pack(fill="x", pady=(4, 4))

        self._separator(controls)
        ttk.Label(controls, text="INSPECT EDGE", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        sel_frame = ttk.Frame(controls)
        sel_frame.pack(fill="x", pady=2)
        ttk.Label(sel_frame, text="Edge index").pack(side="left")
        self.selected_spin = ttk.Spinbox(
            sel_frame, from_=0, to=0, textvariable=self.selected_index, width=10,
            increment=1, wrap=True,
        )
        self.selected_spin.pack(side="right")
        ttk.Label(controls, textvariable=self.selected_info, wraplength=290, justify="left").pack(anchor="w", pady=(4, 10))

        self._separator(controls)
        ttk.Label(controls, textvariable=self.stats, wraplength=290, justify="left", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        view_frame = ttk.Frame(outer, relief="sunken", borderwidth=1)
        view_frame.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(view_frame, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def _separator(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=8)

    def _add_spinbox(self, parent, label, variable, low, high, increment=1):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label).pack(side="left")
        spin = ttk.Spinbox(frame, from_=low, to=high, textvariable=variable, width=10, increment=increment)
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
            self.edge_results = []
            self.file_name.set(f"{path}  ({img.width} x {img.height})")
            self.stats.set("No run yet")
            self._update_selection_range()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))

    def run_fill(self):
        if self.target is None:
            messagebox.showwarning("No target", "Open a 1-bit PNG first.")
            return
        try:
            tolerance = float(self.tolerance.get())
            fill_brush = int(self.fill_brush.get())
            min_len = float(self.min_length.get())
            max_dist = float(self.max_search_distance.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid settings", "One or more fields aren't valid numbers.")
            return

        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.regions = trace_and_simplify(self.target, tolerance)
            self.edge_results = []

            for region in self.regions:
                self.edge_results.extend(build_interior_fill_stamps(
                    region, self.target,
                    fill_brush=fill_brush,
                    min_length=min_len,
                    max_search_distance=max_dist,
                ))

            self._update_stats()
            self._update_selection_range()
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Fill failed", str(exc))
        finally:
            self.root.config(cursor="")

    def _update_selection_range(self):
        n = len(self.edge_results)
        self.selected_spin.configure(to=max(0, n - 1))
        if n == 0:
            self.selected_info.set("No edges yet")
            return
        if self.selected_index.get() >= n:
            self.selected_index.set(0)
        else:
            self._update_selected_info()

    def _update_selected_info(self):
        n = len(self.edge_results)
        if n == 0:
            self.selected_info.set("No edges yet")
            return
        i = max(0, min(self.selected_index.get(), n - 1))
        e = self.edge_results[i]
        p0r = tuple(round(c, 1) for c in e.p0)
        p1r = tuple(round(c, 1) for c in e.p1)
        lines = [f"Edge {i} of {n}  ({e.ring_kind}, edge {e.edge_index})", f"p0={p0r}  p1={p1r}"]
        if e.stamp is not None:
            s = e.stamp
            lines.append(f"width={s.scale_x*2*PLATEAU_FRACTION:.1f} length={s.scale_z*2*PLATEAU_FRACTION:.1f} rot={s.rotation:.1f}")
        else:
            lines.append(f"NO STAMP -- {e.fallback_reason}")
        self.selected_info.set("\n".join(lines))

    def _update_stats(self):
        n_stamps = sum(1 for e in self.edge_results if e.stamp is not None)
        n_fallback = len(self.edge_results) - n_stamps
        lines = [
            f"Regions: {len(self.regions)}",
            f"Stamps: {n_stamps:,}   Fallbacks: {n_fallback:,}",
        ]
        self.stats.set("\n".join(lines))

    def _stamp_corners(self, s: Stamp) -> list[tuple[float, float]]:
        """
        Compute the plateau rectangle's 4 world-space corners directly
        from the Stamp's own fields via explicit trig, rather than
        building an axis-aligned patch and calling Image.rotate().

        The rotate+paste approach silently assumed a match between
        PIL's rotation handedness (counter-clockwise AS DISPLAYED --
        which is clockwise in raw pixel-index terms, since PIL's own
        +y points down) and the atan2(x, z) angle convention borrowed
        from cart_paths.py. That match was never actually verified --
        only the angle FORMULA was checked, which isn't the same as
        checking the RENDERING matches it -- and the mismatch showed
        up as visible bow-tie seams where adjacent stamps didn't
        actually line up despite their centers being correct.

        Reconstructing by direct trig removes the ambiguity by
        construction: build_interior_fill_stamps computed rotation as
        atan2(perp_x, perp_z) where perp was already a unit vector, so
        by the definition of atan2, (sin(theta), cos(theta)) recovers
        that exact same perp -- nothing left to get backwards.
        """
        theta = math.radians(s.rotation)
        perp = (math.sin(theta), math.cos(theta))     # "length" (forward) direction
        across = (math.cos(theta), -math.sin(theta))  # "width" (perpendicular) direction

        half_len = s.scale_z * PLATEAU_FRACTION
        half_wid = s.scale_x * PLATEAU_FRACTION

        near = (s.x - perp[0] * half_len, s.z - perp[1] * half_len)
        far = (s.x + perp[0] * half_len, s.z + perp[1] * half_len)

        return [
            (near[0] - across[0] * half_wid, near[1] - across[1] * half_wid),
            (near[0] + across[0] * half_wid, near[1] + across[1] * half_wid),
            (far[0] + across[0] * half_wid, far[1] + across[1] * half_wid),
            (far[0] - across[0] * half_wid, far[1] - across[1] * half_wid),
        ]

    def _on_selection_change(self):
        try:
            self._update_selected_info()
        except (tk.TclError, ValueError):
            pass  # mid-edit in the spinbox entry, ignore until it settles
        self.redraw()

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
        base.paste(Image.new("RGB", (w, h), (60, 60, 60)), mask=mask_img)
        base = base.convert("RGBA")

        draw = ImageDraw.Draw(base, "RGBA")  # "RGBA" mode = real alpha compositing,
        # not overwrite -- needed so overlapping stamps still visibly
        # darken instead of the last-drawn one just replacing pixels.

        if self.show_fallbacks.get():
            for e in self.edge_results:
                if e.stamp is None:
                    draw.line([e.p0, e.p1], fill=FALLBACK_COLOR, width=3)

        if self.show_stamps.get():
            for e in self.edge_results:
                if e.stamp is not None:
                    draw.polygon(self._stamp_corners(e.stamp), fill=(*STAMP_COLOR, 150))

        if self.show_rings.get():
            for region in self.regions:
                ext = region["exterior_simplified"]
                if len(ext) >= 2:
                    draw.line(ext + [ext[0]], fill=OUTER_RING_COLOR, width=2)
                for hole in region["holes_simplified"]:
                    if len(hole) >= 2:
                        draw.line(hole + [hole[0]], fill=INNER_RING_COLOR, width=2)

        # Highlight the selected edge on top of everything else: its
        # stamp outlined in bright yellow (if it has one), the edge
        # itself as a thick white line, and p0/p1 as distinctly
        # colored dots so direction is visible too.
        n = len(self.edge_results)
        if n > 0:
            i = max(0, min(self.selected_index.get(), n - 1))
            e = self.edge_results[i]
            if e.stamp is not None:
                draw.polygon(self._stamp_corners(e.stamp), outline=(255, 255, 0, 255), width=2)
            draw.line([e.p0, e.p1], fill=(255, 255, 255, 255), width=2)
            r = 3
            draw.ellipse((e.p0[0]-r, e.p0[1]-r, e.p0[0]+r, e.p0[1]+r), fill=(80, 220, 80, 255))   # p0 = green
            draw.ellipse((e.p1[0]-r, e.p1[1]-r, e.p1[0]+r, e.p1[1]+r), fill=(220, 80, 220, 255))  # p1 = magenta

        base = base.convert("RGB")
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
    app = FallLineFillApp(root)
    root.mainloop()