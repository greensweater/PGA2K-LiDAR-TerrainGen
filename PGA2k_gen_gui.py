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

import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

SCRIPT_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = SCRIPT_DIR / "PGA2k_gen.py"

# Reused directly rather than duplicated -- these are plain, side-effect-free
# JSON helpers already tested as part of the CLI (see PGA2k_gen.py).
sys.path.insert(0, str(SCRIPT_DIR))
from PGA2k_gen import load_project, save_project  # noqa: E402

PREVIEW_FILES = [
    "preview_lidar_heightmap.png",
    "preview_lidar.png",
    "preview_hex.png",
    "preview_stamps.png",
    "preview_height.png",
    "preview_error.png",
]

# Game version -> Courses folder name under .../AppData/LocalLow/2K/.
# Windows-specific path (AppData/LocalLow only exists on Windows, which is
# also the only platform The Golf Club / PGA 2K actually runs on) -- only
# 2019 is wired up for now, per the request to add 2K21/23/25 later.
GAME_VERSIONS = {
    "The Golf Club 2019": "The Golf Club 2019",
}


class PGAGenGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PGA2K Terrain Compiler")
        root.geometry("1200x760")

        self.working_dir = tk.StringVar()
        self.course_name = tk.StringVar()
        self.log_queue: queue.Queue = queue.Queue()
        self.running = False
        self._preview_imgtk = None  # keep a reference so tkinter doesn't GC it
        self._suppress_course_name_save = False

        self._build_layout()
        self._poll_log_queue()

        self.working_dir.trace_add("write", lambda *a: self._on_working_dir_changed())
        self.course_name.trace_add("write", lambda *a: self._on_course_name_changed())

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        row1 = ttk.Frame(top)
        row1.pack(fill="x")
        ttk.Label(row1, text="Working directory:").pack(side="left")
        ttk.Entry(row1, textvariable=self.working_dir, width=60).pack(
            side="left", padx=4, fill="x", expand=True
        )
        ttk.Button(row1, text="Browse...", command=self._browse_working_dir).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Label(row2, text="Course name:").pack(side="left")
        ttk.Entry(row2, textvariable=self.course_name, width=40).pack(
            side="left", padx=4, fill="x"
        )

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        # Horizontal split: preview (more room, per request) on the left,
        # log on the right; the sash between them resizes width, not height.
        right = ttk.PanedWindow(main, orient="horizontal")
        right.pack(side="left", fill="both", expand=True)

        self._build_step_buttons(left)
        self._build_preview_panel(right)
        self._build_log_panel(right)

    def _build_step_buttons(self, parent: ttk.Frame) -> None:
        self._add_step_button(parent, "Init", self._run_init)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.projection_var = tk.StringVar()
        ttk.Label(parent, text="Projection EPSG (optional):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.projection_var, width=14).pack(anchor="w")
        self._add_step_button(parent, "Ingest LAZ", self._run_ingest_laz)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Ingest OSM", self._run_ingest_osm)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.course_file_var = tk.StringVar()
        ttk.Label(parent, text="Course file (.course):").pack(anchor="w")
        course_file_row = ttk.Frame(parent)
        course_file_row.pack(anchor="w", fill="x")
        ttk.Entry(course_file_row, textvariable=self.course_file_var, width=18).pack(side="left")
        ttk.Button(course_file_row, text="...", width=3, command=self._browse_course_file).pack(side="left")
        self._add_step_button(parent, "Ingest Course", self._run_ingest_course)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Generate Terrain", self._run_generate_terrain)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.tolerance_var = tk.StringVar(value="2.0")
        self.resolution_var = tk.StringVar(value="200")
        self.min_region_cells_var = tk.StringVar(value="2")
        self.max_new_var = tk.StringVar()
        ttk.Label(parent, text="Error tolerance (m):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.tolerance_var, width=14).pack(anchor="w")
        ttk.Label(parent, text="Error grid resolution:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.resolution_var, width=14).pack(anchor="w")
        ttk.Label(parent, text="Min region cells:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.min_region_cells_var, width=14).pack(anchor="w")
        ttk.Label(parent, text="Max new stamps (optional):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.max_new_var, width=14).pack(anchor="w")
        self._add_step_button(parent, "Refine Terrain", self._run_refine_terrain)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Output Terrain", self._run_output_terrain)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.repack_filename_var = tk.StringVar()
        ttk.Label(parent, text="Repack filename:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.repack_filename_var, width=20).pack(anchor="w")
        self._add_step_button(parent, "Repack", self._run_repack)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.game_version_var = tk.StringVar(value=next(iter(GAME_VERSIONS)))
        ttk.Label(parent, text="Game version:").pack(anchor="w")
        ttk.Combobox(
            parent, textvariable=self.game_version_var, values=list(GAME_VERSIONS),
            state="readonly", width=20,
        ).pack(anchor="w")
        self._add_step_button(parent, "Copy to Game Folder", self._run_copy_to_game)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self._add_step_button(parent, "Visualize", self._run_visualize)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=10)
        self.status_label = ttk.Label(parent, text="Idle", foreground="gray")
        self.status_label.pack(anchor="w")

    def _build_log_panel(self, paned: ttk.PanedWindow) -> None:
        frame = ttk.Frame(paned)
        paned.add(frame, weight=1)
        ttk.Label(frame, text="Output log").pack(anchor="w")

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
        ttk.Button(header, text="Refresh", command=self._refresh_preview_and_slider).pack(side="left")

        version_row = ttk.Frame(frame)
        version_row.pack(fill="x", pady=(4, 0))
        ttk.Label(version_row, text="Version:").pack(side="left")
        self.preview_version = tk.IntVar(value=0)
        self.preview_version_scale = ttk.Scale(
            version_row, from_=0, to=0, orient="horizontal",
            variable=self.preview_version, command=self._on_preview_version_changed,
        )
        self.preview_version_scale.pack(side="left", fill="x", expand=True, padx=4)
        self.preview_version_label = ttk.Label(version_row, text="current", width=10)
        self.preview_version_label.pack(side="left")

        # Scroll wheel over either the slider or the image itself steps
        # through versions -- Windows/Mac send <MouseWheel> with event.delta;
        # Linux sends <Button-4>/<Button-5> instead.
        for widget in (self.preview_version_scale,):
            widget.bind("<MouseWheel>", self._on_preview_scroll)
            widget.bind("<Button-4>", self._on_preview_scroll)
            widget.bind("<Button-5>", self._on_preview_scroll)

        self.preview_label = ttk.Label(frame, text="(no preview loaded)", anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_label.bind("<MouseWheel>", self._on_preview_scroll)
        self.preview_label.bind("<Button-4>", self._on_preview_scroll)
        self.preview_label.bind("<Button-5>", self._on_preview_scroll)

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
        project = load_project(Path(wd))
        self._suppress_course_name_save = True
        try:
            self.course_name.set(project.get("course_name", ""))
        finally:
            self._suppress_course_name_save = False
        self._refresh_preview_and_slider()

    def _on_course_name_changed(self) -> None:
        if self._suppress_course_name_save:
            return
        wd = self.working_dir.get().strip()
        if not wd or not Path(wd).is_dir():
            return
        save_project(Path(wd), {"course_name": self.course_name.get()})

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
        self._run_step(args, wd)

    def _run_ingest_osm(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "ingest-osm"], wd)

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
            self._run_step(["--step", "generate-terrain"], wd)

    def _run_refine_terrain(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "refine-terrain"]
        tol = self.tolerance_var.get().strip()
        if tol:
            args += ["--error-tolerance", tol]
        res = self.resolution_var.get().strip()
        if res:
            args += ["--resolution", res]
        min_cells = self.min_region_cells_var.get().strip()
        if min_cells:
            args += ["--min-region-cells", min_cells]
        max_new = self.max_new_var.get().strip()
        if max_new:
            args += ["--max-new-stamps", max_new]
        self._run_step(args, wd)

    def _run_output_terrain(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "output-terrain"], wd)

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
            self._run_step(["--step", "visualize"], wd)

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

        version = self.game_version_var.get()
        folder_name = GAME_VERSIONS[version]
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
        step_name = extra_args[1]
        self.status_label.config(text=f"Running {step_name}...", foreground="orange")
        self._clear_log()

        cmd = [sys.executable, str(CLI_SCRIPT), str(working_dir)] + extra_args
        self._append_log(f"$ {' '.join(cmd)}\n\n")

        thread = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        thread.start()

    def _run_subprocess(self, cmd: list[str]) -> None:
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self.log_queue.put(("line", line))
            proc.wait()
            self.log_queue.put(("done", proc.returncode))
        except Exception as e:
            self.log_queue.put(("error", str(e)))

    def _poll_log_queue(self) -> None:
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "line":
                    self._append_log(payload)
                elif kind == "done":
                    self.running = False
                    if payload == 0:
                        self.status_label.config(text="Done", foreground="green")
                    else:
                        self.status_label.config(text=f"Failed (exit {payload})", foreground="red")
                    self._refresh_preview_and_slider()
                    self.running = False
                    self.status_label.config(text="Error", foreground="red")
                    self._append_log(f"\n[GUI error] {payload}\n")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

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
        version=0 is the current (unsuffixed) file; version=1,2,...
        are the archived previous runs (see visualize.py's
        _archive_existing: preview_error.png, preview_error_1.png, ...).
        """
        name = self.preview_choice.get()
        if version == 0:
            return working_dir / name
        stem, suffix = Path(name).stem, Path(name).suffix
        return working_dir / f"{stem}_{version}{suffix}"

    def _max_preview_version(self, working_dir: Path) -> int:
        """Highest archived version present for the currently chosen preview."""
        n = 0
        while self._versioned_preview_path(working_dir, n + 1).exists():
            n += 1
        return n

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

    def _show_preview(self) -> None:
        wd = self.working_dir.get().strip()
        if not wd:
            return
        if not _HAVE_PIL:
            self.preview_label.config(
                text="(Pillow not installed -- pip install pillow for image previews)", image=""
            )
            return

        version = int(round(self.preview_version.get()))
        path = self._versioned_preview_path(Path(wd), version)
        if not path.exists():
            self.preview_label.config(text=f"(no {path.name} yet)", image="")
            self._preview_imgtk = None
            return

        try:
            img = Image.open(path)
            img.thumbnail((900, 900), Image.LANCZOS)
            self._preview_imgtk = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self._preview_imgtk, text="")
        except Exception as e:
            self.preview_label.config(text=f"(couldn't load {path.name}: {e})", image="")


def main() -> int:
    if not CLI_SCRIPT.exists():
        print(
            f"error: {CLI_SCRIPT} not found -- this GUI must sit in the same folder as PGA2k_gen.py",
            file=sys.stderr,
        )
        return 1

    root = tk.Tk()
    PGAGenGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
