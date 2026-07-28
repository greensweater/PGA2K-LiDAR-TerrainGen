#!/usr/bin/env python3
"""
PGA2k_gen_gui.py

Minimal desktop GUI wrapping PGA2k_gen.py's steps as buttons: set a
working directory, click a step, watch its output stream in, see
whatever diagnostic preview it produced.

Deliberately thin: every step runs PGA2k_gen.py as a subprocess with
the exact same arguments the CLI takes, rather than re-implementing or
calling into its internals directly. That means there's exactly one
place pipeline behavior lives -- this GUI can't drift out of sync with
the CLI, and anything that works from the command line works here.

Requires: tkinter (stdlib) + Pillow (for preview images -- the GUI
still works without Pillow, previews just won't render).
"""

from __future__ import annotations

import queue
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

PREVIEW_FILES = [
    "preview_lidar_heightmap.png",
    "preview_lidar.png",
    "preview_hex.png",
    "preview_stamps.png",
    "preview_height.png",
    "preview_error.png",
]


class PGAGenGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PGA2K Terrain Compiler")
        root.geometry("1100x720")

        self.working_dir = tk.StringVar()
        self.log_queue: queue.Queue = queue.Queue()
        self.running = False
        self._preview_imgtk = None  # keep a reference so tkinter doesn't GC it

        self._build_layout()
        self._poll_log_queue()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Working directory:").pack(side="left")
        ttk.Entry(top, textvariable=self.working_dir, width=60).pack(
            side="left", padx=4, fill="x", expand=True
        )
        ttk.Button(top, text="Browse...", command=self._browse_working_dir).pack(side="left")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y", padx=(0, 8))

        right = ttk.PanedWindow(main, orient="vertical")
        right.pack(side="left", fill="both", expand=True)

        self._build_step_buttons(left)
        self._build_log_panel(right)
        self._build_preview_panel(right)

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
        self._add_step_button(parent, "Generate Terrain", self._run_generate_terrain)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.tolerance_var = tk.StringVar(value="2.0")
        self.subdivision_var = tk.StringVar(value="2.0")
        self.max_new_var = tk.StringVar()
        ttk.Label(parent, text="Error tolerance (m):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.tolerance_var, width=14).pack(anchor="w")
        ttk.Label(parent, text="Subdivision factor:").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.subdivision_var, width=14).pack(anchor="w")
        ttk.Label(parent, text="Max new stamps (optional):").pack(anchor="w")
        ttk.Entry(parent, textvariable=self.max_new_var, width=14).pack(anchor="w")
        self._add_step_button(parent, "Refine Terrain", self._run_refine_terrain)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        self.course_dir_var = tk.StringVar()
        ttk.Label(parent, text="Course dir (optional):").pack(anchor="w")
        course_row = ttk.Frame(parent)
        course_row.pack(anchor="w", fill="x")
        ttk.Entry(course_row, textvariable=self.course_dir_var, width=18).pack(side="left")
        ttk.Button(course_row, text="...", width=3, command=self._browse_course_dir).pack(side="left")
        self._add_step_button(parent, "Output Terrain", self._run_output_terrain)

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
            text_row, height=15, wrap="word", state="disabled",
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
        dropdown.bind("<<ComboboxSelected>>", lambda e: self._show_preview())
        ttk.Button(header, text="Refresh", command=self._show_preview).pack(side="left")

        self.preview_label = ttk.Label(frame, text="(no preview loaded)", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

    def _add_step_button(self, parent: ttk.Frame, label: str, command) -> ttk.Button:
        btn = ttk.Button(parent, text=label, command=command, width=22)
        btn.pack(anchor="w", pady=2)
        return btn

    # ------------------------------------------------------------------
    # Folder pickers
    # ------------------------------------------------------------------

    def _browse_working_dir(self) -> None:
        d = filedialog.askdirectory(title="Select working directory")
        if d:
            self.working_dir.set(d)

    def _browse_course_dir(self) -> None:
        d = filedialog.askdirectory(title="Select course directory")
        if d:
            self.course_dir_var.set(d)

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
        sub = self.subdivision_var.get().strip()
        if sub:
            args += ["--subdivision-factor", sub]
        max_new = self.max_new_var.get().strip()
        if max_new:
            args += ["--max-new-stamps", max_new]
        self._run_step(args, wd)

    def _run_output_terrain(self) -> None:
        wd = self._require_working_dir()
        if not wd:
            return
        args = ["--step", "output-terrain"]
        course_dir = self.course_dir_var.get().strip()
        if course_dir:
            args += ["--course-dir", course_dir]
        self._run_step(args, wd)

    def _run_visualize(self) -> None:
        wd = self._require_working_dir()
        if wd:
            self._run_step(["--step", "visualize"], wd)

    # ------------------------------------------------------------------
    # Subprocess execution
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
                    self._show_preview()
                elif kind == "error":
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

    def _show_preview(self) -> None:
        wd = self.working_dir.get().strip()
        if not wd:
            return
        if not _HAVE_PIL:
            self.preview_label.config(
                text="(Pillow not installed -- pip install pillow for image previews)", image=""
            )
            return

        path = Path(wd) / self.preview_choice.get()
        if not path.exists():
            self.preview_label.config(text=f"(no {path.name} yet)", image="")
            self._preview_imgtk = None
            return

        try:
            img = Image.open(path)
            img.thumbnail((720, 720), Image.LANCZOS)
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
