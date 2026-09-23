# AGENTS.md — project memory for AI agents working in this repo

Working notes for agents (Claude Code, Hermes, etc.). Human-facing docs are
`README.md`; optimization findings live in `OPTIMIZATION_AUDIT.md`.

## Start here (do these first, in a fresh session)

1. `git branch --show-current` → must be `hermes-experiment` (not main).
2. Read `OPTIMIZATION_AUDIT.md` → what's already been found/fixed/verified clean.
3. Skim the rest of this file → repo layout + conventions.

Context is limited in this setup: keep reads targeted (offset/limit windows,
`grep`/`awk` for scanning, not whole-file reads), and append findings to
`OPTIMIZATION_AUDIT.md` incrementally so progress survives a context reset.

## `.course` file format + extraction

`.course` files are **gzip-compressed UTF-16LE JSON** (NOT zip — `unzip`
fails). Outer JSON = meta, whose `binaryData` holds base64(gzip(UTF-16LE
JSON)) blobs for `CourseDescription`, `CourseMetadata`, `Thumbnail`.
Extract/repack with the repo tools (no third-party deps; system python3
works):

- `python3 util/course_extract.py <file.course> <out_dir>` →
  `out_dir/CourseDescription.json` (base fields),
  `CourseDescription_nodes/<key>.json` (one file per top-level list/dict
  value — **node files are the schema surface to diff**), `CourseMetadata.json`,
  `meta.json`, `Thumbnail.jpg` + `Thumbnail_meta.json`.
- `python3 util/course_repack.py <course_dir> <out.course>` (inverts the
  above; `step_repack` calls it as a subprocess).

Schema discovery workflow (used for v2019/v2021/v2023): extract two
versions' `.course` files and diff the `CourseDescription_nodes/` sets and
entry shapes. v2023 findings live in `V2023_SCHEMA.md`; the v2023 work plan
in `V2023_TASKS.md`.

## Repo layout (what actually matters)

Python golf-course terrain generator: ingests LAZ LiDAR + OpenStreetMap, emits
PGA2k `.course` files. Two big top-level entry points:

- `PGA2k_gen.py` (~6.3k lines) — the CLI orchestrator. Each `step_*` function is
  one pipeline stage (`init`, `ingest-laz`, `ingest-osm`, `generate-terrain`,
  `refine-terrain`, `write-terrain`, `write-water`, `repack`, ...). **This is the
  hot path.**
- `PGA2k_gen_gui.py` (~6.9k lines) — tkinter GUI wrapping the same steps.
- `terrain/` — the real math: `contour_layers.py`, `adaptive_refine.py`,
  `streams.py`, `hexgrid.py`.
- `ingest/` — `laz_reader.py` (PointCloud + cKDTree), `heightmap.py`,
  `tree_detection.py`, `osm.py`.
- `course_output/` — output builders: `water.py`, `objects.py`, `parking.py`,
  `collections.py`, `out_of_bounds.py`.
- `ref/` and `util/viz/` — reference / visualization scripts. **NOT on the
  generation pipeline.** Out of scope for performance work; they still trip
  generic scans.

Pipeline hot path = the `step_*` functions in `PGA2k_gen.py` plus the
`terrain/` + `ingest/` + `course_output/` functions they call.

## Key conventions

- **Work on a dedicated branch, not main.** The `hermes-experiment` branch was
  created as an isolated sandbox for agent changes. Do not push experimental
  work to `main`/`master` without explicit ask.
- **`project.json` is the cross-step state bus.** Steps persist derived values
  (e.g. `course_crop_origin_in_full_frame_x/z`) here and re-read them, so a step
  does NOT need to reload the full point cloud to get a value an earlier step
  already computed. When you see a step reload something heavy, first check
  whether the value is already in `project.json`.
- **Stamps are layered, not snapshot.** Each pass writes only the stamps *it*
  added (`stamps/stamps_N.json`); `load_all_stamps` reconstructs the effective
  list. Deleting the highest-numbered file is the natural "undo".

## Already-verified performance state (do not re-audit)

The hot path has been audited (see `OPTIMIZATION_AUDIT.md`); these are known
**clean** — don't re-flag them as findings:

- `find_error_hotspots` (`terrain/adaptive_refine.py`) — distance transforms
  are cached across iterations, only recomputed when the matching-sign claim
  changes; samples a pre-rasterized grid, not a KD-tree per hotspot.
- `_poisson_pack_band` (`terrain/contour_layers.py`) — one static EDT per band
  + a spatial hash grid for O(1)-amortized overlap rejection.
- `fit_water_tiles` (`course_output/water.py`) — Douglas-Peucker simplifies the
  polygon before the per-edge loop; ray-cast depth is per-edge.
- `load_all_stamps` — nested loop is O(total stamps), no per-stamp I/O.
- `step_refine_terrain` / `step_generate_terrain` — heavy I/O is single-shot,
  cached by the ingest steps.

**Fixed (F1):** `step_generate_terrain` used to load the entire point cloud
just to write `course_origin_x`/`course_origin_y` into `project.json` — values
with zero readers. Removed. If you ever see a similar "load everything to write
a value" pattern, the win is deleting it (verify the value is truly unread
repo-wide first).

## How to audit this repo (short version)

1. AST pre-scan for repeated heavy loads + deep-nested loops (script, not LLM)
   → shortlist.
2. For each lead, read the code and prove the expensive thing is actually
   wasted (result unused, or recomputed in a loop). A single heavy load used
   once is invisible to a "duplicate" scan but is often the biggest win — grep
   for its *consumers*, not just duplicates.
3. Append findings to `OPTIMIZATION_AUDIT.md` incrementally as you go.
4. Read-only during the audit; apply fixes only after approval.

The full generic method lives in the Hermes `performance-optimization-audit`
skill.
