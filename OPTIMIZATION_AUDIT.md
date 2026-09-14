# PGA2K-LiDAR-TerrainGen — Low-Hanging-Fruit Optimization Audit

Branch: `hermes-experiment` (findings appended per pass; applied fixes
marked FIXED).
Method: cheap AST pre-scan (repeated heavy loads, deep-nested loops) → targeted
verification by reading actual code. Findings appended per pass.

## Pre-scan summary
- **No duplicate heavy loads** within a single function (heightmap/pointcloud/stamps
  are each loaded once per step). Good — the classic "re-read in a loop" bug is absent.
- `json.load` repeats are all single-file, harmless.
- Deep-nested loops (for-depth >= 2) flagged as O(n^2)+ candidates — verifying below.

## Findings

### F1 — `step_generate_terrain` loads the full point cloud to produce two unused floats  [BIG]
- **Location:** `PGA2k_gen.py:2641-2648` (load+crop), `:2834-2835` (only consumer).
- **What happens:** `step_generate_terrain` runs `PointCloud.load(pointcloud_path)` on the
  entire LAS/LAZ and `recentered_crop(...)`, then uses `course_cloud.origin_x/.origin_y`
  for exactly one thing: writing `course_origin_x`/`course_origin_y` into `project.json`.
- **Why it's dead weight:** a repo-wide grep shows `course_origin_x`/`course_origin_y`
  have **no readers** — only the two writes at 2834/2835. Every real consumer reads
  `course_crop_origin_in_full_frame_x/z` (saved by `step_ingest_osm` at :1228-1229 and
  read at :1333, :1548, `PGA2k_gen_gui.py:5523, :6774`), all from the project file —
  none reload the point cloud. `full_cloud`/`course_cloud` appear nowhere else in the
  function body (verified by full-file grep, not just the read window).
- **Fix:** delete lines 2641-2648 (the `PointCloud.load` + `recentered_crop` + their
  try/except) and the two `course_origin_*` entries in the `save_project` at 2833-2835.
- **Impact:** removes a heavy full-point-cloud parse (potentially the single most
  expensive I/O op in this step) from the main generation path.
- **Risk:** effectively zero — the produced values are never read, and nothing in the
  function depends on the crop. Output is byte-identical. (Only caveat: confirm no
  external/manual tooling reads `course_origin_x` out of `project.json`; not found in repo.)
  - Keep the pointcloud existence guard at :2523-2527 (useful "run ingest-laz first"
    hint) — only the load+crop at 2641-2648 go.

### F2 — preview re-rendered at full zoomed resolution on every zoom tick  [BIG — FIXED]
- **Location:** `PGA2k_gen_gui.py` `_show_preview` (resize block) + `_set_preview_image`.
- **What happened:** every scroll tick ran the *entire* render tail —
  `geo_img.resize(w*zoom, h*zoom, LANCZOS)` + optional elevation-band
  numpy/composite + objects-layer composite + a fresh `ImageTk.PhotoImage`
  — at the zoomed pixel size, all on the Tk mainloop. Measured on the
  1959x1780 composite: LANCZOS alone ~170ms at 100%, ~490ms at 200%,
  ~930ms at 300%; with PhotoImage encode the 300% tick hit ~1.5s.
  The earlier zoom bugs (zoom in the composite/geo-overlay cache keys)
  were already fixed — this was the remaining per-tick cost.
- **Fix (applied, `hermes-experiment`):** per-zoom-level render cache.
  Zoom is quantized to ¼ steps; the finished RGBA for a
  `(geo_overlay_key, quantized-zoom, overlay-inputs)` pair is cached
  (LRU, capped at 3 entries / ~45MP so a 300% render can't dominate
  memory), and `_set_preview_image` reuses the `PhotoImage` when it
  gets the same PIL object back (identity-keyed). LANCZOS is kept for
  downscaling only; upscaling uses BICUBIC (faster, adds nothing).
  Cursor-anchored zoom, scrollregion, centering and course picking are
  unchanged — they operate on the actually-rendered image size.
- **Verified headless (xvfb, real GUI instance, synthetic 1959x1780
  preview):** steady-state zoom tick (cache hit) ~9ms with a PIL
  resize counter proving 0 resizes; all level-crossing renders,
  cursor anchoring (data point under pointer identical before/after),
  scrollregion, and course-metre picking pass.
- **Note:** a Tk-native `canvas.itemconfigure(zoom=)` was tried first
  and is NOT possible on this system's Tk 9.0.4 (canvas `-zoom`/`-scale`
  options are unknown even at item-creation time), hence the
  quantized-level cache instead.

### Verified clean (no low-hanging fruit)
Checked the core hot-path functions; none have redundant I/O or avoidable recompute:
- `step_generate_terrain` — other I/O (heightmap, height mask) is single-shot and
  cached by the ingest steps; the two `load_project` calls at :2518/:2529 are a minor
  redundancy (first is only taken when `method is None`) — not worth a change.
- `step_refine_terrain` → `terrain/adaptive_refine.py:find_error_hotspots` — distance
  transforms are cached across iterations and only recomputed when the matching-sign
  claim changes; samples a pre-rasterized ground grid, not a KD-tree per hotspot.
- `terrain/contour_layers.py:_poisson_pack_band` — one static EDT for the whole band
  (not per tier) + a spatial hash grid for O(1)-amortized overlap rejection.
- `course_output/water.py:fit_water_tiles` — simplifies the polygon (Douglas-Peucker)
  before the per-edge loop; ray-cast depth is per-edge, not per-vertex.
- `PGA2k_gen.py:load_all_stamps` — nested loop is O(total stamps), no per-stamp I/O.

`ref/` and `util/viz/` deep loops (the for-depth-3/4 hits) are reference/visualization
scripts, not on the main generation pipeline — out of scope for this pass.

## Bottom line
Two real wins found: **F1** (dead full point-cloud load in
`step_generate_terrain` — applied earlier, `91d6c77`) and **F2**
(per-tick full-resolution preview re-render on zoom — applied,
quantized per-level render cache + PhotoImage reuse). Everything else
on the hot path is already carefully optimized; no other mechanical
low-hanging fruit found.

