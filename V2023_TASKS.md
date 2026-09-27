# PGA2K v2023 implementation — task list

Working task breakdown for adding **v2023** support to PGA2K-LiDAR-TerrainGen.
Headline feature: **spline fences and walls**. Secondary v2023 capability
flagged in the registry: texture painting (`has_texture_paint`).

**How this list is meant to be executed** (read first — this is the contract
with fresh sessions): development runs **piecemeal, one task or subtask at a
time, each in a fresh session** to keep context small. So every task below
carries the fields a fresh session needs to work without prior context:

- **Context** — what the task is, why it exists, and what must already be
  done (dependency).
- **Where** — the files/functions to touch (with line anchors where stable)
  and the reference material to read first.
- **Steps** — the subtasks, each self-contained enough to be the *entire*
  prompt of a fresh session: a subtask statement is the goal; its indented
  bullets are the concrete steps, file anchors, and decisions to record.
- **Done when** — a concrete, checkable acceptance criterion (the fresh
  session reports success only if it can demonstrate this).
- **Decisions to record** (where non-obvious) — the exact text to append to
  `V2023_SCHEMA.md` or `V2023_TASKS.md` so the next session inherits it.

Schema facts live in `V2023_SCHEMA.md` (the single source of truth — do NOT
re-derive from the `.course` file when the schema doc already answers it;
re-extract only when a sample is re-pushed, e.g. after
`git pull` shows a new commit touching `templates/2023_fences.course`).

Every task is written to split cleanly into subtasks (indented bullets).
Tasks are ordered by dependency — Phase 0 gates Phase 1, Phase 1 gates
Phase 2, Phase 2 gates Phase 3.

**User-confirmed facts (2026-09-23, reduce Phase 0 scope):**
- v2023 uses the **same asset-path convention** as v2021 (`theme_scheme="path"`),
  plus **new assets** introduced in v2023 (catalog work needed — see 1.4 / 2.1.4).
- v2023 writes **`placedObjects3.json`** (same filename as v2021).
- Fences/walls: **both** an object-tile-along-line (parking.py-style) definition
  and a spline-based definition are valid v2023 representations (see 3.2).
- New feature request: fill splines with **"clear generated objects"** stamps —
  a brush-paint exclusion zone like the OOB feature (see 3.4).

**CONFIRMED from `templates/2023_fences.course` (see `V2023_SCHEMA.md`):**
- **Fences/walls are `Value.objectPaths[]` entries in `placedObjects3.json`** —
  a spline path with a fence material + per-path rule fields. This is the
  *native* v2023 fence; it is NOT `Value.splines[]` (surface spline) and NOT
  placed `items[]` (object tiling). The parking-style object tiling remains a
  valid *alternative* (Andy), but the confirmed game-written mechanism is
  objectPaths.
- Each `objectPath` = `{path:{waypoints[], width, hasCurves, state}, height,
  spacingRule, flexibilityRule, heightRule, spacing}`. Waypoints are 2D
  `{x, y}` (y = course Z) with `pointOne`/`pointTwo` bezier handles.
- Rule fields map to the described options: **`spacingRule` 0–3 = end-cap style
  (JSON: 0 none / 1 ends only / 2 spaced / 3 spline points — in-game verified 2026-09-26)**, **`heightRule` 0=contoured / 1=stepped**,
  **`hasCurves` true/false = curved/straight**, `height` = vertical offset
  (negative = buried), `spacing` = post spacing (m), `width` = fence width (m).
  `flexibilityRule`/`state` semantics still unconfirmed.
- **Per-asset option matrix is real and incomplete** — different fence types
  expose different option sets (brick wall rows show all 4 spacingRules;
  stone wall supports all 4 caps per the author — the single sample row only
  exercised `spacingRule=0` for brevity; canvas/hedge rows don't cover the
  full set). Must confirm which options are valid per asset (in-game).
- v2023 confirms **tilted items are legal** (GolfCartPrefab rot.x/z=±10°) and
  lists the **5 hole-sign colors** (HoleSign01–05; north→south = green/red/blue/
  white/black; 01 black (the v2021 sign), 02 white, 03 blue, 04 red, 05 green
  — see 0.3a.4).
- New v2023 fence assets are under `Assets/CourseGen/Detail/Walls/**`
  (`*Post*Prefab`, `*SplinePostPrefab`), incl. Asia/Brick; carts + hole signs
  under `Assets/CourseProps/**`.
- Node renames v2021→v2023: `userLayers`→`userLayers2`, `weather2`→`weather3`,
  base keys `holes`→`holes2`, `surfaceSplines`→`surfaceSplines2`; `version` 25→31;
  new base keys `blurHeight`/`useV28FairwaySeed`/`useV30Rough`.
- **Clear-generated-objects paint CONFIRMED** (second export, commit `8376f0e`):
  entries in the existing `userLayers2.json` **`surfaces` layer** with
  **`surfaceCategory: 5`** — brush stamps identical in shape to OOB entries
  (`type` 8 = round, 15 = smooth square — same constants as
  `course_output/out_of_bounds.py`), scale = stamp radius (m), value 1.0,
  tool 0, y "-Infinity". Known categories so far: 5 = clear objects, 9 = water
  (`water.py`).

**STILL UNCONFIRMED:** per-asset option matrix; `flexibilityRule`/`state`
meaning; in-game confirmation that category 5
suppresses generated objects (and which kinds); whether v2019/v2021 honor
category 5; `holes2`/`surfaceSplines2` entry shapes (sample empty).

> Branch: `hermes-experiment` (never `main`).
> Read `AGENTS.md` + this file's Phase 0 notes before starting.
> venv for any probe: `cd <repo> && uv venv .venv && uv pip install -r requirements.txt`.

---

## Phase 0 — Schema confirmation (MOSTLY DONE — see `V2023_SCHEMA.md`)

The repo's standing rule (stated in `game_versions.py` docstring): a version
is only "confirmed" after diffing an extracted `.course` the way `hhills3_2019`
vs `hhills3_2021` were. **Done on 2026-09-23** against
`templates/2023_fences.course` (fence/objectPaths schema, commit `39a54da`;
clear-objects stamps added in the second export, commit `8376f0e`), findings
in `V2023_SCHEMA.md`. Remaining items below are the confirmed gaps.

- [x] **0.1 Obtain a v2023 `.course` reference file** — landed as
      `templates/2023_fences.course` (fence-focused sample, not a blank
      template — fine for schema, but has authored fence content).
- [x] **0.2 Extract + diff against v2021** — node renames, base-key renames,
      version 25→31, objectPaths mechanism (see `V2023_SCHEMA.md`).
- [x] **0.3 Characterize the fence/wall schema (headline feature)** —
      objectPaths shape + rule fields → option mapping captured.
- [ ] **0.3a Remaining schema gaps (need a second sample or in-game trial)**
  - [ ] 0.3a.1 **Per-asset option matrix**: which `spacingRule` / `hasCurves` /
        `heightRule` values are valid per fence asset (stone wall = all 4 cap
        options per author; canvas/hedge don't expose all). In-game trial per
        asset, or a richer sample course.
  - [ ] 0.3a.2 **`flexibilityRule` / `state` semantics** (values seen: 0/1;
        fl=0 only on straight brick rows; state=1 only on the retaining wall).
  - [x] 0.3a.3 **`spacingRule` → cap-style ordering** — JSON 0 none, 1 ends only,
        2 spaced, 3 spline points (E3/E4 zigzags, 2026-09-26; the game menu lists them in a different order).
  - [x] 0.3a.4 **HoleSign0N ↔ color mapping** — 01 black (v2021 sign,
        Andy), 02 white, 03 blue, 04 red, 05 green (derived from positions,
        +z = north; see `V2023_SCHEMA.md` "Hole signs").
  - [ ] 0.3a.5 `holes2` / `surfaceSplines2` entry shapes (sample has both empty)
        — confirm vs `holes` / `surfaceSplines` once a populated course is
        available.
  - [x] 0.3a.6 Confirm `pointOne`/`pointTwo` are bezier handles (vs endpoint
        duplicates) when `hasCurves=false` — yes, same 0.25/0.375 rule either
        way (3.2; `V2023_SCHEMA.md` "objectPath handle rule").
- [ ] **0.3b "Clear generated objects" paint — schema CONFIRMED (2026-09-23, second
      export, commit `8376f0e`)** — see `V2023_SCHEMA.md` "Clear-generated-objects
      paint" section. It reuses the `userLayers2.json` `surfaces` layer with
      `surfaceCategory: 5`; entries are OOB-shaped brush stamps (`type` 8 round /
      15 smooth square, scale = radius in m, value 1.0). Remaining verification:
  - [x] 0.3b.1 **In-game check** (Andy, 2026-09-27, `clearobj_test1.course`):
        category 5 clears every kind of generated scatter (trees, plants, grass,
        rocks) inside the stamp and nothing outside it. Manually placed objects
        (our LIDAR trees) are untouched. The editor also has categories **6**
        (clear trees: removes only generated trees, confirmed in-game) and **11** (clear heavy rough:
        type 72, value 2.0, confirmed in-game); see
        `V2023_SCHEMA.md` "Three clear-stamp categories".
  - [x] 0.3b.2 Shapes confirmed (8 → circle, 15 → square). **`scale` uses the
        same convention as height stamps** (centre-to-edge of the full texture),
        settled with the `scale_test1.course` ruler test (2026-09-27): type 72
        clear and height stamps both end exactly at ±scale, and type 15 clear
        and height stamps show the same falloff between the plateau and active
        edges. The earlier "29 m ≈ scale/2" eyeball read was wrong. Still
        unknown and not blocking: what `value` means (1.0, or 2.0 for heavy
        rough); copy the game's values as they are.
  - [ ] 0.3b.3 v2019/v2021 support. **Tools confirmed (Andy, 2026-09-27)**:
        the earlier editors have "clear trees" and "clear objects", but **no
        "clear heavy rough"** (category 11 is v2023-only). **Still open: how they
        are stored.** v2021 userLayers has both `surfaces` (empty in our sample)
        and a separate `clearTrees` key, so clear-trees may live in
        `clearTrees` rather than as a `surfaceCategory: 6` entry. To settle it:
        get a v2021 (and v2019) course with one clear-trees and one
        clear-objects stamp placed in the editor, then extract it and read the
        `userLayers` entries.
  - [ ] 0.3b.4 Record answers in `V2023_SCHEMA.md` and lift the 3.4 gate.
- [x] **0.4 Document findings** — `V2023_SCHEMA.md` written; `VERSION_SCHEMAS`
      `2023` entry still to be populated (that is task 1.1, gated only on
      0.3b for the clear-objects capability flag).

---

## Phase 1 — Version registry + plumbing (low risk, unblocks everything)

- [x] **1.1 Add the `2023` `VersionSchema` entry** (`course_output/game_versions.py`)
  — DONE 2026-09-23. `schema_for("2023")` is fully populated. Hardcoded
  node names were replaced by `schema_for(v).*_filename` in
  `step_write_splines`, `step_write_holes`, `step_write_terrain`,
  `step_write_water`, `_inject_collection_into_course` and
  `step_import_ingame_edits`. `collection_library._height_entries` reads
  either userLayers name. New flag `has_radius_field` (v2023 height/OOB stamps
  drop `radius` as well as `_orientation`/`orientation`; see
  `V2023_SCHEMA.md` userLayers2 section). 1.1.2 was resolved from code, not
  in-game (see `V2023_SCHEMA.md` "Version registry").
  - **Context**: Phase 0 is done (schema in `V2023_SCHEMA.md`). This task
    registers v2023 in the version table so every downstream `schema_for(v)`
    call has something to return.
  - **Where**: `course_output/game_versions.py` — read the whole file first
    (it's small): the `VersionSchema` dataclass fields, the `VERSION_SCHEMAS`
    dict (look at the `2021` entry as the template), `IMPLEMENTED_GAME_VERSIONS`,
    and the module docstring.
  - **Steps**:
  - [x] 1.1.1 Set confirmed fields: `objects_filename="placedObjects3.json"`,
        `theme_scheme="path"`, **`holes_filename="holes2"`**,
        **`splines_filename="surfaceSplines2"`**,
        **`userlayers_filename="userLayers2"`** (v2023 renames the v2021
        nodes — see `V2023_SCHEMA.md` "Node renames" table). Audit every
        reader/writer of these filenames (holes.py, splines.py, userLayers.py,
        course_repack/extract, `_stale_version_node_files`) so the renamed
        nodes are produced, consumed, and stale-file-checked under the new
        names. Grep: `grep -rn 'holes\|surfaceSplines\|userLayers' --include='*.py' course_output/ util/ PGA2k_gen.py`
        (filter comments).
  - [x] 1.1.2 **Verify `placedObjects3` placement in v2023** (resolved: the
        "absent from base" was an extractor artifact, see `V2023_SCHEMA.md`
        "Version registry"): the v2021
        template carries `placedObjects3` as a *base* key (empty, inlined in
        CourseDescription), while v2023's base does NOT list it — the sample
        has it only as a node file. Confirm in-game that a repacked course
        with `CourseDescription_nodes/placedObjects3.json` (and the key absent
        from base) is loaded, and adjust `_ensure_course_baseline` /
        `course_repack.py` accordingly if the game wants it in the base.
  - [x] 1.1.3 Set capability flags: `has_fences=True` (objectPaths),
        `has_texture_paint` (deferred, Phase 4), and a clear-objects flag
        (e.g. `has_clear_objects=True` — the schema is confirmed; add the field
        to `VersionSchema` now, set it per version once 0.3b.3 resolves
        v2019/v2021 support).
  - [x] 1.1.4 Add `"2023"` to `IMPLEMENTED_GAME_VERSIONS`.
  - [x] 1.1.5 Update the module docstring to drop the "unconfirmed" note for
        2023.
  - **Done when**: `schema_for("2023")` returns a fully-populated entry
    (no placeholder/None fields); `python3 -c "from course_output.game_versions import schema_for; print(schema_for('2023'))"`
    runs clean in the repo venv.
  - **Decisions to record**: any field whose meaning required a guess (e.g.
    whether `placedObjects3` belongs in base) → append to `V2023_SCHEMA.md`
    under "Version registry" with the decision + evidence.
- [x] **1.2 Thread v2023 through version-gated code** — DONE 2026-09-23.
  Branch audit (location → v2023 behavior); Phase 2/3 don't need to redo it:

  | location | v2023 behavior |
  |---|---|
  | `_resolve_write_objects_params` / `step_import_ingame_edits` "isn't implemented" guards | pass (2023 is in `IMPLEMENTED_GAME_VERSIONS`) |
  | `_build_placed_objects` `game_version == "2019"` (trees/stakes/clusters/spline fills) | v2021 path-keyed builders (same placedObjects3 items/clusters/splines shape) |
  | same fn, waterfalls/splashes `== "2019"` | `build_*_v2021` |
  | `_apply_course_theme` `!= "2019"` → return | no-op (template encodes the look) |
  | `_inject_collection_into_course` `== "2019"` | `build_collection_objects_v2021`; splines/stamps go to the schema filenames |
  | `object_clusters._NO_SPLINE_FILL_VERSIONS = {"2019"}` | spline fills enabled |
  | `stamp_to_entry` / `oob_records_to_entries` | trimmed stamp shape via `has_orientation_fields` / `has_radius_field` |
  | `_stale_version_node_files` | v2021 `holes.json`/`surfaceSplines.json`/`userLayers.json` leftovers flagged stale in a 2023 project |
  | GUI `_sync_tree_assets_enabled` `!= "2019"` | tree-asset picker enabled |
  | GUI `GAME_VERSION_FOLDERS` | `Documents/My Games/PGA TOUR 2K23/Courses` (not LocalLow) |
  | `step_refine_terrain` | no version branch at all, nothing to do |

  Not yet version-aware (Phase 2 scope): `write_user_layers`' blank fallback
  schema is the v2021 key set (only used if the node file is missing);
  `water[]` entries keep the v2021 shape (unconfirmed for v2023).
  - **Context**: after 1.1, `schema_for("2023")` works; now make sure no code
    path hard-rejects or mis-handles v2023.
  - **Where**: `PGA2k_gen.py` (the `"isn't implemented yet"` guard at
    `PGA2k_gen.py:2009` (a second copy at :5370), `step_write_objects`,
    `step_refine_terrain` at `PGA2k_gen.py:3816`) and
    `PGA2k_gen_gui.py`. Start with:
    `grep -n 'game_version\|is_v2021\|is_v2019' PGA2k_gen.py PGA2k_gen_gui.py`.
  - **Steps**:
  - [x] 1.2.1 Audit every `game_version == "2019"` / `!= "2019"` / `is_v2021`
        branch (notably `PGA2k_gen_gui.py:2639` — the v2021+ picker gating
        uses `!= "2019"` so v2023 falls into the v2021 side automatically;
        `step_write_objects`; `step_refine_terrain` at `PGA2k_gen.py:3816`)
        and decide v2023's behavior in each. Prefer `schema_for(v).has_*`
        flags over string comparisons where a capability drives the branch.
  - [x] 1.2.2 Add a v2023 branch to the `game_version` resolution/error path
        (`PGA2k_gen.py:2009` and the second guard at :5370 — "isn't implemented
        yet") so v2023 is no longer rejected.
  - **Done when**: `python PGA2k_gen.py --game-version 2023 ...` passes version
    resolution (fails later only if a template isn't there yet — that's 1.3),
    and no `grep` hit for a version branch leaves v2023 unhandled.
  - **Decisions to record**: a one-line table (branch location → v2023
    behavior) appended to `V2023_TASKS.md` under this task, so Phase 2/3
    sessions don't re-audit.
- [x] **1.3 Add a v2023 theme/template baseline** — DONE 2026-09-23
  (`templates/2023_rustic.course`; CLI `--game-version` choices come from
  `GAME_VERSIONS`, and the GUI selector/tooltip read the registry). Checked
  with `ingest-course` → `push-blank-template` → `repack` on a
  `game_version=2023` project. **In-game VERIFIED 2026-09-23 (Andy):** a
  `push-blank-template` output (`LIDAR-2023-20260923183452`) opens in v2023.
  - **Context**: version resolution needs a resolvable template file to go
    end-to-end.
  - **Where**: `templates/` (existing `2019_*.course`, `2021_*.course`, and the
    reference `2023_fences.course`), `resolve_course_template` (grep in
    `PGA2k_gen.py`), the GUI version selector + tooltip (reads
    `IMPLEMENTED_GAME_VERSIONS` / `THEMES_V2019`), and the CLI
    `--game-version` argparse choices.
  - **Steps**:
  - [ ] 1.3.1 Confirm which themes exist in v2023 (do v2019/v2021 theme IDs
        carry over? new set?). *Partial:* the sample's `theme: 11` = rustic,
        same id as v2021. The full v2023 theme list needs Andy (in-game).
  - [x] 1.3.2 Add `templates/2023_{theme}.course` (from the confirmed v2023
        template file) so `resolve_course_template` can resolve v2023.
        *Rustic only*, same as 2019/2021. More themes = save a blank per
        theme from the game.
  - [x] 1.3.3 Expose v2023 + its theme list in the GUI version selector and
        tooltip (currently reads `IMPLEMENTED_GAME_VERSIONS`/`THEMES_V2019`).
  - [x] 1.3.4 Expose v2023 in the CLI `--game-version` choices/help.
  - **Done when**: `resolve_course_template("2023", theme)` returns a real file
    for every advertised theme; GUI + CLI both list v2023.
- [x] **1.4 Extend the asset catalog for new v2023 assets** — DONE
  2026-09-23 for everything in the sample. 10 new entries (4 fences, the
  cart, 5 hole signs) with `type: null` + `min_game_version: "2023"`: they
  have no v2019 id, so `(category, type)` lookups now build from
  `V2019_KEYED_ENTRIES` and `_resolve_v2019_key` skips type-null entries.
  `fence_options` sits on all 8 fence posts (`FENCE_ENTRIES`). Stone wall:
  spacingRule complete (author); its other rules only have the sample's
  values (`complete: false`) — they need the in-game trial (0.3a.1). Assets
  outside the sample still need an in-game capture.
  - **Context**: v2023 adds fence/wall + prop assets (full list in
    `V2023_SCHEMA.md` "Fence/wall assets" + "Props seen in the sample"); the
    builder needs them in the catalog before Phase 2/3 can select them.
  - **Where**: `course_output/asset_catalog.json` + `asset_catalog.py` (read
    both first — learn the existing entry shape and the v2019/v2021→v2023
    mapping pattern).
  - **Steps**:
  - [x] 1.4.1 Add the new v2023 assets (from `V2023_SCHEMA.md`) to
        `course_output/asset_catalog.json` / `asset_catalog.py` — fence/wall
        materials (the 8 in the sample — see `V2023_SCHEMA.md` "Fence/wall
        assets": StoneWall, WoodFences, CanvasFence Red/Black, Hedge,
        Asia_BrickWalls, UniFence, RetainWall; the sample is fence-focused,
        check the game for the full set), plus carts/hole-signs from
        `Assets/CourseProps/**` if the catalog covers props.
  - [x] 1.4.2 Record v2019/v2021→v2023 asset-path mappings for shared props
        (the existing catalog pattern, e.g. the fence-post stake mapping) so
        a prop re-written across versions stays the identical asset.
        *Identity mapping:* all 4 shared wall posts appear in the v2023 sample
        under exactly their v2021 catalog paths (recorded in the catalog's
        `v2023_note`).
  - [x] 1.4.3 **Fence option matrix** (data): add a per-fence-asset field for
        the allowed `spacingRule`/`hasCurves`/`heightRule` set — seed from the
        author's statement (stone wall = all 4 caps) and the sample rows, mark
        unknown cells explicitly; this feeds 3.2.3 validation and 0.3a.1.
  - **Done when**: `asset_catalog` loads clean; every v2023 fence/wall asset
    path from `V2023_SCHEMA.md` is present and resolvable; the option-matrix
    field exists with stone wall fully populated.

**Exit:** `--game-version 2023` is accepted end-to-end and resolves a template,
even if fence generation is not yet wired (existing features re-target to v2023).
**MET 2026-09-23**: the v2023 blank round-trips into the game. Still open
from Phase 1, none of it blocking Phase 2: the full v2023 theme list (1.3.1)
and per-fence option confirmation (0.3a.1).

---

## Phase 2 — Re-target existing features to v2023 (before new feature)

Goal: prove the whole *existing* pipeline (trees, clusters, splines, holes,
water, parking, range nets) emits correct v2023 output, using the per-version
builder convention. This de-risks Phase 3 by isolating the new-feature work.

- [x] **2.1 Per-version builders for placed objects** (`course_output/objects.py`)
  — DONE 2026-09-23. The only v2023 difference is the group envelope
  (`objectPaths` + `IsEmpty` on every group). Entry shapes are unchanged, so
  **no `build_X_v2023` copies**: `placed_object_groups_to_v2023` runs once
  after `merge_object_groups` in `_build_placed_objects` and
  `_inject_collection_into_course` (gated on `has_fences`).
  `merge_object_groups` skips scalar fields (it used to crash on `IsEmpty`).
  v2023-first assets are usable by path through collections/in-game
  imports (shawnee run: 16 GolfCart items, hole signs with `OffsetIndex`).
  Details and evidence are in `V2023_SCHEMA.md` "Phase 2 re-target decisions".
  - **Context**: Phase 1 done — v2023 resolves end-to-end. Now make the
    existing object-emitting builders produce v2023 records.
  - **Where**: `course_output/objects.py` — read the `build_tree_objects_v2019`
    / `build_tree_objects_v2021` pair first (that's the pattern to copy),
    plus the dispatch in `step_pack_objects`/`step_write_objects` in
    `PGA2k_gen.py`.
  - **Steps**:
  - [x] 2.1.1 Follow the `build_tree_objects_v2019` / `_v2021` pattern: add
        `build_tree_objects_v2023` (and any `_v2023` counterpart for stakes,
        clusters, spline-fill records, stream drops).
  - [x] 2.1.2 v2023 uses the same asset-path scheme as v2021
        (`theme_scheme="path"`) — reuse/extend the v2021 asset-path resolution
        rather than guessing a numeric triple; confirm no new required fields
        in the diff (Phase 0.2.2).
  - [x] 2.1.3 Wire the v2023 builders into `step_pack_objects` /
        `step_write_objects` dispatch.
  - [x] 2.1.4 Make sure v2023-only assets (from 1.4) are selectable/usable in
        the builders' asset pools.
  - **Done when**: running the pipeline for v2023 emits
    `CourseDescription_nodes/placedObjects3.json` with valid tree/stake/cluster
    records (spot-check 2–3 entries against the v2023 sample's entry shapes —
    same keys, same path-scheme).
- [x] **2.2 Re-target the other node writers** — DONE 2026-09-23.
  holes/userLayers/OOB were already version-aware from Phase 1. Added the
  v2023 blank-userLayers fallback. Water entries keep the v2021 shape
  (unconfirmed, pending the 2.3.2 load). splines/parking/range_nets have no
  version-specific shape (parking and range nets reach placedObjects3 as
  collection objects). 2.2.2 verified: a 2021 `course/` baseline with
  `game_version=2023` makes `repack` refuse (`holes.json,
  surfaceSplines.json, userLayers.json` flagged stale). Every template
  carries `userLayers(2).json`, so a 2021↔2023 switch is always caught.
  - **Context**: everything that writes a node file must honor the v2023
    renames + shapes from `V2023_SCHEMA.md` "Node renames".
  - **Where**: `course_output/` — `holes.py`, `userLayers.py`, `splines.py`,
    `water.py`, `parking.py`, `range_nets.py`, `out_of_bounds.py`; grep each
    for `2021`/`2019` branches and hardcoded filenames.
  - **Steps**:
  - [x] 2.2.1 `holes.py`, `userLayers.py`, `splines.py`, `water.py`,
        `parking.py`, `range_nets.py`, `out_of_bounds.py` — add/verify v2023
        branches per the confirmed schema (`holes2`, `surfaceSplines2`,
        `userLayers2`; water keeps `surfaceCategory: 9`).
  - [x] 2.2.2 Confirm the v2023 `objects_filename` is what gets written and
        that a stale v2019/v2021 node file in `course/` doesn't leak into the
        repack (see the existing "leftover different-game_version file" note).
  - **Done when**: a v2023 run produces every node file the v2023 sample has
    (`V2023_SCHEMA.md` "Confirmed node set") with v2023 names, and a
    pre-existing 2021 node file in `course/` is either cleaned or not
    repacked.
- [x] **2.3 End-to-end verification (no fence feature yet)** — DONE
  2026-09-23. **In-game VERIFIED (Andy):** the course opened and every
  item on the checklist below rendered correctly. Test course:
  a copy of Andy's shawnee project (v2021 → 2023 via `ingest-course`), run
  through write-terrain/-water/-splines/-holes/-objects + repack →
  `LIDAR-2023-shawnee-phase2.course` (copied into the v2023 Courses folder).
  Node set = the sample's set plus `holes2`/`surfaceSplines2`; base keys and
  `version: 31` match. Shape diff vs `2023_fences` shows only expected gaps:
  `objectPaths`/`surfaces` cat-5 (Phase 3), real `position.y` on
  waterfalls/elevated collection members, int `OffsetIndex`.
  `import-ingame-edits` of the output against the project reports "No
  differences".
  **In-game check (Andy):** load `LIDAR-2023-shawnee-phase2` in v2023 and
  confirm: trees (1563), 4 ponds + 179 stream water tiles, 129
  waterfalls/splashes, 18 holes with pins, surface splines, building stakes,
  object-spline fills (1227) + 12 cluster stamps, parking cars/golf carts,
  range nets, hole signs showing the right numbers, and collection props at
  their designed elevation.
  - **Context**: gate for Phase 3 — proves the whole pre-existing pipeline
    emits correct v2023 output.
  - **Where**: full CLI pipeline (`PGA2k_gen.py`) on a test course with trees,
    water, holes, parking, range nets; the v2023 game for the load check.
  - **Steps**:
  - [x] 2.3.1 Run the full CLI pipeline targeting `--game-version 2023` on a
        test course; confirm each node file is produced.
  - [x] 2.3.2 Load the generated `.course` in the v2023 game and confirm trees,
        water, holes, parking, range nets render correctly (spot-check a few).
  - [x] 2.3.3 Diff generated v2023 node files against the confirmed reference
        to catch structural drift.
  - **Done when**: in-game load succeeds with all pre-existing features
    rendering; structural diff vs `2023_fences.course` node files shows no
    unexpected key/shape differences (record any legitimate ones in
    `V2023_SCHEMA.md`).

**Exit:** a v2023 `.course` with all *pre-existing* features is playable in-game.
**MET 2026-09-23** (`LIDAR-2023-shawnee-phase2`, Andy). Phase 3 is unblocked.

---

## Phase 3 — Spline fences and walls (headline feature)

Build only after Phase 2 proves the plumbing; the core schema is confirmed
(`V2023_SCHEMA.md`) — remaining unknowns are per-asset option matrix and
`flexibilityRule`/`state` semantics (0.3a), which affect parameter *validation*,
not the builder's basic shape.

**Confirmed mechanism (from the sample course):** fences/walls are
`Value.objectPaths[]` entries in `placedObjects3.json` — a spline path with a
fence material (asset path under `Assets/CourseGen/Detail/Walls/**`) plus
per-path rule fields:

```
objectPath = {
  path: { waypoints: [{pointOne:{x,y}, pointTwo:{x,y}, waypoint:{x,y}}...],
          width: float, hasCurves: bool, state: int },
  height: float,          # vertical offset (negative = buried)
  spacingRule: int,       # 0 none / 1 ends only / 2 spaced / 3 spline points (JSON values)
  flexibilityRule: int,   # TBD (0/1 observed)
  heightRule: int,        # 0 = contoured (follow terrain), 1 = stepped
  spacing: float          # post/panel spacing (m)
}
```

Waypoints are 2D `{x, y}` with `y` = course Z (matches the items frame). The
parking-style object tiling (placed `items[]`) remains a valid *alternative*
per the course author, but objectPaths is the native game-written mechanism —
prioritize it; treat the object-tile version as a later optional variant.

- [x] **3.1 OSM input path** — DONE 2026-09-23: kinds `fence`/`wall`/`hedge`
      (all lines), routing in `course_output/fences.py`, probe
      `util/probe_fence_tags.py` passes 20/20; decisions + mapping table in
      `V2023_SCHEMA.md` "OSM fence/wall ingest + asset routing".
  - **Context**: Phase 2 done. OSM fence/wall ways must become features the
    fence builder can consume. Today `classify_way` returns `None` for every
    fence/wall/hedge tag (verified this session by probing
    `ingest.osm.classify_way` inline).
  - **Where**: `ingest/osm.py` — read `classify_way` + the existing `kind`
    assignments (water/hazard etc.) to copy the pattern; grep every `kind`
    consumer before adding a new one.
  - **Steps**:
  - [x] 3.1.1 Add `classify_way` branches for fence/wall tags
        (`barrier=fence`, `barrier=wall`, `barrier=hedge`, `barrier=chain`,
        `natural=hedge` — probe first; today they all return `None`).
  - [x] 3.1.2 Assign a `kind` string to each (e.g. `fence`, `wall`, `hedge`)
        and confirm it routes to exactly the v2023 fence consumer and no
        unintended existing consumer (grep every kind consumer before adding).
  - [x] 3.1.3 Decide area vs line semantics per tag (fence/wall are lines; a
        hedge could be a closed area — confirm against OSM conventions and the
        target rendering).
  - [x] 3.1.4 Tag→asset routing: map OSM tags to a v2023 fence asset
        (using the asset catalog from 1.4) — e.g. `material=stone` →
        `StoneWall*`, `material=wood` → `WoodFences*`, `material=chain_link` →
        `UniFence*`, `natural=hedge` → `HedgeSpline*`, `wall=brick` →
        `Asia_BrickWalls*`; record the mapping table in `V2023_SCHEMA.md`.
  - **Done when**: a small probe script (create it under `util/`, e.g.
    `util/probe_fence_tags.py` — none exists yet) on an OSM extract containing
    fence/wall ways shows the ways classified to the new kinds and routed to
    the expected asset paths.
- [x] **3.2 Feature → objectPath record builder** — DONE 2026-09-23:
      `course_output/fences.py` (`build_fence_records`,
      `fence_records_to_groups_v2023`, presets, `pga_fence_*` overrides,
      matrix validation, same-style way joining); `python -m unittest
      tests.test_fences` 15/15. Decisions in `V2023_SCHEMA.md` "objectPath
      handle rule + builder decisions". `state`=closed is inferred — check
      in 3.5.3.
  - **Context**: 3.1 done — fence ways are classified features. Now convert a
    feature line into the exact `objectPath` record shape (in
    `V2023_SCHEMA.md` "objectPaths rule fields").
  - **Where**: new `course_output/fences.py` (or `splines.py` addition —
    decide by reading which file owns `Value.splines[]` emission and whether
    fence output belongs with it); the resample/simplify helpers already used
    for OOB (`course_output/out_of_bounds.py`) and water.
  - **Steps**:
  - [x] 3.2.1 Feature line → objectPath record. Waypoints: resample/simplify
        the OSM way (Douglas-Peucker, as elsewhere) into 2D `{x, y}`
        waypoints in the course frame; set `pointOne`/`pointTwo` handles per
        the curve style (0.3a.6 — confirm handle semantics before emitting
        curves).
  - [x] 3.2.2 Map fence style → rule fields: `spacingRule` (cap style),
        `heightRule` (contoured vs stepped), `hasCurves`, `spacing`, `width`,
        `height` offset. Defaults per asset from the sample rows
        (`V2023_SCHEMA.md` matrix).
  - [x] 3.2.3 **Per-asset option validation**: enforce the option matrix
        (1.4.3) — warn/skip when an OSM-tag-requested option isn't valid for
        the chosen asset. Keep the matrix data-driven (asset_catalog.json) so
        it grows as 0.3a completes.
  - [x] 3.2.4 Support the "trick" recipes as named presets (sample-proven,
        `V2023_SCHEMA.md` "Tricks"): curb (buried stone, h≈-1.5), railroad
        (buried black canvas, h≈-2.69), retaining wall (`RetainWall*`,
        w=2.5, h≈-0.7, contoured, 4 wp, state=1).
  - [x] 3.2.5 Handle closed vs open fence runs and shared corners between
        adjacent ways (no double posts / no gaps).
  - **Done when**: unit test feeds a synthetic way (straight + curved, open +
        closed) and produces objectPath records byte-comparable in structure
        to the sample rows (same keys, 2D waypoints, handles populated).
- [x] **3.3 Pipeline wiring** — DONE 2026-09-24 (code-level): `--step
      generate-fences` → `fences.json` → write-objects (v2023 only) → repack;
      GUI Objects → "Fences & Walls (v2023)" panel. Done-when met on a
      `bouldercreek2` copy with injected fence ways (5 objectPaths + 3150
      items in the repacked v31 course). In-game check is 3.5. Decisions in
      `V2023_SCHEMA.md` "Fence pipeline wiring".
  - **Context**: 3.2 done — records build in isolation; now wire them into
    the pipeline like range nets.
  - **Where**: `PGA2k_gen.py` — read `step_generate_range_nets` +
    `step_write_objects`/`step_pack_objects` + `step_repack` as the pattern;
    `PGA2k_gen_gui.py` for the step UI.
  - **Steps**:
  - [x] 3.3.1 `step_generate_fences` in `PGA2k_gen.py` following the
        `step_generate_range_nets` pattern (kind → features.geojson → tiler).
  - [x] 3.3.2 Emit objectPath entries into `placedObjects3.json` — merge with
        the existing `step_write_objects`/`step_pack_objects` output for the
        same Key.path (fence assets get their own Key entries alongside
        items/clusters/splines) and include in `step_repack`.
  - [x] 3.3.3 GUI button/menu entry + CLI step, with per-way style options
        (cap style, contoured/stepped, curved/straight, spacing, burial depth).
  - [x] 3.3.4 Persist derived state (origins, frame, chosen assets) in
        `project.json` per the state-bus convention.
  - **Done when**: a v2023 run with fence ways in the OSM extract produces a
    repacked `.course` whose `placedObjects3.json` contains the fence
    objectPaths alongside normal items.
- [x] **3.3a (DONE 2026-09-27: filename rule) generated courses fail to load in v2023** —
      fixed by `game_safe_course_stem`. Closed by a full pipeline build
      repacked under a formerly failing name
      (`LIDAR-2023-bouldercreek-fences` → `LIDAR_2023_bouldercreek_fences`,
      3.5.2), which loads in-game (Andy, 2026-09-27).
      Investigation history:
  - Symptom: every course built this session shows in the v2023 course list but
    "Failed to load"; in-game **Delete also fails** on them. A blank pushed via the
    GUI's "Push Blank to Game" loads fine. `LIDAR-2023-shawnee-phase2` still loads.
  - **Not the fences**: `FT2-1-S-compact` (shawnee, repacked with no content change,
    no fences) fails too; so do all no-fence variants.
  - **Ruled out**: file ACLs / Mark-of-the-Web (identical to working files);
    container format (FT3-1/2 match the game's gzip headers, BOMs, compact JSON,
    Thumbnail `_id`, exactly); id convention (`offlineSave<filename>` tried, FT3-1);
    JSON spacing/size (FT2-1 compact, 19.9M chars like shawnee). Unity
    `Player.log` has no course-load error at all (game gives up before reading?);
    the profile `.data` holds no course list.
  - Every file that loads was last written by the game itself (compact JSON, gzip
    flg=0, id `offlineSave<filename>`); shawnee-phase2 and the 9/23 blank were
    our pipeline output when first loaded, then re-saved by the game.
  - Not yet tested: whether a *full* course (not a blank) freshly built by the
    normal GUI flow (Repack → Copy to Game Folder) loads today; whether size
    matters (all failures ~1.4–1.6 MB, working blank ~40 KB); diff the
    shipped `blank_template.course` against its game-rewritten copy to see what
    the game changes.
  - Test files to delete when done (game UI can't): `LIDAR-2023-bouldercreek-fences`,
    `FenceTest-*`, `FT2-*`, `FT3-*`, `BX*`, `BY*`, `BZ*`, `FENCETEST_*` in `Documents/My Games/PGA TOUR 2K23/Courses`.
    Build scripts: session scratchpad (`make_variants*.py`) — not in repo.
  - **2026-09-25 session 2 findings (offline, no game run):**
    - **Lead: compression level.** The v2023 writer uses **zlib level 1** for
      every gzip layer (header flg=0 xfl=0 os=255, mtime = local wall-clock s).
      Level 1 reproduces shawnee-phase2, the rewritten pushed blank and
      `2023_fences` **byte for byte** (scratch `gamefmt.py`). `course_repack.py`
      uses `gzip.compress` (level 9), and the FT3-1/2 "gamefmt" variants used
      level 6, so **no large test file matched the game's bytes**. CD
      decompression ratio: game ≈26×, all failures 32–39×; blanks ≈5× (so the
      level wouldn't matter for them if the game caps the ratio or pre-sizes
      its buffer).
    - Caveat: the 9/23 shawnee-phase2 first load was reportedly level-9
      pipeline output, which contradicts this. No pre-game copy survives to
      check it.
    - Ruled out: content — FT2-1 nodes == shawnee-phase2 nodes after JSON
      normalization; only `_id`/name/timestamps differ. Also ruled out: a
      game-side course registry or cache (profile `.data` blobs have no
      `offlineSave` ids; nothing else under AppData/Steam userdata/Documents
      was written 16:10–16:45); a game update (Steam `LastUpdated` = May 2026);
      and file attributes and ADS streams.
    - Loaded blank: push wrote a random `offlineSave<hex>` id, and the game
      rewrote it to `offlineSave<filename>` (+ Thumbnail `_id`) on load. So
      random ids are fine.
    - IL2CPP metadata shows a course budget (`IsCourseTooLarge`,
      `FileSizeCostLimit`, `ObjectCostLimit`, `MemoryCostLimit`). It's content-
      based, so it can't explain FT2-1 vs shawnee (identical content), but
      remember it if a big course fails after the encoding fix.
  - **Bisection set built (in game folder); in-game check needed.** Restart
    the game first so every file is picked up in a fresh scan:
    - `BX1-shawnee-level1`: shawnee-phase2 re-encoded game-exact; only the
      id/name are new.
    - `BX2-shawnee-level9`: byte-for-byte same text as BX1, but the
      CourseDescription blob uses level 9.
    - `BX3-ourfences-level1`: FT2-4 content (our 5 fence objectPaths),
      game-exact encoding.
    - `BX4-bcfences-level1`: `LIDAR-2023-bouldercreek-fences` content (full
      pipeline output + fences), game-exact encoding.
    - Read: BX1 loads + BX2 fails ⇒ compression level confirmed → fix
      `course_repack.py` (level 1 + game header, compact JSON). BX1 fails ⇒
      "new file/id" is the problem, not encoding. BX1+BX2 both load ⇒ the
      encoding doesn't matter; look at what BX3/4 do.
  - **BX result (Andy, 2026-09-26): BX1–BX4 all load; all FT2-* still fail;
    restarting the game makes no difference.** BX2 was still level 9 when it
    loaded, so **the level is ruled out**. (The game re-saves a course on
    load, and BX1–4 are level 1 now.) BX3/BX4 prove that fence objectPaths
    load in v2023, and they're ready for the 3.5.3 in-game check. The 5 test
    runs are synthetic lines near the hole 18 tee (shawnee) / hole 2 tee
    (bouldercreek).
  - **Round 2, BY1–BY4:** each starts from game-exact BX1 and switches ONE
    factor to FT2-1's form. BY1 = CourseMetadata written with spaces
    (`": "`). BY2 = `course_repack.py` outer layer (`gzip.open`, level 9,
    FNAME header). BY3 = FT2-1 thumbnail (no BOM, timeStamp-first key
    order). BY4 = FT2-1 CourseDescription text (repack moves nodes last).
    The one that fails is the cause. The loading blank also has
    BY1–BY3's traits, but its metadata has no tee/pin objects; best guess is
    BY1.
  - **BY result: all four fail.** Each differed from loading BX1 by a single
    factor, so none of those factors is the cause. **New hypothesis: the
    name.** Across all 26 files tested, every loader's filename ends in a
    digit (`2023_fences` is the exception, but its stored CD name ends in a
    digit), and every failure's filename and CD name both end in a letter.
  - **Round 3, BZ1–BZ5** (game-exact BX1 content; only filename/`_id`/name
    change): BZ1 = letter/letter (predict fail). BZ2 = digit/digit (load).
    BZ3 = filename digit, name letter (load). BZ4 = filename letter, name
    digit (load). **BZ5 = byte-for-byte copy of failing FT2-1, renamed to end
    in a digit (load)** — the decisive one.
  - **BZ result (Andy, 2026-09-26): ROOT CAUSE = FILENAME.** BZ1 fails, BZ2
    loads, BZ3 loads, BZ4 fails. BZ5 (the renamed FT2-1 bytes) loads, and it
    lists as "FT2-1-S-compact". Andy's renamed `hinckleyhills` / `hhills3`
    load too. Rule that fits all ~30 files: **a filename with a hyphen whose
    last `-` segment is all letters fails, for both load and delete.** The
    stored name (CD + CM) doesn't matter. Nor does the encoding (every BY file
    failed only because of its `-spaced` / `-ours` style name). Likely cause:
    the game names parts `<id>-Meta` / `<id>-Thumb` and strips a `-<Letters>`
    tail as a part suffix.
  - **Fix (code, 2026-09-26):** `game_safe_course_stem` (hyphens →
    underscores) is applied in `step_repack` (the saved `repack_filename` is
    the safe one, so Copy to Game Folder follows) and in the GUI's push-*
    copies. `course_repack.py` is unchanged. **Remaining check:** load a
    normal pipeline build repacked under a formerly failing name (e.g.
    `LIDAR-2023-bouldercreek-fences` → `LIDAR_2023_bouldercreek_fences`).
  - **Fences in BX3/BX4 load but aren't visible.** Frames are consistent
    and the JSON matches the sample key for key. Offline evidence for
    burial: the terrain under BX3's fences is ~287 m of flatten stamps,
    while the blank's datum is 3.23 m and every sample fence sits near it.
    If objectPaths are grounded on base terrain, ours are ~284 m
    underground. Tested by the 3.5 harness below (`push-fence-test`, burial
    rows D1–D3).
- [ ] **3.3b Optional variant: object-tile fences** (deferred; only if wanted)
  - **Where**: `course_output/parking.py` / `range_nets.py` — the
    object-tile-along-line pattern (tiler places post/panel `items[]`).
  - **Steps**:
  - [ ] 3.3b.1 Tiler following the `parking.py`/`range_nets.py`
        object-tile-along-line pattern, placing fence post/panel `items[]`
        (reuse existing tiling/rotation — no hand-rolled sampling).
  - [ ] 3.3b.2 Per-way choice between objectPaths vs object-tile representation.
- [ ] **3.4 Fill splines with "clear generated objects" stamps (new feature)**
      — **schema confirmed** (0.3b): `userLayers2.json` → `surfaces[]` entries
      with `surfaceCategory: 5`, OOB-shaped stamps (`type` 8 round / 15 smooth
      square, `scale` = centre-to-edge in m, the same convention as height stamps,
      `value` 1.0, `tool` 0, y "-Infinity"). 0.3b.1/0.3b.2 are closed
      in-game (2026-09-27): the paint clears only the game's procedural
      scatter; categories are 5 objects / 6 trees / 11 heavy rough (type 72,
      value 2.0). Only 0.3b.3 (v2019/v2021 support) is still open.
  - **Context**: Phase 2 done; schema confirmed from the second sample export
    (commit `8376f0e`, see `V2023_SCHEMA.md` "Clear-generated-objects paint").
  - **Where**: `course_output/out_of_bounds.py` is the blueprint (brush-stamp
    emission along a shape); `course_output/water.py` already writes
    `surfaceCategory: 9` stamps into the same `surfaces` array (read how it
    merges); `step_write_terrain` in `PGA2k_gen.py` for the fold-in point.
  - **Steps**:
  - [ ] 3.4.1 **Record builder**: follow the `out_of_bounds.py` blueprint —
        version-agnostic frozen records in their own file
        (e.g. `clear_objects.json`), entries formatted for the target layer by
        a `_v2023` (or shared) formatter that emits the exact confirmed shape
        (`surfaceCategory: 5`, brush `type`, `scale`, `value: 1.0`).
  - [ ] 3.4.2 **Input path**: how the fill area is designated — OSM tag on a
        closed way (e.g. `pga_clear_objects=yes`)? a spline drawn in the GUI?
        (mirror however OOB gets its shape — check `step_generate_oob` input).
        Note the sample uses *brush stamps*, not a spline outline — the
        generator must either emit a stamp chain along the region boundary
        (OOB does exactly this: round caps per vertex + stretched squares per
        edge) or a coarse stamp grid over the region; decide which and record
        the decision in `V2023_SCHEMA.md`.
  - [ ] 3.4.3 **Merge into userLayers**: `step_write_terrain` (or the
        userLayers writer) must append these entries to the existing `surfaces`
        array alongside water entries (`surfaceCategory: 9`) — same layer,
        different category; confirm no key-ordering or dedup assumptions break.
  - [ ] 3.4.4 **Generator-side suppression** (the core logic): when
        `step_generate_trees` / other object-generating steps place objects,
        test each candidate point against the clear-objects region (point-in-
        region test against the filled area) and skip it. Centralize the
        containment test so every generator uses the same rule. (This is
        independent of the paint: it keeps our generator consistent with areas
        painted in-game, and gives us a containment primitive to reuse.)
  - [ ] 3.4.5 **Scope decision**: in-game, the paint only clears the game's
        *own* procedural scatter; our placedObjects3 trees are unaffected
        (0.3b.1). So the generator-side suppression (3.4.4) is a choice we
        make, not something the game does for us. Also decide which categories
        to emit: 5 objects, 6 trees, 11 heavy rough. Also decide whether it
        applies in v2023 only or also v2019/v2021 (pending 0.3b.3 —
        does it apply in v2023 only or also v2019/v2021 (pending 0.3b.3 —
        gate the v2019/v2021 wiring on that answer).
  - [ ] 3.4.6 **Wiring**: `step_generate_clear_objects` (+ clear flag,
        mirroring `step_generate_oob`/`_clear_oob`), fold into
        `step_write_terrain` like OOB is (`oob_entries` pattern), `project.json`
        enable flag.
  - [ ] 3.4.7 **GUI**: brush/draw area + apply/clear controls, mirroring the
        OOB UI.
  - [ ] 3.4.8 **Verification**: paint a clear region over a tree-dense area,
        regenerate, confirm generated trees/objects are absent inside the
        region and present just outside it; confirm manually placed objects are
        unaffected; also load the generated course in-game and confirm the
        paint is visible/enforced (closes 0.3b.1/0.3b.2).
  - **Done when**: in-game, the painted region visibly excludes generated
    objects inside and leaves outside areas untouched; `V2023_SCHEMA.md`
    records which object kinds are suppressed and the v2019/v2021 support
    answer (0.3b.3).
- [ ] **3.5 Fence + clear-objects verification**
  - **Context**: everything in Phase 3 built; this is the gate before
    declaring v2023 done.
  - **Harness (2026-09-26):** `--step push-fence-test` / GUI Objects →
    Fences → "Push Fence Test to Game". It builds a blank v2023 course with
    labelled rows from game-frame (-900, 900) (`--fence-test-origin`; the
    grid is ±1000 — a first build at (-1900, 1900) was off-map and showed
    nothing, and the step now refuses off-map origins), and
    `fence_test_legend.txt` in the working dir maps each row to its
    position:
    - A: the sample's own 13 objectPaths as a control.
    - B1–B20: every asset at its defaults, then BrickWall spacingRule 0–3 /
      stepped / straight, WoodFences spacingRule 0/1/3 (unverified), and the
      3 presets.
    - C: an orientation L.
    - D: a +15 m flatten pad with runs on it (D1), lifted +15 (D2), and
      across its edges (D3).

    Build 1 at (-900, 900): `FENCETEST_2023_20260926173345`. **Andy's
    result: all of A, B, C, D render correctly.** Details are in
    `V2023_SCHEMA.md` "In-game fence test results". Summary:
    - Fences follow height stamps, so burial is ruled out.
    - spacingRule (JSON): 0 none, 1 ends only, 2 spaced, 3 spline points (E3/E4). WoodFences takes
      all 4.
    - Stepped at height 0 is invisible (B13).
    - The retaining wall was 0.3 m high; it's now -1.015.

    Build 2: `FENCETEST_2023_20260926180724`, which adds the E block. **Andy's
    results, all as expected:**
    - E1 (stepped at datum) sits on the surface; E2 (datum+3) floats. So
      stepped `height` is absolute.
    - E3 (spacingRule 1) shows ends only; E4 (3) shows spline points.
    - E5 (curved) is curved; E6 (straight) is a V.
    - The retaining wall (B8/B20 at -1.015) is flush.
    - **The fences in the BX3/BX4 full courses render too**, near the 18th
      hole. They'd only been looked for at the wrong spot.

    That settles 3.5.4 (presets) and the harness side of 3.5.3. What's left
    is a real-OSM pipeline build (3.5.2/3.5.3).
  - **Where**: the tag probe script from 3.1 (e.g. `util/probe_fence_tags.py`),
    a real OSM extract with fence/wall ways, the v2023 game.
  - **Steps**:
  - [x] 3.5.1 Extend the 3.1 probe to cover a representative set of
        fence/wall/hedge tags end-to-end (tag → kind → objectPath record).
        DONE 2026-09-27: 31 synthetic cases, routing to fence types.
  - [x] 3.5.2 Generate a course with real OSM fence/wall ways; diff emitted
        objectPaths against `2023_fences.course`'s rows (same rule-field
        shapes, sane waypoints). DONE 2026-09-27 (code-level):
    - Andy hand-drew 70 barrier ways into `bouldercreek2/map.osm` (the
      real OSM had none).
    - Andy's fence sampler (`LIDAR-2023-20260927105132`) gave all 28 menu
      types. They're now `FENCE_TYPES`; see `V2023_SCHEMA.md` "Fence
      types". Picket and the two railing walls are 2-part.
    - Any `barrier=*` way is now a fence (gates excepted), and
      `barrier=range_net` is accepted as a range-net alias.
    - `chain_link` routing fixed: it went to UniFence ("Fence - metal") and
      now goes to WireFence.
    - Pipeline: `ingest-course` (the reset) → `ingest-osm` → write-terrain/
      water/splines/holes → generate-range-nets → generate-fences →
      pack-objects → write-objects → repack as
      `LIDAR-2023-bouldercreek-fences`, which came out as
      `LIDAR_2023_bouldercreek_fences.course`.
    - Diff (scratch `diff_fences.py`): 69 objectPaths. All have the same
      value types and key order as all 44 game-written rows (2023_fences +
      sampler). The group envelope matches and the file is v31. No
      off-grid, zero-length or bad-handle waypoints; 3 closed runs.
      Alongside: 3150 trees + 178 range-net items.
    - Rule fields equal the sampler's menu row for every asset except stone
      wall and retaining wall. Those keep the defaults the 9/26 fence test
      verified.
    - The file has been copied to the game's Courses folder.
  - [x] 3.5.3 Load in the v2023 game: fences/walls render on their spline,
        grounded/contoured correctly, cap styles match `spacingRule`, stepped
        rows sit at the offset height; no collision with surface splines.
        **DONE 2026-09-27 (Andy: "course looks good").** Not covered by this
        course: leveled/stepped rows (3.5.6) and the multi-part types
        (picket, railings), which get checked in the 3.5.6 in-game pass.
        What was loaded:
        `LIDAR_2023_bouldercreek_fences` (Boulder Creek). This also closes
        3.3a, since it's a full pipeline build under a formerly failing name.
        Look for:
        - 57 hedge runs
        - 3 chain-link (WireFence) runs, 1 of them a closed loop
        - 3 wood 2-rail runs
        - 1 white canvas run
        - 1 wooden-panels run
        - 2 low classic brick runs
        - 1 stone wall
        - 1 closed retaining wall (33 nodes)
        - 2 range-net chains
        None are stepped (3.5.6 is still open). Also confirm that
        chain_link → WireFence reads as chain-link. The multi-part types
        (picket, railings) aren't in this course; test them by setting
        `pga_fence_asset` on a way.
  - [x] 3.5.4 Verify the trick presets (curb / railroad / retaining wall)
        in-game. Done via the fence test (B18–B20), 2026-09-26; the retaining
        wall height was corrected to -1.015.
  - [x] 3.5.6 **Leveled (stepped) fences need an absolute height** (found by the fence
        test). **DONE 2026-09-27:** Andy loaded `LIDAR_2023_bouldercreek_leveled`.
        The leveled picket (over a 1.5 m slope), leveled stone wall, and
        brick with railings all look correct.
    - Semantics (Andy, in-game): `heightRule=1` = ONE height for the whole
      run, which the editor sets to the run's minimum ground.
    - Implemented as `fences.apply_leveled_heights`, called from
      `_build_placed_objects` at write-objects: min terrain along the run
      (1 m sampling) + `output_height_shift_m` + the offset tag. See
      `V2023_SCHEMA.md` "Leveled fence height". The GUI's HEIGHT choice
      now reads "leveled".
    - Tests: `LeveledHeightTest` (5 cases). Checked on a bouldercreek2
      scratch copy: heights match an independent terrain sweep within 6 mm.
    - Done when: a leveled OSM fence in-game sits level at its run's low
      point, like one placed in the editor. Test it on a way over a slope
      (`pga_fence_heightRule=1`); the picket and railing parts should line
      up in the same load.
  - [ ] 3.5.7 (future, from Andy) **Retaining wall along pond edges.** The
        retaining wall draws a ground-textured cap tile about 1.9 m back
        from the spline, at cap height (Γ profile). Place the spline about
        2 m inside the water contour, and set `height` by trig from the bank
        slope over a ~1.5 m run so the tile clips into the terrain. See
        `V2023_SCHEMA.md` "In-game fence test results".
  - [ ] 3.5.5 Regression: confirm untouched features (trees, water, splines)
        are unchanged when fences/clear-objects are present vs absent.
    - **Fences: DONE 2026-09-27** on a bouldercreek2 scratch copy (71
      objectPaths, incl. leveled + multi-part), write-objects with vs
      without `fences.json`:
      - all 20 other nodes byte-identical (userLayers2, surfaceSplines2,
        holes2, ...);
      - in `placedObjects3`, the 15 non-fence groups (trees, range-net
        items) are identical in content and order; fences add only their
        own 12 groups.
    - Clear-objects half still open: re-run once 3.4 exists.

**Exit:** OSM fence/wall ways become correctly-grounded objectPath fences/walls
in a playable v2023 `.course` (trick presets included), clear-objects fill
suppresses generated objects where painted, with no regression to existing
features.

---

## Phase 4 — Texture painting (secondary v2023 capability, optional/deferred)

Only if a texture-painting schema is confirmed. Independent of fences.

- [ ] **4.1** Capture the paint schema from a reference `.course`
  (extract + diff per AGENTS.md ".course file format" section).
  - **Where**: a v2023 `.course` with texture paint (none in the repo yet —
    needs a new sample from the game), diffed against `2023_fences.course`.
- [ ] **4.2** Set `has_texture_paint=True` once implemented.
- [ ] **4.3** Build the paint generator + wire a step; verify in-game.
  - **Done when**: a v2023 course with painted texture regions loads and
    renders correctly in-game.

---

## Phase 5 — Docs + cleanup

- [ ] **5.1** Update `README.md` line 45 ("TODO: v2023, v2025") to reflect
        v2023 support (fences/walls + texture paint status).
- [ ] **5.2** Update `game_versions.py` and `objects.py` module docstrings to
        drop "not implemented yet" for v2023.
- [ ] **5.3** Update the `pga2k-terragen` skill with the confirmed v2023
        fence/wall schema + pipeline shape (so future sessions don't
        re-derive it).
- [ ] **5.4** Record the finished feature in `~/completed-tasks.md`.
  - **Done when (phase)**: all Phase 5 items checked; `git log` shows the
    v2023 work on `hermes-experiment` with no leftover "TODO: v2023" markers
    in docs that are now stale.

---

### Cross-cutting rules (apply to every task)
- Commit on `hermes-experiment`, never `main`.
- Per-version logic = explicit `build_X_v2023`, never a generic version-parameterized
  function (repo convention, see `game_versions.py` docstring).
- Grep every consumer of a feature `kind` before adding/renaming it.
- Verify with the repo venv, not system python (needs `overpy`/`shapely`/`pyproj`).
- Confirm the real schema before building — the whole point of Phase 0.
