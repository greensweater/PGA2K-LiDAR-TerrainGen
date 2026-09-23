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
  (none/spaced/points/ends)**, **`heightRule` 0=contoured / 1=stepped**,
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
  white/black — the 0N↔color mapping needs an in-game check).
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
meaning; `spacingRule`→cap-style ordering; in-game confirmation that category 5
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
  - [ ] 0.3a.3 **`spacingRule` → cap-style ordering**: which of 0–3 is
        none / spaced / points / ends.
  - [ ] 0.3a.4 **HoleSign0N ↔ color mapping** (author: N→S = green, red, blue,
        white, black; need in-game check to bind 01–05 to colors).
  - [ ] 0.3a.5 `holes2` / `surfaceSplines2` entry shapes (sample has both empty)
        — confirm vs `holes` / `surfaceSplines` once a populated course is
        available.
  - [ ] 0.3a.6 Confirm `pointOne`/`pointTwo` are bezier handles (vs endpoint
        duplicates) when `hasCurves=false`.
- [ ] **0.3b "Clear generated objects" paint — schema CONFIRMED (2026-09-23, second
      export, commit `8376f0e`)** — see `V2023_SCHEMA.md` "Clear-generated-objects
      paint" section. It reuses the `userLayers2.json` `surfaces` layer with
      `surfaceCategory: 5`; entries are OOB-shaped brush stamps (`type` 8 round /
      15 smooth square, scale = radius in m, value 1.0). Remaining verification:
  - [ ] 0.3b.1 **In-game check**: load `2023_fences.course` in v2023 — confirm
        the two stamps visibly suppress generated objects inside their area and
        do nothing outside it; note which object kinds are suppressed (trees?
        auto-placed props? all generated?).
  - [ ] 0.3b.2 Confirm the two stamp shapes render as expected (round stamp →
        circular exclusion; square stamp → square) and record the in-game
        semantics of `scale` (radius vs diameter) and `value`.
  - [ ] 0.3b.3 Check whether v2019/v2021 honor `surfaceCategory: 5` (v2021
        userLayers has the `surfaces` key, empty in the sample; v2021 also has a
        `clearTrees` key — determine if that is the same or a different
        mechanism).
  - [ ] 0.3b.4 Record answers in `V2023_SCHEMA.md` and lift the 3.4 gate.
- [x] **0.4 Document findings** — `V2023_SCHEMA.md` written; `VERSION_SCHEMAS`
      `2023` entry still to be populated (that is task 1.1, gated only on
      0.3b for the clear-objects capability flag).

---

## Phase 1 — Version registry + plumbing (low risk, unblocks everything)

- [ ] **1.1 Add the `2023` `VersionSchema` entry** (`course_output/game_versions.py`)
  - **Context**: Phase 0 is done (schema in `V2023_SCHEMA.md`). This task
    registers v2023 in the version table so every downstream `schema_for(v)`
    call has something to return.
  - **Where**: `course_output/game_versions.py` — read the whole file first
    (it's small): the `VersionSchema` dataclass fields, the `VERSION_SCHEMAS`
    dict (look at the `2021` entry as the template), `IMPLEMENTED_GAME_VERSIONS`,
    and the module docstring.
  - **Steps**:
  - [ ] 1.1.1 Set confirmed fields: `objects_filename="placedObjects3.json"`,
        `theme_scheme="path"`, **`holes_filename="holes2"`**,
        **`splines_filename="surfaceSplines2"`**,
        **`userlayers_filename="userLayers2"`** (v2023 renames the v2021
        nodes — see `V2023_SCHEMA.md` "Node renames" table). Audit every
        reader/writer of these filenames (holes.py, splines.py, userLayers.py,
        course_repack/extract, `_stale_version_node_files`) so the renamed
        nodes are produced, consumed, and stale-file-checked under the new
        names. Grep: `grep -rn 'holes\|surfaceSplines\|userLayers' --include='*.py' course_output/ util/ PGA2k_gen.py`
        (filter comments).
  - [ ] 1.1.2 **Verify `placedObjects3` placement in v2023**: the v2021
        template carries `placedObjects3` as a *base* key (empty, inlined in
        CourseDescription), while v2023's base does NOT list it — the sample
        has it only as a node file. Confirm in-game that a repacked course
        with `CourseDescription_nodes/placedObjects3.json` (and the key absent
        from base) is loaded, and adjust `_ensure_course_baseline` /
        `course_repack.py` accordingly if the game wants it in the base.
  - [ ] 1.1.3 Set capability flags: `has_fences=True` (objectPaths),
        `has_texture_paint` (deferred, Phase 4), and a clear-objects flag
        (e.g. `has_clear_objects=True` — the schema is confirmed; add the field
        to `VersionSchema` now, set it per version once 0.3b.3 resolves
        v2019/v2021 support).
  - [ ] 1.1.4 Add `"2023"` to `IMPLEMENTED_GAME_VERSIONS`.
  - [ ] 1.1.5 Update the module docstring to drop the "unconfirmed" note for
        2023.
  - **Done when**: `schema_for("2023")` returns a fully-populated entry
    (no placeholder/None fields); `python3 -c "from course_output.game_versions import schema_for; print(schema_for('2023'))"`
    runs clean in the repo venv.
  - **Decisions to record**: any field whose meaning required a guess (e.g.
    whether `placedObjects3` belongs in base) → append to `V2023_SCHEMA.md`
    under "Version registry" with the decision + evidence.
- [ ] **1.2 Thread v2023 through version-gated code**
  - **Context**: after 1.1, `schema_for("2023")` works; now make sure no code
    path hard-rejects or mis-handles v2023.
  - **Where**: `PGA2k_gen.py` (the `"isn't implemented yet"` guard at
    `PGA2k_gen.py:2009` (a second copy at :5370), `step_write_objects`,
    `step_refine_terrain` at `PGA2k_gen.py:3816`) and
    `PGA2k_gen_gui.py`. Start with:
    `grep -n 'game_version\|is_v2021\|is_v2019' PGA2k_gen.py PGA2k_gen_gui.py`.
  - **Steps**:
  - [ ] 1.2.1 Audit every `game_version == "2019"` / `!= "2019"` / `is_v2021`
        branch (notably `PGA2k_gen_gui.py:2639` — the v2021+ picker gating
        uses `!= "2019"` so v2023 falls into the v2021 side automatically;
        `step_write_objects`; `step_refine_terrain` at `PGA2k_gen.py:3816`)
        and decide v2023's behavior in each. Prefer `schema_for(v).has_*`
        flags over string comparisons where a capability drives the branch.
  - [ ] 1.2.2 Add a v2023 branch to the `game_version` resolution/error path
        (`PGA2k_gen.py:2009` and the second guard at :5370 — "isn't implemented
        yet") so v2023 is no longer rejected.
  - **Done when**: `python PGA2k_gen.py --game-version 2023 ...` passes version
    resolution (fails later only if a template isn't there yet — that's 1.3),
    and no `grep` hit for a version branch leaves v2023 unhandled.
  - **Decisions to record**: a one-line table (branch location → v2023
    behavior) appended to `V2023_TASKS.md` under this task, so Phase 2/3
    sessions don't re-audit.
- [ ] **1.3 Add a v2023 theme/template baseline**
  - **Context**: version resolution needs a resolvable template file to go
    end-to-end.
  - **Where**: `templates/` (existing `2019_*.course`, `2021_*.course`, and the
    reference `2023_fences.course`), `resolve_course_template` (grep in
    `PGA2k_gen.py`), the GUI version selector + tooltip (reads
    `IMPLEMENTED_GAME_VERSIONS` / `THEMES_V2019`), and the CLI
    `--game-version` argparse choices.
  - **Steps**:
  - [ ] 1.3.1 Confirm which themes exist in v2023 (do v2019/v2021 theme IDs
        carry over? new set?).
  - [ ] 1.3.2 Add `templates/2023_{theme}.course` (from the confirmed v2023
        template file) so `resolve_course_template` can resolve v2023.
  - [ ] 1.3.3 Expose v2023 + its theme list in the GUI version selector and
        tooltip (currently reads `IMPLEMENTED_GAME_VERSIONS`/`THEMES_V2019`).
  - [ ] 1.3.4 Expose v2023 in the CLI `--game-version` choices/help.
  - **Done when**: `resolve_course_template("2023", theme)` returns a real file
    for every advertised theme; GUI + CLI both list v2023.
- [ ] **1.4 Extend the asset catalog for new v2023 assets**
  - **Context**: v2023 adds fence/wall + prop assets (full list in
    `V2023_SCHEMA.md` "Fence/wall assets" + "Props seen in the sample"); the
    builder needs them in the catalog before Phase 2/3 can select them.
  - **Where**: `course_output/asset_catalog.json` + `asset_catalog.py` (read
    both first — learn the existing entry shape and the v2019/v2021→v2023
    mapping pattern).
  - **Steps**:
  - [ ] 1.4.1 Add the new v2023 assets (from `V2023_SCHEMA.md`) to
        `course_output/asset_catalog.json` / `asset_catalog.py` — fence/wall
        materials (the 8 in the sample — see `V2023_SCHEMA.md` "Fence/wall
        assets": StoneWall, WoodFences, CanvasFence Red/Black, Hedge,
        Asia_BrickWalls, UniFence, RetainWall; the sample is fence-focused,
        check the game for the full set), plus carts/hole-signs from
        `Assets/CourseProps/**` if the catalog covers props.
  - [ ] 1.4.2 Record v2019/v2021→v2023 asset-path mappings for shared props
        (the existing catalog pattern, e.g. the fence-post stake mapping) so
        a prop re-written across versions stays the identical asset.
  - [ ] 1.4.3 **Fence option matrix** (data): add a per-fence-asset field for
        the allowed `spacingRule`/`hasCurves`/`heightRule` set — seed from the
        author's statement (stone wall = all 4 caps) and the sample rows, mark
        unknown cells explicitly; this feeds 3.2.3 validation and 0.3a.1.
  - **Done when**: `asset_catalog` loads clean; every v2023 fence/wall asset
    path from `V2023_SCHEMA.md` is present and resolvable; the option-matrix
    field exists with stone wall fully populated.

**Exit:** `--game-version 2023` is accepted end-to-end and resolves a template,
even if fence generation is not yet wired (existing features re-target to v2023).

---

## Phase 2 — Re-target existing features to v2023 (before new feature)

Goal: prove the whole *existing* pipeline (trees, clusters, splines, holes,
water, parking, range nets) emits correct v2023 output, using the per-version
builder convention. This de-risks Phase 3 by isolating the new-feature work.

- [ ] **2.1 Per-version builders for placed objects** (`course_output/objects.py`)
  - **Context**: Phase 1 done — v2023 resolves end-to-end. Now make the
    existing object-emitting builders produce v2023 records.
  - **Where**: `course_output/objects.py` — read the `build_tree_objects_v2019`
    / `build_tree_objects_v2021` pair first (that's the pattern to copy),
    plus the dispatch in `step_pack_objects`/`step_write_objects` in
    `PGA2k_gen.py`.
  - **Steps**:
  - [ ] 2.1.1 Follow the `build_tree_objects_v2019` / `_v2021` pattern: add
        `build_tree_objects_v2023` (and any `_v2023` counterpart for stakes,
        clusters, spline-fill records, stream drops).
  - [ ] 2.1.2 v2023 uses the same asset-path scheme as v2021
        (`theme_scheme="path"`) — reuse/extend the v2021 asset-path resolution
        rather than guessing a numeric triple; confirm no new required fields
        in the diff (Phase 0.2.2).
  - [ ] 2.1.3 Wire the v2023 builders into `step_pack_objects` /
        `step_write_objects` dispatch.
  - [ ] 2.1.4 Make sure v2023-only assets (from 1.4) are selectable/usable in
        the builders' asset pools.
  - **Done when**: running the pipeline for v2023 emits
    `CourseDescription_nodes/placedObjects3.json` with valid tree/stake/cluster
    records (spot-check 2–3 entries against the v2023 sample's entry shapes —
    same keys, same path-scheme).
- [ ] **2.2 Re-target the other node writers**
  - **Context**: everything that writes a node file must honor the v2023
    renames + shapes from `V2023_SCHEMA.md` "Node renames".
  - **Where**: `course_output/` — `holes.py`, `userLayers.py`, `splines.py`,
    `water.py`, `parking.py`, `range_nets.py`, `out_of_bounds.py`; grep each
    for `2021`/`2019` branches and hardcoded filenames.
  - **Steps**:
  - [ ] 2.2.1 `holes.py`, `userLayers.py`, `splines.py`, `water.py`,
        `parking.py`, `range_nets.py`, `out_of_bounds.py` — add/verify v2023
        branches per the confirmed schema (`holes2`, `surfaceSplines2`,
        `userLayers2`; water keeps `surfaceCategory: 9`).
  - [ ] 2.2.2 Confirm the v2023 `objects_filename` is what gets written and
        that a stale v2019/v2021 node file in `course/` doesn't leak into the
        repack (see the existing "leftover different-game_version file" note).
  - **Done when**: a v2023 run produces every node file the v2023 sample has
    (`V2023_SCHEMA.md` "Confirmed node set") with v2023 names, and a
    pre-existing 2021 node file in `course/` is either cleaned or not
    repacked.
- [ ] **2.3 End-to-end verification (no fence feature yet)**
  - **Context**: gate for Phase 3 — proves the whole pre-existing pipeline
    emits correct v2023 output.
  - **Where**: full CLI pipeline (`PGA2k_gen.py`) on a test course with trees,
    water, holes, parking, range nets; the v2023 game for the load check.
  - **Steps**:
  - [ ] 2.3.1 Run the full CLI pipeline targeting `--game-version 2023` on a
        test course; confirm each node file is produced.
  - [ ] 2.3.2 Load the generated `.course` in the v2023 game and confirm trees,
        water, holes, parking, range nets render correctly (spot-check a few).
  - [ ] 2.3.3 Diff generated v2023 node files against the confirmed reference
        to catch structural drift.
  - **Done when**: in-game load succeeds with all pre-existing features
    rendering; structural diff vs `2023_fences.course` node files shows no
    unexpected key/shape differences (record any legitimate ones in
    `V2023_SCHEMA.md`).

**Exit:** a v2023 `.course` with all *pre-existing* features is playable in-game.

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
  spacingRule: int,       # 0-3 = end-cap style (none/spaced/points/ends, order TBD)
  flexibilityRule: int,   # TBD (0/1 observed)
  heightRule: int,        # 0 = contoured (follow terrain), 1 = stepped
  spacing: float          # post/panel spacing (m)
}
```

Waypoints are 2D `{x, y}` with `y` = course Z (matches the items frame). The
parking-style object tiling (placed `items[]`) remains a valid *alternative*
per the course author, but objectPaths is the native game-written mechanism —
prioritize it; treat the object-tile version as a later optional variant.

- [ ] **3.1 OSM input path (currently all `None`)**
  - **Context**: Phase 2 done. OSM fence/wall ways must become features the
    fence builder can consume. Today `classify_way` returns `None` for every
    fence/wall/hedge tag (verified this session by probing
    `ingest.osm.classify_way` inline).
  - **Where**: `ingest/osm.py` — read `classify_way` + the existing `kind`
    assignments (water/hazard etc.) to copy the pattern; grep every `kind`
    consumer before adding a new one.
  - **Steps**:
  - [ ] 3.1.1 Add `classify_way` branches for fence/wall tags
        (`barrier=fence`, `barrier=wall`, `barrier=hedge`, `barrier=chain`,
        `natural=hedge` — probe first; today they all return `None`).
  - [ ] 3.1.2 Assign a `kind` string to each (e.g. `fence`, `wall`, `hedge`)
        and confirm it routes to exactly the v2023 fence consumer and no
        unintended existing consumer (grep every kind consumer before adding).
  - [ ] 3.1.3 Decide area vs line semantics per tag (fence/wall are lines; a
        hedge could be a closed area — confirm against OSM conventions and the
        target rendering).
  - [ ] 3.1.4 Tag→asset routing: map OSM tags to a v2023 fence asset
        (using the asset catalog from 1.4) — e.g. `material=stone` →
        `StoneWall*`, `material=wood` → `WoodFences*`, `material=chain_link` →
        `UniFence*`, `natural=hedge` → `HedgeSpline*`, `wall=brick` →
        `Asia_BrickWalls*`; record the mapping table in `V2023_SCHEMA.md`.
  - **Done when**: a small probe script (create it under `util/`, e.g.
    `util/probe_fence_tags.py` — none exists yet) on an OSM extract containing
    fence/wall ways shows the ways classified to the new kinds and routed to
    the expected asset paths.
- [ ] **3.2 Feature → objectPath record builder**
  - **Context**: 3.1 done — fence ways are classified features. Now convert a
    feature line into the exact `objectPath` record shape (in
    `V2023_SCHEMA.md` "objectPaths rule fields").
  - **Where**: new `course_output/fences.py` (or `splines.py` addition —
    decide by reading which file owns `Value.splines[]` emission and whether
    fence output belongs with it); the resample/simplify helpers already used
    for OOB (`course_output/out_of_bounds.py`) and water.
  - **Steps**:
  - [ ] 3.2.1 Feature line → objectPath record. Waypoints: resample/simplify
        the OSM way (Douglas-Peucker, as elsewhere) into 2D `{x, y}`
        waypoints in the course frame; set `pointOne`/`pointTwo` handles per
        the curve style (0.3a.6 — confirm handle semantics before emitting
        curves).
  - [ ] 3.2.2 Map fence style → rule fields: `spacingRule` (cap style),
        `heightRule` (contoured vs stepped), `hasCurves`, `spacing`, `width`,
        `height` offset. Defaults per asset from the sample rows
        (`V2023_SCHEMA.md` matrix).
  - [ ] 3.2.3 **Per-asset option validation**: enforce the option matrix
        (1.4.3) — warn/skip when an OSM-tag-requested option isn't valid for
        the chosen asset. Keep the matrix data-driven (asset_catalog.json) so
        it grows as 0.3a completes.
  - [ ] 3.2.4 Support the "trick" recipes as named presets (sample-proven,
        `V2023_SCHEMA.md` "Tricks"): curb (buried stone, h≈-1.5), railroad
        (buried black canvas, h≈-2.69), retaining wall (`RetainWall*`,
        w=2.5, h≈-0.7, contoured, 4 wp, state=1).
  - [ ] 3.2.5 Handle closed vs open fence runs and shared corners between
        adjacent ways (no double posts / no gaps).
  - **Done when**: unit test feeds a synthetic way (straight + curved, open +
        closed) and produces objectPath records byte-comparable in structure
        to the sample rows (same keys, 2D waypoints, handles populated).
- [ ] **3.3 Pipeline wiring**
  - **Context**: 3.2 done — records build in isolation; now wire them into
    the pipeline like range nets.
  - **Where**: `PGA2k_gen.py` — read `step_generate_range_nets` +
    `step_write_objects`/`step_pack_objects` + `step_repack` as the pattern;
    `PGA2k_gen_gui.py` for the step UI.
  - **Steps**:
  - [ ] 3.3.1 `step_generate_fences` in `PGA2k_gen.py` following the
        `step_generate_range_nets` pattern (kind → features.geojson → tiler).
  - [ ] 3.3.2 Emit objectPath entries into `placedObjects3.json` — merge with
        the existing `step_write_objects`/`step_pack_objects` output for the
        same Key.path (fence assets get their own Key entries alongside
        items/clusters/splines) and include in `step_repack`.
  - [ ] 3.3.3 GUI button/menu entry + CLI step, with per-way style options
        (cap style, contoured/stepped, curved/straight, spacing, burial depth).
  - [ ] 3.3.4 Persist derived state (origins, frame, chosen assets) in
        `project.json` per the state-bus convention.
  - **Done when**: a v2023 run with fence ways in the OSM extract produces a
    repacked `.course` whose `placedObjects3.json` contains the fence
    objectPaths alongside normal items.
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
      square, `scale` = radius m, `value` 1.0, `tool` 0, y "-Infinity"). The
      only remaining gate is 0.3b.1 (in-game suppression semantics), which
      does not block writing the formatter — build against the confirmed
      shape, verify semantics in 3.4.8.
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
  - [ ] 3.4.5 **Scope decision**: which generated objects are suppressed where
        the fill is painted (trees only? all auto-generated placements?) and
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
  - **Where**: the tag probe script from 3.1 (e.g. `util/probe_fence_tags.py`),
    a real OSM extract with fence/wall ways, the v2023 game.
  - **Steps**:
  - [ ] 3.5.1 Extend the 3.1 probe to cover a representative set of
        fence/wall/hedge tags end-to-end (tag → kind → objectPath record).
  - [ ] 3.5.2 Generate a course with real OSM fence/wall ways; diff emitted
        objectPaths against `2023_fences.course`'s rows (same rule-field
        shapes, sane waypoints).
  - [ ] 3.5.3 Load in the v2023 game: fences/walls render on their spline,
        grounded/contoured correctly, cap styles match `spacingRule`, stepped
        rows sit at the offset height; no collision with surface splines.
  - [ ] 3.5.4 Verify the trick presets (curb / railroad / retaining wall)
        in-game.
  - [ ] 3.5.5 Regression: confirm untouched features (trees, water, splines)
        are unchanged when fences/clear-objects are present vs absent.

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
