# PGA2K v2023 implementation — task list

Working task breakdown for adding **v2023** support to PGA2K-LiDAR-TerrainGen.
Headline feature: **spline fences and walls**. Secondary v2023 capability
flagged in the registry: texture painting (`has_texture_paint`).

Every task is written to split cleanly into subtasks (indented bullets).
Tasks are ordered by dependency — do Phase 0 before anything else, because the
real v2023 schema is still **unconfirmed** (see `course_output/game_versions.py`).

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
  expose different option sets (stone wall exercises all 4 spacingRules; canvas
  / hedge rows don't). Must confirm which options are valid per asset (in-game).
- v2023 confirms **tilted items are legal** (GolfCartPrefab rot.x/z=±10°) and
  lists the **5 hole-sign colors** (HoleSign01–05; north→south = green/red/blue/
  white/black — the 0N↔color mapping needs an in-game check).
- New v2023 fence assets are under `Assets/CourseGen/Detail/Walls/**`
  (`*Post*Prefab`, `*SplinePostPrefab`), incl. Asia/Brick; carts + hole signs
  under `Assets/CourseProps/**`.
- Node renames v2021→v2023: `userLayers`→`userLayers2`, `weather2`→`weather3`,
  base keys `holes`→`holes2`, `surfaceSplines`→`surfaceSplines2`; `version` 25→31;
  new base keys `blurHeight`/`useV28FairwaySeed`/`useV30Rough`.
- **STILL UNCONFIRMED:** the "clear generated objects" layer (sample course has
  no painted example — need a course with that feature); `flexibilityRule`/`state`
  meaning; per-asset option matrix; `spacingRule`→cap-style ordering.

> Branch: `hermes-experiment` (never `main`).
> Read `AGENTS.md` + this file's Phase 0 notes before starting.
> venv for any probe: `cd <repo> && uv venv .venv && uv pip install -r requirements.txt`.

---

## Phase 0 — Schema confirmation (MOSTLY DONE — see `V2023_SCHEMA.md`)

The repo's standing rule (stated in `game_versions.py` docstring): a version
is only "confirmed" after diffing an extracted `.course` the way `hhills3_2019`
vs `hhills3_2021` were. **Done on 2026-09-23** against
`templates/2023_fences.course` (commit `39a54da`), findings in `V2023_SCHEMA.md`.
Remaining items below are the confirmed gaps.

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
- [ ] **0.3b Characterize the "clear generated objects" paint (new feature)** —
      UNRESOLVED: the sample course contains no painted example.
  - [ ] 0.3b.1 Find the layer/entries the game uses to mark spline fill areas
        where auto-generated objects must be suppressed (userLayers2.json new
        key? a paint layer like `outOfBounds`? v2021's userLayers had a
        `clearTrees` key — check if that survives in v2023).
  - [ ] 0.3b.2 Capture the stamp/entry shape: brush types, value semantics,
        whether it's a spline region or brush stamps, and which generated
        object types it clears (trees? all? per-layer?).
  - [ ] 0.3b.3 Confirm the generator's suppression contract: does the engine
        skip *generated* objects only (trees, auto-placed), leaving manual
        placed objects alone?
  - [ ] 0.3b.4 **Action: Andy to paint a clear-objects region in-game and
        re-export a sample course** (same trick as 2023_fences.course) once the
        v2023 build is in hand.
- [x] **0.4 Document findings** — `V2023_SCHEMA.md` written; `VERSION_SCHEMAS`
      `2023` entry still to be populated (that is task 1.1, gated only on
      0.3b for the clear-objects capability flag).

---

## Phase 1 — Version registry + plumbing (low risk, unblocks everything)

- [ ] **1.1 Add the `2023` `VersionSchema` entry** (`course_output/game_versions.py`)
  - [ ] 1.1.1 Set confirmed fields: `objects_filename="placedObjects3.json"`,
        `theme_scheme="path"`, **`holes_filename="holes2"`**,
        **`splines_filename="surfaceSplines2"`**,
        **`userlayers_filename="userLayers2"`** (v2023 renames the v2021
        nodes — see `V2023_SCHEMA.md`). Audit every reader/writer of these
        filenames (holes.py, splines.py, userLayers.py, course_repack/extract,
        `_stale_version_node_files`) so the renamed nodes are produced,
        consumed, and stale-file-checked under the new names.
  - [ ] 1.1.2 **Verify `placedObjects3` placement in v2023**: the v2021
        template carries `placedObjects3` as a *base* key (empty, inlined in
        CourseDescription), while v2023's base does NOT list it — the sample
        has it only as a node file. Confirm in-game that a repacked course
        with `CourseDescription_nodes/placedObjects3.json` (and the key absent
        from base) is loaded, and adjust `_ensure_course_baseline` /
        `course_repack.py` accordingly if the game wants it in the base.
  - [ ] 1.1.3 Set capability flags: `has_fences=True` (objectPaths), plus
        `has_texture_paint` and a clear-objects flag (add to `VersionSchema`
        once 0.3b confirms the layer).
  - [ ] 1.1.4 Add `"2023"` to `IMPLEMENTED_GAME_VERSIONS`.
  - [ ] 1.1.5 Update the module docstring to drop the "unconfirmed" note for
        2023.
- [ ] **1.2 Thread v2023 through version-gated code**
  - [ ] 1.2.1 Audit every `game_version == "2019"` / `!= "2019"` / `is_v2021`
        branch (notably `PGA2k_gen_gui.py:2637`, `step_write_objects`,
        `step_refine_terrain` at `PGA2k_gen.py:4192`) and decide v2023's
        behavior in each. Prefer `schema_for(v).has_*` flags over string
        comparisons where a capability drives the branch.
  - [ ] 1.2.2 Add a v2023 branch to the `game_version` resolution/error path
        (`PGA2k_gen.py:1957` "isn't implemented yet" guard) so v2023 is no
        longer rejected.
- [ ] **1.3 Add a v2023 theme/template baseline**
  - [ ] 1.3.1 Confirm which themes exist in v2023 (do v2019/v2021 theme IDs
        carry over? new set?).
  - [ ] 1.3.2 Add `templates/2023_{theme}.course` (from the confirmed v2023
        template file) so `resolve_course_template` can resolve v2023.
  - [ ] 1.3.3 Expose v2023 + its theme list in the GUI version selector and
        tooltip (currently reads `IMPLEMENTED_GAME_VERSIONS`/`THEMES_V2019`).
  - [ ] 1.3.4 Expose v2023 in the CLI `--game-version` choices/help.
- [ ] **1.4 Extend the asset catalog for new v2023 assets**
  - [ ] 1.4.1 Add the new v2023 assets (from 0.3.5) to
        `course_output/asset_catalog.json` / `asset_catalog.py` — fence/wall
        materials (wood, chain, brick, hedge...), and any others the diff
        surfaced.
  - [ ] 1.4.2 Record v2019/v2021→v2023 asset-path mappings for shared props
        (the existing catalog pattern, e.g. the fence-post stake mapping) so
        a prop re-written across versions stays the identical asset.

**Exit:** `--game-version 2023` is accepted end-to-end and resolves a template,
even if fence generation is not yet wired (existing features re-target to v2023).

---

## Phase 2 — Re-target existing features to v2023 (before new feature)

Goal: prove the whole *existing* pipeline (trees, clusters, splines, holes,
water, parking, range nets) emits correct v2023 output, using the per-version
builder convention. This de-risks Phase 3 by isolating the new-feature work.

- [ ] **2.1 Per-version builders for placed objects** (`course_output/objects.py`)
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
- [ ] **2.2 Re-target the other node writers**
  - [ ] 2.2.1 `holes.py`, `userLayers.py`, `splines.py`, `water.py`,
        `parking.py`, `range_nets.py`, `out_of_bounds.py` — add/verify v2023
        branches per the confirmed schema.
  - [ ] 2.2.2 Confirm the v2023 `objects_filename` is what gets written and
        that a stale v2019/v2021 node file in `course/` doesn't leak into the
        repack (see the existing "leftover different-game_version file" note).
- [ ] **2.3 End-to-end verification (no fence feature yet)**
  - [ ] 2.3.1 Run the full CLI pipeline targeting `--game-version 2023` on a
        test course; confirm each node file is produced.
  - [ ] 2.3.2 Load the generated `.course` in the v2023 game and confirm trees,
        water, holes, parking, range nets render correctly (spot-check a few).
  - [ ] 2.3.3 Diff generated v2023 node files against the confirmed reference
        to catch structural drift.

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
        (`material`/`wall`/fence type → `StoneWall*`, `WoodFences*`,
        `CanvasFence*`, `HedgeSpline*`, `Asia_BrickWalls*`, `UniFence*`,
        `RetainWall*`), using the asset catalog (1.4).
- [ ] **3.2 Feature → objectPath record builder**
  - [ ] 3.2.1 New `course_output/fences.py` (or `splines.py` addition):
        feature line → objectPath record. Waypoints: resample/simplify the OSM
        way (Douglas-Peucker, as elsewhere) into 2D `{x, y}` waypoints in the
        course frame; set `pointOne`/`pointTwo` handles per the curve style
        (0.3a.6 — confirm handle semantics before emitting curves).
  - [ ] 3.2.2 Map fence style → rule fields: `spacingRule` (cap style),
        `heightRule` (contoured vs stepped), `hasCurves`, `spacing`, `width`,
        `height` offset. Defaults per asset from the sample rows.
  - [ ] 3.2.3 **Per-asset option validation**: enforce the 0.3a.1 option
        matrix (warn/skip when an OSM-tag-requested option isn't valid for the
        chosen asset). Keep the matrix data-driven (asset_catalog.json) so it
        grows as 0.3a completes.
  - [ ] 3.2.4 Support the "trick" recipes as named presets (sample-proven):
        curb (buried stone, h≈-1.5), railroad (buried black canvas, h≈-2.69),
        retaining wall (`RetainWall*`, w=2.5, h≈-0.7, contoured).
  - [ ] 3.2.5 Handle closed vs open fence runs and shared corners between
        adjacent ways (no double posts / no gaps).
- [ ] **3.3 Pipeline wiring**
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
- [ ] **3.3b Optional variant: object-tile fences** (deferred; only if wanted)
  - [ ] 3.3b.1 Tiler following the `parking.py`/`range_nets.py`
        object-tile-along-line pattern, placing fence post/panel `items[]`
        (reuse existing tiling/rotation — no hand-rolled sampling).
  - [ ] 3.3b.2 Per-way choice between objectPaths vs object-tile representation.
- [ ] **3.4 Fill splines with "clear generated objects" stamps (new feature)**
  - [ ] 3.4.1 **Schema (gated on 0.3b)**: capture the layer + entry shape from
        a painted sample course before writing the formatter.
  - [ ] 3.4.2 **Scope decision**: which generated objects are suppressed where
        the fill is painted (trees only? all auto-generated placements?) and
        does it apply in v2023 only or also v2019/v2021 (check whether
        v2021's userLayers `clearTrees` key is the same mechanism).
  - [ ] 3.4.3 **Input path**: how the fill area is designated — OSM tag on a
        closed way (e.g. `pga_clear_objects=yes`)? a spline drawn in the GUI?
        (mirror however OOB gets its shape).
  - [ ] 3.4.4 **Record builder**: follow the `out_of_bounds.py` blueprint —
        version-agnostic frozen records in their own file
        (e.g. `clear_objects.json`), entries formatted for the target layer by
        a `_v2023` (or shared) formatter.
  - [ ] 3.4.5 **Generator-side suppression** (the core logic): when
        `step_generate_trees` / other object-generating steps place objects,
        test each candidate point against the clear-objects region (point-in-
        region test against the filled splines) and skip it. Centralize the
        containment test so every generator uses the same rule.
  - [ ] 3.4.6 **Wiring**: `step_generate_clear_objects` (+ clear flag,
        mirroring `step_generate_oob`/`_clear_oob`), fold into
        `step_write_terrain` like OOB is (`oob_entries` pattern), `project.json`
        enable flag.
  - [ ] 3.4.7 **GUI**: brush/draw area + apply/clear controls, mirroring the
        OOB UI.
  - [ ] 3.4.8 **Verification**: paint a clear region over a tree-dense area,
        regenerate, confirm generated trees/objects are absent inside the
        region and present just outside it; confirm manually placed objects are
        unaffected.
- [ ] **3.5 Fence + clear-objects verification**
  - [ ] 3.5.1 Probe tag→kind→objectPath-record for a representative set of
        fence/wall/hedge tags (extend `scripts/probe_features.py`).
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

Only if Phase 0.5 confirms a texture-painting schema. Independent of fences.

- [ ] **4.1** Capture the paint schema from the reference `.course`.
- [ ] **4.2** Set `has_texture_paint=True` once implemented.
- [ ] **4.3** Build the paint generator + wire a step; verify in-game.

---

## Phase 5 — Docs + cleanup

- [ ] **5.1** Update `README.md` line 45 ("TODO: v2023, v2025") to reflect
        v2023 support (fences/walls + texture paint status).
- [ ] **5.2** Update `game_versions.py` and `objects.py` module docstrings to
        drop "not implemented yet" for v2023.
- [ ] **5.3** Update the `pga2k-terragen` skill with the confirmed v2023 fence/wall
        schema + pipeline shape (so future sessions don't re-derive it).
- [ ] **5.4** Record the finished feature in `~/completed-tasks.md`.

---

### Cross-cutting rules (apply to every task)
- Commit on `hermes-experiment`, never `main`.
- Per-version logic = explicit `build_X_v2023`, never a generic version-parameterized
  function (repo convention, see `game_versions.py` docstring).
- Grep every consumer of a feature `kind` before adding/renaming it.
- Verify with the repo venv, not system python (needs `overpy`/`shapely`/`pyproj`).
- Confirm the real schema before building — the whole point of Phase 0.
