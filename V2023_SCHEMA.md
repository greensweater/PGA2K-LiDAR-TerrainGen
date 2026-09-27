# V2023 schema reference — confirmed from `templates/2023_fences.course`

Source: `templates/2023_fences.course` (pushed 2026-09-23, commit `39a54da`),
extracted with `util/course_extract.py` into `~/scratch/v2023/2023/`.
Compared against `templates/2021_rustic.course` (extracted to `~/scratch/v2023/2021/`).

This file records what is CONFIRMED. Anything unconfirmed is marked as such.
It backs `V2023_TASKS.md` Phase 0 (0.2–0.4 are now largely done).

## Container format

Identical to v2021: gzip-compressed UTF-16LE outer JSON (`meta.json` shape)
whose `binaryData.CourseDescription` is base64(gzip(utf-16le JSON)). Same for
`CourseMetadata`/`Thumbnail`. `util/course_repack.py` / `course_extract.py`
work unchanged for v2023.

## Version + base CourseDescription diff (2021 v25 → 2023 v31)

- `"version"`: 25 → 31.
- Node files renamed: `userLayers` → **`userLayers2.json`**,
  `weather2` → **`weather3.json`**.
- Base keys renamed: `holes` → **`holes2`**, `surfaceSplines` →
  **`surfaceSplines2`** (base keys, not node files — the sample has both
  empty, so no node file was emitted).
- New base keys: `blurHeight`, `useV28FairwaySeed`, `useV30Rough`.
- **`placedObjects3`: base key present in v2021 (empty, inlined), ABSENT from
  the v2023 base** — in v2023 it appears only as a node file. Whether the
  game wants it in the v2023 base too is unverified (task 1.1.2).
- `CourseMetadata.json`, `meta.json`, `Thumbnail` — structurally unchanged.

## userLayers2.json (v2023)

### Node renames (v2021 → v2023)

| v2021 | v2023 |
|---|---|
| `userLayers.json` | `userLayers2.json` |
| `holes.json` | `holes2.json` (base key `holes` → `holes2`) |
| `surfaceSplines.json` | `surfaceSplines2.json` (base key renamed too) |
| `weather2.json` | `weather3.json` |

Confirmed node set (v2023, from the sample): `fairwayEdge, fairwayEdge2,
fairwayNoise, grassNoise, greenEdge, greenNoise, hazardEdge, hazardVariance,
perturbationNoise, placedObjects3, roughEdge, roughNoise, surfaces,
teeColours, terrainNoise, treeOptions, unplayableNoise, userLayers2,
weather3`. (`holes2`/`surfaceSplines2` exist as base keys but were empty in
the sample, so no node file was emitted for them.)

Sample keys (blank-ish course): `treeDensity`, `terrainHeight`, `height`,
`surfaces`, `outOfBounds`, `crowdLocations`, `water`. The v2021
`userLayers.json` additionally had `deletedHazards`, `newHazards`, `objects`,
`clearTrees`, `addTrees`, `hazards`, `trees`, `green` — likely omitted when
empty in v2023 (or relocated). The "clear generated objects" paint is
**confirmed** to live in this same `surfaces` array with
`surfaceCategory: 5` — see "Clear-generated-objects paint" below.

The `height` layer entry is **trimmed vs v2021**: the game writes only
`tool/position/rotation/scale/type/value/holeId` — v2021's `_orientation`,
`radius` and `orientation` are gone (v2019 had `radius` only). The
`surfaces` clear-objects stamps have the same trimmed shape. `y: "-Infinity"`
= terrain-grounded, same as v2021. (Corrected 2026-09-23; this section
earlier said "unchanged".) The writers follow this through
`has_orientation_fields=False` / `has_radius_field=False` in the 2023
`VersionSchema`. `water[]` entry shape is still unconfirmed: the sample's
`water` is empty, so water entries are written in v2021 shape.

## placedObjects3.json entries (v2023)

Same v2021 shape: `[{Key: {path}, Value: {items, clusters, splines,
objectPaths, IsEmpty}}]`.

- `items[]` — placed objects: position/rotation/scale; `y: "-Infinity"`
  grounds to terrain. v2023 confirms **tilted items are legal**:
  `GolfCartPrefab` items carry rotation x/z = ±10° (tilt). Per the course
  author, of the 9 golf-cart entries only this one is tilt-enabled.
- **`objectPaths[]` — NEW in v2023. The fence/wall mechanism.** Shape:

  ```json
  {
    "path": {
      "waypoints": [
        {"pointOne": {"x": ..., "y": ...},
         "pointTwo": {"x": ..., "y": ...},
         "waypoint": {"x": ..., "y": ...}},
        ...
      ],
      "width": 4.0,
      "hasCurves": true,
      "state": 0
    },
    "height": 0.0,
    "spacingRule": 2,
    "flexibilityRule": 1,
    "heightRule": 0,
    "spacing": 4.0
  }
  ```

  - `waypoints` are 2D `{x, y}` where `y` is the course's Z axis (a fence at
    waypoint y≈127 sits at items-frame z≈127). `pointOne`/`pointTwo` are the
    bezier tangent handles of the waypoint; when `hasCurves: false` they
    appear to coincide with straight-segment endpoints (verify).
  - Fence assets live under `Assets/CourseGen/Detail/Walls/**`
    (`*Post*Prefab` / `*SplinePostPrefab`); carts/signs under
    `Assets/CourseProps/**`.

### Fence/wall assets (v2023, from the sample)

`Assets/CourseGen/Detail/Walls/` (8 in the sample): `StoneWallAPostAPrefab`,
`WoodFencesAPostAPrefab`, `CanvasFence01AARedPostAPrefab`,
`CanvasFence01AABlackPostAPrefab`, `HedgeSplinePostPrefab`,
`Asia_Walls/Asia_BrickWalls_PostPrefab`, `UniFencePostAPrefab`,
`RetainWallAPostAPrefab`. (The sample is fence-focused, not exhaustive —
other fence assets may exist in the game; 1.4.1 should check.)

### Props seen in the sample (v2023)

`Assets/CourseProps/Equipment/GolfCartPrefab` (one tilted entry — tilt legal
in v2023) and `Assets/CourseProps/Signs/HoleSign0[1-5]Prefab` (5 hole signs).

## objectPaths rule fields → observed options

Confirmed from the 12 fence objectPaths in the sample (all `spacing` 4.0
except one 1.3; `width` 4.0 except retaining wall 2.5):

| field | observed values | mapped meaning (per course author's description) |
|---|---|---|
| `spacingRule` | 0, 1, 2, 3 | **end caps: 0 none / 1 ends only / 2 spaced / 3 spline points** (JSON values, in-game verified 2026-09-26; the game menu lists them in a different order) |
| `heightRule` | 0, 1 | **0 = contoured** (follows the stamped terrain; `height` = offset), **1 = leveled** (the editor's name; "stepped" in older notes): ONE absolute `height` for the whole run, which the editor sets to the run's **minimum ground** (Andy, 2026-09-27); sample rows use 3.23 = the blank's flat datum. See "Leveled fence height" |
| `hasCurves` | true, false | **curved vs straight** segments |
| `flexibilityRule` | 0, 1 | unconfirmed — candidate: rigid panels vs flexible/flexing along path (fl=0 seen on the straight brick rows) |
| `state` | 0 (11 rows), 1 (retaining wall, 4 waypoints) | unconfirmed |
| `height` | 0.0, 3.23, -1.498, -0.715, -2.688 | vertical offset from terrain; negative = buried |
| `spacing` | 4.0, 1.3 | post/panel spacing along the path (m) |
| `width` | 4.0, 2.5 | fence width (m) |

**Asset × option matrix (partial — author: "may have to determine which
options are available per asset"):**

| asset (CourseGen/Detail/Walls/…) | spacingRule seen | hasCurves seen | heightRule seen | flexibilityRule seen |
|---|---|---|---|---|
| `StoneWallAPostAPrefab` | 0 | true | 0 | 1 |
| `WoodFencesAPostAPrefab` | 2 | true | 0 | 1 |
| `UniFencePostAPrefab` | 2 | true | 0 | 1 |
| `CanvasFence01AARedPostAPrefab` | 2 | true | 0, 1 | 1 |
| `CanvasFence01AABlackPostAPrefab` | 2 | true | 0 | 1 |
| `HedgeSplinePostPrefab` | 1 | true | 0, 1 | 1 |
| `Asia_BrickWalls_PostPrefab` | 0, 1, 2, 3 | false, true | 0, 1 | 0, 1 |
| `RetainWallAPostAPrefab` | 0 | true (4 wp, state=1) | 0 | 1 |

Per-asset option matrix must still be confirmed in-game (0.3a.1) — the
author confirmed only that the **stone wall** has the full 4-cap option set
(omitted from the course for brevity); canvas/hedge rows in the sample don't
cover every value either.

### In-game fence test results (FENCETEST at (-900, 900), Andy, 2026-09-26)

Built with `--step push-fence-test`; the row ids are from its legend.

- **Every asset renders from our records** (B1–B8), as do the sample's own
  13 rows (A).
- **Orientation is correct** (C): waypoint `x` is +x and `y` is course +z,
  with no mirroring.
- **objectPaths follow the userLayers height stamps** (D1 sits exactly on
  the +15 m flatten pad). D3 contours across the pad's edges; D2's
  `height=+15` floats about 15 m above. So `height` is an offset from the
  *stamped* terrain when `heightRule=0`.
- **`spacingRule`**, the JSON value (verified in-game on the 4-point zigzags
  E3/E4):
  - 0 = none
  - 1 = ends only
  - 2 = spaced (by `spacing`)
  - 3 = spline points (a cap at every waypoint)

  The game's menu lists them as none / spline points / spaced / ends only
  (Andy). That's the UI order, *not* the stored values; E3 (value 1) showed
  ends only and E4 (value 3) showed spline points. 1 and 3 look identical on
  2-point runs (B10/B12, B16/B17).
- **`heightRule=1` (stepped): `height` is an ABSOLUTE elevation**, not an
  offset. E1 at height = datum 3.23 sat on the surface; E2 at datum+3
  floated; B13 at 0 was underground. `heightRule=0`: `height` is an offset
  from the stamped terrain. (Refined 2026-09-27: see "Leveled fence
  height" below — one height per run, the run's lowest ground.)
- **`hasCurves`**: true draws a smooth curve through the waypoints (E5);
  false draws straight segments, so a 3-point bend is a V (E6).
- **The fences in the full course render** (BX3/BX4, near hole 18). My
  earlier "not visible" came from pointing at the wrong spot (the hole 18
  tee from CourseMetadata `teePositions`); the fences are near the 18th hole. WoodFences accepts
  all four, so the catalog set is now complete.
- **`heightRule=1` with `height` 0 doesn't render** (B13). Every sample
  stepped row carries `height ≈ 3.23`, the blank's datum. Hypothesis: when
  stepped, `height` is an **absolute elevation**, so 0 is about 3 m
  underground. E1 (height = datum) and E2 (datum+3) test this.
- `hasCurves=false` looks identical to curved on a straight 2-point run
  (B14), as expected. E5 vs E6 show it on a bend.
- **Retaining wall** renders 0.3 m high at the sample's `-0.715`, so
  `RETAINING_WALL_HEIGHT_M` is now **-1.015**. Verified flush in-game. The asset is special: it
  draws a tile that "refracts" the ground texture at the level of the wall's
  cap, set about 1.9 m back from the spline (an upside-down L / Γ profile).
  Use for pond edges: place the spline about 2 m inside the contour line,
  then set `height` by trig from the bank slope over a ~1.5 m run, so the
  tile overlaps and clips into the terrain. That's a future builder feature,
  not implemented.

## "Tricks" from the sample course (reproducible recipes)

- **Curb**: stone fence partially buried (`height ≈ -1.5`, contoured,
  sp=0/fl=1/ht=0).
- **Railroad track**: black canvas fence partially buried
  (`height ≈ -2.69`, sp=2/fl=1/ht=0).
- **Retaining wall hugging the heightmap**: `RetainWallAPostAPrefab`,
  `width 2.5`, `height ≈ -0.715`, contoured (ht=0), 4 waypoints, `state=1`.

## Hole signs (5, `Assets/CourseProps/Signs/HoleSign0[1-5]Prefab`)

Single `items` entry each, one per color. Author placed them north→south in
the course as **green, red, blue, white, black**. File positions
(course-local, z ascending = south→north or vice versa — direction not
known to us): `HoleSign01 z≈121.7`, `02 z≈123.8`, `03 z≈126.2`,
`04 z≈128.8`, `05 z≈131.1` (all x≈-430, rot.y≈279.3°).
**Resolved 2026-09-23 (Andy):** black is the v2021 sign (`HoleSign01Prefab`,
already used in v2021 projects). Green/red/blue/white are the four v2023
additions. 01 = black is the southernmost sign (lowest z), so **+z = north**,
and the mapping follows from the author's N→S order: **01 black, 02 white,
03 blue, 04 red, 05 green**. 02–05 are inferred from the sample's positions,
not individually viewed; recorded in `asset_catalog.json` descriptions.

## Clear-generated-objects paint (CONFIRMED, second export 2026-09-23)

The re-exported `2023_fences.course` (commit `8376f0e`) added
**clear_generated_objects stamps** — they live in the **existing
`userLayers2.json` `surfaces` layer** as `surfaceCategory: 5` entries (NOT a
new key). Two stamps were provided:

- square: `type: 15` at `(-575.4, z 165.1)`, scale 58.8456268, value 1.0
- round:  `type: 8`  at `(-751.7, z 176.5)`, scale 58.8456268, value 1.0

`type` 8 (round) / 15 (smooth square) are the SAME brush ids the OOB feature
uses (`OOB_ROUND_BRUSH=8`, `OOB_SQUARE_BRUSH=15` in
`course_output/out_of_bounds.py`) — the course author's expectation that they
correspond to known stamp shapes is borne out by the data; the in-game visual
match still needs a quick check (0.3b.2). Entry shape is identical to a
height-layer stamp (tool 0, y "-Infinity" = terrain-grounded, value 1.0) but
with `surfaceCategory` instead of the height layer's `value` semantics.

Known `surfaceCategory` values so far: **5 = clear generated objects** (this
sample), **9 = water** (`course_output/water.py`
`WATER_SURFACE_CATEGORY=9`, written with `type: 15` stamps). Category 5 was
unused by this repo's writers in both v2019/v2021 — v2021's userLayers.json
had the key set `deletedHazards, newHazards, objects, clearTrees, addTrees,
treeDensity, hazards, terrainHeight, height, trees, green, surfaces,
outOfBounds, crowdLocations, water` and the v2021 sample's `surfaces` array
was empty, so whether v2021 also honors category 5 is unconfirmed (0.3b.3).

## Version registry (task 1.1 decisions, 2026-09-23)

- **`placedObjects3` is a top-level CourseDescription key in v2023, same as
  v2021.** It only *looked* absent from the v2023 base because
  `util/course_extract.py` moves every non-empty list/dict value into a node
  file (the v2021 template's `placedObjects3` was empty, so it stayed inline).
  `course_repack.py` lets a node file override the base key, so no
  `_ensure_course_baseline` / repack change was needed. Settled from the code,
  no in-game check needed (task 1.1.2).
- `holes2` / `surfaceSplines2` are the same kind of key (empty in the
  sample). Writers now emit `holes2.json` / `surfaceSplines2.json` /
  `userLayers2.json` via `schema_for(v).*_filename`, and repack puts them
  under those keys.
- `has_pins_field=True` for 2023 is **inherited from v2021, not confirmed**:
  the sample's `holes2` is empty (0.3a.5).
- `has_clear_objects=True` for 2023 only. v2019/v2021 stay False until 0.3b.3.
- Theme: the v2023 sample carries `theme: 11` (= rustic), the same id as
  the v2021 rustic template, so theme ids appear to carry over.
  `templates/2023_rustic.course` is a blank built from `2023_fences.course`:
  the placedObjects3 node is dropped and restored as an empty base key, the
  userLayers2 `surfaces` stamps are cleared, `height` is cut down to the
  single map-wide type-72 base stamp (the same one the v2021 template
  carries), and it is renamed `rustic_2023` / `offlineSaverustic_2023`.
  The thumbnail still shows the fence scene. Other themes need a blank saved
  from the game.
- **v2023 saves courses to `%USERPROFILE%\Documents\My Games\PGA TOUR
  2K23\Courses`**, not `AppData\LocalLow\2K\...` like v2019/v2021 (its
  LocalLow folder holds only logs). The GUI's `GAME_VERSION_FOLDERS` now maps
  each version to a full home-relative path.

## Phase 2 re-target decisions (2026-09-23)

- **placedObjects3 group envelope differs; entries don't.** The game writes
  every v2023 group as `Value: {items, clusters, splines, objectPaths,
  IsEmpty}` (all 14 sample groups, `objectPaths: []` when the group has no
  fence, `IsEmpty: false`). v2021 game saves (`templates/rangenets.course`,
  v25) write `{items, clusters, splines}` only. Item/cluster/spline entry
  shapes are identical. So there are **no `build_X_v2023` copies** of the
  tree/stake/cluster/spline-fill/waterfall builders (no per-version
  resolution logic differs). Instead `objects.placed_object_groups_to_v2023`
  applies the envelope once, after `merge_object_groups`, gated on
  `schema_for(v).has_fences`. `IsEmpty` is derived as "all four lists empty";
  the sample only shows `false` on non-empty groups, so the `true` case is
  inferred.
- `merge_object_groups` now merges list fields only. Before this, merging a
  game-saved v2023 group (scalar `IsEmpty`) crashed with `extend(False)`,
  for example `_inject_collection_into_course` into a v2023 course that
  already has objects.
- `userLayers2.json` blank fallback (`_BLANK_USER_LAYERS_SCHEMA_V2023`) = the
  sample's 7 keys. Only used when the node file is missing.
- **`water[]` entries still use the v2021 shape** (`_orientation`/`radius`/
  `orientation` included). No v2023 water sample exists, and a field the
  game ignores is safer than dropping one it needs. The in-game load (2.3.2)
  decides. If the game re-saves water trimmed like `height`, gate
  `water._water_entry` on `has_orientation_fields`/`has_radius_field`.
- `holes2`/`surfaceSplines2` are written in the v2021 entry shape
  (unconfirmed, sample empty — same as open item 4 below).
- Collection `options.OffsetIndex` is written as an int (`0`); the game
  writes a float (`0.0`). JSON-number-equivalent, so it's left alone unless
  the in-game load says otherwise.
- **Catalog corrected:** `HoleSign01Prefab` (black) was already used in
  v2021 (`~/.pga2k/collections/hole_sign.json`, 2026-09-20), so its
  `min_game_version` is now `"2021"`. Andy confirmed the other four signs
  are new in v2023. `GolfCartPrefab` is still unverified for v2021.
- **In-game VERIFIED 2026-09-23 (Andy):** `LIDAR-2023-shawnee-phase2`
  (shawnee re-targeted to 2023) opens in v2023 with every pre-existing
  feature rendering correctly: trees, ponds + stream water, waterfalls,
  holes/pins, splines, stakes, fills/clusters, parking, range nets, hole
  signs, elevated collection props. So the v2021-shaped `water[]`,
  `holes2` and `surfaceSplines2` entries and the int `OffsetIndex` are all
  accepted by v2023. That's accepted, not confirmed to match what the game
  writes: a game re-save could still trim fields.

## OSM fence/wall ingest + asset routing (task 3.1, 2026-09-23)

**Kinds** (`ingest/osm.py` `classify_way`, checked *last* so a way that
already classifies as something else keeps its kind, e.g. `building=yes +
barrier=wall` stays `building`):

| OSM tag | kind |
|---|---|
| `barrier=wall`, `barrier=retaining_wall`, `barrier=city_wall` | `wall` |
| `barrier=hedge`, `natural=hedge` | `hedge` |
| **any other `barrier=*`** (2026-09-27, Andy: "anything with a barrier tag is a fence of some kind") | `fence` |

Exceptions: `barrier=range_nets` / `range_net` (singular accepted) stay
`range_net`. Passage barriers (`gate`, `lift_gate`, `stile`, `bollard`,
`cattle_grid`, ... — `_NON_FENCE_BARRIERS`) are dropped. Drawn as a way, a
gate spans the gap in a fence, so a fence there would close the opening.

**Always lines, never areas**, even for a closed ring or `hedge area=yes`.
OSM draws a perimeter fence as a closed way, and the objectPath runs along
it. A closed way stays a closed `LineString` (`coords[0] == coords[-1]`), so
3.2 can still tell it's a closed run. No spline/hole/mask/water consumer
matches these kinds: `feature_to_spline` returns None, and `mask` defaults
True.

**Tag → fence type** (`course_output/fences.py` `fence_type_for_tags`,
first match wins; superseded 2026-09-27 by the fence-types table below —
the original 3.1 table routed to 8 assets, and wrongly sent chain_link to
UniFence, which is "Fence - metal"):

1. `wall=retaining_wall` / `barrier=retaining_wall` → `retaining_wall`
2. kind `hedge` → `hedge`
3. `material=*`, then `wall=*`, then `fence_type=*`, then `barrier=*`,
   through `_MATERIAL_TYPES`:
   - stone / dry_stone / flint / city_wall → `stone_wall`; cobblestone → `stone_chunky`
   - brick → `brick_low`; concrete / concrete_block / cinder_block → `brick_cinder`
   - wood / pole / rail → `wood_rustic`; split_rail → `three_rail_natural`
   - palisade / wood_panel / wooden_panels / panel / board / privacy → `wood_panels`
   - chain_link / wire / mesh / barbed_wire / electric → `chain_link`
   - metal / steel / metal_bars / bars / railing / guard_rail / handrail / chain → `metal`
   - hedge → `hedge` (Andy tags hedges `barrier=fence material=hedge`)
   - canvas → `canvas_green`; white_canvas / red_canvas / black_canvas /
     blue_canvas / green_canvas → that colour
   - **any fence-type name verbatim** (`material=picket`, `material=high_metal_fence`, ...)
4. kind default: fence → `wood_rustic`, wall → `stone_wall`

Probe: `python util/probe_fence_tags.py [extract.osm]`. With no argument it
checks a synthetic extract (31 cases incl. regressions); with an extract it
reports.

## Fence types (the game's fence menu, 2026-09-27)

Captured from Andy's fence sampler `LIDAR-2023-20260927105132` (one run of
each menu entry, placed with the editor's defaults). The menu is
**alphabetical by asset name**: sorted by position, the sampler's runs are
Andy's list 1–28 in reverse, and every name-identifiable entry lines up.
`FENCE_TYPES` in `course_output/fences.py` holds this table; `pga_fence_asset`
takes the type name, the menu label, or a part asset.

**Multi-part types:** three entries write two objectPaths over the same
waypoints, one per part asset. `build_fence_records` emits one record per
part. On these types `width`/`height`/`heightRule` overrides apply to both
parts (`SHARED_RULE_FIELDS`), and the other overrides apply to the first part only.

| # | menu name | type name | part asset(s) (`Walls/…`) | sampler rules (sR/flex/curves/spacing) |
|---|---|---|---|---|
| 1 | Brick wall - Asian green cap | `asian_green_cap` | `Asia_Walls/Asia_BrickWalls_PostPrefab` | 1/1/T/4 |
| 2 | Brick wall - Asian black cap | `asian_black_cap` | `Asia_Walls/Asia_Walls_PostPrefab` | 1/1/T/4 |
| 3 | Black canvas wall | `canvas_black` | `CanvasFence01AABlackPostAPrefab` | 2/1/T/4 |
| 4 | Blue canvas wall | `canvas_blue` | `CanvasFence01AABluePostAPrefab` | 2/1/T/4 |
| 5 | Brick with railings | `brick_with_railings` | `BrickWallsLowRailsAPostAPrefab` + `LowMetalFenceInnerPost01Prefab` | 0/1/T/4 + 2/1/T/0.1, both h=-0.594 (lowered to railing height) |
| 6 | Stone wall - chunky rounded | `stone_chunky` | `Brit_Walls/Brit_CobbleWalls_GeneratePostPrefab` | 3/1/T/4 |
| 7 | Brick wall - pavers | `brick_pavers` | `Brit_Walls/Brit_LowWalls_GeneratePostPrefab` | 2/1/T/4 |
| 8 | Brick wall - red brick | `brick_red` | `Brit_Walls/Brit_BrickWalls_01_GeneratePostPrefab` | 2/1/T/4 |
| 9 | Brick wall - big stone | `brick_big_stone` | `Brit_Walls/Brit_BrickWalls_02_GeneratePostPrefab` | 2/1/T/4 |
| 10 | Brick wall - cinder | `brick_cinder` | `Brit_Walls/Brit_BrickWalls_03_GeneratePostPrefab` | 2/1/T/4 |
| 11 | Brick wall - quarried | `brick_quarried` | `Brit_Walls/Brit_Walls_01_GeneratePostPrefab` | 2/1/T/4 |
| 12 | Green canvas wall | `canvas_green` | `CanvasFence01AAPostAPrefab` | 2/1/T/4 |
| 13 | Hedge | `hedge` | `HedgeSplinePostPrefab` | 1/1/T/4 |
| 14 | High brick wall - classic brick | `brick_high` | `BrickWallsHighAPostAPrefab` | 2/1/T/4 |
| 15 | High metal fence | `high_metal_fence` | `BrickWallsHighRailsAPostAPrefab` + `HighMetalFenceInnerPost01Prefab` | 0/1/T/4 + 2/1/T/0.1, both h=-0.104 (lowered to railing height) |
| 16 | Brick wall - Asian panels | `asian_panels` | `Asia_Walls/Asia_KoreanWalls_PostPrefab` | 1/1/T/4 |
| 17 | Low brick wall - classic brick | `brick_low` | `BrickWallsLowAPostAPrefab` | 2/1/T/4 |
| 18 | Picket fence | `picket` | `PicketFencePicket01Prefab` + `PicketFencePost01APrefab` | 2/0/F/0.1 + 2/0/F/4 |
| 19 | Red canvas wall | `canvas_red` | `CanvasFence01AARedPostAPrefab` | 2/1/T/4 |
| 20 | Retaining wall | `retaining_wall` | `RetainWallAPostAPrefab` | 0/0/F/4 |
| 21 | Stone wall | `stone_wall` | `StoneWallAPostAPrefab` | 2/1/T/4 |
| 22 | Fence - 3-rail white | `three_rail_white` | `TriFence01Post01APrefab` | 2/1/T/4 |
| 23 | Fence - 3-rail natural | `three_rail_natural` | `TriFence02Post01APrefab` | 2/1/T/4 |
| 24 | Fence - metal | `metal` | `UniFencePostAPrefab` | 2/1/T/4 |
| 25 | White canvas wall | `canvas_white` | `CanvasFence01AAWhitePostAPrefab` | 2/1/T/4 |
| 26 | Wire fence - chain-link | `chain_link` | `WireFenceAPostAPrefab` | 2/1/T/4 |
| 27 | Wooden fence - 2-rail rustic | `wood_rustic` | `WoodFencesAPostAPrefab` | 2/1/T/4 |
| 28 | Wooden panels | `wood_panels` | `WoodFencesBPostCPrefab` | 2/1/T/4 |

All rows are width 4, heightRule 0, and height 0 unless noted. **Part
defaults** (`ASSET_DEFAULT_RULES`) are these sampler values, except for the 8
assets the 9/26 fence test already verified, which keep their
`2023_fences.course` defaults: stone wall sR 0 (menu 2); Asian green cap
0/0/F (menu 1/1/T); retaining wall w 2.5, h -1.015, curves T, flex 1 (menu
w 4, h 0, curves F, flex 0). The sampler's values are merged into each
asset's `fence_options` in `asset_catalog.json` (every new set is
`complete: false`).

## objectPath handle rule + builder decisions (task 3.2, 2026-09-23)

**Handle rule (resolves 0.3a.6), derived from the sample rows.**
`pointOne`/`pointTwo` are real bezier handles, populated the same way
whether `hasCurves` is true or false:

- **Open run, end waypoints:** the outer handle sits on the waypoint (the
  first waypoint's `pointOne` and the last one's `pointTwo`). The inner
  handle is **0.25 ×** the segment, pointing toward the neighbour.
- **Interior / closed-run waypoints (smooth):** both handles lie along the
  tangent `normalize(next − prev)`. Each is **0.375 ×** its own adjacent
  segment length, so they're asymmetric when the segments differ.
- `course_output/fences.py` rebuilds 12 of the 13 sample rows from their
  waypoints alone to within 0.1 mm (`tests/test_fences.py`). The exception
  is the `WoodFences` row: its handles are 1.03 m, which is 0.25 × 4.1 m
  on a 15.2 m segment. Its end was evidently dragged after drawing without
  the handles being re-derived, so the game tolerates stale/arbitrary
  handles.

**`state` = 1 → closed loop (inferred).** The sample's only `state=1` row is
its only closed run: the retaining wall, 4 waypoints, first not repeated,
and the last waypoint's `pointTwo` points back at the first. The builder
writes `state` 1 for closed runs and 0 for open ones. Confirm in-game
(3.5.3).

**Builder choices (ours, not the game's):**
- **Corners.** An OSM way is a polyline with real corners, but the game
  editor smooths every interior waypoint (the sample's 4-point diamond
  renders as a circle). For an interior waypoint that turns more than
  `CORNER_ANGLE_DEG` (45°), or any waypoint when `hasCurves=false`, the
  builder writes *corner handles*: each handle sits on its own segment
  (0.25 ×), so the bezier stays straight through a sharp corner. Gentle
  bends keep smooth handles.
- **Joining ways.** Ways with the same resolved style (asset + every rule
  field) are joined end-to-end where exactly two ends meet (0.5 m
  tolerance), giving one post per shared node and no gap. At a 3-way
  junction the runs stay separate. A joined run whose ends meet becomes
  closed. Where different styles meet, each keeps its own end posts;
  that can't be avoided.
- **Simplify.** Douglas-Peucker at 0.25 m, applied as a ring for closed
  runs so the seam isn't pinned.
- **Per-asset defaults** are that asset's plain (contoured, unburied)
  sample row. Brick defaults to `flexibilityRule` 0 and `hasCurves` false,
  as in its sample row. The retaining wall defaults to its only sample
  row: width 2.5, height −0.715.
- **Presets** (`pga_fence_preset`): `curb` (stone, h −1.498), `railroad`
  (black canvas, h −2.688), `retaining_wall` (w 2.5, h −0.715).
- **Per-way overrides** use this project's own tags `pga_fence_asset`
  (label or path) and `pga_fence_<field>` (e.g. `pga_fence_spacingRule=3`).
  Precedence: asset default < preset < field tag. The GUI (3.3.3) will
  write these.
- **Option-matrix check:** a value outside a `complete` option set warns
  and reverts to the asset default. A value outside an incomplete set
  warns "unverified" and is kept, so in-game trials can extend
  `asset_catalog.json`.
- **Leveled (`heightRule=1`)** height is resolved at write time. See
  "Leveled fence height" (superseded the original "doesn't auto-set a
  height" note, 3.5.6).
- **Output:** `fences.json` stores the frozen `FenceRecord`s in the local
  frame. `fence_records_to_groups_v2023` emits `{Key:{path}, Value:
  {objectPaths}}` groups for `merge_object_groups` +
  `placed_object_groups_to_v2023`.

## Fence pipeline wiring (task 3.3, 2026-09-24)

- **`--step generate-fences`** (`step_generate_fences`): crops
  `features.geojson` to the course, builds from the fence/wall/hedge
  kinds, and writes **`fences.json`**. It prints per-way style warnings.
  Flags: `--fence-simplify-tol`, `--fence-endpoint-tol`,
  `--fence-corner-angle` (negative = smooth everywhere, like the game
  editor) and `--clear-fences`.
- **Not routed through `objects.json` / pack-objects.** Fences are paths,
  not placed objects, and there's nothing to pack. `_build_placed_objects`
  reads `fences.json` directly (the same way it reads `streams.json`) and
  adds `fence_records_to_groups_v2023` groups before `merge_object_groups`
  + `placed_object_groups_to_v2023`. Only v2023+ (`has_fences`) gets them;
  v2019/v2021 print a NOTE and drop them. `step_repack` needed no change,
  since it packs `course/` as-is. `step_import_ingame_edits` only diffs
  `items`, so fences never show up as user edits. The flip side: fences
  drawn by hand in the game aren't imported.
- **`project.json` keys:** `fence_simplify_tol_m`, `fence_endpoint_tol_m`
  and `fence_corner_angle_deg` (settings, CLI → persisted), plus
  `fence_count` and `fence_assets` (what the last run produced). The
  corner angle is also frozen into each `fences.json` record, so
  write-objects stays a pure formatter.
- **Per-way style** lives in `pga_fence_*` tags on `features.geojson`. The
  GUI's Objects → "Fences & Walls (v2023)" panel writes them with Apply to
  Selected / Reset Selected, acting on the Splines-tab selection.
  `ingest-osm` carries them over a re-ingest, per tag, and a value coming
  from OSM itself wins, the same rule as `pga_cluster_fills`. The Splines
  tab detail column shows each fence row's resolved asset.
- **End-to-end (code-level) check:** a copy of `bouldercreek2` (v2023) with
  6 injected fence ways runs `generate-fences` → `write-objects` →
  `repack`. The result is a version-31 `.course` whose `placedObjects3`
  has 5 objectPaths (1 closed wood square; 2 brick ways joined into 1 run;
  a curved hedge; chain-link with `spacingRule=3`, warned as unverified; a
  curb preset) alongside 3150 items, and every group carries the v2023
  envelope. **Not yet loaded in-game** (that's 3.5).
- `bouldercreek2/course/` was reset to the v2023 baseline
  (`--step ingest-course`) on 2026-09-27 for 3.5.2.

## Leveled fence height (task 3.5.6, 2026-09-27)

**Game semantics (Andy, in-game):**
- `heightRule=0` (contoured): each panel skews along the ground, so the
  top follows the terrain; `height` is an offset from the ground.
- `heightRule=1` (the editor calls it **leveled**): the whole run sits at
  **one absolute `height`**, and the editor sets it to the run's
  **minimum** ground height. Andy first thought each panel anchored at
  its own start height (true steps), but an in-game test showed it
  doesn't.

**Pipeline rule:** in `fences.json`, `height` stays an **offset in both
modes** (`pga_fence_height`; default 0 = what the editor does). At
write-objects, `_build_placed_objects` resolves leveled runs with
`fences.apply_leveled_heights`: absolute `height` = min over the run of
`TerrainModel.evaluate` + `output_height_shift_m` + offset. Details:
- Sampling: every 1 m along every segment, including a closed run's wrap
  segment, so a dip between waypoints counts. The terrain model is the
  one `_cached_terrain_model_for_objects` builds for elevated collection
  objects.
- Multi-part types share points and offset, so both parts get the same
  height.
- `push-fence-test` rows keep explicit absolute heights and bypass this.

Checked on a bouldercreek2 scratch copy: the emitted heights match an
independent 0.25 m terrain sweep to within 6 mm, and sit at each run's
low point.

## Course filename rule (CONFIRMED in-game, 2026-09-26)

A full pipeline build repacked through `game_safe_course_stem`
(`LIDAR_2023_bouldercreek_fences`) loads too (2026-09-27).

The game can't load, or delete, a `.course` whose **filename** has a hyphen
followed by an all-letter last segment (`FT2-1-S-compact`,
`BZ1-shawnee-good`, `LIDAR-2023-bouldercreek-fences`). Byte-identical files
renamed `BZ2-shawnee-good2` / `BZ5-FT2-1-copy5` load, and so do
`hinckleyhills`, `hhills3` and `2023_fences`. The stored name (CD `name`,
CM `name`) and the `_id` don't matter: the game rewrites `_id` to
`offlineSave<filename>` (and the Thumbnail `_id` to `…-Thumb`) the first time
it opens a course, and it re-saves every course it opens. Likely cause: the
game names a course's parts `<id>-Meta` / `<id>-Thumb`, so it strips a
`-<Letters>` tail as a part suffix. The pipeline writes game filenames
through `PGA2k_gen.game_safe_course_stem` (hyphens → underscores).

Also found in the 3.3a bisection: the game writes gzip at **zlib level 1**
(header flg=0, xfl=0, os=255, mtime = local wall-clock seconds), but it
reads our level 9 fine. Encoding level, BOMs, JSON whitespace and key order
all load.

## What this does NOT confirm (still open)

1. **Per-asset fence option matrix**: partly answered by the in-game fence
   test above (WoodFences: all 4 spacingRules). The rest still needs
   per-asset trials.
2. `flexibilityRule` semantics; `state` = closed loop is inferred, not
   confirmed (see "objectPath handle rule" above).
3. ~~`spacingRule` ordering~~ and ~~stepped height~~: both resolved (see "In-game fence
   test results").
4. Whether `holes2`/`surfaceSplines2` entry shapes differ from
   `holes`/`surfaceSplines` (sample has both empty).
5. Clear-objects details: in-game confirmation that category 5 actually
   suppresses generated objects (and which kinds), whether v2021 honors it too,
   and the exact `value` semantics of the surface stamp.
