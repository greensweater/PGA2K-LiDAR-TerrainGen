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

The `height` layer entry shape is unchanged (tool/position/rotation/scale/
type/value/holeId); `y: "-Infinity"` = terrain-grounded, same as v2021.

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
| `spacingRule` | 0, 1, 2, 3 | **end caps, 4 options: none / spaced / points / ends** (order unverified — needs in-game check) |
| `heightRule` | 0, 1 | **0 = contoured** (follows heightmap), **1 = stepped** (fixed raised height; all ht=1 rows carry `height ≈ 3.23`) |
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
→ **In-game subtask: confirm the HoleSign0N ↔ color mapping** (depends on
which z direction is "north" in-game).

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

## What this does NOT confirm (still open)

1. **Per-asset fence option matrix** (which spacingRule/hasCurves/heightRule
   are valid per fence asset; stone wall confirmed full 4 caps by the author)
   — needs in-game trial per asset.
2. `flexibilityRule` and `state` semantics.
3. `spacingRule` → cap-style ordering (which of 0–3 is none/spaced/points/ends).
4. Whether `holes2`/`surfaceSplines2` entry shapes differ from
   `holes`/`surfaceSplines` (sample has both empty).
5. Clear-objects details: in-game confirmation that category 5 actually
   suppresses generated objects (and which kinds), whether v2021 honors it too,
   and the exact `value` semantics of the surface stamp.
