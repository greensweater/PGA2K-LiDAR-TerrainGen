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
  **`surfaceSplines2`**.
- New base keys: `blurHeight`, `useV28FairwaySeed`, `useV30Rough`.
- `placedObjects3` is the SAME key name in both (confirmed: v2023 writes
  `placedObjects3.json`, matching the user-confirmed fact).
- `CourseMetadata.json`, `meta.json`, `Thumbnail` — structurally unchanged.

## userLayers2.json (v2023)

Sample keys (blank-ish course): `treeDensity`, `terrainHeight`, `height`,
`surfaces`, `outOfBounds`, `crowdLocations`, `water`. The v2021
`userLayers.json` additionally had `deletedHazards`, `newHazards`, `objects`,
`clearTrees`, `addTrees`, `hazards`, `trees`, `green` — likely omitted when
empty in v2023 (or relocated). **UNCONFIRMED**: whether a "clear generated
objects" layer exists in v2023 and its key name — the sample course has no
clear-objects paint. Need a course with that feature painted (see task 0.3b).

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

The brick wall sample exercises all four spacingRule values → the author
described the **stone fence** as having the full option set (caps ×4,
stepped/contoured, curved/straight); canvas/hedge rows in the sample do not
exercise every value. Per-asset option matrix must be confirmed in-game.

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

## What this does NOT confirm (still open for Phase 0)

1. The **clear-generated-objects** layer: name, entry shape, and which
   generated object types it suppresses (2021's userLayers has `clearTrees`;
   v2023 sample has no painted example).
2. Per-asset fence option matrix (which spacingRule/hasCurves/heightRule are
   valid per fence type) — needs in-game trial per asset.
3. `flexibilityRule` and `state` semantics.
4. Whether `holes2`/`surfaceSplines2` entry shapes differ from
   `holes`/`surfaceSplines` (sample has both empty).
5. `spacingRule` → cap-style ordering (which of 0–3 is none/spaced/points/ends).
