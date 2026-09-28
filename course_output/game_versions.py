"""
course_output/game_versions.py

Single source of truth for what varies across PGA 2K's .course schema
versions (2019 -> 2021 -> 2023 -> 2025) -- node filenames and per-
version capability flags. Every other course_output/*.py module and
PGA2k_gen.py imports GAME_VERSIONS/IMPLEMENTED_GAME_VERSIONS/
DEFAULT_GAME_VERSION/schema_for from here rather than owning them
(objects.py used to own the version constants; moved here so
holes.py/userLayers.py can depend on version info without an odd
dependency on objects.py).

This is deliberately a registry of filenames + capability flags, NOT a
generic schema-mapping/transform engine -- most of what 2023 (spline
fences) and 2025 (spline water, texture painting) add is new
*generation logic* with new inputs, not a reshaping of data this
project already produces, so a declarative field mapper couldn't
shortcut writing it anyway. Where the *same* logical data (a placed
tree, a cluster) really is encoded differently across versions, the
resolution logic itself differs behaviorally (v2019's numeric
category/type/theme triple needs a theme to resolve at all; v2021+'s
asset-path string doesn't; v2019 does species-bucket routing, v2021
does asset-pool random choice + per-tag override) -- so that stays
hand-written per version as an explicit build_X_v2019/_v2021 function
pair (see objects.py), never a generic version-parameterized function.
This registry only centralizes the mechanical bits: which node file a
version writes to, and which capabilities (has_object_splines, etc.)
gate which branch a writer takes.

"2023" is confirmed: templates/2023_fences.course was diffed against
2021_rustic.course the same way hhills3_2019/hhills3_2021 were (findings
in V2023_SCHEMA.md). "2025" is still unconfirmed and gets no entry here
until its schema is diffed the same way.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VersionSchema:
    version: str
    objects_filename: str  # placedObjects2.json / placedObjects3.json
    holes_filename: str = "holes.json"
    splines_filename: str = "surfaceSplines.json"
    userlayers_filename: str = "userLayers.json"
    # "numeric" = v2019's {category,type,theme} triple; "path" = v2021+'s
    # {"path": "Assets/..."} Unity asset path.
    theme_scheme: str = "numeric"
    has_object_splines: bool = False  # v2021+ Value.splines[] fill regions
    has_pins_field: bool = False  # v2021+ holes.json "pins"
    has_orientation_fields: bool = False  # v2021 only: userLayers height/OOB-entry _orientation/orientation
    has_radius_field: bool = True  # v2019/v2021 height/OOB entries carry "radius": 0.0; v2023 dropped it
    # v2023+ placedObjects3 Value.objectPaths[] (schema confirmed, fence
    # writer not built yet). Also gates the v2023 group envelope
    # (objectPaths + IsEmpty on every group, objects.placed_object_groups_to_v2023).
    has_fences: bool = False
    has_texture_paint: bool = False  # v2025+ (not v2023) -- UNCONFIRMED placeholder, not implemented
    # userLayers "surfaces" clear stamps for the game's procedural scatter:
    # surfaceCategory 5 = clear generated objects (trees/plants/grass/rocks),
    # 6 = clear generated trees. Confirmed from editor exports in all three
    # versions (V2023_SCHEMA.md "Clear-generated-objects paint").
    has_clear_objects: bool = True
    # surfaceCategory 11 = clear generated heavy rough (type 72, value 2.0).
    # v2023 only -- the v2019/v2021 editors have no such tool.
    has_clear_heavy_rough: bool = False
    has_spline_water: bool = False  # v2025+ -- UNCONFIRMED placeholder, not implemented


VERSION_SCHEMAS: dict[str, VersionSchema] = {
    "2019": VersionSchema(version="2019", objects_filename="placedObjects2.json"),
    "2021": VersionSchema(
        version="2021", objects_filename="placedObjects3.json", theme_scheme="path",
        has_object_splines=True, has_pins_field=True, has_orientation_fields=True,
    ),
    # v2023 renames three v2021 nodes (holes -> holes2, surfaceSplines ->
    # surfaceSplines2, userLayers -> userLayers2); placedObjects3 keeps its
    # name. All four are top-level CourseDescription keys, same as v2021
    # (see V2023_SCHEMA.md "Version registry"). Its brush-stamp entries
    # (height/surfaces) are trimmed to tool/position/rotation/scale/type/
    # value/holeId -- no _orientation/radius/orientation. has_pins_field
    # is inherited from v2021 unconfirmed (the sample's holes2 is empty).
    "2023": VersionSchema(
        version="2023", objects_filename="placedObjects3.json",
        holes_filename="holes2.json", splines_filename="surfaceSplines2.json",
        userlayers_filename="userLayers2.json", theme_scheme="path",
        has_object_splines=True, has_pins_field=True, has_orientation_fields=False,
        has_radius_field=False, has_fences=True, has_clear_heavy_rough=True,
    ),
    # "2025" added once its real schema is confirmed the same way -- not
    # populated speculatively.
}

# Kept as strings (not ints) since "2019" etc. are display/config labels,
# not quantities -- nothing here does arithmetic on a version.
GAME_VERSIONS = tuple(VERSION_SCHEMAS)
IMPLEMENTED_GAME_VERSIONS = ("2019", "2021", "2023")
DEFAULT_GAME_VERSION = "2019"


def schema_for(game_version: str) -> VersionSchema:
    try:
        return VERSION_SCHEMAS[game_version]
    except KeyError:
        raise ValueError(
            f"Unknown game_version {game_version!r} -- expected one of {GAME_VERSIONS}"
        ) from None
