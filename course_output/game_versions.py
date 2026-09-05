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
fences, texture painting) and 2025 (spline water) add is new
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

"2023"/"2025" entries are UNCONFIRMED placeholders -- their real
schemas haven't been diffed against an extracted .course file the way
2019 vs 2021 was (see hhills3_2019/hhills3_2021). They exist here so
GAME_VERSIONS can list them for UI/CLI purposes, but
IMPLEMENTED_GAME_VERSIONS keeps them out of anything that would
actually try to build output for them.
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
    has_orientation_fields: bool = False  # v2021+ userLayers height-entry _orientation/orientation
    has_fences: bool = False  # v2023+ -- UNCONFIRMED placeholder, not implemented
    has_texture_paint: bool = False  # v2023+ -- UNCONFIRMED placeholder, not implemented
    has_spline_water: bool = False  # v2025+ -- UNCONFIRMED placeholder, not implemented


VERSION_SCHEMAS: dict[str, VersionSchema] = {
    "2019": VersionSchema(version="2019", objects_filename="placedObjects2.json"),
    "2021": VersionSchema(
        version="2021", objects_filename="placedObjects3.json", theme_scheme="path",
        has_object_splines=True, has_pins_field=True, has_orientation_fields=True,
    ),
    # "2023"/"2025" entries added once their real schemas are confirmed the
    # same way hhills3_2019/hhills3_2021 were diffed -- not populated
    # speculatively.
}

# Kept as strings (not ints) since "2019" etc. are display/config labels,
# not quantities -- nothing here does arithmetic on a version.
GAME_VERSIONS = tuple(VERSION_SCHEMAS)
IMPLEMENTED_GAME_VERSIONS = ("2019", "2021")
DEFAULT_GAME_VERSION = "2019"


def schema_for(game_version: str) -> VersionSchema:
    try:
        return VERSION_SCHEMAS[game_version]
    except KeyError:
        raise ValueError(
            f"Unknown game_version {game_version!r} -- expected one of {GAME_VERSIONS}"
        ) from None
