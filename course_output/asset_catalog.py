"""
course_output/asset_catalog.py

Loads asset_catalog.json -- the category/type -> v2021+ asset path
mapping (and, for "nature" categories, the per-asset planting density
needed to size a cluster stamp) the user derived by placing every
catalog item in a v2019 file and re-saving it from v2021, then diffing
item positions between the two (see that file's own "note" field for
the exact method). Powers the "fill selected splines with clusters" GUI
action -- course_output/object_clusters.py does the actual raster-fill/
group-building -- by letting the user pick a real named asset instead
of a bare numeric type id, and get a correctly-sized cluster back.

Also carries per-prefab native dimensions (native_height_m /
native_canopy_radius_m, category-0 only), measured in-game by the user
(see the JSON's "native_size_note" and util/extract_tree_dimensions.py).
course_output/objects.py._tree_scale uses native_height_m to size a
placed tree to its LIDAR-detected height instead of a course-relative
guess -- keyed by asset path for v2021+ (NATIVE_TREE_HEIGHT_BY_PATH),
by per-theme numeric type id for v2019 (native_tree_height_v2019, gated
on CATALOG_SOURCE_THEME_ID since those ids are theme-specific).
native_canopy_radius_m is captured but not yet consumed (scale is
uniform x=y=z).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ASSET_CATALOG_JSON = Path(__file__).resolve().parent / "asset_catalog.json"


@dataclass(frozen=True, slots=True)
class AssetCategory:
    id: int
    description: str
    cluster_radius: Optional[float]  # None = not a scatterable/"nature" category -- see NATURE_CATEGORY_IDS


@dataclass(frozen=True, slots=True)
class AssetEntry:
    category: int
    type: int
    theme: bool
    path: str
    spacing: Optional[float]  # None = no measured planting density -- can't be cluster-filled, see CLUSTERABLE_ENTRIES
    # In-game-measured native dimensions at scale 1.0, meters. category-0
    # only, and (for the v2019 numeric type id) valid ONLY for
    # CATALOG_SOURCE_THEME_ID -- type ids are per-theme. None = not
    # measured; course_output/objects.py._tree_scale then falls back to
    # the course-relative height remap for that prefab. See the JSON's
    # "native_size_note" for how they're captured.
    native_height_m: Optional[float] = None
    native_canopy_radius_m: Optional[float] = None
    # Free-text, human-written blurb for entries whose prefab name is
    # useless at a glance (e.g. "NPBush08A") -- purely cosmetic, never
    # read by the write-objects pipeline. None = no blurb written yet;
    # display() then falls back to `label`.
    description: Optional[str] = None

    @property
    def label(self) -> str:
        """Last path segment, e.g. 'Lombardy_Popular_Desktop01' -- what the GUI's asset picker shows."""
        return self.path.rsplit("/", 1)[-1]

    @property
    def display(self) -> str:
        """`description` (with the raw prefab name parenthesized) when one's been
        written, else just `label` -- what any GUI list of assets should show."""
        return f"{self.description} ({self.label})" if self.description else self.label


def _load() -> tuple[dict[int, AssetCategory], list[AssetEntry], Optional[int]]:
    data = json.loads(ASSET_CATALOG_JSON.read_text())
    categories = {
        c["id"]: AssetCategory(id=c["id"], description=c["description"], cluster_radius=c.get("cluster_radius"))
        for c in data["categories"]
    }
    entries = [
        AssetEntry(
            category=e["category"], type=e["type"], theme=e["theme"], path=e["path"],
            spacing=e.get("spacing"),
            native_height_m=e.get("native_height_m"),
            native_canopy_radius_m=e.get("native_canopy_radius_m"),
            description=e.get("description"),
        )
        for e in data["entries"]
    ]
    return categories, entries, data.get("source_theme_id")


ASSET_CATEGORIES, ASSET_ENTRIES, CATALOG_SOURCE_THEME_ID = _load()

# Native tree heights (m) for calibrated placed-tree scale in
# course_output/objects.py. v2021+ keys placed objects by global asset
# path; v2019 by a per-theme numeric type id, so the type-keyed lookup is
# only meaningful for CATALOG_SOURCE_THEME_ID (see native_tree_height_v2019).
NATIVE_TREE_HEIGHT_BY_PATH: dict[str, float] = {
    e.path: e.native_height_m for e in ASSET_ENTRIES if e.native_height_m
}
_NATIVE_TREE_HEIGHT_BY_TYPE: dict[int, float] = {
    e.type: e.native_height_m
    for e in ASSET_ENTRIES if e.category == 0 and e.native_height_m
}


def native_tree_height_v2019(type_id: int, theme: Optional[int]) -> Optional[float]:
    """Native height (m) for a v2019 category-0 tree `type_id` under
    `theme`, or None if unknown or `theme` isn't the theme this catalog
    was captured from -- v2019 type ids are per-theme, so a native height
    measured for rustic's "type 14" must not be applied to another
    theme's "type 14"."""
    if theme is None or CATALOG_SOURCE_THEME_ID is None or theme != CATALOG_SOURCE_THEME_ID:
        return None
    return _NATIVE_TREE_HEIGHT_BY_TYPE.get(type_id)

# "Nature" categories -- the only ones with a cluster_radius (the game's
# own max scatter-stamp size for that category) at all, so the only ones
# a cluster fill makes sense for. Trees/bushes, rocks, grass, ground-
# cover and detail plants today; driven by the data, not a hardcoded id
# list, so a future catalog addition (or removal) Just Works.
NATURE_CATEGORY_IDS = {c.id for c in ASSET_CATEGORIES.values() if c.cluster_radius is not None}

# Only entries with real measured spacing can be cluster-filled (see
# asset_catalog.json's own "missing_spacing_data"/null-spacing notes) --
# an entry without one has no way to compute a sane instance count.
CLUSTERABLE_ENTRIES = [e for e in ASSET_ENTRIES if e.category in NATURE_CATEGORY_IDS and e.spacing]


def cluster_count(cluster_radius: float, spacing: float) -> int:
    """
    Instance count for a cluster_radius stamp of an asset with this
    spacing (average center-to-center distance between instances),
    inverting the catalog's own back-calculation formula (see
    asset_catalog.json's "note"): spacing = radius / sqrt(count / pi).
    """
    return max(1, round(math.pi * (cluster_radius / spacing) ** 2))
