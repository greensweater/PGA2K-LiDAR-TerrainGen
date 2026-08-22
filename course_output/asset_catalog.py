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

    @property
    def label(self) -> str:
        """Last path segment, e.g. 'Lombardy_Popular_Desktop01' -- what the GUI's asset picker shows."""
        return self.path.rsplit("/", 1)[-1]


def _load() -> tuple[dict[int, AssetCategory], list[AssetEntry]]:
    data = json.loads(ASSET_CATALOG_JSON.read_text())
    categories = {
        c["id"]: AssetCategory(id=c["id"], description=c["description"], cluster_radius=c.get("cluster_radius"))
        for c in data["categories"]
    }
    entries = [
        AssetEntry(category=e["category"], type=e["type"], theme=e["theme"], path=e["path"], spacing=e.get("spacing"))
        for e in data["entries"]
    ]
    return categories, entries


ASSET_CATEGORIES, ASSET_ENTRIES = _load()

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
