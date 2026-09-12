"""Loader for vehicle_catalog.json -- the PGA 2K 'Land' vehicle (parked-car)
prefabs, their colour variants, and the editor's pitch/roll tilt clamp.

Groundwork for the parking-lot auto-fill feature (line an OSM service road /
amenity=parking with cars). Same "pure helper, imported directly by the GUI
and the CLI step" shape as asset_catalog.py / tree_themes.py.

The catalog was reverse-engineered from a hand-authored reference capture
(C:/Users/andy_/Desktop/golf/py/cars): one prop of every car in a single row,
colour-annotated by the user, with the four Car56 variants set to the editor's
tilt max/min so the clamp range could be read off.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VEHICLE_CATALOG_JSON = Path(__file__).resolve().parent / "vehicle_catalog.json"

# Editor clamp on a vehicle prop's tilt, degrees. rotation.x = pitch,
# rotation.z = roll, rotation.y = yaw (free). Bank parked cars within this
# range to the local ground slope rather than leaving them axis-flat.
VEHICLE_PITCH_LIMIT_DEG = 10.0
VEHICLE_ROLL_LIMIT_DEG = 10.0

# Fallback footprint numbers (meters). Only lateral shoulder-to-shoulder
# spacing was measurable from the single-row capture; length is a guess.
VEHICLE_SIDE_SPACING_M = {"car": 2.25, "van": 2.70}
VEHICLE_LENGTH_M = {"car": 4.5, "van": 5.5}


@dataclass(frozen=True, slots=True)
class VehicleVariant:
    path: str
    color: str
    group: str            # user's model-group label ("1".."7", "v")
    model_id: str          # "56", "51", ... , "Van01"
    vehicle_class: str     # "car" | "van"

    @property
    def side_spacing_m(self) -> float:
        return VEHICLE_SIDE_SPACING_M.get(self.vehicle_class, 2.25)

    @property
    def length_m(self) -> float:
        return VEHICLE_LENGTH_M.get(self.vehicle_class, 4.5)


def _load() -> tuple[list[VehicleVariant], dict]:
    if not VEHICLE_CATALOG_JSON.exists():
        return [], {}
    data = json.loads(VEHICLE_CATALOG_JSON.read_text(encoding="utf-8"))
    out: list[VehicleVariant] = []
    for model in data.get("models", []):
        for v in model.get("variants", []):
            out.append(VehicleVariant(
                path=v["path"],
                color=v["color"],
                group=str(model.get("group", "")),
                model_id=str(model.get("model_id", "")),
                vehicle_class=model.get("class", "car"),
            ))
    return out, data


VEHICLE_VARIANTS: list[VehicleVariant]
_CATALOG_META: dict
VEHICLE_VARIANTS, _CATALOG_META = _load()

VEHICLE_PATHS: list[str] = [v.path for v in VEHICLE_VARIANTS]
VEHICLE_PATH_SET: frozenset[str] = frozenset(VEHICLE_PATHS)


def variants_by_color(color: str) -> list[VehicleVariant]:
    """Every variant of the given colour (case-insensitive). The user's
    spelling wins -- e.g. 'fuschia', not 'fuchsia'."""
    c = color.strip().lower()
    return [v for v in VEHICLE_VARIANTS if v.color == c]


def variant_for_path(path: str) -> Optional[VehicleVariant]:
    for v in VEHICLE_VARIANTS:
        if v.path == path:
            return v
    return None


def pick_pool(
    colors: "list[str] | None" = None,
    classes: "list[str] | None" = None,
) -> list[VehicleVariant]:
    """The variant pool a fill should draw from, filtered by colour and/or
    class ('car'/'van'). Empty filters mean 'all'. Returns every variant
    (full catalog) when nothing matches, so a fill never ends up empty."""
    pool = VEHICLE_VARIANTS
    if colors:
        want = {c.strip().lower() for c in colors}
        pool = [v for v in pool if v.color in want] or pool
    if classes:
        wantc = {c.strip().lower() for c in classes}
        pool = [v for v in pool if v.vehicle_class in wantc] or pool
    return list(pool)


def random_pool_sequence(
    count: int,
    rng: random.Random,
    colors: "list[str] | None" = None,
    classes: "list[str] | None" = None,
) -> list[VehicleVariant]:
    """`count` variants drawn with replacement from the filtered pool, no
    two adjacent identical where the pool allows it -- for scattering a row
    of parked cars that doesn't read as copy-pasted."""
    pool = pick_pool(colors, classes)
    if not pool:
        return []
    seq: list[VehicleVariant] = []
    for _ in range(count):
        choice = rng.choice(pool)
        if len(pool) > 1 and seq and choice.path == seq[-1].path:
            choice = rng.choice([p for p in pool if p.path != seq[-1].path])
        seq.append(choice)
    return seq
