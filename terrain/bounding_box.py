@dataclass(slots=True)
class BoundingBox:
    min_x: float
    min_z: float
    max_x: float
    max_z: float
