@dataclass(slots=True)
class Stamp:
    """A single PGA terrain stamp in world coordinates."""

    x: float
    z: float

    radius: float
    amplitude: float

    brush: int

    rotation: float = 0.0
