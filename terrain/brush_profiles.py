@dataclass(frozen=True, slots=True)
class BrushProfile:
    brush_id: int
    samples: np.ndarray

BRUSH_PROFILES = {
    8: BrushProfile(...),
    9: BrushProfile(...),
    10: BrushProfile(...),
    54: BrushProfile(...)
}
