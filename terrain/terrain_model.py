class TerrainModel:

    def evaluate(self, x: float, z: float) -> float:
        ...

    def evaluate_many(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        ...

    def render(
        self,
        resolution: int
    ) -> np.ndarray:
        ...
