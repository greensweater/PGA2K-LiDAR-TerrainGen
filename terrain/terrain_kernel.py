class TerrainKernel:
    def __init__(self, profile: BrushProfile):
        ...

    def sample(self, r: float) -> float:
        """
        Sample normalized brush amplitude.

        Parameters
        ----------
        r
            Normalized distance from center.
            0 = center
            1 = brush edge

        Returns
        -------
        float
            Normalized height (0..1)
        """
