from enum import Enum


class AlphaMode(str, Enum):
    """Controls which absorption/dispersion terms are included in the equation of state."""

    NO_ABSORPTION = "no_absorption"
    NO_DISPERSION = "no_dispersion"
    STOKES = "stokes"

    def __str__(self):
        return self.value


################################################################
# literals that link the discrete cosine and sine transform types with
# their type definitions in the functions dtt1D, dtt2D, and dtt3D


class DiscreteCosine(Enum):
    TYPE_1 = 1
    TYPE_2 = 2
    TYPE_3 = 3
    TYPE_4 = 4


class DiscreteSine(Enum):
    TYPE_1 = 1
    TYPE_2 = 2
    TYPE_3 = 3
    TYPE_4 = 4


################################################################
