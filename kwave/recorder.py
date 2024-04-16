from dataclasses import dataclass
from typing import List

from kwave.data import Vector
from kwave.kgrid import kWaveGrid


@dataclass
class Recorder(object):
    def __init__(self):
        # flags which control which parameters are recorded
        self.p = True  #: time-varying acoustic pressure
        self.p_max = False  #: maximum pressure over simulation
        self.p_min = False  #: minimum pressure over simulation
        self.p_rms = False  #: root-mean-squared pressure over simulation
        self.p_max_all = False  #: maximum pressure over simulation at all grid points
        self.p_min_all = False  #: minimum pressure over simulation at all grid points
        self.p_final = False  #: final pressure field at all grid points
        self.u = False  #: time-varying particle velocity
        self.u_split_field = False  #: compressional and shear components of time-varying particle velocity
        self.u_non_staggered = False  #: time-varying particle velocity on non-staggered grid
        self.u_max = False  #: maximum particle velocity over simulation
        self.u_min = False  #: minimum particle velocity over simulation
        self.u_rms = False  #: root-mean-squared particle velocity over simulation
        self.u_max_all = False  #: maximum particle velocity over simulation at all grid points
        self.u_min_all = False  #: minimum particle velocity over simulation at all grid points
        self.u_final = False  #: final particle velocity field at all grid points
        self.I = False  #: time-varying acoustic intensity
        self.I_avg = False  #: time-averaged acoustic intensity

        self.cuboid_corners_list = None

        self.x1_inside, self.x2_inside = None, None
        self.y1_inside, self.y2_inside = None, None
        self.z1_inside, self.z2_inside = None, None

    def set_flags_from_list(self, flags_list: List[str], is_elastic_code: bool) -> None:
        """
        Set Recorder flags that are present in the string list to True

        Args:
            flags_list: String list of flags that should be set to True
            is_elastic_code: Is the simulation elastic

        Returns:
            None
        """
        # check the contents of the cell array are valid inputs
        allowed_flags = self.get_allowed_flags(is_elastic_code)
        for record_element in flags_list:
            assert record_element in allowed_flags, f"{record_element} is not a valid input for sensor.record"

            if record_element == "p":  # custom logic for 'p'
                continue
            else:
                setattr(self, record_element, True)

        # set self.record_p to false if a user input for sensor.record
        # is given and 'p' is not set (default is true)
        self.p = "p" in flags_list

    def set_index_variables(self, kgrid: kWaveGrid, pml_size: Vector, is_pml_inside: bool, is_axisymmetric: bool) -> None:
        """
        Assign the index variables

        Args:
            kgrid: kWaveGrid instance
            pml_size: Size of the PML
            is_pml_inside: Whether the PML is inside the grid defined by the user
            is_axisymmetric: Whether the simulation is axisymmetric

        Returns:
            None
        """
        if not is_pml_inside:
            self.x1_inside = pml_size.x + 1.0
            self.x2_inside = kgrid.Nx - pml_size.x
            if kgrid.dim == 2:
                if is_axisymmetric:
                    self.y1_inside = 1
                else:
                    self.y1_inside = pml_size.y + 1.0
                self.y2_inside = kgrid.Ny - pml_size.y
            elif kgrid.dim == 3:
                self.y1_inside = pml_size.y + 1.0
                self.y2_inside = kgrid.Ny - pml_size.y
                self.z1_inside = pml_size.z + 1.0
                self.z2_inside = kgrid.Nz - pml_size.z
        else:
            self.x1_inside = 1.0
            self.x2_inside = kgrid.Nx
            if kgrid.dim == 2:
                self.y1_inside = 1.0
                self.y2_inside = kgrid.Ny
            if kgrid.dim == 3:
                self.z1_inside = 1.0
                self.z2_inside = kgrid.Nz

    @staticmethod
    def get_allowed_flags(is_elastic_code):
        """
        Get the list of allowed flags for a given simulation type

        Args:
            is_elastic_code: Whether the simulation is axisymmetric

        Returns:
            List of allowed flags for a given simulation type
        """
        allowed_flags = [
            "p",
            "p_max",
            "p_min",
            "p_rms",
            "p_max_all",
            "p_min_all",
            "p_final",
            "u",
            "u_max",
            "u_min",
            "u_rms",
            "u_max_all",
            "u_min_all",
            "u_final",
            "u_non_staggered",
            "I",
            "I_avg",
        ]
        if is_elastic_code:  # pragma: no cover
            allowed_flags += ["u_split_field"]
        return allowed_flags

    def is_set(self, attrs: List[str]) -> List[bool]:
        """
        Check if the attributes are set

        Args:
            attrs: Attributes to check

        Returns:
            List of individual boolean results
        """
        return [getattr(self, a) for a in attrs]
