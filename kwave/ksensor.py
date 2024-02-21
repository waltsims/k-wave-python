from dataclasses import dataclass

import numpy as np

from kwave.utils.matrix import expand_matrix


class kSensor(object):
    def __init__(self, mask=None, record=None):
        self._mask = mask
        # cell array of the acoustic parameters to record in the form Recorder
        self.record = record
        # record the time series from the beginning by default
        # time index at which the sensor should start recording the data specified by sensor.record
        self._record_start_index = 1

        # Directivity of the individiual sensor points
        self.directivity = None

        # time varying pressure enforced as a Dirichlet boundary condition over sensor.mask
        self.time_reversal_boundary_data = None
        # two element array specifying the center frequency and percentage bandwidth
        # of a frequency domain Gaussian filter applied to the sensor_data
        self.frequency_response = None

    @property
    def mask(self):
        """
        Binary matrix or a set of Cartesian points where the pressure is recorded at each time-step
        """
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val

    def expand_grid(self, expand_size) -> None:
        """
        Enlarge the sensor mask (for Cartesian sensor masks and cuboid corners,
        this has already been converted to a binary mask for display in inputChecking)

        Args:
            expand_size: the number of elements to add in each dimension

        Returns:
            None
        """
        self.mask = expand_matrix(self.mask, expand_size, 0)

    @property
    def record_start_index(self):
        """
        Time index to start recording if transducer is used as a sensor
        """
        return self._record_start_index

    @record_start_index.setter
    def record_start_index(self, val):
        # force the user index to be an integer
        self._record_start_index = int(round(val))


@dataclass
class kSensorDirectivity(object):
    #: matrix of directivity angles (direction of maximum
    #: response) for each sensor element defined in
    #: sensor.mask. The angles are in radians where 0 = max
    #: sensitivity in x direction (up/down) and pi/2 or -pi/2
    #: = max sensitivity in y direction (left/right)
    angle: np.ndarray = None

    #: string defining the directivity pattern, valid inputs
    #: are 'pressure' (spatial averaging over the sensor
    #: surface equivalent to a sinc function) and 'gradient'
    pattern: str = "pressure"

    #: equivalent element size (the larger the element size the more directional the response)
    size: float = None

    #: list of the unique directivity angles
    unique_angles: np.ndarray = None

    #: It is precomputed to allow data casting, as kgrid.kx (etc) are computed on the fly.
    wavenumbers: np.ndarray = None

    def set_default_size(self, kgrid) -> None:
        """
        Set the element size based on the kGrid

        Args:
            kgrid: Instance of `~kwave.kgrid.kWaveGrid` class

        Returns:
            None
        """
        DEFAULT_SIZE = 10
        self.size = DEFAULT_SIZE * max(kgrid.dx, kgrid.dy)

    def set_unique_angles(self, sensor_mask) -> None:
        """
        Assign unique_angles from sensor_mask

        Args:
            sensor_mask:

        Returns:
            None
        """
        self.unique_angles = np.unique(self.angle[sensor_mask == 1])

    def set_wavenumbers(self, kgrid) -> None:
        """
        Assign the wavenumber vectors

        Args:
            kgrid: Instance of `~kwave.kgrid.kWaveGrid` class

        Returns:
            None
        """
        self.wavenumbers = np.vstack([kgrid.ky.T, kgrid.kx.T])
