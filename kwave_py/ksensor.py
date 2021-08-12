from dataclasses import dataclass
import numpy as np

from kwave_py.utils import expand_matrix


class kSensor(object):

    def __init__(self, mask=None, record=None):
        self._mask = mask                   #: binary matrix or a set of Cartesian points where the pressure is recorded at each time-step
        self.record = record                #: cell array of the acoustic parameters to record in the form Recorder
        # record the time series from the beginning by default
        self._record_start_index = 1        #: time index at which the sensor should start recording the data specified by sensor.record
        self.record_mode = None
        self.directivity = None
        self.time_reversal_boundary_data = None     #: time varying pressure enforced as a Dirichlet boundary condition over sensor.mask
        self.frequency_response = None              #: two element array specifying the center frequency and percentage bandwidth of a frequency domain Gaussian filter applied to the sensor_data

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val

    def expand_grid(self, expand_size):
        self.mask = expand_matrix(self.mask, expand_size, 0)

    @property
    def record_start_index(self):
        return self._record_start_index

    @record_start_index.setter
    def record_start_index(self, val):
        # force the user index to be an integer
        self._record_start_index = int(round(val))


@dataclass
class kSensorDirectivity(object):
    angle               : np.ndarray    = None
    pattern             : str           = 'pressure'
    size                : float         = None
    unique_angles       : np.ndarray    = None
    wavenumbers         : np.ndarray    = None

    def set_default_size(self, kgrid):
        DEFAULT_SIZE = 10
        self.size = DEFAULT_SIZE * max(kgrid.dx, kgrid.dy)

    def set_unique_angles(self, sensor_mask):
        self.unique_angles = np.unique(self.angle[sensor_mask == 1])

    def set_wavenumbers(self, kgrid):
        self.wavenumbers = np.vstack([kgrid.ky.T, kgrid.kx.T])
