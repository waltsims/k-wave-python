import numpy as np
import logging

from kwave.kgrid import kWaveGrid
from kwave.ksensor import kSensor
from kwave.utils.checks import is_number
from kwave.utils.data import get_smallest_possible_type
from kwave.utils.matlab import matlab_find, matlab_mask, unflatten_matlab_mask
from kwave.utils.matrix import expand_matrix
from kwave.utils.signals import get_win


# force value to be a positive integer
def make_pos_int(val):
    return np.abs(val).astype(int)


class kWaveTransducerSimple(object):
    def __init__(
        self,
        kgrid: kWaveGrid,
        number_elements=128,
        element_width=1,
        element_length=20,
        element_spacing=0,
        position=None,
        radius=float("inf"),
    ):
        """
        Args:
            kgrid: kWaveGrid object
            number_elements: the total number of transducer elements
            element_width: the width of each element in grid points
            element_length: the length of each element in grid points
            element_spacing: the spacing (kerf width) between the transducer elements in grid points
            position: the position of the corner of the transducer in the grid
            radius: the radius of curvature of the transducer [m]

        """

        # allocate the grid size and spacing
        self.stored_grid_size = [kgrid.Nx, kgrid.Ny, kgrid.Nz]  # size of the grid in which the transducer is defined
        self.grid_spacing = [kgrid.dx, kgrid.dy, kgrid.dz]  # corresponding grid spacing

        self.number_elements = make_pos_int(number_elements)
        self.element_width = make_pos_int(element_width)
        self.element_length = make_pos_int(element_length)
        self.element_spacing = make_pos_int(element_spacing)

        if position is None:
            position = [1, 1, 1]
        self.position = make_pos_int(position)

        assert np.isinf(radius), "Only a value of transducer.radius = inf is currently supported"
        self.radius = radius

        # check the transducer fits into the grid
        if np.sum(self.position == 0):
            raise ValueError("The defined transducer must be positioned within the grid")
        elif (
            self.position[1] + self.number_elements * self.element_width + (self.number_elements - 1) * self.element_spacing
        ) > self.stored_grid_size[1]:
            raise ValueError("The defined transducer is too large or positioned outside the grid in the y-direction")
        elif (self.position[2] + self.element_length) > self.stored_grid_size[2]:
            logging.log(logging.INFO, self.position[2])
            logging.log(logging.INFO, self.element_length)
            logging.log(logging.INFO, self.stored_grid_size[2])
            raise ValueError("The defined transducer is too large or positioned outside the grid in the z-direction")
        elif self.position[0] > self.stored_grid_size[0]:
            raise ValueError("The defined transducer is positioned outside the grid in the x-direction")

    @property
    def element_pitch(self):
        return (self.element_spacing + self.element_width) * self.grid_spacing[1]

    @property
    def transducer_width(self):
        """

        Total width of the transducer in grid points

        Returns:
            the overall length of the transducer

        """
        return self.number_elements * self.element_width + (self.number_elements - 1) * self.element_spacing


class NotATransducer(kSensor):
    def __init__(
        self,
        transducer: kWaveTransducerSimple,
        kgrid: kWaveGrid,
        active_elements=None,
        focus_distance=float("inf"),
        elevation_focus_distance=float("inf"),
        receive_apodization="Rectangular",
        transmit_apodization="Rectangular",
        sound_speed=1540,
        input_signal=None,
        steering_angle_max=None,
        steering_angle=None,
    ):
        """
        'time_reversal_boundary_data' and 'record' fields should not be defined
        for the objects of this class

        Args:
            kgrid: kWaveGrid object
            active_elements: the transducer elements that are currently active elements
            elevation_focus_distance: the focus depth in the elevation direction [m]
            receive_apodization: transmit apodization
            transmit_apodization: receive apodization
            sound_speed: sound speed used to calculate beamforming delays [m/s]
            focus_distance: focus distance used to calculate beamforming delays [m]
            input_signal:
            steering_angle_max: max steering angle [deg]
            steering_angle: steering angle [deg]

        """

        super().__init__()
        assert isinstance(transducer, kWaveTransducerSimple)
        self.transducer = transducer
        # time index to start recording if transducer is used as a sensor
        self.record_start_index = 1
        # stored value of appended_zeros (accessed using get and set methods).
        # This is used to set the number of zeros that are appended and prepended to the input signal.
        self.stored_appended_zeros = "auto"
        # stored value of the minimum beamforming delay.
        # This is used to offset the delay mask so that all the delays are >= 0
        self.stored_beamforming_delays_offset = "auto"
        # stored value of the steering_angle_max (accessed using get and set methods).
        # This can be set by the user and is used to derive the two parameters above.
        self.stored_steering_angle_max = "auto"
        # stored value of the steering_angle (accessed using get and set methods)
        self.stored_steering_angle = 0

        ####################################
        # if the sensor is a transducer, check that the simulation is in 3D
        assert kgrid.dim == 3, "Transducer inputs are only compatible with 3D simulations."

        ####################################

        # allocate the temporal spacing
        if is_number(kgrid.dt):
            self.dt = kgrid.dt
        elif kgrid.t_array is not None:
            self.dt = kgrid.t_array[1] - kgrid.t_array[0]
        else:
            raise ValueError("kgrid.dt or kgrid.t_array must be explicitly defined")

        if active_elements is None:
            active_elements = np.ones((transducer.number_elements, 1))
        self.active_elements = active_elements

        self.elevation_focus_distance = elevation_focus_distance

        # check the length of the input
        assert (
            not is_number(receive_apodization) or len(receive_apodization) == self.number_active_elements
        ), "The length of the receive apodization input must match the number of active elements"
        self.receive_apodization = receive_apodization

        # check the length of the input
        assert (
            not is_number(transmit_apodization) or len(transmit_apodization) == self.number_active_elements
        ), "The length of the transmit apodization input must match the number of active elements"
        self.transmit_apodization = transmit_apodization

        # check to see the sound_speed is positive
        assert sound_speed > 0, "transducer.sound_speed must be greater than 0"
        self.sound_speed = sound_speed

        self.focus_distance = focus_distance

        if input_signal is not None:
            input_signal = np.squeeze(input_signal)
            assert input_signal.ndim == 1, "transducer.input_signal must be a one-dimensional array"
            self.stored_input_signal = np.atleast_2d(input_signal).T  # force the input signal to be a column vector

        if steering_angle_max is not None:
            # set the maximum steering angle using the set method (this avoids having to duplicate error checking)
            self.steering_angle_max = steering_angle_max

        if steering_angle is not None:
            # set the maximum steering angle using the set method (this
            # avoids having to duplicate error checking)
            self.steering_angle = steering_angle

        # assign the data type for the transducer matrix based on the
        # number of different elements (uint8 supports 255 numbers so
        # most of the time this data type will be used)
        mask_type = get_smallest_possible_type(transducer.number_elements, "uint")

        # create an empty transducer mask (the grid points within
        # element 'n' are all given the value 'n')
        assert transducer.stored_grid_size is not None
        self.indexed_mask = np.zeros(transducer.stored_grid_size, dtype=mask_type)

        # create a second empty mask used for the elevation beamforming
        # delays (the grid points across each element are numbered 1 to
        # M, where M is the number of grid points in the elevation
        # direction)
        self.indexed_element_voxel_mask = np.zeros(transducer.stored_grid_size, dtype=mask_type)

        # create the corresponding indexing variable 1:M
        element_voxel_index = np.tile(np.arange(transducer.element_length) + 1, (transducer.element_width, 1))

        # for each transducer element, calculate the grid point indices
        for element_index in range(0, transducer.number_elements):
            # assign the current transducer position
            element_pos_x = transducer.position[0]
            element_pos_y = transducer.position[1] + (transducer.element_width + transducer.element_spacing) * element_index
            element_pos_z = transducer.position[2]

            element_pos_x = element_pos_x - 1
            element_pos_y = element_pos_y - 1
            element_pos_z = element_pos_z - 1

            # assign the grid points within the current element the
            # index of the element
            self.indexed_mask[
                element_pos_x,
                element_pos_y : element_pos_y + transducer.element_width,
                element_pos_z : element_pos_z + transducer.element_length,
            ] = element_index + 1

            # assign the individual grid points an index corresponding
            # to their order across the element
            self.indexed_element_voxel_mask[
                element_pos_x,
                element_pos_y : element_pos_y + transducer.element_width,
                element_pos_z : element_pos_z + transducer.element_length,
            ] = element_voxel_index

        # double check the transducer fits within the desired grid size
        assert (
            np.array(self.indexed_mask.shape) == transducer.stored_grid_size
        ).all(), "Desired transducer is larger than the input grid_size"

        self.sxx = self.syy = self.szz = self.sxy = self.sxz = self.syz = None
        self.u_mode = self.p_mode = None
        self.ux = self.uy = self.uz = None

    @staticmethod
    def isfield(_):
        # return field_name in dir(self)
        # return eng.isfield(self.transducer, field_name)
        return False  # this method call was returning false always because Matlab 'isfield' calls are false for classes

    def __contains__(self, item):
        return self.isfield(item)

    @property
    def beamforming_delays(self):
        """
        calculate the beamforming delays based on the focus and steering settings

        """
        # calculate the element pitch in [m]
        element_pitch = self.transducer.element_pitch

        # create indexing variable
        element_index = np.arange(-(self.number_active_elements - 1) / 2, (self.number_active_elements + 1) / 2)

        # check for focus depth
        if np.isinf(self.focus_distance):
            # calculate time delays for a steered beam
            delay_times = element_pitch * element_index * np.sin(self.steering_angle * np.pi / 180) / self.sound_speed  # [s]

        else:
            # calculate time delays for a steered and focussed beam
            delay_times = (
                self.focus_distance
                / self.sound_speed
                * (
                    1
                    - np.sqrt(
                        1
                        + (element_index * element_pitch / self.focus_distance) ** 2
                        - 2 * (element_index * element_pitch / self.focus_distance) * np.sin(self.steering_angle * np.pi / 180)
                    )
                )
            )  # [s]

        # convert the delays to be in units of time points
        delay_times = (delay_times / self.dt).round().astype(int)
        return delay_times

    @property
    def beamforming_delays_offset(self):
        """
        Offset used to make all the delays in the delay_mask positive (either set to 'auto' or based on the setting for steering_angle_max)

        Returns:
            the stored value of the offset used to force the values in delay_mask to be >= 0

        """

        return self.stored_beamforming_delays_offset

    @property
    def mask(self):
        """
        Allow mask query to allow compatability with regular sensor structure - return the active sensor mask

        """

        return self.active_elements_mask

    @property
    def indexed_active_elements_mask(self):
        # copy the indexed elements mask
        mask = self.indexed_mask
        if mask is None:
            return None

        mask = np.copy(mask)

        # remove inactive elements from the mask
        for element_index in range(self.transducer.number_elements):
            if not self.active_elements[element_index]:
                mask[mask == (element_index + 1)] = 0  # +1 compatibility

        # force the lowest element index to be 1
        lowest_active_element_index = matlab_find(self.active_elements)[0][0]
        mask[mask != 0] = mask[mask != 0] - lowest_active_element_index + 1
        return mask

    @property
    def indexed_elements_mask(self):  # nr
        return self.indexed_mask

    @property
    def steering_angle(self):  # nr
        return self.stored_steering_angle

    # set the stored value of the steering angle
    @steering_angle.setter
    def steering_angle(self, steering_angle):
        # force to be scalar
        steering_angle = float(steering_angle)

        # check if the steering angle is between -90 and 90
        assert -90 < steering_angle < 90, "Input for steering_angle must be betweeb -90 and 90 degrees."

        # check if the steering angle is less than the maximum steering angle
        if self.stored_steering_angle_max != "auto" and (abs(steering_angle) > self.stored_steering_angle_max):
            raise ValueError("Input for steering_angle cannot be greater than steering_angle_max.")

        # update the stored value
        self.stored_steering_angle = steering_angle

    @property
    def steering_angle_max(self):
        return self.stored_steering_angle_max

    @steering_angle_max.setter
    def steering_angle_max(self, steering_angle_max):
        # force input to be scalar and positive
        steering_angle_max = float(steering_angle_max)

        # check the steering angle is within range
        assert -90 < steering_angle_max < 90, "Input for steering_angle_max must be between -90 and 90."

        # check the maximum steering angle is greater than the current steering angle
        assert (
            self.stored_steering_angle_max == "auto" or abs(self.stored_steering_angle) <= steering_angle_max
        ), "Input for steering_angle_max cannot be less than the current steering_angle."

        # overwrite the stored value
        self.stored_steering_angle_max = steering_angle_max

        # store a copy of the current value for the steering angle
        current_steering_angle = self.stored_steering_angle

        # overwrite with the user defined maximum value
        self.stored_steering_angle = steering_angle_max

        # set the beamforming delay offset to zero (this means the  delays will contain negative
        # values which we can use to derive the new values for the offset)
        self.stored_appended_zeros = 0
        self.stored_beamforming_delays_offset = 0

        # get the element beamforming delays and reverse
        delay_times = -self.beamforming_delays

        # get the maximum and minimum beamforming delays
        min_delay, max_delay = delay_times.min(), delay_times.max()

        # add the maximum and minimum elevation delay if the elevation focus is not set to infinite
        if not np.isinf(self.elevation_focus_distance):
            max_delay = max_delay + max(self.elevation_beamforming_delays)
            min_delay = min_delay + min(self.elevation_beamforming_delays)

        # set the beamforming offset to the difference between the
        # maximum and minimum delay
        self.stored_appended_zeros = max_delay - min_delay

        # set the minimum beamforming delay (converting to a positive number)
        self.stored_beamforming_delays_offset = -min_delay

        # reset the previous value of the steering angle
        self.stored_steering_angle = current_steering_angle

    @property
    def elevation_beamforming_mask(self):  # nr
        # get elevation beamforming mask
        delay_mask = self.delay_mask(2)

        # extract the active elements
        delay_mask = delay_mask[self.active_elements_mask != 0]

        # force delays to start from zero
        delay_mask = delay_mask - delay_mask.min()

        # create an empty output mask
        mask = np.zeros((delay_mask.size, delay_mask.max() + 1))

        # populate the mask by setting 1's at the index given by the delay time
        for index in range(delay_mask.size):
            mask[index, delay_mask[index]] = 1

        # flip the mask so the shortest delays are at the right
        return np.fliplr(mask)

    @property
    def input_signal(self):
        signal = self.stored_input_signal

        # check the signal is not empty
        assert signal is not None, "Transducer input signal is not defined"

        # automatically prepend and append zeros if the beamforming
        # delay offset is set

        # check if the beamforming delay offset is set. If so, use this
        # number to prepend and append this number of zeros to the
        # input signal. Otherwise, calculate how many zeros are needed
        # and prepend and append these.
        stored_appended_zeros = self.stored_appended_zeros
        if stored_appended_zeros != "auto":
            # use the current value of the beamforming offset to add
            # zeros to the input signal
            signal = np.vstack([np.zeros((stored_appended_zeros, 1)), signal, np.zeros((stored_appended_zeros, 1))])

        else:
            # get the current delay beam forming
            delay_mask = self.delay_mask()

            # find the maximum delay
            delay_max = delay_mask.max()

            # count the number of leading zeros in the input signal
            leading_zeros = matlab_find(signal)[0, 0] - 1

            # count the number of trailing zeros in the input signal
            trailing_zeros = matlab_find(np.flipud(signal))[0, 0] - 1

            # check the number of leading zeros is sufficient given the
            # maximum delay
            if leading_zeros < delay_max + 1:
                logging.log(logging.INFO, f"  prepending transducer.input_signal with {delay_max - leading_zeros + 1} leading zeros")

                # prepend extra leading zeros
                signal = np.vstack([np.zeros((delay_max - leading_zeros + 1, 1)), signal])

            # check the number of leading zeros is sufficient given the
            # maximum delay
            if trailing_zeros < delay_max + 1:
                logging.log(logging.INFO, f"  appending transducer.input_signal with {delay_max - trailing_zeros + 1} trailing zeros")

                # append extra trailing zeros
                signal = np.vstack([signal, np.zeros((delay_max - trailing_zeros + 1, 1))])

        return signal

    @property
    def number_active_elements(self):
        return int(self.active_elements.sum())

    @property
    def appended_zeros(self):
        """
        Number of zeros appended to input signal to allow a single time series to be used
        within kspaceFirstOrder3D (either set to 'auto' or based on the setting for steering_angle_max)

        """

        return self.stored_appended_zeros

    @property
    def grid_size(self):
        """
        Returns:
             grid size

        """
        return self.transducer.stored_grid_size

    @property
    def active_elements_mask(self):
        """
        Returns:
            A binary mask showing the locations of the active elements

        """
        indexed_mask = np.copy(self.indexed_mask)
        active_elements = self.active_elements.squeeze()
        number_elements = int(self.transducer.number_elements)

        # copy the indexed elements mask
        mask = indexed_mask

        # remove inactive elements from the mask
        for element_index in range(1, number_elements + 1):
            mask[mask == element_index] = active_elements[element_index - 1]

        # convert remaining mask to binary
        mask[mask != 0] = 1

        return mask

    @property
    def all_elements_mask(self):
        """
        Returns:
            A binary mask showing the locations of all the elements (both active and inactive)

        """

        mask = np.copy(self.indexed_mask)
        mask[mask != 0] = 1
        return mask

    def expand_grid(self, expand_size):
        self.indexed_mask = expand_matrix(self.indexed_mask, expand_size, 0)

    def retract_grid(self, retract_size):
        indexed_mask = self.indexed_mask
        retract_size = np.array(retract_size[0]).astype(np.int_)

        self.indexed_mask = indexed_mask[
            retract_size[0] : -retract_size[0], retract_size[1] : -retract_size[1], retract_size[2] : -retract_size[2]
        ]

    @property
    def transmit_apodization_mask(self):
        """
        convert the transmit wave apodization into the form of a element mask,
        where the apodization values are placed at the grid points
        belonging to the active transducer elements. These values are
        then extracted in the correct order within
        kspaceFirstOrder_inputChecking using apodization =
        transmit_apodization_mask(active_elements_mask ~= 0)

        """

        # get transmit apodization
        apodization = self.get_transmit_apodization()

        # create an empty mask;
        mask = np.zeros(self.transducer.stored_grid_size)

        # assign the apodization values to every grid point in the transducer
        mask_index = self.indexed_active_elements_mask
        mask_index = mask_index[mask_index != 0]
        mask[self.active_elements_mask == 1] = apodization[mask_index - 1, 0]  # -1 for conversion
        return mask

    def get_transmit_apodization(self):
        """
        Returns:
            return the transmit apodization, converting strings of window
            type to actual numbers using getWin

        """

        # check if a user defined apodization is given and whether this
        # is still the correct size (in case the number of active
        # elements has changed)
        if is_number(self.transmit_apodization):
            assert (
                self.transmit_apodization.size == self.number_active_elements
            ), "The length of the transmit apodization input must match the number of active elements"

            # assign apodization
            apodization = self.transmit_apodization
        else:
            # if the number of active elements is greater than 1,
            # create apodization using getWin, otherwise, assign 1
            if self.number_active_elements > 1:
                apodization, _ = get_win(int(self.number_active_elements), type_=self.transmit_apodization)
            else:
                apodization = 1
        apodization = np.array(apodization)
        return apodization

    def delay_mask(self, mode=None):
        """
        mode == 1: both delays
        mode == 2: elevation only
        mode == 3: azimuth only

        """
        # assign the delays to a new mask using the indexed_element_mask
        indexed_active_elements_mask_copy = self.indexed_active_elements_mask
        mask = np.zeros(self.transducer.stored_grid_size, dtype=np.float32)

        if indexed_active_elements_mask_copy is None:
            return mask

        active_elements_index = matlab_find(indexed_active_elements_mask_copy)

        # calculate azimuth focus delay times provided they are not all zero
        if (not np.isinf(self.focus_distance) or (self.steering_angle != 0)) and (mode is None or mode != 2):
            # get the element beamforming delays and reverse
            delay_times = -self.beamforming_delays

            # add delay times
            # mask[active_elements_index] = delay_times[indexed_active_elements_mask_copy[active_elements_index]]
            mask[unflatten_matlab_mask(mask, active_elements_index, diff=-1)] = matlab_mask(
                delay_times, matlab_mask(indexed_active_elements_mask_copy, active_elements_index, diff=-1), diff=-1
            ).squeeze()

        # calculate elevation focus time delays provided each element is longer than one grid point
        if not np.isinf(self.elevation_focus_distance) and (self.transducer.element_length > 1) and (mode is None or mode != 3):
            # get elevation beamforming delays
            elevation_delay_times = self.elevation_beamforming_delays

            # get current mask
            element_voxel_mask = self.indexed_element_voxel_mask

            # add delay times
            mask[unflatten_matlab_mask(mask, active_elements_index - 1)] += matlab_mask(
                elevation_delay_times, matlab_mask(element_voxel_mask, active_elements_index - 1) - 1
            )[:, 0]  # -1s compatibility

        # shift delay times (these should all be >= 0, where a value of 0 means no delay)
        if self.stored_appended_zeros == "auto":
            mask[unflatten_matlab_mask(mask, active_elements_index - 1)] -= mask[
                unflatten_matlab_mask(mask, active_elements_index - 1)
            ].min()  # -1s compatibility
        else:
            mask[unflatten_matlab_mask(mask, active_elements_index - 1)] += self.stored_beamforming_delays_offset  # -1s compatibility
        return mask.astype(np.uint8)

    @property
    def elevation_beamforming_delays(self):
        """
        Calculate the elevation beamforming delays based on the focus setting

        """
        if not np.isinf(self.elevation_focus_distance):
            # create indexing variable

            element_index = np.arange(-(self.transducer.element_length - 1) / 2, (self.transducer.element_length + 1) / 2)

            # calculate time delays for a focussed beam
            delay_times = self.elevation_focus_distance - np.sqrt(
                (element_index * self.transducer.grid_spacing[2]) ** 2 + self.elevation_focus_distance**2
            )
            delay_times /= self.sound_speed

            # convert the delays to be in units of time points and then reverse
            delay_times = -np.round(delay_times / self.dt).astype(np.int32)

        else:
            # create an empty array
            delay_times = np.zeros((1, self.transducer.element_length))
        return delay_times

    def get_receive_apodization(self):
        """
        Get the current receive apodization setting.
        """
        # Example implementation, adjust based on actual logic
        if is_number(self.receive_apodization):
            assert (
                self.receive_apodization.size == self.number_active_elements
            ), "The length of the receive apodization input must match the number of active elements"
            return self.receive_apodization
        else:
            if self.number_active_elements > 1:
                apodization, _ = get_win(int(self.number_active_elements), type_=self.receive_apodization)
            else:
                apodization = 1
        return np.array(apodization)

    def scan_line(self, sensor_data):
        """
        Apply beamforming and apodization to the sensor data.
        """
        # Get the current apodization setting
        apodization = self.get_receive_apodization()

        # Get the current beamforming weights and reverse
        delays = -self.beamforming_delays

        # Offset the received sensor_data by the beamforming delays and apply receive apodization
        for element_index in range(self.number_active_elements):
            if delays[element_index] > 0:
                # Shift element data forwards
                sensor_data[element_index, :] = (
                    np.pad(sensor_data[element_index, delays[element_index] :], (0, delays[element_index]), "constant")
                    * apodization[element_index]
                )
            elif delays[element_index] < 0:
                # Shift element data backwards
                sensor_data[element_index, :] = (
                    np.pad(
                        sensor_data[element_index, : sensor_data.shape[1] + delays[element_index]], (-delays[element_index], 0), "constant"
                    )
                    * apodization[element_index]
                )

        # Form the line summing across the elements
        line = np.sum(sensor_data, axis=0)
        return line

    def combine_sensor_data(self, sensor_data):
        # check the data is the correct size
        if sensor_data.shape[0] != (self.number_active_elements * self.transducer.element_width * self.transducer.element_length):
            raise ValueError(
                "The number of time series in the input sensor_data must "
                "match the number of grid points in the active tranducer elements."
            )

        # get index of which element each time series belongs to
        # Tricky things going on here
        ind = self.indexed_active_elements_mask[0].T[self.indexed_active_elements_mask[0].T > 0]

        # create empty output
        sensor_data_sum = np.zeros((int(self.number_active_elements), sensor_data.shape[1]))

        # check if an elevation focus is set
        if np.isinf(self.elevation_focus_distance):
            raise NotImplementedError

            # # loop over time series and sum
            # for ii = 1:length(ind)
            #     sensor_data_sum(ind(ii), :) = sensor_data_sum(ind(ii), :) + sensor_data(ii, :);
            # end

        else:
            # get the elevation delay for each grid point of each
            # active transducer element (this is given in units of grid
            # points)
            dm = self.delay_mask(2)
            # dm = dm[self.active_elements_mask != 0]
            dm = dm[0].T[self.active_elements_mask[0].T != 0]
            dm = dm.astype(np.int32)

            # loop over time series, shift and sum
            for ii in range(len(ind)):
                # FARID: something nasty can be here
                end = -dm[ii] if dm[ii] != 0 else sensor_data_sum.shape[-1]
                sensor_data_sum[ind[ii] - 1, 0:end] = sensor_data_sum[ind[ii] - 1, 0:end] + sensor_data[ii, dm[ii] :]

        # divide by number of time series in each element
        sensor_data_sum = sensor_data_sum * (1 / (self.transducer.element_width * self.transducer.element_length))
        return sensor_data_sum
