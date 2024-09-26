import logging
from dataclasses import dataclass

import numpy as np

from kwave.data import Vector
from kwave.kWaveSimulation_helper import (
    display_simulation_params,
    set_sound_speed_ref,
    expand_grid_matrices,
    create_absorption_variables,
    scale_source_terms_func,
)
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.ktransducer import NotATransducer
from kwave.options.simulation_options import SimulationOptions, SimulationType
from kwave.recorder import Recorder
from kwave.utils.checks import check_stability
from kwave.utils.colormap import get_color_map
from kwave.utils.conversion import cast_to_type, cart2grid
from kwave.utils.data import get_smallest_possible_type, get_date_string
from kwave.utils.dotdictionary import dotdict
from kwave.utils.filters import smooth
from kwave.utils.matlab import matlab_find, matlab_mask
from kwave.utils.matrix import num_dim2


@dataclass
class kWaveSimulation(object):
    def __init__(
        self, kgrid: kWaveGrid, source: kSource, sensor: NotATransducer, medium: kWaveMedium, simulation_options: SimulationOptions
    ):
        self.precision = None
        self.kgrid = kgrid
        self.medium = medium
        self.source = source
        self.sensor = sensor
        self.options = simulation_options

        # =========================================================================
        # FLAGS WHICH DEPEND ON USER INPUTS (THESE SHOULD NOT BE MODIFIED)
        # =========================================================================
        # flags which control the characteristics of the sensor
        #: Whether time reversal simulation is enabled

        # check if performing time reversal, and replace inputs to explicitly use a
        # source with a dirichlet boundary condition
        if self.sensor.time_reversal_boundary_data is not None:
            # define a new source structure
            source = {"p_mask": self.sensor.p_mask, "p": np.flip(self.sensor.time_reversal_boundary_data, 2), "p_mode": "dirichlet"}

            # define a new sensor structure
            Nx, Ny, Nz = self.kgrid.Nx, self.kgrid.Ny, self.kgrid.Nz
            sensor = kSensor(mask=np.ones((Nx, Ny, max(1, Nz))), record=["p_final"])
            # set time reversal flag
            self.userarg_time_rev = True
        else:
            # set time reversal flag
            self.userarg_time_rev = False

            #: Whether sensor.mask should be re-ordered.
            #: True if sensor.mask is Cartesian with nearest neighbour interpolation which is calculated using a binary mask
            #: and thus must be re-ordered
            self.reorder_data = False

            #: Whether the sensor.mask is binary
            self.binary_sensor_mask = True

            # check if the sensor mask is defined as a list of cuboid corners
            if self.sensor.mask is not None and self.sensor.mask.shape[0] == (2 * self.kgrid.dim):
                self.userarg_cuboid_corners = True
            else:
                self.userarg_cuboid_corners = False

            #: If tse sensor is an object of the kWaveTransducer class
            self.transducer_sensor = False

            self.record = Recorder()

        # transducer source flags
        #: transducer is object of kWaveTransducer class
        self.transducer_source = False

        #: Apply receive elevation focus on the transducer
        self.transducer_receive_elevation_focus = False

        # general
        self.COLOR_MAP = get_color_map()  #: default color map
        self.ESTIMATE_SIM_TIME_STEPS = 50  #: time steps used to estimate simulation time
        self.HIGHEST_PRIME_FACTOR_WARNING = 7  #: largest prime factor before warning
        self.KSPACE_CFL = 0.3  #: default CFL value used if kgrid.t_array is set to 'auto'
        self.PSTD_CFL = 0.1  #: default CFL value used if kgrid.t_array is set to 'auto'

        # source types
        self.SOURCE_S_MODE_DEF = "additive"  #: source mode for stress sources
        self.SOURCE_P_MODE_DEF = "additive"  #: source mode for pressure sources
        self.SOURCE_U_MODE_DEF = "additive"  #: source mode for velocity sources

        # filenames
        self.STREAM_TO_DISK_FILENAME = "temp_sensor_data.bin"  #: default disk stream filename
        self.LOG_NAME = ["k-Wave-Log-", get_date_string()]  #: default log filename

        self.calling_func_name = None
        logging.log(logging.INFO, f"  start time: {get_date_string()}")

        self.c_ref, self.c_ref_compression, self.c_ref_shear = [None] * 3
        self.transducer_input_signal = None

        #: Indexing variable corresponding to the location of all the pressure source elements
        self.p_source_pos_index = None
        #: Indexing variable corresponding to the location of all the velocity source elements
        self.u_source_pos_index = None
        #: Indexing variable corresponding to the location of all the stress source elements
        self.s_source_pos_index = None

        #: Delay mask that accounts for the beamforming delays and elevation focussing
        self.delay_mask = None

        self.absorb_nabla1 = None  #: absorbing fractional Laplacian operator
        self.absorb_tau = None  #: absorbing fractional Laplacian coefficient
        self.absorb_nabla2 = None  #: dispersive fractional Laplacian operator
        self.absorb_eta = None  #: dispersive fractional Laplacian coefficient

        self.dt = None  #: Alias to kgrid.dt
        self.rho0 = None  #: Alias to medium.density
        self.c0 = None  #: Alias to medium.sound_speed
        self.index_data_type = None

    @property
    def equation_of_state(self):
        """
        Returns:
            Set equation of state variable
        """
        if self.medium.absorbing:
            if self.medium.stokes:
                return "stokes"
            else:
                return "absorbing"
        else:
            return "loseless"

    @property
    def use_sensor(self):
        """
        Returns:
            False if no output of any kind is required

        """
        return self.sensor is not None

    @property
    def blank_sensor(self):
        """
        Returns
            True if sensor.mask is not defined but _max_all or _final variables are still recorded

        """
        fields = ["p", "p_max", "p_min", "p_rms", "u", "u_non_staggered", "u_split_field", "u_max", "u_min", "u_rms", "I", "I_avg"]
        if not (isinstance(self.sensor, NotATransducer) or any(self.record.is_set(fields)) or self.time_rev):
            return True
        return False

    @property
    def kelvin_voigt_model(self):
        """
        Returns:
            Whether the simulation is elastic with absorption

        """
        return False

    @property
    def nonuniform_grid(self):
        """
        Returns:
            True if the computational grid is non-uniform

        """
        return self.kgrid.nonuniform

    @property
    def time_rev(self):
        """
        Returns:
            True for time reversal simulaions using sensor.time_reversal_boundary_data

        """
        if self.sensor is not None and not isinstance(self.sensor, NotATransducer):
            if not self.options.simulation_type.is_elastic_simulation() and self.sensor.time_reversal_boundary_data is not None:
                return True
        else:
            return self.userarg_time_rev

    @property
    def elastic_time_rev(self):
        """
        Returns:
            True if using time reversal with the elastic code

        """
        return False

    @property
    def compute_directivity(self):
        """
        Returns:
            True if directivity calculations in 2D are used by setting sensor.directivity_angle

        """
        if self.sensor is not None and not isinstance(self.sensor, NotATransducer):
            if self.kgrid.dim == 2:
                # check for sensor directivity input and set flag
                directivity = self.sensor.directivity
                if directivity is not None and directivity.angle is not None:
                    return True
        return False

    @property
    def cuboid_corners(self):
        """
        Returns:
            Whether the sensor.mask is a list of cuboid corners
        """
        if self.sensor is not None and not isinstance(self.sensor, NotATransducer):
            if not self.blank_sensor and self.sensor.mask.shape[0] == 2 * self.kgrid.dim:
                return True
        return self.userarg_cuboid_corners

    ##############
    # flags which control the types of source used
    ##############
    @property
    def source_p0(self):  # initial pressure
        """
        Returns:
            Whether initial pressure source is present (default=False)

        """
        flag = False  # default
        if not isinstance(self.source, NotATransducer) and self.source.p0 is not None:
            # set flag
            flag = True
        return flag

    @property
    def source_p0_elastic(self):  # initial pressure in the elastic code
        """
        Returns:
            Whether initial pressure source is present in the elastic code (default=False)

        """
        # Not clear where this flag is set
        return False

    @property
    def source_p(self):
        """
        Returns:
            Whether time-varying pressure source is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.p is not None:
            # set source flag to the length of the source, this allows source.p
            # to be shorter than kgrid.Nt
            flag = len(self.source.p[0])
        return flag

    @property
    def source_p_labelled(self):  # time-varying pressure with labelled source mask
        """
        Returns:
            True/False if labelled/binary source mask, respectively.

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.p is not None:
            # check if the mask is binary or labelled
            p_unique = np.unique(self.source.p_mask)
            flag = not (p_unique.size <= 2 and p_unique.sum() == 1)
        return flag

    @property
    def source_ux(self) -> bool:
        """
        Returns:
            Whether time-varying particle velocity source is used in X-direction

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.ux is not None:
            # set source flgs to the length of the sources, this allows the
            # inputs to be defined independently and be of any length
            flag = len(self.source.ux[0])
        return flag

    @property
    def source_uy(self) -> bool:
        """
        Returns:
            Whether time-varying particle velocity source is used in Y-direction

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.uy is not None:
            # set source flgs to the length of the sources, this allows the
            # inputs to be defined independently and be of any length
            flag = len(self.source.uy[0])
        return flag

    @property
    def source_uz(self) -> bool:
        """
        Returns:
            Whether time-varying particle velocity source is used in Z-direction

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.uz is not None:
            # set source flgs to the length of the sources, this allows the
            # inputs to be defined independently and be of any length
            flag = len(self.source.uz[0])
        return flag

    @property
    def source_u_labelled(self):
        """
        Returns:
            Whether time-varying velocity source with labelled source mask is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.u_mask is not None:
            # check if the mask is binary or labelled
            u_unique = np.unique(self.source.u_mask)
            if u_unique.size <= 2 and u_unique.sum() == 1:
                # binary source mask
                flag = False
            else:
                # labelled source mask
                flag = True
        return flag

    @property
    def source_sxx(self):
        """
        Returns:
            Whether time-varying stress source in X->X direction is present (default=False)
        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.sxx is not None:
            flag = len(self.source.sxx[0])
        return flag

    @property
    def source_syy(self):
        """
        Returns:
            Whether time-varying stress source in Y->Y direction is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.syy is not None:
            flag = len(self.source.syy[0])
        return flag

    @property
    def source_szz(self):
        """
        Returns:
            Whether time-varying stress source in Z->Z direction is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.szz is not None:
            flag = len(self.source.szz[0])
        return flag

    @property
    def source_sxy(self):
        """
        Returns:
            Whether time-varying stress source in X->Y direction is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.sxy is not None:
            flag = len(self.source.sxy[0])
        return flag

    @property
    def source_sxz(self):
        """
        Returns:
            Whether time-varying stress source in X->Z direction is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.sxz is not None:
            flag = len(self.source.sxz[0])
        return flag

    @property
    def source_syz(self):
        """
        Returns:
            Whether time-varying stress source in Y->Z direction is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.syz is not None:
            flag = len(self.source.syz[0])
        return flag

    @property
    def source_s_labelled(self):
        """
        Returns:
            Whether time-varying stress source with labelled source mask is present (default=False)

        """
        flag = False
        if not isinstance(self.source, NotATransducer) and self.source.s_mask is not None:
            # check if the mask is binary or labelled
            s_unique = np.unique(self.source.s_mask)
            if s_unique.size <= 2 and s_unique.sum() == 1:
                # binary source mask
                flag = False
            else:
                # labelled source mask
                flag = True
        return flag

    @property
    def use_w_source_correction_p(self):
        """
        Returns:
            Whether to use the w source correction instead of the k-space source correction for pressure sources
        """
        flag = False
        if not isinstance(self.source, NotATransducer):
            if self.source.p is not None and self.source.p_frequency_ref is not None:
                flag = True
        return flag

    @property
    def use_w_source_correction_u(self):
        """
        Returns:
            Whether to use the w source correction instead of the k-space source correction for velocity sources
        """
        flag = False
        if not isinstance(self.source, NotATransducer):
            if any([(getattr(self.source, k) is not None) for k in ["ux", "uy", "uz", "u_mask"]]):
                if self.source.u_frequency_ref is not None:
                    flag = True
        return flag

    def input_checking(self, calling_func_name) -> None:
        """
        Check the input fields for correctness and validness

        Args:
            calling_func_name: Name of the script that calls this function

        Returns:
            None
        """
        self.calling_func_name = calling_func_name

        k_dim = self.kgrid.dim

        self.check_calling_func_name_and_dim(calling_func_name, k_dim)

        # run subscript to check optional inputs
        self.options = SimulationOptions.option_factory(self.kgrid, self.options)
        opt = self.options

        # TODO(Walter): clean this up with getters in simulation options pml size
        pml_x_size, pml_y_size, pml_z_size = opt.pml_x_size, opt.pml_y_size, opt.pml_z_size
        pml_size = Vector([pml_x_size, pml_y_size, pml_z_size])

        is_elastic_code = opt.simulation_type.is_elastic_simulation()
        self.print_start_status(is_elastic_code=is_elastic_code)
        self.set_index_data_type()

        user_medium_density_input = self.check_medium(self.medium, self.kgrid.k, simulation_type=opt.simulation_type)

        # select the reference sound speed used in the k-space operator
        self.c_ref, self.c_ref_compression, self.c_ref_shear = set_sound_speed_ref(self.medium, opt.simulation_type)

        self.check_source(k_dim, self.kgrid.Nt)
        self.check_sensor(k_dim)
        self.check_kgrid_time()
        self.precision = self.select_precision(opt)
        self.check_input_combinations(opt, user_medium_density_input, k_dim, pml_size, self.kgrid.N)

        # run subscript to display time step, max supported frequency etc.
        display_simulation_params(self.kgrid, self.medium, is_elastic_code)

        self.smooth_and_enlarge(self.source, k_dim, Vector(self.kgrid.N), opt)
        self.create_sensor_variables()
        self.create_absorption_vars()
        self.assign_pseudonyms(self.medium, self.kgrid)
        self.scale_source_terms(opt.scale_source_terms)
        self.create_pml_indices(
            kgrid_dim=self.kgrid.dim,
            kgrid_N=Vector(self.kgrid.N),
            pml_size=pml_size,
            pml_inside=opt.pml_inside,
            is_axisymmetric=opt.simulation_type.is_axisymmetric(),
        )

    @staticmethod
    def check_calling_func_name_and_dim(calling_func_name, kgrid_dim) -> None:
        """
        Check correct function has been called for the dimensionality of kgrid

        Args:
            calling_func_name: Name of the script that makes calls to kWaveSimulation
            kgrid_dim: Dimensionality of the kWaveGrid

        Returns:
            None
        """
        assert not calling_func_name.startswith(("pstdElastic", "kspaceElastic")), "Elastic simulation is not supported."

        if calling_func_name == "kspaceFirstOrder1D":
            assert kgrid_dim == 1, f"kgrid has the wrong dimensionality for {calling_func_name}."
        elif calling_func_name in ["kspaceFirstOrder2D", "pstdElastic2D", "kspaceElastic2D", "kspaceFirstOrderAS"]:
            assert kgrid_dim == 2, f"kgrid has the wrong dimensionality for {calling_func_name}."
        elif calling_func_name in ["kspaceFirstOrder3D", "pstdElastic3D", "kspaceElastic3D"]:
            assert kgrid_dim == 3, f"kgrid has the wrong dimensionality for {calling_func_name}."

    @staticmethod
    def print_start_status(is_elastic_code: bool) -> None:
        """
        Update command-line status with the start time

        Args:
            is_elastic_code: is the simulation elastic

        Returns:
            None
        """
        if is_elastic_code:  # pragma: no cover
            logging.log(logging.INFO, "Running k-Wave elastic simulation...")
        else:
            logging.log(logging.INFO, "Running k-Wave simulation...")
        logging.log(logging.INFO, f"  start time: {get_date_string()}")

    def set_index_data_type(self) -> None:
        """
        Pre-calculate the data type needed to store the matrix indices given the
        total number of grid points: indexing variables will be created using this data type to save memory

        Returns:
            None
        """
        total_grid_points = self.kgrid.total_grid_points
        self.index_data_type = get_smallest_possible_type(total_grid_points, "uint", default="double")

    @staticmethod
    def check_medium(medium, kgrid_k, simulation_type: SimulationType) -> bool:
        """
        Check the properties of the medium structure for correctness and validity

        Args:
            medium: kWaveMedium instance
            kgrid_k: kWaveGrid.k matrix
            is_elastic: Whether the simulation is elastic
            is_axisymmetric: Whether the simulation is axisymmetric

        Returns:
            Medium Density
        """

        # if using the fluid code, allow the density field to be blank if the medium is homogeneous
        if (not simulation_type.is_elastic_simulation()) and medium.density is None and medium.sound_speed.size == 1:
            user_medium_density_input = False
            medium.density = 1
        else:
            medium.ensure_defined("density")
            user_medium_density_input = True

        # check medium absorption inputs for the fluid code
        is_absorbing = any(medium.is_defined("alpha_coeff", "alpha_power"))
        is_stokes = simulation_type.is_axisymmetric() or medium.alpha_mode == "stokes"
        medium.set_absorbing(is_absorbing, is_stokes)

        if is_absorbing:
            medium.check_fields(kgrid_k.shape)
        return user_medium_density_input

    def check_sensor(self, kgrid_dim) -> None:
        """
        Check the Sensor properties for correctness and validity

        Args:
            k_dim: kWaveGrid dimensionality

        Returns:
            None
        """
        # =========================================================================
        # CHECK SENSOR STRUCTURE INPUTS
        # =========================================================================
        # check sensor fields
        if self.sensor is not None:
            # check the sensor input is valid
            # TODO FARID move this check as a type checking
            assert isinstance(
                self.sensor, (kSensor, NotATransducer)
            ), "sensor must be defined as an object of the kSensor or kWaveTransducer class."

            # check if sensor is a transducer, otherwise check input fields
            if not isinstance(self.sensor, NotATransducer):
                if kgrid_dim == 2:
                    # check for sensor directivity input and set flag
                    directivity = self.sensor.directivity
                    if directivity is not None and self.sensor.directivity.angle is not None:
                        # make sure the sensor mask is not blank
                        assert self.sensor.mask is not None, "The mask must be defined for the sensor"

                        # check sensor.directivity.pattern and sensor.mask have the same size
                        assert (
                            directivity.angle.shape == self.sensor.mask.shape
                        ), "sensor.directivity.angle and sensor.mask must be the same size."

                        # check if directivity size input exists, otherwise make it
                        # a constant times kgrid.dx
                        if directivity.size is None:
                            directivity.set_default_size(self.kgrid)

                        # find the unique directivity angles
                        # assign the wavenumber vectors
                        directivity.set_unique_angles(self.sensor.mask)
                        directivity.set_wavenumbers(self.kgrid)

                # check for time reversal inputs and set flags
                if not self.options.simulation_type.is_elastic_simulation() and self.sensor.time_reversal_boundary_data is not None:
                    self.record.p = False

                # check for sensor.record and set usage flgs - if no flgs are
                # given, the time history of the acoustic pressure is recorded by
                # default
                if self.sensor.record is not None:
                    # check for time reversal data
                    if self.time_rev:
                        logging.log(logging.WARN, "sensor.record is not used for time reversal reconstructions")

                    # check the input is a cell array
                    assert isinstance(self.sensor.record, list), 'sensor.record must be given as a list, e.g. ["p", "u"]'

                    # check the sensor record flgs
                    self.record.set_flags_from_list(self.sensor.record, self.options.simulation_type.is_elastic_simulation())

                # enforce the sensor.mask field unless just recording the max_all
                # and _final variables
                fields = ["p", "p_max", "p_min", "p_rms", "u", "u_non_staggered", "u_split_field", "u_max", "u_min", "u_rms", "I", "I_avg"]
                if any(self.record.is_set(fields)):
                    assert self.sensor.mask is not None

                # check if sensor mask is a binary grid, a set of cuboid corners,
                # or a set of Cartesian interpolation points
                if not self.blank_sensor:
                    if (kgrid_dim == 3 and num_dim2(self.sensor.mask) == 3) or (
                        kgrid_dim != 3 and (self.sensor.mask.shape == self.kgrid.k.shape)
                    ):
                        # check the grid is binary
                        assert self.sensor.mask.sum() == (
                            self.sensor.mask.size - (self.sensor.mask == 0).sum()
                        ), "sensor.mask must be a binary grid (numeric values must be 0 or 1)."

                        # check the grid is not empty
                        assert self.sensor.mask.sum() != 0, "sensor.mask must be a binary grid with at least one element set to 1."

                    elif self.sensor.mask.shape[0] == 2 * kgrid_dim:
                        # make sure the points are integers
                        assert np.all(self.sensor.mask % 1 == 0), "sensor.mask cuboid corner indices must be integers."

                        # store a copy of the cuboid corners
                        self.record.cuboid_corners_list = self.sensor.mask

                        # check the list makes sense
                        if np.any(self.sensor.mask[self.kgrid.dim :, :] - self.sensor.mask[: self.kgrid.dim, :] < 0):
                            if kgrid_dim == 1:
                                raise ValueError("sensor.mask cuboid corners must be defined " "as [x1, x2; ...]." " where x2 => x1, etc.")
                            elif kgrid_dim == 2:
                                raise ValueError(
                                    "sensor.mask cuboid corners must be defined " "as [x1, y1, x2, y2; ...]." " where x2 => x1, etc."
                                )
                            elif kgrid_dim == 3:
                                raise ValueError(
                                    "sensor.mask cuboid corners must be defined"
                                    " as [x1, y1, z1, x2, y2, z2; ...]."
                                    " where x2 => x1, etc."
                                )

                        # check the list are within bounds
                        if np.any(self.sensor.mask < 1):
                            raise ValueError("sensor.mask cuboid corners must be within the grid.")
                        else:
                            if kgrid_dim == 1:
                                if np.any(self.sensor.mask > self.kgrid.Nx):
                                    raise ValueError("sensor.mask cuboid corners must be within the grid.")
                            elif kgrid_dim == 2:
                                if np.any(self.sensor.mask[[0, 2], :] > self.kgrid.Nx) or np.any(
                                    self.sensor.mask[[1, 3], :] > self.kgrid.Ny
                                ):
                                    raise ValueError("sensor.mask cuboid corners must be within the grid.")
                            elif kgrid_dim == 3:
                                if (
                                    np.any(self.sensor.mask[[0, 3], :] > self.kgrid.Nx)
                                    or np.any(self.sensor.mask[[1, 4], :] > self.kgrid.Ny)
                                    or np.any(self.sensor.mask[[2, 5], :] > self.kgrid.Nz)
                                ):
                                    raise ValueError("sensor.mask cuboid corners must be within the grid.")

                        # create a binary mask for display from the list of corners
                        # TODO FARID mask should be option_factory in sensor not here
                        self.sensor.mask = np.zeros_like(self.kgrid.k, dtype=bool)
                        cuboid_corners_list = self.record.cuboid_corners_list
                        for cuboid_index in range(cuboid_corners_list.shape[1]):
                            if self.kgrid.dim == 1:
                                self.sensor.mask[cuboid_corners_list[0, cuboid_index] : cuboid_corners_list[1, cuboid_index]] = 1
                            if self.kgrid.dim == 2:
                                self.sensor.mask[
                                    cuboid_corners_list[0, cuboid_index] : cuboid_corners_list[2, cuboid_index],
                                    cuboid_corners_list[1, cuboid_index] : cuboid_corners_list[3, cuboid_index],
                                ] = 1
                            if self.kgrid.dim == 3:
                                self.sensor.mask[
                                    cuboid_corners_list[0, cuboid_index] : cuboid_corners_list[3, cuboid_index],
                                    cuboid_corners_list[1, cuboid_index] : cuboid_corners_list[4, cuboid_index],
                                    cuboid_corners_list[2, cuboid_index] : cuboid_corners_list[5, cuboid_index],
                                ] = 1
                    else:
                        # check the Cartesian sensor mask is the correct size
                        # (1 x N, 2 x N, 3 x N)
                        assert (
                            self.sensor.mask.shape[0] == kgrid_dim and num_dim2(self.sensor.mask) <= 2
                        ), f"Cartesian sensor.mask for a {kgrid_dim}D simulation must be given as a {kgrid_dim} by N array."

                        # set Cartesian mask flag (this is modified in
                        # createStorageVariables if the interpolation setting is
                        # set to nearest)
                        self.binary_sensor_mask = False

                        # extract Cartesian data from sensor mask
                        if kgrid_dim == 1:
                            # align sensor data as a column vector to be the
                            # same as kgrid.x_vec so that calls to interp1
                            # return data in the correct dimension
                            self.sensor_x = np.reshape((self.sensor.mask, (-1, 1)))

                            # add sensor_x to the record structure for use with
                            # the _extractSensorData subfunction
                            self.record.sensor_x = self.sensor_x
                            "record.sensor_x = sensor_x;"

                        elif kgrid_dim == 2:
                            self.sensor_x = self.sensor.mask[0, :]
                            self.sensor_y = self.sensor.mask[1, :]
                        elif kgrid_dim == 3:
                            self.sensor_x = self.sensor.mask[0, :]
                            self.sensor_y = self.sensor.mask[1, :]
                            self.sensor_z = self.sensor.mask[2, :]

                        # compute an equivalent sensor mask using nearest neighbour
                        # interpolation, if flgs.time_rev = false and
                        # cartesian_interp = 'linear' then this is only used for
                        # display, if flgs.time_rev = true or cartesian_interp =
                        # 'nearest' this grid is used as the sensor.mask
                        self.sensor.mask, self.order_index, self.reorder_index = cart2grid(
                            self.kgrid, self.sensor.mask, self.options.simulation_type.is_axisymmetric()
                        )

                        # if in time reversal mode, reorder the p0 input data in
                        # the order of the binary sensor_mask
                        if self.time_rev:
                            raise NotImplementedError
                            """
                            # append the reordering data
                            new_col_pos = length(sensor.time_reversal_boundary_data(1, :)) + 1;
                            sensor.time_reversal_boundary_data(:, new_col_pos) = order_index;
        
                            # reorder p0 based on the order_index
                            sensor.time_reversal_boundary_data = sort_rows(sensor.time_reversal_boundary_data, new_col_pos);
        
                            # remove the reordering data
                            sensor.time_reversal_boundary_data = sensor.time_reversal_boundary_data(:, 1:new_col_pos - 1);
                            """
            else:
                # set transducer sensor flag
                self.transducer_sensor = True
                self.record.p = False

                # check to see if there is an elevation focus
                if not np.isinf(self.sensor.elevation_focus_distance):
                    # set flag
                    self.transducer_receive_elevation_focus = True

                    # get the elevation mask that is used to extract the correct values
                    # from the sensor data buffer for averaging
                    self.transducer_receive_mask = self.sensor.elevation_beamforming_mask

        # check for directivity inputs with time reversal
        if kgrid_dim == 2 and self.use_sensor and self.compute_directivity and self.time_rev:
            logging.log(logging.WARN, "sensor directivity fields are not used for time reversal.")

    def check_source(self, k_dim, k_Nt) -> None:
        """
        Check the source properties for correctness and validity

        Args:
            kgrid_dim: kWaveGrid dimension
            k_Nt: Number of time steps in kWaveGrid

        Returns:
            None
        """
        # =========================================================================
        # CHECK SOURCE STRUCTURE INPUTS
        # =========================================================================

        # check source inputs
        if not isinstance(self.source, (kSource, NotATransducer)):
            # allow an invalid or empty source input if computing time reversal,
            # otherwise return error
            assert self.time_rev, "source must be defined as an object of the kSource or kWaveTransducer classes."

        elif not isinstance(self.source, NotATransducer):
            # --------------------------
            # SOURCE IS NOT A TRANSDUCER
            # --------------------------

            """
                check allowable source types
                
                Depending on the kgrid dimensionality and the simulation type, 
                    following fields are allowed & might be use:
                
                kgrid.dim == 1:
                    non-elastic code:
                        ['p0', 'p', 'p_mask', 'p_mode', 'p_frequency_ref', 'ux', 'u_mask', 'u_mode', 'u_frequency_ref']
                kgrid.dim == 2:
                    non-elastic code:
                        ['p0', 'p', 'p_mask', 'p_mode', 'p_frequency_ref', 'ux', 'uy', 'u_mask', 'u_mode', 'u_frequency_ref']
                    elastic code:
                        ['p0', 'sxx', 'syy', 'sxy', 's_mask', 's_mode', 'ux', 'uy', 'u_mask', 'u_mode']
                kgrid.dim == 3:
                    non-elastic code:
                        ['p0', 'p', 'p_mask', 'p_mode', 'p_frequency_ref', 'ux', 'uy', 'uz', 'u_mask', 'u_mode', 'u_frequency_ref']
                    elastic code:
                        ['p0', 'sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz', 's_mask', 's_mode', 'ux', 'uy', 'uz', 'u_mask', 'u_mode']
            """

            self.source.validate(self.kgrid)

            # check for a time varying pressure source input
            if self.source.p is not None:
                # check the source mode input is valid
                if self.source.p_mode is None:
                    self.source.p_mode = self.SOURCE_P_MODE_DEF

                if self.source_p > k_Nt:
                    logging.log(logging.WARN, "  source.p has more time points than kgrid.Nt, remaining time points will not be used.")

                # create an indexing variable corresponding to the location of all the source elements
                self.p_source_pos_index = matlab_find(self.source.p_mask)

                # check if the mask is binary or labelled
                p_unique = np.unique(self.source.p_mask)

                # create a second indexing variable
                if p_unique.size <= 2 and p_unique.sum() == 1:
                    # set signal index to all elements
                    self.p_source_sig_index = ":"
                else:
                    # set signal index to the labels (this allows one input signal
                    # to be used for each source label)
                    self.p_source_sig_index = self.source.p_mask(self.source.p_mask != 0)

                # convert the data type depending on the number of indices
                self.p_source_pos_index = cast_to_type(self.p_source_pos_index, self.index_data_type)
                if self.source_p_labelled:
                    self.p_source_sig_index = cast_to_type(self.p_source_sig_index, self.index_data_type)

            # check for time varying velocity source input and set source flag
            if any([(getattr(self.source, k) is not None) for k in ["ux", "uy", "uz", "u_mask"]]):
                # check the source mode input is valid
                if self.source.u_mode is None:
                    self.source.u_mode = self.SOURCE_U_MODE_DEF

                # create an indexing variable corresponding to the location of all
                # the source elements
                self.u_source_pos_index = matlab_find(self.source.u_mask)

                # check if the mask is binary or labelled
                u_unique = np.unique(self.source.u_mask)

                # create a second indexing variable
                if u_unique.size <= 2 and u_unique.sum() == 1:
                    # set signal index to all elements
                    self.u_source_sig_index = ":"
                else:
                    # set signal index to the labels (this allows one input signal
                    # to be used for each source label)
                    self.u_source_sig_index = self.source.u_mask[self.source.u_mask != 0]

                # convert the data type depending on the number of indices
                self.u_source_pos_index = cast_to_type(self.u_source_pos_index, self.index_data_type)
                if self.source_u_labelled:
                    self.u_source_sig_index = cast_to_type(self.u_source_sig_index, self.index_data_type)

            # check for time varying stress source input and set source flag
            if any([(getattr(self.source, k) is not None) for k in ["sxx", "syy", "szz", "sxy", "sxz", "syz", "s_mask"]]):
                # create an indexing variable corresponding to the location of all
                # the source elements
                raise NotImplementedError
                "s_source_pos_index = find(source.s_mask != 0);"

                # check if the mask is binary or labelled
                "s_unique = unique(source.s_mask);"

                # create a second indexing variable
                if eng.eval("numel(s_unique) <= 2 && sum(s_unique) == 1"):  # noqa: F821
                    # set signal index to all elements
                    eng.workspace["s_source_sig_index"] = ":"  # noqa: F821

                else:
                    # set signal index to the labels (this allows one input signal
                    # to be used for each source label)
                    s_source_sig_index = source.s_mask(source.s_mask != 0)  # noqa

                f"s_source_pos_index = {self.index_data_type}(s_source_pos_index);"
                if self.source_s_labelled:
                    f"s_source_sig_index = {self.index_data_type}(s_source_sig_index);"

        else:
            # ----------------------
            # SOURCE IS A TRANSDUCER
            # ----------------------

            # if the sensor is a transducer, check that the simulation is in 3D
            assert k_dim == 3, "Transducer inputs are only compatible with 3D simulations."

            # get the input signal - this is appended with zeros if required to
            # account for the beamforming delays (this will throw an error if the
            # input signal is not defined)
            self.transducer_input_signal = self.source.input_signal

            # get the delay mask that accounts for the beamforming delays and
            # elevation focussing; this is used so that a single time series can be
            # applied to the complete transducer mask with different delays
            delay_mask = self.source.delay_mask()

            # set source flag - this should be the length of signal minus the
            # maximum delay
            self.transducer_source = self.transducer_input_signal.size - delay_mask.max()

            # get the active elements mask
            active_elements_mask = self.source.active_elements_mask

            # get the apodization mask if not set to 'Rectangular' and convert to a
            # linear array
            if self.source.transmit_apodization == "Rectangular":
                self.transducer_transmit_apodization = 1
            else:
                self.transducer_transmit_apodization = self.source.transmit_apodization_mask
                self.transducer_transmit_apodization = self.transducer_transmit_apodization[active_elements_mask != 0]

            # create indexing variable corresponding to the active elements
            # and convert the data type depending on the number of indices
            self.u_source_pos_index = matlab_find(active_elements_mask).astype(self.index_data_type)

            # convert the delay mask to an indexing variable (this doesn't need to
            # be modified if the grid is expanded) which tells each point in the
            # source mask which point in the input_signal should be used
            delay_mask = matlab_mask(delay_mask, active_elements_mask != 0)  # compatibility

            # convert the data type depending on the maximum value of the delay
            # mask and the length of the source
            smallest_type = get_smallest_possible_type(delay_mask.max(), "uint")
            if smallest_type is not None:
                delay_mask = delay_mask.astype(smallest_type)

            # move forward by 1 as a delay of 0 corresponds to the first point in the input signal
            self.delay_mask = delay_mask + 1

            # clean up unused variables
            del active_elements_mask

    def check_kgrid_time(self) -> None:
        """
        Check time-related kWaveGrid inputs

        Returns:
            None
        """

        # check kgrid for t_array existance, and create if not defined
        if isinstance(self.kgrid.t_array, str) and self.kgrid.t_array == "auto":
            # check for time reversal mode
            if self.time_rev:
                raise ValueError("kgrid.t_array (Nt and dt) must be defined explicitly in time reversal mode.")

            # check for time varying sources
            if (not self.source_p0_elastic) and (
                self.source_p
                or self.source_ux
                or self.source_uy
                or self.source_uz
                or self.source_sxx
                or self.source_syy
                or self.source_szz
                or self.source_sxy
                or self.source_sxz
                or self.source_syz
            ):
                raise ValueError("kgrid.t_array (Nt and dt) must be defined explicitly when using a time-varying source.")

            # create the time array using the compressional sound speed
            self.kgrid.makeTime(self.medium.sound_speed, self.KSPACE_CFL)

        # check kgrid.t_array for stability given medium properties
        if not self.options.simulation_type.is_elastic_simulation():
            # calculate the largest timestep for which the model is stable

            dt_stability_limit = check_stability(self.kgrid, self.medium)

            # give a warning if the timestep is larger than stability limit allows
            if self.kgrid.dt > dt_stability_limit:
                logging.log(logging.WARN, "  time step may be too large for a stable simulation.")

    @staticmethod
    def select_precision(opt: SimulationOptions):
        """
        Select the minimal precision for storing the data

        Args:
            opt: SimulationOptions instance

        Returns:
            Minimal precision for variable allocation

        """
        # set storage variable type based on data_cast - this enables the
        # output variables to be directly created in the data_cast format,
        # rather than creating them in double precision and then casting them
        if opt.data_cast == "off":
            precision = "double"
        elif opt.data_cast == "single":
            precision = "single"
        elif opt.data_cast == "gsingle":
            precision = "single"
        elif opt.data_cast == "gdouble":
            precision = "double"
        elif opt.data_cast == "gpuArray":
            raise NotImplementedError("gpuArray is not supported in Python-version")
        elif opt.data_cast == "kWaveGPUsingle":
            precision = "single"
        elif opt.data_cast == "kWaveGPUdouble":
            precision = "double"
        else:
            raise ValueError("'Unknown ''DataCast'' option'")
        return precision

    def check_input_combinations(self, opt: SimulationOptions, user_medium_density_input: bool, k_dim, pml_size, kgrid_N) -> None:
        """
        Check the input combinations for correctness and validity

        Args:
            opt: SimulationOptions instance
            user_medium_density_input: Medium Density
            k_dim: kWaveGrid dimensionality
            pml_size: Size of the PML
            kgrid_N: kWaveGrid size in each direction

        Returns:
            None
        """
        # =========================================================================
        # CHECK FOR VALID INPUT COMBINATIONS
        # =========================================================================

        # enforce density input if velocity sources or output are being used
        if not user_medium_density_input and (
            self.source_ux or self.source_uy or self.source_uz or self.record.u or self.record.u_max or self.record.u_rms
        ):
            raise ValueError(
                "medium.density must be explicitly defined " "if velocity inputs or outputs are used, even in homogeneous media."
            )

        # TODO(walter): move to check medium
        # enforce density input if nonlinear equations are being used
        if not user_medium_density_input and self.medium.is_nonlinear():
            raise ValueError("medium.density must be explicitly defined if medium.BonA is specified.")

        # check sensor compatability options for flgs.compute_directivity
        if self.use_sensor and k_dim == 2 and self.compute_directivity and not self.binary_sensor_mask and opt.cartesian_interp == "linear":
            raise ValueError(
                "sensor directivity fields are only compatible " "with binary sensor masks or " "CartInterp" " set to " "nearest" "."
            )

        # check for split velocity output
        if self.record.u_split_field and not self.binary_sensor_mask:
            raise ValueError("The option sensor.record = {" "u_split_field" "} is only compatible " "with a binary sensor mask.")

        # check input options for data streaming *****
        if opt.stream_to_disk:
            if not self.use_sensor or self.time_rev:
                raise ValueError(
                    "The optional input "
                    "StreamToDisk"
                    " is currently only compatible "
                    "with forward simulations using a non-zero sensor mask."
                )
            elif self.sensor.record is not None and self.sensor.record.ismember(self.record.flags[1:]).any():
                raise ValueError(
                    "The optional input " "StreamToDisk" " is currently only compatible " "with sensor.record = {" "p" "} (the default)."
                )

        is_axisymmetric = self.options.simulation_type.is_axisymmetric()
        # make sure the PML size is smaller than the grid if PMLInside is true
        if opt.pml_inside and (
            (k_dim == 1 and (pml_size.x * 2 > self.kgrid.Nx))
            or (k_dim == 2 and not is_axisymmetric and ((pml_size.x * 2 > kgrid_N[0]) or (pml_size.y * 2 > kgrid_N[1])))
            or (k_dim == 2 and is_axisymmetric and ((pml_size.x * 2 > kgrid_N[0]) or (pml_size.y > kgrid_N[1])))
            or (k_dim == 3 and ((pml_size.x * 2 > kgrid_N[0]) or (pml_size.x * 2 > kgrid_N[1]) or (pml_size.z * 2 > kgrid_N[2])))
        ):
            raise ValueError("The size of the PML must be smaller than the size of the grid.")

        # make sure the PML is inside if using a non-uniform grid
        if self.nonuniform_grid and not opt.pml_inside:
            raise ValueError("''PMLInside'' must be true for simulations using non-uniform grids.")

        # check for compatible input options if saving to disk
        # modified by Farid | disabled temporarily!
        # if k_dim == 3 and isinstance(self.options.save_to_disk, str) and
        #                   (not self.use_sensor or not self.binary_sensor_mask or self.time_rev):
        #     raise ValueError('The optional input ''SaveToDisk'' is currently only compatible
        #                       with forward simulations using a non-zero binary sensor mask.')

        # check the record start time is within range
        record_start_index = self.sensor.record_start_index
        if self.use_sensor and ((record_start_index > self.kgrid.Nt) or (record_start_index < 1)):
            raise ValueError("sensor.record_start_index must be between 1 and the number of time steps.")

        # ensure 'WSWA' symmetry if using axisymmetric code with 'SaveToDisk'
        if is_axisymmetric and self.options.radial_symmetry != "WSWA" and isinstance(self.options.save_to_disk, str):
            # display a warning only if using WSWS symmetry (not WSWA-FFT)
            if self.options.radial_symmetry.startswith("WSWS"):
                logging.log(
                    logging.WARN, "  Optional input " "RadialSymmetry" " changed to " "WSWA" " for compatability with " "SaveToDisk" "."
                )

            # update setting
            self.options.radial_symmetry = "WSWA"

        # ensure p0 smoothing is switched off if p0 is empty
        if not self.source_p0:
            self.options.smooth_p0 = False

        # start log if required
        if opt.create_log:
            raise NotImplementedError(f"diary({self.LOG_NAME}.txt');")

        # update command line status
        if self.time_rev:
            logging.log(logging.INFO, "  time reversal mode")

        # cleanup unused variables
        for k in list(self.__dict__.keys()):
            if k.endswith("_DEF"):
                delattr(self, k)

    def smooth_and_enlarge(self, source, k_dim, kgrid_N, opt: SimulationOptions) -> None:
        """
        Smooth and enlarge grids

        Args:
            source: kWaveSource instance
            k_dim: kWaveGrid dimensionality
            kgrid_N: kWaveGrid size in each direction
            opt: SimulationOptions

        Returns:
            None
        """

        # smooth the initial pressure distribution p0 if required, and then restore
        # the maximum magnitude
        #   NOTE 1: if p0 has any values at the edge of the domain, the smoothing
        #   may cause part of p0 to wrap to the other side of the domain
        #   NOTE 2: p0 is smoothed before the grid is expanded to ensure that p0 is
        #   exactly zero within the PML
        #   NOTE 3: for the axisymmetric code, p0 is smoothed assuming WS origin
        #   symmetry
        if self.source_p0 and self.options.smooth_p0:
            # update command line status
            logging.log(logging.INFO, "  smoothing p0 distribution...")

            if self.options.simulation_type.is_axisymmetric():
                if self.options.radial_symmetry in ["WSWA-FFT", "WSWA"]:
                    # create a new kWave grid object with expanded radial grid
                    kgrid_exp = kWaveGrid([kgrid_N.x, kgrid_N.y * 4], [self.kgrid.dx, self.kgrid.dy])

                    # mirror p0 in radial dimension using WSWA symmetry
                    self.source.p0 = self.source.p0.astype(float)
                    p0_exp = np.zeros((kgrid_exp.Nx, kgrid_exp.Ny))
                    p0_exp[:, kgrid_N.y * 0 + 0 : kgrid_N.y * 1] = self.source.p0
                    p0_exp[:, kgrid_N.y * 1 + 1 : kgrid_N.y * 2] = -np.fliplr(self.source.p0[:, 1:])
                    p0_exp[:, kgrid_N.y * 2 + 0 : kgrid_N.y * 3] = -self.source.p0
                    p0_exp[:, kgrid_N.y * 3 + 1 : kgrid_N.y * 4] = np.fliplr(self.source.p0[:, 1:])

                elif self.options.radial_symmetry in ["WSWS-FFT", "WSWS"]:
                    # create a new kWave grid object with expanded radial grid
                    kgrid_exp = kWaveGrid([kgrid_N.x, kgrid_N.y * 2 - 2], [self.kgrid.dx, self.kgrid.dy])

                    # mirror p0 in radial dimension using WSWS symmetry
                    p0_exp = np.zeros((kgrid_exp.Nx, kgrid_exp.Ny))
                    p0_exp[:, 1 : kgrid_N.y] = source.p0
                    p0_exp[:, kgrid_N.y + 0 : kgrid_N.y * 2 - 2] = np.fliplr(source.p0[:, 1:-1])

                # smooth p0
                p0_exp = smooth(p0_exp, True)

                # trim back to original size
                source.p0 = p0_exp[:, 0 : self.kgrid.Ny]

                # clean up unused variables
                del kgrid_exp
                del p0_exp
            else:
                source.p0 = smooth(source.p0, True)

        # expand the computational grid if the PML is set to be outside the input
        # grid defined by the user
        if opt.pml_inside is False:
            expand_results = expand_grid_matrices(
                self.kgrid,
                self.medium,
                self.source,
                self.sensor,
                self.options,
                dotdict(
                    {
                        "p_source_pos_index": self.p_source_pos_index,
                        "u_source_pos_index": self.u_source_pos_index,
                        "s_source_pos_index": self.s_source_pos_index,
                    }
                ),
                dotdict(
                    {
                        "axisymmetric": self.options.simulation_type.is_axisymmetric(),
                        "use_sensor": self.use_sensor,
                        "blank_sensor": self.blank_sensor,
                        "cuboid_corners": self.cuboid_corners,
                        "source_p0": self.source_p0,
                        "source_p": self.source_p,
                        "source_ux": self.source_ux,
                        "source_uy": self.source_uy,
                        "source_uz": self.source_uz,
                        "transducer_source": self.transducer_source,
                        "source_sxx": self.source_sxx,
                        "source_syy": self.source_syy,
                        "source_szz": self.source_szz,
                        "source_sxy": self.source_sxy,
                        "source_sxz": self.source_sxz,
                        "source_syz": self.source_syz,
                    }
                ),
            )
            self.kgrid, self.index_data_type, self.p_source_pos_index, self.u_source_pos_index, self.s_source_pos_index = expand_results

        # get maximum prime factors
        if self.options.simulation_type.is_axisymmetric():
            prime_facs = self.kgrid.highest_prime_factors(self.options.radial_symmetry[:4])
        else:
            prime_facs = self.kgrid.highest_prime_factors()

        # give warning for bad dimension sizes
        if prime_facs.max() > self.HIGHEST_PRIME_FACTOR_WARNING:
            prime_facs = prime_facs[prime_facs != 0]
            logging.log(logging.WARN, f"Highest prime factors in each dimension are {prime_facs}")
            logging.log(logging.WARN, "Use dimension sizes with lower prime factors to improve speed")
        del prime_facs

        # smooth the sound speed distribution if required
        if opt.smooth_c0 and num_dim2(self.medium.sound_speed) == k_dim and self.medium.sound_speed.size > 1:
            logging.log(logging.INFO, "  smoothing sound speed distribution...")
            self.medium.sound_speed = smooth(self.medium.sound_speed)

        # smooth the ambient density distribution if required
        if opt.smooth_rho0 and num_dim2(self.medium.density) == k_dim and self.medium.density.size > 1:
            logging.log(logging.INFO, "smoothing density distribution...")
            self.medium.density = smooth(self.medium.density)

    def create_sensor_variables(self) -> None:
        """
        Create the sensor related variables

        Returns:
            None
        """
        # define the output variables and mask indices if using the sensor
        if self.use_sensor:
            if not self.blank_sensor or self.options.save_to_disk:
                if self.cuboid_corners:
                    # create empty list of sensor indices
                    self.sensor_mask_index = []

                    # loop through the list of cuboid corners, and extract the
                    # sensor mask indices for each cube
                    for cuboid_index in range(self.record.cuboid_corners_list.shape[1]):
                        # create empty binary mask
                        temp_mask = np.zeros_like(self.kgrid.k, dtype=bool)

                        if self.kgrid.dim == 1:
                            self.sensor.mask[
                                self.record.cuboid_corners_list[0, cuboid_index] : self.record.cuboid_corners_list[1, cuboid_index]
                            ] = 1
                        if self.kgrid.dim == 2:
                            self.sensor.mask[
                                self.record.cuboid_corners_list[0, cuboid_index] : self.record.cuboid_corners_list[2, cuboid_index],
                                self.record.cuboid_corners_list[1, cuboid_index] : self.record.cuboid_corners_list[3, cuboid_index],
                            ] = 1
                        if self.kgrid.dim == 3:
                            self.sensor.mask[
                                self.record.cuboid_corners_list[0, cuboid_index] : self.record.cuboid_corners_list[3, cuboid_index],
                                self.record.cuboid_corners_list[1, cuboid_index] : self.record.cuboid_corners_list[4, cuboid_index],
                                self.record.cuboid_corners_list[2, cuboid_index] : self.record.cuboid_corners_list[5, cuboid_index],
                            ] = 1

                        # extract mask indices
                        self.sensor_mask_index.append(matlab_find(temp_mask))
                    self.sensor_mask_index = np.array(self.sensor_mask_index)

                    # cleanup unused variables
                    del temp_mask

                else:
                    # create mask indices (this works for both normal sensor and
                    # transducer inputs)
                    self.sensor_mask_index = np.where(self.sensor.mask.flatten(order="F") != 0)[0] + 1  # +1 due to matlab indexing
                    self.sensor_mask_index = np.expand_dims(self.sensor_mask_index, -1)  # compatibility, n => [n, 1]

                # convert the data type depending on the number of indices (this saves
                # memory)
                self.sensor_mask_index = cast_to_type(self.sensor_mask_index, self.index_data_type)

            else:
                # set the sensor mask index variable to be empty
                self.sensor_mask_index = []

    def create_absorption_vars(self) -> None:
        """
        Create absorption variables for the fluid code based on
        the expanded and smoothed values of the medium parameters (if not saving to disk)

        Returns:
            None
        """
        if not self.options.simulation_type.is_elastic_simulation() and not self.options.save_to_disk:
            self.absorb_nabla1, self.absorb_nabla2, self.absorb_tau, self.absorb_eta = create_absorption_variables(
                self.kgrid, self.medium, self.equation_of_state
            )

    def assign_pseudonyms(self, medium: kWaveMedium, kgrid: kWaveGrid) -> None:
        """
        Shorten commonly used field names (these act only as pointers provided that the values aren't modified)
        (done after enlarging and smoothing the grids)

        Args:
            medium: kWaveMedium instance
            kgrid: kWaveGrid instance

        Returns:
            None
        """
        self.dt = float(kgrid.dt)
        self.rho0 = medium.density
        self.c0 = medium.sound_speed

    def scale_source_terms(self, is_scale_source_terms) -> None:
        """
        Scale the source terms based on the expanded and smoothed values of the medium parameters

        Args:
            is_scale_source_terms: Should the source terms be scaled

        Returns:
            None
        """
        if not is_scale_source_terms:
            return
        try:
            p_source_pos_index = self.p_source_pos_index
        except AttributeError:
            p_source_pos_index = None

        try:
            s_source_pos_index = self.s_source_pos_index
        except AttributeError:
            s_source_pos_index = None

        try:
            u_source_pos_index = self.u_source_pos_index
        except AttributeError:
            u_source_pos_index = None

        self.transducer_input_signal = scale_source_terms_func(
            self.c0,
            self.dt,
            self.kgrid,
            self.source,
            p_source_pos_index,
            s_source_pos_index,
            u_source_pos_index,
            self.transducer_input_signal,
            dotdict(
                {
                    "nonuniform_grid": self.nonuniform_grid,
                    "source_ux": self.source_ux,
                    "source_uy": self.source_uy,
                    "source_uz": self.source_uz,
                    "transducer_source": self.transducer_source,
                    "source_p": self.source_p,
                    "source_p0": self.source_p0,
                    "use_w_source_correction_p": self.use_w_source_correction_p,
                    "use_w_source_correction_u": self.use_w_source_correction_u,
                    "source_sxx": self.source_sxx,
                    "source_syy": self.source_syy,
                    "source_szz": self.source_szz,
                    "source_sxy": self.source_sxy,
                    "source_sxz": self.source_sxz,
                    "source_syz": self.source_syz,
                }
            ),
        )

    def create_pml_indices(self, kgrid_dim, kgrid_N: Vector, pml_size, pml_inside, is_axisymmetric):
        """
        Define index variables to remove the PML from the display if the optional
        input 'PlotPML' is set to false

        Args:
            kgrid_dim: kWaveGrid dimensinality
            kgrid_N: kWaveGrid size in each direction
            pml_size: Size of the PML
            pml_inside: Whether the PML is inside the grid defined by the user
            is_axisymmetric: Whether the simulation is axisymmetric

        """
        # comment by Farid: PlotPML is always False in Python version,
        #                       therefore if statement removed
        if kgrid_dim == 1:
            self.x1 = pml_size.x + 1.0
            self.x2 = kgrid_N.x - pml_size.x
        elif kgrid_dim == 2:
            self.x1 = pml_size.x + 1.0
            self.x2 = kgrid_N.x - pml_size.x
            if is_axisymmetric:
                self.y1 = 1.0
            else:
                self.y1 = pml_size.y + 1.0
            self.y2 = kgrid_N.y - pml_size.y
        elif kgrid_dim == 3:
            self.x1 = pml_size.x + 1.0
            self.x2 = kgrid_N.x - pml_size.x
            self.y1 = pml_size.y + 1.0
            self.y2 = kgrid_N.y - pml_size.y
            self.z1 = pml_size.z + 1.0
            self.z2 = kgrid_N.z - pml_size.z

        # define index variables to allow original grid size to be maintained for
        # the _final and _all output variables if 'PMLInside' is set to false
        # if self.record is None:
        #     self.record = Recorder()
        self.record.set_index_variables(self.kgrid, pml_size, pml_inside, is_axisymmetric)
