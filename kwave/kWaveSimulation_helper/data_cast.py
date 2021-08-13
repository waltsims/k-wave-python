from kwave import kWaveMedium, kWaveGrid, SimulationOptions
from kwave.utils import cast_to_type
from kwave.utils import dotdict


def dataCast(data_cast, medium: kWaveMedium, kgrid: kWaveGrid, opt: SimulationOptions, values: dotdict, flags: dotdict):
    # NOTE from Farid: this method is not used now.
    # Therefore, there still exists 'self' calls at the end.
    # Originally, self was referring to the kWaveSimulation class

    # update command line status
    print(f'  casting variables to {data_cast} type...')

    # create list of variable names used in all dimensions
    if flags.elastic_code:  # pragma: no cover
        cast_variables = ['dt', 'mu', 'lambda']
    else:
        cast_variables = ['dt', 'kappa', 'c0', 'rho0']

    # create a separate list for indexing variables
    cast_index_variables = []

    # add variables specific to simulations in certain dimensions
    if kgrid.dim == 1:
        # variables used in the fluid code
        cast_variables = cast_variables + ['ddx_k', 'shift_pos', 'shift_neg','pml_x', 'pml_x_sgx', 'rho0_sgx_inv']

    elif kgrid.dim == 2:
        # variables used in both fluid and elastic codes
        cast_variables = cast_variables + ['ddx_k_shift_pos', 'ddx_k_shift_neg', 'pml_x', 'pml_y',
                                           'pml_x_sgx', 'pml_y_sgy', 'rho0_sgx_inv', 'rho0_sgy_inv']

        # y-dimension shift and derivative variables (these differ between the fluid/elastic code and the axisymmetric code)
        if flags.axisymmetric:

            # y-axis variables
            cast_variables = cast_variables + ['y_vec', 'y_vec_sg']

            # derivative and shift variables
            if opt.radial_symmetry in ['WSWA-FFT','WSWS-FFT','WS-FFT']:
                cast_variables = cast_variables + ['ddy_k', 'y_shift_pos', 'y_shift_neg']

            elif opt.radial_symmetry == 'WSWA':
                cast_variables = cast_variables + ['ddy_k_wswa', 'ddy_k_hahs']

            elif opt.radial_symmetry == 'WSWS':
                cast_variables = cast_variables + ['ddy_k_wsws', 'ddy_k_haha']

        else:
            # derivative and shift variables in the regular code
            cast_variables = cast_variables + ['ddy_k_shift_pos', 'ddy_k_shift_neg']

        # extra variables only used in elastic code
        if flags.elastic_code:  # pragma: no cover

            # variables used in both lossless and lossy case
            cast_variables = cast_variables +  ['mu_sgxy', 'mpml_x', 'mpml_y', 'mpml_x_sgx', 'mpml_y_sgy']

            # extra variables only used in the lossy case
            if flags.kelvin_voigt_model:
                cast_variables = cast_variables + ['chi', 'eta', 'eta_sgxy']

    elif kgrid.dim == 3:
        # variables used in both fluid and elastic codes
        cast_variables = cast_variables + ['ddx_k_shift_pos', 'ddy_k_shift_pos', 'ddz_k_shift_pos',
                                           'ddx_k_shift_neg', 'ddy_k_shift_neg', 'ddz_k_shift_neg',
                                           'pml_x', 'pml_y', 'pml_z',
                                           'pml_x_sgx', 'pml_y_sgy', 'pml_z_sgz',
                                           'rho0_sgx_inv', 'rho0_sgy_inv', 'rho0_sgz_inv']

        # extra variables only used in elastic code
        if flags.elastic_code:  # pragma: no cover
            # variables used in both lossless and lossy case
            cast_variables = cast_variables + ['mu_sgxy', 'mu_sgxz', 'mu_sgyz',
                                               'mpml_x', 'mpml_y', 'mpml_z',
                                               'mpml_x_sgx', 'mpml_y_sgy', 'mpml_z_sgz']

            # extra variables only used in the lossy case
            if flags.kelvin_voigt_model:
                cast_variables = cast_variables + ['chi', 'eta', 'eta_sgxy', 'eta_sgxz', 'eta_sgyz']

    # add sensor mask variables
    if flags.use_sensor:
        cast_index_variables += ['sensor_mask_index']
        if flags.binary_sensor_mask and (values.record.u_non_staggered or values.record.I or values.record.I_avg):
            if kgrid.dim == 1:
                cast_index_variables += ['record.x_shift_neg']
            elif kgrid.dim == 2:
                cast_index_variables += ['record.x_shift_neg', 'record.y_shift_neg']
            elif kgrid.dim == 3:
                cast_index_variables += ['record.x_shift_neg', 'record.y_shift_neg', 'record.z_shift_neg']

    # additional variables only used if the medium is absorbing
    if values.equation_of_state == 'absorbing':
        cast_variables += ['absorb_nabla1', 'absorb_nabla2', 'absorb_eta', 'absorb_tau']

    # additional variables only used if the propagation is nonlinear
    if medium.is_nonlinear():
        cast_variables += ['medium.BonA']

    # additional variables only used if there is an initial pressure source
    if flags.source_p0:
        cast_variables += ['source.p0']

    # additional variables only used if there is a time varying pressure source term
    if flags.source_p:
        cast_variables += ['source.p']
        cast_index_variables += ['p_source_pos_index']
        if flags.source_p_labelled:
            cast_index_variables += ['p_source_sig_index']

    # additional variables only used if there is a time varying velocity source term
    if flags.source_ux or flags.source_uy or flags.source_uz:
        cast_index_variables += ['u_source_pos_index']
        if self.source_u_labelled:
            cast_index_variables += ['u_source_sig_index']

    if flags.source_ux:
        cast_variables += ['source.ux']
    if flags.source_uy:
        cast_variables += ['source.uy']
    if flags.source_uz:
        cast_variables += ['source.uz']

    # additional variables only used if there is a time varying stress source term
    if flags.source_sxx or flags.source_syy or flags.source_szz or flags.source_sxy or flags.source_sxz or flags.source_syz:
        cast_index_variables += ['s_source_pos_index']
        if flags.source_s_labelled:
            cast_index_variables += ['s_source_sig_index']

    if flags.source_sxx:
        cast_variables += ['source.sxx']
    if flags.source_syy:
        cast_variables += ['source.syy']
    if flags.source_szz:
        cast_variables += ['source.szz']
    if flags.source_sxy:
        cast_variables += ['source.sxy']
    if flags.source_sxz:
        cast_variables += ['source.sxz']
    if flags.source_syz:
        cast_variables += ['source.syz']

    # addition variables only used if there is a transducer source
    if flags.transducer_source:
        cast_variables += ['transducer_input_signal']
        cast_index_variables += ['u_source_pos_index', 'delay_mask', 'self.transducer_source', 'transducer_transmit_apodization']

    # addition variables only used if there is a transducer sensor with an elevation focus
    if flags.transducer_sensor and flags.transducer_receive_elevation_focus:
        cast_index_variables += ['sensor_data_buffer', 'transducer_receive_mask']

    # additional variables only used with nonuniform grids
    if flags.nonuniform_grid:
        if kgrid.dim == 1:
            cast_index_variables += ['kgrid.dxudxn']
        elif kgrid.dim == 2:
            cast_index_variables += ['kgrid.dxudxn', 'kgrid.dyudyn']
        elif kgrid.dim == 3:
            cast_index_variables += ['kgrid.dxudxn', 'kgrid.dyudyn', 'kgrid.dzudzn']

    # additional variables only used for Cartesian sensor masks with linear interpolation
    if flags.use_sensor and not flags.binary_sensor_mask and not flags.time_rev:
        if kgrid.dim == 1:
            cast_variables += ['record.grid_x', 'record.sensor_x']
        else:
            cast_variables += ['record.tri', 'record.bc']

    # additional variables only used in 2D if sensor directivity is defined
    if flags.compute_directivity:
        cast_variables += ['sensor.directivity_angle', 'sensor.directivity_unique_angles', 'sensor.directivity_wavenumbers']

    # cast variables
    for cast_var in cast_variables:
        if '.' in cast_var:
            part1, part2 = cast_var.split('.')
            subdict = getattr(self, part1)
            subdict[part2] = cast_to_type(subdict[part2], data_cast)
        else:
            setattr(self, cast_var, cast_to_type(getattr(self, cast_var), data_cast))

    # cast index variables only if casting to the GPU
    if data_cast.startswith('kWaveGPU'):
        raise NotImplementedError
