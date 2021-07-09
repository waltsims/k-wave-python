from kwave_py.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave_py.kspaceFirstOrder import *
from kwave_py.kWaveSimulation import kWaveSimulation
from kwave_py.utils import *
from kwave_py.enums import DiscreteCosine


@kspaceFirstOrderC()
def kspaceFirstOrderASC(**kwargs):
    # generate the input file and save to disk
    kspaceFirstOrderAS(**kwargs)
    return kwargs['SaveToDisk']


def kspaceFirstOrderAS(kgrid, medium, source, sensor, **kwargs):
    # start the timer and store the start time
    TicToc.tic()

    k_sim = kWaveSimulation(kgrid, medium, source, sensor, **kwargs)
    k_sim.input_checking('kspaceFirstOrderAS')

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)
    k_sim.rho0 = np.atleast_1d(k_sim.rho0)
    if num_dim2(k_sim.rho0) == 2 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x, k_sim.kgrid.y]
        k_sim.rho0_sgx = interpolate2D(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2, k_sim.kgrid.y])
        k_sim.rho0_sgy = interpolate2D(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y + k_sim.kgrid.dy / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0
        k_sim.rho0_sgy = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx
    k_sim.rho0_sgy_inv = 1 / k_sim.rho0_sgy

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx
        del k_sim.rho0_sgy
    k_sim.rho0_sgz = None

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the PML operators based on the reference sound speed and PML settings
    Nx, Ny = k_sim.kgrid.Nx, k_sim.kgrid.Ny
    dx, dy = k_sim.kgrid.dx, k_sim.kgrid.dy
    dt = k_sim.kgrid.dt
    pml_x_alpha, pml_y_alpha = options.pml_x_alpha, options.pml_y_alpha
    pml_x_size, pml_y_size = options.pml_x_size, options.pml_y_size
    c_ref = k_sim.c_ref

    k_sim.pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False,                 1, False)
    k_sim.pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True and options.use_sg, 1, False)
    k_sim.pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False,                 2, True)
    k_sim.pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True and options.use_sg, 2, True)

    # define the k-space, derivative, and shift operators
    # for the x (axial) direction, the operators are the same as normal
    kx_vec = k_sim.kgrid.k_vec.x
    k_sim.ddx_k_shift_pos = ifftshift( 1j * kx_vec * np.exp( 1j * kx_vec * dx/2) ).T
    k_sim.ddx_k_shift_neg = ifftshift( 1j * kx_vec * np.exp(-1j * kx_vec * dx/2) ).T

    # for the y (radial) direction
    # when using DTTs:
    #    - there is no explicit grid shift (this is done by choosing DTTs
    #      with the appropriate symmetry)
    #    - ifftshift isn't required as the wavenumbers start from DC
    # when using FFTs:
    #    - the grid is expanded, and the fields replicated in the radial
    #      dimension to give the required symmetry
    #    - the derivative and shift operators are defined as normal
    if options.radial_symmetry in ['WSWA-FFT', 'WSWS-FFT']:
        # create a new kWave grid object with expanded radial grid
        if options.radial_symmetry == 'WSWA-FFT':
            # extend grid by a factor of x4 to account for
            # symmetries in WSWA
            kgrid_exp = kWaveGrid([Nx, Ny * 4], [dx, dy])
        elif options.radial_symmetry == 'WSWS-FFT':
            # extend grid by a factor of x2 - 2 to account for
            # symmetries in WSWS
            kgrid_exp = kWaveGrid([Nx, Ny * 2 - 2], [dx, dy])
        # define operators, rotating y-direction for use with bsxfun
        k_sim.ddy_k       = ifftshift( 1j * options.k_vec.y ).T
        k_sim.y_shift_pos = ifftshift( np.exp( 1j * kgrid_exp.k_vec.y * kgrid_exp.dy/2) ).T
        k_sim.y_shift_neg = ifftshift( np.exp(-1j * kgrid_exp.k_vec.y * kgrid_exp.dy/2) ).T

        # define the k-space operator
        if options.use_kspace:
            k_sim.kappa = ifftshift(sinc(c_ref * kgrid_exp.k * dt / 2))
            if (k_sim.source_p and (k_sim.source.p_mode == 'additive')) or ((k_sim.source_ux or k_sim.source_uy) and (k_sim.source.u_mode == 'additive')):
                k_sim.source_kappa = ifftshift(np.cos (c_ref * kgrid_exp.k * dt / 2))
        else:
            k_sim.kappa = 1
            k_sim.source_kappa = 1
    elif options.radial_symmetry in ['WSWA', 'WSWS']:
        if options.radial_symmetry == 'WSWA':
            # get the wavenumbers and implied length for the DTTs
            ky_vec, M = k_sim.kgrid.ky_vec_dtt(DiscreteCosine.TYPE_3)

            # define the derivative operators
            k_sim.ddy_k_wswa = -ky_vec.T
            k_sim.ddy_k_hahs =  ky_vec.T
        elif options.radial_symmetry == 'WSWS':
            # get the wavenumbers and implied length for the DTTs
            ky_vec, M = k_sim.kgrid.ky_vec_dtt(DiscreteCosine.TYPE_1)

            # define the derivative operators
            k_sim.ddy_k_wsws = -ky_vec[1:].T
            k_sim.ddy_k_haha =  ky_vec[1:].T

        # define the k-space operator
        if options.use_kspace:
            # define scalar wavenumber
            k_dtt = np.sqrt(np.tile(ifftshift(k_sim.kgrid.k_vec.x)**2, [1, k_sim.kgrid.Ny]) + np.tile((ky_vec.T)**2, [k_sim.kgrid.Nx, 1]))

            # define k-space operators
            k_sim.kappa = sinc(c_ref * k_dtt * k_sim.kgrid.dt / 2)
            if (k_sim.source_p and (k_sim.source.p_mode == 'additive')) or ((k_sim.source_ux or k_sim.source_uy) and (k_sim.source.u_mode == 'additive')):
                k_sim.source_kappa = np.cos(c_ref * k_dtt * k_sim.kgrid.dt / 2)

            # cleanup unused variables
            del k_dtt

        else:
            k_sim.kappa = 1
            k_sim.source_kappa = 1

    # define staggered and non-staggered grid axial distance
    k_sim.y_vec    = (k_sim.kgrid.y_vec - k_sim.kgrid.y_vec[0]).T
    k_sim.y_vec_sg = (k_sim.kgrid.y_vec - k_sim.kgrid.y_vec[0] + k_sim.kgrid.dy/2).T

    # option to run simulations without the spatial staggered grid is not
    # supported for the axisymmetric code
    assert options.use_sg, 'Optional input ''UseSG'' is not supported for axisymmetric simulations.'

    # =========================================================================
    # SAVE DATA TO DISK FOR RUNNING SIMULATION EXTERNAL TO MATLAB
    # =========================================================================

    # save to disk option for saving the input matrices to disk for running
    # simulations using k-Wave++
    if options.save_to_disk:
        # store the pml size for resizing transducer object below
        retract_size = [[options.pml_x_size, options.pml_y_size, options.pml_z_size]]

        # run subscript to save files to disk
        save_to_disk_func(k_sim.kgrid, k_sim.medium, k_sim.source, k_sim.options,
                          dotdict({
                              'ddx_k_shift_pos': k_sim.ddx_k_shift_pos,
                              'ddx_k_shift_neg': k_sim.ddx_k_shift_neg,
                              'dt': k_sim.dt,
                              'c0': k_sim.c0,
                              'c_ref': k_sim.c_ref,
                              'rho0': k_sim.rho0,
                              'rho0_sgx': k_sim.rho0_sgx,
                              'rho0_sgy': k_sim.rho0_sgy,
                              'rho0_sgz': k_sim.rho0_sgz,
                              'p_source_pos_index': k_sim.p_source_pos_index,
                              'u_source_pos_index': k_sim.u_source_pos_index,
                              's_source_pos_index': k_sim.s_source_pos_index,
                              'transducer_input_signal': k_sim.transducer_input_signal,
                              'delay_mask': k_sim.delay_mask,
                              'sensor_mask_index': k_sim.sensor_mask_index,
                              'record': k_sim.record,
                          }),
                          dotdict({
                              'source_p': k_sim.source_p,
                              'source_p0': k_sim.source_p0,

                              'source_ux': k_sim.source_ux,
                              'source_uy': k_sim.source_uy,
                              'source_uz': k_sim.source_uz,

                              'source_sxx': k_sim.source_sxx,
                              'source_syy': k_sim.source_syy,
                              'source_szz': k_sim.source_szz,
                              'source_sxy': k_sim.source_sxy,
                              'source_sxz': k_sim.source_sxz,
                              'source_syz': k_sim.source_syz,

                              'transducer_source': k_sim.transducer_source,
                              'nonuniform_grid': k_sim.nonuniform_grid,
                              'elastic_code': k_sim.elastic_code,
                              'axisymmetric': k_sim.axisymmetric,
                              'cuboid_corners': k_sim.cuboid_corners,
                          }))

        # run subscript to resize the transducer object if the grid has been expanded
        retract_transducer_grid_size(k_sim.source, k_sim.sensor, retract_size, k_sim.options.pml_inside)

        # exit matlab computation if required
        if options.save_to_disk_exit:
            return
