from numpy.fft import ifftshift
from kwave_py.kWaveSimulation import kWaveSimulation
from kwave_py.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave_py.kspaceFirstOrder import *
from kwave_py.utils import *


@kspaceFirstOrderG
def kspaceFirstOrder1DG(**kwargs):  # pragma: no cover
    sensor_data = kspaceFirstOrder1DC(**kwargs)  # pass inputs to CPU version
    return sensor_data


@kspaceFirstOrderC()
def kspaceFirstOrder1DC(**kwargs):  # pragma: no cover
    # generate the input file and save to disk
    kspaceFirstOrder1D(**kwargs)
    return kwargs['SaveToDisk']


def kspaceFirstOrder1D(kgrid, medium, source, sensor, **kwargs):  # pragma: no cover
    # start the timer and store the start time
    TicToc.tic()

    k_sim = kWaveSimulation(kgrid, medium, source, sensor, **kwargs)
    k_sim.input_checking('kspaceFirstOrder1D')

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)
    if len(k_sim.rho0) > 1 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x]
        k_sim.rho0_sgx = interpolate3D(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the PML operators based on the reference sound speed and PML settings
    Nx, Ny = k_sim.kgrid.Nx, k_sim.kgrid.Ny
    dx, dy = k_sim.kgrid.dx, k_sim.kgrid.dy
    dt = k_sim.kgrid.dt
    pml_x_alpha = options.pml_x_alpha
    pml_x_size = options.pml_x_size
    c_ref = k_sim.c_ref

    k_sim.pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 1)
    k_sim.pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True and options.use_sg, 1)

    # define the k-space derivative operator
    kx_vec = np.array(k_sim.kgrid.k_vec.x)
    k_sim.ddx_k = ifftshift(1j * kx_vec)
    if options.use_sg:
        k_sim.shift_pos = np.fft.ifftshift(np.exp(  1j * kx_vec * dx/2))[None, :]
        k_sim.shift_neg = np.fft.ifftshift(np.exp( -1j * kx_vec * dx/2))[None, :]
    else:
        k_sim.shift_pos = np.ones(1)
        k_sim.shift_neg = np.ones(1)

    # create k-space operators (the option flgs.use_kspace exists for debugging)
    if options.use_kspace:
        k = k_sim.kgrid.k
        k_sim.kappa = np.fft.ifftshift(np.sinc(c_ref * k * dt / 2))
        if (k_sim.source_p and k_sim.source.p_mode == 'additive') or ((k_sim.source_ux or k_sim.source_uy or k_sim.source_uz) and k_sim.source.u_mode == 'additive'):
            k_sim.source_kappa = np.fft.ifftshift(np.cos(c_ref * k * dt / 2))
    else:
        k_sim.kappa          = 1
        k_sim.source_kappa   = 1

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
