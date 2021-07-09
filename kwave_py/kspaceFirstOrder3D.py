from numpy.fft import ifftshift

from kwave_py.kWaveSimulation import kWaveSimulation
from kwave_py.kWaveSimulation_helper import retract_transducer_grid_size, save_to_disk_func
from kwave_py.kspaceFirstOrder import *
from kwave_py.utils import *


@kspaceFirstOrderG
def kspaceFirstOrder3DG(**kwargs):
    sensor_data = kspaceFirstOrder3DC(**kwargs)  # pass inputs to CPU version
    return sensor_data


@kspaceFirstOrderC()
def kspaceFirstOrder3DC(**kwargs):
    # generate the input file and save to disk
    kspaceFirstOrder3D(**kwargs)
    return kwargs['SaveToDisk']


def kspaceFirstOrder3D(kgrid, medium, source, sensor, **kwargs):
    # start the timer and store the start time
    TicToc.tic()

    k_sim = kWaveSimulation(kgrid, medium, source, sensor, **kwargs)
    k_sim.input_checking('kspaceFirstOrder3D')

    # =========================================================================
    # CALCULATE MEDIUM PROPERTIES ON STAGGERED GRID
    # =========================================================================
    options = k_sim.options

    # interpolate the values of the density at the staggered grid locations
    # where sgx = (x + dx/2, y, z), sgy = (x, y + dy/2, z), sgz = (x, y, z + dz/2)
    k_sim.rho0 = np.atleast_1d(k_sim.rho0)
    if k_sim.rho0.ndim == 3 and options.use_sg:
        # rho0 is heterogeneous and staggered grids are used
        grid_points = [k_sim.kgrid.x, k_sim.kgrid.y, k_sim.kgrid.z]
        k_sim.rho0_sgx = interpolate3D(grid_points, k_sim.rho0, [k_sim.kgrid.x + k_sim.kgrid.dx / 2, k_sim.kgrid.y, k_sim.kgrid.z])
        k_sim.rho0_sgy = interpolate3D(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y + k_sim.kgrid.dy / 2, k_sim.kgrid.z])
        k_sim.rho0_sgz = interpolate3D(grid_points, k_sim.rho0, [k_sim.kgrid.x, k_sim.kgrid.y, k_sim.kgrid.z + k_sim.kgrid.dz / 2])
    else:
        # rho0 is homogeneous or staggered grids are not used
        k_sim.rho0_sgx = k_sim.rho0
        k_sim.rho0_sgy = k_sim.rho0
        k_sim.rho0_sgz = k_sim.rho0

    # invert rho0 so it doesn't have to be done each time step
    k_sim.rho0_sgx_inv = 1 / k_sim.rho0_sgx
    k_sim.rho0_sgy_inv = 1 / k_sim.rho0_sgy
    k_sim.rho0_sgz_inv = 1 / k_sim.rho0_sgz

    # clear unused variables if not using them in _saveToDisk
    if not options.save_to_disk:
        del k_sim.rho0_sgx
        del k_sim.rho0_sgy
        del k_sim.rho0_sgz

    # =========================================================================
    # PREPARE DERIVATIVE AND PML OPERATORS
    # =========================================================================

    # get the PML operators based on the reference sound speed and PML settings
    Nx, Ny, Nz = k_sim.kgrid.Nx, k_sim.kgrid.Ny, k_sim.kgrid.Nz
    dx, dy, dz = k_sim.kgrid.dx, k_sim.kgrid.dy, k_sim.kgrid.dz
    dt = k_sim.kgrid.dt
    pml_x_alpha, pml_y_alpha, pml_z_alpha = options.pml_x_alpha, options.pml_y_alpha, options.pml_z_alpha
    pml_x_size, pml_y_size, pml_z_size = options.pml_x_size, options.pml_y_size, options.pml_z_size
    c_ref = k_sim.c_ref

    k_sim.pml_x     = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, False, 1)
    k_sim.pml_x_sgx = get_pml(Nx, dx, dt, c_ref, pml_x_size, pml_x_alpha, True and options.use_sg, 1)
    k_sim.pml_y     = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, False, 2)
    k_sim.pml_y_sgy = get_pml(Ny, dy, dt, c_ref, pml_y_size, pml_y_alpha, True and options.use_sg, 2)
    k_sim.pml_z     = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, False, 3)
    k_sim.pml_z_sgz = get_pml(Nz, dz, dt, c_ref, pml_z_size, pml_z_alpha, True and options.use_sg, 3)

    # define the k-space derivative operators, multiply by the staggered
    # grid shift operators, and then re-order using ifftshift (the option
    # flgs.use_sg exists for debugging)
    kx_vec, ky_vec, kz_vec = k_sim.kgrid.k_vec
    kx_vec, ky_vec, kz_vec = np.array(kx_vec), np.array(ky_vec), np.array(kz_vec)
    if options.use_sg:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift( 1j * kx_vec * np.exp( 1j * kx_vec * dx/2) ).T
        k_sim.ddx_k_shift_neg = np.fft.ifftshift( 1j * kx_vec * np.exp( -1j * kx_vec * dx/2) ).T
        k_sim.ddy_k_shift_pos = np.fft.ifftshift( 1j * ky_vec * np.exp( 1j * ky_vec * dy/2) ).T
        k_sim.ddy_k_shift_neg = np.fft.ifftshift( 1j * ky_vec * np.exp( -1j * ky_vec * dy/2) ).T
        k_sim.ddz_k_shift_pos = np.fft.ifftshift( 1j * kz_vec * np.exp( 1j * kz_vec * dz/2) ).T
        k_sim.ddz_k_shift_neg = np.fft.ifftshift( 1j * kz_vec * np.exp( -1j * kz_vec * dz/2) ).T
    else:
        k_sim.ddx_k_shift_pos = np.fft.ifftshift( 1j * kx_vec ).T
        k_sim.ddx_k_shift_neg = np.fft.ifftshift( 1j * kx_vec ).T
        k_sim.ddy_k_shift_pos = np.fft.ifftshift( 1j * ky_vec ).T
        k_sim.ddy_k_shift_neg = np.fft.ifftshift( 1j * ky_vec ).T
        k_sim.ddz_k_shift_pos = np.fft.ifftshift( 1j * kz_vec ).T
        k_sim.ddz_k_shift_neg = np.fft.ifftshift( 1j * kz_vec ).T

    # force the derivative and shift operators to be in the correct direction for use with BSXFUN
    k_sim.ddy_k_shift_pos = k_sim.ddy_k_shift_pos.T
    k_sim.ddy_k_shift_neg = k_sim.ddy_k_shift_neg.T

    ddz_k_shift_pos = k_sim.ddz_k_shift_pos  # N x 1
    ddz_k_shift_pos = np.expand_dims(ddz_k_shift_pos, axis=-1).transpose((1, 2, 0))
    k_sim.ddz_k_shift_pos = ddz_k_shift_pos

    ddz_k_shift_neg = k_sim.ddz_k_shift_neg  # N x 1
    ddz_k_shift_neg = np.expand_dims(ddz_k_shift_neg, axis=-1).transpose((1, 2, 0))
    k_sim.ddz_k_shift_neg = ddz_k_shift_neg

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
