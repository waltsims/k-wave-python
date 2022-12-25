output_file = 'collectedValues/kWaveGrid.mat';
test_step_idx = 0;
test_var_expectations = {};


Nx = 10;
dx = 0.1;
Ny = 14;
dy = 0.05;
Nz = 9;
dz = 0.13;
recorder = utils.TestRecorder(output_file);

for dim = 1:3
    disp(dim)

    if dim == 1
        kgrid = kWaveGrid(Nx, dx);
    elseif dim == 2
        kgrid = kWaveGrid(Nx, dx, Ny, dy);
    else
        kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
    end

    recorder.recordObject('kgrid', kgrid);
    recorder.increment();

    kgrid.setTime(52, 0.0001);
    recorder.recordObject('kgrid', kgrid);
    recorder.increment();

    [t_array, dt] = kgrid.makeTime(1596);
    recorder.recordObject('kgrid', kgrid);
    recorder.recordVariable('returned_t_array', t_array);
    recorder.recordVariable('returned_dt', dt);
    recorder.increment();


    for ii = 1:8
        [k, M] = kgrid.k_dtt(ii);
        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('returned_k', k);
        recorder.recordVariable('returned_M', M);
        recorder.increment();

        [kx_vec_dtt, M] = kgrid.kx_vec_dtt(ii);
        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('returned_kx_vec_dtt', kx_vec_dtt);
        recorder.recordVariable('returned_M', M);
        recorder.increment();

        [ky_vec_dtt, M] = kgrid.ky_vec_dtt(ii);
        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('returned_ky_vec_dtt', ky_vec_dtt);
        recorder.recordVariable('returned_M', M);
        recorder.increment();

        [kz_vec_dtt, M] = kgrid.kz_vec_dtt(ii);
        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('returned_kz_vec_dtt', kz_vec_dtt);
        recorder.recordVariable('returned_M', M);
        recorder.increment();
    end

    highest_prime_factors = kgrid.highest_prime_factors();
    recorder.recordObject('kgrid', kgrid);
    recorder.recordVariable('returned_highest_prime_factors', highest_prime_factors);
    recorder.increment();

    axisymmetric_options = {'WSWA', 'WSWS'};
    for ii = 1:numel(axisymmetric_options)
        axisymmetric = axisymmetric_options{ii};

        if strcmp(axisymmetric, 'WSWS') && dim == 1
            continue
        end

        highest_prime_factors = kgrid.highest_prime_factors(axisymmetric);
        recorder.recordObject('kgrid', kgrid);
        recorder.recordVariable('returned_highest_prime_factors', highest_prime_factors);
        recorder.increment();
    end

    inp_xn_vec = rand(3, 2);
    inp_dxudxn = rand(4, 7);
    inp_xn_vec_sgx = rand(7, 5);
    inp_dxudxn_sgx = rand(3, 4);
    recorder.recordVariable('inp_xn_vec', inp_xn_vec);
    recorder.recordVariable('inp_dxudxn', inp_dxudxn);
    recorder.recordVariable('inp_xn_vec_sgx', inp_xn_vec_sgx);
    recorder.recordVariable('inp_dxudxn_sgx', inp_dxudxn_sgx);
    kgrid.setNUGrid(dim, inp_xn_vec, inp_dxudxn, inp_xn_vec_sgx, inp_dxudxn_sgx);
    recorder.recordObject('kgrid', kgrid);
    recorder.increment();
end

recorder.saveRecordsToDisk();
