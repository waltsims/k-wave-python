output_file = 'collectedValues/kWaveGrid.mat';
test_step_idx = 0;
test_var_expectations = {};

Nx = 10;
dx = 0.1;
kgrid = kWaveGrid(Nx, dx);

recorder = TestRecorder(output_file);
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

[k, M] = kgrid.k_dtt(1);
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_k', k);
recorder.recordVariable('returned_M', M);
recorder.increment();

[kx_vec_dtt, M] = kgrid.kx_vec_dtt(1);
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_kx_vec_dtt', kx_vec_dtt);
recorder.recordVariable('returned_M', M);
recorder.increment();

[ky_vec_dtt, M] = kgrid.ky_vec_dtt(1);
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_ky_vec_dtt', ky_vec_dtt);
recorder.recordVariable('returned_M', M);
recorder.increment();

[kz_vec_dtt, M] = kgrid.kz_vec_dtt(1);
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_kz_vec_dtt', kz_vec_dtt);
recorder.recordVariable('returned_M', M);
recorder.increment();

highest_prime_factors = kgrid.highest_prime_factors('WSWA');
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_highest_prime_factors', highest_prime_factors);
recorder.increment();

inp_xn_vec = rand(3, 2);
inp_dxudxn = rand(4, 7);
inp_xn_vec_sgx = rand(7, 5);
inp_dxudxn_sgx = rand(3, 4);
recorder.recordVariable('inp_xn_vec', inp_xn_vec);
recorder.recordVariable('inp_dxudxn', inp_dxudxn);
recorder.recordVariable('inp_xn_vec_sgx', inp_xn_vec_sgx);
recorder.recordVariable('inp_dxudxn_sgx', inp_dxudxn_sgx);
kgrid.setNUGrid(1, inp_xn_vec, inp_dxudxn, inp_xn_vec_sgx, inp_dxudxn_sgx);
recorder.recordObject('kgrid', kgrid);
recorder.saveRecordsToDisk();
