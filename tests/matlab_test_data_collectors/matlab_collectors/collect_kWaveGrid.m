output_file = 'collectedValues/kWaveGrid.mat';
test_step_idx = 0;
test_var_expectations = {};

Nx = 10;
dx = 0.1;
kgrid = kWaveGrid(Nx, dx);

recorder = TestRecorder(output_file);
recorder.recordObject('kgrid', kgrid);
recorder.increment();
% recorder.recordExpectedValue('Nx', kgrid.Nx);
% recorder.recordExpectedValue('dx', kgrid.dx);
% recorder.recordExpectedValue('kx_vec', kgrid.kx_vec);
% recorder.recordExpectedValue('k', kgrid.k);
% recorder.recordExpectedValue('kx_max', kgrid.kx_max);
% recorder.recordExpectedValue('k_max', kgrid.k_max);
% recorder.recordExpectedValue('x', kgrid.x);
% recorder.recordExpectedValue('y', kgrid.y);
% recorder.recordExpectedValue('z', kgrid.z);
% recorder.recordExpectedValue('kx', kgrid.kx);
% recorder.recordExpectedValue('x_vec', kgrid.x_vec);
% recorder.recordExpectedValue('y_vec', kgrid.y_vec);
% recorder.recordExpectedValue('z_vec', kgrid.z_vec);
% recorder.recordExpectedValue('x_size', kgrid.x_size);
% recorder.recordExpectedValue('y_size', kgrid.y_size);
% recorder.recordExpectedValue('z_size', kgrid.z_size);
% recorder.recordExpectedValue('t_array', kgrid.t_array);
% recorder.recordExpectedValue('total_grid_points', kgrid.total_grid_points);


kgrid.setTime(52, 0.0001);
recorder.recordObject('kgrid', kgrid);
recorder.increment();
% recorder.recordExpectedValue('Nt', kgrid.Nt);
% recorder.recordExpectedValue('dt', kgrid.dt);
% recorder.recordExpectedValue('t_array', kgrid.t_array);

% recorder.increment();
[t_array, dt] = kgrid.makeTime(1596);
recorder.recordObject('kgrid', kgrid);
recorder.recordVariable('returned_t_array', t_array);
recorder.recordVariable('returned_dt', dt);
recorder.increment();

% recorder.recordExpectedValue('Nt', kgrid.Nt);
% recorder.recordExpectedValue('dt', kgrid.dt);
% recorder.recordExpectedValue('t_array', kgrid.t_array);

recorder.saveRecordsToDisk();

return


%%%%% UNTIL HERE
[k, M] = kgrid.k_dtt(1);
recorder.recordExpectedValue('returned_k', k);
recorder.recordExpectedValue('returned_M', M);

recorder.increment();
[kx_vec_dtt, M] = kgrid.kx_vec_dtt(1);
recorder.recordExpectedValue('returned_kx_vec_dtt', kx_vec_dtt);
recorder.recordExpectedValue('returned_M', M);

recorder.increment();
[ky_vec_dtt, M] = kgrid.ky_vec_dtt(1);
recorder.recordExpectedValue('returned_ky_vec_dtt', ky_vec_dtt);
recorder.recordExpectedValue('returned_M', M);

recorder.increment();
[kz_vec_dtt, M] = kgrid.kz_vec_dtt(1);
recorder.recordExpectedValue('returned_kz_vec_dtt', kz_vec_dtt);
recorder.recordExpectedValue('returned_M', M);

recorder.increment();
highest_prime_factors = kgrid.highest_prime_factors('WSWA');
recorder.recordExpectedValue('returned_highest_prime_factors', highest_prime_factors);

recorder.increment();
recorder.recordExpectedValue('xn', kgrid.xn);
recorder.recordExpectedValue('xn_vec', kgrid.xn_vec);
recorder.recordExpectedValue('yn', kgrid.yn);
recorder.recordExpectedValue('yn_vec', kgrid.yn_vec);
recorder.recordExpectedValue('zn', kgrid.zn);
recorder.recordExpectedValue('zn_vec', kgrid.zn_vec);

recorder.increment();
inp_xn_vec = rand(3, 2);
inp_dxudxn = rand(4, 7);
inp_xn_vec_sgx = rand(7, 5);
inp_dxudxn_sgx = rand(3, 4);
kgrid.setNUGrid(1, inp_xn_vec, inp_dxudxn, inp_xn_vec_sgx, inp_dxudxn_sgx);

recorder.recordExpectedValue('inp_xn_vec', inp_xn_vec);
recorder.recordExpectedValue('inp_dxudxn', inp_dxudxn);
recorder.recordExpectedValue('inp_xn_vec_sgx', inp_xn_vec_sgx);
recorder.recordExpectedValue('inp_dxudxn_sgx', inp_dxudxn_sgx);


recorder.recordExpectedValue('xn_vec', kgrid.xn_vec);
recorder.recordExpectedValue('dxudxn', kgrid.dxudxn);
recorder.recordExpectedValue('xn_vec_sgx', kgrid.xn_vec_sgx);
recorder.recordExpectedValue('dxudxn_sgx', kgrid.dxudxn_sgx);
recorder.recordExpectedValue('nonuniform', kgrid.nonuniform);

recorder.saveRecordsToDisk();
