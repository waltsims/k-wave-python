output_file = 'collectedValues/angularSpectrum.mat';
recorder = utils.TestRecorder(output_file);

input_plane = rand(44, 44, 255);
dx = 1.0000e-04;
dt = 2.0059e-08;
z_pos = 0.0034;
c0 = 1500;
grid_expansion = 50;

[pressure_max, pressure_time] = angularSpectrum(input_plane, dx, ...
    dt, z_pos, c0, 'GridExpansion', grid_expansion);

recorder.recordVariable('input_plane', input_plane);
recorder.recordVariable('dx', dx);
recorder.recordVariable('dt', dt);
recorder.recordVariable('z_pos', z_pos);
recorder.recordVariable('c0', c0);
recorder.recordVariable('grid_expansion', grid_expansion);

recorder.recordVariable('pressure_max', pressure_max);
recorder.recordVariable('pressure_time', pressure_time);

recorder.saveRecordsToDisk();
