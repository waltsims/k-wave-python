output_file = 'collectedValues/angularSpectrumCW.mat';


recorder = utils.TestRecorder(output_file);

input_plane = rand(44, 100);
dx = 1.0000e-04;
z_pos = 0.0034;
f0 = 300000;
c0 = 1500;
grid_expansion = 50;

pressure = angularSpectrumCW(input_plane, dx, ...
    z_pos, f0, c0, 'GridExpansion', grid_expansion);

recorder.recordVariable('input_plane', input_plane);
recorder.recordVariable('dx', dx);
recorder.recordVariable('z_pos', z_pos);
recorder.recordVariable('f0', f0);
recorder.recordVariable('c0', c0);
recorder.recordVariable('grid_expansion', grid_expansion);

recorder.recordVariable('pressure', pressure);

recorder.saveRecordsToDisk();
