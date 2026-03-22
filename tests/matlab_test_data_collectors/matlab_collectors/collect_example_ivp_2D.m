% Collect reference data for 2D IVP integration test.
% Mirrors examples/new_api_ivp_2D.py and tests/integration/test_ivp_2D.py
output_file = 'collectedValues/example_ivp_2D.mat';
recorder = utils.TestRecorder(output_file);

% Grid
Nx = 128; Ny = 128;
dx = 0.1e-3; dy = 0.1e-3;
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.makeTime(1500);

% Medium
medium.sound_speed = 1500;
medium.density = 1000;

% Source: disc at center (1-indexed, same as Python make_disc which uses 1-indexed centers)
source.p0 = makeDisc(Nx, Ny, 64, 64, 5);

% Sensor: full grid
sensor.mask = ones(Nx, Ny);

% Run simulation with default PML settings
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', true);

% Record outputs
recorder.recordVariable('sensor_data_p', sensor_data.p);
recorder.recordVariable('sensor_data_p_final', sensor_data.p_final);
recorder.recordVariable('Nt', kgrid.Nt);
recorder.recordVariable('dt', kgrid.dt);

recorder.saveRecordsToDisk();
