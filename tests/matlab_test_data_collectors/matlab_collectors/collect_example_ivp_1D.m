% Collect reference data for 1D IVP integration test.
% Homogeneous medium, single-point impulse, two-point sensor.
output_file = 'collectedValues/example_ivp_1D.mat';
recorder = utils.TestRecorder(output_file);

% Grid
Nx = 256;
dx = 0.1e-3;
kgrid = kWaveGrid(Nx, dx);
kgrid.makeTime(1500);

% Homogeneous medium
medium.sound_speed = 1500;
medium.density = 1000;

% Source: single-point impulse at center
source.p0 = zeros(Nx, 1);
source.p0(Nx/2 + 1) = 1.0;  % MATLAB 1-indexed: Nx/2+1 = 129 = Python's Nx//2 = 128

% Sensor: two points
sensor.mask = zeros(Nx, 1);
sensor.mask(Nx/4 + 1) = 1;    % Python: Nx//4 = 64
sensor.mask(3*Nx/4 + 1) = 1;  % Python: 3*Nx//4 = 192
sensor.record = {'p'};

% Run (Smooth=false to match Python smooth_p0=False)
sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor, ...
    'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', false);

% Record outputs
recorder.recordVariable('sensor_data_p', sensor_data.p);
recorder.recordVariable('Nt', kgrid.Nt);
recorder.recordVariable('dt', kgrid.dt);

recorder.saveRecordsToDisk();
