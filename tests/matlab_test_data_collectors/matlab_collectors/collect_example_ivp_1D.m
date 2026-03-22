% Collect reference data for 1D IVP integration test.
% Mirrors examples/ivp_1D_simulation.py and tests/integration/test_ivp_1D.py
output_file = 'collectedValues/example_ivp_1D.mat';
recorder = utils.TestRecorder(output_file);

% Grid
Nx = 512;
dx = 0.05e-3;
kgrid = kWaveGrid(Nx, dx);

% Heterogeneous medium
sound_speed = 1500 * ones(Nx, 1);
sound_speed(1:floor(Nx/3)) = 2000;

density = 1000 * ones(Nx, 1);
density(4*floor(Nx/5)+1:end) = 1500;

medium.sound_speed = sound_speed;
medium.density = density;

% Time stepping (CFL = 0.3)
% makeTime signature: makeTime(sound_speed, cfl, t_end)
kgrid.makeTime(sound_speed, 0.3);

% Source: smooth sinusoidal pulse
p0 = zeros(Nx, 1);
x0 = 281;  % MATLAB 1-indexed (Python uses 280, 0-indexed)
width = 100;
pulse = 0.5 * (sin((0:width) * pi / width - pi/2) + 1);
p0(x0:x0+width) = pulse;
source.p0 = p0;

% Sensor: two points
sensor_mask = zeros(Nx, 1);
sensor_mask(floor(Nx/4) + 1) = 1;    % MATLAB 1-indexed (Python Nx//4 = 128, 0-indexed)
sensor_mask(3*floor(Nx/4) + 1) = 1;  % Python 3*Nx//4 = 384, 0-indexed
sensor.mask = sensor_mask;

% Run
sensor_data = kspaceFirstOrder1D(kgrid, medium, source, sensor, ...
    'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', true);

% Record outputs
recorder.recordVariable('sensor_data_p', sensor_data.p);
recorder.recordVariable('Nt', kgrid.Nt);
recorder.recordVariable('dt', kgrid.dt);

recorder.saveRecordsToDisk();
