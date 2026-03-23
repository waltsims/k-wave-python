% Collect reference data for photoacoustic waveforms integration test.
% Mirrors examples/ivp_photoacoustic_waveforms and tests/integration/test_photoacoustic.py
output_file = 'collectedValues/example_photoacoustic.mat';
recorder = utils.TestRecorder(output_file);

Nx = 64;
dx = 1e-3 / Nx;
source_radius = 2;
source_sensor_distance = 10;
dt = 2e-9;
t_end = 300e-9;

%% 2D simulation
kgrid2 = kWaveGrid(Nx, dx, Nx, dx);
Nt = round(t_end / dt);
kgrid2.setTime(Nt, dt);

medium2.sound_speed = 1500;
medium2.density = 1000;

source2.p0 = makeDisc(Nx, Nx, Nx/2, Nx/2, source_radius);

sensor2.mask = zeros(Nx, Nx);
sensor2.mask(Nx/2 + source_sensor_distance, Nx/2) = 1;
sensor2.record = {'p'};

sensor_data_2D = kspaceFirstOrder2D(kgrid2, medium2, source2, sensor2, ...
    'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', true);

recorder.recordVariable('sensor_data_2D_p', sensor_data_2D.p);

%% 3D simulation
kgrid3 = kWaveGrid(Nx, dx, Nx, dx, Nx, dx);
kgrid3.setTime(Nt, dt);

medium3.sound_speed = 1500;
medium3.density = 1000;

source3.p0 = makeBall(Nx, Nx, Nx, Nx/2, Nx/2, Nx/2, source_radius);

sensor3.mask = zeros(Nx, Nx, Nx);
sensor3.mask(Nx/2 + source_sensor_distance, Nx/2, Nx/2) = 1;
sensor3.record = {'p'};

sensor_data_3D = kspaceFirstOrder3D(kgrid3, medium3, source3, sensor3, ...
    'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', true);

recorder.recordVariable('sensor_data_3D_p', sensor_data_3D.p);

recorder.saveRecordsToDisk();
