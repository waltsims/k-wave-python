% Collect step-by-step intermediate values for 2D parity debugging.
% Compares Python solver internals against MATLAB k-Wave.
% Uses a small grid with no smoothing for exact comparison.
output_file = 'collectedValues/example_parity_2D.mat';
recorder = utils.TestRecorder(output_file);

% Small grid for fast debugging
Nx = 64; Ny = 64;
dx = 0.1e-3; dy = 0.1e-3;
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.makeTime(1500);

medium.sound_speed = 1500;
medium.density = 1000;

% Single-point source at center (no smoothing ambiguity)
source.p0 = zeros(Nx, Ny);
source.p0(Nx/2 + 1, Ny/2 + 1) = 1.0;

% Full-grid sensor
sensor.mask = ones(Nx, Ny);
sensor.record = {'p'};

% Record grid parameters
recorder.recordVariable('Nt', kgrid.Nt);
recorder.recordVariable('dt', kgrid.dt);
recorder.recordVariable('c_ref', max(medium.sound_speed(:)));

% Record k-space vectors (MATLAB stores these internally)
kx = kgrid.kx;
ky = kgrid.ky;
recorder.recordVariable('kx', kx);
recorder.recordVariable('ky', ky);

% Record PML operators (from kgrid internals)
% MATLAB stores these as pml_x, pml_y, pml_x_sgx, pml_y_sgy
% We need to access them via the simulation function's internal variables.
% Instead, run 3 steps and record p, ux, uy at each step.

% Run simulation for just 3 steps to compare
input_args = {'PMLInside', true, 'PMLSize', 20, 'PMLAlpha', 2, 'Smooth', false};
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% Record full sensor time-series
recorder.recordVariable('sensor_data_p', sensor_data.p);

% Now run again with RecordMovie-style step access
% Actually, we can use 'ReturnVelocity' to get velocity fields too
sensor2 = sensor;
sensor2.record = {'p', 'u'};
sensor_data2 = kspaceFirstOrder2D(kgrid, medium, source, sensor2, input_args{:});
recorder.recordVariable('sensor_data_ux', sensor_data2.ux);
recorder.recordVariable('sensor_data_uy', sensor_data2.uy);

% Record initial p0 as passed (no smoothing)
recorder.recordVariable('p0', source.p0);

recorder.saveRecordsToDisk();
