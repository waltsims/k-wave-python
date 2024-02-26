clearvars;

% create the computational grid
Nx = 128;           % number of grid points in the x (row) direction
Ny = 128;           % number of grid points in the y (column) direction
dx = 0.1e-3;        % grid point spacing in the x direction [m]
dy = 0.1e-3;        % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]

% create initial pressure distribution using makeDisc
disc_magnitude = 5; % [Pa]
disc_x_pos = 50;    % [grid points]
disc_y_pos = 50;    % [grid points]
disc_radius = 8;    % [grid points]
disc_1 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

disc_magnitude = 3; % [Pa]
disc_x_pos = 80;    % [grid points]
disc_y_pos = 60;    % [grid points]
disc_radius = 5;    % [grid points]
disc_2 = disc_magnitude * makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

source.p0 = disc_1 + disc_2;

% define a centered circular sensor
sensor_radius = 4e-3;   % [m]
num_sensor_points = 50;
sensor.mask = makeCartCircle(sensor_radius, num_sensor_points);
sensor.mask = cart2grid(kgrid, sensor.mask, false);  % otherwise cpu computation will not work

% Example 1: PML with no absorption
input_args = {'PMLAlpha', 0};
run_simulation_and_record_input_output('example_1', kgrid, medium, source, sensor, input_args);

% Example 2: PML with the absorption value set too high
input_args = {'PMLAlpha', 1e6};
run_simulation_and_record_input_output('example_2', kgrid, medium, source, sensor, input_args);

% Example 3: partially effective PML
input_args = {'PMLSize', 2};
run_simulation_and_record_input_output('example_3', kgrid, medium, source, sensor, input_args);

% Example 4: PML set to be outside the computational domain
input_args = {'PMLInside', false};
run_simulation_and_record_input_output('example_4', kgrid, medium, source, sensor, input_args);


% define a new matlab function
function run_simulation_and_record_input_output(name, kgrid, medium, source, sensor, input_args)

    sensor_data = kspaceFirstOrder2DC(kgrid, medium, source, sensor, input_args{:});

    % plot the simulated sensor data
    figure;
    imagesc(sensor_data, [-1, 1]);
    colormap(getColorMap);
    ylabel('Sensor Position');
    xlabel('Time Step');
    colorbar;

    save(sprintf('%s.mat', name), 'sensor_data', 'input_args');

    % save the figure
    saveas(gcf, sprintf('%s.png', name));
end
