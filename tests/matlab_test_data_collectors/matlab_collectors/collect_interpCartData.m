all_params = {
    {1600, 10, 64, 0.2e-3},...
    {1540, 20, 128, 0.1e-3},...
    {1500, 40, 300, 0.5e-4}
    };

output_folder = 'collectedValues/interpCartData/';

k = 0;
for idx=1:length(all_params)
    for dim=2:3
        disp(k);
        params = all_params{idx};
        % create the computational grid
        PML_size = params{2};               % size of the PML in grid points
        Nx = params{3} - 2 * PML_size;      % number of grid points in the x direction
        Ny = Nx;                            % number of grid points in the y direction
        Nz = Nx;                            % number of grid points in the z direction
        dx = params{4};                     % grid point spacing in the x direction [m]
        dy = dx;                            % grid point spacing in the y direction [m]
        dz = dx;                            % grid point spacing in the z direction [m]

        % create initial pressure distribution using makeBall
        ball_magnitude = 10;        % [Pa]
        ball_x_pos = 16;            % [grid points]
        ball_y_pos = 26;            % [grid points]
        ball_z_pos = 22;            % [grid points]
        ball_radius = 3;            % [grid points]

        % define a Cartesian spherical sensor
        sensor_radius = 4e-3;       % [m]
        center_pos = [0, 0, 0];     % [m]
        num_sensor_points = 100;
        sensor_mask = makeCartSphere(sensor_radius, num_sensor_points, center_pos, false);


        % create a binary sensor mask of an equivalent continuous sphere
        sensor_radius_grid_points = round(sensor_radius / kgrid.dx);
        
        % define the properties of the propagation medium
        medium_sound_speed = params{1};	% [m/s]
        if dim == 2
            kgrid = kWaveGrid(Nx, dx, Ny, dy);
            p0_binary = ball_magnitude * makeCircle(Nx, Ny, ball_x_pos, ball_y_pos, ball_radius);
            binary_sensor_mask = makeDisc(kgrid.Nx, kgrid.Ny, 0, 0, sensor_radius_grid_points);
        else
            kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);
            p0_binary = ball_magnitude * makeBall(Nx, Ny, Nz, ball_x_pos, ball_y_pos, ball_z_pos, ball_radius);
            binary_sensor_mask = makeSphere(kgrid.Nx, kgrid.Ny, kgrid.Nz, sensor_radius_grid_points);
        end


        % create the time array
        kgrid.makeTime(medium_sound_speed);
        % mock the simulation
        sensor_data = sin(repmat(1:kgrid.Nt,[num_sensor_points,1]));
        % smooth the initial pressure distribution and restore the magnitude
        p0 = smooth(p0_binary, true);


        % interpolate data to remove the gaps and assign to time reversal data
        trbd = interpCartData(kgrid, sensor_data, sensor_mask, binary_sensor_mask);
        warning('off','all')
        kgrid = struct(kgrid);
        warning('on','all')

        idx_padded = sprintf('%06d', k );
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        filename = [output_folder idx_padded '.mat'];
        save(filename, 'params', 'kgrid', 'sensor_data', 'sensor_mask', 'binary_sensor_mask', 'trbd');


    end
end

