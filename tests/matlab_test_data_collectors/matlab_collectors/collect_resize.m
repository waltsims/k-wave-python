% Generate a list of interpolation methods
% TODO: test also for nearest
% interp_methods = {'nearest', 'linear'};
interp_methods = {'linear'};

% Generate a random 3D volume with dimensions Nx x Ny x Nz
Nx = randi([10,20]);
Ny = randi([10,20]);
Nz = randi([10,20]);
volume = rand([Nx, Ny, Nz]);

% Specify the desired new size of the volume
new_size = [randi([10,20]), randi([10,20]), randi([10,20])];

output_folder = 'collectedValues/resize/';
k = 0;

% Loop through each interpolation method
for method = interp_methods
    % Use the current interpolation method to resize the volume
    resized_volume = resize(volume, new_size, char(method));

    % Save the original volume, resized volume, and interpolation method
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    idx_padded = sprintf('%06d', k );
    filename = [output_folder idx_padded '.mat'];
    save(filename, 'volume', 'resized_volume', 'new_size', 'method');
    k = k + 1;
end
