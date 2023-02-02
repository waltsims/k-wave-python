% Generate a list of interpolation methods

output_file = 'collectedValues/resize.mat';
recorder = utils.TestRecorder(output_file);

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


% Loop through each interpolation method
for method = interp_methods
    % Use the current interpolation method to resize the volume
    resized_volume = resize(volume, new_size, char(method));

    recorder.recordVariable('volume', volume);
    recorder.recordVariable('resized_volume', resized_volume);
    recorder.recordVariable('new_size', new_size);
    recorder.recordVariable('method', method)
    recorder.increment();

end
recorder.saveRecordsToDisk(); 
    
