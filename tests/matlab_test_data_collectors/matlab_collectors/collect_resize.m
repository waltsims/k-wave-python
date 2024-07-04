% Generate a list of interpolation methods

output_file = 'collectedValues/resize.mat';
recorder = utils.TestRecorder(output_file);

interp_methods = {'nearest', 'linear'};

% Generate a random 3D volume with dimensions Nx x Ny x Nz
sizes =  [32, 32, 32; 48, 48, 48; 56, 56, 56];
new_sizes = [16, 16, 16; 32, 32, 32; 48, 48, 48]; 


for i = length(sizes)
    new_size = new_sizes(i,:);
    volume = rand(sizes(i,:));
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
end
recorder.saveRecordsToDisk(); 
