% adapted from getDelteaBLI_test_bli.m in k-wave toolbox v1.4.0

output_file = 'collectedValues/getDeltaBLI.mat';
recorder = utils.TestRecorder(output_file);
% set the number of upsampled points
NN = 10002;

% include the imaginary component of the off-grid delta function
include_imag = true;

% run test for both even and odd off-grid deltas
for loop_index = 1:2
    
    % specify the computational grid
    Nx = 30 + loop_index;
    dx = 2/Nx;
    x_grid = linspace(-1, 1, Nx+1);
    x_grid = x_grid(1:end-1);

    % create a list of delta function positions
    positions = linspace(0, dx, 7);

    % loop through positions
    for position_index = 1:length(positions)
        position = positions(position_index);
        f_grid = getDeltaBLI(Nx, dx, x_grid, position, include_imag);
        recorder.recordVariable('Nx', Nx);
        recorder.recordVariable('dx', dx);
        recorder.recordVariable('x_grid', x_grid);
        recorder.recordVariable('position', position);
        recorder.recordVariable('include_imag', include_imag);
        recorder.recordVariable('f_grid', f_grid);
        recorder.increment();
    end
end
recorder.saveRecordsToDisk();
