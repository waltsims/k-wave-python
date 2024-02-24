% NOTE: it is very important to alternate kgrid.dim between each test case
%       because tolStar has internal state. We want tests to be independent
%       of each other. Alternating the kgrid.dim between test cases will
%       reset the state.
params = { ...
    {
        0.005, ... 
        struct('dx', 0.5, 'dim', 3, 'x_vec', 0.5:0.5:50, 'y_vec', 0.5:0.5:50, 'z_vec', 0.5:0.5:50, 'Nx', 100, 'Ny', 100, 'Nz', 100), 
        [25.0, 25, 25] ...
    }, ...
    {
        0.01, ...
        struct('dx', 1.0, 'dim', 1, 'x_vec', 1:100, 'Nx', 100),  ...
        [50.0, 129.0] ...
    }, ...
    {
        0.01,  ...
        struct('dx', 1.0, 'dim', 2, 'x_vec', 1:100, 'y_vec', 1:100, 'Nx', 100, 'Ny', 100),  ...
        [50.0, 50.0] ...
    }, ...
    {
        0.05, ...
        struct('dx', 1.0, 'dim', 1, 'x_vec', 1:5, 'Nx', 19),  ...
        12.0 ...
    }, ...
    {
        0.01,  ...
        struct('dx', 1.0, 'dim', 3, 'x_vec', 1:100, 'y_vec', 1:100, 'z_vec', 1:100, 'Nx', 100, 'Ny', 100, 'Nz', 100),  ...
        [-5, 50.0, 50] ...
    }, ...
    {
        0.005,  ...
        struct('dx', 0.5, 'dim', 1, 'x_vec', 0.5:0.5:50, 'Nx', 100),  ...
        25.0 ...
    }, ...
    {
        0.005,  ...
        struct('dx', 0.5, 'dim', 2, 'x_vec', 0.5:0.5:50, 'y_vec', 0.5:0.5:50, 'Nx', 100, 'Ny', 100),  ...
        [25.0, 25.0] ...
    }, ...
    {
        0.005,  ...
        struct('dx', 0.5, 'dim', 2, 'x_vec', 0.5:0.3:19, 'y_vec', 0.5:0.49:27, 'Nx', 12, 'Ny', 17),  ...
        [-45.0, 12] ...
    }, ...
};

output_file = 'collectedValues/tolStar.mat';
recorder = utils.TestRecorder(output_file);
for param_idx = 1:length(params)
    
    [lin_ind, is, js, ks] = private_kwave_functions.tolStar(params{param_idx}{:}, false);

    recorder.recordVariable('params', params{param_idx});
    recorder.recordVariable('lin_ind', lin_ind);
    recorder.recordVariable('is', is);
    recorder.recordVariable('js', js);
    recorder.recordVariable('ks', ks);
    recorder.increment();

end
recorder.saveRecordsToDisk();
disp('Done.')
