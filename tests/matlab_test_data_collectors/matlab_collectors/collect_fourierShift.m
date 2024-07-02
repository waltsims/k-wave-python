
% fourierShift does not support more than 4 dims
dims = [1, 2, 3, 4];
data_dims = { ...
    [7,], ...  # 1D
    [8, 2], ...  # 2D
    [4, 8, 3], ...  # 3D
    [5, 10, 5, 4] ...  # 4D
};
shifts = [0, 0.12, 0.29, 0.5, 0.52, 0.85, 1.0];
output_file = 'collectedValues/fourierShift.mat';
idx = 0;
recorder = utils.TestRecorder(output_file);
for data_dim_idx=1:size(dims, 2)
    data_dim = data_dims{data_dim_idx};
    data = rand(data_dim);
    
    for shift=shifts
        shifted_data = fourierShift(data, shift);
        
        % Save output
        recorder.recordVariable('data', data);
        recorder.recordVariable('shift', shift);
        recorder.recordVariable('shifted_data', shifted_data);
        recorder.increment();
        
        for shift_dim=1:size(data_dim, 2)
            shifted_data = fourierShift(data, shift, shift_dim);
            recorder.recordVariable('data', data);
            recorder.recordVariable('shift', shift);
            recorder.recordVariable('shift_dim', shift_dim);
            recorder.recordVariable('shifted_data', shifted_data);
            recorder.increment();
        end
        
    end
    
end

recorder.saveRecordsToDisk();
