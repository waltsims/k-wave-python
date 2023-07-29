all_params = { ...
    {[20, 30], [5]}, ...
    {[-1.5, -12.3], [182]}, ...
    {[-1.5, -12.3], [0]}, ...
    {[0, 0, 0], [0, 0, 0]}, ...
    {[5, 5, 5], [0, 90, 180]}, ...
    {[50, 12, -5], [-13, 27, -180]}, ...
}; 

output_file = 'collectedValues/getAffineMatrix.mat';
recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    affine_matrix = getAffineMatrix(params{:});
    
    recorder.recordVariable('params', params);
    recorder.recordVariable('affine_matrix', affine_matrix);
    recorder.increment();
    
end

recorder.saveRecordsToDisk();
