all_params = { ...
    {[20, 30, 10], [5, 50, 10], 3}, ...
    {[0, 0, 0], [5, 50, 10]}, ...
    {[0, 0, 0], [5, 50, 10], -5}, ...
    {[0, 0, 0], [0, 0, 0], -5}, ...
    {[0, 0, 0], [0, 0, 0], 0}, ...
    {[15, 16, 17], [15, 16, 17]}, ...
}; 

output_file = 'collectedValues/computeLinearTransform.mat';
recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    [rotMat, offsetPos] = computeLinearTransform(params{:});
    
    recorder.recordVariable('params', params);
    recorder.recordVariable('rotMat', rotMat);
    recorder.recordVariable('offsetPos', offsetPos);
    recorder.increment();
end

recorder.saveRecordsToDisk();
