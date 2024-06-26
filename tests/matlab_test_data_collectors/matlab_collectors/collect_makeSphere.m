all_params = { ...
    {40, 30, 36, 10, false, false}, ...
    {17, 23, 10, 4, false, false}, ...
    {3, 4, 5, 1, false, true}, ...
};
output_file = 'collectedValues/makeSphere.mat';

recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    sphere = makeSphere(params{:});

    recorder.recordVariable('params', params);
    recorder.recordVariable('sphere', sphere);
    recorder.increment();
    
end
recorder.saveRecordsToDisk();
