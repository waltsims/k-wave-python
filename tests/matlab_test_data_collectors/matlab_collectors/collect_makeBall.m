all_params = { ...
    {20, 20, 20, 10, 10, 10, 5, false, false}, ...
    {20, 20, 20, 10, 10, 10, 5, false, true}, ...
    {20, 20, 20, 10, 10, 10, 5, false, false}, ...
    {40, 20, 15, 10, 15, 12, 2, false, false}, ...
    {40, 50, 60, 27, 33, 20, 13, false, true}, ...
}; 

recorder = utils.TestRecorder('collectedValues/makeBall.mat');

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    ball = makeBall(params{:});

    recorder.recordVariable('ball', ball);
    recorder.recordVariable('params', params);
    recorder.increment()
    
end

recorder.saveRecordsToDisk();