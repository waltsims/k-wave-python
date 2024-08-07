all_params = { ...
    {20, 20, 10, 10, 5, false}, ...
    {20, 20, 10, 10, 5, false}, ...
    {20, 20, 10, 10, 5, false}, ...
    {40, 20, 10, 15, 2, false}, ...
    {17, 33, 8, 9, 7, false}, ...
    {17, 38, 5, 12, 4, false}, ...
}; 

recorder = utils.TestRecorder('collectedValues/makeDisc.mat');
for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};

    recorder.recordVariable('params', params);
    
    disc = makeDisc(params{:});
    recorder.recordVariable('disc', disc);
    recorder.increment();
    
end
recorder.saveRecordsToDisk();