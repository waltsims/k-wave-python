all_params = { ...
    {[20, 20], 1, [70, 50], 2, false, false}, ...
    {[20, 10, 30], 2,  [30, 50, 10], 3, false, false}, ...
    {[20, 40, 30], 5,[50, 20, 20], 1, false, false}, ...
    {[10, 20, 30], 2,  [40, 30, 20], 2, false, true}, ...
};

output_file = 'collectedValues/makeCartDisc.mat';
recorder = utils.TestRecorder(output_file);

for idx = 1:length(all_params)
    disp(idx);
    params = all_params{idx};

    coordinates = makeCartDisc(params{:});

    recorder.recordVariable('params', params);
    recorder.recordVariable('coordinates', coordinates);
    recorder.increment();
end

recorder.saveRecordsToDisk();
