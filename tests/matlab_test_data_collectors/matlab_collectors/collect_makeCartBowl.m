all_params = { ...
    {[20, 20, 30], 1, 2, [70, 50, 20], 2, false}, ...
    {[20, 10, 30], 2, 3, [30, 50, 10], 3, false}, ...
    {[20, 40, 30], 5, 3, [50, 20, 20], 1, false}, ...
    {[10, 20, 30], 2, 1, [40, 30, 20], 2, false}, ...
};

output_file = 'collectedValues/makeCartBowl.mat';
recorder = utils.TestRecorder(output_file);

for idx = 1:length(all_params)
    disp(idx);
    params = all_params{idx};

    coordinates = makeCartBowl(params{:});

    recorder.recordVariable('params', params);
    recorder.recordVariable('coordinates', coordinates);
    recorder.increment();
end

recorder.saveRecordsToDisk();
