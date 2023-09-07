% radius, num_points, center_pos
all_params = {
    {5, 100, [1,1],  2*pi}, ...
    {1, 10, [5,5],  pi/4}, ...
    {0.5, 10, [0,0],  2*pi}, ...
    {0.05, 20, [0 0]}
    };  
output_file = 'collectedValues/makeCartCircle.mat';
recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
        params = all_params{idx};
        if length(all_params{idx}) < 4
            circle = makeCartCircle(params{1}, params{2}, params{3});
            params = {params{1}, params{2}, params{3}};
        else
            circle = makeCartCircle(params{1}, params{2}, params{3}, params{4});
            params = {params{1}, params{2}, params{3}, params{4}};
        end
        recorder.recordVariable('params', params);
        recorder.recordVariable('circle', circle);
        recorder.increment()

end
recorder.saveRecordsToDisk();
