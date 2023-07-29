% rect_pos, Lx, Ly, theta, num_points, plot_rect
all_params = { ...
    {[20, 20, 30], 10, 10, [70, 50, 20], 500, false}, ...
    {[20, 70], 10, 10, -15, 500, false}, ...
    {[-20, -30], 15, 30, -5.3, 72, false}, ...
    {[-20, -30], 15, 30, -5.3, 72, false}, ...
};
output_file = 'collectedValues/makeCartRect.mat';
recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};

    coordinates = makeCartRect(params{:});

    recorder.recordVariable('params', params);
    recorder.recordVariable('coordinates', coordinates);
    recorder.increment();
end
recorder.saveRecordsToDisk();
