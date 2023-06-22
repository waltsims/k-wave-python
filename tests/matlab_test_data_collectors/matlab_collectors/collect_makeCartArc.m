% radius, num_points, center_pos
all_params = {
    {[0,0], 5.0, 7.0, [2,5], 50}, ...
    {[2,2], 5.0, 7.0, [8,5], 5}, ...
    {[2,9], 8.0, 13.0, [8,5], 5}
    };  
output_file = 'collectedValues/make_cart_arc.mat';

recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)

        params = all_params{idx};
        cart_arc = makeCartArc(params{1}, params{2}, params{3}, params{4}, params{5});

        recorder.recordVariable('params', params);
        recorder.recordVariable('cart_arc', cart_arc);

        recorder.increment();

end
recorder.saveRecordsToDisk();
disp('Done.')
