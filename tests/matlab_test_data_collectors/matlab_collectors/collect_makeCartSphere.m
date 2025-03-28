% radius, num_points, center_pos
all_params = {
    {5, 100, [1,1,8]}, ...
    {1, 10, [5,5,0]}, ...
    {0.5, 10, [0,0,0]}, ...
    {4e-3, 100, [0, 0, 0]}  % Case from interpCartData
    };
output_file = 'collectedValues/make_cart_sphere.mat';

recorder = utils.TestRecorder(output_file);

for idx=1:length(all_params)
    params = all_params{idx};
    cart_sphere = makeCartSphere(params{1}, params{2}, params{3});

    recorder.recordVariable('params', params);
    recorder.recordVariable('cart_sphere', cart_sphere);

    recorder.increment();
end
recorder.saveRecordsToDisk();
disp('Done.')
