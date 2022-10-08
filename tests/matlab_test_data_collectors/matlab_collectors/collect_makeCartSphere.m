rel_path = 'collectedValues/makeCartSphere';
% radius, num_points, center_pos
all_params = {
    {5, 100, [1,1,8]}, ...
    {1, 10, [5,5,0]}, ...
    {0.5, 10, [0,0,0]}
    };

for idx=1:length(all_params)
        sphere = makeCartSphere(all_params{idx}{1}, all_params{idx}{2}, all_params{idx}{3});
        params = all_params{idx};
        idx_padded = sprintf('%06d', idx - 1);
        filename = [rel_path idx_padded '.mat'];

        save(filename, 'params', 'sphere');
end
disp('Done.')
