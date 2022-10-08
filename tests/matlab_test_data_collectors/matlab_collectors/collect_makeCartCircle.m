rel_path = 'collectedValues/makeCartCircle';
% radius, num_points, center_pos
all_params = {
    {5, 100, [1,1],  2*pi}, ...
    {1, 10, [5,5],  pi/4}, ...
    {0.5, 10, [0,0],  2*pi}
    };  

for idx=1:length(all_params)
        cirlce = makeCartCircle(all_params{idx}{1}, all_params{idx}{2}, all_params{idx}{3});
        params = all_params{idx};
        idx_padded = sprintf('%06d', idx - 1);
        filename = [rel_path idx_padded '.mat'];

        save(filename, 'params', 'circle');
end
disp('Done.')
