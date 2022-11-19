% radius, num_points, center_pos
all_params = {
    {5, 100, [1,1],  2*pi}, ...
    {1, 10, [5,5],  pi/4}, ...
    {0.5, 10, [0,0],  2*pi}
    };  
output_folder = 'collectedValues/makeCartCircle/';

for idx=1:length(all_params)
        circle = makeCartCircle(all_params{idx}{1}, all_params{idx}{2}, all_params{idx}{3}, all_params{idx}{4});
        params = all_params{idx};
        idx_padded = sprintf('%06d', idx - 1);

        idx_padded = sprintf('%06d', idx - 1);
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        filename = [output_folder idx_padded '.mat'];
        save(filename, 'params', 'circle');

end
disp('Done.')
