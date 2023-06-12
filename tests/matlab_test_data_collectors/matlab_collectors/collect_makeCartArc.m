% radius, num_points, center_pos
all_params = {
    {[0,0], 5.0, 7.0, [2,5], 50}, ...
    {[2,2], 5.0, 7.0, [8,5], 5}, ...
    {[2,9], 8.0, 13.0, [8,5], 5}
    };  
output_folder = 'collectedValues/makeCartArc/';

for idx=1:length(all_params)
        cart_arc = makeCartArc(all_params{idx}{1}, all_params{idx}{2}, all_params{idx}{3}, all_params{idx}{4}, all_params{idx}{5});
        params = all_params{idx};
        idx_padded = sprintf('%06d', idx - 1);

        idx_padded = sprintf('%06d', idx - 1);
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        filename = [output_folder idx_padded '.mat'];
        save(filename, 'params', 'cart_arc');

end
disp('Done.')
