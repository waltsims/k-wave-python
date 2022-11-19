all_params = { ...
    {rand(4)}, ...
    {rand(8)}, ...
    {rand(7, 12)}, ...
    {rand(8, 6)}, ...
};
output_folder = 'collectedValues/revolve2D/';

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    mat3D = revolve2D(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    filename = [output_folder idx_padded '.mat'];
    save(filename, 'params', 'mat3D');
end
