all_params = { ...
    {36, 13}, ...
    {36, 13, 21, false, true}, ...
    {28, 13, 5, false, false}, ...
};
output_folder = 'collectedValues/makeSphericalSection/';

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    [spherical_section, distance_map] = makeSphericalSection(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    filename = [output_folder idx_padded '.mat'];
    save(filename, 'params', 'spherical_section', 'distance_map');
end
