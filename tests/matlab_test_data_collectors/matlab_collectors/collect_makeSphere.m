all_params = { ...
    {40, 30, 36, 10, false, false}, ...
    {17, 23, 10, 4, false, false}, ...
    {3, 4, 5, 1, false, true}, ...
};
output_folder = 'collectedValues/makeSphere/';

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    sphere = makeSphere(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    filename = [output_folder idx_padded '.mat'];
    save(filename, 'params', 'sphere');
end
