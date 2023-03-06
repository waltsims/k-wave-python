all_params = { ...
    {20, 20, 20, 10, 10, 10, 5, false, false}, ...
    {20, 20, 20, 10, 10, 10, 5, false, true}, ...
    {20, 20, 20, 10, 10, 10, 5, false, false}, ...
    {40, 20, 15, 10, 15, 12, 2, false, false}, ...
    {40, 50, 60, 27, 33, 20, 13, false, true}, ...
}; 

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    ball = makeBall(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    filename = ['collectedValues/makeBall/' idx_padded '.mat'];
    save(filename, 'params', 'ball');
end