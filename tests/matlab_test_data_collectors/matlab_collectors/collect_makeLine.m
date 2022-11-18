
all_params = { ...
    {20, 20, [1, 1], [19, 19]}, ...
    {20, 20, [1, 1], [5, 5]}, ...
    {10, 20, [4, 15], [10, 1]}, ...
    {20, 20, [15, 15], 0.75 * pi, 5}, ...
    {20, 20, [15, 15], 0.75 * pi, -5}, ...
    {20, 10, [15, 10], 0.1 * pi, 5}, ...
}; 

% 

for idx=1:length(all_params)
    disp(idx);
    params = all_params{idx};
    
    if idx == 4
        disp('aaa');
    end
    
    line = makeLine(params{:});
    
    idx_padded = sprintf('%06d', idx - 1);
    filename = ['collectedValues/makeLine/' idx_padded '.mat'];
    save(filename, 'params', 'line');
end
