matrix_sizes = { ...
    [12, 7], ...
    [12, 7, 5], ...
    [1], ...
    [5], ...
    [8, 3, 6, 1, 5], ...
}; 

% 

for idx=1:length(matrix_sizes)
    matrix_size = matrix_sizes{idx};
    matrix = rand(matrix_size);
   
    [min_val, ind] = minND(matrix);
    
    idx_padded = sprintf('%06d', idx - 1);
    filename = ['collectedValues/minND/' idx_padded '.mat'];
    save(filename, 'matrix', 'min_val', 'ind');
end




