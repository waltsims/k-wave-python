dims = [1, 2, 3];


idx = 0;

for dim = dims
    if dim == 1
        matrices = {rand(30, 1), rand(4, 1)};
        exp_coeffs = {[20, 3], [3, 0], [0, 2]};
        is_edge_val = {false, true};
        edge_vals = {0, 10};
    elseif dim == 2
        matrices = {rand(30, 10), rand(8, 5)};
        exp_coeffs = {[20, 3, 0, 4], [3, 4, 1, 2]};
        is_edge_val = {true, true};
        edge_vals = {3, 5};
    elseif dim == 3
        matrices = {rand(30, 10, 8), rand(4, 5, 6)};
        exp_coeffs = {[20, 3, 0, 4, 2, 1], [3, 4, 1, 0, 12, 0]};
        is_edge_val = {true, false};
        edge_vals = {30, 0};
    end
    
    for i=1:length(matrices)
        for j=1:length(exp_coeffs)
            for k=1:length(is_edge_val)
                
                matrix = matrices{i};
                exp_coeff = exp_coeffs{j};

                input_args = { ...
                    exp_coeff ...
                };
            
                if is_edge_val{k}
                    input_args{end+1} = edge_vals{k};
                end
                
                expanded_matrix = expandMatrix(matrix, input_args{:});

                disp(size(matrix))
                disp(size(expanded_matrix))
                
                idx_padded = sprintf('%06d', idx);
                filename = ['collectedValues/expandMatrix/' idx_padded '.mat'];
                save(filename, 'matrix', 'input_args', 'expanded_matrix');
                
                idx = idx + 1;
            end            
        end
    end
end
disp('Done.')
