
% fourierShift does not support more than 4 dims
dims = [1, 2, 3, 4];
data_dims = { ...
    [1, 7], ...  # 1D
    [8, 2], ...  # 2D
    [4, 8, 3], ...  # 3D
    [5, 10, 5, 4] ...  # 4D
};
shifts = [0, 0.12, 0.29, 0.5, 0.52, 0.85, 1.0];
output_folder = 'collectedValues_fourierShift';
idx = 0;

for data_dim_idx=1:size(dims, 2)
    data_dim = data_dims{data_dim_idx};
    data = rand(data_dim);
    
    for shift=shifts
        shifted_data = fourierShift(data, shift);
        
        % Save output
        idx_padded = sprintf('%06d', idx);
        filename = [output_folder '/' idx_padded '.mat'];
        save(filename, 'data', 'shift', 'shifted_data');
        idx = idx + 1;
        
        for shift_dim=1:size(data_dim, 2)
            shifted_data = fourierShift(data, shift, shift_dim);
            
            % Save output
            idx_padded = sprintf('%06d', idx);
            filename = [output_folder '/' idx_padded '.mat'];
            save(filename, 'data', 'shift', 'shift_dim', 'shifted_data');
            idx = idx + 1;
        end
        
    end
    
end

