matrix_sizes = { ...
    [12, 7], ...
    [12, 7, 5], ...
    [1], ...
    [5], ...
    [8, 3, 6, 1, 5], ...
}; 

% 
output_file = 'collectedValues/minND.mat';
recorder = utils.TestRecorder(output_file);

for idx=1:length(matrix_sizes)
    matrix_size = matrix_sizes{idx};
    matrix = rand(matrix_size);
   
    [min_val, ind] = minND(matrix);
    
    recorder.recordVariable('matrix', matrix);
    recorder.recordVariable('min_val', min_val);
    recorder.recordVariable('ind', ind);
    recorder.increment();
end
recorder.saveRecordsToDisk();




