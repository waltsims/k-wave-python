matrix_sizes = { ...
    [12, 7], ...
    [12, 7, 5], ...
    [2, 8, 56], ...
    [5, 10, 10], ...
    [8, 3, 6, 9, 5], ...
}; 

% 
recorder = utils.TestRecorder('collectedValues/maxND.mat');
for idx=1:length(matrix_sizes)
    matrix_size = matrix_sizes{idx};
    matrix = rand(matrix_size);
   
    [max_val, ind] = maxND(matrix);

    recorder.recordVariable('matrix', matrix);
    recorder.recordVariable('max_val', max_val);
    recorder.recordVariable('ind', ind);
    recorder.increment();
    
end

recorder.saveRecordsToDisk();
