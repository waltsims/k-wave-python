rel_path = "collectedValues/writeMatrix";
idx = 0;
for dim=1:3
    for compression_level=1:9
        matrix_name = 'test';
        filename = rel_path + "/" + num2str(idx) + ".h5";
        matrix = single(10.0 * ones([1,dim]));
        writeMatrix(filename,matrix, matrix_name,compression_level);
        idx = idx + 1;
    end
end
disp('Done.')
