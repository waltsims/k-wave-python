rel_path = "collectedValues/writeAttributes";
idx = 0;

matrix_name = 'test';
filename = rel_path + "/" + num2str(idx) + ".h5";
matrix = single(10.0 * ones([1,1]));
writeMatrix(filename,matrix, matrix_name);

writeAttributes(filename)

disp('Done.')
