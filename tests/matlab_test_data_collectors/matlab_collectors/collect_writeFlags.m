rel_path = "collectedValues/writeFlags";
idx = 0;
for dim=1:3
    filename = rel_path + "/" + num2str(idx) + ".h5";
    grid_size = 10 * ones([3,1]);
    grid_spacing = 0.1 * ones([3,1]);
    pml_size = 2 * ones([3,1]);
    pml_alpha = 0.5 * ones([3,1]);
    
    writeGrid(filename,grid_size, grid_spacing, pml_size, pml_alpha, 5, 0.5, 1540)
    writeMatrix(filename,single([0]),'sensor_mask_index')
    writeFlags(filename)
    idx = idx + 1;
end
disp('Done.')
