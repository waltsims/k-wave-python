list_num_colors = [12, 36, 256];
output_folder = 'collectedValues_getColorMap';
idx = 0;


for num_colors=list_num_colors
    color_map = getColorMap(num_colors);
    disp(size(color_map));
    
    % Save output
    idx_padded = sprintf('%06d', idx);
    filename = [output_folder '/' idx_padded '.mat'];
    save(filename, 'num_colors', 'color_map');
    idx = idx + 1;
end
