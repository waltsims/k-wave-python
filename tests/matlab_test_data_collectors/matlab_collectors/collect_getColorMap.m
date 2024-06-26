list_num_colors = [12, 36, 256];
output_file = 'collectedValues/getColorMap.mat';

recorder = utils.TestRecorder(output_file);

for num_colors=list_num_colors
    color_map = getColorMap(num_colors);
    
    recorder.recordVariable('color_map', color_map);
    recorder.recordVariable('num_colors', num_colors);
    recorder.increment();
end
recorder.saveRecordsToDisk();
