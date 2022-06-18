sensor_data_size = [32, 4];

sensor_data = rand(sensor_data_size);
reorder_index = rand(sensor_data_size(1), 1);

reordered_data = reorderBinarySensorData(sensor_data, reorder_index);


idx = 0;
idx_padded = sprintf('%06d', idx);
filename = ['collectedValues_reorderBinarySensorData/' idx_padded '.mat'];
save(filename, 'sensor_data', 'reorder_index', 'reordered_data');
