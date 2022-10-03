mask_size = [16, 23];
kgrid_size = [16, 23];
kgrid_dx_dy = 0.1;
sensor_data_size = [32, 4];

mask = zeros(mask_size(1), mask_size(2));

for i=1:20
    mask(randi(mask_size(1)), randi(mask_size(2))) = 1;
end
sensor = struct('mask', mask);

% sensor_data = rand(sensor_data_size);

kgrid = kWaveGrid(kgrid_size(1), kgrid_dx_dy, kgrid_size(2), kgrid_dx_dy);
reordered_sensor_data = reorderSensorData(kgrid, sensor, sensor_data);

idx = 0;
idx_padded = sprintf('%06d', idx);
filename = ['collectedValues/reorderSensorData/' idx_padded '.mat'];
save(filename, 'mask_size', 'kgrid_size', 'kgrid_dx_dy', ...
    'sensor_data_size', 'mask', 'sensor_data', 'reordered_sensor_data');
