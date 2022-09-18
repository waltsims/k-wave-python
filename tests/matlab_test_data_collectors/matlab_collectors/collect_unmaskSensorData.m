% dims = [1, 2, 3];
% data_dims = { ...
%     [12], ...
%     [20, 40], ...
%     [16, 24, 32], ...
% };
% d_xyz = -1;  % not important
% num_pts_per_dim = 9;
% output_folder = 'collectedValues/unmaskSensorData';
% idx = 0;
% 
% 
% for dim=dims
%     data_dim = data_dims{dim};
% 
%     switch dim
%         case 1
%             kgrid = kWaveGrid( ...
%                 data_dim(1), d_xyz ...
%             );
%             mask_points_x = uint8(rand_vector_in_range(1, data_dim(1)-1, num_pts_per_dim));
% 
%             sensor_mask = zeros(1, data_dim);
%             sensor_mask(mask_points_x) = 1;
%         case 2
%             kgrid = kWaveGrid( ...
%                 data_dim(1), d_xyz, ...
%                 data_dim(2), d_xyz ...
%             );
%             mask_points_x = uint8(rand_vector_in_range(1, data_dim(1)-1, num_pts_per_dim));
%             mask_points_y = uint8(rand_vector_in_range(1, data_dim(2)-1, num_pts_per_dim));
% 
%             sensor_mask = zeros(data_dim);
%             sensor_mask(mask_points_x, mask_points_y) = 1;
%         case 3
%             kgrid = kWaveGrid( ...
%                 data_dim(1), d_xyz, ...
%                 data_dim(2), d_xyz, ...
%                 data_dim(3), d_xyz ...
%             );
%             mask_points_x = uint8(rand_vector_in_range(1, data_dim(1)-1, num_pts_per_dim));
%             mask_points_y = uint8(rand_vector_in_range(1, data_dim(2)-1, num_pts_per_dim));
%             mask_points_z = uint8(rand_vector_in_range(1, data_dim(3)-1, num_pts_per_dim));
% 
%             sensor_mask = zeros(data_dim);
%             sensor_mask(mask_points_x, mask_points_y, mask_points_z) = 1;
%     end
% 
%     sensor.mask = sensor_mask;
%     num_ones_in_mask = sum(sensor_mask, 'all');
%     sensor_data = rand(1, num_ones_in_mask);
% 
%     unmasked_sensor_data = unmaskSensorData(kgrid, sensor, sensor_data);
%     disp(size(unmasked_sensor_data));
% 
%     % Save output
%     idx_padded = sprintf('%06d', idx);
%     filename = [output_folder '/' idx_padded '.mat'];
%     save(filename, 'data_dim', 'sensor_mask', ...
%         'sensor_data', 'unmasked_sensor_data');
%     idx = idx + 1;
%     
% end
