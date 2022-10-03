scan_lines = rand(5, 618);
steering_angles = -4:2:4;
image_size = [0.0500, 0.0522];
c0 = 1540;
dt = 4.3098e-08;
resolution = [128, 128];

idx = 0;
idx_padded = sprintf('%06d', idx);
filename = ['collectedValues/scanConversion/' idx_padded '.mat'];

b_mode = scanConversion(scan_lines, steering_angles, image_size, c0, dt, resolution);
save(filename, 'scan_lines', 'steering_angles', 'image_size', 'c0', 'dt', 'resolution', 'b_mode');
