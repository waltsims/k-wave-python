scan_lines = rand(5, 618);
steering_angles = -4:2:4;
image_size = [0.0500, 0.0522];
c0 = 1540;
dt = 4.3098e-08;
resolution = [128, 128];

recorder = utils.TestRecorder('collectedValues/scanConversion.mat');

recorder.recordVariable('scan_lines', scan_lines);
recorder.recordVariable('steering_angles', steering_angles);
recorder.recordVariable('image_size', image_size);
recorder.recordVariable('c0', c0);
recorder.recordVariable('dt', dt);
recorder.recordVariable('resolution', resolution);


b_mode = scanConversion(scan_lines, steering_angles, image_size, c0, dt, resolution);
recorder.recordVariable('b_mode', b_mode);

recorder.increment()

recorder.saveRecordsToDisk();
