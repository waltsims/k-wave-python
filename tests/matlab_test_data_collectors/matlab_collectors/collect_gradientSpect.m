output_file = 'collectedValues/gradientSpect.mat';
recorder = utils.TestRecorder(output_file);

x = (pi / 20 : pi / 20 : 4 * pi);
y = (pi / 20 : pi / 20 : 4 * pi);

[X, Y] = meshgrid(x, y);

z = sin(X) * sin(Y);

[dy, dx] = gradientSpect(z, [pi/20 pi/20]);

recorder.recordVariable('X', X);
recorder.recordVariable('Y', Y);
recorder.recordVariable('z', z);
recorder.recordVariable('dy', dy);
recorder.recordVariable('dx', dx);

recorder.saveRecordsToDisk();
