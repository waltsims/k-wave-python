output_file = 'collectedValues/gaussian.mat';
recorder = utils.TestRecorder(output_file);


x = (-3 : 0.05 : 3);

y = gaussian(x);

recorder.recordVariable('x', x);
recorder.recordVariable('y', y);

recorder.saveRecordsToDisk();