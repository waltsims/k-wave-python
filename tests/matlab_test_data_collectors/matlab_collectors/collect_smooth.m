output_file = 'collectedValues/smooth.mat';
recorder = utils.TestRecorder(output_file);

img = rand([20, 20]);
out = smooth(img);
recorder.recordVariable('img', img);
recorder.recordVariable('out', out);

recorder.saveRecordsToDisk();   