output_file = 'collectedValues/tenenbaum.mat';
recorder = utils.TestRecorder(output_file);

img2 = ones([5,5]);
out2 = sharpness(img2, 'Tenenbaum');
recorder.recordVariable('img2', img2);
recorder.recordVariable('out2', out2);


img3 = ones([5,5,5]);
out3 = sharpness(img3, 'Tenenbaum');
recorder.recordVariable('img3', img3);
recorder.recordVariable('out3', out3);
recorder.saveRecordsToDisk();