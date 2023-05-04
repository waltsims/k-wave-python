output_file = 'collectedValues/hounsfield2density.mat';
recorder = utils.TestRecorder(output_file);

p = phantom('Modified Shepp-Logan',200);
out = hounsfield2density(p);
recorder.recordVariable('out', out);
recorder.recordVariable('p', p);

recorder.saveRecordsToDisk();