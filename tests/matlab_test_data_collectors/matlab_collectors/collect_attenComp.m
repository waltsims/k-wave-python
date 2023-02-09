output_file = 'collectedValues/attenComp.mat';
recorder = utils.TestRecorder(output_file);

inp_signal = rand(44, 100);
dt = 0.0001;
c = 1500;
alpha_0 = 0.1;
y = 0.5;
fit_type = 'mav';

[out_signal, tfd, cutoff_freq] = attenComp(inp_signal, dt, c, alpha_0, y, 'FitType', fit_type);

recorder.recordVariable('inp_signal', inp_signal);
recorder.recordVariable('dt', dt);
recorder.recordVariable('c', c);
recorder.recordVariable('alpha_0', alpha_0);
recorder.recordVariable('y', y);

recorder.recordVariable('out_signal', out_signal);
recorder.recordVariable('tfd', tfd);
recorder.recordVariable('cutoff_freq', cutoff_freq);
recorder.recordVariable('fit_type', fit_type);

recorder.saveRecordsToDisk();
