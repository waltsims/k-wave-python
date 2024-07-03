params = { ...
    [40e6, 5e6, 4, 50], ...
    [80e6, 20e6, 2, 80], ...
    [10e6, 0.5e6, 1, 100], ...
}; 

recorder = utils.TestRecorder('collectedValues/gaussianFilter.mat');
for param_idx = 1:length(params)
    fs = params{param_idx}(1);
    fc = params{param_idx}(2);
    n_cycles = params{param_idx}(3);
    bw = params{param_idx}(4);
    input_signal = toneBurst(fs, fc, n_cycles);
    output_signal = gaussianFilter(input_signal, fs, fc, bw);
    recorder.recordVariable('input_signal', input_signal);
    recorder.recordVariable('output_signal', output_signal);
    recorder.recordVariable('fs', fs);
    recorder.recordVariable('fc', fc);
    recorder.recordVariable('bw', bw);
    recorder.increment();
end
recorder.saveRecordsToDisk();
disp('Done.')
