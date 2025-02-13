output_file = 'collectedValues/extract_amp_phase.mat';
recorder = utils.TestRecorder(output_file);

% Create test input data
Fs = 10e6;           % 10 MHz sampling frequency
t = 0:1/Fs:1e-6;     % 1 microsecond time array
source_freq = 2e6;    % 2 MHz source frequency

% Create a simple time series with known frequency
data = sin(2*pi*source_freq*t);
data = reshape(data, [1, length(data)]);  % Ensure row vector

% Add some additional test cases with different dimensions
data_2d = repmat(data, [10, 1]);         % 2D data
data_3d = repmat(data, [5, 5, 1]);       % 3D data

% Test parameters
fft_padding = 3;
window = 'Hanning';

% Record input variables
recorder.recordVariable('data', data);
recorder.recordVariable('data_2d', data_2d);
recorder.recordVariable('data_3d', data_3d);
recorder.recordVariable('Fs', Fs);
recorder.recordVariable('source_freq', source_freq);
recorder.recordVariable('fft_padding', fft_padding);
recorder.recordVariable('window', window);

% Extract amplitude and phase for 1D data
[amp, phase, f] = extract_amp_phase(data, Fs, source_freq, 'auto', fft_padding, window);
recorder.recordVariable('amp_1d', amp);
recorder.recordVariable('phase_1d', phase);
recorder.recordVariable('f_1d', f);

% Extract amplitude and phase for 2D data
[amp_2d, phase_2d, f_2d] = extract_amp_phase(data_2d, Fs, source_freq, 'auto', fft_padding, window);
recorder.recordVariable('amp_2d', amp_2d);
recorder.recordVariable('phase_2d', phase_2d);
recorder.recordVariable('f_2d', f_2d);

% Extract amplitude and phase for 3D data
[amp_3d, phase_3d, f_3d] = extract_amp_phase(data_3d, Fs, source_freq, 'auto', fft_padding, window);
recorder.recordVariable('amp_3d', amp_3d);
recorder.recordVariable('phase_3d', phase_3d);
recorder.recordVariable('f_3d', f_3d);

% Test with explicit dimension specification
[amp_dim2, phase_dim2, f_dim2] = extract_amp_phase(data_2d, Fs, source_freq, 2, fft_padding, window);
recorder.recordVariable('amp_dim2', amp_dim2);
recorder.recordVariable('phase_dim2', phase_dim2);
recorder.recordVariable('f_dim2', f_dim2);

recorder.saveRecordsToDisk(); 