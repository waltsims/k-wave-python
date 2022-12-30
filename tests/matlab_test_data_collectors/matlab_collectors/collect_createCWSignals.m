output_file = 'collectedValues/createCWSignals.mat';
recorder = utils.TestRecorder(output_file);

% define sampling parameters
f = 5e6;
T = 1/f;
Fs = 100e6;
dt = 1/Fs;
t_array = 0:dt:10*T;

% define amplitude and phase
amp = getWin(9, 'Gaussian');
phase = linspace(0, 2*pi, 9).';
signal = createCWSignals(t_array, f, amp, phase);
  
recorder.recordVariable('t_array', t_array);
recorder.recordVariable('f', f);
recorder.recordVariable('amp', amp);
recorder.recordVariable('phase', phase);
recorder.recordVariable('signal', signal);

recorder.saveRecordsToDisk();