params = { ...
    {
        randn(1, 1000), ... % 1D data
        1000, ...          % Sampling frequency
        50, ...            % Source frequency
        'Dim', 1, ...      % Optional parameter: dimension
        'FFTPadding', 2, ... % Optional parameter: FFT padding
        'Window', 'Hanning'  % Optional parameter: window type
    }, ...
    {
        randn(1000, 1), ... % 1D data in different dimension
        2000, ...          % Sampling frequency
        100, ...           % Source frequency
        'Dim', 2, ...      % Optional parameter: dimension
        'FFTPadding', 3, ... % Optional parameter: FFT padding
        'Window', 'Blackman' % Optional parameter: window type
    }, ...
    {
        randn(10, 100), ... % 2D data
        500, ...           % Sampling frequency
        10, ...            % Source frequency
        'Dim', 2, ...      % Optional parameter: dimension
        'FFTPadding', 4, ... % Optional parameter: FFT padding
        'Window', 'Hamming'  % Optional parameter: window type
    }, ...
    {
        randn(50, 50, 50), ... % 3D data
        1000, ...             % Sampling frequency
        250, ...              % Source frequency
        'Dim', 3, ...         % Optional parameter: dimension
        'FFTPadding', 5, ...  % Optional parameter: FFT padding
        'Window', 'Hann'      % Optional parameter: window type
    }, ...
    {
        randn(100, 100, 100, 10), ... % 4D data
        2000, ...                     % Sampling frequency
        500, ...                      % Source frequency
        'Dim', 4, ...                 % Optional parameter: dimension
        'FFTPadding', 3, ...          % Optional parameter: FFT padding
        'Window', 'Kaiser'            % Optional parameter: window type
    }, ...
};

output_file = 'collectedValues/extract_amp_phase.mat';
recorder = utils.TestRecorder(output_file);
for param_idx = 1:length(params)
    
    [amp, phase, freq] = extractAmpPhase(params{param_idx}{:});

    recorder.recordVariable('params', params{param_idx});
    recorder.recordVariable('amp', amp);
    recorder.recordVariable('phase', phase);
    recorder.recordVariable('freq', freq);
    recorder.increment();

end
recorder.saveRecordsToDisk();
disp('Done.')