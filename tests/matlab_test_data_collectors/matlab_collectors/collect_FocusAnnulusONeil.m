output_file = 'collectedValues/focusAnnulusONeil.mat';
recorder = utils.TestRecorder(output_file);

% define transducer parameters
radius      = 30e-3;     % [m]
% aperture diameters of the elements given an inner, outer pairs [m]
diameters       = [0 5; 10 15; 20 25].' .* 1e-3;
amplitude      = [0.5e6, 1e6, 0.75e6];     % source pressure [Pa]
source_phase    = deg2rad([0, 10, 20]);     % phase [rad]
frequency   = 1e6;        % [Hz]
sound_speed = 1500;       % [m/s]
density     = 1000;       % [kg/m^3]

% define position vectors
axial_position   = 0:1e-4:250e-3;     % [m]

% evaluate pressures
p_axial = focusedAnnulusONeil(radius, diameters, ...
    amplitude / (sound_speed * density), source_phase, frequency, sound_speed, density, ...
    axial_position);

recorder.recordVariable('radius', radius);
recorder.recordVariable('diameters', diameters);
recorder.recordVariable('amplitude', amplitude);
recorder.recordVariable('source_phase', source_phase);
recorder.recordVariable('frequency', frequency);
recorder.recordVariable('sound_speed', sound_speed);
recorder.recordVariable('density', density);
recorder.recordVariable('axial_position', axial_position);
recorder.recordVariable('p_axial', p_axial);

recorder.saveRecordsToDisk();