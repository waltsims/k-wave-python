output_file = 'collectedValues/focusBowlONeil.mat';
recorder = utils.TestRecorder(output_file);

% define transducer parameters
radius      = 140e-3;     % [m]
diameter    = 120e-3;     % [m]
velocity    = 100e-3;     % [m/s]
frequency   = 1e6;        % [Hz]
sound_speed = 1500;       % [m/s]
density     = 1000;       % [kg/m^3]

% define position vectors
axial_position   = 0:1e-4:250e-3;     % [m]
lateral_position = -15e-3:1e-4:15e-3; % [m]

% evaluate pressure
[p_axial, p_lateral, p_axial_complex] = focusedBowlONeil(radius, diameter, ...
    velocity, frequency, sound_speed, density, ...
    axial_position, lateral_position);

recorder.recordVariable('radius', radius);
recorder.recordVariable('diameter', diameter);
recorder.recordVariable('velocity', velocity);
recorder.recordVariable('frequency', frequency);
recorder.recordVariable('sound_speed', sound_speed);
recorder.recordVariable('density', density);
recorder.recordVariable('axial_position', axial_position);
recorder.recordVariable('lateral_position', lateral_position);
recorder.recordVariable('p_axial', p_axial);
recorder.recordVariable('p_lateral', p_lateral);
recorder.recordVariable('p_axial_complex', p_axial_complex);

recorder.saveRecordsToDisk();
