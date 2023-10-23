% derived from linear_array_transducer script of k-Wave MATLAB

% create data recorder
recorder = utils.TestRecorder('collectedValues/linear_array_transducer.mat');

% =========================================================================
% DEFINE LITERALS
% =========================================================================
    
% select which k-Wave code to run
%   1: MATLAB CPU code
%   2: MATLAB GPU code
%   3: C++ code
%   4: CUDA code
model           = 4;

% medium parameters
c0              = 1500;     % sound speed [m/s]
rho0            = 1000;     % density [kg/m^3]

% source parameters
source_f0       = 1e6;      % source frequency [Hz]
source_amp      = 1e6;      % source pressure [Pa]
source_cycles   = 5;        % number of toneburst cycles
source_focus    = 20e-3;    % focal length [m]
element_num     = 15;       % number of elements
element_width   = 1e-3;     % width [m]
element_length  = 10e-3;    % elevation height [m]
element_pitch   = 2e-3;     % pitch [m]

% transducer position
translation     = [5e-3, 0, 8e-3];
rotation        = [0, 20, 0];

% grid parameters
grid_size_x     = 40e-3;    % [m]
grid_size_y     = 20e-3;    % [m]
grid_size_z     = 40e-3;    % [m]

% computational parameters
ppw             = 3;        % number of points per wavelength
t_end           = 35e-6;    % total compute time [s]
cfl             = 0.5;      % CFL number

% =========================================================================
% RUN SIMULATION
% =========================================================================

% --------------------
% GRID
% --------------------

% calculate the grid spacing based on the PPW and F0
dx = c0 / (ppw * source_f0);   % [m]

% compute the size of the grid
Nx = roundEven(grid_size_x / dx);
Ny = roundEven(grid_size_y / dx);
Nz = roundEven(grid_size_z / dx);

% create the computational grid
kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

% create the time array
kgrid.makeTime(c0, cfl, t_end);

recorder.recordObject('kgrid', kgrid);

% --------------------
% SOURCE
% --------------------

% set indices for each element
if rem(element_num, 2)
    ids = (1:element_num) - ceil(element_num/2);
else
    ids = (1:element_num) - (element_num + 1)/2;
end

% set time delays for each element to focus at source_focus
time_delays = -(sqrt((ids .* element_pitch).^2 + source_focus.^2) - source_focus) ./ c0;
time_delays = time_delays - min(time_delays);

% create time varying source signals (one for each physical element)
source_sig = source_amp .* toneBurst(1/kgrid.dt, source_f0, source_cycles, 'SignalOffset', round(time_delays / kgrid.dt));

% create empty kWaveArray
karray = kWaveArray('BLITolerance', 0.05, 'UpsamplingRate', 10);

% add rectangular elements
for ind = 1:element_num
    
    % set element y position
    x_pos = 0 - (element_num * element_pitch / 2 - element_pitch / 2) + (ind - 1) * element_pitch;
    
    % add element (set rotation angle to match the global rotation angle)
    karray.addRectElement([x_pos, 0, kgrid.z_vec(1)], element_width, element_length, rotation);
    
end

% move the array
karray.setArrayPosition(translation, rotation)

recorder.recordObject('karray', karray);
% assign binary mask
source.p_mask = karray.getArrayBinaryMask(kgrid);

% assign source signals
source.p = karray.getDistributedSourceSignal(kgrid, source_sig);
    
% --------------------
% MEDIUM
% --------------------

% assign medium properties
medium.sound_speed = c0;
medium.density = rho0;

% --------------------
% SENSOR
% --------------------

% set sensor mask to record central plane
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(:, Ny/2, :) = 1;

% record the pressure
sensor.record = {'p_max'};

recorder.recordObject('sensor', sensor);
recorder.saveRecordsToDisk();
