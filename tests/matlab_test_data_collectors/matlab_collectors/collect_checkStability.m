recorder = utils.TestRecorder("collectedValues/checkstability.mat");

c_min = 1500;
rho_min = 1000;
alpha_coeff_min = 4;
ppw = 3;
freq = 500e3;

Nx = 128;
Ny = 128;
Nz = 128;

dx = c_min / (ppw * freq);

sound = ones(Nx, Ny, Nz) * c_min;
density = ones(Nx, Ny, Nz) * rho_min;
alpha_coeff = ones(Nx, Ny, Nz) * alpha_coeff_min;
alpha_power = 1.43;

medium.density = density;
medium.sound_speed = sound;
medium.alpha_coeff = alpha_coeff;
medium.alpha_power = alpha_power;

kgrid = kWaveGrid(Nx, dx, Ny, dx, Nz, dx);

dt_stability_limit = checkStability(kgrid, medium);
recorder.recordObject('kgrid', kgrid);
recorder.recordObject('medium', medium);    
recorder.recordVariable('dt', dt_stability_limit);
recorder.increment();
medium.alpha_mode = 'no_dispersion';
medium.sound_speed_ref = 'mean';
dt_stability_limit = checkStability(kgrid, medium);
recorder.recordObject('kgrid', kgrid);
recorder.recordObject('medium', medium);    
recorder.recordVariable('dt', dt_stability_limit);
recorder.increment();
medium.alpha_mode = 'no_absorption';
medium.sound_speed_ref = 'min';
dt_stability_limit = checkStability(kgrid, medium);
recorder.recordObject('kgrid', kgrid);
recorder.recordObject('medium', medium);    
recorder.recordVariable('dt', dt_stability_limit);
recorder.increment();

    

recorder.saveRecordsToDisk();   

