params = { ...
    [20, 20, 5, 5, 2, 2*pi], ...
    [100, 40, 15, 8, 4, 2*pi], ...
    [50, 50, 25, 25, 10, 2*pi], ...
    [50, 50, 25, 25, 10, pi], ...
    [50, 50, 25, 25, 10, 0.75 * pi], ...
    [10, 8, 3, 4, 10, 0.1 * pi], ...
    [20, 20, 0, 5, 2, 2*pi], ...
    [20, 20, 5, 0, 2, 2*pi], ...
    [20.3, 19.7, 5, 4, 2, 2*pi], ...
}; 

output_folder = 'collectedValues/makeCircle/';

% 

for idx=1:length(params)
    disp(idx);
    param = params{idx};
    
    Nx = param(1);
    Ny = param(2);
    cx = param(3);
    cy = param(4);
    radius = param(5);
    arc_angle = param(6);
    circle = makeCircle(Nx, Ny, cx, cy, radius, arc_angle);
    
    idx_padded = sprintf('%06d', idx - 1);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    filename = [output_folder idx_padded '.mat'];    save(filename, 'Nx', 'Ny', 'cx', 'cy', 'radius', 'arc_angle', 'circle');
end
