list_of_x = { ...
    [20, 30, 12], ....  % should take the max
    -5, ...
    -0.3, ...
    0, ...
    1e-3, 3e-5, 6e-9, 2e-12, 2e-18, 2e-21, 2e-23, 2e-25, 3e-29, ...
    1e4, 3e7, 6e9, 2e12, 2e18, 2e21, 2e23, 2e25, 3e29, 4e32, ...
};


idx = 0;

for i=1:length(list_of_x)
    x = list_of_x{i};
    
    [x_sc, scale, prefix, prefix_fullname] = scaleSI(x);
 
                
    idx_padded = sprintf('%06d', idx);
    filename = ['collectedValues_scaleSI/' idx_padded '.mat'];
    save(filename, 'x', 'x_sc', 'scale', 'prefix', 'prefix_fullname');

    idx = idx + 1;
end
disp('Done.')
