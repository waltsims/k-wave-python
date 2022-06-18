list_of_seconds = { ...
    10, 300, 5000, 12000, ...
    5^4, 12^10, 8.3^4, ...
    0.23^8, ...
    -3, -3.5 ...
};


idx = 0;

for i=1:length(list_of_seconds)
    seconds = list_of_seconds{i};
    
    time = scaleTime(seconds);
 
                
    idx_padded = sprintf('%06d', idx);
    filename = ['collectedValues_scaleTime/' idx_padded '.mat'];
    save(filename, 'seconds', 'time');

    idx = idx + 1;
end
disp('Done.')
