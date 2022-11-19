%     'nearest' - nearest neighbor interpolation
%     'linear'  - bilinear interpolation
%     'spline'  - spline interpolation
%     'cubic'   - bicubic convolution interpolation for uniformly-spaced
%                 data. This method does not extrapolate and falls back to
%                 'spline' interpolation for irregularly-spaced data.
%     'makima'  - modified Akima cubic interpolation

all_params = { ...
    {1,    randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20])}, ...
    {2,    randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20])}, ...
    {3,    randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20]), randi([10,20])}, ...
    };
output_folder = 'collectedValues/resize/';
k = 0;
for idx=1:length(all_params)
    for type= ["linear"] %["nearest", "linear"]
        type = char(type);
        disp(k);

        dim = all_params{idx}{1};
        Nx = all_params{idx}{2};
        Ny = all_params{idx}{3};
        Nz = all_params{idx}{4};
        nNx = all_params{idx}{5};
        nNy = all_params{idx}{6};
        nNz = all_params{idx}{7};

        if dim == 1
            volume = rand([Nx 1]);
            new_size = nNx;
        elseif dim == 2
            volume = rand([Nx,Ny]);
            new_size = [nNx,nNy];
        else
            volume = rand([Nx,Ny,Nz]);
            new_size = [nNx,nNy, nNz];
        end
        resized_volume = resize(volume,new_size, type);

        idx_padded = sprintf('%06d', k );
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        filename = [output_folder idx_padded '.mat'];
        save(filename, 'params', 'volume', 'resized_volume' , 'new_size', 'type');
        k = k + 1;
    end
end
